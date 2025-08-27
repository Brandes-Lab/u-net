from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer_block_updated import TransformerConfig, TransformerTower
# =============================== CONFIG ===============================

@dataclass
class ModelConfig:
    vocab_size: int = 32
    mask_token_id: int = 4
    mask_prob: float = 0.15
    base_dim: int = 256
    growth: int = 64
    num_scales: int = 4
    kernel_size: int = 5
    initial_conv_kernel: int = 15
    pool_factor: int = 2
    upsample_mode: str = "nearest"
    
    transformer_cfg: TransformerConfig = TransformerConfig()

    @property
    def bottleneck_dim(self) -> int:
        return self.base_dim + self.num_scales * self.growth

    @property
    def total_downsample(self) -> int:
        return self.pool_factor ** self.num_scales


# =========================== BUILDING BLOCKS ===========================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.weight / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class ConvBlock(nn.Module):
    """
    PreNorm + GELU + 1D Conv with same-length padding.

    Expects and returns tensors in [B, L, C] (batch, length, channels).
    Internally transposes to [B, C, L] for Conv1d, then transposes back.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C_in]
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)           # -> [B, C_in, L]
        x = self.conv(x)                # -> [B, C_out, L]
        x = x.transpose(1, 2)           # -> [B, L, C_out]
        return x


class DownBlock(nn.Module):
    """
    One encoder stage:
      - Increase channels by `growth` via two ConvBlocks (with a residual-ish add)
      - Save a skip tensor (pre-pooled)
      - Downsample length by `pool_factor` using MaxPool1d

    Input : [B, L, C_in]
    Skip  : [B, L, C_in + growth]
    Output: [B, L//pool_factor, C_in + growth]
    """
    def __init__(self, in_channels: int, growth: int = 128, pool_factor: int = 2, kernel_size: int = 5) -> None:
        super().__init__()
        out_channels = in_channels + growth
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=pool_factor, stride=pool_factor)
        self.growth = growth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x                                    # [B, L, C_in]
        x = self.conv1(x)                               # [B, L, C_in + growth]

        # Can’t add residual ([B, L, C_in]) directly to x ([B, L, C_in + growth])
        # zero-pad residual to match the new channel dimension:
        pad = torch.zeros_like(x[..., :self.growth])    # [B, L, growth]
        x = x + torch.cat([residual, pad], dim=-1)      # [B, L, C_in + growth]

        x = self.conv2(x)                               # [B, L, C_in + growth]
        skip = x                                        # [B, L, C_in + growth]
        x = x.transpose(1, 2)                           # [B, C, L]
        x = self.pool(x).transpose(1, 2)                # [B, L//2, C]
        return x, skip

# If input x is [B, 512, 512] with growth=96, output from one interation will be:
# x: [B, 256, 608] (sequence halved, channels increased)
# skip: [B, 512, 608] (stored for decoder)

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample_mode: str = "nearest", kernel_size: int = 5) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        target_len = skip.size(1)
        if self.upsample_mode == "linear":
            x = F.interpolate(x.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        else:
            x = F.interpolate(x.transpose(1, 2), size=target_len, mode=self.upsample_mode).transpose(1, 2)
        x = self.conv1(x)
        if skip.shape[-1] != x.shape[-1]:
            skip = skip[..., :x.shape[-1]]
        x = x + skip
        x = self.conv2(x)
        return x




# ================================ MODEL ================================

class ProteinMLMModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # starting with [B, S] = [64, 512], 64 sequences, each 512 tokens long
        # Embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.base_dim) # [B, S, base_dim] = [64, 512, 512] 

        # 1D convolutional layer, Input = [Batch, Channels, Length] = [B, 512, 512], Output = [B, 512, 512]
        self.initial_conv = nn.Conv1d(cfg.base_dim, cfg.base_dim, kernel_size=cfg.initial_conv_kernel, padding=cfg.initial_conv_kernel // 2)
        
        # Conv Block, Input = [B, 512, 512], Output = [B, 512, 512]
        self.conv_block = ConvBlock(cfg.base_dim, cfg.base_dim, kernel_size=cfg.kernel_size)
        
        # Sequence Encoder, decrease seq length, increase number of channels 
            # 512 → 256 → 128 → 64 → 32 → 16 → 8
            # enc_in_channels = [
            #         512 + 1 * 96,  # 608
            #         512 + 2 * 96,  # 704
            #         512 + 3 * 96,  # 800
            #         512 + 4 * 96,  # 896
            #         512 + 5 * 96   # 992
            #         512 + 6 * 96   # 1088
            #     ]
        enc_in_channels = [cfg.base_dim + i * cfg.growth for i in range(cfg.num_scales)]
        self.enc_blocks = nn.ModuleList([
            DownBlock(cin, growth=cfg.growth, pool_factor=cfg.pool_factor, kernel_size=cfg.kernel_size)
            for cin in enc_in_channels
        ])

        self.transformer_tower = TransformerTower(self.cfg.transformer_cfg)

        # Sequence Decoder, Increase seq length, decrease number of channels 
        # in_channels   : [1088, 992, 896, 800, 704, 608]   # From upsample + skip concat
        # out_channels  :  [992, 896, 800, 704, 608, 512]   # Output size of each UpBlock
        self.dec_in_channels  = [cfg.base_dim + i * cfg.growth for i in reversed(range(1, cfg.num_scales + 1))]
        self.dec_out_channels = [cfg.base_dim + i * cfg.growth for i in reversed(range(cfg.num_scales))]
        self.dec_blocks = nn.ModuleList([
            UpBlock(in_ch, out_ch, upsample_mode=cfg.upsample_mode, kernel_size=cfg.kernel_size)
            for in_ch, out_ch in zip(self.dec_in_channels, self.dec_out_channels)
        ])
        self.output_layer = nn.Linear(cfg.base_dim, cfg.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # print("Input token_ids:", token_ids.shape, flush=True)  # [B, L]
        
        # Embedding
        x = self.embed(token_ids)  # [B, L, base_dim]
        # print("After embedding:", x.shape, flush=True)

        # Initial conv
        x = self.initial_conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, base_dim]
        # print("After initial_conv:", x.shape, flush=True)
        
        # Residual conv block
        x = x + self.conv_block(x)  # [B, L, base_dim]
        # print("After conv_block:", x.shape, flush=True)
        
        # Encoder
        skips = []
        for i, block in enumerate(self.enc_blocks):
            x, skip = block(x)
            skips.append(skip)
            # print(f"After encoder block {i}: x = {x.shape}, skip = {skip.shape}", flush=True)

        # Transformer 
        # print("Entering TransformerTower:", x.shape, flush=True)
        x, pair_x = self.transformer_tower(x)
        # print("After transformer_tower: x =", x.shape, flush=True)
        # print("pair_x =", pair_x.shape, flush=True)

        # Decoder
        for i, (block, skip) in enumerate(zip(self.dec_blocks, reversed(skips))):
            x = block(x, skip)
            # print(f"After decoder block {i}: x = {x.shape}", flush=True)

        # Final output layer
        logits = self.output_layer(x)
        # print("Final logits:", logits.shape, flush=True)  # [B, L, vocab_size]

        return logits