from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformer_block import TransformerConfig, TransformerTower

# =============================== CONFIG ===============================

@dataclass
class ModelConfig:
    vocab_size: int = 32
    mask_token_id: int = 4
    mask_prob: float = 0.15
    base_dim: int = 256
    growth: int = 32
    num_scales: int = 2
    kernel_size: int = 5
    initial_conv_kernel: int = 15
    pool_factor: int = 2
    upsample_mode: str = "nearest"
    dilation_schedule: Optional[Tuple[int, ...]] = None
    
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2) * dilation, dilation=dilation) 

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
    def __init__(self, in_channels: int, growth: int = 128, pool_factor: int = 2, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        out_channels = in_channels + growth
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.pool = nn.MaxPool1d(kernel_size=pool_factor, stride=pool_factor)
        self.growth = growth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x                                    # [B, L, C_in]
        x = self.conv1(x)                               # [B, L, C_in + growth]

        # Can't add residual ([B, L, C_in]) directly to x ([B, L, C_in + growth])
        # zero-pad residual to match the new channel dimension:
        pad = torch.zeros_like(x[..., :self.growth])    # [B, L, growth]
        x = x + torch.cat([residual, pad], dim=-1)      # [B, L, C_in + growth]

        x = self.conv2(x)                               # [B, L, C_in + growth]
        skip = x                                        # [B, L, C_in + growth]
        x = x.transpose(1, 2)                           # [B, C, L]
        x = self.pool(x).transpose(1, 2)                # [B, L//2, C]
        return x, skip


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
        
        # Embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.base_dim)
        
        # Initial conv
        self.initial_conv = nn.Conv1d(cfg.base_dim, cfg.base_dim, kernel_size=cfg.initial_conv_kernel, padding=cfg.initial_conv_kernel // 2)
        
        # Conv Block
        self.conv_block = ConvBlock(cfg.base_dim, cfg.base_dim, kernel_size=cfg.kernel_size)
        
        # Encoder
        enc_in_channels = [cfg.base_dim + i * cfg.growth for i in range(cfg.num_scales)]
        dilations = getattr(cfg, "dilation_schedule", None) or (1,)*cfg.num_scales
        self.enc_blocks = nn.ModuleList([
            DownBlock(cin, growth=cfg.growth, pool_factor=cfg.pool_factor, kernel_size=cfg.kernel_size, dilation=dil)
            for cin, dil in zip(enc_in_channels, dilations)
        ])

        # Transformer
        self.transformer_cfg = TransformerConfig(dim=cfg.bottleneck_dim)
        self.transformer_tower = TransformerTower(self.transformer_cfg)

        # Decoder
        self.dec_in_channels  = [cfg.base_dim + i * cfg.growth for i in reversed(range(1, cfg.num_scales + 1))]
        self.dec_out_channels = [cfg.base_dim + i * cfg.growth for i in reversed(range(cfg.num_scales))]
        self.dec_blocks = nn.ModuleList([
            UpBlock(in_ch, out_ch, upsample_mode=cfg.upsample_mode, kernel_size=cfg.kernel_size)
            for in_ch, out_ch in zip(self.dec_in_channels, self.dec_out_channels)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(cfg.base_dim, cfg.vocab_size)

    def forward(
        self, 
        token_ids: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        return_dict: bool = False,
        **kwargs
    ):
        """
        Forward pass with optional HF-style arguments.
        
        Args:
            token_ids: Token IDs [B, L] (original name)
            input_ids: Token IDs [B, L] (HF standard name)
            labels: Ground truth labels (accepted for HF compatibility but not used)
            return_dict: Whether to return dict (for HF compatibility)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            logits [B, L, vocab_size] or dict with 'logits' key
        """
        # Handle both input_ids (HF standard) and token_ids (your original)
        # Priority: token_ids first (for explicit calls), then input_ids
        x = token_ids if token_ids is not None else input_ids
        
        if x is None:
            raise ValueError("Either token_ids or input_ids must be provided")
        
        # Embedding
        x = self.embed(x).clone()  # [B, L, base_dim]
        
        # Initial conv
        x = self.initial_conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Residual conv block
        x = x + self.conv_block(x).clone()
        
        # Encoder
        skips = []
        for block in self.enc_blocks:
            x, skip = block(x)
            skips.append(skip)
        
        # Transformer
        x = self.transformer_tower(x)
        
        # Decoder
        for block, skip in zip(self.dec_blocks, reversed(skips)):
            x = block(x, skip)
        
        # Output
        logits = self.output_layer(x)
        
        # Return format
        if return_dict:
            return {"logits": logits}
        return logits
