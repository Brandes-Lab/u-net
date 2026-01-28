"""
Protein MLM Model with U-Net style architecture using ModernBERT as the bottleneck transformer.

Architecture:
    Embedding → Conv Encoder (downsampling) → ModernBERT → Conv Decoder (upsampling) → Output
    
FSDP Compatible:
- Gradient checkpointing enabled (works with FSDP unlike DDP)
- Clean module structure for FSDP wrapping
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import ModernBertConfig, ModernBertModel


@dataclass
class ModelConfig:
    vocab_size: int = 32
    mask_token_id: int = 4
    pad_token_id: int = 0
    mask_prob: float = 0.15
    
    base_dim: int = 256
    growth: int = 32
    num_scales: int = 2
    kernel_size: int = 5
    initial_conv_kernel: int = 15
    pool_factor: int = 2
    upsample_mode: str = "nearest"
    dilation_schedule: Optional[Tuple[int, ...]] = None
    
    modernbert_num_layers: int = 6
    modernbert_num_attention_heads: int = 8
    modernbert_intermediate_size: int = 1024
    modernbert_hidden_dropout: float = 0.1
    modernbert_attention_dropout: float = 0.1
    modernbert_max_position_embeddings: int = 8192
    
    use_gradient_checkpointing: bool = True  # Enabled for FSDP

    @property
    def bottleneck_dim(self) -> int:
        return self.base_dim + self.num_scales * self.growth

    @property
    def total_downsample(self) -> int:
        return self.pool_factor ** self.num_scales


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=(kernel_size // 2) * dilation, dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class DownBlock(nn.Module):
    """Encoder block - good FSDP wrapping boundary."""
    def __init__(self, in_channels: int, growth: int = 128, pool_factor: int = 2, 
                 kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        out_channels = in_channels + growth
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.pool = nn.MaxPool1d(kernel_size=pool_factor, stride=pool_factor)
        self.growth = growth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.conv1(x)
        pad = torch.zeros_like(x[..., :self.growth])
        x = x + torch.cat([residual, pad], dim=-1)
        x = self.conv2(x)
        skip = x
        x = x.transpose(1, 2)
        x = self.pool(x).transpose(1, 2)
        return x, skip


class UpBlock(nn.Module):
    """Decoder block - good FSDP wrapping boundary."""
    def __init__(self, in_channels: int, skip_channels: int, upsample_mode: str = "nearest", 
                 kernel_size: int = 5, residual_scale_init: float = 0.9):
        super().__init__()
        self.upsample_mode = upsample_mode
        self.skip_channels = skip_channels
        
        self.channel_proj = ConvBlock(in_channels, skip_channels, kernel_size=kernel_size)
        # FSDP requires 1D tensor, not scalar
        self.residual_scale = nn.Parameter(torch.tensor([residual_scale_init]))
        self.skip_proj = ConvBlock(skip_channels, skip_channels, kernel_size=1)
        self.final_conv = ConvBlock(skip_channels, skip_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        target_len = skip.size(1)
        
        x_proj = self.channel_proj(x)
        x_residual = x[..., :self.skip_channels].contiguous()
        x_proj = x_proj + x_residual
        
        if self.upsample_mode == "linear":
            x_up = F.interpolate(x_proj.transpose(1, 2).contiguous(), size=target_len, 
                                 mode="linear", align_corners=False).transpose(1, 2).contiguous()
        else:
            x_up = F.interpolate(x_proj.transpose(1, 2).contiguous(), size=target_len, 
                                 mode=self.upsample_mode).transpose(1, 2).contiguous()
        
        x_up = x_up * self.residual_scale
        
        skip_processed = self.skip_proj(skip)
        x_combined = x_up + skip_processed
        
        out = x_combined + self.final_conv(x_combined)
        return out


class ProteinMLMModel(nn.Module):
    """
    Protein MLM Model with FSDP support.
    
    FSDP wraps DownBlock, UpBlock, and ModernBertEncoderLayer for optimal sharding.
    Gradient checkpointing is applied to ModernBERT layers.
    """
    
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        self.embed = nn.Embedding(cfg.vocab_size, cfg.base_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.embed_scale = math.sqrt(cfg.base_dim)
        
        self.initial_conv = nn.Conv1d(cfg.base_dim, cfg.base_dim, kernel_size=cfg.initial_conv_kernel, 
                                       padding=cfg.initial_conv_kernel // 2)
        nn.init.kaiming_normal_(self.initial_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.initial_conv.bias)
        
        self.conv_block = ConvBlock(cfg.base_dim, cfg.base_dim, kernel_size=cfg.kernel_size)
        
        enc_in_channels = [cfg.base_dim + i * cfg.growth for i in range(cfg.num_scales)]
        dilations = cfg.dilation_schedule or (1,) * cfg.num_scales
        self.enc_blocks = nn.ModuleList([
            DownBlock(cin, growth=cfg.growth, pool_factor=cfg.pool_factor, 
                     kernel_size=cfg.kernel_size, dilation=dil)
            for cin, dil in zip(enc_in_channels, dilations)
        ])
        
        modernbert_config = ModernBertConfig(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.bottleneck_dim,
            num_hidden_layers=cfg.modernbert_num_layers,
            num_attention_heads=cfg.modernbert_num_attention_heads,
            intermediate_size=cfg.modernbert_intermediate_size,
            hidden_dropout_prob=cfg.modernbert_hidden_dropout,
            attention_probs_dropout_prob=cfg.modernbert_attention_dropout,
            max_position_embeddings=cfg.modernbert_max_position_embeddings,
            pad_token_id=cfg.pad_token_id,
            _attn_implementation="sdpa",
        )
        self.modernbert = ModernBertModel(modernbert_config)
        
        # Enable gradient checkpointing for ModernBERT
        if cfg.use_gradient_checkpointing:
            self.modernbert.gradient_checkpointing_enable()
        
        dec_in_channels = [cfg.bottleneck_dim]
        dec_skip_channels = []
        for i in reversed(range(cfg.num_scales)):
            skip_ch = cfg.base_dim + (i + 1) * cfg.growth
            dec_skip_channels.append(skip_ch)
            if i > 0:
                dec_in_channels.append(skip_ch)
        
        self.dec_blocks = nn.ModuleList([
            UpBlock(in_ch, skip_ch, upsample_mode=cfg.upsample_mode, kernel_size=cfg.kernel_size)
            for in_ch, skip_ch in zip(dec_in_channels, dec_skip_channels)
        ])
        
        self.final_output_dim = cfg.base_dim + cfg.growth
        self.output_layer = nn.Linear(self.final_output_dim, cfg.vocab_size)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.02)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        x = self.embed(input_ids) * self.embed_scale
        x = self.initial_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.conv_block(x)
        
        skips = []
        for block in self.enc_blocks:
            x, skip = block(x)
            skips.append(skip)
        
        if attention_mask is not None:
            bottleneck_len = x.size(1)
            downsampled_mask = F.max_pool1d(
                attention_mask.float().unsqueeze(1),
                kernel_size=self.cfg.total_downsample,
                stride=self.cfg.total_downsample
            ).squeeze(1).long()
            downsampled_mask = downsampled_mask[:, :bottleneck_len]
        else:
            downsampled_mask = None
        
        x = x.contiguous()
        modernbert_output = self.modernbert(inputs_embeds=x, attention_mask=downsampled_mask,
                                            output_hidden_states=False, return_dict=True)
        x = modernbert_output.last_hidden_state
        
        for block, skip in zip(self.dec_blocks, reversed(skips)):
            x = block(x, skip)
        
        logits = self.output_layer(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), labels.reshape(-1),
                                   ignore_index=-100, reduction="mean")
        
        return {"loss": loss, "logits": logits}

    def get_output_embeddings(self):
        return self.output_layer
    
    def set_output_embeddings(self, new_embeddings):
        self.output_layer = new_embeddings