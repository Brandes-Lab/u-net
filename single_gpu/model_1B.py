"""
Protein MLM Model with U-Net style architecture using ModernBERT as the bottleneck transformer.

Architecture:
    Embedding → Conv Encoder (downsampling) → ModernBERT → Conv Decoder (upsampling) → Output
    
Key features:
- Proper weight initialization for stable training
- Gradient checkpointing support for memory efficiency
- Compatible with HuggingFace Trainer
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertModel


# =============================== CONFIG ===============================

@dataclass
class ModelConfig:
    """Configuration for the ProteinMLMModel."""
    vocab_size: int = 32
    mask_token_id: int = 4
    pad_token_id: int = 0
    mask_prob: float = 0.15
    
    # U-Net encoder/decoder settings
    base_dim: int = 256
    growth: int = 32
    num_scales: int = 2
    kernel_size: int = 5
    initial_conv_kernel: int = 15
    pool_factor: int = 2
    upsample_mode: str = "nearest"
    dilation_schedule: Optional[Tuple[int, ...]] = None
    
    # ModernBERT settings
    modernbert_num_layers: int = 6
    modernbert_num_attention_heads: int = 8
    modernbert_intermediate_size: int = 1024
    modernbert_hidden_dropout: float = 0.1
    modernbert_attention_dropout: float = 0.1
    modernbert_max_position_embeddings: int = 8192
    
    # Training settings
    use_gradient_checkpointing: bool = False

    @property
    def bottleneck_dim(self) -> int:
        """Channel dimension at the bottleneck (after all downsampling)."""
        return self.base_dim + self.num_scales * self.growth

    @property
    def total_downsample(self) -> int:
        """Total downsampling factor."""
        return self.pool_factor ** self.num_scales


# =========================== BUILDING BLOCKS ===========================

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
        self.conv = nn.Conv1d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            padding=(kernel_size // 2) * dilation, 
            dilation=dilation
        )
        
        # Initialize conv weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C_in]
        x = self.norm(x)
        x = self.act(x)
        x = x.transpose(1, 2)  # -> [B, C_in, L]
        x = self.conv(x)       # -> [B, C_out, L]
        x = x.transpose(1, 2)  # -> [B, L, C_out]
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
    def __init__(
        self, 
        in_channels: int, 
        growth: int = 128, 
        pool_factor: int = 2, 
        kernel_size: int = 5, 
        dilation: int = 1
    ) -> None:
        super().__init__()
        out_channels = in_channels + growth
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self.pool = nn.MaxPool1d(kernel_size=pool_factor, stride=pool_factor)
        self.growth = growth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x  # [B, L, C_in]
        x = self.conv1(x)  # [B, L, C_in + growth]

        # Zero-pad residual to match the new channel dimension
        pad = torch.zeros_like(x[..., :self.growth])  # [B, L, growth]
        x = x + torch.cat([residual, pad], dim=-1)    # [B, L, C_in + growth]

        x = self.conv2(x)   # [B, L, C_in + growth]
        skip = x            # [B, L, C_in + growth]
        x = x.transpose(1, 2)                         # [B, C, L]
        x = self.pool(x).transpose(1, 2)              # [B, L//2, C]
        return x, skip


class UpBlock(nn.Module):
    """
    AlphaGenome-style decoder stage:
      - Project input channels down to match skip connection channels via conv
      - Upsample with learnable residual scaling
      - Process skip connection with 1x1 conv before adding
      - Final conv refinement with residual connection

    Input : [B, L_low, C_in]
    Skip  : [B, L_high, C_skip]
    Output: [B, L_high, C_skip]
    """
    def __init__(
        self, 
        in_channels: int, 
        skip_channels: int,
        upsample_mode: str = "nearest", 
        kernel_size: int = 5,
        residual_scale_init: float = 0.9
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.skip_channels = skip_channels
        
        # Project input channels to match skip channels (replaces truncation)
        self.channel_proj = ConvBlock(in_channels, skip_channels, kernel_size=kernel_size)
        
        # Learnable residual scale (initialized to 0.9 like AlphaGenome)
        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))
        
        # 1x1 conv to process skip connection before adding
        self.skip_proj = ConvBlock(skip_channels, skip_channels, kernel_size=1)
        
        # Final refinement conv with residual
        self.final_conv = ConvBlock(skip_channels, skip_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # x: [B, L_low, C_in], skip: [B, L_high, C_skip]
        target_len = skip.size(1)
        
        # Project channels down to match skip (learned projection, not truncation)
        x_proj = self.channel_proj(x)  # [B, L_low, C_skip]
        
        # Residual connection with channel cropping (for the residual path only)
        x_residual = x[..., :self.skip_channels].contiguous()  # [B, L_low, C_skip]
        x_proj = x_proj + x_residual
        
        # Upsample to match skip connection length
        if self.upsample_mode == "linear":
            x_up = F.interpolate(
                x_proj.transpose(1, 2).contiguous(), size=target_len, mode="linear", align_corners=False
            ).transpose(1, 2).contiguous()
        else:
            x_up = F.interpolate(
                x_proj.transpose(1, 2).contiguous(), size=target_len, mode=self.upsample_mode
            ).transpose(1, 2).contiguous()
        
        # Apply learnable residual scale
        x_up = x_up * self.residual_scale
        
        # Process skip connection with 1x1 conv (learned transformation)
        skip_processed = self.skip_proj(skip)  # [B, L_high, C_skip]
        
        # Add processed skip connection
        x_combined = x_up + skip_processed  # [B, L_high, C_skip]
        
        # Final refinement with residual
        out = x_combined + self.final_conv(x_combined)  # [B, L_high, C_skip]
        
        return out


# ================================ MODEL ================================

class ProteinMLMModel(nn.Module):
    """
    Protein Masked Language Model with U-Net architecture and ModernBERT bottleneck.
    
    Architecture:
        1. Embedding layer
        2. Initial convolution
        3. Encoder (downsampling conv blocks)
        4. ModernBERT transformer at bottleneck
        5. Decoder (upsampling conv blocks with skip connections)
        6. Output projection to vocabulary
    
    For use with HuggingFace Trainer, the forward method accepts `labels` 
    and returns a dict with 'loss' and 'logits'.
    """
    
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        # Embedding layer with scaled initialization
        self.embed = nn.Embedding(cfg.vocab_size, cfg.base_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        
        # Embedding scale factor (common in transformers)
        self.embed_scale = math.sqrt(cfg.base_dim)
        
        # Initial 1D convolution
        self.initial_conv = nn.Conv1d(
            cfg.base_dim, cfg.base_dim, 
            kernel_size=cfg.initial_conv_kernel, 
            padding=cfg.initial_conv_kernel // 2
        )
        nn.init.kaiming_normal_(self.initial_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.initial_conv.bias)
        
        # Pre-encoder conv block
        self.conv_block = ConvBlock(cfg.base_dim, cfg.base_dim, kernel_size=cfg.kernel_size)
        
        # Encoder blocks (downsampling)
        enc_in_channels = [cfg.base_dim + i * cfg.growth for i in range(cfg.num_scales)]
        dilations = cfg.dilation_schedule or (1,) * cfg.num_scales
        self.enc_blocks = nn.ModuleList([
            DownBlock(
                cin, 
                growth=cfg.growth, 
                pool_factor=cfg.pool_factor, 
                kernel_size=cfg.kernel_size, 
                dilation=dil
            )
            for cin, dil in zip(enc_in_channels, dilations)
        ])
        
        # ModernBERT at bottleneck
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
            _attn_implementation="sdpa",  # Use SDPA for stability
        )
        self.modernbert = ModernBertModel(modernbert_config)
        
        # Enable gradient checkpointing if requested
        if cfg.use_gradient_checkpointing:
            self.modernbert.gradient_checkpointing_enable()
        
        # Decoder blocks (upsampling)
        dec_in_channels = [cfg.bottleneck_dim]
        dec_skip_channels = []
        for i in reversed(range(cfg.num_scales)):
            skip_ch = cfg.base_dim + (i + 1) * cfg.growth
            dec_skip_channels.append(skip_ch)
            if i > 0:
                dec_in_channels.append(skip_ch)
        
        self.dec_blocks = nn.ModuleList([
            UpBlock(
                in_channels=in_ch, 
                skip_channels=skip_ch,
                upsample_mode=cfg.upsample_mode, 
                kernel_size=cfg.kernel_size
            )
            for in_ch, skip_ch in zip(dec_in_channels, dec_skip_channels)
        ])
        
        # Final output dimension after decoder
        self.final_output_dim = cfg.base_dim + cfg.growth
        
        # Output projection with conservative initialization
        self.output_layer = nn.Linear(self.final_output_dim, cfg.vocab_size)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.02)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for MLM training.
        
        Args:
            input_ids: Token IDs [B, L]
            labels: Target labels for MLM loss [B, L], -100 for non-masked positions
            attention_mask: Attention mask [B, L] (optional)
            
        Returns:
            Dict with 'loss' (if labels provided) and 'logits'
        """
        # Embedding with scaling
        x = self.embed(input_ids) * self.embed_scale  # [B, L, base_dim]
        
        # Initial convolution
        x = self.initial_conv(x.transpose(1, 2)).transpose(1, 2)  # [B, L, base_dim]
        
        # Pre-encoder residual conv block
        x = x + self.conv_block(x)  # [B, L, base_dim]
        
        # Encoder (collect skip connections)
        skips = []
        for block in self.enc_blocks:
            x, skip = block(x)
            skips.append(skip)
        
        # ModernBERT at bottleneck
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
        
        modernbert_output = self.modernbert(
            inputs_embeds=x,
            attention_mask=downsampled_mask,
            output_hidden_states=False,
            return_dict=True
        )
        x = modernbert_output.last_hidden_state
        
        # Decoder (use skip connections in reverse order)
        for block, skip in zip(self.dec_blocks, reversed(skips)):
            x = block(x, skip)
        
        # Output projection
        logits = self.output_layer(x)  # [B, L, vocab_size]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="mean"
            )
        
        return {"loss": loss, "logits": logits}

    def get_output_embeddings(self):
        """Required for some HF utilities."""
        return self.output_layer
    
    def set_output_embeddings(self, new_embeddings):
        """Required for some HF utilities."""
        self.output_layer = new_embeddings