import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# ----------------------------------------------- Config ----------------------------------------------- #
@dataclass
class TransformerConfig:
    dim: int = 512                      # Embedding dimension
    depth: int = 6                      # Number of Transformer layers
    heads: int = 8                      # Number of attention heads
    rope_max_position: int = 2048       # RoPE uses this for max relative distance

# ----------------------------------------------- Transformer Tower Class ----------------------------------------------- #
class TransformerTower(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.mha_blocks = nn.ModuleList([MHA_Block(cfg) for _ in range(cfg.depth)])
        self.mlp_blocks = nn.ModuleList([MLPBlock(cfg) for _ in range(cfg.depth)])

    def forward(self, x):
        # x: [B, S, C]

        for i in range(self.cfg.depth): 
            # Input = [B, S, C], Output = [B, S, C]
            x = x + self.mha_blocks[i](x, attention_bias=None)

            # Input = [B, S, C], Output = [B, S, C]
            x = x + self.mlp_blocks[i](x)

        return x   # x: [B, S, C]

 
# ----------------------------------------------- MHA + MLP Blocks ----------------------------------------------- #

class MHA_Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads  # Automatically computed
        self.q_proj = nn.Linear(cfg.dim, cfg.dim)
        self.k_proj = nn.Linear(cfg.dim, cfg.dim)
        self.v_proj = nn.Linear(cfg.dim, cfg.dim)
        self.out_proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm = RMSNorm(cfg)
        self.cfg = cfg

    def forward(self, x, attention_bias=None):
        x = self.norm(x)
        B, S, _ = x.shape

        q = self.q_proj(x).reshape(B, S, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(B, S, self.heads, self.head_dim)
        v = self.v_proj(x).reshape(B, S, self.heads, self.head_dim)

        q = apply_rope(q, self.cfg)
        k = apply_rope(k, self.cfg)

        q_t = q.transpose(1, 2)  # [B, H, S, head_dim]
        k_t = k.transpose(1, 2)  # [B, H, S, head_dim]
        v_t = v.transpose(1, 2)  # [B, H, S, head_dim]

        attn_logits = torch.einsum("bhid,bhjd->bhij", q_t, k_t)
        attn_logits = attn_logits / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_logits, dim=-1)

        context = torch.einsum("bhij,bhjd->bhid", attn_weights, v_t)
        context = context.transpose(1, 2).reshape(B, S, -1)
        out = self.out_proj(context)
        return out




def apply_rope(x: torch.Tensor, cfg: TransformerConfig, positions: torch.Tensor = None) -> torch.Tensor:
    if positions is None:
        positions = torch.arange(x.shape[1], device=x.device)  # [L]

    D = x.shape[-1]
    assert D % 2 == 0, "Last dimension (head_dim) must be even for RoPE"

    num_freq = D // 2
    max_position = cfg.rope_max_position

    # Create frequency terms
    freq = 1.0 / torch.logspace(
        start=0,
        end=math.log10(max_position),
        steps=num_freq,
        device=x.device
    )

    # Compute position-dependent angle matrix
    theta = torch.einsum('s,f->sf', positions.float(), freq)  # [L, D//2]

    # Get sin and cos for even/odd dims
    sin_theta = torch.sin(theta)[None, :, None, :]  # [1, L, 1, D//2]
    cos_theta = torch.cos(theta)[None, :, None, :]  # [1, L, 1, D//2]

    # Split x into even and odd dims
    x1 = x[..., ::2]  # [B, L, H, D//2]
    x2 = x[..., 1::2]  # [B, L, H, D//2]

    # Apply rotation
    x_rotated = torch.cat([x1 * cos_theta - x2 * sin_theta,
                           x1 * sin_theta + x2 * cos_theta], dim=-1)  # [B, L, H, D]

    return x_rotated



class RMSNorm(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.eps = 1e-8
        self.weight = nn.Parameter(torch.ones(cfg.dim))  

    def forward(self, x):
        return x * self.weight / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)



class MLPBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.norm = RMSNorm(cfg)
        self.linear1 = nn.Linear(cfg.dim, cfg.dim * 2)
        self.linear2 = nn.Linear(cfg.dim * 2, cfg.dim)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x