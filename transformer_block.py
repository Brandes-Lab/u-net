import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# ----------------------------------------------- Config ----------------------------------------------- #
@dataclass
class TransformerConfig:
    dim: int = 512                     # channel dim after 4 scales: base_dim + 4 * growth = 256 + 64*4
    depth: int = 6                      
    seq_len: int = 32                   # sequence length after 4 scales (512 // 2^4), Downsampled length after encoder, [B, 32, 512]
    pair_len: int = 32                   # seq_len // repeat_factor
    heads: int = 8
    pair_dim: int = 32               
    rope_max_position: int = 512
    repeat_factor: int = 1


# ----------------------------------------------- Transformer Tower Class ----------------------------------------------- #
class TransformerTower(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.sequence_to_pair = sequence_to_pair_block(cfg)
        self.row_attention = row_attention_block(cfg)
        self.pair_mlp = pair_MLP_block(cfg)
        self.attn_bias_block = AttentionBiasBlock(cfg)
        self.mha_blocks = nn.ModuleList([MHA_Block(cfg) for _ in range(cfg.depth)])
        self.mlp_blocks = nn.ModuleList([MLPBlock(cfg) for _ in range(cfg.depth)])

    def forward(self, x):
        # Input: x.shape = [B, 8, 1088]
        pair_x = None
        for i in range(self.cfg.depth):
            if i % 2 == 0:
                # projects x into pairwise space: computes q, k, pos → then outputs attention matrix.
                y = self.sequence_to_pair(x)                      # [B, P, P, pair_dim] = [B, 8, 8, 64]
                pair_x = y if pair_x is None else pair_x + y     # [B, P, P, pair_dim]
                pair_x = pair_x + self.row_attention(pair_x)             # [B, P, P, pair_dim]
                pair_x = pair_x + self.pair_mlp(pair_x)                  # [B, P, P, pair_dim]

            # Maps pair_x → [B, 8, 8, 8] → reshaped to [B, heads, seq_len, seq_len]
            attn_bias = self.attn_bias_block(pair_x)   # attn_bias.shape = [B, 8, 8, 8]

            # Input = [B, 8, 1088], Output = [B, 8, 1088]
            x = x + self.mha_blocks[i](x, attention_bias=attn_bias)

            # Input = [B, 8, 1088], Output = [B, 8, 1088]
            x = x + self.mlp_blocks[i](x)

        return x, pair_x            # x: [B, 8, 1088], pair_x: [B, 8, 8, 64]

class sequence_to_pair_block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=cfg.seq_len // cfg.pair_len, stride=cfg.seq_len // cfg.pair_len)  #  AvgPool1d(kernel_size=1, stride=1)
        self.q_proj = nn.Linear(cfg.dim, cfg.heads * cfg.pair_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, cfg.heads * cfg.pair_dim, bias=False)
        self.pos_proj = nn.Linear(64, cfg.heads * cfg.pair_dim)
        self.q_bias = nn.Parameter(torch.zeros(1, 1, cfg.heads, cfg.pair_dim))
        self.k_bias = nn.Parameter(torch.zeros(1, 1, cfg.heads, cfg.pair_dim))
        self.linear_out = nn.Linear(cfg.heads, cfg.pair_dim)
        self.yq_proj = nn.Linear(cfg.dim, cfg.pair_dim, bias=False)
        self.yk_proj = nn.Linear(cfg.dim, cfg.pair_dim, bias=False)
        self.cfg = cfg

    def forward(self, x):
        B = x.shape[0]
        cfg = self.cfg
        
        x_ds = self.pool(x.transpose(1, 2)).transpose(1, 2)                                  # [B, P, C] = [B, 8, 1088]
        P = x_ds.shape[1]

        #  Project to Query and Key
        q = self.q_proj(x_ds).reshape(B, P, cfg.heads, cfg.pair_dim)                        # [B, P, H, C] = [B, 8, 8, 64]
        k = self.k_proj(x_ds).reshape(B, P, cfg.heads, cfg.pair_dim)                        # [B, P, H, C] = [B, 8, 8, 64]

        # print(f"P: {P}")  # Should be 32 in this case
        pos_features = central_mask_features(P, 64).to(x.device)                # [2P-1, 64] = [15, 64]
        pos_enc = self.pos_proj(pos_features).reshape(2 * P - 1, cfg.heads, cfg.pair_dim)  # [15, 8, 64]

        # q + q_bias = [B, Q, H, C] = [B, 8, 8, 64]
        # pos_enc = [2P-1, H, C] = [15, 8, 64]
        # einsum: [B, Q=8, H=8, D=64], [P=15, H=8, D=64] → [B, Q, P, H] = [B, 8, 15, 8]
        rel_q_a = relative_shift(torch.einsum('bqhc,phc->bqph', q + self.q_bias, pos_enc))  # Output = [B, 8, 8, 8]
        # print("rel_q_a:", rel_q_a.shape)
        rel_k_a = relative_shift(torch.einsum('bkhc,phc->bkph', k + self.k_bias, pos_enc))  # Output = [B, 8, 8, 8]
        
        # dot-product: [B, Q=8, K=8, H=8]
        dot = torch.einsum('bqhc,bkhc->bqkh', q, k) 
        a = dot + (rel_q_a + rel_k_a.permute(0, 2, 1, 3)) / 2                               # Final a: [B, 8, 8, 8]
        
        # print("dot:", dot.shape)                 # [B, P, P, H]
        # print("rel_q_a:", rel_q_a.shape)        # [B, P, P, H]
        # print("rel_k_a:", rel_k_a.shape)        # [B, P, P, H]  
        
        yq = self.yq_proj(F.gelu(x_ds))                                                     # [B, P, pair_dim] =  [B, 8, 64]
        yk = self.yk_proj(F.gelu(x_ds))                                                     # [B, P, pair_dim] =  [B, 8, 64]
        
        # [B, P, P, pair_dim] + [B, P, 1, pair_dim] + [B, 1, P, pair_dim] = [B, P, P, pair_dim]
        pair_activations = self.linear_out(a) + yq[:, :, None, :] + yk[:, None, :, :]       # [B, 8, 8, 64] + [B, 8, 1, 64] + [B, 1, 8, 64]
        
        # [B, P, P, pair_dim]
        return pair_activations                                                              # [B, 8, 8, 64]


def central_mask_features(sequence_length: int, feature_size: int):
    # Generates a [2P-1, feature_dim] tensor with central symmetric features
    # Positional range: -P+1 to P-1
    rel_pos = torch.arange(2 * sequence_length - 1) - (sequence_length - 1)                  # [2*sequence_length - 1]
    widths = torch.arange(feature_size // 2) + torch.logspace(
        start=0,
        end=math.log10(sequence_length - feature_size // 2 + 1),
        steps=feature_size // 2
    )  # list of distance threshodls [feature_size//2]
    embeddings = (widths[None, :] > rel_pos[:, None]).float()                                # [2*sequence_length - 1, feature_size//2]
    signed = torch.sign(rel_pos).unsqueeze(-1) * embeddings
    return torch.cat([embeddings, signed], dim=-1)                                           # [2*sequence_length - 1, feature_size//2]

def relative_shift(x):
    """
    Input: x of shape [B, P, 2P-1, H]
    Output: shifted x of shape [B, P, P, H]
    """
    B, P, _, H = x.shape
    x = F.pad(x, (0, 0, 0, 0, 1, 0))          # pad 2nd dim (width) on the left
    x = x.view(B, P + 1, 2 * P - 1, H)    # [B, P+1, 2P-1, H]
    x = x[:, 1:]                          # Drop first slice to restore original length
    return x[:, :, :P]                  # Truncate to [B, P, P, H]


class row_attention_block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.dim = cfg.pair_dim  # Use pair_dim
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        q = self.q_proj(x)  # [B, P, P, dim]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Attention along the row dimension (last P)
        logits = torch.einsum('bpPf,bpkf->bpPk', q, k) / math.sqrt(self.dim)
        weights = F.softmax(logits, dim=-1)

        out = torch.einsum('bpPk,bpkf->bpPf', weights, v)
        return out  # [B, P, P, dim] = [B, 8, 8, 64]



class pair_MLP_block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(cfg.pair_dim, cfg.pair_dim * 2)
        self.linear2 = nn.Linear(cfg.pair_dim * 2, cfg.pair_dim)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))  # [B, P, P, pair_dim] = [B, 8, 8, 64]

 
# ----------------------------------------------- MHA + MLP Blocks ----------------------------------------------- #

class MHA_Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.heads = cfg.heads                  # 8
        self.q_dim = cfg.pair_dim               # 32
        self.v_dim = cfg.pair_dim * 3 // 2      # 48

        self.q_proj = nn.Linear(cfg.dim, self.heads * self.q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.dim, self.q_dim, bias=False)   # Single shared head
        self.v_proj = nn.Linear(cfg.dim, self.v_dim)               # Single shared head

        self.out_proj = nn.Linear(self.heads * self.v_dim, cfg.dim)
        self.norm = RMSNorm(cfg)
        self.cfg = cfg

    def forward(self, x, attention_bias):
        # x: [B, S, C]
        x = self.norm(x)
        B, S, _ = x.shape

        # === Project ===
        q = self.q_proj(x).reshape(B, S, self.heads, self.q_dim)   # [B, S, H, q_dim]
        k = self.k_proj(x)                                         # [B, S, q_dim]
        v = self.v_proj(x)                                         # [B, S, v_dim]

        # === Apply RoPE ===
        q = apply_rope(q, self.cfg)                                # [B, S, H, q_dim]
        k = apply_rope(k.unsqueeze(2), self.cfg).squeeze(2)        # [B, S, q_dim]

        # === Compute attention scores ===
        # Expand k: [B, S, q_dim] → [B, 1, S, q_dim] for broadcast
        k_exp = k.unsqueeze(1)                                     # [B, 1, S, q_dim]
        q_t = q.transpose(1, 2)                                    # [B, H, S, q_dim]
        attn_logits = torch.einsum("bhid,bjsd->bhij", q_t, k_exp)  # [B, H, S, S]

        # === Apply attention bias and softmax ===
        attn_logits = torch.tanh((attn_logits + attention_bias) / 5.0) * 5.0
        attn_weights = F.softmax(attn_logits, dim=-1)              # [B, H, S, S]

        # === Apply attention weights to shared value ===
        # v: [B, S, v_dim] → [B, 1, S, v_dim] for broadcast
        v_exp = v.unsqueeze(1)                                     # [B, 1, S, v_dim]
        context = torch.einsum("bhij,bjsd->bhid", attn_weights, v_exp)  # [B, H, S, v_dim]

        # === Final projection ===
        context = context.transpose(1, 2).reshape(B, S, -1)        # [B, S, H*v_dim]
        out = self.out_proj(context)                               # [B, S, C]
        return out




# def apply_rope(x: torch.Tensor, cfg: TransformerConfig, positions: torch.Tensor = None) -> torch.Tensor:
#     """
#     Applies Rotary Positional Embeddings (RoPE) to tensor `x` using configuration from `cfg`.
    
#     Args:
#         x: Tensor of shape [B, L, H, D] (batch, seq_len, heads, head_dim)
#         cfg: TransformerConfig object (must include rope_max_position)
#         positions: Optional tensor of shape [L], position indices. Defaults to range(L).
    
#     Returns:
#         Tensor of shape [B, L, H, D] with RoPE applied.
#     """
#     if positions is None:
#         positions = torch.arange(x.shape[1], device=x.device)  # [L]

#     D = x.shape[-1]
#     assert D % 2 == 0, "Last dimension (head_dim) must be even for RoPE"

#     num_freq = D // 2
#     max_position = cfg.rope_max_position

#     # Create frequency terms
#     freq = 1.0 / torch.logspace(
#     start=0,
#     end=math.log10(max_position),
#     steps=num_freq,
#     device=x.device
#     )

#     # Compute position-dependent angle matrix
#     theta = torch.einsum('s,f->sf', positions.float(), freq)  # [L, D//2]
#     theta = torch.cat([theta, theta], dim=-1)                 # [L, D]

#     # Split x into even/odd for rotation
#     x1, x2 = x[..., ::2], x[..., 1::2]                         # [B, L, H, D//2] each

#     # Broadcast sin/cos for rotation
#     sin_theta = torch.sin(theta)[None, :, None, :]            # [1, L, 1, D]
#     cos_theta = torch.cos(theta)[None, :, None, :]            # [1, L, 1, D]

#     # Apply RoPE
#     return torch.cat([
#         x1 * cos_theta - x2 * sin_theta,
#         x1 * sin_theta + x2 * cos_theta
#     ], dim=-1)                                                # [B, L, H, D]

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


class AttentionBiasBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.linear = nn.Linear(cfg.pair_dim, cfg.heads, bias=False)  # in_dim = pair_dim, out_heads = heads
        self.repeat_factor = cfg.repeat_factor
        self.cfg = cfg

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, P, P, pair_dim]  (pairwise activations)
        Returns:
            Tensor of shape [B, heads, S, S] for attention bias
        """
        x = self.linear(F.gelu(x))                          # [B, P, P, heads]
        x = x.repeat_interleave(self.repeat_factor, dim=1)  # Expand row
        x = x.repeat_interleave(self.repeat_factor, dim=2)  # Expand col
        return x.permute(0, 3, 1, 2)                        # [B, heads, S, S] = [B, 8, 8, 8]



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


