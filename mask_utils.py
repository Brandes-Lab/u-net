import torch
import torch.nn.functional as F
from typing import Optional


# --------------------------- Masking for MLM ---------------------------

def mask_input_hf(input_ids: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int, mask_token_id: int, special_token_ids: torch.Tensor,
                mask_prob: float = 0.15, avoid_special_in_random: bool = True, allowed_random_ids: Optional[torch.Tensor] = None):
    
    device = input_ids.device
    B, L = input_ids.shape

    is_special = torch.zeros_like(input_ids, dtype=torch.bool, device=device)
    for tid in special_token_ids.tolist():
        is_special |= (input_ids == tid)
    maskable = (attention_mask == 1) & (~is_special)

    probs = torch.rand((B, L), device=device)
    mask = (probs < mask_prob) & maskable

    labels = input_ids.clone()
    labels[~mask] = -100
    labels[attention_mask == 0] = -100

    rand = torch.rand((B, L), device=device)
    masked_input = input_ids.clone()

    masked_input[mask & (rand < 0.8)] = mask_token_id

    replace_random = mask & (rand >= 0.8) & (rand < 0.9)
    if avoid_special_in_random:
        idx = torch.randint(0, allowed_random_ids.numel(), size=(B, L), device=device)
        rnd = allowed_random_ids[idx]
        masked_input[replace_random] = rnd[replace_random]
    else:
        rnd = torch.randint(0, vocab_size, size=(B, L), device=device)
        masked_input[replace_random] = rnd[replace_random]

    return masked_input, labels

