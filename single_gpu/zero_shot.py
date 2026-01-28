"""
Zero-shot Variant Effect Prediction (VEP) evaluation.

This module provides functions to evaluate protein language models on their ability
to predict the pathogenicity of amino acid variants using a zero-shot approach
(log-odds scoring of masked position predictions).

Includes both single-sequence and batched implementations for efficiency.
"""

import math
import time
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


# --------------------------- Single-sequence VEP ---------------------------

def compute_log_odds_vep(
    model, 
    tokenizer, 
    seq: str, 
    pos: int, 
    ref: str, 
    alt: str, 
    device: torch.device, 
    min_length: int, 
    max_length: int = 8192,
    eps: float = 1e-10
) -> Optional[float]:
    """
    Compute log-odds score for a single variant.
    
    Args:
        model: The protein language model
        tokenizer: Tokenizer for the model
        seq: Protein sequence
        pos: Position of the variant (0-indexed)
        ref: Reference amino acid
        alt: Alternate amino acid
        device: Device to run inference on
        min_length: Minimum sequence length
        max_length: Maximum sequence length (crop longer sequences)
        eps: Small epsilon for numerical stability
        
    Returns:
        Log-odds score (alt vs ref), or None if invalid
    """
    if len(seq) < min_length:
        return None
    if pos < 0 or pos >= len(seq) or seq[pos] != ref:
        return None
    if pos >= max_length:
        return None

    # Crop sequence and create masked version
    crop = seq[:max_length]
    crop_list = list(crop)
    crop_list[pos] = tokenizer.mask_token
    masked_seq = "".join(crop_list)

    # Tokenize
    enc = tokenizer(
        masked_seq, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length, 
        padding=False
    )
    input_ids = enc["input_ids"].to(device)

    # Find mask position
    mask_pos = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_pos) == 0:
        return None
    mask_pos = mask_pos.item()

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_ids)
        
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output
            
        probs = F.softmax(logits[0, mask_pos], dim=-1)

    # Get token IDs for ref and alt
    ref_id = tokenizer.convert_tokens_to_ids(ref)
    alt_id = tokenizer.convert_tokens_to_ids(alt)
    
    if ref_id is None or alt_id is None or ref_id < 0 or alt_id < 0:
        return None

    # Compute log-odds with numerical stability
    log_odds = (torch.log(probs[alt_id] + eps) - torch.log(probs[ref_id] + eps)).item()
    
    if math.isnan(log_odds) or math.isinf(log_odds):
        return None
        
    return log_odds


# --------------------------- Batched VEP ---------------------------

def prepare_vep_batch(
    df: pd.DataFrame,
    tokenizer,
    min_length: int,
    max_length: int = 8192
) -> Tuple[List[dict], List[int]]:
    """
    Prepare VEP data for batched processing.
    
    Returns:
        valid_items: List of dicts with keys:
            - 'masked_seq': The sequence with mask token at variant position
            - 'mask_pos_in_seq': Position of mask in the sequence
            - 'ref_id': Token ID for reference amino acid
            - 'alt_id': Token ID for alternate amino acid
            - 'original_idx': Original index in dataframe
            - 'label': Ground truth label
            - 'seq_len': Length of the sequence
        invalid_indices: List of indices that were skipped
    """
    valid_items = []
    invalid_indices = []
    
    for idx, row in df.iterrows():
        seq = row["sequence"]
        pos = int(row["pos"])
        ref = row["ref"]
        alt = row["alt"]
        label = int(row["label"])
        
        # Validation checks
        if len(seq) < min_length:
            invalid_indices.append(idx)
            continue
        if pos < 0 or pos >= len(seq) or seq[pos] != ref:
            invalid_indices.append(idx)
            continue
        if pos >= max_length:
            invalid_indices.append(idx)
            continue
        
        # Get token IDs
        ref_id = tokenizer.convert_tokens_to_ids(ref)
        alt_id = tokenizer.convert_tokens_to_ids(alt)
        if ref_id is None or alt_id is None or ref_id < 0 or alt_id < 0:
            invalid_indices.append(idx)
            continue
        
        # Create masked sequence
        crop = seq[:max_length]
        crop_list = list(crop)
        crop_list[pos] = tokenizer.mask_token
        masked_seq = "".join(crop_list)
        
        valid_items.append({
            'masked_seq': masked_seq,
            'mask_pos_in_seq': pos,
            'ref_id': ref_id,
            'alt_id': alt_id,
            'original_idx': idx,
            'label': label,
            'seq_len': len(crop),
        })
    
    return valid_items, invalid_indices


def run_batched_vep_inference(
    model,
    tokenizer,
    device,
    valid_items: List[dict],
    batch_size: int = 64,
    max_length: int = 8192,
    eps: float = 1e-10
) -> List[Optional[float]]:
    """
    Run batched inference for VEP scoring.
    
    Groups sequences by similar length to minimize padding waste,
    then runs batched forward passes.
    """
    if not valid_items:
        return []
    
    model.eval()
    
    # Sort by sequence length for efficient batching
    sorted_indices = sorted(range(len(valid_items)), key=lambda i: valid_items[i]['seq_len'])
    
    scores = [None] * len(valid_items)
    
    # Process in batches
    for batch_start in range(0, len(sorted_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(sorted_indices))
        batch_indices = sorted_indices[batch_start:batch_end]
        
        batch_seqs = [valid_items[i]['masked_seq'] for i in batch_indices]
        
        # Tokenize batch with padding
        encodings = tokenizer(
            batch_seqs,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        # Find mask positions
        mask_positions = []
        for b in range(input_ids.size(0)):
            mask_pos = (input_ids[b] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_pos) > 0:
                mask_positions.append(mask_pos[0].item())
            else:
                mask_positions.append(-1)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
            
            if isinstance(output, dict):
                logits = output["logits"]
            else:
                logits = output
        
        # Extract scores
        for b, orig_idx in enumerate(batch_indices):
            mask_pos = mask_positions[b]
            
            if mask_pos < 0:
                scores[orig_idx] = None
                continue
            
            item = valid_items[orig_idx]
            ref_id = item['ref_id']
            alt_id = item['alt_id']
            
            probs = F.softmax(logits[b, mask_pos], dim=-1)
            log_odds = (torch.log(probs[alt_id] + eps) - torch.log(probs[ref_id] + eps)).item()
            
            if math.isnan(log_odds) or math.isinf(log_odds):
                scores[orig_idx] = None
            else:
                scores[orig_idx] = log_odds
    
    return scores


def run_vep_eval_batched(
    model,
    tokenizer,
    device,
    df: pd.DataFrame,
    step: int,
    csv_path: Optional[str],
    min_length: int,
    max_length: int = 8192,
    batch_size: int = 64,
) -> Optional[float]:
    """
    Run batched VEP evaluation.
    
    Returns:
        AUC-ROC score, or None if not enough valid data
    """
    start_time = time.time()
    
    print(f"Running batched zero-shot VEP evaluation @ step {step}", flush=True)
    
    # Prepare data
    valid_items, invalid_indices = prepare_vep_batch(df, tokenizer, min_length, max_length)
    
    print(f"  Valid sequences: {len(valid_items):,}", flush=True)
    print(f"  Invalid/skipped: {len(invalid_indices):,}", flush=True)
    
    if len(valid_items) == 0:
        print(f"[VEP Eval] Step {step}: No valid sequences", flush=True)
        return None
    
    # Run batched inference
    scores = run_batched_vep_inference(
        model, tokenizer, device, valid_items, batch_size, max_length
    )
    
    # Map scores back to original indices
    log_odds_scores = [None] * len(df)
    labels = [None] * len(df)
    
    for item, score in zip(valid_items, scores):
        orig_idx = item['original_idx']
        if isinstance(orig_idx, int):
            pos = orig_idx
        else:
            pos = df.index.get_loc(orig_idx)
        log_odds_scores[pos] = score
        labels[pos] = item['label']
    
    df_out = df.copy()
    df_out["log_odds"] = log_odds_scores
    
    # Compute AUC
    valid_mask = df_out["log_odds"].notnull()
    valid_count = valid_mask.sum()
    
    elapsed = time.time() - start_time
    
    if valid_count >= 10 and len(set(df_out.loc[valid_mask, "label"])) > 1:
        auc = roc_auc_score(
            df_out.loc[valid_mask, "label"],
            -df_out.loc[valid_mask, "log_odds"]  # Negative: lower log-odds = pathogenic
        )
        print(f"[VEP Eval] Step {step}: AUC = {auc:.4f} ({valid_count:,} variants, {elapsed:.1f}s)", flush=True)
        return auc
    else:
        print(f"[VEP Eval] Step {step}: Not enough valid data ({valid_count} variants)", flush=True)
        return None


# --------------------------- Main Entry Point ---------------------------

def run_vep_eval(
    model,
    tokenizer,
    device,
    df: pd.DataFrame,
    step: int,
    csv_path: Optional[str],
    min_length: int,
    max_length: int = 8192,
    batch_size: int = 64,
    use_batched: bool = True,
) -> Optional[float]:
    """
    Run VEP evaluation - main entry point.
    
    By default uses batched evaluation for speed.
    """
    if use_batched:
        return run_vep_eval_batched(
            model, tokenizer, device, df, step, csv_path,
            min_length, max_length, batch_size
        )
    else:
        # Single-sequence implementation (slower)
        print(f"Running zero-shot VEP evaluation @ step {step}", flush=True)
        log_odds_scores = []
        labels = []

        for _, row in df.iterrows():
            score = compute_log_odds_vep(
                model, tokenizer,
                row["sequence"], int(row["pos"]),
                row["ref"], row["alt"],
                device, min_length, max_length
            )
            log_odds_scores.append(score)
            labels.append(int(row["label"]))

        df_out = df.copy()
        df_out["log_odds"] = log_odds_scores

        valid_mask = df_out["log_odds"].notnull()
        if valid_mask.sum() >= 10 and len(set(df_out.loc[valid_mask, "label"])) > 1:
            auc = roc_auc_score(
                df_out.loc[valid_mask, "label"],
                -df_out.loc[valid_mask, "log_odds"]
            )
            print(f"[VEP Eval] Step {step}: AUC = {auc:.4f}", flush=True)
            return auc
        else:
            print(f"[VEP Eval] Step {step}: Not enough valid data", flush=True)
            return None