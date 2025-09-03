import math, os, time, csv
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

from mask_utils import mask_input_hf


# --------------------------- VEP Eval ---------------------------

def compute_log_odds_vep(model, tokenizer, seq, pos, ref, alt, device, min_length, max_length=8192):
    if len(seq) < min_length:
        return None
    if pos < 0 or pos >= len(seq) or seq[pos] != ref:
        return None
    if pos >= max_length:
        return None

    crop = seq[:max_length]
    crop_list = list(crop)
    crop_list[pos] = tokenizer.mask_token
    masked_seq = "".join(crop_list)

    enc = tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    input_ids = enc["input_ids"].to(device)

    mask_pos = (input_ids[0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        probs = F.softmax(logits[0, mask_pos], dim=-1)

    ref_id = tokenizer.convert_tokens_to_ids(ref)
    alt_id = tokenizer.convert_tokens_to_ids(alt)
    if ref_id is None or alt_id is None or ref_id < 0 or alt_id < 0:
        return None

    return (torch.log(probs[alt_id]) - torch.log(probs[ref_id])).item()

def run_vep_eval(model, tokenizer, device, df, step, csv_path, min_length, max_length=8192):
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
        # csv_exists = os.path.exists(csv_path)
        # pd.DataFrame([[step, auc]], columns=["step", "auc"]).to_csv(
        #     csv_path, mode='a', header=not csv_exists, index=False
        # )
        return auc
    else:
        print(f"[VEP Eval] Step {step}: Not enough valid data", flush=True)
        return None


# --------------------------- Eval ---------------------------

@torch.no_grad()
def evaluate_epoch(model, dl, vocab_size, mask_token_id, special_token_ids, allowed_random_ids, amp, mask_prob):
    model.eval()
    total_loss, total_batches = 0.0, 0
    for batch in dl:
        input_ids = batch["input_ids"].to(next(model.parameters()).device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(next(model.parameters()).device, non_blocking=True)

        masked, labels = mask_input_hf(
            input_ids, attention_mask,
            vocab_size, mask_token_id, special_token_ids,
            mask_prob=mask_prob, avoid_special_in_random=True,
            allowed_random_ids=allowed_random_ids
        )
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(masked)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)
