# train.py
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
import pandas as pd
from typing import List, Dict, Optional
from datasets import load_dataset
from model import ModelConfig, ProteinMLMModel
from itertools import islice
import argparse
from zero_shot_vep import run_vep_eval, evaluate_epoch
from mask_utils import mask_input_hf
import wandb
import math, os, time, csv  

# --------------------------- Small utils ---------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def append_csv_row(path: str, fieldnames: List[str], row: Dict):
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row)

# --------------------------- Collator ---------------------------

class TokenizePadCollator:
    def __init__(self, tokenizer, max_length=512, pad_to="longest"):
        assert pad_to in ("longest", "max_length")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to = pad_to

    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=self.pad_to,
            return_tensors="pt",
        )
        return enc


# --------------------------- Main ---------------------------

def main():
    

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args_cli = parser.parse_args()

    args = {
        "tokenizer_dir": "/gpfs/data/brandeslab/Project/HuggingfaceTransformer/char_tokenizer",
        "train_file": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/train/train_10M.txt",
        "val_file": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/val/val.txt",
        "log_dir": "./logs_unet",
        "batch_size": 64,
        "epochs": 500,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "amp": False,
        "base_dim": 256,
        "growth": 64,
        "num_scales": 4,
    }

    run_name = "unet_10M_seqs"
    wandb.init(project="huggingface_bert_sweep", name=run_name, entity="sinha-anushka12-na")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Device: {device}", flush=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args["tokenizer_dir"])
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id
    special_ids = tokenizer.all_special_ids

    cfg = ModelConfig(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        base_dim=args["base_dim"],
        growth=args["growth"],
        num_scales=args["num_scales"],
    )
    model = ProteinMLMModel(cfg).to(device)

    min_len = 2 ** cfg.num_scales

    # Dataset
    train_stream = load_dataset("text", data_files={"train": args["train_file"]}, split="train", streaming=True)
    # filter out short sequences (< min_len)
    train_stream = train_stream.filter(lambda ex: len(ex["text"]) >= min_len)
    shuffled_train_stream = train_stream.shuffle(buffer_size=100_000, seed=42)
    
    val_dataset = load_dataset("text", data_files={"val": args["val_file"]}, split="val", streaming=True)
    # Filter out short sequences
    val_dataset = val_dataset.filter(lambda ex: len(ex["text"]) >= min_len)

    collator = TokenizePadCollator(tokenizer, max_length=512, pad_to="longest")

    train_dl = DataLoader(
        shuffled_train_stream, batch_size=args["batch_size"], num_workers=0,
        collate_fn=collator, drop_last=True, pin_memory=(device == "cuda")
    )
    val_dl = DataLoader(val_dataset, batch_size=args["batch_size"], num_workers=0, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"])
    special_token_ids = torch.tensor(special_ids, device=device, dtype=torch.long)
    allowed_random_ids = torch.tensor(
        [i for i in range(vocab_size) if i not in set(special_token_ids.tolist())],
        device=device, dtype=torch.long
    )

    log_dir = args["log_dir"]
    ensure_dir(log_dir)
    train_csv = os.path.join(log_dir, "train_losses.csv")
    val_csv = os.path.join(log_dir, "val_losses.csv")
    vep_csv = os.path.join(log_dir, "vep_eval.csv")

    vep_df = pd.read_csv("/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv")
    global_step = 0
    best_val = math.inf

    for epoch in range(1, args["epochs"] + 1):
        print(f"\n=== Epoch {epoch}/{args['epochs']} ===", flush=True)
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_dl, start=1):
            global_step += 1
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            masked, labels = mask_input_hf(
                input_ids, attention_mask,
                vocab_size, mask_token_id, special_token_ids,
                mask_prob=cfg.mask_prob, avoid_special_in_random=True,
                allowed_random_ids=allowed_random_ids
            )

            with torch.amp.autocast(device_type='cuda', enabled=args["amp"]):
                logits = model(masked)
                logits_flat = logits.view(-1, vocab_size)              # [B*L, V]
                labels_flat = labels.view(-1)                          # [B*L]
                mask = labels_flat != -100
                masked_logits = logits_flat[mask]
                masked_labels = labels_flat[mask]
                loss = F.cross_entropy(masked_logits, masked_labels, reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # --- Debug info at first step ---
            if global_step == 1:
                print("Logits shape:", logits.shape)
                print("Unique labels:", torch.unique(labels))
                print("Masked tokens:", (labels != -100).sum().item())
                print("Batch tokens:", labels.numel())

            # --- Train loss every 50 steps ---
            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                print(f"[Train step {global_step}] avg_loss={avg_loss:.4f}", flush=True)
                # append_csv_row(train_csv, ["step", "train_loss"],
                #             {"step": global_step, "train_loss": float(avg_loss)})
                wandb.log({
                        "train_loss": avg_loss,
                        "epoch": epoch,
                        "elapsed_hours": (time.time() - start_time) / 3600,},
                    step=global_step)
                running_loss = 0.0

            # --- Val loss every 10000 steps ---
            if global_step % 10000 == 0:
                val_loss = evaluate_epoch(model, val_dl, vocab_size, mask_token_id,
                                        special_token_ids, allowed_random_ids, args["amp"], cfg.mask_prob)
                best_val = min(best_val, val_loss)
                print(f"[Val step {global_step}] loss={val_loss:.4f} (best={best_val:.4f})", flush=True)
                # append_csv_row(val_csv, ["step", "val_loss"],
                #             {"step": global_step, "val_loss": float(val_loss)})
                wandb.log({
                    "val_loss": val_loss,
                    "epoch": epoch,
                    "elapsed_hours": (time.time() - start_time) / 3600,}, 
                    step= global_step)

            # --- VEP eval every 5000 steps ---
            if global_step % 5000 == 0:
                auc = run_vep_eval(model, tokenizer, device, vep_df, global_step, vep_csv,
                       min_length=min_len, max_length=512)
                print(f"AUC {auc}")
                if auc is not None:
                    print(f"AUC is none")
                    wandb.log({
                        "zero_shot_vep_auc": auc,
                        "epoch": epoch,
                        "elapsed_hours": (time.time() - start_time) / 3600,}, 
                        step=global_step)
                


if __name__ == "__main__":
    main()
