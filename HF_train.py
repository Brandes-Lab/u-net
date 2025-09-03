import os, math, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling
)
# from transformers.trainer_utils.length_grouping import get_length_grouped_indices

from model import ModelConfig, ProteinMLMModel
from zero_shot_vep import run_vep_eval, evaluate_epoch
import wandb


import numpy as np
import random

# def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=50, generator=None):
#     """
#     Reimplementation of HuggingFace's get_length_grouped_indices

#     Args:
#         lengths (List[int]): list of sequence lengths
#         batch_size (int): batch size used for training
#         mega_batch_mult (int): multiplier to define the size of "mega-batches"
#         generator (random.Random or np.random.Generator): optional seeded generator for reproducibility

#     Returns:
#         List[int]: reordered indices with local length grouping
#     """
#     indices = list(range(len(lengths)))

#     # Shuffle the full index list first
#     if generator is None:
#         random.shuffle(indices)
#     else:
#         generator.shuffle(indices)

#     # Split into mega-batches
#     mega_batch_size = batch_size * mega_batch_mult
#     megabatches = [indices[i:i + mega_batch_size] for i in range(0, len(indices), mega_batch_size)]

#     # Sort each megabatch by descending length
#     def sort_by_length(batch):
#         return sorted(batch, key=lambda idx: -lengths[idx])

#     sorted_megabatches = [sort_by_length(mb) for mb in megabatches]

#     # Flatten back into a single list
#     return [idx for mb in sorted_megabatches for idx in mb]

def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=50, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - sorted by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """
    # Default for mega_batch_mult: 50 or the number to get 4 megabatches, whichever is smaller.
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        # Just in case, for tiny datasets
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]

    # The rest is to get the biggest batch first.
    # Since each megabatch is sorted by descending length, the longest element is the first
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    # Switch to put the longest element in first position
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]


# --------------------------- Main ---------------------------


def main():
    args = {
        "tokenizer_dir": "/gpfs/data/brandeslab/Project/HuggingfaceTransformer/char_tokenizer",
        "train_data_path": "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train",
        "val_data_path": "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation",
        "batch_size": 64,
        "epochs": 500,
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "amp": False,
        "base_dim": 256,
        "growth": 64,
        "num_scales": 4,
    }

    start_time = time.time()
    run_name = "unet_pre_tokenized"
    wandb.init(project="huggingface_bert_sweep", name=run_name, entity="sinha-anushka12-na")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Device: {device}", flush=True)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args["tokenizer_dir"])
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id

    cfg = ModelConfig(
        vocab_size=vocab_size,
        mask_token_id=mask_token_id,
        base_dim=args["base_dim"],
        growth=args["growth"],
        num_scales=args["num_scales"]
    )
    model = ProteinMLMModel(cfg).to(device)

    # Discard sequences shorter than the model's minimum length
    min_len = 2 ** cfg.num_scales

    # Load pre-tokenized datasets from disk
    train_ds = load_from_disk(args["train_data_path"])
    # train_ds = train_ds.select(range(min(len(train_ds), 10_000)))

    # val_ds = load_from_disk(args["val_data_path"])

    # Filter out short sequences that can't be downsampled
    train_ds = train_ds.filter(
    lambda x: x["length"] >= min_len,
    num_proc=16,                # or fewer depending on available CPUs
    desc="Filtering short sequences"
    )
    # lengths = np.array(train_ds["length"])
    # valid_indices = np.where(lengths >= min_len)[0]
    # train_ds = train_ds.select(valid_indices)
    # train_ds = train_ds.filter(lambda x: x["length"] >= min_len)
    # val_ds = val_ds.filter(lambda x: x["length"] >= min_len)
    
    # Batch sequences of similar lengths together
    train_lengths = train_ds["length"]  
    # Generate a list of shuffled + length-grouped indices:
    # - First shuffles all indices randomly
    # - Then breaks into mega-batches (e.g. 50 × batch_size)
    # - Then sorts each mega-batch by descending length (local sorting)
    grouped_indices = get_length_grouped_indices(train_lengths, args["batch_size"])
    # Reorder the dataset using the grouped indices
    # The resulting dataset has roughly similar-length sequences near each other,
    # so the DataLoader can form more efficient mini-batches (less padding)
    train_ds = train_ds.select(grouped_indices)     # creates a new dataset where rows are reordered based on the grouped_indices list.

    # Huggingface's built-in MLM collator handles masking + labels
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mask_prob
    )

    # DataLoaders for train/val
    train_dl = DataLoader(
        train_ds, shuffle=False, batch_size=args["batch_size"],
        collate_fn=collator, num_workers=16, pin_memory=(device == "cuda")
    )

    # === Inspect first 100 batches ===
    pad_id = tokenizer.pad_token_id
    batch_lens = []

    print("\nInspecting first 100 batches...\n")
    for i, batch in enumerate(train_dl):
        input_ids = batch["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            lengths = (input_ids != pad_id).sum(dim=1).tolist()
            min_len = min(lengths)
            max_len = max(lengths)
            batch_lens.append((i, min_len, max_len))

            print(f"Batch {i:03d} → min length = {min_len}, max length = {max_len}, batch size = {len(lengths)}")

        if i >= 100:
            break


    if batch_lens:
        batch_ids, min_lens, max_lens = zip(*batch_lens)
        plt.figure(figsize=(10, 4))
        plt.plot(batch_ids, max_lens, label='Max length')
        plt.plot(batch_ids, min_lens, label='Min length')
        plt.fill_between(batch_ids, min_lens, max_lens, alpha=0.2, label='Length spread')
        plt.xlabel("Batch Index")
        plt.ylabel("Sequence Length")
        plt.title("Min/Max Token Length per Batch (after length grouping)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("unet_group_by_len.png")
        plt.show()

    # val_dl = DataLoader(
    #     val_ds, shuffle=False, batch_size=args["batch_size"],
    #     collate_fn=collator, num_workers=4, pin_memory=(device == "cuda")
    # )

    # AdamW optimizer 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args["lr"],
        weight_decay=args["weight_decay"]
    )

    # Load ClinVar dataset for zero-shot VEP evaluation
    vep_df = pd.read_csv("/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv")
    global_step = 0
    best_val = math.inf

     # ======================== Training Loop ========================

    for epoch in range(1, args["epochs"] + 1):
        print(f"\n=== Epoch {epoch}/{args['epochs']} ===", flush=True)
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_dl, start=1):
            global_step += 1

            # Transfer batch to GPU (non-blocking for performance)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Forward pass with AMP if enabled
            with torch.amp.autocast(device_type='cuda', enabled=args["amp"]):               # Use automatic mixed precision for this block if I'm training on CUDA and the --amp flag is set.
                logits = model(input_ids)                                                   # input_ids: Tensor of shape [batch_size, sequence_length]
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean"
                )

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # --- First step debug info ---
            if global_step == 1:
                print("Logits shape:", logits.shape)
                print("Unique labels:", torch.unique(labels))
                print("Masked tokens:", (labels != -100).sum().item())
                print("Batch tokens:", labels.numel())

            # --- Log training loss every 50 steps ---
            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                print(f"[Train step {global_step}] avg_loss={avg_loss:.4f}", flush=True)
                wandb.log({
                    "train_loss": avg_loss,
                    "epoch": epoch,
                    "elapsed_hours": (time.time() - start_time) / 3600,
                }, step=global_step)
                running_loss = 0.0

            # # --- Evaluate on validation set every 10,000 steps ---
            # if global_step % 10000 == 0:
            #     val_loss = evaluate_epoch(model, val_dl, vocab_size, args["amp"])
            #     best_val = min(best_val, val_loss)
            #     print(f"[Val step {global_step}] loss={val_loss:.4f} (best={best_val:.4f})", flush=True)
            #     wandb.log({
            #         "val_loss": val_loss,
            #         "epoch": epoch,
            #         "elapsed_hours": (time.time() - start_time) / 3600,
            #     }, step=global_step)

            # --- Run zero-shot VEP evaluation every 5,000 steps ---
            if global_step % 10000 == 0:
                auc = run_vep_eval(
                    model, tokenizer, device,
                    df=vep_df,
                    step=global_step,
                    csv_path=None,
                    min_length=min_len,
                    max_length=8192
                )
                print(f"AUC {auc}")
                if auc is not None:
                    wandb.log({
                        "zero_shot_vep_auc": auc,
                        "epoch": epoch,
                        "elapsed_hours": (time.time() - start_time) / 3600,
                    }, step=global_step)


if __name__ == "__main__":
    main()