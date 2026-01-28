# =========================
# HF_ddp.py (DDP-integrated)
# =========================

# --- Imports (every line explained) ---
import os  # standard lib: access env vars (LOCAL_RANK, MASTER_ADDR/PORT), paths for save_model
import time  # standard lib: timing for elapsed-hours logging / evaluation timers
import random  # standard lib: ensure python-level RNG is seeded identically on all ranks for determinism

import torch  # PyTorch core: tensors, distributed init/introspection, device utilities
import torch.nn.functional as F  # PyTorch functional API: cross_entropy for MLM loss
import torch.distributed as dist  # PyTorch distributed: DDP helpers (is_initialized, get_rank, barrier, all_gather_object)

import numpy as np  # numerical utils: used for simple stats and array slicing
import pandas as pd  # tabular IO: read ClinVar CSV for zero-shot VEP evaluation

from datasets import load_from_disk  # HuggingFace Datasets: memory-mapped datasets from disk

# Transformers bits: tokenizer, data collator, Trainer APIs, and callbacks
from transformers import (
    PreTrainedTokenizerFast,            # fast tokenizer loader from a local directory
    DataCollatorForLanguageModeling,    # builds masked LM batches (applies [MASK] and labels with -100 for ignore)
    Trainer,                            # high-level training loop (DDP-aware)
    TrainingArguments,                  # configuration object for Trainer (DDP, logging, saving, etc.)
    TrainerCallback                     # base class for custom callbacks (logging, VEP, etc.)
)

# Your local modules (unchanged APIs)
from model_650m import ModelConfig, ProteinMLMModel  # your model config and model
from zero_shot_vep import run_vep_eval          # your zero-shot VEP implementation (single-process version)
import wandb  # experiment tracking

# -----------------------------
# DDP helpers & rank utilities
# -----------------------------

def ddp_is_initialized() -> bool:
    """Return True if torch.distributed is initialized."""
    # We call dist.is_available() first to avoid exceptions on systems without distributed compiled in
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    """Return this process's rank or 0 if not in DDP."""
    # If DDP is not initialized, treat as single-process rank 0
    return dist.get_rank() if ddp_is_initialized() else 0

def get_world_size() -> int:
    """Return world size or 1 if not in DDP."""
    # World size is total number of processes participating in DDP
    return dist.get_world_size() if ddp_is_initialized() else 1

def barrier():
    """Cross-process synchronization barrier (no-op if not in DDP)."""
    # Useful to keep side-effects (like writing files) in a consistent order
    if ddp_is_initialized():
        dist.barrier()

def print_rank0(*args, **kwargs):
    """Print only from rank 0 to avoid duplicated stdout."""
    if get_rank() == 0:
        print(*args, **kwargs)

# -------------------------------
# Length grouping (DDP-friendly)
# -------------------------------

def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=50, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices has similar lengths.

    DDP NOTE:
      - We ensure the same RNG seed across ranks so each process computes identical `indices`.
      - That keeps the dataset ordering consistent before the Trainer’s DDP samplers shard it per rank.
    """
    # If user didn't pass a generator, create a deterministic one with a fixed seed so *all ranks* produce same order
    if generator is None:
        generator = torch.Generator()
        # Fixed seed for reproducibility; change if you want a different shuffle each run
        generator.manual_seed(42)

    # Original heuristics preserved
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    # identical random permutation across ranks thanks to fixed generator
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size
    # slice into megabatches
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    # sort each megabatch by length (desc) so OOM—if any—happens early
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]

    # place the global longest item as the first example overall
    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    # flatten back into a single list of indices
    return [i for megabatch in megabatches for i in megabatch]

# ----------------------------------------
# Custom Trainer with explicit MLM loss
# ----------------------------------------

class ProteinMLMTrainer(Trainer):
    """Custom Trainer that handles the MLM loss computation (unchanged, DDP-safe)."""

    def __init__(self, *args, debug_mode=False, **kwargs):
        # forward all Trainer init params to base class
        super().__init__(*args, **kwargs)
        # store debug flag and a counter for first few steps of verbose stat printing
        self.debug_mode = debug_mode
        self.step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the masked LM loss explicitly via F.cross_entropy to match your original loop.
        This runs on each rank in DDP; gradients are reduced by DDP automatically.
        """
        # fetch token IDs and labels from the batch dict
        input_ids = inputs["input_ids"]
        labels = inputs["labels"]

        # forward pass: your model signature expects token_ids=...
        logits = model(token_ids=input_ids)

        # cross entropy over vocabulary, ignoring non-masked tokens labeled as -100
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction="mean"
        )

        # lightweight debug prints for first few steps per rank (kept as in your file)
        if self.debug_mode and self.step_count < 5:
            self.step_count += 1
            num_masked = (labels != -100).sum().item()
            total_tokens = labels.numel()
            print_rank0(f"\n=== DEBUG Step {self.step_count} ===")
            print_rank0(f"Loss: {loss.item():.6f}")
            print_rank0(f"Logits shape: {logits.shape}")
            print_rank0(f"Labels shape: {labels.shape}")
            print_rank0(f"Masked tokens: {num_masked} / {total_tokens} ({num_masked/total_tokens:.2%})")
            print_rank0(f"Vocab size: {vocab_size}")
            print_rank0(f"Logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
            print_rank0(f"Logits mean/std: {logits.mean().item():.4f} / {logits.std().item():.4f}")
            print_rank0(f"Expected initial loss (random): ~{torch.log(torch.tensor(vocab_size, dtype=torch.float32)).item():.2f}")

        # return semantics unchanged
        return (loss, {"logits": logits}) if return_outputs else loss

# ----------------------------------------
# DDP-aware zero-shot VEP evaluation
# ----------------------------------------

class ZeroShotVEPEvaluationCallback(TrainerCallback):
    """
    DDP-safe VEP callback:
      - Shards rows by rank
      - Computes per-rank scores
      - all_gather_object() to rank 0
      - Rank 0 computes AUC + logs to W&B
    """

    def __init__(self, tokenizer, input_csv, max_len=8192, eval_every_n_steps=10000, batch_size=8):
        # keep tokenizer to build masked inputs
        self.tokenizer = tokenizer
        # path to ClinVar-like CSV with columns: sequence, pos, ref, alt, label
        self.input_csv = input_csv
        # truncate long sequences safely
        self.max_len = max_len
        # run cadence in steps
        self.eval_every_n_steps = eval_every_n_steps
        # micro-batch size for inference on each rank
        self.batch_size = batch_size
        # wall-clock timer origin
        self.start_time = time.time()
        # load dataframe once (each rank loads, inexpensive with memory mapping or local FS)
        self.df = pd.read_csv(
            input_csv,
            usecols=["sequence", "pos", "ref", "alt", "label"],
            dtype={"pos": np.int32, "label": np.int8},
        )

    def _compute_log_odds_batch(self, model, seqs, poses, refs, alts):
        """
        Core batch: mask the ref position, score alt vs ref from MLM logits at [MASK].
        Returns: list of log-odds (or None) aligned with input mini-batch.
        """
        # build masked sequences and remember which are valid to map outputs back
        masked_seqs, valid_indices = [], []
        for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
            # skip overly long sequences
            if len(seq) > self.max_len:
                continue
            # skip if position out of range or ref mismatch
            if pos >= len(seq) or seq[pos] != ref:
                continue
            # replace ref AA with [MASK]
            masked_seq = list(seq)
            masked_seq[pos] = self.tokenizer.mask_token
            masked_seqs.append("".join(masked_seq))
            valid_indices.append(i)

        # if all invalid, return all None to preserve alignment
        if not masked_seqs:
            return [None] * len(seqs)

        # tokenize the masked sequences to tensors on the correct device
        inputs = self.tokenizer(
            masked_seqs,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_len,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # forward pass without grad
        with torch.no_grad():
            logits = model(**inputs)

        # find which token is [MASK] in each row
        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)

        # prefill with None per input slot
        results = [None] * len(seqs)
        for _, (batch_idx, token_idx) in enumerate(mask_indices):
            # map local-batch index back to original sub-batch index
            input_idx = valid_indices[batch_idx]
            # convert ref/alt amino acids to vocab ids
            ref_id = self.tokenizer.convert_tokens_to_ids(refs[input_idx])
            alt_id = self.tokenizer.convert_tokens_to_ids(alts[input_idx])
            # skip if any token not in vocab
            if ref_id is None or alt_id is None:
                continue
            # probabilities at the masked position
            prob = torch.nn.functional.softmax(logits[batch_idx, token_idx], dim=0)
            # log-odds score (alt vs ref)
            log_odds = (torch.log(prob[alt_id]) - torch.log(prob[ref_id])).item()
            # store into slot aligned to original mini-batch
            results[input_idx] = log_odds

        return results

    def _run_once(self, model, step_id):
        """
        One full DDP evaluation pass across all rows:
          - each rank processes a strided shard
          - gather predictions to rank 0
          - rank 0 computes AUC and logs
        """
        # capture rank and world_size for messages/sharding
        rank, world_size = get_rank(), get_world_size()
        print_rank0(f"[VEP] World size = {world_size}")

        # elapsed time for logs
        elapsed_hours = (time.time() - self.start_time) / 3600
        start_time = time.time()

        # columns to numpy for fast slicing
        seqs = self.df["sequence"].values
        poses = self.df["pos"].values
        refs  = self.df["ref"].values
        alts  = self.df["alt"].values
        labels = self.df["label"].to_numpy(dtype=np.int8)

        # total examples and this-rank strided indices
        n = len(labels)
        all_idx = np.arange(n)
        shard_idx = all_idx[rank::world_size]

        # allocate predictions for this shard (nan for missing)
        preds_shard = np.full(len(shard_idx), np.nan, dtype=np.float32)

        # keep model mode and switch to eval
        was_training = model.training
        model.eval()
        try:
            # minibatch loop over this shard only
            for i in range(0, len(shard_idx), self.batch_size):
                batch_ids   = shard_idx[i: i + self.batch_size]
                batch_seqs  = seqs[batch_ids]
                batch_poses = poses[batch_ids]
                batch_refs  = refs[batch_ids]
                batch_alts  = alts[batch_ids]

                # compute per-example masked log-odds
                scores = self._compute_log_odds_batch(model, batch_seqs, batch_poses, batch_refs, batch_alts)

                # fill shard predictions; your convention uses negative log-odds
                for j, s in enumerate(scores):
                    if s is not None:
                        preds_shard[i + j] = -float(s)

                # occasional progress trace (per rank)
                if (i + self.batch_size) % 5000 < self.batch_size:
                    print(f"[Rank {rank}] VEP progress: {i + self.batch_size}/{len(shard_idx)}", flush=True)
        finally:
            # restore train mode if it was training
            if was_training:
                model.train()

        # --- Gather to rank 0 ---
        # Convert to python lists for all_gather_object (works for variable-length shards)
        preds_list   = preds_shard.tolist()
        labels_list  = labels[shard_idx].tolist()
        idx_list     = shard_idx.tolist()

        # Prepare receive containers length = world_size
        gathered_preds   = [None for _ in range(world_size)]
        gathered_labels  = [None for _ in range(world_size)]
        gathered_indices = [None for _ in range(world_size)]

        # all_gather_object collects lists of arbitrary python objects per rank
        dist.all_gather_object(gathered_preds,   preds_list)   if ddp_is_initialized() else None
        dist.all_gather_object(gathered_labels,  labels_list)  if ddp_is_initialized() else None
        dist.all_gather_object(gathered_indices, idx_list)     if ddp_is_initialized() else None

        # only rank 0 assembles and logs
        if get_rank() == 0:
            # flat array for all predictions
            flat_preds = np.full(n, np.nan, dtype=np.float32)
            # scatter each rank’s shard back to global index slots
            for preds, idxs in zip(gathered_preds, gathered_indices):
                if preds is None or idxs is None:
                    continue
                flat_preds[np.array(idxs)] = np.array(preds, dtype=np.float32)

            # mask rows with valid predictions
            mask = ~np.isnan(flat_preds)
            # guard: enough positives and negatives to compute ROC AUC
            if mask.sum() >= 10 and (labels[mask].min() != labels[mask].max()):
                from sklearn.metrics import roc_auc_score  # local import to avoid overhead if never used
                auc = roc_auc_score(labels[mask], flat_preds[mask])
                print(f"[VEP] AUC at step {step_id}: {auc:.4f}", flush=True)
                # Log only from rank 0
                if wandb.run is not None:
                    wandb.log({"zero_shot_vep_auc": auc, "step": step_id, "elapsed_hours": elapsed_hours})
            else:
                print(f"[VEP] Skipping AUC at step {step_id} (insufficient label variety or coverage)")

            elapsed = time.time() - start_time
            print(f"[VEP TIMER] Zero-shot VEP took {elapsed:.2f} seconds", flush=True)

        # synchronize all ranks before proceeding
        barrier()

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """Run once at step 0 to get a baseline AUC (optional)."""
        if state.global_step == 0:
            self._run_once(model, step_id=state.global_step)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Periodic VEP eval based on eval_every_n_steps."""
        if state.global_step > 0 and state.global_step % self.eval_every_n_steps == 0:
            self._run_once(model, step_id=state.global_step)
        return control

# -------------------------
# Loss logging fix callback
# -------------------------

class LossLoggingCallback(TrainerCallback):
    """Log true per-step loss (divide Trainer’s accumulated loss)."""

    def __init__(self, gradient_accumulation_steps):
        # store GA steps to correct the aggregated loss
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        # HF Trainer emits 'loss' after GA; we add 'loss_per_step' for clarity
        if logs is not None and "loss" in logs:
            actual_loss = logs["loss"] / self.gradient_accumulation_steps
            logs["loss_per_step"] = actual_loss
            # log only from rank 0 (avoid duplicate W&B series)
            if wandb.run is not None and get_rank() == 0 and state.global_step % args.logging_steps == 0:
                wandb.log({
                    "train_loss_per_step": actual_loss,
                    "train_loss_accumulated": logs["loss"]
                }, step=state.global_step)

# -----------------
# Main entry point
# -----------------

def main():
    # -------------------- Static args (kept from your HF.py) --------------------
    model_size="1B"
    args = {
        "tokenizer_dir": "/gpfs/data/brandeslab/Project/HuggingfaceTransformer/char_tokenizer",  # tokenizer path
        "train_data_path": "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train",  # train ds
        "val_data_path": "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation", # val ds (optional)
        "output_dir": f"./outputs/unet_{model_size}",                     # checkpoints/output root
        "batch_size": 2,                               # per-device batch size
        "epochs": 500,                                 # num_train_epochs (if using epoch schedule)
        "lr": 1e-4,                                    # learning rate
        "weight_decay": 0.0,                           # AdamW weight decay
        "gradient_accumulation_steps": 128,              # GA to reach bigger effective batch
        "base_dim": 896,                               # your model config param
        "growth": 224,                                  # your model config param
        "num_scales": 4,                               # your model config param
        "bf16": True,                                  # enable bfloat16
        "log_interval": 100,                          # Trainer logging_steps
        "vep_interval": 200,                         # VEP eval cadence
        "save_steps": 450,                           # checkpoint cadence
        "group_by_length": True                        # enable deterministic pre-grouping
    }

    # -------------------- Rank / world info for logging & seeding --------------------
    rank = get_rank()                              # integer rank (0..world_size-1)
    world_size = get_world_size()                  # total number of processes
    print_rank0(f"[DDP] world_size = {world_size}")# report DDP info once

    # -------------------- Seeds for determinism across ranks --------------------
    seed = 1337                                   # fixed seed (tweak if needed)
    random.seed(seed)                             # seed Python RNG
    np.random.seed(seed)                          # seed NumPy RNG
    torch.manual_seed(seed)                       # seed CPU RNG
    torch.cuda.manual_seed_all(seed)              # seed all GPU RNGs

    # -------------------- W&B init (rank 0 only) --------------------
    run_name = "unet_1B"                       # same run name you had
    if get_rank() == 0:
        wandb.init(project="long_runs", name=run_name, entity="sinha-anushka12-na", reinit=False)
    else:
        # disable network logging from non-zero ranks to avoid collisions
        os.environ["WANDB_MODE"] = "disabled"

    # -------------------- Tokenizer --------------------
    print_rank0("[Setup] Loading tokenizer...", flush=True)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args["tokenizer_dir"])  # load tokenizer from directory
    vocab_size = tokenizer.vocab_size                                              # capture vocab size
    mask_token_id = tokenizer.mask_token_id                                        # needed by collator/model cfg
    pad_token_id = tokenizer.pad_token_id                                          # pad id (not directly used here)

    # -------------------- Model config & model --------------------
    cfg = ModelConfig(
        vocab_size=vocab_size,                 # vocab size for MLM head
        mask_token_id=mask_token_id,           # mask token id
        base_dim=args["base_dim"],             # your architecture param
        growth=args["growth"],                 # your architecture param
        num_scales=args["num_scales"],         # your architecture param
        dilation_schedule=(2, 2, 2, 2)         # keep your dilation pattern
    )
    from transformer_block import TransformerConfig
    cfg.transformer_cfg = TransformerConfig(
        dim=cfg.bottleneck_dim,
        depth=36,
        heads=28,
        rope_max_position=8192
    )
    model = ProteinMLMModel(cfg)               # instantiate your model
    # NOTE: do NOT call model.to(rank) here; let HF Trainer place the model on the right device per process
    # NOTE: do NOT torch.compile by default in DDP until stability confirmed; you can enable later if desired
    # -------------------- Parameter count (rank 0 only) --------------------

    # -------------------- Derived constants --------------------
    min_len = 2 ** cfg.num_scales              # minimum sequence length used by your VEP/eval constraints

    # -------------------- Datasets --------------------
    print_rank0("[Setup] Loading datasets...", flush=True)
    train_ds = load_from_disk(args["train_data_path"])   # memory map train dataset
    # val_ds = load_from_disk(args["val_data_path"])     # optional: enable if you have validation

    # -------------------- Optional deterministic length grouping --------------------
    if args["group_by_length"]:
        print_rank0("[Setup] Applying deterministic length grouping...", flush=True)
        train_lengths = train_ds["length"]                                      # vector of sequence lengths
        grouped_indices = get_length_grouped_indices(train_lengths, args["batch_size"])  # same order on all ranks
        train_ds = train_ds.select(grouped_indices)                             # reorder dataset identically

    # -------------------- Data collator --------------------
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,                # tokenizer used for masking and id conversions
        mlm=True,                           # enable masked language modeling
        mlm_probability=cfg.mask_prob       # use your model config’s mask probability
    )

    # -------------------- VEP input (CSV) --------------------
    vep_csv = "/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv"  # path used by your pipeline

    # -------------------- TrainingArguments (DDP enabled) --------------------
    training_args = TrainingArguments(
        output_dir=args["output_dir"],                 # where to write checkpoints
        overwrite_output_dir=True,                     # overwrite if exists
        num_train_epochs=args["epochs"],               # long-run epochs (you can switch to max_steps if preferred)
        per_device_train_batch_size=args["batch_size"],# batch size per GPU/process
        gradient_accumulation_steps=args["gradient_accumulation_steps"],  # GA for larger effective batch
        learning_rate=args["lr"],                      # LR
        weight_decay=args["weight_decay"],             # weight decay
        bf16=args["bf16"],                             # bfloat16 compute
        logging_steps=args["log_interval"],            # log every N optimizer steps
        save_steps=args["save_steps"],                 # save every N steps
        #save_total_limit=3,                            # keep last N checkpoints
        dataloader_num_workers=4,                     # workers per process
        dataloader_pin_memory=True,                    # pin host memory
        dataloader_persistent_workers=True,            # keep workers alive between epochs
        dataloader_prefetch_factor=2,                  # prefetch per worker
        group_by_length=False,                         # we already pre-grouped deterministically
        remove_unused_columns=False,                   # keep all columns (model expects custom keys)
        report_to="wandb" if get_rank() == 0 else "none",                             # W&B integration (rank 0 only actually sends)
        run_name=run_name,                             # same run name
        logging_first_step=True,                       # log at step 0
        save_strategy="steps",                         # save based on step cadence
        # evaluation_strategy="steps",                 # uncomment with a real val_ds
        # eval_steps=args["log_interval"],             # cadence for evaluation
        load_best_model_at_end=False,                  # no best-of metric restore
        metric_for_best_model="loss",                  # used only if load_best_model_at_end=True
        logging_nan_inf_filter=False,                  # preserve raw loss values

        # --- DDP-specific bits ---
        ddp_backend="nccl",                            # NCCL backend for multi-GPU/multi-node
        ddp_timeout=3000,                              # seconds before NCCL operations timeout (long jobs)
        ddp_find_unused_parameters=False,              # faster if your graph is well-behaved (no unused params)
        # NOTE: We do not set local_rank manually; HF reads LOCAL_RANK from torchrun.
    )

    # -------------------- Callbacks --------------------
    vep_callback = ZeroShotVEPEvaluationCallback(
        tokenizer=tokenizer,               # tokenizer for building masked inputs
        input_csv=vep_csv,                 # CSV path with (sequence,pos,ref,alt,label)
        max_len=8192,                      # max length cap
        eval_every_n_steps=args["vep_interval"],  # cadence in steps
        batch_size=8                       # eval micro-batch per rank
    )

    loss_callback = LossLoggingCallback(
        gradient_accumulation_steps=args["gradient_accumulation_steps"]  # for correcting logged loss
    )
    # -------------------- Disable W&B on non-zero ranks --------------------

    # -------------------- Trainer --------------------
    trainer = ProteinMLMTrainer(
        model=model,                # your model
        args=training_args,         # training config (includes DDP settings)
        data_collator=collator,     # MLM collator
        train_dataset=train_ds,     # training dataset (HF will shard per rank)
        # eval_dataset=val_ds,      # attach if using validation
        callbacks=[vep_callback, loss_callback],  # DDP-safe callbacks
        debug_mode=True,            # enable initial debug prints (rank0 only prints)
    )

    # -------------------- Train --------------------
    print_rank0("\n[Training] Starting training...\n", flush=True)  # announce training on rank 0
    trainer.train()                                                 # HF handles DDP setup and loop

    # -------------------- Save only from rank 0 --------------------
    if get_rank() == 0:
        trainer.save_model(os.path.join(args["output_dir"], "final_model"))  # serialize to disk once
        print("\n[Training] Training completed!", flush=True)                # final message

# Standard Python entry guard
if __name__ == "__main__":
    main()  # call main




