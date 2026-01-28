"""
Training script for Protein MLM Model using FSDP (Fully Sharded Data Parallel).

Architecture: U-Net style Conv Encoder → ModernBERT → Conv Decoder

FSDP shards model parameters, gradients, and optimizer states across GPUs,
dramatically reducing per-GPU memory usage compared to DDP.
"""

import os
import math
import time
import functools
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    ModuleWrapPolicy,
)
import torch.backends.cuda
import torch.backends.cudnn
import numpy as np
import pandas as pd
from datasets import load_from_disk

from transformers import (
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)
from transformers.models.modernbert.modeling_modernbert import ModernBertEncoderLayer

import wandb
from tqdm import tqdm

from model_1B import ModelConfig, ProteinMLMModel
from data_collator import ProteinMLMCollator
from zero_shot import run_vep_eval


# ====================== DISTRIBUTED UTILITIES ======================

def setup_distributed():
    """Initialize distributed training."""
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def print_rank0(msg: str) -> None:
    if is_main_process():
        print(msg, flush=True)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


# ====================== PERFORMANCE OPTIMIZATIONS ======================

def setup_performance_optimizations() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print_rank0("[Setup] TF32 and cuDNN benchmark enabled")


# ====================== LENGTH GROUPING ======================

def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=50, generator=None):
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i:i+megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(mb, key=lambda i: lengths[i], reverse=True) for mb in megabatches]

    megabatch_maximums = [lengths[mb[0]] for mb in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for mb in megabatches for i in mb]


# ====================== FSDP UTILITIES ======================

def get_fsdp_wrap_policy(model):
    """
    Create FSDP wrap policy that wraps:
    1. ModernBERT encoder layers (transformer layers)
    2. Large conv blocks
    """
    # Wrap ModernBERT encoder layers and our custom blocks
    from model_1B import DownBlock, UpBlock, ConvBlock
    
    auto_wrap_policy = ModuleWrapPolicy(
        {ModernBertEncoderLayer, DownBlock, UpBlock}
    )
    return auto_wrap_policy


def get_fsdp_config():
    """Get FSDP configuration with mixed precision."""
    # Mixed precision policy for bf16 training
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    return {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,  # Shard everything
        "mixed_precision": bf16_policy,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "cpu_offload": None,  # No CPU offload for speed
        "device_id": torch.cuda.current_device(),
        "use_orig_params": True,  # Required for torch.compile compatibility
        "limit_all_gathers": True,  # Memory optimization
    }


def save_fsdp_checkpoint(model, optimizer, scheduler, step, output_dir):
    """Save FSDP checkpoint using full state dict."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_state = model.state_dict()
        
        if is_main_process():
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            torch.save({
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "step": step,
            }, os.path.join(checkpoint_dir, "checkpoint.pt"))
            
            print_rank0(f"[Checkpoint] Saved at step {step}")
    
    barrier()


def load_fsdp_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load FSDP checkpoint - must be called by all ranks."""
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    
    # Only rank 0 loads from disk, then broadcasts
    if is_main_process():
        print_rank0(f"[Resume] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        step = checkpoint["step"]
    else:
        checkpoint = None
        step = 0
    
    # Broadcast step to all ranks
    step_tensor = torch.tensor([step], dtype=torch.long, device="cuda")
    dist.broadcast(step_tensor, src=0)
    step = step_tensor.item()
    
    # Load model state dict - all ranks must participate
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
        if is_main_process():
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Non-rank-0 processes still need to participate in the collective
            model.load_state_dict({})
    
    # Optimizer and scheduler only loaded on rank 0, but we skip for simplicity
    # (optimizer states are complex with FSDP, we'll let them reinitialize)
    if is_main_process() and checkpoint is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            # Skip optimizer for now - FSDP optimizer state loading is complex
            print_rank0(f"[Resume] Scheduler restored, optimizer will warm up")
        except Exception as e:
            print_rank0(f"[Resume] Warning: Could not restore scheduler: {e}")
    
    barrier()
    return step


# ====================== TRAINING LOOP ======================

def train_step(model, batch, device):
    """Single training step."""
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    return outputs["loss"]


def evaluate_vep(model, tokenizer, device, vep_df, step, min_length=4, max_length=8192):
    """Run VEP evaluation."""
    model.eval()
    
    # For FSDP, we need to use the model directly
    auc = run_vep_eval(
        model,
        tokenizer,
        device,
        df=vep_df,
        step=step,
        csv_path=None,
        min_length=min_length,
        max_length=max_length,
        batch_size=32,
        use_batched=True,
        use_amp=True,
    )
    
    model.train()
    return auc


# ====================== MAIN ======================

def main():
    # ======================== Setup ========================
    local_rank = get_local_rank()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        raise RuntimeError("FSDP requires CUDA")
    
    setup_distributed()
    setup_performance_optimizations()
    
    # ======================== Configuration ========================
    
    tokenizer_dir = os.environ.get("TOKENIZER_DIR", "/gpfs/data/brandeslab/Project/HuggingfaceTransformer/char_tokenizer")
    train_data_path = os.environ.get("TRAIN_DATA_PATH", "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train")
    vep_data_path = os.environ.get("VEP_DATA_PATH", "/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv")
    output_dir = os.environ.get("OUTPUT_DIR", "./outputs/unet_modernbert_1B_fsdp")
    
    # Model config - gradient checkpointing enabled (works with FSDP!)
    model_config = ModelConfig(
        vocab_size=32,
        mask_token_id=4,
        pad_token_id=0,
        base_dim=1024,
        growth=192,
        num_scales=2,
        modernbert_num_layers=28,
        modernbert_num_attention_heads=16,
        modernbert_intermediate_size=6400,
        use_gradient_checkpointing=True,  # FSDP supports gradient checkpointing!
    )
    
    # Training params - can use larger batch with FSDP + grad checkpointing
    run_name = "unet_modernbert_1B_fsdp"
    per_device_batch_size = 8  # Back to 8 with FSDP!
    gradient_accumulation_steps = 2
    # Effective: 8 * 2 GPUs * 2 = 32
    
    learning_rate = 4e-5
    weight_decay = 0.01
    max_steps = 250000
    warmup_steps = 2000
    log_interval = 100
    save_interval = 10000
    vep_interval = 10000
    
    start_time = time.time()
    
    # ======================== Wandb ========================
    
    if is_main_process():
        wandb.init(project="protein_mlm_1B", name=run_name, entity="sinha-anushka12-na")
    else:
        os.environ["WANDB_DISABLED"] = "true"
    
    print_rank0(f"[Setup] World size: {get_world_size()}, Device: {device}")
    print_rank0(f"[Setup] FSDP enabled with gradient checkpointing")
    
    if torch.cuda.is_available():
        print_rank0(f"[Setup] GPU: {torch.cuda.get_device_name(local_rank)}")
        print_rank0(f"[Setup] GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory/1e9:.1f}GB")
    
    # ======================== Tokenizer ========================
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    model_config.vocab_size = tokenizer.vocab_size
    model_config.mask_token_id = tokenizer.mask_token_id
    model_config.pad_token_id = tokenizer.pad_token_id or 0
    
    print_rank0(f"[Setup] Vocab: {model_config.vocab_size}, Bottleneck: {model_config.bottleneck_dim}")
    
    # ======================== Data ========================
    
    print_rank0("[Data] Loading...")
    train_ds = load_from_disk(train_data_path)
    print_rank0(f"[Data] Size: {len(train_ds):,}")
    
    # Length grouping
    print_rank0("[Data] Applying length grouping...")
    generator = torch.Generator().manual_seed(42)
    grouped_indices = get_length_grouped_indices(
        train_ds["length"], 
        per_device_batch_size * get_world_size(), 
        generator=generator
    )
    train_ds = train_ds.select(grouped_indices)
    
    # Data collator
    data_collator = ProteinMLMCollator(tokenizer=tokenizer, mlm_probability=model_config.mask_prob)
    
    # Distributed sampler
    from torch.utils.data import DataLoader, DistributedSampler
    
    sampler = DistributedSampler(
        train_ds,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=False,  # Already shuffled via length grouping
        drop_last=True,
    )
    
    dataloader = DataLoader(
        train_ds,
        batch_size=per_device_batch_size,
        sampler=sampler,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    barrier()
    
    # ======================== Model ========================
    
    print_rank0("[Model] Initializing...")
    model = ProteinMLMModel(model_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print_rank0(f"[Model] Parameters: {total_params:,}")
    print_rank0(f"[Model] Gradient checkpointing: {model_config.use_gradient_checkpointing}")
    
    # ======================== FSDP Wrapping ========================
    
    print_rank0("[FSDP] Wrapping model...")
    
    fsdp_config = get_fsdp_config()
    wrap_policy = get_fsdp_wrap_policy(model)
    
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        **fsdp_config
    )
    
    print_rank0("[FSDP] Model wrapped successfully")
    
    # Print FSDP sharding info
    if is_main_process():
        for name, param in model.named_parameters():
            if param.requires_grad:
                print_rank0(f"[FSDP] {name}: {param.shape}, sharded={param.is_sharded if hasattr(param, 'is_sharded') else 'N/A'}")
                break  # Just print first one as example
    
    barrier()
    
    # ======================== Optimizer & Scheduler ========================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=weight_decay,
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    
    # ======================== Resume from Checkpoint ========================
    
    start_step = 0
    checkpoint_path = None
    
    # Only rank 0 checks for checkpoints
    if is_main_process():
        if os.path.isdir(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                candidate_path = os.path.join(output_dir, latest, "checkpoint.pt")
                if os.path.exists(candidate_path):
                    checkpoint_path = candidate_path
                    print_rank0(f"[Resume] Found checkpoint: {checkpoint_path}")
    
    # Broadcast checkpoint path to all ranks
    path_list = [checkpoint_path]
    dist.broadcast_object_list(path_list, src=0)
    checkpoint_path = path_list[0]
    
    # All ranks load checkpoint together
    if checkpoint_path is not None:
        start_step = load_fsdp_checkpoint(model, optimizer, scheduler, checkpoint_path)
        print_rank0(f"[Resume] Resuming from step {start_step}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ======================== VEP Data ========================
    
    vep_df = pd.read_csv(vep_data_path)
    
    # ======================== Training Loop ========================
    
    effective_batch = per_device_batch_size * get_world_size() * gradient_accumulation_steps
    
    print_rank0(f"\n{'='*60}")
    print_rank0(f"FSDP Training Config:")
    print_rank0(f"  Model: {total_params/1e9:.2f}B params")
    print_rank0(f"  GPUs: {get_world_size()}")
    print_rank0(f"  Batch: {per_device_batch_size} x {get_world_size()} x {gradient_accumulation_steps} = {effective_batch}")
    print_rank0(f"  Learning rate: {learning_rate}")
    print_rank0(f"  Max steps: {max_steps:,}")
    print_rank0(f"  Gradient checkpointing: {model_config.use_gradient_checkpointing}")
    print_rank0(f"  FSDP sharding: FULL_SHARD")
    print_rank0(f"{'='*60}\n")
    
    model.train()
    optimizer.zero_grad()
    
    global_step = start_step
    total_tokens = 0
    tokens_since_log = 0
    accumulated_loss = 0.0
    last_log_time = time.time()
    
    # Create infinite iterator
    epoch = 0
    data_iter = iter(dataloader)
    
    # Progress bar
    if is_main_process():
        pbar = tqdm(total=max_steps, initial=start_step, desc="Training")
    
    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Count tokens
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            batch_tokens = attention_mask.sum().item() * get_world_size()
        else:
            batch_tokens = batch["input_ids"].numel() * get_world_size()
        total_tokens += batch_tokens
        tokens_since_log += batch_tokens
        
        # Forward pass
        loss = train_step(model, batch, device)
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        accumulated_loss += loss.item()
        
        # Optimizer step
        if (global_step + 1) % gradient_accumulation_steps == 0 or global_step == max_steps - 1:
            # Gradient clipping (FSDP handles this)
            model.clip_grad_norm_(1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        global_step += 1
        
        # Logging
        if global_step % log_interval == 0:
            current_time = time.time()
            time_delta = current_time - last_log_time
            tokens_per_sec = tokens_since_log / time_delta if time_delta > 0 else 0
            
            avg_loss = accumulated_loss / log_interval * gradient_accumulation_steps
            lr = scheduler.get_last_lr()[0]
            
            if is_main_process():
                print_rank0(f"[Step {global_step:,}] loss={avg_loss:.4f}, lr={lr:.2e}, tok/s={tokens_per_sec:.0f}")
                
                wandb.log({
                    "loss": avg_loss,
                    "learning_rate": lr,
                    "tokens_per_second": tokens_per_sec,
                    "total_tokens": total_tokens,
                    "epoch": epoch,
                }, step=global_step)
                
                pbar.update(log_interval)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            
            accumulated_loss = 0.0
            tokens_since_log = 0
            last_log_time = current_time
        
        # First step logging
        if global_step == 1:
            print_rank0(f"[Step 1] Training started successfully!")
        
        # VEP Evaluation - ALL ranks must participate due to FSDP sharding
        if global_step % vep_interval == 0:
            print_rank0(f"\n[VEP Eval @ step {global_step}]")
            
            # All ranks run evaluation (FSDP requires this)
            auc = evaluate_vep(model, tokenizer, device, vep_df, global_step)
            
            # Only rank 0 logs
            if is_main_process():
                print_rank0(f"[VEP Eval] AUC: {auc}\n")
                if auc is not None:
                    wandb.log({"zero_shot_vep_auc": auc}, step=global_step)
            
            model.train()
            barrier()  # Sync all ranks after evaluation
            torch.cuda.empty_cache()
        
        # Checkpointing
        if global_step % save_interval == 0:
            save_fsdp_checkpoint(model, optimizer, scheduler, global_step, output_dir)
    
    # ======================== Final Save ========================
    
    save_fsdp_checkpoint(model, optimizer, scheduler, global_step, output_dir)
    
    if is_main_process():
        pbar.close()
        total_time = (time.time() - start_time) / 3600
        print_rank0(f"\n{'='*60}")
        print_rank0(f"Training Complete!")
        print_rank0(f"Total time: {total_time:.2f} hours")
        print_rank0(f"Total tokens: {total_tokens:,}")
        print_rank0(f"{'='*60}\n")
        wandb.finish()
    
    cleanup_distributed()


if __name__ == "__main__":
    import sys
    import traceback
    
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
    
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR rank {get_rank()}: {e}", flush=True)
        traceback.print_exc()
        cleanup_distributed()
        sys.exit(1)