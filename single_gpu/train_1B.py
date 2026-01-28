"""
Training script for Protein MLM Model using HuggingFace Trainer.

Architecture: U-Net style Conv Encoder → ModernBERT → Conv Decoder
Uses zero_shot.py for VEP evaluation.

Key fixes from original:
- Custom data collator that works with character-level protein tokenizers
- Proper weight initialization for stable training (loss starts at ~3.5)
- Optimized for A100 GPUs with gradient checkpointing option
"""

import os
import math
import time
from typing import Optional, Dict, Any

import torch
import torch.backends.cuda
import torch.backends.cudnn
import numpy as np
import pandas as pd
from datasets import load_from_disk

from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb

from model_1B import ModelConfig, ProteinMLMModel
from data_collator import ProteinMLMCollator
from zero_shot import run_vep_eval


# ====================== PERFORMANCE OPTIMIZATIONS ======================

def setup_performance_optimizations():
    """Enable various CUDA optimizations for faster training."""
    # Enable TF32 for faster matmuls on Ampere GPUs (A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    print("[Setup] TF32 enabled for matmul and cudnn", flush=True)
    print("[Setup] cuDNN benchmark mode enabled", flush=True)


# ====================== LENGTH GROUPING UTILITY ======================

def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=50, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices 
    correspond to elements of similar lengths. This minimizes padding waste.
    """
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist() 
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True) 
        for megabatch in megabatches
    ]

    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]


# ====================== CUSTOM CALLBACKS ======================

class VEPEvaluationCallback(TrainerCallback):
    """Callback to run zero-shot VEP evaluation at specified intervals."""
    
    def __init__(
        self, 
        model: ProteinMLMModel,
        tokenizer: PreTrainedTokenizerFast,
        vep_df: pd.DataFrame,
        eval_interval: int = 10000,
        min_length: int = 16,
        max_length: int = 8192,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.vep_df = vep_df
        self.eval_interval = eval_interval
        self.min_length = min_length
        self.max_length = max_length
        self.start_time = time.time()
        
        # Pre-calculate sequence counts
        self.total_sequences = len(vep_df)
        lengths = vep_df['sequence'].str.len() if 'sequence' in vep_df.columns else None
        if lengths is not None:
            self.valid_sequences = ((lengths >= min_length) & (lengths <= max_length)).sum()
        else:
            self.valid_sequences = None
        
    def on_step_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        if state.global_step > 0 and state.global_step % self.eval_interval == 0:
            device = next(self.model.parameters()).device
            
            # Ensure model is in eval mode
            was_training = self.model.training
            self.model.eval()
            
            print(f"\n{'='*60}", flush=True)
            print(f"[VEP Eval step {state.global_step}]", flush=True)
            print(f"  Total sequences in dataset: {self.total_sequences:,}", flush=True)
            if self.valid_sequences is not None:
                print(f"  Sequences within length range [{self.min_length}, {self.max_length}]: {self.valid_sequences:,}", flush=True)
            print(f"{'='*60}", flush=True)
            
            auc = run_vep_eval(
                self.model,
                self.tokenizer,
                device,
                df=self.vep_df,
                step=state.global_step,
                csv_path=None,
                min_length=self.min_length,
                max_length=self.max_length,
                batch_size=32,
                use_batched=True,
            )
            
            print(f"[VEP Eval step {state.global_step}] AUC: {auc}", flush=True)
            print(f"{'='*60}\n", flush=True)
            
            if auc is not None:
                wandb.log({
                    "zero_shot_vep_auc": auc,
                    "elapsed_hours": (time.time() - self.start_time) / 3600,
                }, step=state.global_step)
            
            # Restore training mode if needed
            if was_training:
                self.model.train()
        
        return control


class LoggingCallback(TrainerCallback):
    """Custom callback for additional logging."""
    
    def __init__(self):
        self.start_time = time.time()
        
    def on_log(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        logs: Dict[str, Any] = None,
        **kwargs
    ):
        if logs is not None:
            logs["elapsed_hours"] = (time.time() - self.start_time) / 3600
            
            loss = logs.get("loss", None)
            step = state.global_step
            lr = logs.get("learning_rate", 0)
            tokens_seen = logs.get("num_input_tokens_seen", 0)
            
            if loss is not None:
                print(f"[Step {step:,}] loss={loss:.4f}, lr={lr:.2e}, tokens_seen={tokens_seen:,}", flush=True)
            
        return control
    
    def on_step_end(
        self, 
        args: TrainingArguments, 
        state: TrainerState, 
        control: TrainerControl, 
        **kwargs
    ):
        if state.global_step == 1:
            print(f"[Step 1] Training started successfully", flush=True)
        return control


# ====================== CUSTOM TRAINER ======================

class ProteinMLMTrainer(Trainer):
    """Custom Trainer with token tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_tokens_seen = 0
        self.train_start_time = None
        self.tokens_since_last_log = 0
        self.last_log_time = None
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute MLM loss and track tokens."""
        # Count tokens (non-padding)
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                batch_tokens = attention_mask.sum().item()
            else:
                pad_token_id = self.processing_class.pad_token_id if self.processing_class else 0
                batch_tokens = (input_ids != pad_token_id).sum().item()
            
            self.total_tokens_seen += batch_tokens
            self.tokens_since_last_log += batch_tokens
        
        outputs = model(**inputs)
        loss = outputs["loss"]
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log to add token metrics."""
        current_time = time.time()
        
        if self.train_start_time is None:
            self.train_start_time = current_time
            self.last_log_time = current_time
        
        time_since_last_log = current_time - self.last_log_time if self.last_log_time else 1.0
        if time_since_last_log > 0:
            tokens_per_second = self.tokens_since_last_log / time_since_last_log
        else:
            tokens_per_second = 0
        
        logs["train_tokens_per_second"] = tokens_per_second
        logs["num_input_tokens_seen"] = self.total_tokens_seen
        
        self.tokens_since_last_log = 0
        self.last_log_time = current_time
        
        super().log(logs, start_time)


# ====================== MAIN TRAINING FUNCTION ======================

def main():
    # ======================== Configuration ========================
    
    # Paths - NYU HPC environment
    tokenizer_dir = "/gpfs/data/brandeslab/Project/HuggingfaceTransformer/char_tokenizer"
    train_data_path = "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train"
    val_data_path = "/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation"
    vep_data_path = "/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv"
    output_dir = "./outputs/unet_modernbert_1B_lr5e-4"
    
    # Alternative: set via environment variables
    import os as _os
    tokenizer_dir = _os.environ.get("TOKENIZER_DIR", tokenizer_dir)
    train_data_path = _os.environ.get("TRAIN_DATA_PATH", train_data_path)
    vep_data_path = _os.environ.get("VEP_DATA_PATH", vep_data_path)
    output_dir = _os.environ.get("OUTPUT_DIR", output_dir)
    
    # Model configuration (~1.3B parameters)
    model_config = ModelConfig(
        vocab_size=32,  # Will be updated from tokenizer
        mask_token_id=4,  # Will be updated from tokenizer
        pad_token_id=0,  # Will be updated from tokenizer
        base_dim=1024,
        growth=192,
        num_scales=2,  # 8x downsampling
        #dilation_schedule=(2, 2, 2),
        # ModernBERT settings
        modernbert_num_layers=28,
        modernbert_num_attention_heads=16,
        modernbert_intermediate_size=6400,
        # Enable gradient checkpointing for memory efficiency
        use_gradient_checkpointing=True,
    )
    
    # Training hyperparameters
    run_name = "unet_modernbert_1B_lr5e-4"
    batch_size = 8  # Increased due to gradient checkpointing
    gradient_accumulation_steps = 4  # Effective batch size = 32
    learning_rate = 5e-4
    weight_decay = 0.01
    max_steps = 500000
    warmup_steps = 2000
    log_interval = 100
    vep_interval = 10000
    save_steps = 10000
    
    # ======================== Setup ========================
    
    start_time = time.time()
    
    # Performance optimizations
    setup_performance_optimizations()
    
    # Initialize wandb
    wandb.init(
        project="protein_mlm_1B", 
        name=run_name, 
        entity="sinha-anushka12-na",
        reinit=True,
        resume="never",
    )
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Device: {device}", flush=True)
    
    if torch.cuda.is_available():
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[Setup] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    
    # Update model config with tokenizer info
    model_config.vocab_size = tokenizer.vocab_size
    model_config.mask_token_id = tokenizer.mask_token_id
    model_config.pad_token_id = tokenizer.pad_token_id or 0
    
    print(f"[Setup] Vocab size: {model_config.vocab_size}", flush=True)
    print(f"[Setup] Mask token ID: {model_config.mask_token_id}", flush=True)
    print(f"[Setup] Pad token ID: {model_config.pad_token_id}", flush=True)
    print(f"[Setup] Bottleneck dim: {model_config.bottleneck_dim}", flush=True)
    
    # Minimum sequence length
    min_seq_len = 2 ** model_config.num_scales
    print(f"[Setup] Minimum sequence length: {min_seq_len}", flush=True)
    
    # ======================== Load Data ========================
    
    print("[Data] Loading datasets...", flush=True)
    train_ds = load_from_disk(train_data_path)
    print(f"[Data] Train dataset size: {len(train_ds):,}", flush=True)
    
    # Print length statistics
    if "length" in train_ds.features:
        lengths = np.array(train_ds["length"])
        print(f"[Data] Sequence lengths - Mean: {np.mean(lengths):.0f}, "
              f"Median: {np.median(lengths):.0f}, "
              f"P95: {np.percentile(lengths, 95):.0f}, "
              f"Max: {np.max(lengths)}", flush=True)
    
    # Length grouping
    print("[Data] Applying length grouping (one-time)...", flush=True)
    train_lengths = train_ds["length"]
    grouped_indices = get_length_grouped_indices(train_lengths, batch_size)
    train_ds = train_ds.select(grouped_indices)
    print("[Data] Length grouping complete.", flush=True)
    
    # ======================== Data Collator ========================
    
    # Use custom collator instead of HuggingFace's DataCollatorForLanguageModeling
    data_collator = ProteinMLMCollator(
        tokenizer=tokenizer,
        mlm_probability=model_config.mask_prob,
    )
    
    # Verify collator works correctly
    print("[Data] Testing data collator...", flush=True)
    sample_batch = data_collator([train_ds[i] for i in range(4)])
    print(f"[Data] Sample batch - input_ids shape: {sample_batch['input_ids'].shape}", flush=True)
    print(f"[Data] Sample batch - labels non-masked: {(sample_batch['labels'] != -100).sum().item()}", flush=True)
    
    # ======================== Model ========================
    
    print("[Model] Initializing ProteinMLMModel with U-Net + ModernBERT...", flush=True)
    model = ProteinMLMModel(model_config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total parameters: {total_params:,}", flush=True)
    print(f"[Model] Trainable parameters: {trainable_params:,}", flush=True)
    print(f"[Model] Gradient checkpointing: {model_config.use_gradient_checkpointing}", flush=True)
    
    # Verify model produces reasonable loss
    print("[Model] Verifying model initialization...", flush=True)
    model.eval()
    with torch.no_grad():
        test_batch = {k: v.to(device) for k, v in sample_batch.items()}
        test_output = model(**test_batch)
        init_loss = test_output["loss"].item()
        expected_loss = math.log(model_config.vocab_size)
        print(f"[Model] Initial loss: {init_loss:.4f} (expected ~{expected_loss:.4f})", flush=True)
        
        if init_loss > expected_loss * 3:
            print(f"[WARNING] Initial loss is much higher than expected! Check model initialization.", flush=True)
    model.train()
    
    # ======================== Training Arguments ========================
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        
        # Training hyperparameters
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        
        # Learning rate schedule
        lr_scheduler_type="cosine",
        
        # Optimizer
        optim="adamw_torch_fused",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-6,
        max_grad_norm=1.0,
        
        # Precision
        bf16=True,  # Use bf16 on A100 (better than fp16)
        bf16_full_eval=True,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_steps=log_interval,
        logging_first_step=True,
        report_to="wandb",
        
        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        
        # DataLoader settings
        dataloader_num_workers=4,  # Reduced from 16
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        dataloader_drop_last=True,
        
        # Misc
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        label_names=["labels"],
    )
    
    # ======================== Callbacks ========================
    
    vep_df = pd.read_csv(vep_data_path)
    
    callbacks = [
        LoggingCallback(),
        VEPEvaluationCallback(
            model=model,
            tokenizer=tokenizer,
            vep_df=vep_df,
            eval_interval=vep_interval,
            min_length=min_seq_len,
            max_length=8192,
        ),
    ]
    
    # ======================== Trainer ========================
    
    trainer = ProteinMLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    
    # ======================== Training ========================
    
    print("\n" + "=" * 60, flush=True)
    print("Starting Training", flush=True)
    print(f"  Architecture: U-Net + ModernBERT", flush=True)
    print(f"  Model size: {total_params/1e9:.2f}B parameters", flush=True)
    print(f"  Batch size: {batch_size} x {gradient_accumulation_steps} = {batch_size * gradient_accumulation_steps} effective", flush=True)
    print(f"  Learning rate: {learning_rate}", flush=True)
    print(f"  Warmup steps: {warmup_steps}", flush=True)
    print(f"  Max steps: {max_steps:,}", flush=True)
    print("=" * 60 + "\n", flush=True)
    
    # Check for checkpoint
    last_checkpoint = None
    if os.path.isdir(output_dir):
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint:
            print(f"[Resume] Found checkpoint: {last_checkpoint}", flush=True)
    
    # Train
    #train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    train_result = trainer.train()
    
    # Save final model
    trainer.save_model()
    
    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    print("\n" + "=" * 60, flush=True)
    print("Training Complete!", flush=True)
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours", flush=True)
    print("=" * 60 + "\n", flush=True)
    
    wandb.finish()


if __name__ == "__main__":
    import sys
    import traceback
    
    # Force unbuffered output
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
    
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}", flush=True)
        print(f"FATAL ERROR: {type(e).__name__}: {e}", flush=True)
        print(f"{'='*60}", flush=True)
        traceback.print_exc()
        sys.exit(1)