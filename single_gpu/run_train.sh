#!/bin/bash -l
#SBATCH --job-name=train_1B_unet
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gpfs/scratch/an4477/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/an4477/slurm_logs/%x_%j.err

# === Load modules ===
module load anaconda3/gpu/new

# === Activate environment ===
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# === Cache location ===
export HF_HOME=/gpfs/scratch/an4477/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

echo "=============================================="
echo "PROTEIN MLM 1B TRAINING"
echo "=============================================="
echo "Date: $(date)"
echo "Caching to: $HF_HOME"
echo ""

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# === Navigate to project ===
cd /gpfs/data/brandeslab/Project/u-net/mb/one

# === Torch compile settings ===
export TORCHINDUCTOR_DISABLE_CUDA_GRAPH=1
export TORCH_COMPILE_DEBUG=0

# === Run training ===
python train_1B.py

echo ""
echo "=============================================="
echo "Training complete at $(date)"
echo "=============================================="