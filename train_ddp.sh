#!/bin/bash -l
#SBATCH --job-name=train_unet_ddp
#SBATCH --partition=reservation
#SBATCH --reservation=brandeslab_reservation
#SBATCH --gres=gpu:a100:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=13-00:00:00
#SBATCH --output=/gpfs/scratch/an4477/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/an4477/slurm_logs/%x_%j.err

# === Load modules ===
#module purge
#module load anaconda3/gpu/new
#module load cuda/11.8

# === Activate environment ===
#source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
#conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert
# === Cache location (use your own space) ===
export HF_HOME=/gpfs/scratch/an4477/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME
echo "Caching to: $HF_HOME"

# === Navigate to project ===
cd /gpfs/data/brandeslab/Project/u-net/

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1
export TORCHELASTIC_ERROR_FILE="${PWD}/elastic_${SLURM_JOB_ID}.log"


# === Run distributed training ===
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  --redirects 3 \
  --tee 3 \
  HF_DDP100.py
