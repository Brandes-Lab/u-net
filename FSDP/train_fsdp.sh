#!/bin/bash -l
#SBATCH --job-name=train_1B_fsdp
#SBATCH --partition=a100_dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/scratch/an4477/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/an4477/slurm_logs/%x_%j.err

module load anaconda3/gpu/new
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

export HF_HOME=/gpfs/scratch/an4477/cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME
unset TRANSFORMERS_CACHE

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 29500-29999 -n 1)

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^lo,docker
export OMP_NUM_THREADS=8

# FSDP specific settings
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=============================================="
echo "PROTEIN MLM 1B FSDP TRAINING"
echo "=============================================="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs: 2 x A100 80GB"
echo ""
echo "FSDP Configuration:"
echo "  Sharding: FULL_SHARD"
echo "  Gradient Checkpointing: Enabled"
echo "  Batch: 8 per GPU, 2 accum = 32 effective"
echo ""

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

cd /gpfs/data/brandeslab/Project/u-net/mb/one/FSDP

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_fsdp.py

echo ""
echo "Exit code: $?"
echo "Complete: $(date)"