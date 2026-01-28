#!/bin/bash
#SBATCH --job-name=train_u-net
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=28-00:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err

# srun --job-name=train_bert --partition=a100_long --gres=gpu:a100:1 --cpus-per-task=16 --mem=100G --time=28-00:00:00
# srun --job-name=train_bert_ddp --partition=a100_dev --gres=gpu:a100:2 --cpus-per-task=16 --mem=100G --time=04:00:00 --pty /bin/bash

# === Load and activate conda environment ===
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# Set Hugging Face cache location to non-home directory
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
echo "Caching to: $HF_HOME"

cd /gpfs/data/brandeslab/Project/u-net/

# python train_hf.py \
#   --tokenizer_dir /gpfs/data/brandeslab/Project/HuggingfaceTransformer/char_tokenizer \
#   --dataset_dir   /gpfs/data/brandeslab/Data/tokenized_datasets/uniref90_tokenized_single_char_512 \
#   --log_dir       /gpfs/data/brandeslab/Project/u-net/logs_unet \
#   --batch_size 64 --epochs 500 --lr 1e-4 --num_workers 16 

python HF_train.py
