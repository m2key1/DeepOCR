#!/bin/bash
#SBATCH --job-name=deepocr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=5-00:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err

# Adjust the partition, GPU count and memory to your cluster; the launch line below follows
# SLURM_GPUS_ON_NODE.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

# Keeps a system-wide CUDA install from shadowing the cuBLAS that ships with the torch wheel.
unset LD_LIBRARY_PATH

# Per-GPU batch, sized for the widest bucket on a 141 GB card; halve it for 80 GB cards. With 8 GPUs
# the effective batch is 640 and the learning rates scale by sqrt(640 // BASE_BATCH_SIZE).
export OCR_BATCH=80

# Dataloader workers per rank. Each spawn-context worker holds its own copy of the dataset and up
# to prefetch_factor batches, so host memory scales with ranks x workers. 6 keep a step fed.
export OCR_NUM_WORKERS=6

# Requeue-safe: after a node failure the job resumes from runs/latest_run/last_checkpoint.pt.
export OCR_RESUME=auto

# Width-bucketed batches change shape every step; expandable segments keep the caching allocator
# from fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# If the first allreduce hangs on a node without NVLink, add NCCL_P2P_DISABLE=1.
export NCCL_DEBUG=WARN

.venv/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${SLURM_GPUS_ON_NODE:-8}" \
    train.py
