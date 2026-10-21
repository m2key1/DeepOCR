#!/bin/bash
#SBATCH --job-name=decoder_lm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err

# Pretrains the character decoder as a standalone LM. pretrain_decoder.py is single-process, so
# one GPU is enough; BATCH_SIZE 512 needs about 22 GB.
#
# --text-dirs with no values suppresses the data_sources default, which would harvest the raw
# pairs and so train the LM on data/valid's transcriptions. Emit the text from data/train only,
# with build_kenlm.py --dump-raw-text. The output lands at OUTPUT_PATH in pretrain_decoder.py,
# which is what PRETRAINED_DECODER_LM in train.py expects.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p runs

# Keeps a system-wide CUDA install from shadowing the cuBLAS that ships with the torch wheel.
unset LD_LIBRARY_PATH

.venv/bin/python pretrain_decoder.py \
    --text-dirs \
    --text-file data/lm/train_text.txt
