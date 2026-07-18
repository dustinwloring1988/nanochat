#!/bin/bash

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Download pre-trained d34 model and tokenizer from HuggingFace (~8.6GB)
python -m scripts.download_d34

# SFT fine-tune the pre-trained model
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --model-tag d34 --run=$WANDB_RUN

# Evaluate
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

