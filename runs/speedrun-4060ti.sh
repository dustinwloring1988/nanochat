#!/bin/bash

# Speed run configured for a single NVIDIA RTX 4060 Ti 16GB GPU.
# Shorter than the full 8-GPU speedrun (~1.5 hours) but long enough
# to reliably measure CORE metric and validation BPB for regression detection.
# Expected runtime: ~15-25 minutes depending on model size.

# Source uv environment
. $HOME/.local/bin/env

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv --python 3.11
# install the repo dependencies (gpu extra for CUDA support)
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate
# Set C compiler for Triton (needed for torch.compile on 4060Ti)
export CC=gcc
export PYTORCH_COMPILE_OFF=1
export TORCHINDUCTOR_CACHE_DIR="/root/.cache/torchinductor"

# -----------------------------------------------------------------------------
# Tokenizer (skip if already trained)

TOKENIZER_EXISTS=0
[ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] && TOKENIZER_EXISTS=1

if [ "$TOKENIZER_EXISTS" -eq 0 ]; then
    # Download a small subset of pretraining data (~1 shard = ~100MB)
    python -m nanochat.dataset -n 1
    # Also kick off a background download of more shards while training proceeds
    python -m nanochat.dataset -n 20 &
    DATASET_DOWNLOAD_PID=$!
    # train the tokenizer with vocab size 2**15 = 32768 on ~250M chars
    python -m scripts.tok_train --max-chars=250000000
fi
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Skip base training if checkpoints already exist
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints" ]; then
    # Waiting for dataset download to complete in background
    [ -n "$DATASET_DOWNLOAD_PID" ] && wait $DATASET_DOWNLOAD_PID

    # Use a shallow model (depth=8) that fits in 16GB VRAM on 4060Ti.
    # aspect_ratio=64 gives model_dim=512, which is lightweight but still
    # produces meaningful CORE metric and BPB results.
    # --device-batch-size=4 to avoid OOM on 16GB GPU.
    # --target-param-data-ratio=8 (Chinchilla-optimal-ish, lower than default 12
    #   for faster convergence on smaller datasets).
    # Note: omitted --fp8 (requires H100+; 4060Ti uses BF16/FP32).
    # Use --run=dummy to skip wandb logging (works in no-TTY containers).
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=8 \
    --target-param-data-ratio=8 \
    --device-batch-size=4 \
    --run=dummy \
    --num-iterations=100
fi

# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=4

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# run SFT and eval the model (shorter SFT run)
if [ ! -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints" ]; then
    torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --run=dummy
    torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft
fi

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"