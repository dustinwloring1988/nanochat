#!/bin/bash

# Speed run configured for a single NVIDIA RTX 4060 Ti 16GB GPU.
# Uses Protocol v1.0.0 with 32K vocabulary tokenizer.
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
# Tokenizer (skip if already trained with protocol v1.0.0)

TOKENIZER_EXISTS=0
TOKENIZER_VALID=0

if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] && [ -f "$NANOCHAT_BASE_DIR/tokenizer/metadata.json" ]; then
    TOKENIZER_EXISTS=1
    # Check protocol version
    PROTOCOL_VERSION=$(python -c "import json; print(json.load(open('$NANOCHAT_BASE_DIR/tokenizer/metadata.json'))['protocol_version'])" 2>/dev/null || echo "unknown")
    if [ "$PROTOCOL_VERSION" = "1.0.0" ]; then
        TOKENIZER_VALID=1
        echo "✓ Found tokenizer with protocol v1.0.0"
    else
        echo "⚠ Found tokenizer with protocol $PROTOCOL_VERSION (need v1.0.0)"
    fi
fi

if [ "$TOKENIZER_VALID" -eq 0 ]; then
    echo "Training tokenizer with protocol v1.0.0..."
    # Download a small subset of pretraining data (~1 shard = ~100MB)
    python -m nanochat.dataset -n 1
    # Also kick off a background download of more shards while training proceeds
    python -m nanochat.dataset -n 20 &
    DATASET_DOWNLOAD_PID=$!
    # Train tokenizer with protocol v1.0.0
    # 32,000 lexical vocab + 222 control tokens = 32,222 total
    python -m scripts.tok_train --max-chars=250000000
    echo "✓ Tokenizer trained with protocol v1.0.0"
fi

# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining with protocol v1.0.0)

# Verify tokenizer before training
TOKENIZER_VOCAB=$(python -c "from nanochat.tokenizer import get_tokenizer; print(get_tokenizer().get_vocab_size())" 2>/dev/null || echo "0")
if [ "$TOKENIZER_VOCAB" != "32000" ]; then
    echo "✗ ERROR: Tokenizer vocab size is $TOKENIZER_VOCAB (expected 32000)"
    echo "  Please retrain tokenizer with protocol v1.0.0"
    exit 1
fi

# Check if we need to train base model
NEED_BASE_TRAIN=0
if [ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints" ]; then
    NEED_BASE_TRAIN=1
else
    # Check if existing checkpoint uses correct vocab size
    BASE_VOCAB=$(python -c "import json; meta=json.load(open('$NANOCHAT_BASE_DIR/base_checkpoints/d8/meta_000100.json')); print(meta.get('model_config', {}).get('vocab_size', 0))" 2>/dev/null || echo "0")
    if [ "$BASE_VOCAB" != "32000" ]; then
        echo "⚠ Existing base checkpoint uses vocab size $BASE_VOCAB (need 32000)"
        echo "  Moving old checkpoint and retraining with protocol v1.0.0..."
        mv "$NANOCHAT_BASE_DIR/base_checkpoints" "$NANOCHAT_BASE_DIR/base_checkpoints_old"
        NEED_BASE_TRAIN=1
    fi
fi

if [ "$NEED_BASE_TRAIN" -eq 1 ]; then
    echo "Training base model with protocol v1.0.0 (32K vocab)..."
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
    echo "✓ Base model training complete (protocol v1.0.0)"
else
    echo "✓ Base model checkpoint found (protocol v1.0.0)"
fi

# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=4

# -----------------------------------------------------------------------------
# SFT (chat fine-tuning with protocol v1.0.0)
# Uses chat template with reasoning structure and tool calling

# Check if we need to train SFT model
NEED_SFT_TRAIN=0
if [ ! -d "$NANOCHAT_BASE_DIR/chatsft_checkpoints" ]; then
    NEED_SFT_TRAIN=1
else
    # Check if existing SFT checkpoint is compatible
    # (This is a simple check - in production you'd want more validation)
    if [ -f "$NANOCHAT_BASE_DIR/chatsft_checkpoints/d8/meta_001889.json" ]; then
        echo "✓ SFT checkpoint found"
    else
        echo "⚠ SFT checkpoint incomplete, retraining..."
        mv "$NANOCHAT_BASE_DIR/chatsft_checkpoints" "$NANOCHAT_BASE_DIR/chatsft_checkpoints_old"
        NEED_SFT_TRAIN=1
    fi
fi

if [ "$NEED_SFT_TRAIN" -eq 1 ]; then
    echo "Training SFT model with protocol v1.0.0 (chat template + reasoning + tools)..."
    torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --run=dummy
    echo "✓ SFT training complete"
    
    echo "Evaluating SFT model..."
    torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft
else
    echo "✓ SFT checkpoint found (protocol v1.0.0)"
fi

echo ""
echo "========================================================================"
echo "Speedrun Complete!"
echo "========================================================================"
echo "Protocol: v1.0.0"
echo "Tokenizer: 32,000 vocab (32K lexical + 222 control tokens)"
echo "Base model: $NANOCHAT_BASE_DIR/base_checkpoints"
echo "SFT model: $NANOCHAT_BASE_DIR/chatsft_checkpoints"
echo ""
echo "Test the model:"
echo "  python -m scripts.chat_cli -p \"What is 2+2?\""
echo "  python -m scripts.chat_cli  # Interactive mode"
echo "========================================================================"