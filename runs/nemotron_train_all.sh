#!/bin/bash
# Complete Nemotron pretraining pipeline
# Runs Stage 1, Stage 2a, and Stage 2b sequentially

set -e

echo "=========================================="
echo "Nemotron Complete Training Pipeline"
echo "=========================================="
echo ""
echo "This will run:"
echo "  Stage 1: Code-focused (2048 tokens)"
echo "  Stage 2a: Math Textbooks (8192 tokens)"
echo "  Stage 2b: InfiniByte Reasoning (32768 tokens)"
echo ""
echo "WARNING: This will take a very long time!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Build Docker image if needed
if ! docker images | grep -q "nanochat"; then
    echo "Building Docker image..."
    docker build -t nanochat -f docker/Dockerfile .
fi

# -----------------------------------------------------------------------------
# Stage 1: Code-focused pretraining
# -----------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Starting Stage 1..."
echo "=========================================="

mkdir -p checkpoints/stage1
mkdir -p logs/stage1

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd):/workspace" \
    -w /workspace \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/workspace/.cache/huggingface \
    nanochat \
    python scripts/nemotron_train.py \
        --config configs/nemotron_stage1_config.py

echo "Stage 1 complete! Checkpoint: checkpoints/stage1/final.pt"

# -----------------------------------------------------------------------------
# Stage 2a: Math Textbooks
# -----------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Starting Stage 2a..."
echo "=========================================="

mkdir -p checkpoints/stage2a
mkdir -p logs/stage2a

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd):/workspace" \
    -w /workspace \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/workspace/.cache/huggingface \
    nanochat \
    python scripts/nemotron_train.py \
        --config configs/nemotron_stage2_config.py \
        --stage stage2a

echo "Stage 2a complete! Checkpoint: checkpoints/stage2a/final.pt"

# -----------------------------------------------------------------------------
# Stage 2b: InfiniByte Reasoning
# -----------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "Starting Stage 2b..."
echo "=========================================="

mkdir -p checkpoints/stage2b
mkdir -p logs/stage2b

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$(pwd):/workspace" \
    -w /workspace \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e HF_HOME=/workspace/.cache/huggingface \
    nanochat \
    python scripts/nemotron_train.py \
        --config configs/nemotron_stage2_config.py \
        --stage stage2b

echo "Stage 2b complete! Checkpoint: checkpoints/stage2b/final.pt"

# -----------------------------------------------------------------------------
# Complete!
# -----------------------------------------------------------------------------

echo ""
echo "=========================================="
echo "ALL STAGES COMPLETE!"
echo "=========================================="
echo "Checkpoints:"
echo "  Stage 1:  checkpoints/stage1/final.pt"
echo "  Stage 2a: checkpoints/stage2a/final.pt"
echo "  Stage 2b: checkpoints/stage2b/final.pt"
echo "=========================================="
