#!/bin/bash
# Curriculum training optimized for single NVIDIA RTX 4060Ti (16GB VRAM)
# 
# Model: depth=6 (very small for fast iteration testing)
# Training: Multi-stage curriculum with context length scheduling
# Dataset: Mixed sources (ClimbMix + Nemotron specialized datasets)
#
# This script is designed for rapid experimentation and testing of the
# curriculum pipeline, not for training production-quality models.

echo "=========================================="
echo "NanoChat Curriculum Training - 4060Ti"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  GPU: Single RTX 4060Ti (16GB VRAM)"
echo "  Model: depth=6 (small test model)"
echo "  Curriculum: 4 stages with context 2K→16K"
echo "  Purpose: Fast iteration testing"
echo ""

# Environment setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p $NANOCHAT_BASE_DIR

# Python environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "WARNING: .venv not found, using system Python"
fi

# Wandb setup (optional)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy  # Set to a name to enable wandb logging
fi

echo "Data directory: $NANOCHAT_BASE_DIR"
echo "Wandb run: $WANDB_RUN"
echo ""

# ===================================================================
# Stage 1: Download dataset shards
# ===================================================================

echo "=========================================="
echo "Downloading Dataset Shards"
echo "=========================================="
echo ""
echo "Downloading ClimbMix shards for curriculum training..."
echo "  - Stage 1 (foundation): ~40 shards (~4GB)"
echo "  - Additional shards will be downloaded as needed"
echo ""

# Download initial shards for foundation stage
# For depth=6, we use more data (doubled from original)
python -m nanochat.data_registry --download climbmix -n 40 -w 4

echo ""
echo "Dataset download complete!"
echo ""

# ===================================================================
# Stage 2: Curriculum Pretraining
# ===================================================================

echo "=========================================="
echo "Curriculum Pretraining (depth=6)"
echo "=========================================="
echo ""
echo "Training with 4-stage curriculum:"
echo "  Stage 1: Foundation (2K→4K context)"
echo "  Stage 2: Reasoning+Code (4K→8K context)"
echo "  Stage 3: Long Context (8K→16K context)"
echo "  Stage 4: Consolidation (16K context)"
echo ""

# Training hyperparameters optimized for 4060Ti (16GB VRAM)
# These are aggressive settings for fast iteration, not optimal quality
python -m scripts.base_train_curriculum \
    --config config/pretraining_curriculum.yaml \
    --depth 6 \
    --aspect-ratio 64 \
    --head-dim 128 \
    --max-seq-len 2048 \
    --window-pattern "SSL" \
    --device-batch-size 24 \
    --total-batch-size 98304 \
    --target-param-data-ratio 16 \
    --embedding-lr 0.3 \
    --unembedding-lr 0.008 \
    --matrix-lr 0.02 \
    --scalar-lr 0.5 \
    --weight-decay 0.28 \
    --warmup-steps 40 \
    --warmdown-ratio 0.65 \
    --final-lr-frac 0.05 \
    --eval-every 100 \
    --eval-tokens $((10 * 98304)) \
    --core-metric-every 500 \
    --core-metric-max-per-task 100 \
    --sample-every 500 \
    --save-every 500 \
    --model-tag "d6_curriculum_4060ti" \
    --run "$WANDB_RUN"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: $NANOCHAT_BASE_DIR/base_checkpoints/d6_curriculum_4060ti/"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python -m scripts.base_eval --model-tag d6_curriculum_4060ti"
echo "  2. Chat: python -m scripts.chat_cli"
echo "  3. SFT: python -m scripts.sft_train_curriculum"
echo ""
echo "Performance notes for depth=6:"
echo "  - This is a tiny model for testing the pipeline"
echo "  - Expected training time: ~2-4 hours on 4060Ti"
echo "  - For production models, use depth=20+ on multi-GPU"
echo ""
