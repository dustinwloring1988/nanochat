#!/bin/bash
# Initialize and run the approved pretraining pipeline in Docker

set -e

echo "=========================================="
echo "NanoChat Pretraining Pipeline"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Download tokenizer training corpus (samples)"
echo "  2. Train tokenizer on mixed corpus"
echo "  3. Run curriculum-based pretraining"
echo ""
echo "Supervised fine-tuning is intentionally not run by this workflow."
echo "Note: This is configured for single GPU (4060Ti)"
echo "      Uses sampled datasets for fast testing"
echo ""

# Activate venv
source /workspace/.venv/bin/activate

# Create necessary directories
mkdir -p $NANOCHAT_BASE_DIR/curriculum_data
mkdir -p $NANOCHAT_BASE_DIR/base_checkpoints

# Step 1: Train tokenizer on mixed curriculum corpus
echo "=========================================="
echo "STEP 1: Training Tokenizer"
echo "=========================================="
echo ""

if [ -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "✓ Tokenizer already trained at $NANOCHAT_BASE_DIR/tokenizer/ (skipping, set FORCE_RETRAIN=1 to override)"
else
    echo "Building mixed corpus from multiple sources..."
    echo "This will download ~6 shards per source (~4GB total)"
    echo ""

    python -m scripts.tok_train_curriculum \
        --mix default \
        --total-chars 2000000000 \
        --shards-per-source 30

    echo ""
    echo "Tokenizer training complete!"
fi
echo ""

# Step 1.5: Download all Nemotron subsets needed for pretraining
echo "=========================================="
echo "STEP 1.5: Downloading Pretraining Datasets"
echo "=========================================="
echo ""

if [ -f "$NANOCHAT_BASE_DIR/base_checkpoints/d6_curriculum_4060ti/model_003776.pt" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "✓ Base model already trained (skipping dataset download)"
else
    echo "Ensuring all Nemotron subsets are downloaded..."
    echo ""

    python -c "
from nanochat.data_registry import get_source
from nanochat.tokenizer_corpus import _download_huggingface_dataset

source_v1 = get_source('nemotron_v1')
print(f'Downloading Nemotron v1 subsets: {source_v1.subsets}')
_download_huggingface_dataset(source_v1, num_samples_per_subset=10000)

source_v1_1 = get_source('nemotron_v1_1')
print(f'Downloading Nemotron v1.1 subsets: {source_v1_1.subsets}')
_download_huggingface_dataset(source_v1_1, num_samples_per_subset=5000)

source_v1_2 = get_source('nemotron_v1_2')
print(f'Downloading Nemotron v1.2 subsets: {source_v1_2.subsets}')
_download_huggingface_dataset(source_v1_2, num_samples_per_subset=5000)

source_code_reasoning = get_source('code_reasoning')
print(f'Downloading Code Reasoning subsets: {source_code_reasoning.subsets}')
_download_huggingface_dataset(source_code_reasoning, num_samples_per_subset=5000)

source_knowledge_pile = get_source('knowledge_pile')
print(f'Downloading Knowledge Pile dataset')
_download_huggingface_dataset(source_knowledge_pile, num_samples_per_subset=10000)

print('All pretraining datasets downloaded!')
"
fi
echo ""

# Step 2: Run curriculum pretraining (skip eval - needs base data)
echo "=========================================="
echo "STEP 2: Curriculum Pretraining"
echo "=========================================="
echo ""

if [ -f "$NANOCHAT_BASE_DIR/base_checkpoints/d6_curriculum_4060ti/model_003776.pt" ] && [ "${FORCE_RETRAIN:-0}" != "1" ]; then
    echo "✓ Base model d6_curriculum_4060ti already trained at step 3776 (skipping, set FORCE_RETRAIN=1 to override)"
else
    echo "Starting curriculum-based pretraining..."
    echo "Using runs/curriculum_4060ti.sh"
    echo ""

    bash runs/curriculum_4060ti.sh
fi
echo ""
echo "=========================================="
echo "Pretraining Pipeline Complete"
echo "=========================================="
echo ""
echo "Base model: $NANOCHAT_BASE_DIR/base_checkpoints/d6_curriculum_4060ti/"
echo ""
echo "SFT is not run by this workflow. The legacy SFT entry point is disabled,"
echo "and production SFT requires a separately approved data and quality plan."
echo ""
echo "For the separately approved fixed-context runtime/resume probe, use:"
echo "  python -m scripts.sft_smoke --help"
echo ""
