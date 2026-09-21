#!/bin/bash
# Initialize and run full curriculum training pipeline in Docker

set -e

echo "=========================================="
echo "NanoChat Curriculum Training Pipeline"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Download tokenizer training corpus (samples)"
echo "  2. Train tokenizer on mixed corpus"
echo "  3. Run curriculum-based pretraining"
echo "  4. Run supervised fine-tuning (SFT)"
echo ""
echo "Note: This is configured for single GPU (4060Ti)"
echo "      Uses sampled datasets for fast testing"
echo ""

# Activate venv
source /workspace/.venv/bin/activate

# Create necessary directories
mkdir -p $NANOCHAT_BASE_DIR/curriculum_data
mkdir -p $NANOCHAT_BASE_DIR/base_checkpoints
mkdir -p $NANOCHAT_BASE_DIR/sft_checkpoints

# Step 1: Train tokenizer on mixed curriculum corpus
echo "=========================================="
echo "STEP 1: Training Tokenizer"
echo "=========================================="
echo ""
echo "Building mixed corpus from multiple sources..."
echo "This will download ~3 shards per source (~2GB total)"
echo ""

python -m scripts.tok_train_curriculum \
    --mix default \
    --total-chars 2000000000 \
    --shards-per-source 15

echo ""
echo "Tokenizer training complete!"
echo ""

# Step 1.5: Download all Nemotron subsets needed for pretraining
echo "=========================================="
echo "STEP 1.5: Downloading Pretraining Datasets"
echo "=========================================="
echo ""
echo "Ensuring all Nemotron subsets are downloaded..."
echo ""

# Use Python to download all subsets from the curriculum
python -c "
from nanochat.data_registry import get_source
from nanochat.tokenizer_corpus import _download_huggingface_dataset

# Download all Nemotron v1 subsets
source_v1 = get_source('nemotron_v1')
print(f'Downloading Nemotron v1 subsets: {source_v1.subsets}')
_download_huggingface_dataset(source_v1, num_samples_per_subset=10000)

# Download all Nemotron v1.1 subsets  
source_v1_1 = get_source('nemotron_v1_1')
print(f'Downloading Nemotron v1.1 subsets: {source_v1_1.subsets}')
_download_huggingface_dataset(source_v1_1, num_samples_per_subset=5000)

# Download all Nemotron v1.2 subsets
source_v1_2 = get_source('nemotron_v1_2')
print(f'Downloading Nemotron v1.2 subsets: {source_v1_2.subsets}')
_download_huggingface_dataset(source_v1_2, num_samples_per_subset=5000)

print('All Nemotron subsets downloaded!')
"

echo ""

# Step 2: Run curriculum pretraining (skip eval - needs base data)
echo "=========================================="
echo "STEP 2: Curriculum Pretraining"
echo "=========================================="
echo ""
echo "Starting curriculum-based pretraining..."
echo "Using runs/curriculum_4060ti.sh"
echo ""

bash runs/curriculum_4060ti.sh

echo ""
echo "=========================================="
echo "STEP 3: Supervised Fine-Tuning (SFT)"
echo "=========================================="
echo ""
echo "Starting multi-dataset curriculum SFT..."
echo "Using 5 datasets across 3 stages (instruction → coding/agents → consolidation)"
echo ""

# Run multi-dataset curriculum SFT
python -m scripts.sft_train_curriculum \
    --config config/sft_curriculum.yaml \
    --model-tag d6_curriculum_4060ti \
    --device-batch-size 4 \
    --total-batch-size 16384 \
    --eval-every 200

echo ""
echo "SFT curriculum training complete!"
echo ""

echo ""
echo "=========================================="
echo "Training Pipeline Complete!"
echo "=========================================="
echo ""
echo "Base model: $NANOCHAT_BASE_DIR/base_checkpoints/d6_curriculum_4060ti/"
echo "Chat model: $NANOCHAT_BASE_DIR/sft_checkpoints/d6_curriculum_4060ti_sft_curriculum/"
echo ""
echo "Training stages completed:"
echo "  ✓ Tokenizer (mixed corpus with 5% SFT samples)"
echo "  ✓ Curriculum pretraining (4 stages, multi-source)"
echo "  ✓ SFT curriculum (3 stages, 5 datasets)"
echo ""
echo "Next steps:"
echo "  - Chat with model: python -m scripts.chat_cli"
echo "  - Evaluate model: python -m scripts.chat_eval"
echo ""
