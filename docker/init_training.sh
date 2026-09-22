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
echo "STEP 3: Supervised Fine-Tuning (SFT)"
echo "=========================================="
echo ""

# Check if base model exists
BASE_MODEL_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d6_curriculum_4060ti"
if [ ! -d "$BASE_MODEL_DIR" ]; then
    echo "ERROR: Base model not found at $BASE_MODEL_DIR"
    echo ""
    echo "Pretraining may have failed. Check logs above for errors."
    echo "SFT requires a pretrained base model to continue."
    echo ""
    exit 1
fi

echo "✓ Base model found at $BASE_MODEL_DIR"
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
echo ""
echo "=========================================="
echo "STEP 4: Benchmarking & Evaluation"
echo "=========================================="
echo ""

# 4a: Inference Performance Benchmark (latency, throughput, MBU)
echo "Running inference performance benchmark..."
python -m scripts.infer_bench -i sft -g d6_curriculum_4060ti_sft_curriculum

# 4b: Model Quality Evaluation (ChatCORE: ARC, MMLU, GSM8K, HumanEval)
echo ""
echo "Running chat quality benchmark (ChatCORE)..."
python -m scripts.chat_eval -i sft -g d6_curriculum_4060ti_sft_curriculum

echo ""
echo "=========================================="
echo "Training & Benchmarking Pipeline Complete!"
echo "=========================================="
echo ""
echo "Base model: $NANOCHAT_BASE_DIR/base_checkpoints/d6_curriculum_4060ti/"
echo "Chat model: $NANOCHAT_BASE_DIR/sft_checkpoints/d6_curriculum_4060ti_sft_curriculum/"
echo ""
echo "Training stages completed:"
echo "  ✓ Tokenizer (mixed corpus with 5% SFT samples)"
echo "  ✓ Curriculum pretraining (4 stages, multi-source)"
echo "  ✓ SFT curriculum (3 stages, 5 datasets)"
echo "  ✓ Benchmarks (infer_bench + chat_eval)"
echo ""
echo "Next steps:"
echo "  - Chat with model: python -m scripts.chat_cli --model-tag d6_curriculum_4060ti_sft_curriculum"
echo ""
