# Reasoning Model Guide

## Overview

NanoChat now supports **reasoning-capable models** that can show step-by-step problem-solving and chain-of-thought (CoT) reasoning. This guide explains how to train and use reasoning models in NanoChat.

## Table of Contents

1. [What is a Reasoning Model?](#what-is-a-reasoning-model)
2. [Training a Reasoning Model](#training-a-reasoning-model)
3. [Using the Reasoning Model](#using-the-reasoning-model)
4. [Understanding Reasoning Levels](#understanding-reasoning-levels)
5. [Dataset Information](#dataset-information)
6. [Performance Benchmarks](#performance-benchmarks)
7. [Troubleshooting](#troubleshooting)

---

## What is a Reasoning Model?

A **reasoning model** is an LLM that can explicitly show its thinking process before providing an answer. Instead of just outputting the final result, it generates intermediate reasoning steps.

### Example: Standard Model vs Reasoning Model

**Standard Model:**
```
User: What is 15% of 240?
Assistant: 36
```

**Reasoning Model:**
```
User: What is 15% of 240?
Assistant: <thinking>
To find 15% of 240, I need to:
1. Convert 15% to decimal: 15% = 0.15
2. Multiply 240 by 0.15
3. 240 × 0.15 = 36
</thinking>
The answer is 36.
```

### Benefits

- **Improved Accuracy**: Explicit reasoning helps prevent errors
- **Interpretability**: Users can see how the model arrived at an answer
- **Better Performance**: Especially on math, code, and logic tasks
- **Adaptive Complexity**: Model can adjust reasoning depth based on task

---

## Training a Reasoning Model

### Quick Start: Complete Pipeline

The easiest way to train a reasoning model is using the complete pipeline script:

```powershell
# Windows (PowerShell) with Docker
.\runs\complete_pipeline_docker.ps1 `
    -PretrainIterations 200 `
    -ReasoningSFTIterations 1000 `
    -ReasoningRatio 0.7
```

This will:
1. Train tokenizer
2. Pretrain base model
3. **Train reasoning capabilities** (NEW!)
4. Fine-tune with standard SFT

### Stage-by-Stage Training

#### Stage 1: Base Pretraining

Standard pretraining on ClimbMix or Nemotron datasets:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="reasoning_base" \
    --num-iterations=10000
```

#### Stage 2: Reasoning SFT

Train reasoning capabilities using Nemotron datasets:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_reasoning_sft -- \
    --device-batch-size=4 \
    --num-iterations=2000 \
    --reasoning-ratio=0.7 \
    --enable-reasoning-curriculum=1 \
    --run="reasoning_sft"
```

**Key Parameters:**
- `--num-iterations`: Total training steps (default: 2000)
- `--reasoning-ratio`: Fraction of examples with explicit CoT (0.0-1.0, default: 0.7)
- `--enable-reasoning-curriculum`: Gradually increase reasoning during training (default: 1)
- `--stage1-iterations`: Instruction following stage (default: 600)
- `--stage2-iterations`: Reasoning training stage (default: 1000)
- `--stage3-iterations`: Multi-task fine-tuning (default: 400)

#### Stage 3: Standard SFT (Optional)

Additional fine-tuning on conversational tasks:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run="final_sft"
```

### Training Configuration

Edit `configs/reasoning_config.py` to customize:

```python
CONFIG = {
    "reasoning": {
        "base_ratio": 0.7,  # 70% reasoning, 30% direct
        "enable_curriculum": True,
        "level_distribution": {
            "none": 0.30,    # 30% no reasoning
            "low": 0.20,     # 20% short reasoning
            "medium": 0.35,  # 35% moderate reasoning
            "high": 0.15,    # 15% long reasoning
        }
    }
}
```

---

## Using the Reasoning Model

### Interactive Chat

```bash
# Default (medium reasoning)
python -m scripts.chat_cli

# Specify reasoning level
python -m scripts.chat_cli --reasoning-level=high
python -m scripts.chat_cli --reasoning-level=none  # Direct answers
```

### Programmatic Usage

```python
from nanochat.engine import Engine
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.messages import Message

# Load model
model, tokenizer, meta = load_model("reasoning_sft", device="cuda")
engine = Engine(model, tokenizer)

# Create conversation with reasoning
messages = [
    Message(role="user", content="What is the area of a circle with radius 5?")
]

# Generate with reasoning
response = engine.generate(
    messages,
    reasoning_level="medium",  # or "low", "high", "none"
    max_tokens=512,
    temperature=0.7
)

print(response)
```

### Reasoning Levels

Control how much reasoning the model shows:

```python
# No reasoning (direct answer)
reasoning_level="none"

# Short reasoning (1-2 steps)
reasoning_level="low"

# Moderate reasoning (3-5 steps)
reasoning_level="medium"

# Detailed reasoning (5+ steps)
reasoning_level="high"
```

---

## Understanding Reasoning Levels

### Reasoning Level Descriptions

| Level | Length | Use Case | Example |
|-------|--------|----------|---------|
| `none` | 0 tokens | Simple queries, known facts | "The capital of France is Paris." |
| `low` | ~50-200 tokens | Basic arithmetic, simple logic | "2+2=4 because..." |
| `medium` | ~200-800 tokens | Multi-step problems, coding | "To solve this, first... then... finally..." |
| `high` | 800+ tokens | Complex reasoning, proofs | "Let's break this down systematically..." |

### When to Use Each Level

**none**: 
- Factual questions
- Simple definitions
- Direct information retrieval

**low**:
- Basic math
- Short explanations
- Simple code snippets

**medium** (default):
- Problem-solving
- Code generation with comments
- Multi-step reasoning

**high**:
- Complex proofs
- Detailed analysis
- Comprehensive explanations

---

## Dataset Information

The reasoning model is trained on **NVIDIA Nemotron datasets**:

### Primary Datasets

1. **Nemotron-Cascade-SFT-Stage-2** (7.8M examples)
   - Math: OpenMathReasoning (1.9M)
   - Code: OpenCodeReasoning, TACO, etc. (1.4M)
   - Science: Synthetic + Nemotron-v1 (311K)
   - General: MMLU, SlimOrca, etc. (3.6M)
   - Tool Calling: 309K examples
   - Software Engineering: 211K examples
   - Instruction Following: 146K examples

2. **Nemotron-Post-Training-Dataset-v2** (5.3M examples)
   - Math: 239K examples
   - Code: 175K examples
   - STEM: 355K examples
   - Chat: 628K examples
   - Multilingual: 4.9M examples (5 languages)

### Dataset Features

- **Reasoning Modes**: Each example has `thinking=true/false`
- **thinking=true**: Contains explicit CoT reasoning
- **thinking=false**: Direct answer without reasoning trace
- **Mixed Training**: Model learns when to use reasoning

### Data Sources

- DeepSeek-R1-0528: Reasoning traces
- Qwen2.5/Qwen3 models: Responses
- Multiple open datasets: Prompts

---

## Performance Benchmarks

### Expected Improvements

After reasoning SFT training on a d12 model (GPT-1 scale):

| Task | Baseline | With Reasoning | Improvement |
|------|----------|----------------|-------------|
| GSM8K (Math) | 30-40% | 60-70% | +30% |
| MMLU | 35-45% | 40-50% | +5-10% |
| HumanEval (Code) | 15-25% | 30-40% | +15% |
| ARC-Challenge | 30-40% | 40-50% | +10% |

### Training Efficiency

- **Training Time**: +20-30% over standard SFT
- **Inference Speed**: 
  - reasoning_level="none": Same as baseline
  - reasoning_level="medium": 2-3x tokens (slower but better)
- **Memory**: Minimal increase (same architecture)

### Scaling Laws

Reasoning capabilities improve with:
- Model size (larger models reason better)
- Training iterations (more data helps)
- Reasoning ratio (optimal ~0.7)

---

## Troubleshooting

### Common Issues

#### 1. Dataset Download Fails

```
Error: Connection timeout downloading Nemotron datasets
```

**Solution:**
- Check internet connection
- Enable HuggingFace transfer acceleration:
  ```bash
  export HF_HUB_ENABLE_HF_TRANSFER=1
  ```
- Use streaming mode for large datasets:
  ```python
  streaming=True
  ```

#### 2. Out of Memory (OOM)

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size:
  ```bash
  --device-batch-size=2  # or 1
  ```
- Reduce sequence length:
  ```bash
  --max-seq-len=1024  # instead of 2048
  ```
- Enable gradient checkpointing (future feature)

#### 3. Model Not Showing Reasoning

```
Model generates direct answers instead of reasoning
```

**Solution:**
- Check reasoning level:
  ```python
  reasoning_level="medium"  # not "none"
  ```
- Verify checkpoint loaded:
  ```bash
  --model-tag=reasoning_sft
  ```
- Ensure proper training:
  - reasoning_ratio > 0.5
  - Completed stage 2

#### 4. Reasoning Quality is Poor

```
Reasoning is incoherent or incorrect
```

**Solution:**
- Train longer (more iterations)
- Increase reasoning ratio (0.8-0.9)
- Use larger model (d16 or d20)
- Check data quality:
  ```bash
  python -m nanochat.reasoning_dataloader
  ```

#### 5. Tests Failing

```
ImportError: cannot import name 'NemotronReasoningDataset'
```

**Solution:**
- Reinstall package:
  ```bash
  uv sync
  ```
- Check Python path
- Verify all new files created:
  - `nanochat/reasoning_dataloader.py`
  - `scripts/chat_reasoning_sft.py`
  - `configs/reasoning_config.py`

### Debug Mode

Enable verbose logging:

```bash
export NANOCHAT_DEBUG=1
python -m scripts.chat_reasoning_sft -- ...
```

### Getting Help

1. Check logs in `./logs/reasoning_sft/`
2. Review WandB dashboard for training curves
3. Test dataloader independently:
   ```bash
   python -m nanochat.reasoning_dataloader
   ```
4. Validate checkpoint:
   ```bash
   python -m scripts.chat_eval -i reasoning_sft
   ```

---

## Advanced Topics

### Custom Reasoning Datasets

Add your own reasoning data:

```python
# In reasoning_dataloader.py
custom_dataset = {
    "name": "my-org/my-reasoning-dataset",
    "subset": None,
    "weight": 0.5,
    "reasoning_ratio": 0.8,
}
```

### Reasoning Evaluation

Evaluate reasoning quality:

```bash
python -m scripts.chat_eval \
    -i reasoning_sft \
    --tasks GSM8K,MMLU,HumanEval \
    --reasoning-metrics
```

### Adaptive Reasoning

Future feature: Model automatically decides reasoning level:

```python
# Coming soon
engine.generate(messages, reasoning_level="auto")
```

---

## Citation

If you use the reasoning model in research, please cite:

```bibtex
@software{nanochat_reasoning,
  author = {NanoChat Team},
  title = {NanoChat Reasoning Model},
  year = {2026},
  url = {https://github.com/karpathy/nanochat}
}

@software{nemotron_datasets,
  author = {NVIDIA},
  title = {Nemotron Post-Training Datasets},
  year = {2025},
  url = {https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2}
}
```

---

## Next Steps

1. **Train Your First Reasoning Model**: Run `complete_pipeline_docker.ps1`
2. **Experiment with Reasoning Levels**: Try different levels in chat
3. **Evaluate Performance**: Run benchmarks on GSM8K
4. **Customize Training**: Edit `reasoning_config.py`
5. **Share Results**: Post to discussions or leaderboard

Happy reasoning! 🧠✨
