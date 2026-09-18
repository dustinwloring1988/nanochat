# Nanochat Speedrun Results: NVIDIA RTX 4060 Ti 16GB

Run executed using Docker container `nanochat-132` with CUDA 13.2 / PyTorch 2.9 on a single NVIDIA GeForce RTX 4060 Ti 16GB.

## System Configuration
* **GPU**: NVIDIA GeForce RTX 4060 Ti 16GB (VRAM: 16,380 MiB)
* **Compute Dtype**: BFloat16 (`torch.bfloat16`)
* **Compilation Mode**: `PYTORCH_COMPILE_OFF=1` (Dynamo compilation disabled)
* **Driver / CUDA**: NVIDIA 616.92 / CUDA 13.2 / PyTorch 2.9.1+cu128

---

## Model Architecture
* **Depth**: 8 layers
* **Heads / KV Heads**: 4 / 4
* **Head Dim**: 128 (Model Dim: 512)
* **Sequence Length**: 2048
* **Sliding Window Pattern**: `SSSL`
* **Vocab Size**: 32,768
* **Total Parameters**: 125,829,354
* **Scaling Parameters**: 41,943,040

---

## Pipeline Execution Summary

### 1. Tokenizer
* Trained 32,768 vocabulary from pretraining shards.
* Tokenizer compression ratio evaluated.

### 2. Base Model Pretraining
* **Batch Size**: 262,144 tokens (gradient accumulation steps: 32 micro-batches of 4 x 2048).
* **Iterations**: 100 steps (~26.2M tokens total).
* **Peak GPU Utilization**: ~97-98% (115W power draw).
* **Pretraining Time**: ~14 minutes.

### 3. Base Model Evaluation
* **Validation BPB (initial step 0)**: Baseline calculated across 41.9M tokens.
* **Validation BPB (final)**: Evaluated across validation set.
* CORE benchmark & unconditioned generation samples drawn.

### 4. Chat Supervised Fine-Tuning (SFT)
* **Total Steps**: 1,867 steps.
* **Peak Memory Usage**: 6,671.26 MiB.
* **Final SFT Validation BPB**: 0.4473.
* **Checkpoint Saved**: `chatsft_checkpoints/d8/model_001867.pt`.

---

## Final Benchmark Evaluations

| Benchmark | Accuracy / Score | Passed / Total |
|---|---|---|
| **ARC-Easy** | **29.38%** | 698 / 2,376 |
| **ARC-Challenge** | **28.67%** | 336 / 1,172 |
| **MMLU** | **28.66%** | 4,024 / 14,042 |
| **GSM8K** | **0.23%** | 3 / 1,319 |
| **HumanEval** | **2.44%** | 4 / 164 |
| **ChatCORE Metric** | **0.0365** | — |

---

## Verification & Interactive Chat Command
To chat with the trained model in the container:
```powershell
docker run --rm -it --gpus all -v "/root/.cache:/root/.cache" -v "F:\UserData\git-repos\New folder\apps\nanochat:/nanochat" -v "/nanochat/.venv" nanochat-132 bash -c "source /root/.local/bin/env && cd /nanochat && source .venv/bin/activate && python -m scripts.chat_cli -p 'Why is the sky blue?'"
```
