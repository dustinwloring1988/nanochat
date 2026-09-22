# NanoChat Curriculum Training - Changelog

## v0.2.0 - Multi-Dataset SFT Curriculum & Baseline Run (2026-09-20 to 2026-09-22)

### Summary
Complete implementation, end-to-end verification, and baseline benchmark execution of the multi-source, multi-dataset curriculum training pipeline (Tokenizer → 4-stage Pretraining → 3-stage SFT → Automated Benchmarking).

### Completed Features

#### Multi-Dataset SFT Curriculum
- **New script:** `scripts/sft_train_curriculum.py` - Multi-dataset curriculum SFT supporting context ramps and dynamic shapes.
- **New config:** `config/sft_curriculum.yaml` - 3-stage SFT curriculum definition:
  - Stage 1 (Instruction Alignment): 8K context, 65% SmolTalk + 20% Nemotron-Multilingual + 10% Hunter-Alpha + 3% Nemotron-SWE + 2% Claude-Fable
  - Stage 2 (Coding & Agents): 16K context, 35% SmolTalk + 10% Nemotron-Multilingual + 15% Hunter-Alpha + 38% Nemotron-SWE + 2% Claude-Fable
  - Stage 3 (Consolidation): 16K→32K context ramp, 45% SmolTalk + 10% Nemotron-Multilingual + 10% Hunter-Alpha + 32% Nemotron-SWE + 3% Claude-Fable
- **New dataset Tasks:**
  - `tasks/nemotron_multilingual.py` - NVIDIA multilingual (34K examples, Hindi code subset)
  - `tasks/hunter_alpha.py` - Tool-using coding agent (1.2K examples)
  - `tasks/nemotron_swe.py` - Repository-level SWE agents (5.1K examples)
  - `tasks/claude_fable.py` - Curated high-quality examples (63 examples)
- **Dataset Registry:** All 5 SFT sources registered in `nanochat/data_registry.py`.

#### End-to-End Automated Pipeline & Benchmarking
- Updated `docker/init_training.sh` to run unattended end-to-end:
  - Step 1: Mixed tokenizer training (~2B characters)
  - Step 1.5: Pretraining datasets download
  - Step 2: 4-stage curriculum pretraining (`d6_curriculum_4060ti`)
  - Step 3: 3-stage multi-dataset SFT curriculum
  - Step 4: Automated Benchmarking (`infer_bench` and `chat_eval`)
- Updated `nanochat/checkpoint_manager.py` to seamlessly resolve checkpoints in both `sft_checkpoints/` and `chatsft_checkpoints/`.

### Bug Fixes During Baseline Run
- **Torch Compile Dynamic Shapes:** Switched `torch.compile(model, dynamic=True)` in `scripts/sft_train_curriculum.py` to prevent `torch._dynamo` hitting cache recompilation limits and falling back to eager mode during the Stage 3 context ramp.
- **RoPE Embedding Cache Expansion:** Increased default rotary embedding buffer in `nanochat/gpt.py` from `config.sequence_len * 10` (20,480) to `max(config.sequence_len * 10, 65536)`, and added dynamic buffer doubling fallback in `forward` to support sequences up to 32,768+ tokens.
- **SFT Checkpoint Metadata:** Fixed `scripts/sft_train_curriculum.py` to persist `model_config` in `meta_*.json` checkpoints so that downstream evaluation tools (`chat_eval`, `infer_bench`, `chat_cli`) can automatically reconstruct the model architecture.
- **Task API & Batching:** Fixed `TaskMixture` weighting, dataset formatting parsers, and validation skipping for curriculum pretraining.

### Baseline Benchmark Results (RTX 4060 Ti 16GB, depth=6 test model)

#### 1. Inference Performance (`infer_bench`)
- **Model Parameters:** 73,531,538 (50.3M bfloat16, 23.2M float32) | Weight bytes: 185 MiB
- **Prefill (Prompt: 2,048 tokens):** **133,775 tok/s** | TTFT: **15.4 ms**
- **Decode Performance:**
  - Batch 1: **235 tok/s** (TPOT: 4.13 ms) | Peak VRAM: 1.04 GiB
  - Batch 8: **1,828 tok/s** (TPOT: 4.23 ms) | Peak VRAM: 1.04 GiB
  - Batch 32: **5,885 tok/s** (TPOT: 5.39 ms) | Peak VRAM: 1.17 GiB
  - Batch 128: **9,490 tok/s** (TPOT: 13.44 ms) | Peak VRAM: 3.09 GiB

#### 2. Chat Quality Evaluation (`chat_eval`, 50 problems/task)
- **ARC-Easy:** **26.00%** (Baseline: 25.0%)
- **ARC-Challenge:** **22.00%** (Baseline: 25.0%)
- **MMLU:** **28.00%** (Baseline: 25.0%)
- **GSM8K:** **0.00%** (Open-ended math reasoning)
- **HumanEval:** **0.00%** (Open-ended code synthesis)
- **ChatCORE Score:** **0.0027**

---

## v0.1.0 - Initial Curriculum Implementation (2026-09-19 to 2026-09-20)

### Summary
Forked from [karpathy/nanochat](https://github.com/karpathy/nanochat) and implemented comprehensive multi-stage curriculum training infrastructure.

### Core Infrastructure Added

#### Data Management
- **New module:** `nanochat/data_registry.py` - Centralized dataset registry
  - Support for pretraining sources: ClimbMix, Nemotron v1/v1.1/v1.2
  - License tracking and compliance metadata
  - Download management with retry logic
- **New module:** `nanochat/multi_source_dataloader.py` - Multi-source training
  - Weighted source sampling
  - Per-source document buffers
  - Multi-subset support for Nemotron datasets
  - DDP-aware distributed loading

#### Curriculum System
- **New module:** `nanochat/curriculum.py` - Stage-based curriculum scheduling
  - CurriculumStage dataclass for stage definitions
  - CurriculumScheduler for stage transitions
  - Context length interpolation within stages
  - Source weight management per stage
  - Full serialization for checkpoint resume
- **New module:** `nanochat/tokenizer_corpus.py` - Mixed corpus builder
  - Sample documents from multiple sources
  - Configurable mixing ratios
  - Default mix: 60/15/10/10 (ClimbMix/Nemotron-v1/v1.1/v1.2) + 5% SFT

#### Configuration Files
- **New config:** `config/pretraining_curriculum.yaml` - 4-stage pretraining curriculum
- **New config:** `config/sft_curriculum.yaml` - 3-stage SFT curriculum  
- **New config:** `config/data_sources.yaml` - Complete dataset documentation

#### Training Scripts
- **New script:** `scripts/tok_train_curriculum.py` - Train tokenizer on mixed corpus
- **New script:** `scripts/base_train_curriculum.py` - Curriculum-aware pretraining
- **New script:** `scripts/sft_train_curriculum.py` - SFT with curriculum (initial)
- **New run:** `runs/curriculum_4060ti.sh` - Single-GPU training optimized for RTX 4060Ti

### Docker Support
- **New file:** `Dockerfile` - PyTorch 2.5.1 with CUDA 12.4
- **New file:** `docker-compose.yml` - Service definitions for training
- **New file:** `docker/entrypoint.sh` - Container initialization
- **New file:** `docker/init_training.sh` - Full automated training pipeline
- **New file:** `.dockerignore` - Exclude data files from builds

### Key Features
- **4-stage pretraining curriculum:**
  1. Foundation (55%): 2K→4K context, mostly ClimbMix
  2. Reasoning+Code (28%): 4K→8K context, more Nemotron-v1
  3. Long Context (12%): 8K→16K context, balanced mixing
  4. Consolidation (5%): 16K context, return to foundation mix
  
- **Multi-source data mixing:**
  - Weighted sampling at document level
  - Per-stage source weight adjustments
  - Subset-level weights for Nemotron datasets (6+5+4 subsets)
  
- **Context length scheduling:**
  - Gradual ramps within stages (e.g., 2K → 4K in stage 1)
  - Linear interpolation based on stage progress
  - Automatic adjustment as curriculum advances

### Technical Details
- Multi-subset dataloader: Separate iterators per subset with weighted sampling
- Curriculum state serialization: Full resume from any checkpoint
- Backward compatible: All original scripts unchanged
- Docker optimization: Dependencies cached separately from code
- Dataset sampling: Configurable sample sizes for fast testing

### Bug Fixes Applied
- Fixed Nemotron subset names to match HuggingFace BuilderConfig:
  - v1: RQA, STEM-SFT, Math-Textbooks, InfiniByte-Reasoning, Wiki-Rewrite, Scientific-Coding
  - v1.1: Code-Concepts, Multiple-Choice, Formal-Logic, Unconditional-Algorithmic, Economics
  - v1.2: Fact-Seeking, Multiple-Choice, Generative, Moral-Scenarios
- Fixed multi-subset dataloader: Create separate iterator per subset
- Fixed validation skip during curriculum training (no base_data directory needed)
- Added Nemotron pre-download step before pretraining starts

### Testing Performed
- Tokenizer training on mixed corpus: ✅
- Multi-source dataloader initialization: ✅
- Curriculum stage transitions: ✅
- Docker build and container launch: ✅
- End-to-end training (100 steps): ✅
- Checkpoint save/resume: ✅

### Known Limitations
- SFT curriculum uses SmolTalk only (multi-dataset support came in v0.2.0)
- Nemotron datasets require manual HuggingFace setup (direct download not yet implemented)
- Context length changes don't dynamically recreate dataloader (logged but not applied)
- Subset weights defined in config but not yet used by dataloader (default subset only)

### Performance Metrics (depth=6 on RTX 4060Ti)
- Tokenizer: ~30-45 minutes
- Pretraining: ~2-4 hours (1,888 steps)
- Memory: ~12GB VRAM @ batch_size=24, context=2048
- Throughput: ~141K tokens/sec
- Total: ~3-5 hours for full pipeline

### License Compliance
- ClimbMix: CC-BY-NC-4.0 (research/non-commercial only)
- Nemotron v1/v1.1/v1.2: Various CC-BY licenses with considerations
- Documentation: All licenses tracked in config/data_sources.yaml

---

## Repository Origin

**Forked from:** [karpathy/nanochat](https://github.com/karpathy/nanochat)  
**Original commit:** [Latest as of 2026-09-19]  
**License:** MIT

This changelog documents modifications and additions made to implement curriculum-based training with multi-source datasets. The original nanochat codebase remains fully functional and backward compatible.
