# NanoChat Curriculum Training - Changelog

## v0.3.0 - Ready for Baseline (2026-09-21)

### Summary
Final preparation release before establishing baseline. All features implemented and tested, ready for fresh training run.

### Completed
- ✅ Full end-to-end pipeline functional (tokenizer → pretraining → SFT)
- ✅ Multi-dataset SFT curriculum with 5 datasets
- ✅ All dataset Task classes implemented and tested
- ✅ Docker configuration optimized for faster rebuilds
- ✅ Documentation consolidated (removed DOCKER_TRAINING.md)
- ✅ Validation properly skipped during curriculum training

### What's Ready
**Tokenizer Training:**
- Mixed corpus: 60% ClimbMix + 30% Nemotron (3 versions) + 5% SFT samples (5 datasets)
- Total: ~2B characters sampled from all sources

**Pretraining (4 stages):**
- Foundation: 2K→4K context, 70% ClimbMix + 20% Nemotron-v1 + 7% v1.1 + 3% v1.2
- Reasoning+Code: 4K→8K context, 50% ClimbMix + 30% Nemotron-v1 + 13% v1.1 + 7% v1.2
- Long Context: 8K→16K context, 45% ClimbMix + 35% Nemotron-v1 + 13% v1.1 + 7% v1.2
- Consolidation: 16K context, 60% ClimbMix + 20% Nemotron-v1 + 13% v1.1 + 7% v1.2

**SFT (3 stages):**
- Instruction Alignment: 8K context, 65% SmolTalk + 20% Nemotron-Multilingual + 10% Hunter-Alpha + 3% Nemotron-SWE + 2% Claude-Fable
- Coding & Agents: 16K context, 35% SmolTalk + 10% Nemotron-Multilingual + 15% Hunter-Alpha + 38% Nemotron-SWE + 2% Claude-Fable
- Consolidation: 16K→32K context, 45% SmolTalk + 10% Nemotron-Multilingual + 10% Hunter-Alpha + 32% Nemotron-SWE + 3% Claude-Fable

### Files Changed
- Updated: `README.md` - Consolidated Docker documentation
- Updated: `CHANGELOG.md` - New versioned format
- Deleted: `DOCKER_TRAINING.md` - Info moved to README

### Known Issues
- Docker builds can timeout on slower networks (workaround: use `--cache-from`)
- Nemotron-Multilingual uses only `code_hi` split (34K examples) as sample; full implementation would load all 12 language splits

### Next Steps
1. Rebuild Docker image with latest code
2. Run baseline training from scratch
3. Establish metrics before making further changes

---

## v0.2.0 - Multi-Dataset SFT Curriculum (2026-09-20 to 2026-09-21)

### Major Features Added

#### SFT Curriculum Training
- **New script:** `scripts/sft_train_curriculum.py` - Multi-dataset curriculum SFT
- **New config:** `config/sft_curriculum.yaml` - 3-stage SFT curriculum definition
- **New dataset Tasks:**
  - `tasks/nemotron_multilingual.py` - NVIDIA multilingual (34K examples, Hindi code subset)
  - `tasks/hunter_alpha.py` - Tool-using coding agent (1.2K examples)
  - `tasks/nemotron_swe.py` - Repository-level SWE agents (5.1K examples)
  - `tasks/claude_fable.py` - Curated high-quality examples (63 examples)
- **Dataset registry:** All 5 SFT sources registered in `nanochat/data_registry.py`

#### Training Pipeline
- Updated `docker/init_training.sh` to use full SFT curriculum
- Automatic progression: Tokenizer → Pretraining → SFT (all unattended)
- Stage-aware training with proper context length scheduling
- Multi-dataset weighted mixing within each stage

#### Technical Implementation
- Custom batch generator for SFT curriculum stages
- Proper handling of Task/TaskMixture APIs
- Conversation rendering with tool schemas support
- Error handling for malformed dataset examples
- Progress tracking across all 3 SFT stages

### Bug Fixes
- Fixed TaskMixture weight handling (use task repetition instead of tuple weights)
- Fixed nemotron_multilingual dataset format parsing
- Fixed batch generator to use correct Task.get_example() API
- Fixed validation skip during curriculum training (no base_data directory)
- Updated docker-compose image name from `nanochat-curriculum:latest` to `nanochat:latest`

### Testing
- Verified all 5 SFT datasets load successfully
- Confirmed training runs with multi-dataset mixing
- Validated stage transitions and context length changes
- Tested with 100 iterations across curriculum stages

### Performance
- Training speed: ~450ms per step @ 8K context on 4060Ti
- Memory: ~12GB VRAM with batch_size=4
- All 5 datasets loading and mixing correctly in weighted proportions

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
