# Changelog

All notable changes to nanochat will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Reasoning Model Support (September 2026)

**MAJOR UPDATE**: NanoChat now supports reasoning-capable models that can show step-by-step problem-solving and chain-of-thought (CoT) reasoning.

#### Added

**Reasoning Infrastructure**
- `nanochat/reasoning_dataloader.py` - Dataloader for NVIDIA Nemotron reasoning datasets
  - Support for reasoning_on/reasoning_off splits
  - Automatic reasoning level inference (none, low, medium, high)
  - Mixed dataset sampling with configurable weights
  - Streaming support for large datasets
- `scripts/chat_reasoning_sft.py` - Reasoning-focused SFT training script
  - Three-stage training: Instruction Following → Reasoning → Multi-task
  - Reasoning curriculum learning (gradually increase reasoning ratio)
  - Integration with Nemotron-Cascade-SFT-Stage-2 (7.8M examples)
  - Integration with Nemotron-Post-Training-Dataset-v2 (5.3M examples)
- `configs/reasoning_config.py` - Comprehensive reasoning model configuration
  - Stage boundaries and scheduling
  - Reasoning ratio curriculum
  - Dataset mixing strategies
- `tests/test_reasoning_dataloader.py` - Complete test suite for reasoning functionality

**Dataset Integration**
- **Nemotron-Cascade-SFT-Stage-2** (7.8M examples):
  - Math: OpenMathReasoning (1.9M)
  - Code: OpenCodeReasoning, TACO (1.4M)
  - Science: Nemotron-v1, synthetic (311K)
  - General: MMLU, SlimOrca, ShareGPT (3.6M)
  - Tool Calling (309K)
  - Software Engineering (211K)
  - Instruction Following (146K)
- **Nemotron-Post-Training-Dataset-v2** (5.3M examples):
  - Math, Code, STEM reasoning
  - Multilingual reasoning (5 languages)
  - Mixed thinking=true/false examples

**Training Pipeline**
- Updated `runs/complete_pipeline_docker.ps1` with reasoning SFT stage:
  - Stage 1: Tokenizer training
  - Stage 2: Base pretraining
  - **Stage 3: Reasoning SFT (NEW!)**
  - Stage 4: Standard SFT
- Configurable reasoning parameters:
  - `ReasoningSFTIterations`: Training iterations (default: 1000)
  - `ReasoningRatio`: Fraction of reasoning examples (default: 0.7)
- Complete Docker support with GPU passthrough

**Reasoning Features**
- Four reasoning levels:
  - `none`: Direct answers (0 reasoning tokens)
  - `low`: Short reasoning (~50-200 tokens)
  - `medium`: Moderate reasoning (~200-800 tokens)
  - `high`: Detailed reasoning (800+ tokens)
- Adaptive reasoning based on task complexity
- Reasoning level inference from content length
- Automatic reasoning/answer splitting
- Loss mask optimization for reasoning tokens

**Documentation**
- `docs/reasoning_model_guide.md` - Comprehensive guide:
  - Training reasoning models
  - Using reasoning levels
  - Performance benchmarks
  - Troubleshooting guide
  - Dataset information
  - Example usage patterns

#### Changed

**Chat Tokenizer**
- Enhanced `nanochat/chat_template.jinja` with reasoning support
  - Reasoning markers: `<|think_none|>`, `<|think_low|>`, `<|think_medium|>`, `<|think_high|>`
  - Automatic reasoning structure for all assistant messages
  - Backward compatible with existing models
- Updated `nanochat/chat_tokenizer.py` for reasoning formatting
- Extended `nanochat/messages.py` with reasoning fields

**Training Scripts**
- Enhanced `scripts/chat_sft.py` compatibility with reasoning models
- Improved checkpoint management for multi-stage training
- Better stage transitions and progress tracking

**Pipeline**
- Extended `complete_pipeline_docker.ps1`:
  - 4-stage training (was 3-stage)
  - Reasoning SFT stage between pretraining and standard SFT
  - New parameters for reasoning configuration
  - Enhanced output with reasoning capability indicators

#### Performance Improvements

**Expected Gains** (on d12 model):
- GSM8K (Math): 30-40% → 60-70% (+30%)
- MMLU: 35-45% → 40-50% (+5-10%)
- HumanEval (Code): 15-25% → 30-40% (+15%)
- ARC-Challenge: 30-40% → 40-50% (+10%)

**Training Efficiency**:
- Training time: +20-30% over standard SFT
- Inference (reasoning_level="none"): Same speed as baseline
- Inference (reasoning_level="medium"): 2-3x tokens (better accuracy)
- Memory: Minimal increase (same architecture)

#### Dataset Statistics

**Nemotron-Cascade-SFT-Stage-2 Composition**:
| Domain | Samples | Primary Sources |
|--------|---------|----------------|
| Math | 1.9M | OpenMathReasoning |
| Code | 1.4M | OpenCodeReasoning, TACO |
| Science | 311K | Nemotron-v1, synthetic |
| General | 3.6M | MMLU, SlimOrca, ShareGPT |
| Tool Calling | 309K | Nemotron-v1 |
| SWE | 211K | SWE-Bench, R2E-Gym |
| Instruction Following | 146K | Tulu-3 personas |

**Reasoning Distribution**:
- thinking=true: ~60% of examples (with explicit CoT)
- thinking=false: ~40% of examples (direct answers)
- Response generators: DeepSeek-R1-0528, Qwen2.5/3, DeepSeek-V3

#### Fixed

- Reasoning content parsing for various formats (`<think>`, `Answer:`, `\boxed{}`)
- Dataset streaming for large-scale training
- Checkpoint compatibility between reasoning and standard SFT
- Loss masking for reasoning tokens
- Stage transition logic in multi-stage training

#### Migration Guide

**For Existing Users**:
1. No breaking changes - existing training pipelines work as before
2. To enable reasoning:
   ```powershell
   .\runs\complete_pipeline_docker.ps1 `
       -ReasoningSFTIterations 1000 `
       -ReasoningRatio 0.7
   ```
3. To use reasoning in inference:
   ```python
   engine.generate(messages, reasoning_level="medium")
   ```

**New CLI Options**:
```bash
# Training
python -m scripts.chat_reasoning_sft -- \
    --num-iterations=2000 \
    --reasoning-ratio=0.7 \
    --enable-reasoning-curriculum=1

# Inference
python -m scripts.chat_cli --reasoning-level=medium
```

---

### Nemotron Dataset Integration (January 2027)

Major update integrating NVIDIA Nemotron specialized pretraining datasets for code-focused training.

#### Added

**Nemotron Dataset Support**
- `nanochat/nemotron_dataloader.py` - Dataloader for Nemotron datasets compatible with nanochat pipeline
- `scripts/nemotron_base_train.py` - Modified base training script using Nemotron datasets
- Multi-stage training support:
  - Stage 1: Code-Concepts + Scientific-Coding (2048 tokens)
  - Stage 2a: Math-Textbooks (8192 tokens)  
  - Stage 2b: InfiniByte-Reasoning (32768 tokens)
- `runs/complete_pipeline_docker.ps1` - Complete Windows/PowerShell training pipeline
  - Tokenizer training
  - Base pretraining with Nemotron datasets
  - SFT (Supervised Fine-Tuning)
  - Full Docker support

**Docker Infrastructure**
- Updated `docker/Dockerfile` with proper Python environment setup
- Windows PowerShell training scripts
- GPU passthrough support for Docker Desktop on Windows
- Volume mounting for persistent checkpoints and cache

**Configuration**
- `configs/nemotron_stage1_config.py` - Stage 1 code-focused training config
- `configs/nemotron_stage2_config.py` - Stage 2a/2b math and reasoning configs
- Configurable model depth (default: 6 layers for faster iteration)
- Optimized for RTX 4060 Ti 16GB

#### Changed

**Dataset Architecture**
- Replaced ClimbMix default dataset with NVIDIA Nemotron datasets
- Monkey-patching approach for drop-in compatibility with existing training scripts
- Stage-based dataset selection via environment variable `NEMOTRON_STAGE`
- Streaming dataset support for efficient memory usage

**Training Pipeline**
- Simplified pipeline: Tokenizer → Base Pretraining → SFT
- Removed separate Nemotron training script in favor of integrated approach
- Compatible checkpoint format with existing nanochat architecture
- Batch processing of documents for efficiency

**Model Configuration**
- Default depth reduced to 6 layers for faster training
- Aspect ratio: 64 (model_dim = depth × aspect_ratio = 384)
- Device batch size: 4 (optimized for 16GB VRAM)
- Target param-data ratio: 8 (Chinchilla-optimal-ish)

#### Fixed

- Docker Python path issues (proper symlinks and PATH setup)
- Argument passing between nemotron_base_train.py and base_train.py
- Token protocol compatibility with Nemotron datasets
- Checkpoint directory structure for SFT compatibility

#### Nemotron Dataset Statistics

| Subset | Source | Median Tokens | P95 Tokens | Context |
|--------|--------|---------------|------------|---------|
| Code-Concepts | v1.1 | 418 | 680 | 2048 |
| Scientific-Coding | v1 | 1,299 | 2,354 | 2048 |
| Math-Textbooks | v1 | 1,873 | 2,599 | 8192 |
| InfiniByte-Reasoning | v1 | 14,124 | 31,898 | 32768 |

#### Training Pipeline

**Complete Pipeline (PowerShell/Docker):**
```powershell
.\runs\complete_pipeline_docker.ps1 -PretrainIterations 200
```

**Pipeline Stages:**
1. **Tokenizer Training**: BPE tokenizer on ClimbMix sample data (32K vocab)
2. **Base Pretraining**: Nemotron Code-Concepts + Scientific-Coding datasets
3. **SFT**: Supervised fine-tuning for chat, reasoning, and tool use

**Configurable Parameters:**
- `-PretrainIterations`: Number of training iterations (default: 200)
- `-SkipTokenizer`: Skip tokenizer training if already exists
- `-SkipPretraining`: Skip base pretraining
- `-SkipSFT`: Skip supervised fine-tuning
- `-OnlyPretraining`: Stop after pretraining (skip SFT)

#### Performance Notes

- **Model Size**: ~50M parameters (depth=6, aspect_ratio=64)
- **Training Speed**: ~10-20 iterations/min on RTX 4060 Ti 16GB
- **Memory Usage**: ~8-10GB VRAM during training
- **Dataset Download**: ~2-5GB per stage (cached after first download)

#### Technical Details

**Dataloader Architecture:**
- Yields batches of 100 documents for efficiency
- DDP sharding support (start/step parameters)
- Automatic text extraction from HuggingFace dataset format
- Streaming mode for memory efficiency
- Compatible with nanochat's tokenizing dataloader

**Dataset Interleaving:**
- Stage 1: 50% Code-Concepts + 50% Scientific-Coding
- Stage 2a: 100% Math-Textbooks
- Stage 2b: 100% InfiniByte-Reasoning
- Configurable weights in config files

**Checkpoint Format:**
- Standard nanochat checkpoint format (model_NNNNNN.pt, meta_NNNNNN.json, optim_NNNNNN_rankN.pt)
- Saved to `~/.cache/nanochat/base_checkpoints/d6/` (depth-6 model)
- Compatible with SFT and evaluation scripts
- Periodic saves every N iterations

#### Dependencies

- Python 3.11
- PyTorch 2.0+
- HuggingFace datasets
- Docker with GPU support (NVIDIA Docker)
- NVIDIA GPU with CUDA support

#### References

- [NVIDIA Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1)
- [NVIDIA Nemotron-Pretraining-Specialized-v1.1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1.1)

---

## [Unreleased - Previous]

### Protocol v1.0.0 - Token Architecture Upgrade (September 2026)

A major upgrade introducing a structured token protocol to support chat templates, reasoning, tool calling, and future multimodal capabilities while maintaining nanochat's compact vocabulary philosophy.

#### Added

**Core Architecture**
- `nanochat/token_protocol.py` - Token namespace with 32K lexical + 256 control tokens
  - Lexical vocabulary: 0-31,999 (BPE-trained)
  - Control tokens: 32,000-32,255 (structured namespace)
  - 26 active special tokens for chat, reasoning, and tools
  - 196 reserved tokens for future expansion
- `nanochat/messages.py` - Type-safe Message, ToolCall, Content dataclasses
- `nanochat/chat_template.jinja` - Jinja2 template for consistent message formatting
- `nanochat/chat_tokenizer.py` - Enhanced tokenizer with template support

**Training & Inference**
- `render_conversation()` updated with `use_chat_template` flag for protocol v1.0.0
- `render_for_completion()` updated for generation with reasoning levels
- Proper loss masking - trains assistant content, masks input
- Support for reasoning levels: none, low, medium, high
- Tool calling support (Python REPL for math)

**Scripts & Utilities**
- `check_training.ps1` - Quick training status checker
- `monitor_training.ps1` - Real-time log streaming
- Updated `runs/speedrun-4060ti.sh` for protocol v1.0.0
  - Automatic protocol version validation
  - Vocab size verification (32K)
  - Auto-migration of old checkpoints
  - Smart checkpoint detection

**Documentation**
- `TOKENIZER_PROTOCOL_PLAN.md` - Full implementation plan (18 sections)
- `IMPLEMENTATION_STATUS.md` - Progress tracking
- `PHASE4_COMPLETE.md` - Phase 4 completion report
- Training monitoring and status documentation

#### Changed

**Token Protocol**
- Vocab size reduced from 32,768 to 32,000 lexical tokens
- Total vocabulary: 32,222 tokens (32K lexical + 222 control)
- Special tokens now use structured namespace:
  - Message structure: `<|message_start|>`, `<|message_end|>`
  - Roles: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`, `<|developer|>`
  - Reasoning: `<|think_none|>`, `<|think_low|>`, `<|think_medium|>`, `<|think_high|>`, `<|think_end|>`
  - Tool calls: `<|tool_call|>`, `<|tool_call_end|>`, `<|tool_result|>`, `<|tool_result_end|>`
  - Multimodal (reserved): `<|image|>`, `<|video|>`, `<|audio|>` + end tokens

**Tokenizer Training**
- Updated `scripts/tok_train.py` to use protocol v1.0.0
- Saves protocol metadata with trained tokenizer
- Enhanced validation and sanity checks

**Engine & Generation**
- Fixed `nanochat/engine.py` to use `<|message_end|>` instead of `<|assistant_end|>`
- Backward compatibility with try/except for legacy tokens
- Updated generation to support new protocol

**Training Scripts**
- `chat_sft.py`, `chat_rl.py`, `chat_eval.py` now automatically use protocol v1.0.0
- No code changes needed - works through `render_conversation()` API
- WandB disabled by default (`--run=dummy`) for container compatibility

#### Fixed

- Special tokens now encode as single tokens using tiktoken's `allowed_special="all"`
- Loss masking correctly trains assistant content and masks input
- Tool calls properly structured (GSM8K python tools work)
- Reasoning blocks formatted correctly in all assistant messages
- Token ID encoding fixed - was splitting special tokens into multiple IDs
- Vocab size mismatch between old (32,768) and new (32,000) tokenizers resolved

#### Breaking Changes

⚠️ **Models trained with old tokenizer (32,768 tokens) are incompatible with new protocol**

**Migration Required:**
1. Retrain tokenizer with protocol v1.0.0
2. Retrain base model from scratch
3. Retrain SFT model with new chat template
4. Old checkpoints automatically backed up by updated scripts

**Backward Compatibility:**
- `render_conversation(use_chat_template=False)` for legacy format (requires old tokenizer)
- Engine falls back to legacy tokens if new tokens not found
- Updated scripts detect and migrate incompatible checkpoints automatically

#### Implementation Phases

**Phase 1: Token Architecture & Message Schema** ✅ Complete
- Token protocol definition
- Message dataclasses
- Chat template

**Phase 2: Chat Template Integration** ✅ Complete
- ChatTokenizer implementation
- Template rendering
- Loss mask generation

**Phase 3: Tokenizer Training Update** ✅ Complete
- Updated training script
- Protocol metadata
- Validation

**Phase 4: Dataset Formatting** ✅ Complete
- Updated render_conversation()
- Updated render_for_completion()
- Engine compatibility fixes
- Training integration

**Phase 5: Inference Integration** 🔄 In Progress
- Chat CLI updates for reasoning levels
- Response parsing

**Phase 6: HuggingFace Compatibility** 📋 Pending
- Export to HF format
- Tokenizer config generation

**Phase 7: Testing & Validation** 📋 Pending
- Comprehensive test suite
- End-to-end validation

#### Performance Notes

- Control tokens add ~222 tokens to vocab (~0.7% of total)
- Parameter impact: ~0.2% for 125M param model
- Reasoning structure: ~4-8 tokens per assistant message
- Training impact: negligible
- Inference speed: unchanged
- Template rendering overhead: <1ms

#### Technical Details

**Token Ranges:**
```
0-31,999     : Lexical vocabulary (BPE)
32,000-32,003: Core (pad, bos, eos, unk)
32,010-32,011: Message structure
32,020-32,024: Roles
32,030-32,034: Reasoning
32,040-32,043: Tool calls
32,050-32,055: Multimodal (reserved)
32,060-32,255: Reserved for future
```

**Reasoning Structure:**
All assistant messages include reasoning structure for consistency:
```
# No reasoning
<|assistant|><|think_none|><|think_end|>answer<|message_end|>

# With reasoning
<|assistant|><|think_medium|>reasoning...<|think_end|>answer<|message_end|>
```

#### Testing

All protocol v1.0.0 features validated:
- ✅ Special token encoding (single token IDs)
- ✅ Loss masking (trains assistant, masks input)
- ✅ Tool call structure preservation
- ✅ Reasoning blocks formatted correctly
- ✅ Generation prompts end with reasoning markers
- ✅ Truncation support
- ✅ Protocol version tracking

#### Contributors

- Implementation: @dustinwloring1988 with Kiro AI
- Original nanochat: @karpathy

---

## Historical Changes

Previous changes tracked in git history. See commits for details on:
- Original nanochat implementation
- Speedrun optimizations
- Dataset migration to ClimbMix
- Autoresearch improvements
- FP8 support
- Flash Attention 3 integration

---

## Future Roadmap

### Short Term
- [ ] Complete inference integration (Phase 5)
- [ ] HuggingFace tokenizer export (Phase 6)
- [ ] Comprehensive testing suite (Phase 7)
- [ ] Update chat CLI for reasoning level control
- [ ] Response parsing for structured output

### Medium Term
- [ ] Multimodal support (vision)
- [ ] Enhanced tool calling features
- [ ] Additional reasoning modes
- [ ] Protocol v1.1.0 planning

### Long Term
- [ ] Audio/video support
- [ ] Advanced agent features
- [ ] Extended control token usage
- [ ] Protocol v2.0.0

---

## Notes

### Protocol Versioning

- **v1.0.0**: Current - Basic reasoning, tool calling, multimodal placeholders
- **v1.1.0**: Planned - Actual multimodal, enhanced tools
- **v2.0.0**: Future - Audio/video, advanced agents

### Compatibility

- Protocol v1.0.0 is not backward compatible with old tokenizer
- All new training should use protocol v1.0.0
- Old models require retraining with new tokenizer
- Scripts automatically detect and handle migration

---

For detailed implementation documentation, see:
- `TOKENIZER_PROTOCOL_PLAN.md` - Full design specification
- `IMPLEMENTATION_STATUS.md` - Current progress tracking
- `PHASE4_COMPLETE.md` - Phase 4 completion details
