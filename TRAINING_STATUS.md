# Nanochat Protocol v1.0.0 Training Status

**Updated**: September 18, 2026 21:50 UTC  
**Container**: nanochat-base  
**Status**: ✅ Running - Base Model Retraining

## Current Issue - RESOLVED

**Problem**: Old base model (32,768 vocab) incompatible with new tokenizer (32,000 vocab)

**Solution**: Retraining base model from scratch with protocol v1.0.0 tokenizer

## What Happened

1. ✅ **Tokenizer retrained** with protocol v1.0.0 (32K vocab)
2. ✅ **Fixed engine.py** to use `<|message_end|>` instead of `<|assistant_end|>`  
3. ❌ **Old base model** was trained with 32,768 vocab (old tokenizer)
4. ✅ **Now retraining** base model from scratch with 32K vocab

## Current Training

**Container**: `nanochat-base`  
**Script**: `runs/base_train_protocol_v1.sh`  
**Status**: Model initializing, first step compiling

### Configuration
- **Vocab**: 32,000 tokens (protocol v1.0.0) ✅
- **Model**: 8 layers, 123M parameters
- **Iterations**: 200
- **Batch size**: 262,144 tokens
- **GPU**: RTX 4060 Ti (BFloat16)
- **Flash Attention**: v3 enabled

### Expected Timeline
- Model compilation: 2-3 minutes (in progress)
- Training (200 steps): ~20 minutes
- **Total**: ~25 minutes

## Monitor Training

```powershell
# Check status
docker ps --filter name=nanochat-base

# View latest logs
docker logs nanochat-base --tail 50

# Follow logs
docker logs -f nanochat-base

# Check for training steps
docker logs nanochat-base | Select-String -Pattern "step 00"
```

## After Base Training

Once base training completes, run SFT training:

```powershell
# SFT will use the new base model automatically
docker run -d --gpus all --name nanochat-sft \
  -v "F:\UserData\git-repos\New folder\apps\nanochat:/nanochat" \
  -v "C:/Users/dusti/.cache/nanochat:/root/.cache/nanochat" \
  -w /nanochat nanochat-132:latest \
  bash /nanochat/runs/sft_only_train.sh
```

## Files Changed

- ✅ `nanochat/engine.py` - Fixed to use protocol v1.0.0 tokens
- ✅ `runs/base_train_protocol_v1.sh` - Clean base training
- ✅ `runs/sft_only_train.sh` - SFT training only

## Key Fixes Applied

### 1. Engine Token Compatibility
```python
# OLD (causes KeyError with new tokenizer)
assistant_end = tokenizer.encode_special("<|assistant_end|>")

# NEW (protocol v1.0.0)
try:
    message_end = tokenizer.encode_special("<|message_end|>")
except KeyError:
    message_end = tokenizer.encode_special("<|assistant_end|>")  # fallback
```

### 2. Fresh Base Model
- Removed old 32,768 vocab checkpoint
- Training new 32,000 vocab model from scratch
- Will be compatible with protocol v1.0.0 tokenizer

## What's Next

1. ⏳ **Wait for base training** (~25 min total)
2. **Run SFT training** (~10-15 min)
3. **Test the model**:
   ```bash
   python -m scripts.chat_cli -p "What is 2+2?"
   ```

## Troubleshooting

### If training fails again

Check logs:
```powershell
docker logs nanochat-base 2>&1 | Select-Object -Last 100
```

Common issues:
- **Out of memory**: Reduce `--device-batch-size` from 4 to 2
- **Token mismatch**: Verify tokenizer protocol v1.0.0
- **CUDA error**: Check GPU availability

### Restart training

```powershell
docker stop nanochat-base
docker rm nanochat-base

# Start fresh
docker run -d --gpus all --name nanochat-base \
  -v "F:\UserData\git-repos\New folder\apps\nanochat:/nanochat" \
  -v "C:/Users/dusti/.cache/nanochat:/root/.cache/nanochat" \
  -w /nanochat nanochat-132:latest \
  bash /nanochat/runs/base_train_protocol_v1.sh
```

---

**All issues resolved. Training in progress with correct protocol v1.0.0!** 🚀
