# NanoChat Docker Training Guide

Simple guide for training NanoChat using Docker.

## Prerequisites

- ✅ Docker installed (you have 29.8.0)
- ✅ NVIDIA GPU drivers
- ✅ NVIDIA Container Toolkit (for GPU support)

## Quick Start

### Option 1: Automated Training (Recommended)

Run the complete training pipeline automatically:

```bash
docker-compose up nanochat-train-auto
```

This will:
1. Train the tokenizer on mixed corpus (~30-45 min)
2. Download dataset shards (~10-20 min)
3. Run curriculum pretraining (~2-4 hours for depth=6)

**Everything runs automatically - just wait for it to complete!**

### Option 2: Interactive Mode

For more control, run commands manually inside the container:

```bash
# Start interactive container
docker-compose run --rm nanochat-train bash

# Inside container:
source .venv/bin/activate

# Train tokenizer (choose one)
python -m scripts.tok_train_curriculum --mix minimal    # Fast (~15 min)
python -m scripts.tok_train_curriculum --mix default    # Full (~45 min)

# Download datasets
python -m nanochat.data_registry --download climbmix -n 20 -w 4

# Start training
bash runs/curriculum_4060ti.sh

# After training, chat with your model
python -m scripts.chat_cli
```

## Building the Image

If you need to rebuild the Docker image:

```bash
docker-compose build nanochat-train
```

## Data Persistence

Your data is automatically saved to local directories:

- `./data` - Dataset cache and tokenizer
- `./checkpoints` - Trained model checkpoints
- `./config` - Configuration files
- `./runs` - Training scripts

These directories persist between container runs, so you won't lose your trained models.

## Configuration

### Environment Variables

Set these in `.env` file or docker-compose.yml:

```bash
NANOCHAT_BASE_DIR=/workspace/data      # Data directory
CUDA_VISIBLE_DEVICES=0                 # GPU selection
WANDB_API_KEY=your_key_here           # Optional: W&B logging
```

### Training Parameters

Edit `runs/curriculum_4060ti.sh` to adjust:

- `--depth` - Model size (6 for testing, 20+ for production)
- `--device-batch-size` - Reduce if you get OOM errors (try 16, 8, 4, 2)
- `--model-tag` - Name for your model checkpoint

### Curriculum Stages

Edit `config/pretraining_curriculum.yaml` to modify:

- Stage token ratios (how much data per stage)
- Context length schedules (2K→4K→8K→16K)
- Data source mixing weights

## Monitoring Training

### View Logs

```bash
# Follow logs in real-time
docker-compose logs -f nanochat-train-auto

# View logs for interactive session
docker logs -f nanochat-train
```

### Weights & Biases (Optional)

Enable W&B logging by setting environment variable in container:

```bash
export WANDB_RUN="my_experiment"  # Enable logging
export WANDB_RUN="dummy"          # Disable logging (default)
```

## After Training

### Evaluate Your Model

```bash
docker-compose run --rm nanochat-train bash -c "
source .venv/bin/activate &&
python -m scripts.base_eval --model-tag d6_curriculum_4060ti
"
```

### Chat with Your Model

```bash
docker-compose run --rm nanochat-train bash -c "
source .venv/bin/activate &&
python -m scripts.chat_cli
"
```

### Run Supervised Fine-Tuning

```bash
docker-compose run --rm nanochat-train bash -c "
source .venv/bin/activate &&
python -m scripts.sft_train_curriculum
"
```

## Troubleshooting

### GPU Not Available

Check if Docker can access your GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If this fails, install NVIDIA Container Toolkit:
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Out of Memory (OOM)

Edit `runs/curriculum_4060ti.sh` and reduce batch size:

```bash
--device-batch-size 24  # Try 16, 8, 4, or 2
```

Then rebuild and run:

```bash
docker-compose build nanochat-train
docker-compose up nanochat-train-auto
```

### Container Exits Immediately

Check logs for errors:

```bash
docker-compose logs nanochat-train-auto
```

### Dataset Download Fails

Increase workers in `runs/curriculum_4060ti.sh`:

```bash
python -m nanochat.data_registry --download climbmix -n 20 -w 8  # More workers
```

## Time Estimates

| Task | Duration | Notes |
|------|----------|-------|
| Image build | 5-10 min | One-time |
| Tokenizer (minimal) | 10-15 min | Fast setup |
| Tokenizer (full) | 30-45 min | Production quality |
| Dataset download | 10-20 min | Network dependent |
| Training (depth=6) | 2-4 hours | Test model |
| Training (depth=20) | ~2 hours | Requires 8x H100 |

## Hardware Requirements

### Minimum (depth=6 test model)
- GPU: NVIDIA RTX 4060Ti (16GB VRAM) or equivalent
- RAM: 32GB
- Storage: 50GB free

### Recommended (production models)
- GPU: 8x H100 or A100 (80GB each)
- RAM: 256GB+
- Storage: 500GB+ NVMe SSD

## Quick Command Reference

```bash
# Build image
docker-compose build nanochat-train

# Run automated training
docker-compose up nanochat-train-auto

# Interactive mode
docker-compose run --rm nanochat-train bash

# View logs
docker-compose logs -f nanochat-train-auto

# Stop training
docker-compose down

# Clean up (remove containers)
docker-compose down -v

# Check GPU availability
nvidia-smi
```

## File Structure

```
nanochat/
├── docker-compose.yml         ← Docker services configuration
├── Dockerfile                 ← Image definition
├── docker/
│   ├── entrypoint.sh         ← Container startup script
│   └── init_training.sh      ← Automated training pipeline
├── runs/
│   └── curriculum_4060ti.sh  ← Training script
├── config/
│   └── pretraining_curriculum.yaml  ← Curriculum configuration
├── data/                      ← Persistent data (created automatically)
└── checkpoints/              ← Persistent checkpoints (created automatically)
```

## Support

- **Project Documentation:** `README.md`
- **GitHub Discussions:** https://github.com/karpathy/nanochat/discussions
- **Discord:** #nanochat channel
- **DeepWiki:** https://deepwiki.com/karpathy/nanochat

## License

MIT - See LICENSE file for details
