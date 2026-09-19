# Complete Training Pipeline: Base Pretraining + SFT + RL
# Adapted for RTX 4060 Ti 16GB with Docker
# Based on speedrun-4060ti-2x but configurable

param(
    [switch]$SkipTokenizer = $false,
    [switch]$SkipPretraining = $false,
    [switch]$SkipSFT = $false,
    [switch]$OnlyPretraining = $false,
    [int]$PretrainIterations = 200,  # Default: 200 (same as speedrun-2x)
    [int]$DataShards = 20  # Number of data shards to download
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Complete NanoChat Training Pipeline" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Pipeline stages:" -ForegroundColor Yellow
Write-Host "  1. Tokenizer training (if needed)" -ForegroundColor Yellow
Write-Host "  2. Base pretraining ($PretrainIterations iterations)" -ForegroundColor Yellow
Write-Host "  3. SFT (Supervised Fine-Tuning)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Data shards: $DataShards" -ForegroundColor White
Write-Host "  Pretraining iterations: $PretrainIterations" -ForegroundColor White
Write-Host ""

# Build Docker image if needed
$imageExists = docker images | Select-String "nanochat"
if (-not $imageExists) {
    Write-Host "Building Docker image..." -ForegroundColor Yellow
    docker build -t nanochat -f docker/Dockerfile .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Docker build failed!" -ForegroundColor Red
        exit 1
    }
}

# Create cache directory
$NANOCHAT_BASE_DIR = "$env:USERPROFILE\.cache\nanochat"
New-Item -ItemType Directory -Force -Path $NANOCHAT_BASE_DIR | Out-Null

# -----------------------------------------------------------------------------
# Stage 1: Tokenizer Training
# -----------------------------------------------------------------------------

if (-not $SkipTokenizer) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Stage 1: Tokenizer Training" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    
    $tokenizerExists = Test-Path "$NANOCHAT_BASE_DIR\tokenizer\tokenizer.pkl"
    
    if (-not $tokenizerExists) {
        Write-Host "Training tokenizer on Nemotron data..." -ForegroundColor Yellow
        Write-Host "Note: This will download a sample of Nemotron datasets" -ForegroundColor Yellow
        
        # Train tokenizer on Nemotron data
        # We'll use the standard ClimbMix approach but you could adapt this to use Nemotron
        # For now, download a small amount of ClimbMix data for tokenizer training
        Write-Host "Downloading sample data for tokenizer..." -ForegroundColor Yellow
        docker run --rm `
            -v "${PWD}:/workspace" `
            -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
            -w /workspace `
            -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
            nanochat `
            python -m nanochat.dataset -n 1
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Data download failed!" -ForegroundColor Red
            exit 1
        }
        
        # Train tokenizer
        Write-Host "Training tokenizer..." -ForegroundColor Yellow
        docker run --rm `
            -v "${PWD}:/workspace" `
            -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
            -w /workspace `
            -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
            nanochat `
            python -m scripts.tok_train --max-chars=250000000
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Tokenizer training failed!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "✓ Tokenizer trained successfully" -ForegroundColor Green
    } else {
        Write-Host "✓ Tokenizer already exists, skipping..." -ForegroundColor Green
    }
    
    # Evaluate tokenizer
    Write-Host "Evaluating tokenizer..." -ForegroundColor Yellow
    docker run --rm `
        -v "${PWD}:/workspace" `
        -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
        -w /workspace `
        -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
        nanochat `
        python -m scripts.tok_eval
}

# -----------------------------------------------------------------------------
# Stage 2: Base Pretraining
# -----------------------------------------------------------------------------

if (-not $SkipPretraining) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Stage 2: Nemotron Base Pretraining" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Iterations: $PretrainIterations" -ForegroundColor Cyan
    Write-Host "Datasets: Nemotron Code-Concepts + Scientific-Coding" -ForegroundColor Cyan
    Write-Host "Model: depth=6, aspect_ratio=64 (fast training)" -ForegroundColor Cyan
    Write-Host ""
    
    $checkpointExists = Test-Path "$NANOCHAT_BASE_DIR\base_checkpoints"
    
    if (-not $checkpointExists) {
        Write-Host "Starting Nemotron base pretraining..." -ForegroundColor Yellow
        Write-Host "Note: First run will download Nemotron datasets from HuggingFace" -ForegroundColor Yellow
        Write-Host "This will take several hours to days depending on iterations." -ForegroundColor Yellow
        Write-Host ""
        
        # Run base training using Nemotron datasets
        docker run --rm `
            --gpus all `
            --ipc=host `
            -v "${PWD}:/workspace" `
            -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
            -w /workspace `
            -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
            -e NEMOTRON_STAGE=stage1 `
            -e OMP_NUM_THREADS=1 `
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True `
            -e PYTORCH_COMPILE_OFF=1 `
            -e HF_HUB_ENABLE_HF_TRANSFER=1 `
            nanochat `
            bash -c "source /nanochat/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 -m scripts.nemotron_base_train -- --depth=6 --target-param-data-ratio=8 --device-batch-size=4 --run=dummy --num-iterations=$PretrainIterations"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Nemotron pretraining failed!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "✓ Nemotron pretraining completed!" -ForegroundColor Green
    } else {
        Write-Host "✓ Pretraining checkpoint exists, skipping..." -ForegroundColor Green
    }
    
    # Evaluate base model
    Write-Host ""
    Write-Host "Evaluating pretrained model..." -ForegroundColor Yellow
    docker run --rm `
        --gpus all `
        -v "${PWD}:/workspace" `
        -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
        -w /workspace `
        -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
        nanochat `
        bash -c "source /nanochat/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=4"
}

if ($OnlyPretraining) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "Pretraining Complete!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "Checkpoint: checkpoints/stage1/final.pt" -ForegroundColor Green
    Write-Host ""
    Write-Host "Run with -OnlyPretraining:`$false to continue with SFT" -ForegroundColor Yellow
    exit 0
}

# -----------------------------------------------------------------------------
# Stage 3: SFT (Supervised Fine-Tuning)
# -----------------------------------------------------------------------------

if (-not $SkipSFT) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Stage 3: Supervised Fine-Tuning (SFT)" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    
    $sftExists = Test-Path "$NANOCHAT_BASE_DIR\chatsft_checkpoints"
    
    if (-not $sftExists) {
        Write-Host "Starting SFT training..." -ForegroundColor Yellow
        Write-Host "Teaching model conversation, tool use, and multiple choice..." -ForegroundColor Yellow
        Write-Host ""
        
        # Run SFT
        docker run --rm `
            --gpus all `
            --ipc=host `
            -v "${PWD}:/workspace" `
            -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
            -w /workspace `
            -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
            nanochat `
            bash -c "torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --run=dummy --load-from=checkpoints/stage1/final.pt"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "SFT training failed!" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "✓ SFT training completed!" -ForegroundColor Green
    } else {
        Write-Host "✓ SFT checkpoint exists, skipping..." -ForegroundColor Green
    }
    
    # Evaluate SFT model
    Write-Host ""
    Write-Host "Evaluating SFT model..." -ForegroundColor Yellow
    docker run --rm `
        --gpus all `
        -v "${PWD}:/workspace" `
        -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
        -w /workspace `
        -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
        nanochat `
        bash -c "torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft"
}

# -----------------------------------------------------------------------------
# Complete!
# -----------------------------------------------------------------------------

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Checkpoints:" -ForegroundColor Green
Write-Host "  Pretrained: checkpoints/stage1/final.pt" -ForegroundColor White
Write-Host "  SFT: $NANOCHAT_BASE_DIR\chatsft_checkpoints\" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Test the model:" -ForegroundColor White
Write-Host "     docker run -it --rm --gpus all -v `"${PWD}:/workspace`" -v `"${NANOCHAT_BASE_DIR}:/root/.cache/nanochat`" -w /workspace -e NANOCHAT_BASE_DIR=/root/.cache/nanochat nanochat python -m scripts.chat_cli -p `"Why is the sky blue?`"" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Interactive chat:" -ForegroundColor White
Write-Host "     docker run -it --rm --gpus all -v `"${PWD}:/workspace`" -v `"${NANOCHAT_BASE_DIR}:/root/.cache/nanochat`" -w /workspace -e NANOCHAT_BASE_DIR=/root/.cache/nanochat nanochat python -m scripts.chat_cli" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Run RL training (optional):" -ForegroundColor White
Write-Host "     docker run --rm --gpus all -v `"${PWD}:/workspace`" -v `"${NANOCHAT_BASE_DIR}:/root/.cache/nanochat`" -w /workspace -e NANOCHAT_BASE_DIR=/root/.cache/nanochat nanochat torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl -- --run=dummy" -ForegroundColor Gray
Write-Host ""
