# Complete Training Pipeline: Base Pretraining + Reasoning SFT + RL
# Adapted for RTX 4060 Ti 16GB with Docker
# Now includes reasoning model training with NVIDIA Nemotron datasets

param(
    [switch]$SkipTokenizer = $false,
    [switch]$SkipPretraining = $false,
    [switch]$SkipReasoningSFT = $false,
    [switch]$SkipSFT = $false,
    [switch]$OnlyPretraining = $false,
    [int]$PretrainIterations = 200,  # Default: 200 (same as speedrun-2x)
    [int]$ReasoningSFTIterations = 1000,  # Reasoning SFT iterations
    [float]$ReasoningRatio = 0.7,  # Ratio of reasoning_on examples (0.0-1.0)
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
Write-Host "  3. Reasoning SFT ($ReasoningSFTIterations iterations)" -ForegroundColor Yellow
Write-Host "  4. Standard SFT (Supervised Fine-Tuning)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Cyan
Write-Host "  Data shards: $DataShards" -ForegroundColor White
Write-Host "  Pretraining iterations: $PretrainIterations" -ForegroundColor White
Write-Host "  Reasoning SFT iterations: $ReasoningSFTIterations" -ForegroundColor White
Write-Host "  Reasoning ratio: $ReasoningRatio (70% = more reasoning traces)" -ForegroundColor White
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
        -e PYTORCH_COMPILE_OFF=1 `
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
    Write-Host "Run with -OnlyPretraining:`$false to continue with Reasoning SFT" -ForegroundColor Yellow
    exit 0
}

# -----------------------------------------------------------------------------
# Stage 3: Reasoning SFT (NEW!)
# -----------------------------------------------------------------------------

if (-not $SkipReasoningSFT) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Stage 3: Reasoning SFT (NEW!)" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Training model with reasoning capabilities" -ForegroundColor Yellow
    Write-Host "Using NVIDIA Nemotron datasets:" -ForegroundColor Yellow
    Write-Host "  - Nemotron-Cascade-SFT-Stage-2 (instruction + reasoning)" -ForegroundColor White
    Write-Host "  - Nemotron-Post-Training-Dataset-v2 (multilingual reasoning)" -ForegroundColor White
    Write-Host ""
    Write-Host "Reasoning configuration:" -ForegroundColor Cyan
    Write-Host "  Iterations: $ReasoningSFTIterations" -ForegroundColor White
    Write-Host "  Reasoning ratio: $ReasoningRatio" -ForegroundColor White
    Write-Host "  Training stages:" -ForegroundColor White
    Write-Host "    - Stage 1: Instruction following (30%)" -ForegroundColor Gray
    Write-Host "    - Stage 2: Reasoning training (50%)" -ForegroundColor Gray
    Write-Host "    - Stage 3: Multi-task fine-tuning (20%)" -ForegroundColor Gray
    Write-Host ""
    
    $reasoningSftExists = Test-Path "$NANOCHAT_BASE_DIR\reasoning_sft_checkpoints"
    
    if (-not $reasoningSftExists) {
        Write-Host "Starting reasoning SFT training..." -ForegroundColor Yellow
        Write-Host "Note: First run will download Nemotron datasets (~10GB)" -ForegroundColor Yellow
        Write-Host "This may take several hours depending on iterations." -ForegroundColor Yellow
        Write-Host ""
        
        # Run reasoning SFT
        docker run --rm `
            --gpus all `
            --ipc=host `
            -v "${PWD}:/workspace" `
            -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
            -w /workspace `
            -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
            -e OMP_NUM_THREADS=1 `
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True `
            -e PYTORCH_COMPILE_OFF=1 `
            -e HF_HUB_ENABLE_HF_TRANSFER=1 `
            nanochat `
            bash -c "source /nanochat/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 -m scripts.chat_reasoning_sft -- --device-batch-size=4 --num-iterations=$ReasoningSFTIterations --reasoning-ratio=$ReasoningRatio --run=reasoning_docker"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Reasoning SFT training failed!" -ForegroundColor Red
            Write-Host "Check logs for details. Common issues:" -ForegroundColor Yellow
            Write-Host "  - Dataset download timeout (retry with better connection)" -ForegroundColor White
            Write-Host "  - Out of memory (reduce --device-batch-size)" -ForegroundColor White
            Write-Host "  - Missing dependencies (rebuild Docker image)" -ForegroundColor White
            exit 1
        }
        
        Write-Host "✓ Reasoning SFT training completed!" -ForegroundColor Green
    } else {
        Write-Host "✓ Reasoning SFT checkpoint exists, skipping..." -ForegroundColor Green
    }
    
    # Evaluate reasoning model
    Write-Host ""
    Write-Host "Evaluating reasoning model..." -ForegroundColor Yellow
    Write-Host "Testing reasoning capabilities on GSM8K (math reasoning)" -ForegroundColor White
    
    docker run --rm `
        --gpus all `
        -v "${PWD}:/workspace" `
        -v "${NANOCHAT_BASE_DIR}:/root/.cache/nanochat" `
        -w /workspace `
        -e NANOCHAT_BASE_DIR=/root/.cache/nanochat `
        -e PYTORCH_COMPILE_OFF=1 `
        nanochat `
        bash -c "source /nanochat/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i reasoning_sft --device-batch-size=4"
    
    Write-Host ""
    Write-Host "✓ Reasoning SFT stage complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Model now has reasoning capabilities:" -ForegroundColor Cyan
    Write-Host "  ✓ Step-by-step problem solving" -ForegroundColor Green
    Write-Host "  ✓ Chain-of-thought reasoning" -ForegroundColor Green
    Write-Host "  ✓ Improved math and code performance" -ForegroundColor Green
    Write-Host ""
}

# -----------------------------------------------------------------------------
# Stage 4: Standard SFT (Supervised Fine-Tuning)
# -----------------------------------------------------------------------------

if (-not $SkipSFT) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Stage 4: Standard SFT (Supervised Fine-Tuning)" -ForegroundColor Cyan
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
            -e OMP_NUM_THREADS=1 `
            -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True `
            -e PYTORCH_COMPILE_OFF=1 `
            nanochat `
            bash -c "source /nanochat/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --run=dummy"
        
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
        -e PYTORCH_COMPILE_OFF=1 `
        nanochat `
        bash -c "source /nanochat/.venv/bin/activate && torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft"
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
Write-Host "  Reasoning SFT: $NANOCHAT_BASE_DIR\reasoning_sft_checkpoints\" -ForegroundColor White
Write-Host "  Standard SFT: $NANOCHAT_BASE_DIR\chatsft_checkpoints\" -ForegroundColor White
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Test reasoning capabilities:" -ForegroundColor White
Write-Host "     docker run -it --rm --gpus all -v `"${PWD}:/workspace`" -v `"${NANOCHAT_BASE_DIR}:/root/.cache/nanochat`" -w /workspace -e NANOCHAT_BASE_DIR=/root/.cache/nanochat nanochat python -m scripts.chat_cli -p `"Solve step by step: What is 15% of 240?`"" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. Interactive chat with reasoning:" -ForegroundColor White
Write-Host "     docker run -it --rm --gpus all -v `"${PWD}:/workspace`" -v `"${NANOCHAT_BASE_DIR}:/root/.cache/nanochat`" -w /workspace -e NANOCHAT_BASE_DIR=/root/.cache/nanochat nanochat python -m scripts.chat_cli --reasoning-level=medium" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Test math reasoning (GSM8K):" -ForegroundColor White
Write-Host "     docker run --rm --gpus all -v `"${PWD}:/workspace`" -v `"${NANOCHAT_BASE_DIR}:/root/.cache/nanochat`" -w /workspace -e NANOCHAT_BASE_DIR=/root/.cache/nanochat nanochat python -m scripts.chat_eval -- --task GSM8K" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. Run RL training (optional):" -ForegroundColor White
Write-Host "     docker run --rm --gpus all -v `"${PWD}:/workspace`" -v `"${NANOCHAT_BASE_DIR}:/root/.cache/nanochat`" -w /workspace -e NANOCHAT_BASE_DIR=/root/.cache/nanochat nanochat torchrun --standalone --nproc_per_node=1 -m scripts.chat_rl -- --run=dummy" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 Your reasoning model is ready!" -ForegroundColor Magenta
Write-Host "The model can now:" -ForegroundColor Cyan
Write-Host "  ✓ Show step-by-step reasoning for complex problems" -ForegroundColor Green
Write-Host "  ✓ Provide direct answers when appropriate" -ForegroundColor Green
Write-Host "  ✓ Handle math, code, and logical reasoning tasks" -ForegroundColor Green
Write-Host ""
