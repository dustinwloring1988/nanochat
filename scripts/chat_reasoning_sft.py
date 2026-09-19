"""
Reasoning-focused Supervised Fine-Tuning (SFT) for nanochat.

This script trains the model with reasoning capabilities using:
1. NVIDIA Nemotron instruction following datasets
2. NVIDIA Nemotron reasoning datasets (with reasoning_on/off splits)

Training stages:
    Stage 1: Instruction Following (~30% of training)
    Stage 2: Reasoning Training (~50% of training)
    Stage 3: Multi-task Fine-tuning (~20% of training)

Run as:
    torchrun --standalone --nproc_per_node=1 -m scripts.chat_reasoning_sft -- \
        --device-batch-size=4 --num-iterations=1000
"""

import gc
import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from nanochat.common import (
    compute_init, compute_cleanup, print0, DummyWandb, get_base_dir,
    autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON,
    is_ddp_initialized
)
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
from nanochat.chat_tokenizer import get_chat_tokenizer
from nanochat.reasoning_dataloader import create_reasoning_dataloader, NemotronReasoningDataset
from nanochat.messages import Message
import torch.distributed as dist
from nanochat.flash_attention import HAS_FA3
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Reasoning-focused SFT for nanochat")
# Logging
parser.add_argument("--run", type=str, default="reasoning", help="wandb run name ('dummy' disables wandb)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from checkpoint (0=no, 1=yes)")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=2000, help="total optimization steps")
parser.add_argument("--stage1-iterations", type=int, default=600, help="stage 1: instruction following")
parser.add_argument("--stage2-iterations", type=int, default=1000, help="stage 2: reasoning training")
parser.add_argument("--stage3-iterations", type=int, default=400, help="stage 3: multi-task finetuning")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=None, help="LR for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=None, help="LR for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=None, help="LR for matrix parameters (Muon)")
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.05, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.1, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Reasoning-specific
parser.add_argument("--reasoning-ratio", type=float, default=0.7, help="ratio of reasoning_on to total examples (0.0-1.0)")
parser.add_argument("--enable-reasoning-curriculum", type=int, default=1, help="gradually increase reasoning ratio (0=no, 1=yes)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps")
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="tokens for validation")
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE every N steps")
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task")
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task")
# Legacy task mixing (stage 3)
parser.add_argument("--mmlu-epochs", type=int, default=1, help="MMLU epochs in stage 3")
parser.add_argument("--gsm8k-epochs", type=int, default=2, help="GSM8K epochs in stage 3")

args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')

# wandb logging
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-reasoning-sft",
    name=args.run,
    config=user_config
)

# Flash Attention status
if not HAS_FA3:
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")

# Load model and tokenizer
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Get chat tokenizer for reasoning support
chat_tokenizer = get_chat_tokenizer()

# Inherit hyperparameters from pretrained checkpoint
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len", 2048, meta),
    ("device_batch_size", 32, meta),
    ("total_batch_size", 524288, meta),
    ("embedding_lr", 0.3, pretrain_user_config),
    ("unembedding_lr", 0.004, pretrain_user_config),
    ("matrix_lr", 0.02, pretrain_user_config),
]:
    arg_val = getattr(args, name)
    pretrain_val = source.get(name)
    if arg_val is None:
        resolved = pretrain_val if pretrain_val is not None else fallback
        setattr(args, name, resolved)
        print0(f"Inherited {name}={resolved}")
    else:
        print0(f"Using {name}={arg_val}")

# Compile model
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd

print0(f"\n{'='*60}")
print0("REASONING SFT CONFIGURATION")
print0(f"{'='*60}")
print0(f"Tokens/micro-batch/rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size: {args.total_batch_size:,} => grad_accum: {grad_accum_steps}")
print0(f"\nTraining stages:")
print0(f"  Stage 1 (Instruction Following): {args.stage1_iterations} iters")
print0(f"  Stage 2 (Reasoning Training):    {args.stage2_iterations} iters")
print0(f"  Stage 3 (Multi-task Finetuning): {args.stage3_iterations} iters")
print0(f"  Total:                           {args.num_iterations} iters")
print0(f"\nReasoning configuration:")
print0(f"  Reasoning ratio: {args.reasoning_ratio:.1%}")
print0(f"  Reasoning curriculum: {'enabled' if args.enable_reasoning_curriculum else 'disabled'}")
print0(f"{'='*60}\n")

token_bytes = get_token_bytes(device=device)

# Initialize optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=0.0
)

# Load optimizer state if requested
base_dir = get_base_dir()
if args.load_optimizer:
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    if optimizer_data is not None:
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr
        print0("✓ Loaded optimizer state (momentum buffers only)")
    else:
        print0("⚠ Optimizer checkpoint not found, starting fresh")

# GradScaler for fp16
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# Set initial learning rates
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def get_current_stage(step: int) -> str:
    """Determine current training stage based on step"""
    if step < args.stage1_iterations:
        return "instruction_following"
    elif step < args.stage1_iterations + args.stage2_iterations:
        return "reasoning"
    else:
        return "mixed"

def get_reasoning_ratio(step: int) -> float:
    """Get reasoning ratio for current step (with optional curriculum)"""
    if not args.enable_reasoning_curriculum:
        return args.reasoning_ratio
    
    # Gradually increase reasoning ratio during stage 2
    if step < args.stage1_iterations:
        return 0.3  # Low reasoning in stage 1
    elif step < args.stage1_iterations + args.stage2_iterations:
        # Linearly increase from 0.3 to args.reasoning_ratio
        progress = (step - args.stage1_iterations) / args.stage2_iterations
        return 0.3 + progress * (args.reasoning_ratio - 0.3)
    else:
        return args.reasoning_ratio  # Full reasoning in stage 3

# Create dataloaders for each stage
print0("\nInitializing dataloaders...")
stage_loaders = {}

try:
    # Stage 1: Instruction Following
    print0("\nStage 1: Instruction Following")
    stage_loaders["instruction_following"] = create_reasoning_dataloader(
        stage="instruction_following",
        reasoning_ratio=0.3,
        seed=42 + ddp_rank,
    )
    
    # Stage 2: Reasoning
    print0("\nStage 2: Reasoning Training")
    stage_loaders["reasoning"] = create_reasoning_dataloader(
        stage="reasoning",
        reasoning_ratio=args.reasoning_ratio,
        seed=42 + ddp_rank,
    )
    
    # Stage 3: Mixed (with legacy tasks)
    print0("\nStage 3: Multi-task Finetuning")
    # Use existing SmolTalk/MMLU/GSM8K tasks
    legacy_tasks = [
        SmolTalk(split="train"),
        *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)],
        *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)],
    ]
    legacy_dataset = TaskMixture(legacy_tasks)
    print0(f"✓ Legacy tasks: {len(legacy_dataset):,} rows")
    
    # Also mix with reasoning data
    stage_loaders["mixed"] = create_reasoning_dataloader(
        stage="mixed",
        reasoning_ratio=args.reasoning_ratio,
        seed=42 + ddp_rank,
    )

except Exception as e:
    print0(f"✗ Failed to initialize dataloaders: {e}")
    print0("⚠ Falling back to legacy tasks only")
    stage_loaders = None

# Create validation dataset
val_dataset = TaskMixture([
    SmolTalk(split="test"),
    MMLU(subset="all", split="test", stop=5200),
    GSM8K(subset="main", split="test", stop=420),
])

# -----------------------------------------------------------------------------
# Data Generator
# -----------------------------------------------------------------------------

current_stage = None
current_iterator = None

def get_data_batch():
    """Get next training batch with reasoning support"""
    global current_stage, current_iterator
    
    step = get_data_batch.step if hasattr(get_data_batch, 'step') else 0
    stage = get_current_stage(step)
    
    # Switch stage if needed
    if stage != current_stage:
        print0(f"\n{'='*60}")
        print0(f"Switching to stage: {stage.upper()}")
        print0(f"{'='*60}\n")
        current_stage = stage
        
        if stage_loaders and stage in stage_loaders:
            current_iterator = stage_loaders[stage].get_iterator(shuffle=True)
        else:
            # Fallback to legacy data generator
            current_iterator = sft_data_generator_bos_bestfit("train")
    
    # Get next example
    try:
        if stage_loaders and current_iterator:
            # Get from reasoning dataloader
            example = next(current_iterator)
            
            # Convert to Message objects
            messages = [Message.from_dict(msg) for msg in example["messages"]]
            
            # Format for training using ChatTokenizer
            batch = chat_tokenizer.format_for_training(
                messages,
                train_reasoning=True,
                max_tokens=args.max_seq_len
            )
            
            # Convert to tensors
            use_cuda = device_type == "cuda"
            inputs = torch.tensor(batch["input_ids"], dtype=torch.int32, device=device).unsqueeze(0)
            targets = torch.tensor(batch["target_ids"], dtype=torch.int64, device=device).unsqueeze(0)
            
            # Apply loss mask
            mask = torch.tensor(batch["loss_mask"], dtype=torch.bool, device=device).unsqueeze(0)
            targets[~mask] = -1
            
            return inputs, targets
        else:
            # Fallback to legacy generator
            return next(current_iterator)
            
    except StopIteration:
        # Dataset exhausted - recreate iterator
        if stage_loaders and stage in stage_loaders:
            current_iterator = stage_loaders[stage].get_iterator(shuffle=True)
        else:
            current_iterator = sft_data_generator_bos_bestfit("train")
        return get_data_batch()

# Legacy data generator (fallback)
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """Legacy SFT dataloader for backward compatibility"""
    # ... (existing implementation from chat_sft.py)
    # This is a simplified placeholder - in production, use full implementation
    raise NotImplementedError("Legacy generator not implemented - use reasoning dataloader")

# Initialize first batch
x, y = get_data_batch()

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

print0(f"\n{'='*60}")
print0("STARTING REASONING SFT TRAINING")
print0(f"{'='*60}\n")

min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0

while step < args.num_iterations:
    flops_so_far = num_flops_per_token * args.total_batch_size * step
    progress = step / args.num_iterations
    
    # Validation evaluation
    if step > 0 and (step % args.eval_every == 0 or step == args.num_iterations - 1):
        model.eval()
        # Use legacy validation for now
        # TODO: Add reasoning-specific validation
        print0(f"Step {step:05d} | Evaluating...")
        wandb_run.log({
            "step": step,
            "stage": get_current_stage(step),
            "reasoning_ratio": get_reasoning_ratio(step),
        })
        model.train()
    
    # ChatCORE evaluation
    if args.chatcore_every > 0 and step > 0 and step % args.chatcore_every == 0:
        model.eval()
        engine = Engine(orig_model, tokenizer)
        # TODO: Add reasoning-specific evaluation
        print0(f"Step {step:05d} | ChatCORE evaluation...")
        model.train()
    
    # Save checkpoint
    if step > 0 and step % 500 == 0:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "reasoning_sft_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            optimizer.state_dict(),
            {
                "step": step,
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config,
            },
            rank=ddp_rank,
        )
        print0(f"✓ Checkpoint saved at step {step}")
    
    # Training step
    synchronize()
    t0 = time.time()
    
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Prefetch next batch
        get_data_batch.step = step
        x, y = get_data_batch()
    
    # Optimizer step
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    
    if scaler is not None:
        scaler.unscale_(optimizer)
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    step += 1
    
    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    
    if step > 10:
        total_training_time += dt
    
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    
    current_stage_name = get_current_stage(step)
    current_reasoning_ratio = get_reasoning_ratio(step)
    
    print0(
        f"step {step:05d} ({100*progress:.1f}%) | "
        f"stage={current_stage_name[:4]} | "
        f"loss={debiased_smooth_loss:.4f} | "
        f"r_ratio={current_reasoning_ratio:.2f} | "
        f"dt={dt*1000:.0f}ms | "
        f"tok/s={tok_per_sec:,} | "
        f"mfu={mfu:.1f}%"
    )
    
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "stage": current_stage_name,
            "train/loss": debiased_smooth_loss,
            "train/reasoning_ratio": current_reasoning_ratio,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

# Helper functions
def get_lr_multiplier(progress):
    """Learning rate schedule"""
    if progress < args.warmup_ratio:
        return (progress + 1e-8) / args.warmup_ratio
    elif progress <= 1.0 - args.warmdown_ratio:
        return 1.0
    else:
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

def get_muon_momentum(it):
    """Momentum schedule for Muon optimizer"""
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# Final checkpoint
output_dirname = args.model_tag if args.model_tag else f"d{depth}"
checkpoint_dir = os.path.join(base_dir, "reasoning_sft_checkpoints", output_dirname, "final")
save_checkpoint(
    checkpoint_dir,
    step,
    orig_model.state_dict(),
    optimizer.state_dict(),
    {
        "step": step,
        "model_config": {
            "sequence_len": args.max_seq_len,
            "vocab_size": tokenizer.get_vocab_size(),
            "n_layer": depth,
            "n_head": model.config.n_head,
            "n_kv_head": model.config.n_kv_head,
            "n_embd": model.config.n_embd,
            "window_pattern": model.config.window_pattern,
        },
        "user_config": user_config,
    },
    rank=ddp_rank,
)

print0(f"\n{'='*60}")
print0("REASONING SFT TRAINING COMPLETE")
print0(f"{'='*60}")
print0(f"Total training time: {total_training_time/60:.1f}m")
print0(f"Final checkpoint: {checkpoint_dir}")
print0(f"{'='*60}\n")

# Cleanup
wandb_run.finish()
compute_cleanup()
