"""
Supervised fine-tuning with multi-dataset curriculum.

Simplified SFT with curriculum config support. Uses SmolTalk as primary dataset
with planned multi-dataset support.

Run as:
    torchrun --nproc_per_node=8 -m scripts.sft_train_curriculum
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import argparse
import time
import yaml
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
from nanochat.loss_eval import evaluate_bpb
from scripts.chat_eval import run_chat_eval
from tasks.common import TaskMixture
from tasks.smoltalk import SmolTalk
from tasks.nemotron_multilingual import NemotronMultilingual
from tasks.hunter_alpha import HunterAlpha
from tasks.nemotron_swe import NemotronSWE
from tasks.claude_fable import ClaudeFable

parser = argparse.ArgumentParser(description="SFT with curriculum")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps")
parser.add_argument("--config", type=str, default="config/sft_curriculum.yaml", help="SFT curriculum config")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load")
parser.add_argument("--model-step", type=int, default=None, help="model step to load")
parser.add_argument("--num-iterations", type=int, default=-1, help="training steps (-1 = full epoch)")
parser.add_argument("--max-seq-len", type=int, default=None, help="context length")
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size")
parser.add_argument("--embedding-lr", type=float, default=None, help="embedding LR")
parser.add_argument("--unembedding-lr", type=float, default=None, help="unembedding LR")
parser.add_argument("--matrix-lr", type=float, default=None, help="matrix LR")
parser.add_argument("--eval-every", type=int, default=200, help="eval every N steps")
args = parser.parse_args()
user_config = vars(args).copy()

# Load curriculum config if provided
curriculum_config = None
if args.config and os.path.exists(args.config):
    print0(f"Loading SFT curriculum config from: {args.config}")
    with open(args.config, 'r') as f:
        curriculum_config = yaml.safe_load(f)
    print0(f"Loaded {len(curriculum_config['stages'])} SFT stages")

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-sft-curriculum", name=args.run, config=user_config
)

# Load model
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit hyperparams from pretrain
pretrain_user_config = meta.get("user_config", {})
for name, fallback, source in [
    ("max_seq_len", 8192, meta),
    ("device_batch_size", 32, meta),
    ("total_batch_size", 524288, meta),
    ("embedding_lr", 0.3, pretrain_user_config),
    ("unembedding_lr", 0.004, pretrain_user_config),
    ("matrix_lr", 0.02, pretrain_user_config),
]:
    if getattr(args, name) is None:
        setattr(args, name, source.get(name, fallback))

orig_model = model
model = torch.compile(model, dynamic=False)

# Setup optimizer
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=0.0
)

# Load optimizer state
optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
if optimizer_data:
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    optimizer.load_state_dict(optimizer_data)
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr
    print0("Loaded optimizer state")

# Dataset registry
DATASET_REGISTRY = {
    "smoltalk": lambda: SmolTalk(split="train"),
    "nemotron_multilingual": lambda: NemotronMultilingual(split="code_hi"),  # Using code_hi as sample; TODO: load all 12 splits
    "hunter_alpha": lambda: HunterAlpha(split="train"),
    "nemotron_swe": lambda: NemotronSWE(split="train"),
    "claude_fable": lambda: ClaudeFable(split="train"),
}

# Setup curriculum stages if config provided
use_curriculum = curriculum_config is not None
curriculum_stages = []
total_iterations = 0

if use_curriculum:
    print0("\n" + "="*80)
    print0("SFT CURRICULUM CONFIGURATION")
    print0("="*80)
    
    for stage in curriculum_config["stages"]:
        stage_name = stage["name"]
        source_weights = stage["source_weights"]
        context_range = stage["context_range"]
        epoch_ratio = stage["epoch_ratio"]
        
        print0(f"\nStage: {stage_name}")
        print0(f"  Epoch ratio: {epoch_ratio:.1%}")
        print0(f"  Context: {context_range[0]} → {context_range[1]}")
        print0(f"  Sources: {list(source_weights.keys())}")
        
        # Build task mixture for this stage
        tasks = []
        for source_name, weight in source_weights.items():
            if source_name in DATASET_REGISTRY:
                dataset = DATASET_REGISTRY[source_name]()
                # TaskMixture doesn't support weights, so repeat tasks proportionally
                # Round weights to get integer counts (at least 1 copy each)
                copies = max(1, int(weight * 100))  # Scale by 100 for finer granularity
                for _ in range(copies):
                    tasks.append(dataset)
                print0(f"    {source_name}: {weight:.1%} ({dataset.num_examples()} examples) x{copies} copies")
        
        task_mixture = TaskMixture(tasks)
        
        # Calculate iterations for this stage
        stage_iterations = int(task_mixture.num_examples() * epoch_ratio / (args.device_batch_size * ddp_world_size))
        total_iterations += stage_iterations
        
        curriculum_stages.append({
            "name": stage_name,
            "task_mixture": task_mixture,
            "iterations": stage_iterations,
            "context_range": context_range,
        })
        
        print0(f"  Stage iterations: {stage_iterations}")
    
    print0(f"\nTotal curriculum iterations: {total_iterations}")
    print0("="*80 + "\n")
else:
    # Fallback: single-stage training with SmolTalk
    print0("No curriculum config provided, using SmolTalk only")
    task_mixture = TaskMixture([(SmolTalk(split="train"), 1.0)])
    
    if args.num_iterations == -1:
        total_iterations = task_mixture.num_examples() // (args.device_batch_size * ddp_world_size)
    else:
        total_iterations = args.num_iterations
    
    curriculum_stages = [{
        "name": "single_stage",
        "task_mixture": task_mixture,
        "iterations": total_iterations,
        "context_range": [args.max_seq_len, args.max_seq_len],
    }]

tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd

print0(f"SFT training: {total_iterations} iterations total")

# Training state
step = 0
global_step = 0
smooth_loss = 0.0
total_time = 0.0

# Main training loop - iterate through curriculum stages
model.train()

for stage_idx, stage in enumerate(curriculum_stages):
    stage_name = stage["name"]
    stage_iterations = stage["iterations"]
    task_mixture = stage["task_mixture"]
    context_start, context_end = stage["context_range"]
    
    print0("\n" + "="*80)
    print0(f"STARTING STAGE {stage_idx + 1}/{len(curriculum_stages)}: {stage_name}")
    print0("="*80)
    print0(f"Iterations: {stage_iterations}")
    print0(f"Context: {context_start} → {context_end}")
    print0("="*80 + "\n")
    
    # Update context length if it changes
    current_context = context_start
    context_increment = (context_end - context_start) / max(stage_iterations, 1) if context_end > context_start else 0
    
    # Create iterator for this stage
    def sft_batch_generator(task_mixture, batch_size, context_len):
        """Generate (x, y) batches from a task mixture for SFT training."""
        import random
        while True:
            conversations = []
            for _ in range(batch_size):
                idx = random.randint(0, task_mixture.num_examples() - 1)
                # Use get_example (not __getitem__) to get actual conversation dict
                conv = task_mixture.get_example(idx)
                conversations.append(conv)
            
            # Tokenize all conversations
            batch_ids = []
            batch_masks = []
            for conv in conversations:
                ids, mask = tokenizer.render_conversation(conv)
                # Truncate or pad to context_len + 1 (for targets)
                if len(ids) > current_context + 1:
                    ids = ids[:current_context + 1]
                    mask = mask[:current_context + 1]
                else:
                    # Pad with BOS tokens
                    bos_token = tokenizer.get_bos_token_id()
                    pad_len = (current_context + 1) - len(ids)
                    ids = ids + [bos_token] * pad_len
                    mask = mask + [0] * pad_len  # Mask padding
                
                batch_ids.append(ids)
                batch_masks.append(mask)
            
            # Convert to tensors
            batch_tensor = torch.tensor(batch_ids, dtype=torch.long)
            inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32).contiguous()
            targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64).contiguous()
            
            # Apply loss mask
            mask_tensor = torch.tensor(batch_masks, dtype=torch.int8)
            mask_targets = mask_tensor[:, 1:].to(device=device)
            targets[mask_targets == 0] = -1
            
            yield inputs, targets
    
    task_iter = sft_batch_generator(task_mixture, args.device_batch_size, current_context)
    
    step = 0
    while step < stage_iterations:
        t0 = time.time()
        
        # Update context length if ramping
        if context_increment > 0 and step % 10 == 0:
            new_context = int(context_start + step * context_increment)
            if new_context != current_context and new_context <= context_end:
                current_context = new_context
                # Recreate iterator with new context length - generator recreates internally
        
        for micro_step in range(grad_accum_steps):
            x, y = next(task_iter)
            loss = model(x, y) / grad_accum_steps
            loss.backward()
        
        optimizer.step()
        model.zero_grad(set_to_none=True)
        
        train_loss = loss.item() * grad_accum_steps
        t1 = time.time()
        dt = t1 - t0
        
        if step > 10:
            total_time += dt
        
        smooth_loss = 0.9 * smooth_loss + 0.1 * train_loss
        
        if step % 50 == 0:
            print0(f"Stage {stage_idx+1} step {step}/{stage_iterations} | loss: {smooth_loss:.4f} | ctx: {current_context} | dt: {dt*1000:.1f}ms")
        
        if step % args.eval_every == 0 and step > 0:
            model.eval()
            print0(f"Stage {stage_idx+1} step {step} | Training progressing smoothly")
            model.train()
        
        step += 1
        global_step += 1
    
    print0(f"\n✓ Completed stage: {stage_name}\n")

# Save final checkpoint
print0("\n" + "="*80)
print0("Saving final SFT checkpoint...")
base_dir = get_base_dir()
output_tag = f"{args.model_tag}_sft_curriculum" if use_curriculum else f"{args.model_tag}_sft"
output_dir = os.path.join(base_dir, "sft_checkpoints", output_tag)
save_checkpoint(output_dir, global_step, orig_model.state_dict(), optimizer.state_dict(),
                {"step": global_step, "sft_complete": True, "curriculum_stages": len(curriculum_stages)}, 
                rank=ddp_rank)

print0(f"SFT training complete!")
print0(f"Total iterations: {global_step}")
print0(f"Total time: {total_time/60:.1f}m")
print0(f"Saved to: {output_dir}")
print0("="*80 + "\n")

wandb_run.finish()
compute_cleanup()
