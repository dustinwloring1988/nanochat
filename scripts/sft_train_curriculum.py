"""Fixed-context, manifest-only supervised fine-tuning entry point."""

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

try:
    import wandb
except ImportError:
    wandb = None

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    print0,
)
from nanochat.tokenizer import get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.research_results import (
    runtime_metadata,
    tokenizer_hash,
    write_json_atomic,
)
from nanochat.sft_manifest import (
    resolve_run_local_path,
    validate_jsonl_dataset_manifest,
)
from nanochat.sft_quality import (
    SFTQualityError,
    build_sft_quality_result,
    validate_sft_quality_plan,
    write_sft_quality_result,
)
from nanochat.sft_runtime import (
    SFT_CONTEXT_TOKENS,
    SFTConversationBatchLoader,
    SFTLoaderError,
    SFTCheckpointTransaction,
    build_effective_token_budget,
    capture_rng_state,
    restore_rng_state,
    validate_fixed_context,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-context, manifest-only SFT")
    parser.add_argument("--run-dir", required=True, help="approved run-local workspace")
    parser.add_argument("--run-id", required=True, help="safe run identifier")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--eval-manifest", default=None)
    parser.add_argument("--quality-plan", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--model-step", type=int, default=None)
    parser.add_argument("--device-type", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--total-batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=SFT_CONTEXT_TOKENS)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume-from-step", type=int, default=-1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--run", default="dummy")
    return parser.parse_args()


def load_fixed_config(path):
    if path is None:
        return {
            "stages": [
                {
                    "name": "fixed",
                    "context_range": [SFT_CONTEXT_TOKENS, SFT_CONTEXT_TOKENS],
                    "source_weights": {"manifest": 1.0},
                }
            ]
        }
    config_path = Path(path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    stages = payload.get("stages") if isinstance(payload, dict) else None
    if not isinstance(stages, list) or len(stages) != 1:
        raise SFTLoaderError(
            "The approved SFT entry point requires exactly one fixed-context stage"
        )
    stage = stages[0]
    if stage.get("context_range") != [SFT_CONTEXT_TOKENS, SFT_CONTEXT_TOKENS]:
        raise SFTLoaderError(
            f"SFT context must be fixed at {SFT_CONTEXT_TOKENS}; long-context is not enabled"
        )
    return payload


def resolve_run_path(path, run_root):
    return resolve_run_local_path(path, run_root)


def resolve_run_root(path):
    candidate = Path(path).expanduser().absolute()
    return resolve_run_local_path(candidate, candidate)


def build_run_manifest(
    args,
    config,
    manifest,
    run_dir,
    budget,
    tokenizer_digest,
    quality_plan=None,
    eval_manifest=None,
):
    payload = {
        "sft_run_schema_version": 1,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "dataset_manifest": str(manifest.manifest_path),
        "dataset_manifest_hash": manifest.manifest_hash,
        "tokenizer_hash": tokenizer_digest,
        "curriculum_config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "context_length": SFT_CONTEXT_TOKENS,
        "device_batch_size": budget.device_batch_size,
        "world_size": budget.world_size,
        "gradient_accumulation_steps": budget.gradient_accumulation_steps,
        "effective_tokens": budget.effective_tokens,
        "num_iterations": args.num_iterations,
        "seed": args.seed,
        "user_config": {
            "model_tag": args.model_tag,
            "model_step": args.model_step,
            "max_seq_len": args.max_seq_len,
            "eval_every": args.eval_every,
            "save_every": args.save_every,
            "compile": args.compile,
        },
        "approval": "AG-1 fixed-context local scope",
        "promotion": "not authorized",
    }
    if quality_plan is not None:
        payload["quality_plan_hash"] = quality_plan.plan_hash
        payload["eval_manifest"] = str(quality_plan.eval_manifest.manifest_path)
        payload["eval_manifest_hash"] = quality_plan.eval_manifest.manifest_hash
        payload["quality_budget"] = quality_plan.budget
        payload["quality_thresholds"] = quality_plan.thresholds
        payload["disjointness"] = quality_plan.disjointness.to_dict()
    elif eval_manifest is not None:
        payload["eval_manifest"] = str(eval_manifest.manifest_path)
        payload["eval_manifest_hash"] = eval_manifest.manifest_hash
    return payload


def save_checkpoint(
    checkpoint_root,
    run_root,
    step,
    model,
    optimizer,
    loader,
    metadata,
    rank,
    world_size,
):
    transaction = SFTCheckpointTransaction(
        checkpoint_root,
        run_root=run_root,
        step=step,
        rank=rank,
        world_size=world_size,
        resume=True,
        metadata=metadata,
    )
    try:
        transaction.write_rank_sidecars(
            rank,
            optimizer.state_dict(),
            loader.state_dict(),
            capture_rng_state(include_cuda=torch.cuda.is_available()),
        )
        if rank == 0:
            transaction.write_model(model.state_dict())
        if world_size > 1:
            dist.barrier()
        if rank == 0:
            transaction.commit()
        if world_size > 1:
            dist.barrier()
    except Exception:
        transaction.rollback()
        raise
    return transaction


def evaluate_heldout(
    model,
    manifest,
    tokenizer,
    *,
    batch_size,
    device,
    rank,
    world_size,
    seed,
    eval_tokens,
):
    loader = SFTConversationBatchLoader(
        manifest,
        tokenizer,
        batch_size=batch_size,
        device=device,
        rank=rank,
        world_size=world_size,
        seed=seed,
        sequence_length=SFT_CONTEXT_TOKENS,
    )
    eval_batches = eval_tokens // (batch_size * SFT_CONTEXT_TOKENS * world_size)
    if (
        eval_batches <= 0
        or eval_batches * batch_size * SFT_CONTEXT_TOKENS * world_size != eval_tokens
    ):
        raise SFTLoaderError("eval token budget must divide the held-out batch")
    token_bytes = get_token_bytes(device=device)
    batches = ((item.inputs, item.targets) for item in loader)
    value = evaluate_bpb(model, batches, eval_batches, token_bytes)
    if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value)):
        raise SFTLoaderError("held-out evaluation did not produce a finite BPB")
    return float(value), eval_batches


def main():
    args = parse_args()
    validate_fixed_context(args.max_seq_len)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", args.run_id):
        raise SFTLoaderError("run-id must be a safe single path component")
    if args.num_iterations <= 0:
        raise SFTLoaderError("num-iterations must be positive")
    if args.eval_every < 0 or args.save_every == 0:
        raise SFTLoaderError("eval-every must be non-negative and save-every positive")
    config = load_fixed_config(args.config)
    run_root = resolve_run_root(args.run_dir)
    run_root.mkdir(parents=True, exist_ok=True)
    run_root = resolve_run_root(run_root)
    manifest_path = resolve_run_path(args.dataset_manifest, run_root)
    manifest = validate_jsonl_dataset_manifest(manifest_path, run_root=run_root)
    val_manifest = None
    quality_plan = None
    if args.eval_manifest is not None:
        val_manifest = validate_jsonl_dataset_manifest(
            resolve_run_path(args.eval_manifest, run_root), run_root=run_root
        )
        if args.quality_plan is None:
            raise SFTLoaderError(
                "a separate eval-manifest requires an approved quality-plan"
            )
        try:
            quality_plan = validate_sft_quality_plan(
                resolve_run_path(args.quality_plan, run_root),
                run_root=run_root,
                train_manifest=manifest,
                eval_manifest=val_manifest,
            )
        except SFTQualityError as exc:
            raise SFTLoaderError(str(exc)) from exc
        if quality_plan.run_id != args.run_id:
            raise SFTLoaderError("quality-plan run-id does not match --run-id")
        if args.resume_from_step != -1:
            raise SFTLoaderError(
                "quality runs require a fresh run; resume cannot redefine the baseline"
            )
    elif args.quality_plan is not None:
        raise SFTLoaderError("quality-plan requires --eval-manifest")
    if args.eval_every and val_manifest is None:
        raise SFTLoaderError("eval-every requires a separate eval-manifest")

    device_type = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    ddp, rank, local_rank, world_size, device = compute_init(
        device_type, seed=args.seed
    )
    micro_batch_tokens = args.device_batch_size * SFT_CONTEXT_TOKENS * world_size
    if args.total_batch_size is None:
        args.total_batch_size = micro_batch_tokens
    if args.total_batch_size <= 0 or args.total_batch_size % micro_batch_tokens:
        raise SFTLoaderError(
            "total-batch-size must be a positive multiple of device batch times 2048 times world size"
        )
    gradient_accumulation_steps = args.total_batch_size // micro_batch_tokens
    budget = build_effective_token_budget(
        args.device_batch_size,
        world_size,
        gradient_accumulation_steps,
        sequence_length=args.max_seq_len,
        total_batch_size=args.total_batch_size,
    )
    if quality_plan is not None:
        quality_budget = quality_plan.budget
        expected_quality_values = {
            "context_length": SFT_CONTEXT_TOKENS,
            "device_batch_size": budget.device_batch_size,
            "world_size": budget.world_size,
            "gradient_accumulation_steps": budget.gradient_accumulation_steps,
            "effective_tokens": budget.effective_tokens,
            "optimization_steps": args.num_iterations,
        }
        for field, expected in expected_quality_values.items():
            if quality_budget.get(field) != expected:
                raise SFTLoaderError(
                    f"quality-plan {field} does not match the requested run"
                )

    base_dir = Path(
        os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
    ).expanduser()
    if not base_dir.is_dir():
        raise SFTLoaderError(
            "an existing read-only base directory is required; refusing to create the trusted cache"
        )
    use_dummy_wandb = args.run == "dummy" or rank != 0
    if not use_dummy_wandb and wandb is None:
        raise RuntimeError("wandb is required for non-dummy SFT runs")
    wandb_run = (
        DummyWandb()
        if use_dummy_wandb
        else wandb.init(project="nanochat-sft-fixed", name=args.run, config=vars(args))
    )
    model, tokenizer, meta = load_model(
        "base", device, phase="train", model_tag=args.model_tag, step=args.model_step
    )
    tokenizer_digest = tokenizer_hash(base_dir)
    if tokenizer_digest is None:
        raise SFTLoaderError("tokenizer hash is unavailable; refusing SFT")
    optimizer = model.setup_optimizer(
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
    )
    orig_model = model
    if args.compile:
        model = torch.compile(orig_model, dynamic=False)
    loader = SFTConversationBatchLoader(
        manifest,
        tokenizer,
        batch_size=args.device_batch_size,
        device=device,
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        sequence_length=args.max_seq_len,
    )

    def make_val_loader():
        if val_manifest is None:
            return None
        return SFTConversationBatchLoader(
            val_manifest,
            tokenizer,
            batch_size=(
                quality_plan.budget["eval_batch_size"]
                if quality_plan is not None
                else args.device_batch_size
            ),
            device=device,
            rank=rank,
            world_size=world_size,
            seed=args.seed + 1,
            sequence_length=args.max_seq_len,
        )

    checkpoint_root = resolve_run_path("checkpoints", run_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    step = 0
    if args.resume_from_step != -1:
        from nanochat.sft_runtime import load_sft_checkpoint

        loaded = load_sft_checkpoint(
            checkpoint_root,
            run_root=run_root,
            step=args.resume_from_step,
            rank=rank,
            device=device,
        )
        orig_model.load_state_dict(loaded.model_state, strict=True)
        optimizer.load_state_dict(loaded.optimizer_state)
        restore_rng_state(loaded.rng_state, device=device)
        loader = SFTConversationBatchLoader(
            manifest,
            tokenizer,
            batch_size=args.device_batch_size,
            device=device,
            rank=rank,
            world_size=world_size,
            seed=args.seed,
            sequence_length=args.max_seq_len,
            resume_state=loaded.loader_state,
        )
        step = args.resume_from_step
    run_manifest = build_run_manifest(
        args,
        config,
        manifest,
        run_root,
        budget,
        tokenizer_digest,
        quality_plan=quality_plan,
        eval_manifest=val_manifest,
    )
    if rank == 0:
        write_json_atomic(run_root / "run_manifest.json", run_manifest)
    batch = next(loader)
    token_bytes = get_token_bytes(device=device)
    baseline_bpb = None
    baseline_eval_batches = None
    if quality_plan is not None:
        orig_model.eval()
        with torch.no_grad():
            baseline_bpb, baseline_eval_batches = evaluate_heldout(
                orig_model,
                val_manifest,
                tokenizer,
                batch_size=quality_plan.budget["eval_batch_size"],
                device=device,
                rank=rank,
                world_size=world_size,
                seed=args.seed + 1,
                eval_tokens=quality_plan.budget["eval_tokens"],
            )
        orig_model.train()
        print0(f"baseline held-out bpb: {baseline_bpb:.6f}")
    for _ in range(step, args.num_iterations):
        if args.eval_every and _ % args.eval_every == 0 and val_manifest is not None:
            model.eval()
            with torch.no_grad():
                if quality_plan is not None:
                    val_bpb, _ = evaluate_heldout(
                        orig_model,
                        val_manifest,
                        tokenizer,
                        batch_size=quality_plan.budget["eval_batch_size"],
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        seed=args.seed + 1,
                        eval_tokens=quality_plan.budget["eval_tokens"],
                    )
                else:
                    val_loader = make_val_loader()
                    val_batches = ((item.inputs, item.targets) for item in val_loader)
                    val_bpb = evaluate_bpb(orig_model, val_batches, 1, token_bytes)
            print0(f"step {_} | validation bpb: {val_bpb:.6f}")
            model.train()
        started = time.perf_counter()
        for _micro in range(budget.gradient_accumulation_steps):
            loss = model(batch.inputs, batch.targets)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("non-finite SFT loss")
            (loss / budget.gradient_accumulation_steps).backward()
            batch = next(loader)
        optimizer.step()
        model.zero_grad(set_to_none=True)
        step += 1
        elapsed = time.perf_counter() - started
        print0(f"step {step}/{args.num_iterations} | dt: {elapsed:.3f}s")
        if step % args.save_every == 0 or step == args.num_iterations:
            metadata = {
                **run_manifest,
                "global_step": step,
                "stage_index": 0,
                "stage_name": config["stages"][0]["name"],
                "stage_local_step": step,
                "device_batch_size": budget.device_batch_size,
                "world_size": budget.world_size,
                "gradient_accumulation_steps": budget.gradient_accumulation_steps,
                "total_batch_size": budget.effective_tokens,
                "effective_tokens": budget.effective_tokens,
                "sequence_length": SFT_CONTEXT_TOKENS,
                "context_length": SFT_CONTEXT_TOKENS,
                "max_seq_len": SFT_CONTEXT_TOKENS,
                "resume_supported": True,
            }
            save_checkpoint(
                checkpoint_root,
                run_root,
                step,
                orig_model,
                optimizer,
                loader,
                metadata,
                rank,
                world_size,
            )
    quality_result = None
    if quality_plan is not None:
        orig_model.eval()
        with torch.no_grad():
            final_bpb, final_eval_batches = evaluate_heldout(
                orig_model,
                val_manifest,
                tokenizer,
                batch_size=quality_plan.budget["eval_batch_size"],
                device=device,
                rank=rank,
                world_size=world_size,
                seed=args.seed + 1,
                eval_tokens=quality_plan.budget["eval_tokens"],
            )
        quality_result = build_sft_quality_result(
            quality_plan,
            baseline_bpb=baseline_bpb,
            final_bpb=final_bpb,
            eval_batches=final_eval_batches,
            tokenizer_hash=tokenizer_digest,
            model_tag=args.model_tag,
            model_step=args.num_iterations,
            runtime=runtime_metadata(),
        )
        if rank == 0:
            write_sft_quality_result(run_root / "quality_result.json", quality_result)
            print0(
                "quality decision: "
                f"{quality_result['decision']['status']} "
                f"(final held-out bpb: {final_bpb:.6f})"
            )
    if rank == 0:
        print0(f"SFT fixed-context run complete: {run_root}")
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
