"""Run the bounded two-step 2048-to-8192 model, loader, resume, and resource proof."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.flash_attention import USE_FA3
from nanochat.gpt import GPT, GPTConfig
from nanochat.long_context import (
    LONG_CONTEXT_SCOPE,
    LongContextStepEvidence,
    build_long_context_approval,
    build_long_context_plan,
    evidence_hash,
    validate_long_context_evidence,
    write_long_context_approval,
)
from nanochat.multi_source_dataloader import (
    tokenizing_distributed_data_loader_multi_source,
)
from nanochat.research_results import dataset_manifest_hash, write_json_atomic
from nanochat.tokenizer import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run the approved long-context proof")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-meta", default=None)
    parser.add_argument("--device-type", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="climbmix")
    parser.add_argument("--tokenizer-batch-size", type=int, default=2)
    parser.add_argument("--buffer-size", type=int, default=8)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--no-save-checkpoint", action="store_true")
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        _json_value(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def _state_delta(left: Any, right: Any) -> float:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError("state tensor types differ")
        if left.shape != right.shape:
            raise AssertionError("state tensor shapes differ")
        return float((left.detach().cpu() - right.detach().cpu()).abs().max())
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError("state mapping keys differ")
        return max((_state_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError("state sequence lengths differ")
        return max((_state_delta(a, b) for a, b in zip(left, right)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if left != right:
        raise AssertionError("state values differ")
    return 0.0


def _batch_equal(
    left: tuple[torch.Tensor, torch.Tensor], right: tuple[torch.Tensor, torch.Tensor]
) -> bool:
    return torch.equal(left[0], right[0]) and torch.equal(left[1], right[1])


def _load_model(
    checkpoint: Path, meta_path: Path, device: torch.device
) -> tuple[GPT, dict, GPTConfig]:
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    model_config = dict(meta["model_config"])
    model_config["sequence_len"] = 8192
    config = GPTConfig(**model_config)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True, assign=True)
    model.train()
    return model, meta, config


def _make_loader(
    tokenizer,
    base_dir: str,
    source: str,
    sequence_length: int,
    device: torch.device,
    seed: int,
    tokenizer_batch_size: int,
    buffer_size: int,
    resume_state_dict=None,
):
    return tokenizing_distributed_data_loader_multi_source(
        tokenizer=tokenizer,
        source_weights={source: 1.0},
        B=1,
        T=sequence_length,
        split="train",
        tokenizer_threads=1,
        tokenizer_batch_size=tokenizer_batch_size,
        buffer_size=buffer_size,
        device=device,
        seed=seed,
        resume_state_dict=resume_state_dict,
    )


def _run_stage(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    loader,
    gradient_accumulation_steps: int,
    device: torch.device,
):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    batch = next(loader)
    first_batch = (batch[0].detach().cpu().clone(), batch[1].detach().cpu().clone())
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(gradient_accumulation_steps):
        loss = model(batch[0], batch[1])
        if not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite long-context loss")
        losses.append(float(loss.detach().cpu()))
        (loss / gradient_accumulation_steps).backward()
        if micro_step + 1 < gradient_accumulation_steps:
            batch = next(loader)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    pending = next(loader)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "loss": sum(losses) / len(losses),
        "losses": losses,
        "elapsed": elapsed,
        "allocated": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 1
        ),
        "reserved": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 1
        ),
        "first_batch": first_batch,
        "pending": (
            pending[0].detach().cpu().clone(),
            pending[1].detach().cpu().clone(),
        ),
        "loader_state": pending[2],
    }


def _make_optimizer(model: GPT, device: torch.device):
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def main():
    args = parse_args()
    if args.device_type not in {"cuda", "cpu"}:
        raise ValueError("long-context proof supports cuda or cpu")
    if args.device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved resource proof")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_root = Path(args.run_dir).expanduser().absolute()
    run_root.mkdir(parents=True, exist_ok=True)
    os.environ["NANOCHAT_BASE_DIR"] = str(Path(args.base_dir).expanduser().absolute())
    device = torch.device(args.device_type)
    checkpoint = Path(args.checkpoint).expanduser().absolute()
    meta_path = (
        Path(args.model_meta).expanduser().absolute()
        if args.model_meta
        else checkpoint.with_name(
            "meta_" + checkpoint.stem.removeprefix("model_") + ".json"
        )
    )
    if not checkpoint.is_file() or not meta_path.is_file():
        raise FileNotFoundError("checkpoint and model metadata are required")
    tokenizer = get_tokenizer()
    plan = build_long_context_plan()
    model, _, model_config = _load_model(checkpoint, meta_path, device)
    optimizer = _make_optimizer(model, device)
    model_config_hash = _hash_json(asdict(model_config))
    data_hash = dataset_manifest_hash(args.base_dir)
    backend = "flash-attn-3" if USE_FA3 else "torch-sdpa"
    dtype = str(COMPUTE_DTYPE).removeprefix("torch.")
    short_loader = _make_loader(
        tokenizer,
        args.base_dir,
        args.source,
        2048,
        device,
        args.seed,
        args.tokenizer_batch_size,
        args.buffer_size,
    )
    short_result = _run_stage(model, optimizer, short_loader, 4, device)
    short_model_state = _cpu_clone(model.state_dict())
    short_optimizer_state = _cpu_clone(optimizer.state_dict())
    short_pending = short_result["pending"]
    short_loader_state = short_result["loader_state"]
    if not args.no_save_checkpoint:
        torch.save(
            {
                "model": short_model_state,
                "optimizer": short_optimizer_state,
                "loader": short_loader_state,
                "plan": plan.to_dict(),
            },
            run_root / "resume-step-0.pt",
        )
    del short_loader
    long_loader = _make_loader(
        tokenizer,
        args.base_dir,
        args.source,
        8192,
        device,
        args.seed,
        args.tokenizer_batch_size,
        args.buffer_size,
    )
    long_result = _run_stage(model, optimizer, long_loader, 1, device)
    long_model_state = _cpu_clone(model.state_dict())
    long_optimizer_state = _cpu_clone(optimizer.state_dict())
    long_pending = long_result["pending"]
    long_loader_state = long_result["loader_state"]
    del model, optimizer, short_model_state, short_optimizer_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.no_save_checkpoint:
        raise RuntimeError("resume proof requires the run-local checkpoint")
    checkpoint_payload = torch.load(
        run_root / "resume-step-0.pt", map_location="cpu", weights_only=True
    )
    resumed_model, _, _ = _load_model(checkpoint, meta_path, device)
    resumed_optimizer = _make_optimizer(resumed_model, device)
    resumed_model.load_state_dict(checkpoint_payload["model"], strict=True)
    resumed_optimizer.load_state_dict(checkpoint_payload["optimizer"])
    initial_model_delta = _state_delta(
        checkpoint_payload["model"], resumed_model.state_dict()
    )
    initial_optimizer_delta = _state_delta(
        checkpoint_payload["optimizer"], resumed_optimizer.state_dict()
    )
    resumed_short_loader = _make_loader(
        tokenizer,
        args.base_dir,
        args.source,
        2048,
        device,
        args.seed,
        args.tokenizer_batch_size,
        args.buffer_size,
        resume_state_dict=short_loader_state,
    )
    resumed_pending = next(resumed_short_loader)
    resume_batch_equal = _batch_equal(
        short_pending,
        (resumed_pending[0].detach().cpu(), resumed_pending[1].detach().cpu()),
    )
    resumed_loader_state = resumed_pending[2]
    resumed_long_loader = _make_loader(
        tokenizer,
        args.base_dir,
        args.source,
        8192,
        device,
        args.seed,
        args.tokenizer_batch_size,
        args.buffer_size,
    )
    resumed_long_result = _run_stage(
        resumed_model, resumed_optimizer, resumed_long_loader, 1, device
    )
    model_delta = _state_delta(long_model_state, resumed_model.state_dict())
    optimizer_delta = _state_delta(long_optimizer_state, resumed_optimizer.state_dict())
    loader_equal = _hash_json(short_loader_state) == _hash_json(resumed_loader_state)
    long_batch_equal = _batch_equal(
        long_pending,
        resumed_long_result["pending"],
    )
    long_first_batch_equal = _batch_equal(
        long_result["first_batch"], resumed_long_result["first_batch"]
    )
    if (
        model_delta != 0.0
        or optimizer_delta != 0.0
        or not resume_batch_equal
        or not loader_equal
        or not long_batch_equal
    ):
        raise AssertionError(
            json.dumps(
                {
                    "model_delta": model_delta,
                    "optimizer_delta": optimizer_delta,
                    "initial_model_delta": initial_model_delta,
                    "initial_optimizer_delta": initial_optimizer_delta,
                    "resume_batch_equal": resume_batch_equal,
                    "loader_equal": loader_equal,
                    "long_batch_equal": long_batch_equal,
                    "long_first_batch_equal": long_first_batch_equal,
                },
                sort_keys=True,
            )
        )
    evidence = [
        LongContextStepEvidence(
            step=0,
            sequence_length=2048,
            runtime_verified=True,
            model_forward_backward=True,
            loader_batch_verified=True,
            resume_model_equal=model_delta == 0.0,
            resume_optimizer_equal=optimizer_delta == 0.0,
            resume_loader_equal=loader_equal,
            resume_next_batch_equal=resume_batch_equal and long_batch_equal,
            loss=short_result["loss"],
            peak_vram_bytes=short_result["allocated"],
            reserved_vram_bytes=short_result["reserved"],
            active_tokens_per_second=8192 / short_result["elapsed"],
            wall_clock_seconds=short_result["elapsed"],
            model_config_hash=model_config_hash,
            loader_state_hash=_hash_json(short_loader_state),
            next_batch_hash=_hash_json(short_pending),
            data_hash=data_hash,
            attention_backend=backend,
            dtype=dtype,
        ),
        LongContextStepEvidence(
            step=1,
            sequence_length=8192,
            runtime_verified=True,
            model_forward_backward=True,
            loader_batch_verified=True,
            resume_model_equal=model_delta == 0.0,
            resume_optimizer_equal=optimizer_delta == 0.0,
            resume_loader_equal=loader_equal,
            resume_next_batch_equal=resume_batch_equal and long_batch_equal,
            loss=long_result["loss"],
            peak_vram_bytes=long_result["allocated"],
            reserved_vram_bytes=long_result["reserved"],
            active_tokens_per_second=8192 / long_result["elapsed"],
            wall_clock_seconds=long_result["elapsed"],
            model_config_hash=model_config_hash,
            loader_state_hash=_hash_json(long_loader_state),
            next_batch_hash=_hash_json(long_pending),
            data_hash=data_hash,
            attention_backend=backend,
            dtype=dtype,
        ),
    ]
    validate_long_context_evidence(plan, evidence)
    approval = build_long_context_approval(
        plan,
        evidence,
        approval_id=args.approval_id,
        approved_by=args.approved_by,
        approved_at=args.approved_at,
    )
    write_long_context_approval(run_root / "long_context_activation.json", approval)
    evidence_payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": LONG_CONTEXT_SCOPE,
        "plan": plan.to_dict(),
        "evidence": [item.to_dict() for item in evidence],
        "evidence_hash": evidence_hash(evidence),
        "approval": approval.to_dict(),
        "model": {
            "checkpoint": str(checkpoint),
            "model_config": asdict(model_config),
            "model_config_hash": model_config_hash,
            "device": str(device),
            "runtime_dtype": dtype,
            "attention_backend": backend,
        },
        "loader": {
            "source": args.source,
            "base_dir": str(Path(args.base_dir).expanduser().absolute()),
            "device_batch_size": 1,
            "world_size": 1,
            "effective_token_batch": 8192,
            "gradient_accumulation_steps": [4, 1],
            "resume_checkpoint": "resume-step-0.pt",
        },
        "resume": {
            "initial_model_max_abs_delta": initial_model_delta,
            "initial_optimizer_max_abs_delta": initial_optimizer_delta,
            "model_max_abs_delta": model_delta,
            "optimizer_max_abs_delta": optimizer_delta,
            "loader_state_equal": loader_equal,
            "next_batch_equal": resume_batch_equal and long_batch_equal,
            "first_batch_equal": long_first_batch_equal,
        },
        "promotion": "not authorized",
    }
    write_json_atomic(run_root / "long_context_evidence.json", evidence_payload)
    print(json.dumps(evidence_payload, sort_keys=True))


if __name__ == "__main__":
    main()
