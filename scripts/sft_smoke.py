"""Bounded fixed-context SFT checkpoint/resume runtime probe."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.sft_manifest import resolve_run_local_path, write_jsonl_dataset_manifest
from nanochat.sft_runtime import (
    SFT_CONTEXT_TOKENS,
    SFTConversationBatchLoader,
    SFTCheckpointTransaction,
    capture_rng_state,
    load_sft_checkpoint,
    restore_rng_state,
    validate_fixed_context,
)
from nanochat.tokenizer import RustBPETokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run a bounded local SFT probe")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--device-type", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--optimizer", choices=("auto", "nanochat", "adamw"), default="auto"
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_hash(tokenizer_dir):
    digest = hashlib.sha256()
    for name in ("tokenizer.pkl", "token_bytes.pt"):
        path = Path(tokenizer_dir) / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def create_manifest(run_root):
    data_dir = run_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "fabricated-train.jsonl"
    records = [
        {
            "conversation": {
                "messages": [
                    {"role": "user", "content": f"fabricated question {index}"},
                    {"role": "assistant", "content": f"fabricated answer {index}"},
                ]
            }
        }
        for index in range(8)
    ]
    data_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = {
        "manifest_schema_version": 1,
        "run_id": "fabricated-sft-probe",
        "provenance": {
            "source": "fabricated-runtime-probe",
            "revision": "1",
            "reviewed": True,
        },
        "datasets": [
            {
                "source": "fabricated",
                "split": "train",
                "license": {"name": "test-only", "redistribution": False},
                "provenance": {"source": "fabricated-runtime-probe", "revision": "1"},
                "files": [
                    {
                        "path": data_path.relative_to(run_root).as_posix(),
                        "sha256": sha256_file(data_path),
                        "byte_count": data_path.stat().st_size,
                        "record_count": len(records),
                    }
                ],
            }
        ],
    }
    return write_jsonl_dataset_manifest(
        run_root / "dataset-manifest.json", payload, run_root
    )


def build_model(vocab_size, device):
    config = GPTConfig(
        sequence_len=SFT_CONTEXT_TOKENS,
        vocab_size=vocab_size,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    return model


def build_optimizer(model, device, requested):
    if requested == "auto":
        requested = "nanochat" if device.type == "cuda" else "adamw"
    if requested == "nanochat":
        return (
            model.setup_optimizer(
                unembedding_lr=0.004,
                embedding_lr=0.2,
                matrix_lr=0.02,
                weight_decay=0.0,
            ),
            requested,
        )
    return torch.optim.AdamW(model.parameters(), lr=1e-3), requested


def run_step(model, optimizer, loader, batch):
    loss = model(batch.inputs, batch.targets)
    if not torch.isfinite(loss).all():
        raise FloatingPointError("non-finite probe loss")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach().cpu()), batch, next(loader)


def compare_state(left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError("tensor state shape mismatch")
        if left.shape != right.shape:
            raise AssertionError("tensor state shape mismatch")
        return float((left.detach().cpu() - right.detach().cpu()).abs().max())
    if isinstance(left, dict):
        if set(left) != set(right):
            raise AssertionError("mapping state keys mismatch")
        return max((compare_state(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError("sequence state length mismatch")
        return max((compare_state(a, b) for a, b in zip(left, right)), default=0.0)
    if left != right:
        raise AssertionError(f"state value mismatch: {left!r} != {right!r}")
    return 0.0


def main():
    args = parse_args()
    validate_fixed_context(SFT_CONTEXT_TOKENS)
    if args.steps != 2:
        raise ValueError("the bounded probe requires exactly two steps")
    run_root = resolve_run_local_path(
        Path(args.run_dir).expanduser().absolute(),
        Path(args.run_dir).expanduser().absolute(),
    )
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = create_manifest(run_root)
    tokenizer = RustBPETokenizer.from_directory(args.tokenizer_dir)
    device = torch.device(args.device_type)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    probe_started = time.perf_counter()
    torch.manual_seed(args.seed)
    model = build_model(tokenizer.get_vocab_size(), device)
    optimizer, optimizer_kind = build_optimizer(model, device, args.optimizer)
    loader = SFTConversationBatchLoader(
        manifest,
        tokenizer,
        batch_size=1,
        device=device,
        rank=0,
        world_size=1,
        seed=args.seed,
        sequence_length=SFT_CONTEXT_TOKENS,
    )
    checkpoint_root = resolve_run_local_path("checkpoints", run_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    first_batch = next(loader)
    first_loss, _, pending_batch = run_step(model, optimizer, loader, first_batch)
    transaction = SFTCheckpointTransaction(
        checkpoint_root,
        run_root=run_root,
        step=1,
        metadata={
            "run_id": "fabricated-sft-probe",
            "seed": args.seed,
            "sequence_length": SFT_CONTEXT_TOKENS,
            "context_length": SFT_CONTEXT_TOKENS,
            "device_batch_size": 1,
            "world_size": 1,
            "gradient_accumulation_steps": 1,
            "total_batch_size": SFT_CONTEXT_TOKENS,
            "effective_tokens": SFT_CONTEXT_TOKENS,
            "dataset_manifest_hash": manifest.manifest_hash,
            "tokenizer_hash": tokenizer_hash(args.tokenizer_dir),
            "resume_supported": True,
            "optimizer_kind": optimizer_kind,
        },
    )
    transaction.write_rank_sidecars(
        0,
        optimizer.state_dict(),
        loader.state_dict(),
        capture_rng_state(include_cuda=device.type == "cuda"),
    )
    transaction.write_model(model.state_dict())
    transaction.commit()
    second_loss, reference_batch, _ = run_step(model, optimizer, loader, pending_batch)
    reference_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    resumed_model = build_model(tokenizer.get_vocab_size(), device)
    resumed_optimizer, _ = build_optimizer(resumed_model, device, optimizer_kind)
    loaded = load_sft_checkpoint(
        checkpoint_root,
        run_root=run_root,
        step=1,
        rank=0,
        device=device,
    )
    resumed_model.load_state_dict(loaded.model_state, strict=True)
    resumed_optimizer.load_state_dict(loaded.optimizer_state)
    restore_rng_state(loaded.rng_state, device=device)
    resumed_loader = SFTConversationBatchLoader(
        manifest,
        tokenizer,
        batch_size=1,
        device=device,
        rank=0,
        world_size=1,
        seed=args.seed,
        sequence_length=SFT_CONTEXT_TOKENS,
        resume_state=loaded.loader_state,
    )
    resumed_batch = next(resumed_loader)
    resumed_loss, resumed_batch, _ = run_step(
        resumed_model, resumed_optimizer, resumed_loader, resumed_batch
    )
    resumed_state = {
        key: value.detach().cpu().clone()
        for key, value in resumed_model.state_dict().items()
    }
    model_delta = compare_state(reference_state, resumed_state)
    batch_equal = torch.equal(
        reference_batch.inputs, resumed_batch.inputs
    ) and torch.equal(reference_batch.targets, resumed_batch.targets)
    if model_delta != 0.0 or not batch_equal:
        raise AssertionError(
            json.dumps(
                {
                    "model_max_abs_delta": model_delta,
                    "next_batch_equal": batch_equal,
                    "reference_loss": second_loss,
                    "resumed_loss": resumed_loss,
                    "reference_input_head": reference_batch.inputs[0, :8].tolist(),
                    "resumed_input_head": resumed_batch.inputs[0, :8].tolist(),
                    "reference_target_head": reference_batch.targets[0, :8].tolist(),
                    "resumed_target_head": resumed_batch.targets[0, :8].tolist(),
                    "reference_cursor": reference_batch.state.get("cursor"),
                    "resumed_cursor": resumed_batch.state.get("cursor"),
                },
                sort_keys=True,
            )
        )
    evidence = {
        "status": "ok",
        "device": str(device),
        "elapsed_seconds": round(time.perf_counter() - probe_started, 6),
        "peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "optimizer": optimizer_kind,
        "sequence_length": SFT_CONTEXT_TOKENS,
        "first_loss": first_loss,
        "resumed_loss": resumed_loss,
        "second_loss": second_loss,
        "model_max_abs_delta": model_delta,
        "next_batch_equal": batch_equal,
        "checkpoint": str(loaded.checkpoint_path),
        "manifest_hash": manifest.manifest_hash,
        "tokenizer_hash": tokenizer_hash(args.tokenizer_dir),
        "external_cache_written": False,
        "quality_claim": False,
    }
    print(json.dumps(evidence, sort_keys=True))


if __name__ == "__main__":
    main()
