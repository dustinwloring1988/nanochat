"""Prepare an approved, pinned local SFT quality package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import HfApi

from nanochat.sft_manifest import write_jsonl_dataset_manifest
from nanochat.sft_quality import write_sft_quality_plan
from tasks.common import normalize_messages

DEFAULT_REPO = "HuggingFaceTB/smol-smoltalk"
DEFAULT_REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a pinned SFT quality package")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-meta", required=True)
    parser.add_argument("--model-tag", default="quality-base")
    parser.add_argument("--train-examples", type=int, default=64)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--optimization-steps", type=int, default=16)
    parser.add_argument("--eval-tokens", type=int, default=65536)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--approved-at", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_rows(
    split: str, count: int, repo: str, revision: str
) -> list[dict[str, Any]]:
    stream = load_dataset(repo, split=split, streaming=True, revision=revision)
    records = []
    for row in stream:
        messages = normalize_messages(row.get("messages", []))
        if len(messages) < 2:
            continue
        records.append({"conversation": {"messages": messages}})
        if len(records) == count:
            break
    if len(records) != count:
        raise RuntimeError(f"{repo}:{split} did not provide {count} valid records")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "record_count": len(records),
    }


def copy_base_checkpoint(
    base_dir: Path, checkpoint: Path, meta: Path, tag: str
) -> None:
    target = base_dir / "base_checkpoints" / tag
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        checkpoint, target / f"model_{checkpoint.stem.removeprefix('model_')}.pt"
    )
    shutil.copy2(meta, target / f"meta_{meta.stem.removeprefix('meta_')}.json")
    optimizer = checkpoint.with_name(
        "optim_" + checkpoint.stem.removeprefix("model_") + "_rank0.pt"
    )
    if optimizer.is_file():
        shutil.copy2(optimizer, target / optimizer.name)


def make_manifest(
    run_root: Path,
    path: Path,
    split: str,
    file_info: dict[str, int | str],
    repo: str,
    revision: str,
):
    payload = {
        "manifest_schema_version": 1,
        "run_id": run_root.name,
        "provenance": {
            "source": repo,
            "revision": revision,
            "reviewed": True,
            "split_method": "pinned Hugging Face source split",
        },
        "datasets": [
            {
                "source": repo,
                "split": split,
                "license": {
                    "name": "Apache-2.0",
                    "url": "https://www.apache.org/licenses/LICENSE-2.0",
                    "redistribution": True,
                },
                "provenance": {
                    "source": repo,
                    "revision": revision,
                    "dataset_info_license": "apache-2.0",
                },
                "files": [
                    {
                        "path": file_info["path"],
                        "sha256": file_info["sha256"],
                        "byte_count": file_info["byte_count"],
                        "record_count": file_info["record_count"],
                    }
                ],
            }
        ],
    }
    return write_jsonl_dataset_manifest(path, payload, run_root)


def main():
    args = parse_args()
    if args.train_examples <= 0 or args.eval_examples <= 0:
        raise ValueError("record counts must be positive")
    if args.optimization_steps <= 0 or args.eval_tokens <= 0:
        raise ValueError("training and evaluation budgets must be positive")
    if args.eval_tokens % 2048:
        raise ValueError("eval-tokens must be divisible by the 2048-token context")
    info = HfApi().dataset_info(args.repo, revision=args.revision)
    license_name = str((info.card_data or {}).get("license", "")).casefold()
    if license_name != "apache-2.0":
        raise RuntimeError(f"unexpected dataset license: {license_name or 'missing'}")
    run_root = Path(args.run_dir).expanduser().absolute()
    run_root.mkdir(parents=True, exist_ok=True)
    data_root = run_root / "data"
    train_records = collect_rows("train", args.train_examples, args.repo, args.revision)
    eval_records = collect_rows("test", args.eval_examples, args.repo, args.revision)
    train_path = data_root / "train.jsonl"
    eval_path = data_root / "heldout.jsonl"
    train_info = {
        "path": train_path.relative_to(run_root).as_posix(),
        **write_jsonl(train_path, train_records),
    }
    eval_info = {
        "path": eval_path.relative_to(run_root).as_posix(),
        **write_jsonl(eval_path, eval_records),
    }
    train_manifest = make_manifest(
        run_root,
        data_root / "train-manifest.json",
        "train",
        train_info,
        args.repo,
        args.revision,
    )
    eval_manifest = make_manifest(
        run_root,
        data_root / "heldout-manifest.json",
        "heldout",
        eval_info,
        args.repo,
        args.revision,
    )
    effective_tokens = 8192
    quality_plan = {
        "quality_plan_schema_version": 1,
        "run_id": run_root.name,
        "train_manifest": str(train_manifest.manifest_path),
        "eval_manifest": str(eval_manifest.manifest_path),
        "approval": {
            "approved": True,
            "reviewed": True,
            "approved_by": args.approved_by,
            "approved_at": args.approved_at,
            "scope": "approved local production SFT quality evaluation; no promotion",
        },
        "budget": {
            "context_length": 2048,
            "device_batch_size": 1,
            "world_size": 1,
            "gradient_accumulation_steps": 4,
            "effective_tokens": effective_tokens,
            "optimization_steps": args.optimization_steps,
            "train_tokens": args.optimization_steps * effective_tokens,
            "eval_batch_size": 1,
            "eval_tokens": args.eval_tokens,
        },
        "thresholds": {
            "pass_max_bpb": 1.6,
            "stop_min_bpb": 2.0,
            "min_improvement": 0.01,
        },
        "provenance": {
            "source": args.repo,
            "revision": args.revision,
            "split_method": "pinned source train/test splits; no cross-split record reuse",
            "license_review": True,
            "dataset_license": "Apache-2.0",
        },
    }
    quality = write_sft_quality_plan(
        run_root / "quality-plan.json", quality_plan, run_root
    )
    base_dir = Path(args.base_dir).expanduser().absolute()
    base_dir.mkdir(parents=True, exist_ok=True)
    copy_base_checkpoint(
        base_dir,
        Path(args.checkpoint).expanduser().absolute(),
        Path(args.model_meta).expanduser().absolute(),
        args.model_tag,
    )
    summary = {
        "run_id": run_root.name,
        "repo": args.repo,
        "revision": args.revision,
        "license": "Apache-2.0",
        "train_manifest_hash": train_manifest.manifest_hash,
        "eval_manifest_hash": eval_manifest.manifest_hash,
        "quality_plan_hash": quality.plan_hash,
        "disjointness": quality.disjointness.to_dict(),
        "base_dir": str(base_dir),
        "model_tag": args.model_tag,
        "promotion": "not authorized",
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
