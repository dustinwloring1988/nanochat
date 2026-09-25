import hashlib
import json

import pytest

from nanochat.sft_manifest import validate_jsonl_dataset_manifest
from nanochat.sft_quality import (
    SFTQualityError,
    build_quality_decision,
    build_sft_quality_result,
    validate_sft_quality_plan,
    validate_sft_quality_result,
)


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "byte_count": path.stat().st_size,
        "record_count": len(records),
    }


def make_manifest(path, split, records, source):
    data_path = path.parent / f"{split}.jsonl"
    file_info = write_jsonl(data_path, records)
    payload = {
        "manifest_schema_version": 1,
        "run_id": "quality-test",
        "provenance": {"reviewed": True, "source": source, "revision": "1"},
        "datasets": [
            {
                "source": source,
                "split": split,
                "license": {"name": "test-license", "redistribution": False},
                "provenance": {"source": source, "revision": "1"},
                "files": [
                    {
                        "path": data_path.relative_to(path.parent).as_posix(),
                        **file_info,
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return validate_jsonl_dataset_manifest(path, run_root=path.parent)


def quality_plan(train_manifest, eval_manifest):
    return {
        "quality_plan_schema_version": 1,
        "run_id": "quality-test",
        "train_manifest": str(train_manifest.manifest_path),
        "eval_manifest": str(eval_manifest.manifest_path),
        "approval": {
            "approved": True,
            "reviewed": True,
            "approved_by": "test-reviewer",
            "approved_at": "2026-09-25",
            "scope": "fixed-context local quality evaluation",
        },
        "budget": {
            "context_length": 2048,
            "device_batch_size": 1,
            "world_size": 1,
            "gradient_accumulation_steps": 2,
            "effective_tokens": 4096,
            "optimization_steps": 2,
            "train_tokens": 8192,
            "eval_batch_size": 1,
            "eval_tokens": 4096,
        },
        "thresholds": {
            "pass_max_bpb": 1.5,
            "stop_min_bpb": 1.8,
            "min_improvement": 0.01,
        },
        "provenance": {
            "source": "test-source",
            "revision": "1",
            "split_method": "deterministic source split",
            "license_review": True,
        },
    }


def test_quality_plan_requires_disjoint_manifests(tmp_path):
    train = make_manifest(
        tmp_path / "train-manifest.json",
        "train",
        [{"messages": [{"role": "user", "content": "train"}]}],
        "test-source",
    )
    evaluation = make_manifest(
        tmp_path / "eval-manifest.json",
        "heldout",
        [{"messages": [{"role": "user", "content": "heldout"}]}],
        "test-source",
    )
    plan = validate_sft_quality_plan(quality_plan(train, evaluation), run_root=tmp_path)
    assert plan.disjointness.overlap_count == 0
    assert plan.disjointness.train_record_count == 1
    assert len(plan.plan_hash) == 64


def test_quality_plan_rejects_content_overlap(tmp_path):
    record = {"messages": [{"role": "user", "content": "same"}]}
    train = make_manifest(tmp_path / "train-manifest.json", "train", [record], "source")
    evaluation = make_manifest(
        tmp_path / "eval-manifest.json", "heldout", [record], "source"
    )
    with pytest.raises(SFTQualityError, match="overlap"):
        validate_sft_quality_plan(quality_plan(train, evaluation), run_root=tmp_path)


def test_quality_decision_and_result_are_threshold_bound(tmp_path):
    train = make_manifest(
        tmp_path / "train-manifest.json",
        "train",
        [{"messages": [{"role": "user", "content": "train"}]}],
        "source",
    )
    evaluation = make_manifest(
        tmp_path / "eval-manifest.json",
        "heldout",
        [{"messages": [{"role": "user", "content": "heldout"}]}],
        "source",
    )
    plan = validate_sft_quality_plan(quality_plan(train, evaluation), run_root=tmp_path)
    decision = build_quality_decision(1.7, 1.4, plan.thresholds)
    assert decision["status"] == "pass"
    result = build_sft_quality_result(
        plan,
        baseline_bpb=1.7,
        final_bpb=1.4,
        eval_batches=2,
        tokenizer_hash="a" * 64,
        model_tag="quality-test",
        model_step=2,
        runtime={"python": "test"},
    )
    validate_sft_quality_result(result)
    assert result["quality_claim"] is True
    assert result["promotion"] == "not authorized"


def test_quality_plan_rejects_missing_approval(tmp_path):
    train = make_manifest(
        tmp_path / "train-manifest.json",
        "train",
        [{"messages": [{"role": "user", "content": "train"}]}],
        "source",
    )
    evaluation = make_manifest(
        tmp_path / "eval-manifest.json",
        "heldout",
        [{"messages": [{"role": "user", "content": "heldout"}]}],
        "source",
    )
    plan = quality_plan(train, evaluation)
    plan["approval"]["approved"] = False
    with pytest.raises(SFTQualityError, match="approval"):
        validate_sft_quality_plan(plan, run_root=tmp_path)


def test_quality_result_rejects_tampered_decision(tmp_path):
    train = make_manifest(
        tmp_path / "train-manifest.json",
        "train",
        [{"messages": [{"role": "user", "content": "train"}]}],
        "source",
    )
    evaluation = make_manifest(
        tmp_path / "eval-manifest.json",
        "heldout",
        [{"messages": [{"role": "user", "content": "heldout"}]}],
        "source",
    )
    plan = validate_sft_quality_plan(quality_plan(train, evaluation), run_root=tmp_path)
    result = build_sft_quality_result(
        plan,
        baseline_bpb=1.7,
        final_bpb=1.4,
        eval_batches=2,
        tokenizer_hash="a" * 64,
        model_tag="quality-test",
        model_step=2,
        runtime={"python": "test"},
    )
    result["decision"]["status"] = "stop"
    with pytest.raises(SFTQualityError, match="decision"):
        validate_sft_quality_result(result)
