from __future__ import annotations

import copy
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanochat.research_results import write_json_atomic
from nanochat.sft_manifest import (
    SFTContractError,
    SFTManifestError,
    ValidatedDatasetManifest,
    canonical_json_hash,
    read_jsonl_records,
    resolve_run_local_path,
    validate_jsonl_dataset_manifest,
)

SFT_QUALITY_PLAN_SCHEMA_VERSION = 1
SFT_QUALITY_RESULT_SCHEMA_VERSION = 1
SFT_CONTEXT_TOKENS = 2048
SFT_QUALITY_DECISION_STATUSES = frozenset({"pass", "stop", "inconclusive"})


class SFTQualityError(ValueError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any, field_name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise SFTQualityError(f"{field_name} must be a positive integer")
    return int(value)


def _finite_float(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SFTQualityError(f"{field_name} must be finite and numeric")
    return float(value)


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SFTQualityError(f"{field_name} must be a non-empty string")
    return value


def _json_copy(value: Any, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise SFTQualityError(f"{field_name} must be JSON-compatible") from exc


def _load_json(path: str | os.PathLike) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SFTQualityError(f"quality plan is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SFTQualityError("quality plan must be an object")
    return payload


def _canonical_path(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))


def _conversation_for_fingerprint(record: dict, location: str) -> dict:
    if "conversation" in record:
        raw = record["conversation"]
    elif "messages" in record:
        raw = {"messages": record["messages"]}
    else:
        raise SFTQualityError(f"{location} has no conversation")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise SFTQualityError(f"{location} has invalid conversation JSON") from exc
    if isinstance(raw, list):
        raw = {"messages": raw}
    if not isinstance(raw, dict) or "messages" not in raw:
        raise SFTQualityError(f"{location} has no messages")
    messages = raw["messages"]
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (TypeError, ValueError) as exc:
            raise SFTQualityError(f"{location} has invalid messages JSON") from exc
    if not isinstance(messages, list) or not messages:
        raise SFTQualityError(f"{location} must contain messages")
    normalized = copy.deepcopy(raw)
    normalized["messages"] = copy.deepcopy(messages)
    return normalized


def manifest_record_fingerprints(manifest: ValidatedDatasetManifest) -> tuple[str, ...]:
    fingerprints = []
    for dataset_file in manifest.files:
        records = read_jsonl_records(dataset_file.path, manifest.run_root)
        for line_number, record in enumerate(records, 1):
            location = f"{dataset_file.relative_path}:{line_number}"
            fingerprints.append(
                canonical_json_hash(_conversation_for_fingerprint(record, location))
            )
    return tuple(fingerprints)


@dataclass(frozen=True)
class SFTDisjointnessEvidence:
    train_manifest_hash: str
    eval_manifest_hash: str
    train_file_hashes: tuple[str, ...]
    eval_file_hashes: tuple[str, ...]
    train_record_count: int
    eval_record_count: int
    overlap_count: int

    def to_dict(self) -> dict:
        return {
            "train_manifest_hash": self.train_manifest_hash,
            "eval_manifest_hash": self.eval_manifest_hash,
            "train_file_hashes": list(self.train_file_hashes),
            "eval_file_hashes": list(self.eval_file_hashes),
            "train_record_count": self.train_record_count,
            "eval_record_count": self.eval_record_count,
            "overlap_count": self.overlap_count,
        }


def validate_disjoint_sft_manifests(
    train_manifest: ValidatedDatasetManifest,
    eval_manifest: ValidatedDatasetManifest,
) -> SFTDisjointnessEvidence:
    if not isinstance(train_manifest, ValidatedDatasetManifest):
        raise SFTQualityError("train_manifest must be validated")
    if not isinstance(eval_manifest, ValidatedDatasetManifest):
        raise SFTQualityError("eval_manifest must be validated")
    if train_manifest.manifest_hash == eval_manifest.manifest_hash:
        raise SFTQualityError("train and evaluation manifests must be different")
    train_files = {_canonical_path(item.path) for item in train_manifest.files}
    eval_files = {_canonical_path(item.path) for item in eval_manifest.files}
    if train_files & eval_files:
        raise SFTQualityError("train and evaluation manifests share a dataset file")
    train_fingerprints = manifest_record_fingerprints(train_manifest)
    eval_fingerprints = manifest_record_fingerprints(eval_manifest)
    overlap = set(train_fingerprints) & set(eval_fingerprints)
    if overlap:
        raise SFTQualityError(
            f"train and evaluation manifests overlap on {len(overlap)} records"
        )
    return SFTDisjointnessEvidence(
        train_manifest_hash=train_manifest.manifest_hash,
        eval_manifest_hash=eval_manifest.manifest_hash,
        train_file_hashes=tuple(item.sha256 for item in train_manifest.files),
        eval_file_hashes=tuple(item.sha256 for item in eval_manifest.files),
        train_record_count=len(train_fingerprints),
        eval_record_count=len(eval_fingerprints),
        overlap_count=0,
    )


def _validated_manifest(
    value: Any,
    run_root: str | os.PathLike,
    field_name: str,
) -> ValidatedDatasetManifest:
    if isinstance(value, ValidatedDatasetManifest):
        return value
    if not isinstance(value, (str, os.PathLike)):
        raise SFTQualityError(f"{field_name} must be a path")
    try:
        path = resolve_run_local_path(value, run_root, require_exists=True)
        return validate_jsonl_dataset_manifest(path, run_root=run_root)
    except (SFTContractError, SFTManifestError) as exc:
        raise SFTQualityError(f"{field_name} is invalid: {exc}") from exc


@dataclass(frozen=True)
class ValidatedSFTQualityPlan:
    payload: dict
    plan_hash: str
    train_manifest: ValidatedDatasetManifest
    eval_manifest: ValidatedDatasetManifest
    disjointness: SFTDisjointnessEvidence

    @property
    def run_id(self) -> str:
        return self.payload["run_id"]

    @property
    def budget(self) -> dict:
        return copy.deepcopy(self.payload["budget"])

    @property
    def thresholds(self) -> dict:
        return copy.deepcopy(self.payload["thresholds"])


def _normalise_plan_payload(payload: dict) -> dict:
    normalized = copy.deepcopy(payload)
    if (
        "quality_plan_schema_version" not in normalized
        and "schema_version" in normalized
    ):
        normalized["quality_plan_schema_version"] = normalized["schema_version"]
    normalized.pop("schema_version", None)
    if "train_manifest" not in normalized and "train_manifest_path" in normalized:
        normalized["train_manifest"] = normalized["train_manifest_path"]
    if "eval_manifest" not in normalized and "eval_manifest_path" in normalized:
        normalized["eval_manifest"] = normalized["eval_manifest_path"]
    normalized.pop("train_manifest_path", None)
    normalized.pop("eval_manifest_path", None)
    if "plan_hash" in normalized:
        normalized.pop("plan_hash")
    return normalized


def validate_sft_quality_plan(
    plan: dict | str | os.PathLike,
    run_root: str | os.PathLike | None = None,
    *,
    train_manifest: ValidatedDatasetManifest | None = None,
    eval_manifest: ValidatedDatasetManifest | None = None,
) -> ValidatedSFTQualityPlan:
    payload = (
        _load_json(plan)
        if isinstance(plan, (str, os.PathLike))
        else copy.deepcopy(plan)
    )
    if not isinstance(payload, dict):
        raise SFTQualityError("quality plan must be an object")
    payload = _normalise_plan_payload(payload)
    required = {
        "quality_plan_schema_version",
        "run_id",
        "train_manifest",
        "eval_manifest",
        "approval",
        "budget",
        "thresholds",
        "provenance",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise SFTQualityError(f"quality plan is missing fields: {missing}")
    if payload["quality_plan_schema_version"] != SFT_QUALITY_PLAN_SCHEMA_VERSION:
        raise SFTQualityError("unsupported quality plan schema version")
    run_id = _text(payload["run_id"], "run_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise SFTQualityError("run_id must be a safe path component")
    if run_root is None:
        if train_manifest is not None:
            run_root = train_manifest.run_root
        else:
            run_root = Path(
                os.path.expanduser(os.fspath(payload["train_manifest"]))
            ).parent
    run_root = Path(os.path.expanduser(os.fspath(run_root)))
    train = _validated_manifest(
        train_manifest if train_manifest is not None else payload["train_manifest"],
        run_root,
        "train_manifest",
    )
    evaluation = _validated_manifest(
        eval_manifest if eval_manifest is not None else payload["eval_manifest"],
        run_root,
        "eval_manifest",
    )
    if Path(os.path.realpath(train.manifest_path)) == Path(
        os.path.realpath(evaluation.manifest_path)
    ):
        raise SFTQualityError("train and evaluation manifests must be different")
    disjointness = validate_disjoint_sft_manifests(train, evaluation)
    approval = payload["approval"]
    if not isinstance(approval, dict):
        raise SFTQualityError("approval must be an object")
    if approval.get("approved") is not True or approval.get("reviewed") is not True:
        raise SFTQualityError("quality plan requires explicit approval")
    _text(approval.get("approved_by"), "approval.approved_by")
    _text(approval.get("approved_at"), "approval.approved_at")
    _text(approval.get("scope"), "approval.scope")
    budget = payload["budget"]
    if not isinstance(budget, dict):
        raise SFTQualityError("budget must be an object")
    context_length = _positive_int(
        budget.get("context_length", SFT_CONTEXT_TOKENS), "budget.context_length"
    )
    if context_length != SFT_CONTEXT_TOKENS:
        raise SFTQualityError("quality SFT context must remain fixed at 2048")
    device_batch_size = _positive_int(
        budget.get("device_batch_size"), "budget.device_batch_size"
    )
    world_size = _positive_int(budget.get("world_size"), "budget.world_size")
    effective_tokens = _positive_int(
        budget.get("effective_tokens"), "budget.effective_tokens"
    )
    optimization_steps = _positive_int(
        budget.get("optimization_steps"), "budget.optimization_steps"
    )
    eval_tokens = _positive_int(budget.get("eval_tokens"), "budget.eval_tokens")
    eval_batch_size = _positive_int(
        budget.get("eval_batch_size"), "budget.eval_batch_size"
    )
    gradient_accumulation_steps = _positive_int(
        budget.get("gradient_accumulation_steps", 1),
        "budget.gradient_accumulation_steps",
    )
    expected_effective = (
        device_batch_size * context_length * world_size * gradient_accumulation_steps
    )
    if expected_effective != effective_tokens:
        raise SFTQualityError("budget effective token arithmetic is invalid")
    if "total_batch_size" in budget and budget["total_batch_size"] != effective_tokens:
        raise SFTQualityError("budget total_batch_size is inconsistent")
    if "train_tokens" in budget and budget["train_tokens"] != (
        optimization_steps * effective_tokens
    ):
        raise SFTQualityError("budget train_tokens is inconsistent")
    evaluation_batch_tokens = eval_batch_size * context_length * world_size
    if eval_tokens % evaluation_batch_tokens:
        raise SFTQualityError(
            "evaluation token budget must divide the evaluation batch"
        )
    thresholds = payload["thresholds"]
    if not isinstance(thresholds, dict):
        raise SFTQualityError("thresholds must be an object")
    pass_max_bpb = _finite_float(
        thresholds.get("pass_max_bpb"), "thresholds.pass_max_bpb"
    )
    stop_min_bpb = _finite_float(
        thresholds.get("stop_min_bpb"), "thresholds.stop_min_bpb"
    )
    min_improvement = _finite_float(
        thresholds.get("min_improvement", 0.0), "thresholds.min_improvement"
    )
    if pass_max_bpb <= 0 or stop_min_bpb <= 0:
        raise SFTQualityError("BPB thresholds must be positive")
    if pass_max_bpb >= stop_min_bpb:
        raise SFTQualityError("pass threshold must be below stop threshold")
    if min_improvement < 0:
        raise SFTQualityError("minimum improvement must be non-negative")
    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise SFTQualityError("provenance must be an object")
    _text(provenance.get("source"), "provenance.source")
    _text(provenance.get("revision"), "provenance.revision")
    _text(provenance.get("split_method"), "provenance.split_method")
    if provenance.get("license_review") is not True:
        raise SFTQualityError("license review must be explicitly approved")
    normalized = _json_copy(payload, "quality plan")
    normalized["train_manifest"] = str(train.manifest_path)
    normalized["eval_manifest"] = str(evaluation.manifest_path)
    plan_hash = canonical_json_hash(normalized)
    return ValidatedSFTQualityPlan(
        payload=normalized,
        plan_hash=plan_hash,
        train_manifest=train,
        eval_manifest=evaluation,
        disjointness=disjointness,
    )


def load_sft_quality_plan(
    path: str | os.PathLike,
    run_root: str | os.PathLike | None = None,
) -> ValidatedSFTQualityPlan:
    return validate_sft_quality_plan(path, run_root=run_root)


def write_sft_quality_plan(
    path: str | os.PathLike,
    payload: dict,
    run_root: str | os.PathLike,
) -> ValidatedSFTQualityPlan:
    resolved = resolve_run_local_path(path, run_root)
    write_json_atomic(resolved, payload)
    return validate_sft_quality_plan(resolved, run_root=run_root)


def build_quality_decision(
    baseline_bpb: float,
    final_bpb: float,
    thresholds: dict,
) -> dict:
    baseline = _finite_float(baseline_bpb, "baseline_bpb")
    final = _finite_float(final_bpb, "final_bpb")
    pass_max = _finite_float(thresholds.get("pass_max_bpb"), "pass_max_bpb")
    stop_min = _finite_float(thresholds.get("stop_min_bpb"), "stop_min_bpb")
    min_improvement = _finite_float(
        thresholds.get("min_improvement", 0.0), "min_improvement"
    )
    if pass_max <= 0 or stop_min <= 0 or pass_max >= stop_min:
        raise SFTQualityError("quality thresholds are invalid")
    if min_improvement < 0:
        raise SFTQualityError("minimum improvement is invalid")
    improvement = baseline - final
    if final <= pass_max and improvement >= min_improvement:
        status = "pass"
        reason = "final held-out BPB meets the pass threshold"
    elif final >= stop_min:
        status = "stop"
        reason = "final held-out BPB meets the stop threshold"
    else:
        status = "inconclusive"
        reason = "final held-out BPB is between the predeclared thresholds"
    return {
        "status": status,
        "reason": reason,
        "baseline_bpb": baseline,
        "final_bpb": final,
        "improvement_bpb": improvement,
        "thresholds": {
            "pass_max_bpb": pass_max,
            "stop_min_bpb": stop_min,
            "min_improvement": min_improvement,
        },
    }


def build_sft_quality_result(
    plan: ValidatedSFTQualityPlan,
    *,
    baseline_bpb: float,
    final_bpb: float,
    eval_batches: int,
    tokenizer_hash: str,
    model_tag: str | None,
    model_step: int | None,
    runtime: dict,
    clean_evaluation: bool = True,
) -> dict:
    if not isinstance(plan, ValidatedSFTQualityPlan):
        raise SFTQualityError("a validated quality plan is required")
    if not _is_int(eval_batches) or eval_batches <= 0:
        raise SFTQualityError("eval_batches must be positive")
    if not isinstance(tokenizer_hash, str) or len(tokenizer_hash) != 64:
        raise SFTQualityError("tokenizer_hash is invalid")
    if not isinstance(runtime, dict):
        raise SFTQualityError("runtime metadata must be an object")
    decision = build_quality_decision(baseline_bpb, final_bpb, plan.thresholds)
    payload = {
        "schema_version": SFT_QUALITY_RESULT_SCHEMA_VERSION,
        "status": "complete",
        "run_id": plan.run_id,
        "quality_plan_hash": plan.plan_hash,
        "train_manifest_hash": plan.train_manifest.manifest_hash,
        "eval_manifest_hash": plan.eval_manifest.manifest_hash,
        "disjointness": plan.disjointness.to_dict(),
        "tokenizer_hash": tokenizer_hash,
        "model_tag": model_tag,
        "model_step": model_step,
        "metrics": {
            "baseline_heldout_bpb": decision["baseline_bpb"],
            "final_heldout_bpb": decision["final_bpb"],
            "improvement_bpb": decision["improvement_bpb"],
        },
        "evaluation": {
            "metric": "heldout_bpb",
            "eval_batches": int(eval_batches),
            "eval_tokens": int(plan.budget["eval_tokens"]),
            "clean_final_checkpoint": bool(clean_evaluation),
        },
        "thresholds": plan.thresholds,
        "decision": decision,
        "runtime": _json_copy(runtime, "runtime"),
        "promotion": "not authorized",
        "quality_claim": decision["status"] == "pass",
    }
    validate_sft_quality_result(payload)
    return payload


def validate_sft_quality_result(result: dict) -> None:
    if not isinstance(result, dict):
        raise SFTQualityError("quality result must be an object")
    if result.get("schema_version") != SFT_QUALITY_RESULT_SCHEMA_VERSION:
        raise SFTQualityError("unsupported quality result schema version")
    if result.get("status") != "complete":
        raise SFTQualityError("quality result is not complete")
    if result.get("promotion") != "not authorized":
        raise SFTQualityError("quality result must not authorize promotion")
    for field in (
        "quality_plan_hash",
        "train_manifest_hash",
        "eval_manifest_hash",
        "tokenizer_hash",
    ):
        value = result.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise SFTQualityError(f"quality result {field} is invalid")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise SFTQualityError("quality result metrics are invalid")
    baseline = _finite_float(
        metrics.get("baseline_heldout_bpb"), "baseline_heldout_bpb"
    )
    final = _finite_float(metrics.get("final_heldout_bpb"), "final_heldout_bpb")
    improvement = _finite_float(metrics.get("improvement_bpb"), "improvement_bpb")
    if not math.isclose(improvement, baseline - final, rel_tol=0.0, abs_tol=1e-12):
        raise SFTQualityError("quality result improvement is inconsistent")
    evaluation = result.get("evaluation")
    if not isinstance(evaluation, dict):
        raise SFTQualityError("quality result evaluation is invalid")
    if evaluation.get("clean_final_checkpoint") is not True:
        raise SFTQualityError("quality result must use the final checkpoint")
    if not _is_int(evaluation.get("eval_batches")) or evaluation["eval_batches"] <= 0:
        raise SFTQualityError("quality result evaluation budget is invalid")
    if not _is_int(evaluation.get("eval_tokens")) or evaluation["eval_tokens"] <= 0:
        raise SFTQualityError("quality result token budget is invalid")
    thresholds = result.get("thresholds")
    decision = result.get("decision")
    if not isinstance(thresholds, dict) or not isinstance(decision, dict):
        raise SFTQualityError("quality result decision is invalid")
    expected = build_quality_decision(baseline, final, thresholds)
    if decision.get("status") not in SFT_QUALITY_DECISION_STATUSES:
        raise SFTQualityError("quality result decision status is invalid")
    for field in ("status", "reason", "baseline_bpb", "final_bpb", "improvement_bpb"):
        if decision.get(field) != expected[field]:
            raise SFTQualityError(f"quality result decision field is invalid: {field}")
    if result.get("quality_claim") is not (decision["status"] == "pass"):
        raise SFTQualityError("quality result claim does not match decision")


def write_sft_quality_result(path: str | os.PathLike, result: dict) -> None:
    validate_sft_quality_result(result)
    write_json_atomic(path, result)


validate_quality_plan = validate_sft_quality_plan
load_quality_plan = load_sft_quality_plan
write_quality_plan = write_sft_quality_plan
validate_disjoint_manifests = validate_disjoint_sft_manifests
build_quality_result = build_sft_quality_result
validate_quality_result = validate_sft_quality_result
write_quality_result = write_sft_quality_result
