from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any

from nanochat.curriculum import CurriculumScheduler, CurriculumStage
from nanochat.dynamic_context import (
    ContextProfile,
    DynamicContextPlan,
    build_context_step_plan,
    validate_resource_proof,
)
from nanochat.research_results import write_json_atomic

LONG_CONTEXT_SCHEMA_VERSION = 1
LONG_CONTEXT_BUCKETS = (2048, 8192)
LONG_CONTEXT_DEVICE_BATCH_SIZE = 1
LONG_CONTEXT_WORLD_SIZE = 1
LONG_CONTEXT_EFFECTIVE_TOKEN_BATCH = 8192
LONG_CONTEXT_OPTIMIZATION_STEPS = 2
LONG_CONTEXT_VRAM_LIMIT_BYTES = 14 * 1024**3
LONG_CONTEXT_SAFETY_MARGIN_BYTES = 256 * 1024**2
LONG_CONTEXT_MINIMUM_TOKENS_PER_SECOND = 1.0
LONG_CONTEXT_SCOPE = "bounded two-step local proof; no production promotion"


class LongContextContractError(ValueError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any, field_name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise LongContextContractError(f"{field_name} must be a positive integer")
    return int(value)


def _finite_positive(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise LongContextContractError(f"{field_name} must be positive and finite")
    return float(value)


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongContextContractError(f"{field_name} must be a non-empty string")
    return value


def _hash(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LongContextContractError(f"{field_name} must be a SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        raise LongContextContractError(f"{field_name} must be a SHA-256 hash") from exc
    return value.casefold()


def _canonical_hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LongContextContractError(
            "long-context payload must be JSON-compatible"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def build_long_context_plan() -> DynamicContextPlan:
    stages = [
        CurriculumStage(
            name="short_context",
            token_ratio=0.5,
            context_range=(2048, 2048),
            source_weights={"climbmix": 1.0},
        ),
        CurriculumStage(
            name="long_context",
            token_ratio=0.5,
            context_range=(8192, 8192),
            source_weights={"climbmix": 1.0},
        ),
    ]
    scheduler = CurriculumScheduler(
        stages,
        total_tokens=LONG_CONTEXT_OPTIMIZATION_STEPS
        * LONG_CONTEXT_EFFECTIVE_TOKEN_BATCH,
        total_steps=LONG_CONTEXT_OPTIMIZATION_STEPS,
    )
    return build_context_step_plan(
        scheduler,
        buckets=LONG_CONTEXT_BUCKETS,
        device_batch_size=LONG_CONTEXT_DEVICE_BATCH_SIZE,
        world_size=LONG_CONTEXT_WORLD_SIZE,
        effective_token_batch=LONG_CONTEXT_EFFECTIVE_TOKEN_BATCH,
    )


@dataclass(frozen=True)
class LongContextApproval:
    schema_version: int
    approval_id: str
    approved: bool
    approved_by: str
    approved_at: str
    scope: str
    plan_hash: str
    evidence_hash: str
    vram_limit_bytes: int
    safety_margin_bytes: int
    minimum_tokens_per_second: float

    def __post_init__(self):
        if self.schema_version != LONG_CONTEXT_SCHEMA_VERSION:
            raise LongContextContractError("unsupported long-context approval schema")
        _text(self.approval_id, "approval_id")
        if self.approved is not True:
            raise LongContextContractError("long-context activation is not approved")
        _text(self.approved_by, "approved_by")
        _text(self.approved_at, "approved_at")
        _text(self.scope, "scope")
        _hash(self.plan_hash, "plan_hash")
        _hash(self.evidence_hash, "evidence_hash")
        if self.vram_limit_bytes != LONG_CONTEXT_VRAM_LIMIT_BYTES:
            raise LongContextContractError("approval must use the 14 GiB VRAM envelope")
        _positive_int(self.safety_margin_bytes, "safety_margin_bytes")
        _finite_positive(self.minimum_tokens_per_second, "minimum_tokens_per_second")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "approval_id": self.approval_id,
            "approved": self.approved,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "scope": self.scope,
            "plan_hash": self.plan_hash,
            "evidence_hash": self.evidence_hash,
            "vram_limit_bytes": self.vram_limit_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "minimum_tokens_per_second": self.minimum_tokens_per_second,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LongContextApproval":
        if not isinstance(data, dict):
            raise LongContextContractError("long-context approval must be an object")
        required = {
            "schema_version",
            "approval_id",
            "approved",
            "approved_by",
            "approved_at",
            "scope",
            "plan_hash",
            "evidence_hash",
            "vram_limit_bytes",
            "safety_margin_bytes",
            "minimum_tokens_per_second",
        }
        if set(data) != required:
            raise LongContextContractError("long-context approval fields are invalid")
        return cls(**data)


@dataclass(frozen=True)
class LongContextStepEvidence:
    step: int
    sequence_length: int
    runtime_verified: bool
    model_forward_backward: bool
    loader_batch_verified: bool
    resume_model_equal: bool
    resume_optimizer_equal: bool
    resume_loader_equal: bool
    resume_next_batch_equal: bool
    loss: float
    peak_vram_bytes: int
    reserved_vram_bytes: int
    active_tokens_per_second: float
    wall_clock_seconds: float
    model_config_hash: str
    loader_state_hash: str
    next_batch_hash: str
    data_hash: str
    attention_backend: str
    dtype: str

    def __post_init__(self):
        if not _is_int(self.step) or self.step < 0:
            raise LongContextContractError("step must be a non-negative integer")
        _positive_int(self.sequence_length, "sequence_length")
        for field in (
            "runtime_verified",
            "model_forward_backward",
            "loader_batch_verified",
            "resume_model_equal",
            "resume_optimizer_equal",
            "resume_loader_equal",
            "resume_next_batch_equal",
        ):
            if not isinstance(getattr(self, field), bool):
                raise LongContextContractError(f"{field} must be boolean")
        if (
            isinstance(self.loss, bool)
            or not isinstance(self.loss, (int, float))
            or not math.isfinite(float(self.loss))
        ):
            raise LongContextContractError("loss must be finite and numeric")
        _positive_int(self.peak_vram_bytes, "peak_vram_bytes")
        _positive_int(self.reserved_vram_bytes, "reserved_vram_bytes")
        _finite_positive(self.active_tokens_per_second, "active_tokens_per_second")
        _finite_positive(self.wall_clock_seconds, "wall_clock_seconds")
        for field in (
            "model_config_hash",
            "loader_state_hash",
            "next_batch_hash",
            "data_hash",
        ):
            _hash(getattr(self, field), field)
        _text(self.attention_backend, "attention_backend")
        _text(self.dtype, "dtype")

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "sequence_length": self.sequence_length,
            "runtime_verified": self.runtime_verified,
            "model_forward_backward": self.model_forward_backward,
            "loader_batch_verified": self.loader_batch_verified,
            "resume_model_equal": self.resume_model_equal,
            "resume_optimizer_equal": self.resume_optimizer_equal,
            "resume_loader_equal": self.resume_loader_equal,
            "resume_next_batch_equal": self.resume_next_batch_equal,
            "loss": float(self.loss),
            "peak_vram_bytes": self.peak_vram_bytes,
            "reserved_vram_bytes": self.reserved_vram_bytes,
            "active_tokens_per_second": float(self.active_tokens_per_second),
            "wall_clock_seconds": float(self.wall_clock_seconds),
            "model_config_hash": self.model_config_hash,
            "loader_state_hash": self.loader_state_hash,
            "next_batch_hash": self.next_batch_hash,
            "data_hash": self.data_hash,
            "attention_backend": self.attention_backend,
            "dtype": self.dtype,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "LongContextStepEvidence":
        if not isinstance(data, dict):
            raise LongContextContractError("step evidence must be an object")
        required = {
            "step",
            "sequence_length",
            "runtime_verified",
            "model_forward_backward",
            "loader_batch_verified",
            "resume_model_equal",
            "resume_optimizer_equal",
            "resume_loader_equal",
            "resume_next_batch_equal",
            "loss",
            "peak_vram_bytes",
            "reserved_vram_bytes",
            "active_tokens_per_second",
            "wall_clock_seconds",
            "model_config_hash",
            "loader_state_hash",
            "next_batch_hash",
            "data_hash",
            "attention_backend",
            "dtype",
        }
        if set(data) != required:
            raise LongContextContractError("step evidence fields are invalid")
        return cls(**data)


def evidence_hash(evidence: Any) -> str:
    if not isinstance(evidence, (list, tuple)):
        raise LongContextContractError("evidence must be a sequence")
    records = [
        item.to_dict() if isinstance(item, LongContextStepEvidence) else item
        for item in evidence
    ]
    return _canonical_hash(records)


def build_long_context_approval(
    plan: DynamicContextPlan,
    evidence: list[LongContextStepEvidence] | tuple[LongContextStepEvidence, ...],
    *,
    approval_id: str,
    approved_by: str,
    approved_at: str,
) -> LongContextApproval:
    if plan.schedule_hash != build_long_context_plan().schedule_hash:
        raise LongContextContractError(
            "approval plan is not the approved long-context plan"
        )
    records = tuple(
        (
            item
            if isinstance(item, LongContextStepEvidence)
            else LongContextStepEvidence.from_dict(item)
        )
        for item in evidence
    )
    validate_long_context_evidence(plan, records)
    return LongContextApproval(
        schema_version=LONG_CONTEXT_SCHEMA_VERSION,
        approval_id=approval_id,
        approved=True,
        approved_by=approved_by,
        approved_at=approved_at,
        scope=LONG_CONTEXT_SCOPE,
        plan_hash=plan.schedule_hash,
        evidence_hash=evidence_hash(records),
        vram_limit_bytes=LONG_CONTEXT_VRAM_LIMIT_BYTES,
        safety_margin_bytes=LONG_CONTEXT_SAFETY_MARGIN_BYTES,
        minimum_tokens_per_second=LONG_CONTEXT_MINIMUM_TOKENS_PER_SECOND,
    )


def validate_long_context_evidence(
    plan: DynamicContextPlan,
    evidence: (
        list[LongContextStepEvidence] | tuple[LongContextStepEvidence, ...] | list[dict]
    ),
    approval: LongContextApproval | None = None,
) -> tuple[LongContextStepEvidence, ...]:
    if not isinstance(plan, DynamicContextPlan):
        raise LongContextContractError("a dynamic context plan is required")
    if plan.to_dict() != build_long_context_plan().to_dict():
        raise LongContextContractError("plan is not the approved 2048 to 8192 profile")
    if not isinstance(evidence, (list, tuple)):
        raise LongContextContractError("evidence must be a sequence")
    records = tuple(
        (
            item
            if isinstance(item, LongContextStepEvidence)
            else LongContextStepEvidence.from_dict(item)
        )
        for item in evidence
    )
    if len(records) != len(plan.steps):
        raise LongContextContractError("evidence must contain one record per plan step")
    by_step = {record.step: record for record in records}
    if set(by_step) != set(range(len(plan.steps))):
        raise LongContextContractError("evidence steps are incomplete")
    ordered = tuple(by_step[index] for index in range(len(plan.steps)))
    resource_records = tuple(
        {
            "sequence_length": record.sequence_length,
            "peak_vram_bytes": record.peak_vram_bytes,
            "active_tokens_per_second": record.active_tokens_per_second,
            "wall_clock_seconds": record.wall_clock_seconds,
            "runtime_verified": record.runtime_verified,
        }
        for record in ordered
    )
    validate_resource_proof(
        plan,
        resource_records,
        LONG_CONTEXT_VRAM_LIMIT_BYTES,
        LONG_CONTEXT_MINIMUM_TOKENS_PER_SECOND,
        LONG_CONTEXT_SAFETY_MARGIN_BYTES,
    )
    for record, step in zip(ordered, plan.steps):
        if record.sequence_length != step.sequence_length:
            raise LongContextContractError("evidence context does not match the plan")
        if (
            record.reserved_vram_bytes + LONG_CONTEXT_SAFETY_MARGIN_BYTES
            > LONG_CONTEXT_VRAM_LIMIT_BYTES
        ):
            raise LongContextContractError(
                "reserved VRAM evidence exceeds the 14 GiB envelope"
            )
        if not all(
            getattr(record, field)
            for field in (
                "runtime_verified",
                "model_forward_backward",
                "loader_batch_verified",
                "resume_model_equal",
                "resume_optimizer_equal",
                "resume_loader_equal",
                "resume_next_batch_equal",
            )
        ):
            raise LongContextContractError(
                "evidence contains an unverified runtime claim"
            )
    if approval is not None:
        if not isinstance(approval, LongContextApproval):
            raise LongContextContractError("approval must be a LongContextApproval")
        if approval.plan_hash != plan.schedule_hash:
            raise LongContextContractError("approval plan hash does not match")
        if approval.evidence_hash != evidence_hash(ordered):
            raise LongContextContractError("approval evidence hash does not match")
    return ordered


def approved_context_profile(
    plan: DynamicContextPlan,
    evidence: (
        list[LongContextStepEvidence] | tuple[LongContextStepEvidence, ...] | list[dict]
    ),
    approval: LongContextApproval,
) -> ContextProfile:
    validate_long_context_evidence(plan, evidence, approval)
    return ContextProfile(
        mode="dynamic",
        sequence_length=LONG_CONTEXT_BUCKETS[-1],
        buckets=LONG_CONTEXT_BUCKETS,
        enabled=True,
    )


def write_long_context_approval(
    path: str | os.PathLike, approval: LongContextApproval
) -> None:
    if not isinstance(approval, LongContextApproval):
        raise LongContextContractError("approval is invalid")
    write_json_atomic(path, approval.to_dict())


def load_long_context_approval(path: str | os.PathLike) -> LongContextApproval:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LongContextContractError(
            "long-context approval is not valid JSON"
        ) from exc
    return LongContextApproval.from_dict(payload)


build_long_context_step_plan = build_long_context_plan
validate_long_context_proof = validate_long_context_evidence
build_activation_approval = build_long_context_approval
