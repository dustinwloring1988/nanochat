import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from nanochat.curriculum import CurriculumScheduler

FIXED_CONTEXT_TOKENS = 2048
DYNAMIC_CONTEXT_ENABLED = False
DYNAMIC_CONTEXT_GATE_SCHEMA_VERSION = 1
DYNAMIC_LOADER_STATE_SCHEMA_VERSION = 2


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any, field_name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _json_copy(value: Any, field_name: str):
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-compatible") from exc
    return json.loads(encoded)


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_bucket_sequence(buckets: Any, require_fixed: bool) -> tuple[int, ...]:
    if not isinstance(buckets, (list, tuple)) or not buckets:
        raise ValueError("buckets must be a non-empty sequence")
    values = tuple(_positive_int(value, "bucket") for value in buckets)
    if tuple(sorted(set(values))) != values:
        raise ValueError("buckets must be strictly increasing and unique")
    if require_fixed and FIXED_CONTEXT_TOKENS not in values:
        raise ValueError(
            f"buckets must include the fixed {FIXED_CONTEXT_TOKENS} bucket"
        )
    return values


def _validate_buckets(buckets: Any) -> tuple[int, ...]:
    return _validate_bucket_sequence(buckets, require_fixed=True)


@dataclass(frozen=True)
class ContextStep:
    step: int
    stage_index: int
    sequence_length: int
    gradient_accumulation_steps: int
    effective_tokens: int

    def __post_init__(self):
        for field_name in (
            "step",
            "stage_index",
            "sequence_length",
            "gradient_accumulation_steps",
            "effective_tokens",
        ):
            value = getattr(self, field_name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.effective_tokens <= 0:
            raise ValueError("effective_tokens must be positive")

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "stage_index": self.stage_index,
            "sequence_length": self.sequence_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_tokens": self.effective_tokens,
        }


@dataclass(frozen=True)
class DynamicContextPlan:
    schema_version: int
    buckets: tuple[int, ...]
    device_batch_size: int
    world_size: int
    effective_token_batch: int
    token_horizon: int
    steps: tuple[ContextStep, ...]
    schedule_hash: str

    def __post_init__(self):
        if (
            not _is_int(self.schema_version)
            or self.schema_version != DYNAMIC_CONTEXT_GATE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported dynamic context gate schema version")
        object.__setattr__(self, "buckets", _validate_buckets(self.buckets))
        object.__setattr__(
            self,
            "device_batch_size",
            _positive_int(self.device_batch_size, "device_batch_size"),
        )
        object.__setattr__(
            self, "world_size", _positive_int(self.world_size, "world_size")
        )
        object.__setattr__(
            self,
            "effective_token_batch",
            _positive_int(self.effective_token_batch, "effective_token_batch"),
        )
        object.__setattr__(
            self,
            "token_horizon",
            _positive_int(self.token_horizon, "token_horizon"),
        )
        if not isinstance(self.steps, tuple) or not self.steps:
            raise ValueError("steps must be a non-empty tuple")
        if any(not isinstance(step, ContextStep) for step in self.steps):
            raise ValueError("steps contains an invalid step")
        if tuple(step.step for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("steps must be contiguous and start at zero")
        stage_indices = [step.stage_index for step in self.steps]
        if any(right < left for left, right in zip(stage_indices, stage_indices[1:])):
            raise ValueError("stage indices must not decrease")
        if len(self.steps) * self.effective_token_batch != self.token_horizon:
            raise ValueError("planned steps do not equal the declared token horizon")
        for step in self.steps:
            if step.sequence_length not in self.buckets:
                raise ValueError("step uses a context outside the approved buckets")
            denominator = (
                self.device_batch_size * step.sequence_length * self.world_size
            )
            if self.effective_token_batch % denominator:
                raise ValueError(
                    "effective token batch is not divisible by bucket batch"
                )
            if (
                self.effective_token_batch // denominator
                != step.gradient_accumulation_steps
            ):
                raise ValueError("gradient accumulation does not match effective batch")
        if not isinstance(self.schedule_hash, str) or len(self.schedule_hash) != 64:
            raise ValueError("dynamic context schedule hash is invalid")
        expected_hash = _canonical_hash(self._hash_payload())
        if self.schedule_hash != expected_hash:
            raise ValueError("dynamic context schedule hash is invalid")

    def _hash_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "buckets": list(self.buckets),
            "device_batch_size": self.device_batch_size,
            "world_size": self.world_size,
            "effective_token_batch": self.effective_token_batch,
            "token_horizon": self.token_horizon,
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_dict(self) -> dict:
        payload = self._hash_payload()
        payload["schedule_hash"] = self.schedule_hash
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> "DynamicContextPlan":
        if not isinstance(data, dict):
            raise ValueError("dynamic context plan must be a mapping")
        required = {
            "schema_version",
            "buckets",
            "device_batch_size",
            "world_size",
            "effective_token_batch",
            "token_horizon",
            "steps",
            "schedule_hash",
        }
        if set(data) != required:
            raise ValueError("dynamic context plan fields are invalid")
        try:
            steps = tuple(
                ContextStep(
                    step=step["step"],
                    stage_index=step["stage_index"],
                    sequence_length=step["sequence_length"],
                    gradient_accumulation_steps=step["gradient_accumulation_steps"],
                    effective_tokens=step["effective_tokens"],
                )
                for step in data["steps"]
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("dynamic context plan contains an invalid step") from exc
        return cls(
            schema_version=data["schema_version"],
            buckets=tuple(data["buckets"]),
            device_batch_size=data["device_batch_size"],
            world_size=data["world_size"],
            effective_token_batch=data["effective_token_batch"],
            token_horizon=data["token_horizon"],
            steps=steps,
            schedule_hash=data["schedule_hash"],
        )


def build_context_step_plan(
    scheduler: CurriculumScheduler,
    buckets: Any,
    device_batch_size: int,
    world_size: int,
    effective_token_batch: int,
) -> DynamicContextPlan:
    if not isinstance(scheduler, CurriculumScheduler):
        raise ValueError("scheduler must be a CurriculumScheduler")
    approved_buckets = _validate_buckets(buckets)
    device_batch_size = _positive_int(device_batch_size, "device_batch_size")
    world_size = _positive_int(world_size, "world_size")
    effective_token_batch = _positive_int(
        effective_token_batch, "effective_token_batch"
    )
    if scheduler.total_steps is None:
        raise ValueError("dynamic context planning requires step-based scheduling")
    if scheduler.total_steps <= 0:
        raise ValueError("scheduler must contain at least one step")
    if scheduler.total_tokens != effective_token_batch * scheduler.total_steps:
        raise ValueError("scheduler token horizon does not match effective batch plan")
    steps = []
    for step in range(scheduler.total_steps):
        stage_index, _ = scheduler.get_stage_progress(step=step)
        sequence_length = scheduler.get_context_length(step=step)
        if sequence_length not in approved_buckets:
            raise ValueError(
                f"scheduler produced non-discrete context {sequence_length} at step {step}"
            )
        denominator = device_batch_size * sequence_length * world_size
        if effective_token_batch % denominator:
            raise ValueError(
                f"effective token batch is not divisible by context {sequence_length}"
            )
        steps.append(
            ContextStep(
                step=step,
                stage_index=stage_index,
                sequence_length=sequence_length,
                gradient_accumulation_steps=effective_token_batch // denominator,
                effective_tokens=effective_token_batch,
            )
        )
    if {step.stage_index for step in steps} != set(range(len(scheduler.stages))):
        raise ValueError("every curriculum stage must receive at least one step")
    payload = {
        "schema_version": DYNAMIC_CONTEXT_GATE_SCHEMA_VERSION,
        "buckets": list(approved_buckets),
        "device_batch_size": device_batch_size,
        "world_size": world_size,
        "effective_token_batch": effective_token_batch,
        "token_horizon": scheduler.total_tokens,
        "steps": [step.to_dict() for step in steps],
    }
    plan_payload = {
        **payload,
        "steps": tuple(steps),
    }
    return DynamicContextPlan(
        **plan_payload,
        schedule_hash=_canonical_hash(payload),
    )


@dataclass(frozen=True)
class ContextProfile:
    mode: str
    sequence_length: int
    buckets: tuple[int, ...]
    enabled: bool


@dataclass(frozen=True)
class ContextFallback:
    profile: ContextProfile
    requested_buckets: tuple[int, ...]
    fallback_used: bool
    reason: str


def fixed_context_profile() -> ContextProfile:
    return ContextProfile(
        mode="fixed",
        sequence_length=FIXED_CONTEXT_TOKENS,
        buckets=(FIXED_CONTEXT_TOKENS,),
        enabled=False,
    )


def resolve_context_request(
    requested_buckets: Any = None,
    *,
    plan: Any = None,
    evidence: Any = None,
    approval: Any = None,
) -> ContextFallback:
    if requested_buckets is None:
        requested = (FIXED_CONTEXT_TOKENS,)
    else:
        requested = _validate_bucket_sequence(requested_buckets, require_fixed=False)
    if plan is not None and evidence is not None and approval is not None:
        from nanochat.long_context import approved_context_profile

        if requested == (2048, 8192):
            return ContextFallback(
                profile=approved_context_profile(plan, evidence, approval),
                requested_buckets=requested,
                fallback_used=False,
                reason="explicit_activation_approved",
            )
    if requested == (FIXED_CONTEXT_TOKENS,):
        return ContextFallback(
            profile=fixed_context_profile(),
            requested_buckets=requested,
            fallback_used=False,
            reason="fixed_context_request",
        )
    return ContextFallback(
        profile=fixed_context_profile(),
        requested_buckets=requested,
        fallback_used=True,
        reason="dynamic_context_not_approved",
    )


_REQUIRED_DYNAMIC_LOADER_FIELDS = {
    "source_names",
    "source_weights",
    "subset_names",
    "subset_weights",
    "source_states",
    "document_buffers",
    "sampler_state",
    "subset_sampler_states",
    "source_token_counts",
    "source_document_counts",
    "pending_batch",
}


def make_dynamic_loader_resume_state(
    plan: DynamicContextPlan,
    active_step: int,
    loader_state: dict,
) -> dict:
    if not isinstance(loader_state, dict):
        raise ValueError("loader_state must be a mapping")
    copied = _json_copy(loader_state, "loader_state")
    if (
        "schema_version" in copied
        and copied["schema_version"] != DYNAMIC_LOADER_STATE_SCHEMA_VERSION
    ):
        raise ValueError("loader_state uses an incompatible schema version")
    missing = sorted(_REQUIRED_DYNAMIC_LOADER_FIELDS - set(copied))
    if missing:
        raise ValueError(f"loader_state is missing fields: {missing}")
    if not _is_int(active_step) or not 0 <= active_step < len(plan.steps):
        raise ValueError("active_step is outside the context plan")
    copied["schema_version"] = DYNAMIC_LOADER_STATE_SCHEMA_VERSION
    copied["context_schedule_hash"] = plan.schedule_hash
    copied["active_step"] = active_step
    copied["active_bucket"] = plan.steps[active_step].sequence_length
    copied["active_stage_index"] = plan.steps[active_step].stage_index
    copied["next_bucket"] = (
        plan.steps[active_step + 1].sequence_length
        if active_step + 1 < len(plan.steps)
        else None
    )
    copied["accumulated_tokens"] = active_step * plan.effective_token_batch
    validate_dynamic_loader_resume_state(copied, plan)
    return copied


def validate_dynamic_loader_resume_state(
    state: Any,
    plan: DynamicContextPlan,
    expected_active_step: int | None = None,
) -> None:
    if not isinstance(state, dict):
        raise ValueError("dynamic loader state must be a mapping")
    if state.get("schema_version") != DYNAMIC_LOADER_STATE_SCHEMA_VERSION:
        raise ValueError("dynamic loader state schema version is incompatible")
    missing = sorted(_REQUIRED_DYNAMIC_LOADER_FIELDS - set(state))
    if missing:
        raise ValueError(f"dynamic loader state is missing fields: {missing}")
    if state.get("context_schedule_hash") != plan.schedule_hash:
        raise ValueError("dynamic loader state schedule hash is incompatible")
    active_step = state.get("active_step")
    if not _is_int(active_step) or not 0 <= active_step < len(plan.steps):
        raise ValueError("dynamic loader state active step is invalid")
    if expected_active_step is not None and active_step != expected_active_step:
        raise ValueError("dynamic loader state active step is incompatible")
    expected_bucket = plan.steps[active_step].sequence_length
    if state.get("active_bucket") != expected_bucket:
        raise ValueError("dynamic loader state active bucket is incompatible")
    if state.get("active_stage_index") != plan.steps[active_step].stage_index:
        raise ValueError("dynamic loader state active stage is incompatible")
    expected_next_bucket = (
        plan.steps[active_step + 1].sequence_length
        if active_step + 1 < len(plan.steps)
        else None
    )
    if state.get("next_bucket") != expected_next_bucket:
        raise ValueError("dynamic loader state next bucket is incompatible")
    if state.get("accumulated_tokens") != active_step * plan.effective_token_batch:
        raise ValueError("dynamic loader state accumulated tokens are incompatible")
    source_names = state["source_names"]
    if (
        not isinstance(source_names, list)
        or not source_names
        or any(not isinstance(name, str) or not name for name in source_names)
        or len(set(source_names)) != len(source_names)
    ):
        raise ValueError("dynamic loader source names are invalid")
    source_weights = state["source_weights"]
    if not isinstance(source_weights, dict) or list(source_weights) != source_names:
        raise ValueError("dynamic loader source weights are invalid")
    for name in source_names:
        weight = source_weights[name]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or weight < 0
        ):
            raise ValueError("dynamic loader source weights are invalid")
    subset_names = state["subset_names"]
    if not isinstance(subset_names, dict) or list(subset_names) != source_names:
        raise ValueError("dynamic loader subset names are invalid")
    for names in subset_names.values():
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("dynamic loader subset names are invalid")
    subset_weights = state["subset_weights"]
    if not isinstance(subset_weights, dict) or any(
        source not in source_names for source in subset_weights
    ):
        raise ValueError("dynamic loader subset weights are invalid")
    for source, weights in subset_weights.items():
        if not isinstance(weights, dict) or list(weights) != subset_names[source]:
            raise ValueError("dynamic loader subset weights are invalid")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or weight < 0
            for weight in weights.values()
        ):
            raise ValueError("dynamic loader subset weights are invalid")
    for name in ("source_states", "document_buffers", "subset_sampler_states"):
        if not isinstance(state[name], dict):
            raise ValueError(f"dynamic loader {name} must be a mapping")
    if not isinstance(state["sampler_state"], list) or not state["sampler_state"]:
        raise ValueError("dynamic loader sampler state is invalid")
    if any(
        not _is_int(value) or not 0 <= value <= 255 for value in state["sampler_state"]
    ):
        raise ValueError("dynamic loader sampler state is invalid")
    for name in ("source_token_counts", "source_document_counts"):
        counts = state[name]
        if not isinstance(counts, dict):
            raise ValueError(f"dynamic loader {name} is invalid")
        if any(not _is_int(value) or value < 0 for value in counts.values()):
            raise ValueError(f"dynamic loader {name} is invalid")
    pending_batch = state["pending_batch"]
    if not isinstance(pending_batch, dict):
        raise ValueError("dynamic loader state must preserve a pending batch")
    expected_shape = (plan.device_batch_size, expected_bucket)
    for name in ("inputs", "targets"):
        matrix = pending_batch.get(name)
        if not isinstance(matrix, list) or len(matrix) != expected_shape[0]:
            raise ValueError(f"pending batch {name} has an invalid batch dimension")
        if any(
            not isinstance(row, list) or len(row) != expected_shape[1] for row in matrix
        ):
            raise ValueError(f"pending batch {name} has an invalid sequence dimension")
        if any(not _is_int(value) for row in matrix for value in row):
            raise ValueError(f"pending batch {name} must contain integer token IDs")


def build_dynamic_checkpoint_metadata(
    plan: DynamicContextPlan,
    active_step: int,
    loader_state: dict,
) -> dict:
    state = make_dynamic_loader_resume_state(plan, active_step, loader_state)
    return {
        "schema_version": DYNAMIC_CONTEXT_GATE_SCHEMA_VERSION,
        "context_plan": plan.to_dict(),
        "active_step": active_step,
        "active_bucket": plan.steps[active_step].sequence_length,
        "loader_state": state,
    }


def validate_dynamic_checkpoint_metadata(
    metadata: Any,
    expected_plan: DynamicContextPlan,
) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("dynamic checkpoint metadata must be a mapping")
    required = {
        "schema_version",
        "context_plan",
        "active_step",
        "active_bucket",
        "loader_state",
    }
    if set(metadata) != required:
        raise ValueError("dynamic checkpoint metadata fields are invalid")
    if metadata["schema_version"] != DYNAMIC_CONTEXT_GATE_SCHEMA_VERSION:
        raise ValueError("dynamic checkpoint schema version is incompatible")
    saved_plan = DynamicContextPlan.from_dict(metadata["context_plan"])
    if saved_plan.to_dict() != expected_plan.to_dict():
        raise ValueError("dynamic checkpoint context plan is incompatible")
    active_step = metadata["active_step"]
    validate_dynamic_loader_resume_state(
        metadata["loader_state"],
        expected_plan,
        expected_active_step=active_step,
    )
    if metadata["active_bucket"] != expected_plan.steps[active_step].sequence_length:
        raise ValueError("dynamic checkpoint active bucket is incompatible")


@dataclass(frozen=True)
class ContextResourceEvidence:
    sequence_length: int
    peak_vram_bytes: int
    active_tokens_per_second: float
    wall_clock_seconds: float
    runtime_verified: bool

    def __post_init__(self):
        object.__setattr__(
            self,
            "sequence_length",
            _positive_int(self.sequence_length, "sequence_length"),
        )
        object.__setattr__(
            self,
            "peak_vram_bytes",
            _positive_int(self.peak_vram_bytes, "peak_vram_bytes"),
        )
        if (
            isinstance(self.active_tokens_per_second, bool)
            or not isinstance(self.active_tokens_per_second, (int, float))
            or not math.isfinite(float(self.active_tokens_per_second))
            or self.active_tokens_per_second <= 0
        ):
            raise ValueError("active_tokens_per_second must be positive and finite")
        if (
            isinstance(self.wall_clock_seconds, bool)
            or not isinstance(self.wall_clock_seconds, (int, float))
            or not math.isfinite(float(self.wall_clock_seconds))
            or self.wall_clock_seconds <= 0
        ):
            raise ValueError("wall_clock_seconds must be positive and finite")
        if not isinstance(self.runtime_verified, bool):
            raise ValueError("runtime_verified must be boolean")


def validate_resource_proof(
    plan: DynamicContextPlan,
    evidence: Any,
    vram_limit_bytes: int,
    minimum_tokens_per_second: float,
    safety_margin_bytes: int,
) -> tuple[ContextResourceEvidence, ...]:
    vram_limit = _positive_int(vram_limit_bytes, "vram_limit_bytes")
    if (
        isinstance(minimum_tokens_per_second, bool)
        or not isinstance(minimum_tokens_per_second, (int, float))
        or not math.isfinite(float(minimum_tokens_per_second))
        or minimum_tokens_per_second <= 0
    ):
        raise ValueError("minimum_tokens_per_second must be positive and finite")
    minimum_throughput = float(minimum_tokens_per_second)
    margin = _positive_int(safety_margin_bytes, "safety_margin_bytes")
    if not isinstance(evidence, (list, tuple)):
        raise ValueError("evidence must be a sequence")
    records = tuple(
        (
            item
            if isinstance(item, ContextResourceEvidence)
            else ContextResourceEvidence(**item)
        )
        for item in evidence
    )
    if len({item.sequence_length for item in records}) != len(records):
        raise ValueError("resource evidence contains duplicate buckets")
    by_bucket = {item.sequence_length: item for item in records}
    if set(by_bucket) != set(plan.buckets):
        raise ValueError("resource evidence must cover every approved bucket")
    for record in records:
        if not record.runtime_verified:
            raise ValueError("resource evidence is not runtime verified")
        if record.peak_vram_bytes + margin > vram_limit:
            raise ValueError("resource evidence exceeds the VRAM safety limit")
        if record.active_tokens_per_second < minimum_throughput:
            raise ValueError("resource evidence is below the throughput requirement")
    return tuple(by_bucket[bucket] for bucket in plan.buckets)
