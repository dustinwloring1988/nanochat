import json
import math
from dataclasses import dataclass
from typing import Dict, List

from nanochat.curriculum import CurriculumScheduler

CURRICULUM_STAGE_TRANSITION_SCHEMA_VERSION = 1

_COUNTER_FIELDS = {
    "stage_index",
    "stage_name",
    "source_token_counts",
    "source_document_counts",
}
_TRANSITION_FIELDS = {
    "schema_version",
    "completed_stage_index",
    "completed_stage_name",
    "current_stage_index",
    "current_stage_name",
    "next_stage_index",
    "next_stage_name",
    "transition_step",
    "next_source_weights",
    "next_subset_weights",
    "stage_composition_counters",
}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_exact_fields(value, field_name: str, required: set) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - required, key=repr)
    if missing or unexpected:
        raise ValueError(
            f"{field_name} fields are invalid: missing={missing}, unexpected={unexpected}"
        )


def _require_index(value, field_name: str, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return int(value)


def _require_name(value, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _weight_map(value, field_name: str) -> Dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    normalised = {}
    for name, weight in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{field_name} names must be non-empty strings")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError(f"{field_name}[{name}] must be numeric")
        try:
            weight = float(weight)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{name}] must be numeric") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{field_name}[{name}] must be finite and non-negative")
        normalised[name] = weight
    total_weight = sum(normalised.values())
    if not math.isfinite(total_weight) or not math.isclose(
        total_weight, 1.0, rel_tol=0.0, abs_tol=0.01
    ):
        raise ValueError(f"{field_name} must sum to approximately 1.0")
    return normalised


def _subset_map(value, source_weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    if not isinstance(value, dict):
        raise ValueError("next_subset_weights must be a mapping")
    normalised = {}
    for source_name, weights in value.items():
        if source_name not in source_weights:
            raise ValueError(f"unknown subset source: {source_name}")
        normalised[source_name] = _weight_map(
            weights, f"next_subset_weights[{source_name}]"
        )
    return normalised


def _count_map(value, field_name: str) -> Dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    normalised = {}
    for name, count in value.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{field_name} names must be non-empty strings")
        if not _is_int(count) or count < 0:
            raise ValueError(f"{field_name}[{name}] must be a non-negative integer")
        normalised[name] = int(count)
    return normalised


@dataclass
class CurriculumStageCompositionCounters:
    stage_index: int
    stage_name: str
    source_token_counts: Dict[str, int]
    source_document_counts: Dict[str, int]

    def __post_init__(self):
        self.stage_index = _require_index(self.stage_index, "stage_index")
        self.stage_name = _require_name(self.stage_name, "stage_name")
        self.source_token_counts = _count_map(
            self.source_token_counts, "source_token_counts"
        )
        self.source_document_counts = _count_map(
            self.source_document_counts, "source_document_counts"
        )
        if list(self.source_token_counts) != list(self.source_document_counts):
            raise ValueError(
                "source token and document counters must contain the same ordered sources"
            )

    def to_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "stage_name": self.stage_name,
            "source_token_counts": dict(self.source_token_counts),
            "source_document_counts": dict(self.source_document_counts),
        }


@dataclass
class CurriculumStageTransitionState:
    schema_version: int
    completed_stage_index: int
    completed_stage_name: str
    current_stage_index: int
    current_stage_name: str
    next_stage_index: int
    next_stage_name: str
    transition_step: int
    next_source_weights: Dict[str, float]
    next_subset_weights: Dict[str, Dict[str, float]]
    stage_composition_counters: List[CurriculumStageCompositionCounters]

    def __post_init__(self):
        if (
            not _is_int(self.schema_version)
            or self.schema_version != CURRICULUM_STAGE_TRANSITION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported stage transition schema_version")
        for field_name in (
            "completed_stage_index",
            "current_stage_index",
            "next_stage_index",
        ):
            setattr(
                self, field_name, _require_index(getattr(self, field_name), field_name)
            )
        for field_name in (
            "completed_stage_name",
            "current_stage_name",
            "next_stage_name",
        ):
            setattr(
                self, field_name, _require_name(getattr(self, field_name), field_name)
            )
        if (
            self.completed_stage_index != self.current_stage_index
            or self.completed_stage_name != self.current_stage_name
        ):
            raise ValueError("completed and current stage identities must match")
        if self.next_stage_index != self.current_stage_index + 1:
            raise ValueError(
                "next stage index must immediately follow the current stage"
            )
        self.transition_step = _require_index(
            self.transition_step, "transition_step", minimum=1
        )
        self.next_source_weights = _weight_map(
            self.next_source_weights, "next_source_weights"
        )
        self.next_subset_weights = _subset_map(
            self.next_subset_weights, self.next_source_weights
        )
        if not isinstance(self.stage_composition_counters, (list, tuple)) or any(
            not isinstance(counter, CurriculumStageCompositionCounters)
            for counter in self.stage_composition_counters
        ):
            raise ValueError("stage_composition_counters contains an invalid counter")
        self.stage_composition_counters = sorted(
            self.stage_composition_counters, key=lambda counter: counter.stage_index
        )
        if tuple(
            counter.stage_index for counter in self.stage_composition_counters
        ) != tuple(range(self.current_stage_index + 1)):
            raise ValueError(
                "stage composition counters must cover every stage through the current stage"
            )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "completed_stage_index": self.completed_stage_index,
            "completed_stage_name": self.completed_stage_name,
            "current_stage_index": self.current_stage_index,
            "current_stage_name": self.current_stage_name,
            "next_stage_index": self.next_stage_index,
            "next_stage_name": self.next_stage_name,
            "transition_step": self.transition_step,
            "next_source_weights": dict(self.next_source_weights),
            "next_subset_weights": {
                source: dict(weights)
                for source, weights in self.next_subset_weights.items()
            },
            "stage_composition_counters": [
                counter.to_dict() for counter in self.stage_composition_counters
            ],
        }

    @staticmethod
    def _validate_against_scheduler(
        state: "CurriculumStageTransitionState", scheduler: CurriculumScheduler
    ) -> None:
        if not isinstance(scheduler, CurriculumScheduler):
            raise ValueError("scheduler must be a CurriculumScheduler")
        if scheduler.stage_step_boundaries is None:
            raise ValueError("stage transition state requires step-based scheduling")
        if state.next_stage_index >= len(scheduler.stages):
            raise ValueError("next stage index is outside the curriculum")
        identities = (
            (state.completed_stage_index, state.completed_stage_name),
            (state.current_stage_index, state.current_stage_name),
            (state.next_stage_index, state.next_stage_name),
        )
        for stage_index, stage_name in identities:
            if stage_name != scheduler.stages[stage_index].name:
                raise ValueError(f"stage name does not match curriculum: {stage_name}")
        for counter in state.stage_composition_counters:
            expected_stage = scheduler.stages[counter.stage_index]
            source_names = list(expected_stage.source_weights)
            if counter.stage_name != expected_stage.name:
                raise ValueError(
                    "composition counter stage name does not match curriculum"
                )
            if list(counter.source_token_counts) != source_names:
                raise ValueError(
                    "composition counter token sources do not match curriculum"
                )
            if list(counter.source_document_counts) != source_names:
                raise ValueError(
                    "composition counter document sources do not match curriculum"
                )
        next_stage = scheduler.stages[state.next_stage_index]
        if list(state.next_source_weights) != list(next_stage.source_weights):
            raise ValueError("next source weight order does not match curriculum")
        expected_source_weights = {
            name: float(weight) for name, weight in next_stage.source_weights.items()
        }
        if state.next_source_weights != expected_source_weights:
            raise ValueError("next source weights do not match curriculum")
        if list(state.next_subset_weights) != list(next_stage.subset_weights):
            raise ValueError("next subset weight order does not match curriculum")
        for source_name, weights in next_stage.subset_weights.items():
            if list(state.next_subset_weights[source_name]) != list(weights):
                raise ValueError("next subset weight order does not match curriculum")
        expected_subset_weights = {
            source: {name: float(weight) for name, weight in weights.items()}
            for source, weights in next_stage.subset_weights.items()
        }
        if state.next_subset_weights != expected_subset_weights:
            raise ValueError("next subset weights do not match curriculum")
        expected_step = scheduler.stage_step_boundaries[state.next_stage_index - 1]
        if state.transition_step != expected_step:
            raise ValueError("transition_step does not match the curriculum boundary")
        if (
            scheduler.get_current_stage_index(step=state.transition_step - 1)
            != state.current_stage_index
        ):
            raise ValueError("current stage does not precede the transition boundary")

    @classmethod
    def from_dict(
        cls, data: dict, scheduler: CurriculumScheduler
    ) -> "CurriculumStageTransitionState":
        try:
            json.dumps(data, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("stage transition state must be JSON-compatible") from exc
        _require_exact_fields(data, "stage transition state", _TRANSITION_FIELDS)
        if not isinstance(data["stage_composition_counters"], list):
            raise ValueError("stage_composition_counters must be a list")
        counters = []
        for counter in data["stage_composition_counters"]:
            _require_exact_fields(counter, "composition counter", _COUNTER_FIELDS)
            counters.append(CurriculumStageCompositionCounters(**counter))
        state = cls(
            schema_version=data["schema_version"],
            completed_stage_index=data["completed_stage_index"],
            completed_stage_name=data["completed_stage_name"],
            current_stage_index=data["current_stage_index"],
            current_stage_name=data["current_stage_name"],
            next_stage_index=data["next_stage_index"],
            next_stage_name=data["next_stage_name"],
            transition_step=data["transition_step"],
            next_source_weights=data["next_source_weights"],
            next_subset_weights=data["next_subset_weights"],
            stage_composition_counters=counters,
        )
        cls._validate_against_scheduler(state, scheduler)
        return state


def assert_stage_transition_has_no_pending_batch(dataloader_state_dict: dict) -> None:
    if not isinstance(dataloader_state_dict, dict):
        raise ValueError("dataloader_state_dict must be a mapping")
    if "pending_batch" not in dataloader_state_dict:
        raise ValueError("stage transition requires explicit pending_batch state")
    if dataloader_state_dict["pending_batch"] is not None:
        raise RuntimeError(
            "stage transition blocked: pending batch must be consumed or checkpointed"
        )
