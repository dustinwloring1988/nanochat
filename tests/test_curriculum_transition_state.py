import copy
import json

import pytest

from nanochat.curriculum import CurriculumScheduler, CurriculumStage
from nanochat.curriculum_state import (
    CURRICULUM_STAGE_TRANSITION_SCHEMA_VERSION,
    CurriculumStageCompositionCounters,
    CurriculumStageTransitionState,
    assert_stage_transition_has_no_pending_batch,
)


def transition_contract():
    stages = [
        CurriculumStage(
            name="foundation",
            token_ratio=0.4,
            context_range=(2048, 2048),
            source_weights={"climbmix": 1.0},
        ),
        CurriculumStage(
            name="specialized",
            token_ratio=0.3,
            context_range=(2048, 2048),
            source_weights={"climbmix": 0.6, "code": 0.4},
        ),
        CurriculumStage(
            name="long_context",
            token_ratio=0.3,
            context_range=(2048, 2048),
            source_weights={"climbmix": 0.5, "code": 0.5},
            subset_weights={"code": {"synthetic": 0.7, "web": 0.3}},
        ),
    ]
    scheduler = CurriculumScheduler(stages, total_tokens=1000, total_steps=100)
    counters = [
        CurriculumStageCompositionCounters(
            stage_index=0,
            stage_name="foundation",
            source_token_counts={"climbmix": 24000},
            source_document_counts={"climbmix": 24},
        ),
        CurriculumStageCompositionCounters(
            stage_index=1,
            stage_name="specialized",
            source_token_counts={"climbmix": 12000, "code": 8000},
            source_document_counts={"climbmix": 12, "code": 8},
        ),
    ]
    state = CurriculumStageTransitionState(
        schema_version=CURRICULUM_STAGE_TRANSITION_SCHEMA_VERSION,
        completed_stage_index=1,
        completed_stage_name="specialized",
        current_stage_index=1,
        current_stage_name="specialized",
        next_stage_index=2,
        next_stage_name="long_context",
        transition_step=70,
        next_source_weights={"climbmix": 0.5, "code": 0.5},
        next_subset_weights={"code": {"synthetic": 0.7, "web": 0.3}},
        stage_composition_counters=counters,
    )
    return scheduler, state


def test_stage_transition_state_round_trip_is_json_compatible():
    scheduler, state = transition_contract()

    payload = state.to_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

    assert json.loads(encoded) == payload
    assert CurriculumStageTransitionState.from_dict(payload, scheduler) == state
    assert state.to_dict() == payload
    assert payload["completed_stage_index"] == 1
    assert payload["current_stage_name"] == "specialized"
    assert payload["next_stage_index"] == 2
    assert payload["transition_step"] == 70


def test_stage_transition_state_rejects_incomplete_state():
    scheduler, state = transition_contract()
    payload = state.to_dict()
    del payload["next_subset_weights"]

    with pytest.raises(ValueError, match="missing"):
        CurriculumStageTransitionState.from_dict(payload, scheduler)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("next_stage_index", 3, "immediately follow"),
        ("next_source_weights", {"climbmix": 0.25, "code": 0.75}, "source weights"),
        (
            "next_subset_weights",
            {"code": {"synthetic": 0.6, "web": 0.4}},
            "subset weights",
        ),
    ],
)
def test_stage_transition_state_rejects_mismatched_state(field, value, message):
    scheduler, state = transition_contract()
    payload = state.to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        CurriculumStageTransitionState.from_dict(payload, scheduler)


def test_stage_transition_state_rejects_malformed_counters():
    scheduler, state = transition_contract()

    missing_stage = state.to_dict()
    missing_stage["stage_composition_counters"].pop()
    with pytest.raises(ValueError, match="cover every stage"):
        CurriculumStageTransitionState.from_dict(missing_stage, scheduler)

    mismatched_name = state.to_dict()
    mismatched_name["stage_composition_counters"][1]["stage_name"] = "other"
    with pytest.raises(ValueError, match="stage name"):
        CurriculumStageTransitionState.from_dict(mismatched_name, scheduler)

    incomplete_counts = state.to_dict()
    incomplete_counts["stage_composition_counters"][1]["source_token_counts"].pop(
        "code"
    )
    with pytest.raises(ValueError, match="same ordered sources"):
        CurriculumStageTransitionState.from_dict(incomplete_counts, scheduler)


def test_stage_transition_state_preserves_composition_counters():
    scheduler, state = transition_contract()
    payload = copy.deepcopy(state.to_dict())

    restored = CurriculumStageTransitionState.from_dict(payload, scheduler)

    assert [
        counter.to_dict() for counter in restored.stage_composition_counters
    ] == payload["stage_composition_counters"]
    assert restored.stage_composition_counters[0].source_token_counts == {
        "climbmix": 24000
    }
    assert restored.stage_composition_counters[1].source_token_counts == {
        "climbmix": 12000,
        "code": 8000,
    }
    payload["stage_composition_counters"][0]["source_token_counts"]["climbmix"] = 0
    assert (
        restored.stage_composition_counters[0].source_token_counts["climbmix"] == 24000
    )


def test_stage_transition_guard_fails_closed_on_pending_batch():
    assert_stage_transition_has_no_pending_batch({"pending_batch": None})
    with pytest.raises(RuntimeError, match="pending batch"):
        assert_stage_transition_has_no_pending_batch(
            {"pending_batch": {"inputs": [[1]], "targets": [[2]]}}
        )
    with pytest.raises(ValueError, match="explicit pending_batch"):
        assert_stage_transition_has_no_pending_batch({})
