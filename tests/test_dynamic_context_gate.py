import copy
import json

import pytest

from nanochat.ai_scientist_experiment import (
    PretrainingExperimentConfig,
    validate_experiment_contract,
)
from nanochat.curriculum import CurriculumScheduler, CurriculumStage
from nanochat.dynamic_context import (
    DYNAMIC_CONTEXT_ENABLED,
    ContextResourceEvidence,
    DynamicContextPlan,
    build_context_step_plan,
    build_dynamic_checkpoint_metadata,
    fixed_context_profile,
    make_dynamic_loader_resume_state,
    resolve_context_request,
    validate_dynamic_checkpoint_metadata,
    validate_dynamic_loader_resume_state,
    validate_resource_proof,
)


def make_scheduler(contexts):
    stages = [
        CurriculumStage(
            name=f"stage_{index}",
            token_ratio=1 / len(contexts),
            context_range=(context, context),
            source_weights={"source": 1.0},
        )
        for index, context in enumerate(contexts)
    ]
    return CurriculumScheduler(
        stages,
        total_tokens=8192 * 8,
        total_steps=8,
    )


def make_loader_state(plan, active_step):
    sequence_length = plan.steps[active_step].sequence_length
    return {
        "source_names": ["source"],
        "source_weights": {"source": 1.0},
        "subset_names": {"source": ["default"]},
        "subset_weights": {},
        "source_states": {"source": {"row_offset": 3}},
        "document_buffers": {"source": [[1, 2], [3]]},
        "sampler_state": [1, 2, 3],
        "subset_sampler_states": {},
        "source_token_counts": {"source": 10},
        "source_document_counts": {"source": 2},
        "pending_batch": {
            "inputs": [[0] * sequence_length],
            "targets": [[1] * sequence_length],
            "shape": [1, sequence_length],
        },
    }


def test_fixed_context_is_the_only_enabled_profile():
    assert DYNAMIC_CONTEXT_ENABLED is False
    profile = fixed_context_profile()
    assert profile.mode == "fixed"
    assert profile.sequence_length == 2048
    assert profile.buckets == (2048,)
    assert profile.enabled is False

    fallback = resolve_context_request((2048, 4096))
    assert fallback.profile == profile
    assert fallback.requested_buckets == (2048, 4096)
    assert fallback.fallback_used is True
    assert fallback.reason == "dynamic_context_not_approved"

    larger_only = resolve_context_request((4096,))
    assert larger_only.profile == profile
    assert larger_only.requested_buckets == (4096,)
    assert larger_only.fallback_used is True


def test_discrete_bucket_plan_aligns_effective_batch_and_scheduler():
    scheduler = make_scheduler((2048, 4096))
    plan = build_context_step_plan(
        scheduler,
        buckets=(2048, 4096),
        device_batch_size=1,
        world_size=1,
        effective_token_batch=8192,
    )

    assert plan.buckets == (2048, 4096)
    assert plan.token_horizon == 65536
    assert [step.sequence_length for step in plan.steps] == [2048] * 4 + [4096] * 4
    assert [step.gradient_accumulation_steps for step in plan.steps] == [4] * 4 + [
        2
    ] * 4
    assert sum(step.effective_tokens for step in plan.steps) == plan.token_horizon
    assert DynamicContextPlan.from_dict(plan.to_dict()) == plan


def test_interpolated_context_is_not_a_discrete_bucket():
    stages = [
        CurriculumStage(
            name="ramp",
            token_ratio=1.0,
            context_range=(2048, 4096),
            source_weights={"source": 1.0},
        )
    ]
    scheduler = CurriculumScheduler(stages, total_tokens=8192 * 8, total_steps=8)

    with pytest.raises(ValueError, match="non-discrete context"):
        build_context_step_plan(
            scheduler,
            buckets=(2048, 4096),
            device_batch_size=1,
            world_size=1,
            effective_token_batch=8192,
        )


def test_effective_batch_must_divide_every_bucket():
    stages = [
        CurriculumStage(
            name="short",
            token_ratio=0.5,
            context_range=(2048, 2048),
            source_weights={"source": 1.0},
        ),
        CurriculumStage(
            name="long",
            token_ratio=0.5,
            context_range=(4096, 4096),
            source_weights={"source": 1.0},
        ),
    ]
    scheduler = CurriculumScheduler(stages, total_tokens=5000 * 8, total_steps=8)

    with pytest.raises(ValueError, match="not divisible"):
        build_context_step_plan(
            scheduler,
            buckets=(2048, 4096),
            device_batch_size=1,
            world_size=1,
            effective_token_batch=5000,
        )


def test_dynamic_resume_state_preserves_loader_and_pending_batch():
    scheduler = make_scheduler((2048, 4096))
    plan = build_context_step_plan(
        scheduler,
        buckets=(2048, 4096),
        device_batch_size=1,
        world_size=1,
        effective_token_batch=8192,
    )
    original = make_loader_state(plan, 4)
    state = make_dynamic_loader_resume_state(plan, 4, original)
    encoded = json.loads(json.dumps(state, sort_keys=True))

    validate_dynamic_loader_resume_state(encoded, plan, expected_active_step=4)
    assert encoded["source_states"] == original["source_states"]
    assert encoded["document_buffers"] == original["document_buffers"]
    assert encoded["pending_batch"] == original["pending_batch"]
    assert encoded["sampler_state"] == original["sampler_state"]


def test_dynamic_checkpoint_metadata_rejects_mismatched_schedule():
    scheduler = make_scheduler((2048, 4096))
    plan = build_context_step_plan(
        scheduler,
        buckets=(2048, 4096),
        device_batch_size=1,
        world_size=1,
        effective_token_batch=8192,
    )
    metadata = build_dynamic_checkpoint_metadata(plan, 4, make_loader_state(plan, 4))
    validate_dynamic_checkpoint_metadata(metadata, plan)

    changed = copy.deepcopy(metadata)
    changed["context_plan"]["schedule_hash"] = "invalid"
    with pytest.raises(ValueError, match="schedule hash"):
        validate_dynamic_checkpoint_metadata(changed, plan)

    changed = copy.deepcopy(metadata)
    changed["loader_state"]["pending_batch"] = None
    with pytest.raises(ValueError, match="pending batch"):
        validate_dynamic_checkpoint_metadata(changed, plan)

    changed = copy.deepcopy(metadata)
    changed["active_bucket"] = 2048
    with pytest.raises(ValueError, match="active bucket"):
        validate_dynamic_checkpoint_metadata(changed, plan)


def test_resource_proof_requires_runtime_evidence_for_every_bucket():
    scheduler = make_scheduler((2048, 4096))
    plan = build_context_step_plan(
        scheduler,
        buckets=(2048, 4096),
        device_batch_size=1,
        world_size=1,
        effective_token_batch=8192,
    )
    evidence = [
        ContextResourceEvidence(2048, 1_000_000, 1000.0, 10.0, True),
        ContextResourceEvidence(4096, 1_500_000, 900.0, 12.0, True),
    ]

    assert validate_resource_proof(plan, evidence, 2_000_000, 500, 100) == tuple(
        evidence
    )
    with pytest.raises(ValueError, match="runtime verified"):
        validate_resource_proof(
            plan,
            [evidence[0], ContextResourceEvidence(4096, 1, 900.0, 12.0, False)],
            2_000_000,
            500,
            100,
        )
    with pytest.raises(ValueError, match="cover every"):
        validate_resource_proof(plan, evidence[:1], 2_000_000, 500, 100)
    with pytest.raises(ValueError, match="VRAM"):
        validate_resource_proof(plan, evidence, 1_000_000, 500, 100)


def test_current_experiment_contract_still_rejects_dynamic_curriculum(tmp_path):
    source = tmp_path / "source"
    config_dir = source / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ai_scientist_pretraining.yaml").write_text(
        "stages:\n"
        "  - name: ramp\n"
        "    token_ratio: 1.0\n"
        "    context_range: [2048, 4096]\n"
        "    source_weights: {climbmix: 1.0}\n"
        "    subset_weights: {}\n",
        encoding="utf-8",
    )
    baseline = PretrainingExperimentConfig()
    candidate = PretrainingExperimentConfig()

    with pytest.raises(ValueError, match="fixed baseline context"):
        validate_experiment_contract(candidate, baseline, source)
