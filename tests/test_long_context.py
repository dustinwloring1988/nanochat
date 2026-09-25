import pytest

from nanochat.dynamic_context import resolve_context_request
from nanochat.long_context import (
    LONG_CONTEXT_SAFETY_MARGIN_BYTES,
    LONG_CONTEXT_VRAM_LIMIT_BYTES,
    LongContextStepEvidence,
    approved_context_profile,
    build_long_context_approval,
    build_long_context_plan,
    load_long_context_approval,
    validate_long_context_evidence,
)


def make_evidence(plan):
    records = []
    for step in plan.steps:
        records.append(
            LongContextStepEvidence(
                step=step.step,
                sequence_length=step.sequence_length,
                runtime_verified=True,
                model_forward_backward=True,
                loader_batch_verified=True,
                resume_model_equal=True,
                resume_optimizer_equal=True,
                resume_loader_equal=True,
                resume_next_batch_equal=True,
                loss=1.25,
                peak_vram_bytes=1_000_000_000,
                reserved_vram_bytes=1_100_000_000,
                active_tokens_per_second=1000.0,
                wall_clock_seconds=1.0,
                model_config_hash="a" * 64,
                loader_state_hash="b" * 64,
                next_batch_hash="c" * 64,
                data_hash="d" * 64,
                attention_backend="test",
                dtype="float32",
            )
        )
    return tuple(records)


def test_approved_long_context_profile_has_exact_budget():
    plan = build_long_context_plan()
    assert [step.sequence_length for step in plan.steps] == [2048, 8192]
    assert [step.gradient_accumulation_steps for step in plan.steps] == [4, 1]
    assert plan.effective_token_batch == 8192
    assert plan.token_horizon == 16384
    evidence = make_evidence(plan)
    approval = build_long_context_approval(
        plan,
        evidence,
        approval_id="test-approval",
        approved_by="test-reviewer",
        approved_at="2026-09-25",
    )
    assert approval.vram_limit_bytes == LONG_CONTEXT_VRAM_LIMIT_BYTES
    assert approval.safety_margin_bytes == LONG_CONTEXT_SAFETY_MARGIN_BYTES
    profile = approved_context_profile(plan, evidence, approval)
    assert profile.mode == "dynamic"
    assert profile.buckets == (2048, 8192)
    assert profile.enabled is True


def test_tracked_activation_matches_approved_plan():
    plan = build_long_context_plan()
    approval = load_long_context_approval("config/long_context_activation.json")
    assert approval.plan_hash == plan.schedule_hash
    assert approval.vram_limit_bytes == LONG_CONTEXT_VRAM_LIMIT_BYTES


def test_dynamic_context_stays_fallback_without_explicit_approval():
    fallback = resolve_context_request((2048, 8192))
    assert fallback.fallback_used is True
    assert fallback.reason == "dynamic_context_not_approved"


def test_long_context_evidence_rejects_unverified_resume():
    plan = build_long_context_plan()
    evidence = list(make_evidence(plan))
    evidence[1] = LongContextStepEvidence(
        **{
            **evidence[1].to_dict(),
            "resume_next_batch_equal": False,
        }
    )
    with pytest.raises(ValueError, match="unverified"):
        validate_long_context_evidence(plan, evidence)
