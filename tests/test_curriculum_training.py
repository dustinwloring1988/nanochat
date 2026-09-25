import math

import pytest
import torch

from nanochat.common import seed_everything
from nanochat.curriculum import CurriculumScheduler, CurriculumStage
from nanochat.multi_source_dataloader import WeightedSourceSampler
from nanochat.research_results import (
    build_training_results,
    load_results,
    validate_results,
    write_json_atomic,
)


def curriculum():
    stages = [
        CurriculumStage(
            name="foundation",
            token_ratio=0.55,
            context_range=(2048, 2048),
            source_weights={"climbmix": 1.0},
        ),
        CurriculumStage(
            name="specialized",
            token_ratio=0.45,
            context_range=(2048, 2048),
            source_weights={"climbmix": 0.5, "code_reasoning": 0.5},
        ),
    ]
    return CurriculumScheduler(stages, total_tokens=1000, total_steps=100)


def test_curriculum_boundaries_and_weights():
    scheduler = curriculum()
    assert scheduler.get_current_stage_index(step=54) == 0
    assert scheduler.get_current_stage_index(step=55) == 1
    assert scheduler.get_source_weights(step=0) == {"climbmix": 1.0}
    assert scheduler.get_source_weights(step=55) == {
        "climbmix": 0.5,
        "code_reasoning": 0.5,
    }


def test_curriculum_boundaries_distribute_rounding_residual():
    stages = [
        CurriculumStage("a", 0.333, (2048, 2048), {"climbmix": 1.0}),
        CurriculumStage("b", 0.333, (2048, 2048), {"climbmix": 1.0}),
        CurriculumStage("c", 0.334, (2048, 2048), {"climbmix": 1.0}),
    ]
    scheduler = CurriculumScheduler(stages, total_tokens=10, total_steps=10)
    assert scheduler.stage_step_boundaries == [3, 7, 10]


def test_weighted_source_sampler_seed():
    first = WeightedSourceSampler({"a": 0.5, "b": 0.5}, seed=42)
    second = WeightedSourceSampler({"a": 0.5, "b": 0.5}, seed=42)
    third = WeightedSourceSampler({"a": 0.5, "b": 0.5}, seed=43)
    first_values = [first.sample() for _ in range(20)]
    second_values = [second.sample() for _ in range(20)]
    third_values = [third.sample() for _ in range(20)]
    assert first_values == second_values
    assert first_values != third_values


def test_seed_everything_reproduces_torch_values():
    seed_everything(42)
    first = torch.rand(8)
    seed_everything(42)
    second = torch.rand(8)
    assert torch.equal(first, second)


def test_canonical_results_round_trip(tmp_path):
    base_dir = tmp_path / "cache"
    tokenizer_dir = base_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.pkl").write_bytes(b"tokenizer")
    (tokenizer_dir / "token_bytes.pt").write_bytes(b"bytes")
    data_dir = base_dir / "base_data_climbmix"
    data_dir.mkdir()
    (data_dir / "train-000.parquet").write_bytes(b"data")
    results_path = tmp_path / "results.json"
    payload = build_training_results(
        root=tmp_path,
        base_dir=base_dir,
        seed=42,
        model_tag="d6",
        user_config={"depth": 6},
        metrics={
            "val_bpb": 1.25,
            "best_val_bpb": 1.2,
            "core_metric": 0.5,
            "final_train_loss": 1.25,
            "training_time_seconds": 10.0,
            "active_training_time_seconds": 10.0,
            "wall_clock_training_seconds": 12.0,
            "tokens_per_second": 1000.0,
            "active_tokens_per_second": 1000.0,
            "wall_clock_tokens_per_second": 833.0,
            "mfu_percent": 12.5,
            "peak_vram_bytes": 1024,
            "total_training_tokens": 10000,
            "parameter_count": 1000,
        },
        curves={
            "step": [0, 10],
            "train_loss": [2.0, 1.5],
            "val_bpb": [1.5, 1.25],
            "tokens_per_second": [900.0, 1000.0],
        },
    )
    write_json_atomic(results_path, payload)
    assert load_results(results_path) == payload
    payload["metrics"]["core_metric"] = -0.1
    payload["metrics"]["mfu_percent"] = None
    validate_results(payload)
    payload["metrics"]["val_bpb"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        validate_results(payload)
