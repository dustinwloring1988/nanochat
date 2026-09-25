import json

import pytest

from nanochat.ai_scientist_experiment import (
    PretrainingExperimentConfig,
    build_command,
    create_plots,
    validate_experiment_contract,
)
from nanochat.research_results import (
    build_resume_contract,
    validate_resume_contract,
)


def result_payload():
    return {
        "curves": {
            "step": [0, 10, 20],
            "train_loss": [2.0, 1.5, 1.25],
            "val_bpb": [1.5, 1.35, 1.25],
            "tokens_per_second": [800.0, 900.0, 1000.0],
        },
        "composition": {
            "schema_version": 1,
            "stages": [
                {
                    "name": "fixed",
                    "token_ratio": 1.0,
                    "context_range": [2048, 2048],
                    "source_weights": {"climbmix": 1.0},
                    "subset_weights": {},
                }
            ],
            "observed_source_tokens": {"climbmix": 100},
            "observed_document_counts": {"climbmix": 1},
            "measurement_status": "observed",
        },
    }


def test_experiment_config_validation():
    config = PretrainingExperimentConfig()
    config.validate()
    with pytest.raises(ValueError, match="divisible"):
        PretrainingExperimentConfig(
            total_batch_size=100,
            device_batch_size=7,
            max_seq_len=16,
        ).validate()
    with pytest.raises(ValueError, match="window_pattern"):
        PretrainingExperimentConfig(window_pattern="XL").validate()
    with pytest.raises(ValueError, match="eval_tokens must be divisible"):
        PretrainingExperimentConfig(eval_tokens=983041).validate()


def test_experiment_contract_preserves_budget_and_context(tmp_path):
    source = tmp_path / "source"
    config_dir = source / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "ai_scientist_pretraining.yaml").write_text(
        "stages:\n"
        "  - name: fixed\n"
        "    token_ratio: 1.0\n"
        "    context_range: [2048, 2048]\n"
        "    source_weights: {climbmix: 1.0}\n"
        "    subset_weights: {}\n",
        encoding="utf-8",
    )
    baseline = PretrainingExperimentConfig()
    candidate = PretrainingExperimentConfig(
        total_batch_size=49152,
        num_iterations=400,
    )
    validate_experiment_contract(candidate, baseline, source)
    validate_experiment_contract(
        PretrainingExperimentConfig(seed=43), baseline, source, allow_seed_variants=True
    )
    with pytest.raises(ValueError, match="seed"):
        validate_experiment_contract(
            PretrainingExperimentConfig(seed=43), baseline, source
        )
    with pytest.raises(ValueError, match="max_seq_len"):
        validate_experiment_contract(
            PretrainingExperimentConfig(max_seq_len=4096, total_batch_size=196608),
            baseline,
            source,
        )
    with pytest.raises(ValueError, match="token budget"):
        validate_experiment_contract(
            PretrainingExperimentConfig(num_iterations=201), baseline, source
        )
    with pytest.raises(ValueError, match="max_peak_vram_bytes"):
        validate_experiment_contract(
            PretrainingExperimentConfig(max_peak_vram_bytes=1 << 62),
            baseline,
            source,
        )


def test_build_command_contains_fixed_contract(tmp_path):
    config = PretrainingExperimentConfig()
    command = build_command(
        config,
        tmp_path,
        tmp_path / "cache",
        tmp_path / "results.json",
    )
    assert command[1:3] == ["-m", "scripts.base_train_curriculum"]
    assert "--num-iterations" in command
    assert command[command.index("--num-iterations") + 1] == "200"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--results-json") + 1] == str(
        tmp_path / "results.json"
    )


def test_resume_contract_rejects_incompatible_configuration():
    contract = build_resume_contract(
        model_config={"n_layer": 6, "sequence_len": 2048},
        seed=42,
        world_size=1,
        rank=0,
        token_horizon=19660800,
        device_batch_size=24,
        max_seq_len=2048,
        total_batch_size=98304,
        curriculum_state={"stages": ["fixed"]},
        dataloader_config={"buffer_size": 1000},
        dataset_manifest_hash="dataset",
        tokenizer_hash="tokenizer",
    )
    validate_resume_contract(contract, contract)
    changed = {**contract, "seed": 43}
    with pytest.raises(ValueError, match="seed"):
        validate_resume_contract(contract, changed)
    with pytest.raises(ValueError, match="version"):
        validate_resume_contract({**contract, "version": 0}, contract)


def test_create_plots(tmp_path):
    result_path = tmp_path / "results.json"
    result_path.write_text(json.dumps(result_payload()), encoding="utf-8")
    create_plots(result_path, tmp_path)
    assert (tmp_path / "validation_bpb.png").is_file()
    assert (tmp_path / "training_loss.png").is_file()
    assert (tmp_path / "tokens_per_second.png").is_file()
    assert (tmp_path / "source_mixture.png").is_file()
