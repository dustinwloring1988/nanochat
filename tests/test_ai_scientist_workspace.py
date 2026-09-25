import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("openai")
pytest.importorskip("omegaconf")
pytest.importorskip("yaml")

import yaml

from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file
from ai_scientist.treesearch.interpreter import Interpreter
from ai_scientist.treesearch.utils.config import load_cfg
from ai_scientist.treesearch.nanochat_adapter import (
    apply_nanochat_result,
    attest_nanochat_artifact,
    experiment_env,
    load_canonical_result,
    prepare_node_workspace,
    verify_nanochat_artifact,
)


def config(tmp_path: Path):
    source = tmp_path / "source"
    for directory in ("nanochat", "scripts", "tasks", "config"):
        (source / directory).mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (source / "config" / "ai_scientist_experiment.json").write_text(
        "{}", encoding="utf-8"
    )
    (source / "config" / "ai_scientist_pretraining.yaml").write_text(
        "stages:\n"
        "  - name: fixed\n"
        "    token_ratio: 1.0\n"
        "    context_range: [2048, 2048]\n"
        "    source_weights:\n"
        "      climbmix: 1.0\n"
        "    subset_weights: {}\n",
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    tokenizer = cache / "tokenizer"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.pkl").write_bytes(b"tokenizer")
    (tokenizer / "token_bytes.pt").write_bytes(b"bytes")
    data = cache / "base_data_climbmix"
    data.mkdir()
    (data / "train-000.parquet").write_bytes(b"data")
    return SimpleNamespace(
        workspace_dir=tmp_path / "workspace",
        log_dir=tmp_path / "logs",
        experiment=SimpleNamespace(
            mode="nanochat",
            source_dir=source,
            cache_dir=cache,
            config_file=source / "config" / "ai_scientist_experiment.json",
            model="opencode/space-bunny-free",
            require_plots=True,
        ),
    )


def test_prepare_node_workspace_is_unique(tmp_path):
    cfg = config(tmp_path)
    first = prepare_node_workspace(cfg)
    second = prepare_node_workspace(cfg)
    assert first["task_id"] != second["task_id"]
    assert first["source"] != second["source"]
    assert (first["source"] / "nanochat").is_dir()
    assert (first["source"] / "pyproject.toml").is_file()
    assert first["working"].is_dir()
    assert (
        json.loads(first["experiment_config"].read_text(encoding="utf-8"))[
            "max_seq_len"
        ]
        == 2048
    )
    assert first["baseline_config"].is_file()


def test_experiment_env_contains_only_runtime_paths(tmp_path):
    cfg = config(tmp_path)
    paths = prepare_node_workspace(cfg)
    environment = experiment_env(cfg, paths, 0)
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["NANOCHAT_SOURCE_DIR"] == str(paths["source"])
    assert environment["NANOCHAT_EXPERIMENT_CONFIG"] == str(paths["experiment_config"])
    assert len(environment["NANOCHAT_BASELINE_CONFIG_HASH"]) == 64
    assert "OPENCODE_API_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert "OPENROUTER_API_KEY" not in environment


def test_run_config_uses_one_model_and_runtime_paths(tmp_path, monkeypatch):
    idea_dir = tmp_path / "idea"
    idea_dir.mkdir()
    idea_path = idea_dir / "idea.json"
    idea_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("NANOCHAT_SHARED_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("NANOCHAT_SOURCE_ROOT", str(tmp_path / "source"))
    run_config = edit_bfts_config_file(
        "bfts_config.yaml",
        str(idea_dir),
        str(idea_path),
        model="openrouter/vendor/model",
    )
    with open(run_config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    assert config["agent"]["code"]["model"] == "openrouter/vendor/model"
    assert config["agent"]["feedback"]["model"] == "openrouter/vendor/model"
    assert config["agent"]["vlm_feedback"]["model"] == "openrouter/vendor/model"
    assert config["agent"]["summary"]["model"] == "openrouter/vendor/model"
    assert config["agent"]["select_node"]["model"] == "openrouter/vendor/model"
    assert config["agent"]["max_nodes"] == 1
    assert config["experiment"]["model"] == "openrouter/vendor/model"
    assert config["experiment"]["source_dir"] == str(tmp_path / "source")
    assert config["experiment"]["cache_dir"] == str(tmp_path / "cache")
    assert config["experiment"]["config_file"] == str(
        tmp_path / "source" / "config" / "ai_scientist_experiment.json"
    )
    loaded = load_cfg(run_config)
    assert loaded.experiment.mode == "nanochat"
    assert loaded.agent.metric_only is True
    assert loaded.agent.max_nodes == 1
    assert loaded.agent.multi_seed_eval.num_seeds == 0
    unlimited_path = edit_bfts_config_file(
        "bfts_config.yaml",
        str(idea_dir),
        str(idea_path),
        model="openrouter/vendor/model",
        max_nodes=0,
    )
    with pytest.raises(ValueError, match="max_nodes=1"):
        load_cfg(unlimited_path)


def canonical_result():
    return {
        "schema_version": 2,
        "status": "ok",
        "primary_metric": {
            "name": "val_bpb",
            "value": 1.25,
            "lower_is_better": True,
        },
        "metrics": {
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
            "peak_vram_bytes": 100,
            "total_training_tokens": 19660800,
            "parameter_count": 1000,
        },
        "curves": {
            "step": [0, 10, 20],
            "train_loss": [2.0, 1.5, 1.25],
            "val_bpb": [1.3, 1.27, 1.25],
            "tokens_per_second": [900.0, 950.0, 1000.0],
        },
        "guardrails": {
            "max_seq_len": 2048,
            "max_peak_vram_bytes": 17179869184,
            "max_training_seconds": 1800,
            "required_plot_count": 4,
            "required_total_training_tokens": 19660800,
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
            "observed_source_tokens": {"climbmix": 19660800},
            "observed_document_counts": {"climbmix": 1},
            "measurement_status": "observed",
        },
        "provenance": {
            "git_commit": None,
            "source_revision": "source-sha256:a",
            "source_hash": "a",
            "tokenizer_hash": "b",
            "dataset_manifest_hash": "c",
            "seed": 42,
            "model_tag": "ai-scientist-candidate",
            "experiment_model": "opencode/space-bunny-free",
            "runtime": {
                "python": "3.11",
                "torch": "2.9.1",
                "cuda": "12.8",
                "cuda_available": True,
                "device": "test-device",
            },
            "user_config": {
                "max_seq_len": 2048,
                "num_iterations": 200,
                "total_batch_size": 98304,
                "device_batch_size": 24,
                "eval_every": 20,
                "eval_tokens": 983040,
                "seed": 42,
            },
        },
    }


def bind_provenance(payload, cfg, paths):
    from nanochat.research_results import (
        dataset_manifest_hash,
        runtime_metadata,
        source_hash,
        source_identity,
        tokenizer_hash,
    )

    payload["provenance"].update(
        {
            "source_hash": source_hash(paths["source"]),
            "source_revision": source_identity(paths["source"]),
            "tokenizer_hash": tokenizer_hash(cfg.experiment.cache_dir),
            "dataset_manifest_hash": dataset_manifest_hash(cfg.experiment.cache_dir),
            "runtime": runtime_metadata(),
        }
    )
    payload["provenance"]["baseline_config_hash"] = paths["config_hash"]


def test_canonical_result_validation(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(canonical_result()), encoding="utf-8")
    assert load_canonical_result(path)["primary_metric"]["value"] == 1.25
    payload = canonical_result()
    payload["metrics"]["tokens_per_second"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive and finite"):
        load_canonical_result(path)
    payload = canonical_result()
    payload["primary_metric"]["value"] = 1.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must equal"):
        load_canonical_result(path)


def test_apply_result_sets_metric_and_plot_state(tmp_path):
    from ai_scientist.treesearch.journal import Node
    from ai_scientist.treesearch.interpreter import ExecutionResult
    from nanochat.research_results import source_hash

    cfg = config(tmp_path)
    paths = prepare_node_workspace(cfg)
    payload = canonical_result()
    bind_provenance(payload, cfg, paths)
    (paths["working"] / "results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    for name in (
        "validation_bpb.png",
        "training_loss.png",
        "tokens_per_second.png",
        "source_mixture.png",
    ):
        (paths["working"] / name).write_bytes(b"png")
    node = Node(plan="baseline", code="print('ok')")
    result = ExecutionResult(["ok"], 1.0, "KeyError", {"key": "val_bpb"}, [])
    apply_nanochat_result(node, result, cfg, paths)
    assert node.is_buggy is False
    assert "post-result execution error ignored" in node.analysis
    assert node.is_buggy_plots is False
    assert node.metric.get_mean_value() == 1.25
    assert Path(node.exp_results_dir, "results.json").is_file()
    assert Path(node.exp_results_dir, "candidate.patch").is_file()


def test_controller_attestation_is_required_and_detects_tampering(tmp_path):
    from ai_scientist.treesearch.interpreter import ExecutionResult
    from ai_scientist.treesearch.journal import Node

    cfg = config(tmp_path)
    paths = prepare_node_workspace(cfg)
    payload = canonical_result()
    bind_provenance(payload, cfg, paths)
    (paths["working"] / "results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    for name in (
        "validation_bpb.png",
        "training_loss.png",
        "tokens_per_second.png",
        "source_mixture.png",
    ):
        (paths["working"] / name).write_bytes(b"png")
    node = Node(plan="candidate", code="print('ok')")
    apply_nanochat_result(
        node, ExecutionResult(["ok"], 1.0, None, None, None), cfg, paths
    )
    with pytest.raises(FileNotFoundError):
        verify_nanochat_artifact(Path(node.exp_results_dir), cfg, b"k" * 32, node.id)
    attest_nanochat_artifact(
        Path(node.exp_results_dir), cfg, b"k" * 32, node.id, paths["task_id"]
    )
    assert (
        verify_nanochat_artifact(Path(node.exp_results_dir), cfg, b"k" * 32, node.id)[
            "primary_metric"
        ]["value"]
        == 1.25
    )
    result_path = Path(node.exp_results_dir, "results.json")
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stale"):
        verify_nanochat_artifact(Path(node.exp_results_dir), cfg, b"k" * 32, node.id)


def test_candidate_source_is_archived_and_reused(tmp_path):
    from ai_scientist.treesearch.interpreter import ExecutionResult
    from ai_scientist.treesearch.journal import Node

    cfg = config(tmp_path)
    first = prepare_node_workspace(cfg)
    changed = first["source"] / "nanochat" / "candidate.py"
    changed.write_text("VALUE = 1\n", encoding="utf-8")
    payload = canonical_result()
    bind_provenance(payload, cfg, first)
    (first["working"] / "results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    for name in (
        "validation_bpb.png",
        "training_loss.png",
        "tokens_per_second.png",
        "source_mixture.png",
    ):
        (first["working"] / name).write_bytes(b"png")
    node = Node(plan="candidate", code="print('ok')")
    apply_nanochat_result(
        node, ExecutionResult(["ok"], 1.0, None, None, None), cfg, first
    )
    assert Path(node.candidate_source, "nanochat", "candidate.py").is_file()
    second = prepare_node_workspace(cfg, node.candidate_source)
    assert (second["source"] / "nanochat" / "candidate.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"


def test_node_serializes_paths_outside_project(tmp_path):
    from ai_scientist.treesearch.journal import Node

    external_plot = tmp_path / "external" / "curve.png"
    node = Node(
        plan="baseline",
        code="print('ok')",
        exp_results_dir=str(tmp_path / "external"),
        plot_paths=[str(external_plot)],
        plot_analyses=[{"analysis": "stable", "plot_path": str(external_plot)}],
    )
    payload = node.to_dict()
    assert payload["exp_results_dir"]
    assert payload["plot_paths"]
    assert payload["plot_analyses"][0]["plot_path"]


def test_parallel_agent_abort_executor_reaps_workers():
    from concurrent.futures import ProcessPoolExecutor

    from ai_scientist.treesearch.parallel_agent import ParallelAgent

    executor = ProcessPoolExecutor(max_workers=1)
    future = executor.submit(time.sleep, 30)
    time.sleep(0.2)
    agent = SimpleNamespace(executor=executor, _is_shutdown=False)
    processes = list(executor._processes.values())
    try:
        ParallelAgent._abort_executor(agent)
        assert agent._is_shutdown is True
        assert all(not process.is_alive() for process in processes)
    finally:
        future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def test_interpreter_removes_provider_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    interpreter = Interpreter(
        tmp_path,
        timeout=10,
        env_vars={"NANOCHAT_TEST_VALUE": "ok"},
        sanitize_environment=True,
    )
    try:
        result = interpreter.run(
            "import os; assert 'OPENCODE_API_KEY' not in os.environ; "
            "assert 'OPENAI_API_KEY' not in os.environ; "
            "assert 'OPENROUTER_API_KEY' not in os.environ; "
            "assert 'AWS_SECRET_ACCESS_KEY' not in os.environ; "
            "assert 'HF_TOKEN' not in os.environ; "
            "assert 'WANDB_API_KEY' not in os.environ; "
            "assert os.environ['NANOCHAT_TEST_VALUE'] == 'ok'",
            True,
        )
        assert result.exc_type is None
    finally:
        interpreter.cleanup_session()


@pytest.mark.skipif(os.name != "posix", reason="process-group assertion requires POSIX")
def test_interpreter_timeout_terminates_descendants(tmp_path):
    marker = tmp_path / "descendant.txt"
    interpreter = Interpreter(tmp_path, timeout=1, sanitize_environment=True)
    code = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', \"import time; time.sleep(3); open({str(marker)!r}, 'w').write('done')\"]); "
        "time.sleep(10)"
    )
    try:
        result = interpreter.run(code, True)
        assert result.exc_type == "TimeoutError"
        time.sleep(1)
        assert not marker.exists()
    finally:
        interpreter.cleanup_session()
