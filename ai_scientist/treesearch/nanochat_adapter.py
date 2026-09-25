import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from .utils.metric import MetricValue, WorstMetricValue

SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    ".python-version",
)
SOURCE_DIRS = (
    "nanochat",
    "scripts",
    "tasks",
    "config",
)
RESULT_FILENAME = "results.json"
ATTESTATION_FILENAME = "controller-attestation.json"
ATTESTATION_SCHEMA_VERSION = 1


def prepare_node_workspace(cfg, baseline_source: str | Path | None = None) -> dict:
    task_id = uuid.uuid4().hex
    workspace = Path(cfg.workspace_dir) / "nodes" / task_id
    source = workspace / "source"
    working = workspace / "working"
    workspace.mkdir(parents=True, exist_ok=False)
    working.mkdir()
    base_source_root = Path(cfg.experiment.source_dir).resolve()
    source_root = base_source_root
    if baseline_source is not None:
        parent_source = Path(baseline_source).resolve()
        artifact_root = Path(cfg.log_dir).resolve()
        if not parent_source.is_dir() or not parent_source.is_relative_to(
            artifact_root
        ):
            raise ValueError(
                "Parent candidate source is outside the experiment artifact root"
            )
        source_root = parent_source
    copy_source_snapshot(source_root, source)
    profile_path = Path(cfg.experiment.config_file).resolve()
    if not profile_path.is_relative_to(source_root):
        relative_profile = profile_path.relative_to(base_source_root)
        profile_path = source_root / relative_profile
    if not profile_path.is_relative_to(source_root):
        raise ValueError("Experiment configuration must be inside the source root")
    with open(profile_path, "r", encoding="utf-8") as handle:
        profile_payload = json.load(handle)

    from nanochat.ai_scientist_experiment import PretrainingExperimentConfig

    resolved_config = PretrainingExperimentConfig.from_mapping(profile_payload)
    serialized_config = (
        json.dumps(asdict(resolved_config), indent=2, sort_keys=True) + "\n"
    )
    experiment_config = working / "experiment.json"
    baseline_config = workspace / "baseline_experiment.json"
    experiment_config.write_text(serialized_config, encoding="utf-8")
    baseline_config.write_text(serialized_config, encoding="utf-8")
    config_hash = hashlib.sha256(serialized_config.encode("utf-8")).hexdigest()
    return {
        "task_id": task_id,
        "workspace": workspace,
        "source": source,
        "working": working,
        "experiment_config": experiment_config,
        "baseline_config": baseline_config,
        "config_hash": config_hash,
    }


def copy_source_snapshot(source_root: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=False)
    for directory in SOURCE_DIRS:
        source = source_root / directory
        if source.exists():
            shutil.copytree(
                source,
                destination / directory,
                ignore=shutil.ignore_patterns(
                    "__pycache__",
                    "*.pyc",
                    "*.pyo",
                    "*.pt",
                    "*.pth",
                    "*.ckpt",
                ),
            )
    for filename in SOURCE_FILES:
        source = source_root / filename
        if source.is_file():
            shutil.copy2(source, destination / filename)


def experiment_env(cfg, paths: dict, gpu_id: int | None) -> dict:
    env = {
        "NANOCHAT_SOURCE_DIR": str(paths["source"]),
        "NANOCHAT_WORKSPACE_DIR": str(paths["workspace"]),
        "NANOCHAT_WORKING_DIR": str(paths["working"]),
        "NANOCHAT_RESULTS_PATH": str(paths["working"] / RESULT_FILENAME),
        "NANOCHAT_EXPERIMENT_CONFIG": str(paths["experiment_config"]),
        "NANOCHAT_BASELINE_EXPERIMENT_CONFIG": str(paths["baseline_config"]),
        "NANOCHAT_BASELINE_CONFIG_HASH": paths["config_hash"],
        "NANOCHAT_EXPERIMENT_MODEL": cfg.experiment.model,
        "AI_SCIENTIST_MODEL": cfg.experiment.model,
        "NANOCHAT_PYTHON": os.environ.get("NANOCHAT_PYTHON", sys.executable),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    if cfg.experiment.cache_dir is not None:
        env["NANOCHAT_SHARED_CACHE"] = str(Path(cfg.experiment.cache_dir).resolve())
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def apply_nanochat_result(node, exec_result, cfg, paths: dict):
    node.absorb_exec_result(exec_result)
    execution_error = exec_result.exc_type
    node.is_buggy = execution_error is not None
    node.analysis = (
        f"Experiment execution failed with {execution_error}: "
        f"{exec_result.exc_info}"
        if node.is_buggy
        else "Canonical nanochat result received"
    )
    result_path = paths["working"] / RESULT_FILENAME
    if not result_path.is_file():
        node.metric = WorstMetricValue()
        node.is_buggy = True
        node.is_buggy_plots = True
        collect_artifacts(
            node, cfg, paths, result_path if result_path.exists() else None
        )
        return

    try:
        payload = load_canonical_result(result_path)
        validate_nanochat_result(payload, cfg, paths)
    except Exception as exc:
        node.metric = WorstMetricValue()
        node.is_buggy = True
        node.is_buggy_plots = True
        node.analysis = f"Invalid canonical result: {exc}"
        collect_artifacts(node, cfg, paths, result_path)
        return

    value = payload["primary_metric"]["value"]
    node.metric = MetricValue(
        value={
            "metric_names": [
                {
                    "metric_name": "val_bpb",
                    "lower_is_better": True,
                    "description": "Held-out validation bits per byte at a fixed token budget",
                    "data": [
                        {
                            "dataset_name": "nanochat",
                            "final_value": value,
                            "best_value": payload["metrics"].get("best_val_bpb", value),
                        }
                    ],
                }
            ]
        }
    )
    plots = sorted(paths["working"].glob("*.png"))
    node.is_buggy_plots = not plots
    node.analysis = (
        f"val_bpb={value:.6f}, runtime={payload['metrics']['training_time_seconds']:.2f}s, "
        f"tokens_per_second={payload['metrics']['tokens_per_second']:.1f}"
    )
    if execution_error:
        node.is_buggy = False
        node.analysis += f"; post-result execution error ignored: {execution_error}"
    collect_artifacts(node, cfg, paths, result_path)


def load_canonical_result(path: Path) -> dict:
    from nanochat.research_results import load_results

    return load_results(path)


def _attestation_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _artifact_paths(artifact_dir: Path) -> dict:
    artifact_dir = artifact_dir.resolve()
    return {
        "source": artifact_dir / "source",
        "working": artifact_dir,
        "experiment_config": artifact_dir / "experiment.json",
        "baseline_config": artifact_dir / "baseline_experiment.json",
    }


def _artifact_config_hash(paths: dict) -> str:
    content = paths["baseline_config"].read_text(encoding="utf-8")
    return hashlib.sha256(content.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _validated_nanochat_artifact(artifact_dir: Path, cfg) -> tuple[dict, dict, Path]:
    artifact_dir = Path(artifact_dir).resolve()
    artifact_root = Path(cfg.log_dir).resolve()
    if not artifact_dir.is_relative_to(artifact_root):
        raise ValueError("Nanochat artifact is outside the controller log root")
    paths = _artifact_paths(artifact_dir)
    result_path = paths["working"] / RESULT_FILENAME
    payload = load_canonical_result(result_path)
    paths["config_hash"] = _artifact_config_hash(paths)
    validate_nanochat_result(payload, cfg, paths)
    return payload, paths, result_path


def verify_nanochat_artifact(
    artifact_dir: Path, cfg, key: bytes, expected_node_id: str | None = None
) -> dict:
    from nanochat.research_results import source_hash

    payload, paths, result_path = _validated_nanochat_artifact(artifact_dir, cfg)
    attestation_path = paths["working"] / ATTESTATION_FILENAME
    with open(attestation_path, "r", encoding="utf-8") as handle:
        attestation = json.load(handle)
    if attestation.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        raise ValueError("Unsupported controller attestation version")
    attested = attestation.get("payload")
    signature = attestation.get("signature")
    if not isinstance(attested, dict) or not isinstance(signature, str):
        raise ValueError("Controller attestation is malformed")
    expected_signature = hmac.new(
        key, _attestation_bytes(attested), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Controller attestation signature is invalid")
    if (
        attested.get("result_sha256")
        != hashlib.sha256(result_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Controller attestation result hash is stale")
    if attested.get("source_sha256") != source_hash(paths["source"]):
        raise ValueError("Controller attestation source hash is stale")
    if attested.get("baseline_config_hash") != paths["config_hash"]:
        raise ValueError("Controller attestation config hash is stale")
    if attested.get("experiment_model") != cfg.experiment.model:
        raise ValueError("Controller attestation model is inconsistent")
    if expected_node_id is not None and attested.get("node_id") != expected_node_id:
        raise ValueError("Controller attestation node is inconsistent")
    return payload


def attest_nanochat_artifact(
    artifact_dir: Path,
    cfg,
    key: bytes,
    node_id: str,
    task_id: str,
) -> dict:
    from nanochat.research_results import source_hash, write_json_atomic

    payload, paths, result_path = _validated_nanochat_artifact(artifact_dir, cfg)
    paths = _artifact_paths(Path(artifact_dir).resolve())
    attested = {
        "node_id": node_id,
        "task_id": task_id,
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "source_sha256": source_hash(paths["source"]),
        "baseline_config_hash": _artifact_config_hash(paths),
        "experiment_model": cfg.experiment.model,
        "controller_pid": os.getpid(),
    }
    write_json_atomic(
        paths["working"] / ATTESTATION_FILENAME,
        {
            "schema_version": ATTESTATION_SCHEMA_VERSION,
            "payload": attested,
            "signature": hmac.new(
                key, _attestation_bytes(attested), hashlib.sha256
            ).hexdigest(),
        },
    )
    return payload


def validate_nanochat_result(payload: dict, cfg, paths: dict) -> None:
    from nanochat.ai_scientist_experiment import (
        CONTROLLER_GUARDRAILS,
        PretrainingExperimentConfig,
        validate_experiment_contract,
    )
    from nanochat.research_results import (
        dataset_manifest_hash,
        runtime_metadata,
        source_hash,
        source_identity,
        tokenizer_hash,
    )

    with open(paths["experiment_config"], "r", encoding="utf-8") as handle:
        candidate = PretrainingExperimentConfig.from_mapping(json.load(handle))
    with open(paths["baseline_config"], "r", encoding="utf-8") as handle:
        baseline = PretrainingExperimentConfig.from_mapping(json.load(handle))
    validate_experiment_contract(candidate, baseline, paths["source"].resolve())

    metrics = payload["metrics"]
    expected_tokens = candidate.num_iterations * candidate.total_batch_size
    if metrics["total_training_tokens"] != expected_tokens:
        raise ValueError("Result training token budget does not match experiment")
    if metrics["peak_vram_bytes"] > CONTROLLER_GUARDRAILS["max_peak_vram_bytes"]:
        raise ValueError("Result exceeded the peak VRAM guardrail")
    if metrics["training_time_seconds"] > CONTROLLER_GUARDRAILS["max_training_seconds"]:
        raise ValueError("Result exceeded the training-time guardrail")
    if (
        metrics["wall_clock_training_seconds"]
        > CONTROLLER_GUARDRAILS["max_training_seconds"]
    ):
        raise ValueError("Result exceeded the wall-clock training-time guardrail")
    if metrics["active_training_time_seconds"] > metrics["wall_clock_training_seconds"]:
        raise ValueError("Active training time cannot exceed wall-clock time")

    guardrails = payload["guardrails"]
    expected_guardrails = {
        "max_seq_len": candidate.max_seq_len,
        **CONTROLLER_GUARDRAILS,
        "required_total_training_tokens": expected_tokens,
    }
    for name, expected in expected_guardrails.items():
        if guardrails.get(name) != expected:
            raise ValueError(f"Result guardrail {name} is inconsistent")

    provenance = payload["provenance"]
    if provenance["seed"] != candidate.seed:
        raise ValueError("Result seed does not match experiment")
    if provenance.get("experiment_model") != cfg.experiment.model:
        raise ValueError("Result experiment model does not match BFTS configuration")
    if provenance.get("baseline_config_hash") != paths["config_hash"]:
        raise ValueError("Result baseline configuration hash does not match node")
    if provenance["source_hash"] != source_hash(paths["source"]):
        raise ValueError("Result source hash does not match candidate snapshot")
    if provenance.get("source_revision") != source_identity(paths["source"]):
        raise ValueError("Result source revision does not match candidate snapshot")
    if provenance.get("model_tag") != candidate.model_tag:
        raise ValueError("Result model tag does not match experiment")
    cache_dir = Path(cfg.experiment.cache_dir).resolve()
    expected_tokenizer_hash = tokenizer_hash(cache_dir)
    expected_dataset_hash = dataset_manifest_hash(cache_dir)
    if expected_tokenizer_hash is None:
        raise ValueError("Shared cache has no tokenizer identity")
    if provenance.get("tokenizer_hash") != expected_tokenizer_hash:
        raise ValueError("Result tokenizer hash does not match shared cache")
    if provenance.get("dataset_manifest_hash") != expected_dataset_hash:
        raise ValueError("Result dataset manifest hash does not match shared cache")
    expected_runtime = runtime_metadata()
    actual_runtime = provenance.get("runtime", {})
    for name in ("python", "torch", "cuda", "cuda_available", "device"):
        if actual_runtime.get(name) != expected_runtime.get(name):
            raise ValueError(f"Result runtime field {name} does not match controller")
    user_config = provenance["user_config"]
    for name in (
        "max_seq_len",
        "num_iterations",
        "total_batch_size",
        "device_batch_size",
        "eval_every",
        "eval_tokens",
        "seed",
    ):
        if user_config.get(name) != getattr(candidate, name):
            raise ValueError(f"Result user_config.{name} does not match experiment")
    if candidate.trainer == "curriculum":
        composition = payload.get("composition")
        if not isinstance(composition, dict):
            raise ValueError("Curriculum results require composition metadata")
        if not (paths["working"] / "source_mixture.png").is_file():
            raise ValueError("Curriculum results require source_mixture.png")
        import yaml

        with open(
            paths["source"] / candidate.curriculum_config, "r", encoding="utf-8"
        ) as handle:
            curriculum_payload = yaml.safe_load(handle)
        expected_stages = curriculum_payload.get("stages", [])
        actual_stages = composition.get("stages", [])
        if len(expected_stages) != len(actual_stages):
            raise ValueError("Composition stage count does not match curriculum")
        for expected, actual in zip(expected_stages, actual_stages):
            for field in ("name", "token_ratio", "context_range", "source_weights"):
                if actual.get(field) != expected.get(field):
                    raise ValueError(f"Composition stage field {field} is inconsistent")

    plot_count = len(list(paths["working"].glob("*.png")))
    minimum_plots = (
        CONTROLLER_GUARDRAILS["required_plot_count"]
        if cfg.experiment.require_plots
        else 0
    )
    if plot_count < minimum_plots:
        raise ValueError("Result is missing required plots")


def collect_artifacts(node, cfg, paths: dict, result_path: Path | None):
    output_dir = (
        Path(cfg.log_dir) / "experiment_results" / f"node_{node.id}_{paths['task_id']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "experiment_code.py", "w", encoding="utf-8") as handle:
        handle.write(node.code)
    candidate_source = output_dir / "source"
    shutil.copytree(
        paths["source"],
        candidate_source,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.pyo", "*.pt", "*.pth", "*.ckpt"
        ),
    )
    node.candidate_source = str(candidate_source)
    if result_path is not None:
        shutil.copy2(result_path, output_dir / RESULT_FILENAME)
    experiment_config = paths["experiment_config"]
    if experiment_config.is_file():
        shutil.copy2(experiment_config, output_dir / experiment_config.name)
    shutil.copy2(paths["baseline_config"], output_dir / "baseline_experiment.json")
    patch_path = output_dir / "candidate.patch"
    patch_path.write_text(
        source_diff(Path(cfg.experiment.source_dir).resolve(), paths["source"]),
        encoding="utf-8",
    )
    node.exp_results_dir = str(output_dir)
    node.plots = []
    node.plot_paths = []
    for plot in sorted(paths["working"].glob("*.png")):
        destination = output_dir / plot.name
        shutil.copy2(plot, destination)
        node.plots.append(str(destination))
        node.plot_paths.append(str(destination))


def source_diff(source_root: Path, candidate_root: Path) -> str:
    chunks = []
    targets = (*SOURCE_DIRS, *SOURCE_FILES)
    for target in targets:
        source = source_root / target
        candidate = candidate_root / target
        if not source.exists() or not candidate.exists():
            continue
        result = subprocess.run(
            ["git", "diff", "--no-index", "--", str(source), str(candidate)],
            capture_output=True,
            text=True,
        )
        if result.returncode not in {0, 1}:
            break
        chunk = result.stdout.replace(source_root.as_posix(), "a").replace(
            candidate_root.as_posix(), "b"
        )
        chunks.append(chunk)
    else:
        return "\n".join(chunks)
    digest = hashlib.sha256()
    for path in sorted(candidate_root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(candidate_root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return f"Unable to create textual diff. Candidate tree hash: {digest.hexdigest()}\n"
