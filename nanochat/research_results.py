import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = 2
SOURCE_DIRS = ("nanochat", "scripts", "tasks", "config")
SOURCE_FILES = ("pyproject.toml", "uv.lock", ".python-version")
REQUIRED_CURVES = ("step", "train_loss", "val_bpb", "tokens_per_second")
REQUIRED_METRICS = (
    "val_bpb",
    "best_val_bpb",
    "core_metric",
    "final_train_loss",
    "training_time_seconds",
    "active_training_time_seconds",
    "wall_clock_training_seconds",
    "tokens_per_second",
    "active_tokens_per_second",
    "wall_clock_tokens_per_second",
    "mfu_percent",
    "peak_vram_bytes",
    "total_training_tokens",
    "parameter_count",
)


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash(root: str | os.PathLike) -> str:
    root_path = Path(root).resolve()
    digest = hashlib.sha256()
    for directory in SOURCE_DIRS:
        base = root_path / directory
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            relative = path.relative_to(root_path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(sha256_file(path).encode("ascii"))
    for filename in SOURCE_FILES:
        path = root_path / filename
        if not path.is_file():
            continue
        digest.update(filename.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def repository_hash(root: str | os.PathLike) -> str:
    root_path = Path(root).resolve()
    excluded_names = {
        ".git",
        ".venv",
        ".venv-ai-scientist",
        "data",
        "experiments",
        "ai_scientist_data",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    digest = hashlib.sha256()
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or any(part in excluded_names for part in path.parts):
            continue
        if path.name in {".env", "Thumbs.db"} or path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def git_commit(root: str | os.PathLike) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def source_identity(root: str | os.PathLike) -> str:
    commit = git_commit(root)
    if commit:
        return f"git:{commit}"
    return f"source-sha256:{source_hash(root)}"


RESUME_CONTRACT_VERSION = 1
RESUME_CONTRACT_FIELDS = (
    "model_config",
    "seed",
    "world_size",
    "rank",
    "token_horizon",
    "device_batch_size",
    "max_seq_len",
    "total_batch_size",
    "curriculum_state",
    "dataloader_config",
    "dataset_manifest_hash",
    "tokenizer_hash",
)


def build_resume_contract(
    model_config: dict,
    seed: int,
    world_size: int,
    rank: int,
    token_horizon: int,
    device_batch_size: int,
    max_seq_len: int,
    total_batch_size: int,
    curriculum_state: dict | None,
    dataloader_config: dict,
    dataset_manifest_hash: str,
    tokenizer_hash: str,
) -> dict:
    if not dataset_manifest_hash or not tokenizer_hash:
        raise ValueError("Resume contract requires dataset and tokenizer hashes")
    return {
        "version": RESUME_CONTRACT_VERSION,
        "model_config": model_config,
        "seed": int(seed),
        "world_size": int(world_size),
        "rank": int(rank),
        "token_horizon": int(token_horizon),
        "device_batch_size": int(device_batch_size),
        "max_seq_len": int(max_seq_len),
        "total_batch_size": int(total_batch_size),
        "curriculum_state": curriculum_state,
        "dataloader_config": dataloader_config,
        "dataset_manifest_hash": dataset_manifest_hash,
        "tokenizer_hash": tokenizer_hash,
    }


def validate_resume_contract(saved: dict, current: dict) -> None:
    if not isinstance(saved, dict) or saved.get("version") != RESUME_CONTRACT_VERSION:
        raise ValueError(
            "Checkpoint resume contract version is missing or incompatible"
        )
    if (
        not isinstance(current, dict)
        or current.get("version") != RESUME_CONTRACT_VERSION
    ):
        raise ValueError("Current resume contract is invalid")
    for field in RESUME_CONTRACT_FIELDS:
        if field not in saved or field not in current:
            raise ValueError(f"Resume contract is missing {field}")
        saved_value = json.dumps(saved[field], sort_keys=True, default=str)
        current_value = json.dumps(current[field], sort_keys=True, default=str)
        if saved_value != current_value:
            raise ValueError(f"Resume contract mismatch: {field}")


def tokenizer_hash(base_dir: str | os.PathLike) -> str | None:
    tokenizer_dir = Path(base_dir) / "tokenizer"
    paths = [tokenizer_dir / "tokenizer.pkl", tokenizer_dir / "token_bytes.pt"]
    if not all(path.is_file() for path in paths):
        return None
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def dataset_manifest_hash(
    base_dir: str | os.PathLike,
    directories: tuple[str, ...] = (
        "base_data_climbmix",
        "curriculum_data",
        "eval_bundle",
        "task_data",
    ),
) -> str:
    base_path = Path(os.environ.get("NANOCHAT_SHARED_CACHE", base_dir)).resolve()
    digest = hashlib.sha256()
    entries = []
    for directory in directories:
        root = base_path / directory
        if not root.exists():
            continue
        entries.extend(path for path in root.rglob("*") if path.is_file())
    for path in sorted(entries):
        relative = path.relative_to(base_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def write_json_atomic(path: str | os.PathLike, payload: dict):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, output_path)


def runtime_metadata() -> dict:
    metadata = {"python": platform.python_version()}
    try:
        import torch

        metadata.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
                "device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            }
        )
    except ImportError:
        metadata["torch"] = None
    return metadata


def build_training_results(
    root: str | os.PathLike,
    base_dir: str | os.PathLike,
    seed: int,
    model_tag: str,
    user_config: dict,
    metrics: dict[str, Any],
    curves: dict[str, list[float]],
    guardrails: dict[str, Any] | None = None,
    composition: dict[str, Any] | None = None,
) -> dict:
    required = REQUIRED_METRICS
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"Missing required training metrics: {', '.join(missing)}")
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "ok",
        "primary_metric": {
            "name": "val_bpb",
            "value": float(metrics["val_bpb"]),
            "lower_is_better": True,
        },
        "metrics": metrics,
        "curves": curves,
        "composition": composition,
        "guardrails": guardrails or {},
        "provenance": {
            "git_commit": git_commit(root),
            "source_revision": source_identity(root),
            "source_hash": source_hash(root),
            "tokenizer_hash": tokenizer_hash(base_dir),
            "dataset_manifest_hash": dataset_manifest_hash(base_dir),
            "seed": int(seed),
            "model_tag": model_tag,
            "experiment_model": os.environ.get("AI_SCIENTIST_MODEL"),
            "baseline_config_hash": os.environ.get("NANOCHAT_BASELINE_CONFIG_HASH"),
            "runtime": runtime_metadata(),
            "user_config": user_config,
        },
    }
    validate_results(payload)
    return payload


def _validate_composition(composition: dict) -> None:
    if composition.get("schema_version") != 1:
        raise ValueError("Unsupported composition schema version")
    stages = composition.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Composition must contain stages")
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("Composition stage must be an object")
        if not isinstance(stage.get("name"), str) or not stage["name"]:
            raise ValueError("Composition stage name is required")
        token_ratio = stage.get("token_ratio")
        if (
            not isinstance(token_ratio, (int, float))
            or isinstance(token_ratio, bool)
            or not math.isfinite(token_ratio)
            or not 0 < token_ratio <= 1
        ):
            raise ValueError("Composition stage token_ratio is invalid")
        context_range = stage.get("context_range")
        if (
            not isinstance(context_range, list)
            or len(context_range) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in context_range
            )
            or context_range[0] > context_range[1]
        ):
            raise ValueError("Composition stage context_range is invalid")
        source_weights = stage.get("source_weights")
        if not isinstance(source_weights, dict) or not source_weights:
            raise ValueError("Composition stage source_weights are required")
        if any(
            not isinstance(name, str)
            or not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight < 0
            for name, weight in source_weights.items()
        ):
            raise ValueError("Composition stage source weights are invalid")
    for name in ("observed_source_tokens", "observed_document_counts"):
        observed = composition.get(name, {})
        if not isinstance(observed, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for key, value in observed.items()
        ):
            raise ValueError(f"Composition {name} is invalid")
    if composition.get("measurement_status") not in {"observed", "not_recorded"}:
        raise ValueError("Composition measurement_status is invalid")


def build_stage_composition(
    stages: list,
    observed_source_tokens: dict[str, int] | None = None,
    observed_document_counts: dict[str, int] | None = None,
    measurement_status: str = "observed",
) -> dict:
    return {
        "schema_version": 1,
        "stages": [
            {
                "name": stage.name,
                "token_ratio": float(stage.token_ratio),
                "context_range": list(stage.context_range),
                "source_weights": {
                    name: float(weight) for name, weight in stage.source_weights.items()
                },
                "subset_weights": {
                    source: {name: float(weight) for name, weight in weights.items()}
                    for source, weights in stage.subset_weights.items()
                },
            }
            for stage in stages
        ],
        "observed_source_tokens": dict(observed_source_tokens or {}),
        "observed_document_counts": dict(observed_document_counts or {}),
        "measurement_status": measurement_status,
    }


def validate_results(payload: dict):
    if not isinstance(payload, dict):
        raise ValueError("Result payload must be an object")
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Unsupported result schema version")
    if payload.get("status") != "ok":
        raise ValueError("Result status is not ok")
    primary = payload.get("primary_metric")
    if not isinstance(primary, dict) or primary.get("name") != "val_bpb":
        raise ValueError("Primary metric must be val_bpb")
    if primary.get("lower_is_better") is not True:
        raise ValueError("val_bpb must be lower-is-better")
    value = primary.get("value")
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise ValueError("Primary metric must be finite")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("Result metrics must be an object")
    for name in REQUIRED_METRICS:
        metric = metrics.get(name)
        if name == "core_metric":
            valid = (
                isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(metric)
            )
        elif name == "mfu_percent":
            valid = metric is None or (
                isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(metric)
                and metric >= 0
            )
        else:
            valid = (
                isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(metric)
                and metric > 0
            )
        if not valid:
            if name == "core_metric":
                message = "Metric core_metric must be finite"
            elif name == "mfu_percent":
                message = "Metric mfu_percent must be non-negative and finite"
            else:
                message = f"Metric {name} must be positive and finite"
            raise ValueError(message)
    if metrics["val_bpb"] != value:
        raise ValueError("Primary metric must equal metrics.val_bpb")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Result provenance must be an object")
    for name in (
        "source_revision",
        "source_hash",
        "tokenizer_hash",
        "dataset_manifest_hash",
        "seed",
        "model_tag",
    ):
        if provenance.get(name) is None:
            raise ValueError(f"Missing provenance field {name}")
    if (
        not isinstance(provenance.get("source_revision"), str)
        or not provenance["source_revision"]
    ):
        raise ValueError("Provenance source_revision must be a non-empty string")
    if not isinstance(provenance.get("user_config"), dict):
        raise ValueError("Provenance user_config must be an object")
    if not isinstance(provenance.get("runtime"), dict):
        raise ValueError("Provenance runtime must be an object")
    guardrails = payload.get("guardrails")
    if not isinstance(guardrails, dict):
        raise ValueError("Result guardrails must be an object")
    composition = payload.get("composition")
    if composition is not None:
        _validate_composition(composition)
    curves = payload.get("curves")
    if not isinstance(curves, dict):
        raise ValueError("Result curves must be an object")
    for name in REQUIRED_CURVES:
        if name not in curves or not curves[name]:
            raise ValueError(f"Result curve {name} must be non-empty")
    lengths = {name: len(series) for name, series in curves.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError("Result curve lengths must match")
    for name, series in curves.items():
        if not isinstance(series, list):
            raise ValueError(f"Result curve {name} must be a list")
        for point in series:
            if (
                not isinstance(point, (int, float))
                or isinstance(point, bool)
                or not math.isfinite(point)
            ):
                raise ValueError(f"Result curve {name} contains a non-finite value")
    steps = curves["step"]
    if any(
        not isinstance(step, int) or isinstance(step, bool) for step in steps
    ) or any(right <= left for left, right in zip(steps, steps[1:])):
        raise ValueError("Result curve steps must be strictly increasing integers")
    if curves["val_bpb"][-1] != metrics["val_bpb"]:
        raise ValueError("Final validation curve point must equal metrics.val_bpb")


def load_results(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_results(payload)
    return payload
