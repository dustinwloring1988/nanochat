import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from nanochat.research_results import write_json_atomic

CONTROLLER_GUARDRAILS = {
    "max_peak_vram_bytes": 17179869184,
    "max_training_seconds": 1800,
    "required_plot_count": 4,
}

FIXED_EXPERIMENT_FIELDS = {
    "trainer": "curriculum",
    "max_seq_len": 2048,
    "seed": 42,
    "model_tag": "ai-scientist-candidate",
    "eval_every": 20,
    "eval_tokens": 983040,
    "core_metric_every": 100,
    "core_metric_max_per_task": 100,
    **CONTROLLER_GUARDRAILS,
}


@dataclass
class PretrainingExperimentConfig:
    trainer: str = "curriculum"
    depth: int = 6
    aspect_ratio: int = 64
    head_dim: int = 128
    max_seq_len: int = 2048
    window_pattern: str = "SSL"
    device_batch_size: int = 24
    total_batch_size: int = 98304
    num_iterations: int = 200
    target_param_data_ratio: float = 16.0
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.008
    matrix_lr: float = 0.02
    scalar_lr: float = 0.5
    weight_decay: float = 0.28
    warmup_steps: int = 40
    warmdown_ratio: float = 0.65
    final_lr_frac: float = 0.05
    eval_every: int = 20
    eval_tokens: int = 983040
    core_metric_every: int = 100
    core_metric_max_per_task: int = 100
    seed: int = 42
    model_tag: str = "ai-scientist-candidate"
    curriculum_config: str = "config/ai_scientist_pretraining.yaml"
    fp8: bool = False
    max_peak_vram_bytes: int = 17179869184
    max_training_seconds: int = 1800
    required_plot_count: int = 4

    def validate(self):
        if self.trainer not in {"base", "curriculum"}:
            raise ValueError("trainer must be base or curriculum")
        for name in (
            "depth",
            "aspect_ratio",
            "head_dim",
            "max_seq_len",
            "device_batch_size",
            "total_batch_size",
            "num_iterations",
            "target_param_data_ratio",
            "eval_tokens",
            "core_metric_every",
            "core_metric_max_per_task",
            "max_peak_vram_bytes",
            "max_training_seconds",
            "required_plot_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.total_batch_size % (self.device_batch_size * self.max_seq_len):
            raise ValueError(
                "total_batch_size must be divisible by device batch times context"
            )
        if self.eval_tokens % (self.device_batch_size * self.max_seq_len):
            raise ValueError(
                "eval_tokens must be divisible by device batch times context"
            )
        if not 0 <= self.warmdown_ratio <= 1:
            raise ValueError("warmdown_ratio must be between zero and one")
        if not 0 <= self.final_lr_frac <= 1:
            raise ValueError("final_lr_frac must be between zero and one")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        for name in (
            "embedding_lr",
            "unembedding_lr",
            "matrix_lr",
            "scalar_lr",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if any(
            character not in {"L", "S"} for character in self.window_pattern.upper()
        ):
            raise ValueError("window_pattern may contain only L and S")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", self.model_tag):
            raise ValueError("model_tag must be a safe filename component")

    @classmethod
    def from_mapping(cls, payload: dict):
        if not isinstance(payload, dict):
            raise ValueError("Experiment configuration must be an object")
        config = cls(**payload)
        config.validate()
        return config


def validate_experiment_contract(
    candidate: PretrainingExperimentConfig,
    baseline: PretrainingExperimentConfig,
    source_dir: Path,
    *,
    allow_seed_variants: bool = False,
):
    candidate.validate()
    baseline.validate()
    for name, expected in FIXED_EXPERIMENT_FIELDS.items():
        if allow_seed_variants and name == "seed":
            continue
        actual = getattr(candidate, name)
        if actual != expected:
            raise ValueError(
                f"Experiment field {name} is fixed at {expected!r}, got {actual!r}"
            )

    expected_tokens = baseline.num_iterations * baseline.total_batch_size
    actual_tokens = candidate.num_iterations * candidate.total_batch_size
    if actual_tokens != expected_tokens:
        raise ValueError(
            f"Training token budget must remain {expected_tokens}, got {actual_tokens}"
        )

    curriculum_path = (source_dir / candidate.curriculum_config).resolve()
    if not curriculum_path.is_relative_to(source_dir):
        raise ValueError("curriculum_config must remain inside the source snapshot")
    if not curriculum_path.is_file():
        raise ValueError(f"Curriculum config not found: {curriculum_path}")
    with open(curriculum_path, "r", encoding="utf-8") as handle:
        curriculum = yaml.safe_load(handle)
    stages = curriculum.get("stages") if isinstance(curriculum, dict) else None
    if not isinstance(stages, list) or not stages:
        raise ValueError("Curriculum config must contain at least one stage")
    for stage in stages:
        context_range = stage.get("context_range") if isinstance(stage, dict) else None
        if context_range != [baseline.max_seq_len, baseline.max_seq_len]:
            raise ValueError(
                "Every curriculum stage must use the fixed baseline context"
            )


def run_from_environment(working_dir: str):
    working_path = Path(working_dir).resolve()
    source_dir = Path(os.environ["NANOCHAT_SOURCE_DIR"]).resolve()
    shared_cache = Path(os.environ["NANOCHAT_SHARED_CACHE"]).resolve()
    config_path = Path(os.environ["NANOCHAT_EXPERIMENT_CONFIG"]).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = PretrainingExperimentConfig.from_mapping(json.load(handle))
    baseline_path = Path(os.environ["NANOCHAT_BASELINE_EXPERIMENT_CONFIG"]).resolve()
    with open(baseline_path, "r", encoding="utf-8") as handle:
        baseline = PretrainingExperimentConfig.from_mapping(json.load(handle))
    validate_experiment_contract(config, baseline, source_dir)
    run_pretraining(config, source_dir, shared_cache, working_path)


def run_pretraining(
    config: PretrainingExperimentConfig,
    source_dir: Path,
    shared_cache: Path,
    working_dir: Path,
):
    config.validate()
    run_cache = working_dir / "nanochat-cache"
    prepare_run_cache(shared_cache, run_cache)
    results_path = working_dir / "results.json"
    command = build_command(config, source_dir, run_cache, results_path)
    environment = os.environ.copy()
    environment["NANOCHAT_BASE_DIR"] = str(run_cache)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    python = Path(os.environ.get("NANOCHAT_PYTHON", sys.executable))
    command[0] = str(python)
    log_path = working_dir / "training.log"
    process = None
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=source_dir,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    if return_code != 0:
        raise RuntimeError(f"nanochat training failed with exit code {return_code}")
    if not results_path.is_file():
        raise RuntimeError("nanochat training did not produce results.json")
    enrich_result_artifact(results_path, config)
    create_plots(results_path, working_dir)
    with open(
        working_dir / "resolved_experiment.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
        handle.write("\n")


def enrich_result_artifact(results_path: Path, config: PretrainingExperimentConfig):
    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {})
    expected_tokens = config.num_iterations * config.total_batch_size
    if metrics.get("total_training_tokens") != expected_tokens:
        raise ValueError(
            f"Result token budget mismatch: expected {expected_tokens}, "
            f"got {metrics.get('total_training_tokens')}"
        )
    guardrails = payload.setdefault("guardrails", {})
    guardrails.update(
        {
            "max_seq_len": config.max_seq_len,
            **CONTROLLER_GUARDRAILS,
            "required_total_training_tokens": expected_tokens,
        }
    )
    payload.setdefault("provenance", {})["experiment_model"] = os.environ.get(
        "AI_SCIENTIST_MODEL"
    )
    write_json_atomic(results_path, payload)


def prepare_run_cache(shared_cache: Path, run_cache: Path):
    if not shared_cache.is_dir():
        raise FileNotFoundError(f"Shared nanochat cache not found: {shared_cache}")
    required_files = (
        "tokenizer/tokenizer.pkl",
        "tokenizer/token_bytes.pt",
        "base_data_climbmix/shard_06542.parquet",
    )
    required_directories = (
        "curriculum_data/karpathy--climbmix-400b-shuffle/default",
        "eval_bundle",
    )
    missing = [
        relative
        for relative in required_files
        if not (shared_cache / relative).is_file()
    ]
    missing.extend(
        relative
        for relative in required_directories
        if not (shared_cache / relative).is_dir()
    )
    if not any((shared_cache / "base_data_climbmix").glob("*.parquet")):
        missing.append("base_data_climbmix/*.parquet")
    if not any(
        (shared_cache / "curriculum_data/karpathy--climbmix-400b-shuffle/default").glob(
            "*.parquet"
        )
    ):
        missing.append("curriculum_data/.../default/*.parquet")
    if missing:
        raise FileNotFoundError(
            "Shared nanochat cache is missing required assets: "
            + ", ".join(sorted(set(missing)))
        )
    run_cache.mkdir(parents=True, exist_ok=True)
    for name in (
        "tokenizer",
        "base_data_climbmix",
        "curriculum_data",
        "eval_bundle",
        "task_data",
    ):
        source = shared_cache / name
        destination = run_cache / name
        if source.exists() and not destination.exists():
            try:
                destination.symlink_to(source, target_is_directory=True)
            except OSError as exc:
                raise RuntimeError(
                    f"Unable to link shared nanochat asset {source} into isolated run cache"
                ) from exc


def build_command(
    config: PretrainingExperimentConfig,
    source_dir: Path,
    run_cache: Path,
    results_path: Path,
) -> list[str]:
    module = (
        "scripts.base_train_curriculum"
        if config.trainer == "curriculum"
        else "scripts.base_train"
    )
    command = [
        sys.executable,
        "-m",
        module,
        "--run",
        "dummy",
        "--depth",
        str(config.depth),
        "--aspect-ratio",
        str(config.aspect_ratio),
        "--head-dim",
        str(config.head_dim),
        "--max-seq-len",
        str(config.max_seq_len),
        "--window-pattern",
        str(config.window_pattern).upper(),
        "--device-batch-size",
        str(config.device_batch_size),
        "--total-batch-size",
        str(config.total_batch_size),
        "--num-iterations",
        str(config.num_iterations),
        "--target-param-data-ratio",
        str(config.target_param_data_ratio),
        "--embedding-lr",
        str(config.embedding_lr),
        "--unembedding-lr",
        str(config.unembedding_lr),
        "--matrix-lr",
        str(config.matrix_lr),
        "--scalar-lr",
        str(config.scalar_lr),
        "--weight-decay",
        str(config.weight_decay),
        "--warmup-steps",
        str(config.warmup_steps),
        "--warmdown-ratio",
        str(config.warmdown_ratio),
        "--final-lr-frac",
        str(config.final_lr_frac),
        "--eval-every",
        str(config.eval_every),
        "--eval-tokens",
        str(config.eval_tokens),
        "--core-metric-every",
        str(config.core_metric_every),
        "--core-metric-max-per-task",
        str(config.core_metric_max_per_task),
        "--sample-every",
        "-1",
        "--save-every",
        "-1",
        "--seed",
        str(config.seed),
        "--model-tag",
        config.model_tag,
        "--results-json",
        str(results_path),
    ]
    if config.trainer == "curriculum":
        curriculum_path = source_dir / config.curriculum_config
        command.extend(["--config", str(curriculum_path)])
    if config.fp8:
        command.append("--fp8")
    return command


def create_plots(results_path: Path, working_dir: Path):
    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    curves = payload["curves"]
    figures = (
        ("val_bpb", "Validation BPB", "validation_bpb.png", True),
        ("train_loss", "Training Loss", "training_loss.png", True),
        ("tokens_per_second", "Tokens per Second", "tokens_per_second.png", False),
    )
    for key, title, filename, log_scale in figures:
        values = curves.get(key, [])
        if not values:
            continue
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(curves["step"], values, linewidth=1.5)
        axis.set_title(f"nanochat {title}")
        axis.set_xlabel("Optimization step")
        axis.set_ylabel(title)
        axis.grid(alpha=0.25)
        if log_scale and all(value > 0 for value in values):
            axis.set_yscale("log")
        figure.tight_layout()
        figure.savefig(working_dir / filename, dpi=140)
        plt.close(figure)
    composition = payload.get("composition")
    if composition:
        mixture = {}
        for stage in composition["stages"]:
            for name, weight in stage["source_weights"].items():
                mixture[name] = mixture.get(name, 0.0) + float(weight) * float(
                    stage["token_ratio"]
                )
        names = sorted(mixture)
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(names, [mixture[name] for name in names])
        axis.set_title("nanochat configured source mixture")
        axis.set_ylabel("Weighted source ratio")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(working_dir / "source_mixture.png", dpi=140)
        plt.close(figure)
