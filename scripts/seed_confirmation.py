from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from ai_scientist.treesearch.nanochat_adapter import (
    attest_nanochat_artifact,
    copy_source_snapshot,
    verify_nanochat_artifact,
)
from nanochat.ai_scientist_experiment import PretrainingExperimentConfig, run_pretraining
from nanochat.research_results import sha256_file, source_hash, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential nanochat seed confirmation")
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/ai_scientist_experiment.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--mean-tolerance", type=float, default=0.02)
    parser.add_argument("--single-seed-tolerance", type=float, default=0.05)
    parser.add_argument("--experiment-model", default="controller-seed-confirmation")
    return parser.parse_args()


def parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if seeds != [42, 43, 44]:
        raise ValueError("seed confirmation requires exactly sequential seeds 42,43,44")
    return seeds


def config_hash(path: Path) -> str:
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_config(path: Path) -> tuple[dict, PretrainingExperimentConfig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, PretrainingExperimentConfig.from_mapping(payload)


def seed_environment(
    *,
    source_dir: Path,
    working_dir: Path,
    experiment_config: Path,
    baseline_config: Path,
    cache_dir: Path,
    experiment_model: str,
) -> None:
    values = {
        "NANOCHAT_SOURCE_DIR": str(source_dir),
        "NANOCHAT_WORKING_DIR": str(working_dir),
        "NANOCHAT_EXPERIMENT_CONFIG": str(experiment_config),
        "NANOCHAT_BASELINE_EXPERIMENT_CONFIG": str(baseline_config),
        "NANOCHAT_BASELINE_CONFIG_HASH": config_hash(baseline_config),
        "NANOCHAT_SHARED_CACHE": str(cache_dir),
        "AI_SCIENTIST_MODEL": experiment_model,
    }
    os.environ.update(values)


def run_seed(
    *,
    seed: int,
    baseline_payload: dict,
    baseline_config: PretrainingExperimentConfig,
    source_root: Path,
    cache_dir: Path,
    output_root: Path,
    experiment_model: str,
    key: bytes,
) -> dict:
    artifact = output_root / f"seed_{seed}"
    artifact.mkdir(parents=True, exist_ok=False)
    source_snapshot = artifact / "source"
    copy_source_snapshot(source_root, source_snapshot)
    config = replace(baseline_config, seed=seed)
    config.validate()
    experiment_path = artifact / "experiment.json"
    baseline_path = artifact / "baseline_experiment.json"
    write_json_atomic(experiment_path, asdict(config))
    write_json_atomic(baseline_path, baseline_payload)
    seed_environment(
        source_dir=source_snapshot,
        working_dir=artifact,
        experiment_config=experiment_path,
        baseline_config=baseline_path,
        cache_dir=cache_dir,
        experiment_model=experiment_model,
    )
    run_pretraining(config, source_snapshot, cache_dir, artifact)
    cfg = SimpleNamespace(
        log_dir=output_root,
        experiment=SimpleNamespace(
            model=experiment_model,
            cache_dir=cache_dir,
            require_plots=True,
        ),
    )
    node_id = f"seed-{seed}"
    attest_nanochat_artifact(
        artifact,
        cfg,
        key,
        node_id,
        "seed-confirmation",
        allow_seed_variants=True,
    )
    verify_nanochat_artifact(
        artifact,
        cfg,
        key,
        expected_node_id=node_id,
        allow_seed_variants=True,
    )
    result_path = artifact / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    attestation = json.loads(
        (artifact / "controller-attestation.json").read_text(encoding="utf-8")
    )
    return {
        "seed": seed,
        "status": "ok",
        "val_bpb": float(result["primary_metric"]["value"]),
        "result_sha256": sha256_file(result_path),
        "source_sha256": source_hash(source_snapshot),
        "attestation_signature": attestation["signature"],
        "attested_node_id": node_id,
    }


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    source_root = args.source_dir.resolve()
    cache_value = args.cache_dir or os.environ.get("NANOCHAT_SHARED_CACHE")
    if not cache_value:
        raise FileNotFoundError("NANOCHAT_SHARED_CACHE or --cache-dir is required")
    cache_dir = Path(cache_value).resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Shared nanochat cache not found: {cache_dir}")
    config_path = args.config
    if not config_path.is_absolute():
        config_path = source_root / config_path
    baseline_payload, baseline_config = load_config(config_path)
    if baseline_config.seed != 42:
        raise ValueError("baseline configuration must use seed 42")
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    key = secrets.token_bytes(32)
    records = []
    for seed in seeds:
        try:
            records.append(
                run_seed(
                    seed=seed,
                    baseline_payload=baseline_payload,
                    baseline_config=baseline_config,
                    source_root=source_root,
                    cache_dir=cache_dir,
                    output_root=output_root,
                    experiment_model=args.experiment_model,
                    key=key,
                )
            )
        except Exception as exc:
            records.append(
                {
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    valid = [record for record in records if record.get("status") == "ok"]
    baseline_record = next(
        (record for record in records if record.get("seed") == 42), None
    )
    values = [record["val_bpb"] for record in valid]
    mean = sum(values) / len(values) if values else None
    baseline_value = baseline_record.get("val_bpb") if baseline_record else None
    mean_ok = (
        mean is not None
        and baseline_value is not None
        and mean <= baseline_value + args.mean_tolerance
    )
    individual_ok = (
        len(valid) == len(seeds)
        and baseline_value is not None
        and all(
            record["val_bpb"] <= baseline_value + args.single_seed_tolerance
            for record in valid
        )
    )
    all_canonical = len(valid) == len(seeds) and all(
        record.get("attestation_signature") for record in valid
    )
    status = "passed" if all_canonical and mean_ok and individual_ok else "failed"
    payload = {
        "schema_version": 1,
        "status": status,
        "seeds": seeds,
        "mean_val_bpb": mean,
        "baseline_val_bpb": baseline_value,
        "mean_tolerance": args.mean_tolerance,
        "single_seed_tolerance": args.single_seed_tolerance,
        "checks": {
            "all_canonical_and_attested": all_canonical,
            "mean_within_tolerance": mean_ok,
            "each_seed_within_tolerance": individual_ok,
        },
        "attestation_key_id": hashlib.sha256(key).hexdigest(),
        "records": records,
    }
    write_json_atomic(output_root / "seed_confirmation.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
