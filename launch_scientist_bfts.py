import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

if not os.environ.get("AI_SCIENTIST_RUN_ID"):
    os.environ["AI_SCIENTIST_RUN_ID"] = uuid.uuid4().hex

from ai_scientist.providers import (
    DEFAULT_MODEL,
    configure_provider_tracing,
    llm_budget,
    preflight_model,
)
from ai_scientist.trace_writer import TraceConfig
from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file, idea_to_markdown
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.utils.token_tracker import token_tracker
from nanochat.research_results import (
    dataset_manifest_hash,
    repository_hash,
    source_hash,
    tokenizer_hash,
    write_json_atomic,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run nanochat AI Scientist experiments"
    )
    parser.add_argument(
        "--load-ideas",
        default="ai_scientist/ideas/nanochat_pretraining.json",
    )
    parser.add_argument("--load-code", action="store_true")
    parser.add_argument("--idea-idx", type=int, default=0)
    parser.add_argument("--attempt-id", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--config", default="bfts_config.yaml")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=1,
        help="Maximum executed training nodes; nanochat currently requires 1",
    )
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--allow-provider-calls",
        action="store_true",
        help="Explicitly allow provider preflight and live LLM calls",
    )
    parser.add_argument("--parent-journal")
    parser.add_argument("--parent-node-id")
    parser.add_argument("--parent-stage", type=int, choices=(1, 2))
    parser.add_argument("--lineage-id")
    return parser.parse_args()


def run_preflight(model: str) -> dict:
    try:
        return preflight_model(model)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def save_token_tracker(idea_dir: Path):
    payload = {
        "models": token_tracker.get_summary(),
        "provider_budget": llm_budget.snapshot(),
        "interactions_captured": False,
    }
    with open(idea_dir / "token_tracker.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_preflight_evidence(
    idea_dir: Path,
    result: dict,
    *,
    model: str,
    run_id: str,
) -> None:
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "model": model,
        "allow_missing_usage": os.environ.get(
            "AI_SCIENTIST_ALLOW_MISSING_USAGE", "0"
        ).strip().lower()
        in {"1", "true", "yes"},
        "budget": llm_budget.snapshot(),
        "result": result,
    }
    write_json_atomic(idea_dir / "preflight.json", payload)


def save_run_status(
    idea_dir: Path,
    status: str,
    *,
    model: str,
    run_id: str,
    max_nodes: int,
    error: str | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "run_id": run_id,
        "model": model,
        "max_nodes": max_nodes,
        "provider_budget": llm_budget.snapshot(),
        "allow_missing_usage": os.environ.get(
            "AI_SCIENTIST_ALLOW_MISSING_USAGE", "0"
        ).strip().lower()
        in {"1", "true", "yes"},
    }
    if error is not None:
        payload["error"] = error
    write_json_atomic(idea_dir / "run_status.json", payload)


def trace_config_from_run_config(config_path: Path, workspace: Path):
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    trace = payload.get("trace", {}) if isinstance(payload, dict) else {}
    if not isinstance(trace, dict) or not trace.get("enabled", False):
        return None
    cache_dir = os.environ.get("NANOCHAT_SHARED_CACHE")
    if not cache_dir:
        raise ValueError("NANOCHAT_SHARED_CACHE is required when tracing is enabled")
    return TraceConfig(
        run_workspace=workspace,
        run_id=trace.get("run_id") or "preflight",
        enabled=True,
        retention_seconds=trace.get("retention_seconds"),
        max_bytes=trace.get("max_bytes"),
        cache_dir=cache_dir,
    )


def build_integrity_manifest(project_root: Path, cache_dir: Path) -> dict:
    return {
        "repository_sha256": repository_hash(project_root),
        "training_source_sha256": source_hash(project_root),
        "cache_dataset_sha256": dataset_manifest_hash(cache_dir),
        "tokenizer_sha256": tokenizer_hash(cache_dir),
    }


def main():
    args = parse_args()
    if args.preflight:
        result = run_preflight(args.model)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result.get("ok"):
            raise SystemExit(1)
        return
    if args.max_nodes != 1:
        raise ValueError("nanochat currently requires --max-nodes 1")
    if not args.allow_provider_calls:
        raise RuntimeError(
            "Provider calls are disabled by default; pass --allow-provider-calls explicitly"
        )

    project_root = Path(__file__).resolve().parent
    handoff_values = (
        args.parent_journal,
        args.parent_node_id,
        args.parent_stage,
        args.lineage_id,
    )
    if any(value is not None for value in handoff_values) and not all(
        value is not None for value in handoff_values
    ):
        raise ValueError("lineage handoff arguments must be provided together")
    ideas_path = Path(args.load_ideas)
    if not ideas_path.is_absolute():
        ideas_path = project_root / ideas_path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    experiment_root = Path(
        os.environ.get("AI_SCIENTIST_EXPERIMENT_DIR", project_root / "experiments")
    )
    os.environ.setdefault("AI_SCIENTIST_EXPERIMENT_DIR", str(experiment_root))
    os.environ["AI_SCIENTIST_ROOT"] = str(project_root)
    with open(ideas_path, "r", encoding="utf-8") as handle:
        ideas = json.load(handle)
    if args.idea_idx < 0 or args.idea_idx >= len(ideas):
        raise ValueError(f"idea_idx {args.idea_idx} is outside the loaded idea list")
    idea = ideas[args.idea_idx]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_idea_name = re.sub(
        r"[^A-Za-z0-9_-]+", "_", str(idea.get("Name", "experiment"))
    ).strip("_")
    if not safe_idea_name:
        raise ValueError("Idea Name must contain a usable filename component")
    idea_dir = (
        experiment_root / f"{timestamp}_{safe_idea_name}_attempt_{args.attempt_id}"
    )
    idea_dir.mkdir(parents=True, exist_ok=False)
    run_id = os.environ.get("AI_SCIENTIST_RUN_ID", "unknown")
    save_run_status(
        idea_dir,
        "created",
        model=args.model,
        run_id=run_id,
        max_nodes=args.max_nodes,
    )
    idea_json_path = idea_dir / "idea.json"
    idea_markdown_path = idea_dir / "idea.md"

    code_path = None
    if args.load_code:
        code_path = str(ideas_path.with_suffix(".py"))
        with open(code_path, "r", encoding="utf-8") as handle:
            idea["Code"] = handle.read()

    idea_to_markdown(idea, str(idea_markdown_path), code_path)
    with open(idea_json_path, "w", encoding="utf-8") as handle:
        json.dump(idea, handle, indent=2)

    run_config_path = edit_bfts_config_file(
        str(config_path),
        str(idea_dir),
        str(idea_json_path.resolve()),
        model=args.model,
        max_nodes=args.max_nodes,
    )
    preflight_trace = trace_config_from_run_config(Path(run_config_path), idea_dir)
    if preflight_trace is not None:
        configure_provider_tracing(preflight_trace)
    try:
        preflight = run_preflight(args.model)
    finally:
        if preflight_trace is not None:
            configure_provider_tracing(None)
    save_preflight_evidence(
        idea_dir,
        preflight,
        model=args.model,
        run_id=run_id,
    )
    preflight_status = "preflight_passed" if preflight.get("ok") else "preflight_failed"
    save_run_status(
        idea_dir,
        preflight_status,
        model=args.model,
        run_id=run_id,
        max_nodes=args.max_nodes,
        error=None if preflight.get("ok") else str(preflight.get("error", "preflight failed")),
    )
    if not preflight.get("ok"):
        raise RuntimeError(
            "Provider preflight failed: "
            + str(preflight.get("error", "selected model is unavailable"))
        )
    cache_dir = Path(os.environ["NANOCHAT_SHARED_CACHE"]).resolve()
    pre_manifest = build_integrity_manifest(project_root, cache_dir)
    write_json_atomic(idea_dir / "integrity_before.json", pre_manifest)
    save_run_status(
        idea_dir,
        "running",
        model=args.model,
        run_id=run_id,
        max_nodes=args.max_nodes,
    )
    try:
        perform_experiments_bfts(
            run_config_path,
            parent_journal_path=args.parent_journal,
            parent_node_id=args.parent_node_id,
            parent_stage=args.parent_stage,
            lineage_id=args.lineage_id,
        )
        post_manifest = build_integrity_manifest(project_root, cache_dir)
        write_json_atomic(idea_dir / "integrity_after.json", post_manifest)
        if pre_manifest != post_manifest:
            raise RuntimeError("Project or shared-cache integrity changed during BFTS")
        save_token_tracker(idea_dir)
    except Exception as exc:
        save_run_status(
            idea_dir,
            "failed",
            model=args.model,
            run_id=run_id,
            max_nodes=args.max_nodes,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    save_run_status(
        idea_dir,
        "complete",
        model=args.model,
        run_id=run_id,
        max_nodes=args.max_nodes,
    )
    print(f"Experiment artifacts: {idea_dir.resolve()}")


if __name__ == "__main__":
    main()
