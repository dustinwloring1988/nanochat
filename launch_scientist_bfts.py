import argparse
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

if not os.environ.get("AI_SCIENTIST_RUN_ID"):
    os.environ["AI_SCIENTIST_RUN_ID"] = uuid.uuid4().hex

from ai_scientist.providers import (
    DEFAULT_MODEL,
    ProviderError,
    llm_budget,
    preflight_model,
)
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
    return parser.parse_args()


def run_preflight(model: str) -> dict:
    try:
        return preflight_model(model)
    except ProviderError as exc:
        return {"ok": False, "error": str(exc)}


def save_token_tracker(idea_dir: Path):
    payload = {
        "models": token_tracker.get_summary(),
        "provider_budget": llm_budget.snapshot(),
        "interactions_captured": False,
    }
    with open(idea_dir / "token_tracker.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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
    preflight = run_preflight(args.model)
    if not preflight.get("ok"):
        raise RuntimeError(
            "Provider preflight failed: "
            + str(preflight.get("error", "selected model is unavailable"))
        )

    project_root = Path(__file__).resolve().parent
    ideas_path = Path(args.load_ideas)
    if not ideas_path.is_absolute():
        ideas_path = project_root / ideas_path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    experiment_root = Path(
        os.environ.get("AI_SCIENTIST_EXPERIMENT_DIR", project_root / "experiments")
    )
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
    cache_dir = Path(os.environ["NANOCHAT_SHARED_CACHE"]).resolve()
    pre_manifest = build_integrity_manifest(project_root, cache_dir)
    write_json_atomic(idea_dir / "integrity_before.json", pre_manifest)
    perform_experiments_bfts(run_config_path)
    post_manifest = build_integrity_manifest(project_root, cache_dir)
    write_json_atomic(idea_dir / "integrity_after.json", post_manifest)
    if pre_manifest != post_manifest:
        raise RuntimeError("Project or shared-cache integrity changed during BFTS")
    save_token_tracker(idea_dir)
    print(f"Experiment artifacts: {idea_dir.resolve()}")


if __name__ == "__main__":
    main()
