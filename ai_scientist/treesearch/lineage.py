from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanochat.research_results import source_hash, write_json_atomic

if TYPE_CHECKING:
    from .agent_manager import AgentManager

LINEAGE_SCHEMA_VERSION = 1


class LineageError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(path: Path) -> str:
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _path_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def _experiment_root() -> Path:
    value = os.environ.get("AI_SCIENTIST_EXPERIMENT_DIR")
    if not value:
        raise LineageError("AI_SCIENTIST_EXPERIMENT_DIR is required for lineage handoff")
    return Path(value).expanduser().resolve()


def _validate_lineage_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise LineageError("lineage_id must be a safe path component")
    return value


def _stage(manager: "AgentManager", stage_number: int):
    from .agent_manager import Stage
    if stage_number == 1:
        return manager.stages[0]
    if stage_number == 2:
        return Stage(
            name="2_baseline_tuning_1_first_attempt",
            description="first_attempt",
            goals=manager.main_stage_goals[2],
            max_iterations=manager._get_max_iterations(2),
            num_drafts=0,
            stage_number=2,
        )
    if stage_number == 3:
        return Stage(
            name="3_creative_research_1_first_attempt",
            description="first_attempt",
            goals=manager.main_stage_goals[3],
            max_iterations=manager._get_max_iterations(3),
            num_drafts=0,
            stage_number=3,
        )
    raise LineageError("lineage handoff supports parent stages 1 and 2 only")


def _copy_parent_artifact(node: Any, handoff_root: Path) -> dict[str, Path]:
    from .nanochat_adapter import copy_source_snapshot

    artifact = Path(node.exp_results_dir).expanduser().resolve()
    if not artifact.is_dir():
        raise LineageError("parent node has no archived experiment artifact")
    source = Path(node.candidate_source).expanduser().resolve()
    if not source.is_dir():
        raise LineageError("parent node has no candidate source snapshot")
    handoff_root.mkdir(parents=True, exist_ok=False)
    copied_artifact = handoff_root / "artifact"
    copied_source = handoff_root / "source"
    copied_artifact.mkdir()
    copy_source_snapshot(source, copied_source)
    for name in (
        "results.json",
        "experiment.json",
        "baseline_experiment.json",
        "controller-attestation.json",
    ):
        candidate = artifact / name
        if candidate.is_file():
            shutil.copy2(candidate, copied_artifact / name)
    for plot in artifact.glob("*.png"):
        shutil.copy2(plot, copied_artifact / plot.name)
    paths = {
        "source": copied_source,
        "working": copied_artifact,
        "experiment_config": copied_artifact / "experiment.json",
        "baseline_config": copied_artifact / "baseline_experiment.json",
    }
    paths["config_hash"] = _config_hash(paths["baseline_config"])
    missing = [
        str(path)
        for name, path in paths.items()
        if name != "config_hash" and not path.exists()
    ]
    if missing:
        raise LineageError(f"parent handoff artifact is incomplete: {missing}")
    return paths


def _load_journal(path: Path):
    from .journal import Journal, Node

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    raw_nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(raw_nodes, list):
        raise LineageError("parent journal has no node list")
    nodes = [Node.from_dict(dict(raw_node)) for raw_node in raw_nodes]
    journal = Journal(nodes=nodes)
    node2parent = payload.get("node2parent", {})
    if not isinstance(node2parent, dict):
        raise LineageError("parent journal relationship map is invalid")
    for child_id, parent_id in node2parent.items():
        child = journal.get_node_by_id(child_id)
        parent = journal.get_node_by_id(parent_id)
        if child is None or parent is None:
            raise LineageError("parent journal relationship is stale")
        child.parent = parent
        parent.children.add(child)
    return journal


def _validate_parent_attestation(node: Any, paths: dict[str, Any]) -> None:
    attestation_path = paths["working"] / "controller-attestation.json"
    if not attestation_path.is_file():
        raise LineageError("parent node has no controller attestation")
    try:
        with open(attestation_path, "r", encoding="utf-8") as handle:
            attestation = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LineageError("parent controller attestation is not valid JSON") from exc
    attested = attestation.get("payload") if isinstance(attestation, dict) else None
    if (
        not isinstance(attested, dict)
        or attestation.get("schema_version") != 1
        or not isinstance(attestation.get("signature"), str)
        or not attestation["signature"]
    ):
        raise LineageError("parent controller attestation is malformed")
    expected = {
        "node_id": node.id,
        "result_sha256": _sha256(paths["working"] / "results.json"),
        "source_sha256": source_hash(paths["source"]),
        "baseline_config_hash": _config_hash(paths["baseline_config"]),
    }
    for name, value in expected.items():
        if attested.get(name) != value:
            raise LineageError(f"parent controller attestation {name} is stale")


def load_parent_handoff(
    cfg,
    manager: AgentManager,
    *,
    parent_journal_path: str | os.PathLike,
    parent_node_id: str,
    parent_stage: int,
    lineage_id: str,
) -> dict[str, Any]:
    if parent_stage not in (1, 2):
        raise LineageError("parent_stage must be 1 or 2")
    lineage_id = _validate_lineage_id(lineage_id)
    from .agent_manager import StageTransition
    from .journal import Journal
    from .nanochat_adapter import load_canonical_result, validate_nanochat_result

    experiment_root = _experiment_root()
    journal_path = Path(parent_journal_path).expanduser().resolve()
    if not _path_under(journal_path, experiment_root):
        raise LineageError("parent journal is outside the experiment root")
    if not journal_path.is_file():
        raise LineageError("parent journal does not exist")
    if not isinstance(parent_node_id, str) or not parent_node_id:
        raise LineageError("parent_node_id is required")
    journal = _load_journal(journal_path)
    parent = journal.get_node_by_id(parent_node_id)
    if parent is None:
        raise LineageError("parent node is not present in the journal")
    if parent.is_buggy or parent.is_buggy_plots or parent.metric is None:
        raise LineageError("parent node is not an accepted canonical candidate")
    handoff_root = Path(cfg.log_dir).resolve() / "handoff" / lineage_id
    paths = _copy_parent_artifact(parent, handoff_root)
    try:
        payload = load_canonical_result(paths["working"] / "results.json")
        validate_nanochat_result(payload, cfg, paths)
        _validate_parent_attestation(parent, paths)
    except Exception as exc:
        raise LineageError(f"parent canonical result validation failed: {exc}") from exc
    parent_copy = copy.deepcopy(parent)
    parent_copy.parent = None
    parent_copy.children = set()
    parent_copy.step = 0
    parent_copy.candidate_source = str(paths["source"])
    parent_copy.exp_results_dir = str(paths["working"])
    stage1 = manager.stages[0]
    target = _stage(manager, parent_stage + 1)
    if parent_stage == 1:
        manager.stages = [stage1, target]
        manager.journals = {
            stage1.name: Journal(nodes=[parent_copy]),
            target.name: Journal(),
        }
        manager.stage_history = [
            StageTransition(
                from_stage=stage1.name,
                to_stage=target.name,
                reason=f"verified parent handoff {lineage_id}",
                config_adjustments={},
            )
        ]
    else:
        stage2 = _stage(manager, 2)
        manager.stages = [stage1, stage2, target]
        manager.journals = {
            stage1.name: Journal(),
            stage2.name: Journal(nodes=[parent_copy]),
            target.name: Journal(),
        }
        manager.stage_history = [
            StageTransition(
                from_stage=stage1.name,
                to_stage=stage2.name,
                reason="lineage stage-2 predecessor",
                config_adjustments={},
            ),
            StageTransition(
                from_stage=stage2.name,
                to_stage=target.name,
                reason=f"verified parent handoff {lineage_id}",
                config_adjustments={},
            ),
        ]
    manager.current_stage = target
    manager.current_stage_number = target.stage_number
    manager.executed_nodes = 0
    manager.completed_stages = [stage.name for stage in manager.stages if stage.stage_number < target.stage_number]
    return {
        "lineage_id": lineage_id,
        "parent_node_id": parent.id,
        "parent_stage": parent_stage,
        "child_stage": target.stage_number,
        "parent_source": str(paths["source"]),
        "parent_artifact": str(paths["working"]),
        "parent_result_sha256": _sha256(paths["working"] / "results.json"),
        "parent_source_sha256": source_hash(paths["source"]),
    }


def finalize_lineage(
    cfg,
    manager: "AgentManager",
    handoff: dict[str, Any],
) -> dict[str, Any]:
    from .journal import Journal

    target = manager.current_stage
    if target is None:
        candidates = [stage for stage in manager.stages if stage.stage_number == handoff["child_stage"]]
        target = candidates[0] if candidates else None
    journal = manager.journals.get(target.name, Journal()) if target is not None else Journal()
    children = [node for node in journal.nodes if node.id != handoff["parent_node_id"]]
    child = children[0] if len(children) == 1 else None
    child_result_sha256 = None
    child_source_sha256 = None
    attestation_present = False
    if child is not None:
        child_artifact = Path(child.exp_results_dir).resolve()
        result_path = child_artifact / "results.json"
        source_path = Path(child.candidate_source).resolve()
        if result_path.is_file():
            child_result_sha256 = _sha256(result_path)
        if source_path.is_dir():
            child_source_sha256 = source_hash(source_path)
        attestation_present = (child_artifact / "controller-attestation.json").is_file()
    status = (
        "complete"
        if child is not None
        and child.is_buggy is False
        and child.is_buggy_plots is False
        and child_result_sha256 is not None
        and child_source_sha256 is not None
        and attestation_present
        else "rejected"
    )
    payload = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "status": status,
        "lineage_id": handoff["lineage_id"],
        "max_nodes": 1,
        "executed_nodes": len(children),
        "parent_stage": handoff["parent_stage"],
        "child_stage": handoff["child_stage"],
        "parent_node_id": handoff["parent_node_id"],
        "child_node_id": child.id if child is not None else None,
        "parent_result_sha256": handoff["parent_result_sha256"],
        "parent_source_sha256": handoff["parent_source_sha256"],
        "child_result_sha256": child_result_sha256,
        "child_source_sha256": child_source_sha256,
        "child_attestation_present": attestation_present,
        "promotion": "not authorized",
    }
    output = Path(cfg.log_dir).resolve() / "lineage.json"
    write_json_atomic(output, payload)
    return payload


def load_lineage_manifest(path: str | os.PathLike) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LineageError("lineage manifest is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != LINEAGE_SCHEMA_VERSION:
        raise LineageError("unsupported lineage manifest")
    if payload.get("status") not in {"complete", "rejected"}:
        raise LineageError("lineage manifest status is invalid")
    if payload.get("max_nodes") != 1 or payload.get("promotion") != "not authorized":
        raise LineageError("lineage manifest violates the one-node boundary")
    return payload
