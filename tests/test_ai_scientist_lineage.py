import json

import pytest

from ai_scientist.treesearch.lineage import LineageError, load_lineage_manifest


def test_lineage_manifest_preserves_one_node_no_promotion_contract(tmp_path):
    path = tmp_path / "lineage.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "max_nodes": 1,
                "promotion": "not authorized",
            }
        ),
        encoding="utf-8",
    )
    assert load_lineage_manifest(path)["status"] == "complete"


def test_lineage_manifest_rejects_promotion(tmp_path):
    path = tmp_path / "lineage.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "max_nodes": 1,
                "promotion": "authorized",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(LineageError, match="one-node boundary"):
        load_lineage_manifest(path)
