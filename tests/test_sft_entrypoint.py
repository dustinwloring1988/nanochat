import json
from pathlib import Path

import pytest

from scripts.sft_train_curriculum import (
    build_run_manifest,
    load_fixed_config,
    resolve_run_root,
)
from nanochat.sft_runtime import (
    SFT_CONTEXT_TOKENS,
    SFTLoaderError,
    build_effective_token_budget,
)

ROOT = Path(__file__).parents[1]


class ManifestStub:
    manifest_path = Path("dataset-manifest.json")
    manifest_hash = "a" * 64


def test_entrypoint_accepts_only_fixed_single_stage_config(tmp_path):
    fixed = load_fixed_config(None)
    assert fixed["stages"][0]["context_range"] == [SFT_CONTEXT_TOKENS] * 2

    long_config = tmp_path / "long.yaml"
    long_config.write_text(
        "stages:\n  - name: long\n    context_range: [8192, 8192]\n",
        encoding="utf-8",
    )
    with pytest.raises(SFTLoaderError, match="fixed at 2048"):
        load_fixed_config(long_config)

    multi_config = tmp_path / "multi.yaml"
    multi_config.write_text(
        "stages:\n"
        "  - name: one\n"
        "    context_range: [2048, 2048]\n"
        "  - name: two\n"
        "    context_range: [2048, 2048]\n",
        encoding="utf-8",
    )
    with pytest.raises(SFTLoaderError, match="exactly one"):
        load_fixed_config(multi_config)


def test_entrypoint_run_root_rejects_trusted_cache(tmp_path, monkeypatch):
    trusted = tmp_path / "trusted-cache"
    trusted.mkdir()
    monkeypatch.setenv("NANOCHAT_SHARED_CACHE", str(trusted))
    with pytest.raises(ValueError, match="trusted|global"):
        resolve_run_root(trusted / "run")


def test_entrypoint_run_manifest_contains_fixed_budget_and_hashes(tmp_path):
    budget = build_effective_token_budget(1, 1, 2)

    class Args:
        run_id = "fabricated-run"
        num_iterations = 2
        seed = 7
        model_tag = None
        model_step = None
        max_seq_len = 2048
        eval_every = 0
        save_every = 1
        compile = False

    manifest = build_run_manifest(
        Args(),
        {"stages": [{"name": "fixed"}]},
        ManifestStub(),
        tmp_path,
        budget,
        "b" * 64,
    )
    assert manifest["context_length"] == 2048
    assert manifest["effective_tokens"] == 4096
    assert len(manifest["curriculum_config_hash"]) == 64
    assert manifest["promotion"] == "not authorized"
    assert "OPENAI_API_KEY" not in json.dumps(manifest)


def test_legacy_chat_sft_entrypoint_is_disabled(capsys):
    from scripts.chat_sft import main

    with pytest.raises(SystemExit):
        main()
    message = capsys.readouterr().err
    assert "pretraining-only" in message
    assert "python -m scripts.sft_smoke --help" in message


def test_default_workflows_do_not_invoke_unapproved_sft():
    workflow_paths = (
        "docker/init_training.sh",
        "runs/speedrun.sh",
        "runs/runcpu.sh",
        "runs/curriculum_4060ti.sh",
    )
    forbidden = (
        "-m scripts.chat_sft",
        "-m scripts.sft_train_curriculum",
        "config/sft_curriculum.yaml",
        "sft_checkpoints",
    )
    for relative_path in workflow_paths:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, relative_path
        assert "python -m scripts.sft_smoke --help" in text


def test_readme_quick_start_is_pretraining_only():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pretraining-only" in text
    assert "Run 3-stage SFT curriculum" not in text
    assert "./data/sft_checkpoints/" not in text
    assert "python -m scripts.sft_smoke --help" in text
