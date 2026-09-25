import json

import pytest
import torch

from nanochat.research_results import sha256_file
from nanochat.sft_runtime import (
    SFT_CONTEXT_TOKENS,
    SFTConversationBatchLoader,
    SFTLoaderError,
    SFTCheckpointTransaction,
    build_effective_token_budget,
    capture_rng_state,
    fixed_context_contract,
    load_sft_checkpoint,
    resolve_run_local_path,
    rollback_sft_checkpoint,
    save_sft_checkpoint,
    validate_dataset_manifest,
    validate_effective_token_budget,
    validate_fixed_context,
    validate_sft_checkpoint_envelope,
)


class FakeTokenizer:
    def __init__(self):
        self.calls = []
        self.fail = False

    def get_bos_token_id(self):
        return 0

    def render_conversation(self, conversation, max_tokens=2048):
        self.calls.append((conversation, max_tokens))
        if self.fail:
            raise RuntimeError("synthetic render failure")
        text = conversation["messages"][-1].get("content", "")
        value = int(str(text))
        ids = [0, value, value + 1, value + 2, value + 3]
        mask = [0, 0, 1, 1, 1]
        if max_tokens < len(ids):
            return ids[:max_tokens], mask[:max_tokens]
        return ids, mask


def make_manifest(tmp_path, count=6, name="train.jsonl"):
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    data_path = data_dir / name
    records = [
        {
            "conversation": {
                "messages": [
                    {"role": "user", "content": f"question-{index}"},
                    {"role": "assistant", "content": str(index)},
                ]
            }
        }
        for index in range(count)
    ]
    data_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    payload = {
        "manifest_schema_version": 1,
        "datasets": [
            {
                "source": "synthetic",
                "split": "train",
                "license": {"name": "test-only"},
                "provenance": {"source": "unit-test", "revision": "1"},
                "files": [
                    {
                        "path": data_path.relative_to(tmp_path).as_posix(),
                        "sha256": sha256_file(data_path),
                        "byte_count": data_path.stat().st_size,
                        "record_count": len(records),
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return manifest_path


def make_rng_state():
    return capture_rng_state()


def test_manifest_validates_jsonl_and_fails_closed_on_stale_files(tmp_path):
    manifest_path = make_manifest(tmp_path)
    manifest = validate_dataset_manifest(manifest_path, run_root=tmp_path)
    assert manifest.schema_version == 1
    assert manifest.files[0].source == "synthetic"
    assert manifest.files[0].split == "train"
    assert manifest.files[0].record_count == 6

    data_path = tmp_path / "data" / "train.jsonl"
    data_path.write_text(data_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        validate_dataset_manifest(manifest_path, run_root=tmp_path)

    data_path.unlink()
    with pytest.raises(ValueError, match="missing|exist"):
        validate_dataset_manifest(manifest_path, run_root=tmp_path)


def test_manifest_requires_source_split_license_and_provenance(tmp_path):
    manifest_path = make_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["datasets"][0]["provenance"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        validate_dataset_manifest(manifest_path, run_root=tmp_path)


def test_run_local_path_rejects_namespaces_and_symlink_escape(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    trusted = tmp_path / "trusted-cache"
    trusted.mkdir()
    monkeypatch.setenv("NANOCHAT_SHARED_CACHE", str(trusted))

    assert resolve_run_local_path("data/train.jsonl", run_root) == (
        run_root / "data" / "train.jsonl"
    )
    with pytest.raises(ValueError, match="global checkpoint namespace"):
        resolve_run_local_path("sft_checkpoints/model.pt", run_root)
    with pytest.raises(ValueError, match="escapes the run root"):
        resolve_run_local_path(trusted / "data.jsonl", run_root)

    outside = tmp_path / "outside"
    outside.mkdir()
    link = run_root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(ValueError, match="symlink"):
        resolve_run_local_path("link/data.jsonl", run_root)


def test_fixed_context_contract_and_budget_reject_non_2048(tmp_path):
    assert SFT_CONTEXT_TOKENS == 2048
    assert fixed_context_contract()["context_range"] == [2048, 2048]
    assert validate_fixed_context(2048) == 2048
    with pytest.raises(ValueError, match="2048"):
        validate_fixed_context(4096)
    with pytest.raises(ValueError, match="2048"):
        validate_effective_token_budget(
            device_batch_size=1,
            world_size=1,
            gradient_accumulation_steps=1,
            sequence_length=4096,
        )

    budget = build_effective_token_budget(2, 2, 3)
    assert budget.effective_tokens == 24576
    assert budget.total_batch_size == 24576
    assert budget.total_sequence_batch == 12
    assert validate_effective_token_budget(2, 2, 3, 24576) == 24576
    assert validate_effective_token_budget(2, 2, 3, total_batch_size=24576) == 24576
    with pytest.raises(ValueError, match="fixed-token formula"):
        validate_effective_token_budget(2, 2, 3, 24575)


def test_loader_uses_renderer_masks_pads_and_has_json_resume_state(tmp_path):
    manifest = validate_dataset_manifest(make_manifest(tmp_path), run_root=tmp_path)
    tokenizer = FakeTokenizer()
    loader = SFTConversationBatchLoader(
        manifest,
        tokenizer,
        batch_size=2,
        rank=0,
        world_size=1,
        seed=17,
        device="cpu",
    )
    batch = next(loader)
    assert tuple(batch.inputs.shape) == (2, 2048)
    assert tuple(batch.targets.shape) == (2, 2048)
    assert tuple(batch.loss_mask.shape) == (2, 2048)
    assert batch.inputs[0, 0].item() == 0
    assert batch.targets[0, -1].item() == -1
    assert tokenizer.calls[0][1] == 2049
    state = json.loads(json.dumps(batch.state, allow_nan=False, sort_keys=True))
    for field in ("cursor", "epoch", "rng_state", "counters", "pending_batch"):
        assert field in state
    assert state["pending_batch"]["shape"] == [2, 2048]


def test_loader_no_fallback_and_rejects_malformed_conversations(tmp_path):
    data_path = tmp_path / "data.jsonl"
    data_path.write_text(
        json.dumps({"conversation": "not-json"}) + "\n",
        encoding="utf-8",
    )
    payload = {
        "manifest_schema_version": 1,
        "datasets": [
            {
                "source": "synthetic",
                "split": "train",
                "license": "test",
                "provenance": "unit",
                "files": [
                    {
                        "path": "data.jsonl",
                        "sha256": sha256_file(data_path),
                        "record_count": 1,
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = validate_dataset_manifest(manifest_path, run_root=tmp_path)
    with pytest.raises(SFTLoaderError, match="non-JSON|fallback"):
        SFTConversationBatchLoader(manifest, FakeTokenizer())

    tokenizer = FakeTokenizer()
    tokenizer.fail = True
    data_path.write_text(
        json.dumps(
            {
                "conversation": {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload["datasets"][0]["files"][0]["sha256"] = sha256_file(data_path)
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = validate_dataset_manifest(manifest_path, run_root=tmp_path)
    with pytest.raises(SFTLoaderError, match="fallback is forbidden"):
        next(SFTConversationBatchLoader(manifest, tokenizer))


def test_loader_resume_reproduces_uninterrupted_stream(tmp_path):
    manifest = validate_dataset_manifest(
        make_manifest(tmp_path, count=8), run_root=tmp_path
    )
    tokenizer = FakeTokenizer()
    reference = SFTConversationBatchLoader(
        manifest, tokenizer, batch_size=2, seed=23, device="cpu"
    )
    first = next(reference)
    first_state = json.loads(json.dumps(first.state, allow_nan=False))
    expected = [(first.inputs.clone(), first.targets.clone())]
    for _ in range(4):
        item = next(reference)
        expected.append((item.inputs.clone(), item.targets.clone()))

    resumed = SFTConversationBatchLoader(
        manifest,
        FakeTokenizer(),
        batch_size=2,
        seed=23,
        device="cpu",
        resume_state=first_state,
    )
    replay = next(resumed)
    assert torch.equal(replay.inputs, expected[0][0])
    assert torch.equal(replay.targets, expected[0][1])
    for expected_inputs, expected_targets in expected[1:]:
        item = next(resumed)
        assert torch.equal(item.inputs, expected_inputs)
        assert torch.equal(item.targets, expected_targets)


def test_loader_is_rank_aware(tmp_path):
    manifest = validate_dataset_manifest(
        make_manifest(tmp_path, count=8), run_root=tmp_path
    )
    rank_zero = SFTConversationBatchLoader(
        manifest, FakeTokenizer(), batch_size=1, rank=0, world_size=2, seed=9
    )
    rank_one = SFTConversationBatchLoader(
        manifest, FakeTokenizer(), batch_size=1, rank=1, world_size=2, seed=9
    )
    assert next(rank_zero).inputs[0, 1].item() != next(rank_one).inputs[0, 1].item()


def test_checkpoint_transaction_rolls_back_without_pointer(tmp_path):
    checkpoint_root = tmp_path / "run-checkpoints"
    transaction = SFTCheckpointTransaction(
        checkpoint_root,
        run_root=tmp_path,
        step=1,
        world_size=1,
    )
    transaction.write_rank_sidecars(0, {"step": 1}, {"cursor": 0}, make_rng_state())
    transaction.write_model({"weight": torch.tensor([1.0])})
    transaction.rollback()
    assert not (checkpoint_root / "step_000001").exists()
    assert not (checkpoint_root / "last_known_good.json").exists()


def test_checkpoint_commit_load_and_last_known_good(tmp_path):
    checkpoint_root = tmp_path / "run-checkpoints"
    transaction = save_sft_checkpoint(
        checkpoint_root,
        run_root=tmp_path,
        step=1,
        model_state={"weight": torch.tensor([3.0])},
        optimizer_state={"step": torch.tensor(1)},
        loader_state={"cursor": 0, "epoch": 1},
        rng_state=make_rng_state(),
    )
    assert transaction.committed is True
    loaded = load_sft_checkpoint(checkpoint_root, run_root=tmp_path)
    assert loaded.envelope["sequence_length"] == 2048
    assert torch.equal(loaded.model_state["weight"], torch.tensor([3.0]))
    assert loaded.loader_state["cursor"] == 0
    envelope = validate_sft_checkpoint_envelope(checkpoint_root / "step_000001")
    assert envelope["completion_status"] == "complete"
    assert (checkpoint_root / "COMPLETED.json").exists() is False


def test_checkpoint_rank_local_sidecars_can_commit_from_shared_staging(tmp_path):
    checkpoint_root = tmp_path / "run-checkpoints"
    rank_zero = SFTCheckpointTransaction(
        checkpoint_root,
        run_root=tmp_path,
        step=1,
        rank=0,
        world_size=2,
    )
    rank_zero.write_rank_sidecars(
        0,
        {"step": torch.tensor(0)},
        {"rank": 0},
        make_rng_state(),
    )
    rank_zero.write_model({"weight": torch.tensor([4.0])})
    rank_one = SFTCheckpointTransaction(
        checkpoint_root,
        run_root=tmp_path,
        step=1,
        rank=1,
        world_size=2,
    )
    rank_one.write_rank_sidecars(
        1,
        {"step": torch.tensor(1)},
        {"rank": 1},
        make_rng_state(),
    )
    rank_zero.commit()
    loaded = load_sft_checkpoint(checkpoint_root, run_root=tmp_path, rank=1)
    assert loaded.loader_state == {"rank": 1}
    assert set(loaded.envelope["optimizer"]) == {"0", "1"}


def test_checkpoint_failed_candidate_preserves_pointer_and_load_fails_closed(tmp_path):
    checkpoint_root = tmp_path / "run-checkpoints"
    save_sft_checkpoint(
        checkpoint_root,
        run_root=tmp_path,
        step=1,
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={"step": torch.tensor(1)},
        loader_state={"cursor": 0},
        rng_state=make_rng_state(),
    )
    pointer_before = (checkpoint_root / "last_known_good.json").read_bytes()
    failed = SFTCheckpointTransaction(
        checkpoint_root,
        run_root=tmp_path,
        step=2,
        world_size=2,
    )
    failed.write_rank_sidecars(0, {"step": 2}, {"cursor": 1}, make_rng_state())
    failed.write_model({"weight": torch.tensor([2.0])})
    with pytest.raises(ValueError, match="sidecars"):
        failed.commit()
    failed.rollback()
    assert (checkpoint_root / "last_known_good.json").read_bytes() == pointer_before
    assert not (checkpoint_root / "step_000002").exists()

    final_path = checkpoint_root / "step_000001"
    (final_path / "loader_rank0.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="stale|hash"):
        load_sft_checkpoint(checkpoint_root, run_root=tmp_path)
    with pytest.raises(ValueError, match="stale|hash"):
        rollback_sft_checkpoint(
            checkpoint_root, run_root=tmp_path, checkpoint_id="step_000001"
        )
    assert (checkpoint_root / "last_known_good.json").read_bytes() == pointer_before
