import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import nanochat.multi_source_dataloader as dataloader
from nanochat.checkpoint_manager import load_dataloader_checkpoint, save_checkpoint


class FakeTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend=0, num_threads=1):
        return [[prepend, int(text)] for text in texts]


def make_parquet(tmp_path, values, row_group_size=2, name="source"):
    path = tmp_path / f"{name}.parquet"
    pq.write_table(
        pa.table({"text": [str(value) for value in values]}),
        path,
        row_group_size=row_group_size,
    )
    return path


def install_fake_source(monkeypatch, path, source_name="source"):
    source = SimpleNamespace(
        subsets=["default"],
        text_column="text",
    )
    monkeypatch.setattr(
        dataloader,
        "list_parquet_files_for_source",
        lambda name, subset: [str(path)] if name == source_name else [],
    )
    monkeypatch.setattr(dataloader, "get_source", lambda name: source)
    monkeypatch.setattr(dataloader, "get_dist_info", lambda: (False, 0, 0, 1))


def make_loader(path, **kwargs):
    options = {
        "tokenizer": FakeTokenizer(),
        "source_weights": {"source": 1.0},
        "B": 1,
        "T": 3,
        "device": "cpu",
        "tokenizer_threads": 1,
        "tokenizer_batch_size": 2,
        "buffer_size": 3,
        "seed": 11,
    }
    options.update(kwargs)
    return dataloader.tokenizing_distributed_data_loader_multi_source(**options)


def clone_batch(inputs, targets):
    return inputs.detach().cpu().clone(), targets.detach().cpu().clone()


def test_fresh_iterator_reads_first_row_group(tmp_path, monkeypatch):
    path = make_parquet(tmp_path, [1, 2, 3, 4], row_group_size=4)
    install_fake_source(monkeypatch, path)

    iterator = dataloader.DocumentBatchIterator(
        source_name="source",
        tokenizer_batch_size=2,
    )
    batches = iter(iterator)
    first, state = next(batches)
    first_state = dataloader._source_state_to_dict(state)
    second, second_state = next(batches)

    assert first == ["1", "2"]
    assert second == ["3", "4"]
    assert first_state["epoch"] == 1
    assert first_state["first_pass"] is True
    assert first_state["pq_idx"] == 0
    assert first_state["rg_idx"] == 0
    assert first_state["row_offset"] == 2
    assert first_state["chunk_offset"] == 2
    assert second_state.row_offset == 4


def test_state_round_trip_preserves_cursor_and_buffered_documents(
    tmp_path, monkeypatch
):
    path = make_parquet(tmp_path, [1, 2, 3, 4, 5, 6, 7, 8], row_group_size=4)
    install_fake_source(monkeypatch, path)

    loader = make_loader(path)
    first_inputs, first_targets, state = next(loader)
    first_inputs, first_targets = clone_batch(first_inputs, first_targets)
    encoded_state = json.loads(json.dumps(state))

    assert encoded_state["schema_version"] == dataloader.LOADER_STATE_SCHEMA_VERSION
    assert encoded_state["source_names"] == ["source"]
    assert encoded_state["iterator_names"] == ["source"]
    assert encoded_state["source_states"]["source"]["row_offset"] == 4
    assert encoded_state["source_states"]["source"]["chunk_offset"] == 4
    assert encoded_state["document_buffers"]["source"] == [[0, 3], [0, 4]]
    assert encoded_state["pending_batch"]["inputs"] == first_inputs.tolist()
    assert encoded_state["pending_batch"]["targets"] == first_targets.tolist()

    expected = [(first_inputs, first_targets)]
    for _ in range(2):
        inputs, targets, _ = next(loader)
        expected.append(clone_batch(inputs, targets))

    resumed = make_loader(path, resume_state_dict=encoded_state)
    resumed_inputs, resumed_targets, _ = next(resumed)
    resumed_inputs, resumed_targets = clone_batch(resumed_inputs, resumed_targets)
    assert torch.equal(resumed_inputs, expected[0][0])
    assert torch.equal(resumed_targets, expected[0][1])

    for expected_inputs, expected_targets in expected[1:]:
        inputs, targets, _ = next(resumed)
        inputs, targets = clone_batch(inputs, targets)
        assert torch.equal(inputs, expected_inputs)
        assert torch.equal(targets, expected_targets)


def test_state_round_trip_preserves_epoch_and_first_pass(tmp_path, monkeypatch):
    path = make_parquet(tmp_path, [1, 2], row_group_size=4)
    install_fake_source(monkeypatch, path)

    loader = make_loader(path, buffer_size=5)
    first_inputs, first_targets, state = next(loader)
    first_inputs, first_targets = clone_batch(first_inputs, first_targets)
    encoded_state = json.loads(json.dumps(state))
    source_state = encoded_state["source_states"]["source"]

    assert source_state["epoch"] > 1
    assert source_state["first_pass"] is False

    expected = [(first_inputs, first_targets)]
    inputs, targets, _ = next(loader)
    expected.append(clone_batch(inputs, targets))

    resumed = make_loader(path, buffer_size=5, resume_state_dict=encoded_state)
    for expected_inputs, expected_targets in expected:
        inputs, targets, _ = next(resumed)
        inputs, targets = clone_batch(inputs, targets)
        assert torch.equal(inputs, expected_inputs)
        assert torch.equal(targets, expected_targets)


def test_state_preserves_ordered_sources_and_subsets(tmp_path, monkeypatch):
    paths = {
        ("source_b", "default"): make_parquet(
            tmp_path, [11, 12, 13, 14], row_group_size=2, name="source_b"
        ),
        ("source_a", "x"): make_parquet(
            tmp_path, [21, 22, 23, 24], row_group_size=2, name="source_a_x"
        ),
        ("source_a", "y"): make_parquet(
            tmp_path, [31, 32, 33, 34], row_group_size=2, name="source_a_y"
        ),
    }
    sources = {
        "source_b": SimpleNamespace(subsets=["default"], text_column="text"),
        "source_a": SimpleNamespace(subsets=["x", "y"], text_column="text"),
    }
    monkeypatch.setattr(
        dataloader,
        "list_parquet_files_for_source",
        lambda name, subset: [str(paths[(name, subset)])],
    )
    monkeypatch.setattr(dataloader, "get_source", lambda name: sources[name])
    monkeypatch.setattr(dataloader, "get_dist_info", lambda: (False, 0, 0, 1))

    options = {
        "tokenizer": FakeTokenizer(),
        "source_weights": {"source_b": 0.5, "source_a": 0.5},
        "subset_weights": {"source_a": {"x": 0.25, "y": 0.75}},
        "B": 1,
        "T": 3,
        "device": "cpu",
        "tokenizer_threads": 1,
        "tokenizer_batch_size": 1,
        "buffer_size": 2,
        "seed": 17,
    }
    loader = dataloader.tokenizing_distributed_data_loader_multi_source(**options)
    first_inputs, first_targets, state = next(loader)
    first_inputs, first_targets = clone_batch(first_inputs, first_targets)
    encoded_state = json.loads(json.dumps(state))

    assert encoded_state["source_names"] == ["source_b", "source_a"]
    assert encoded_state["subset_names"] == {
        "source_b": ["default"],
        "source_a": ["x", "y"],
    }
    assert encoded_state["iterator_names"] == [
        "source_b",
        "source_a::x",
        "source_a::y",
    ]
    assert list(encoded_state["subset_sampler_states"]) == ["source_a"]

    expected = [(first_inputs, first_targets)]
    for _ in range(3):
        inputs, targets, _ = next(loader)
        expected.append(clone_batch(inputs, targets))

    resumed = dataloader.tokenizing_distributed_data_loader_multi_source(
        **options, resume_state_dict=encoded_state
    )
    for expected_inputs, expected_targets in expected:
        inputs, targets, _ = next(resumed)
        inputs, targets = clone_batch(inputs, targets)
        assert torch.equal(inputs, expected_inputs)
        assert torch.equal(targets, expected_targets)


def test_incomplete_and_legacy_resume_state_fail_closed(tmp_path, monkeypatch):
    path = make_parquet(tmp_path, [1, 2, 3, 4], row_group_size=4)
    install_fake_source(monkeypatch, path)

    _, _, state = next(make_loader(path))
    state = json.loads(json.dumps(state))
    incomplete = dict(state)
    del incomplete["document_buffers"]
    with pytest.raises(ValueError):
        next(make_loader(path, resume_state_dict=incomplete))

    legacy = {
        "source_weights": {"source": 1.0},
        "source_states": {"source": {"pq_idx": 0, "rg_idx": 0, "epoch": 1}},
    }
    with pytest.raises(ValueError):
        next(make_loader(path, resume_state_dict=legacy))


def test_checkpoint_boundary_resume_matches_uninterrupted_stream(tmp_path, monkeypatch):
    path = make_parquet(tmp_path, [1, 2, 3, 4, 5, 6], row_group_size=3)
    install_fake_source(monkeypatch, path)
    loader = make_loader(path, buffer_size=2)
    expected = []

    inputs, targets, state = next(loader)
    expected.append(clone_batch(inputs, targets))
    checkpoint_dir = tmp_path / "checkpoints"
    save_checkpoint(
        checkpoint_dir,
        1,
        {},
        None,
        {"step": 1},
        rank=0,
        dataloader_state_dict=state,
    )
    resumed_state = load_dataloader_checkpoint(checkpoint_dir, 1, rank=0)
    for _ in range(3):
        inputs, targets, _ = next(loader)
        expected.append(clone_batch(inputs, targets))

    resumed = make_loader(path, buffer_size=2, resume_state_dict=resumed_state)
    for expected_inputs, expected_targets in expected:
        inputs, targets, _ = next(resumed)
        inputs, targets = clone_batch(inputs, targets)
        assert torch.equal(inputs, expected_inputs)
        assert torch.equal(targets, expected_targets)
