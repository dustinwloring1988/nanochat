import os

import pytest
import torch

from nanochat.checkpoint_manager import (
    dataloader_checkpoint_path,
    load_checkpoint,
    load_dataloader_checkpoint,
    save_checkpoint,
)


def test_rank_local_dataloader_states_round_trip(tmp_path):
    step = 12
    model_data = {"weight": torch.tensor([1.0, 2.0])}
    optimizer_data = {"step": torch.tensor(1)}
    meta_data = {"step": step}
    rank0_state = {
        "schema_version": 1,
        "loader_config": {"ddp_rank": 0, "ddp_world_size": 2},
        "document_buffers": {"train": [[1, 2], [3]]},
        "pending_batch": {"inputs": [[4, 5]], "targets": [[6, 7]]},
    }
    rank1_state = {
        "schema_version": 1,
        "loader_config": {"ddp_rank": 1, "ddp_world_size": 2},
        "document_buffers": {"train": [[8, 9], [10]]},
        "pending_batch": {"inputs": [[11, 12]], "targets": [[13, 14]]},
    }

    save_checkpoint(
        tmp_path,
        step,
        model_data,
        optimizer_data,
        meta_data,
        rank=0,
        dataloader_state_dict=rank0_state,
    )
    assert dataloader_checkpoint_path(tmp_path, step, 0) == str(
        tmp_path / "dataloader_000012_rank0.pt"
    )
    assert load_dataloader_checkpoint(tmp_path, step, 0) == rank0_state
    assert not os.path.exists(dataloader_checkpoint_path(tmp_path, step, 1))

    save_checkpoint(
        tmp_path,
        step,
        model_data,
        optimizer_data,
        meta_data,
        rank=1,
        dataloader_state_dict=rank1_state,
    )
    assert load_dataloader_checkpoint(tmp_path, step, 0) == rank0_state
    assert load_dataloader_checkpoint(tmp_path, step, 1) == rank1_state
    assert rank0_state != rank1_state


def test_missing_rank_local_dataloader_state_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="rank 1"):
        load_dataloader_checkpoint(tmp_path, 4, 1)


def test_save_derives_dataloader_state_from_metadata(tmp_path):
    dataloader_state = {"epoch": 3, "pq_idx": 7, "rg_idx": 11}
    meta_data = {"step": 8, "dataloader_state_dict": dataloader_state}

    save_checkpoint(tmp_path, 8, {}, None, meta_data)

    assert load_dataloader_checkpoint(tmp_path, 8, 0) == dataloader_state
    _, _, loaded_meta_data = load_checkpoint(tmp_path, 8, torch.device("cpu"))
    assert loaded_meta_data == meta_data


def test_explicit_none_does_not_derive_dataloader_state(tmp_path):
    meta_data = {"dataloader_state_dict": {"epoch": 2}}

    save_checkpoint(tmp_path, 5, {}, None, meta_data, dataloader_state_dict=None)

    assert not os.path.exists(dataloader_checkpoint_path(tmp_path, 5, 0))
    _, _, loaded_meta_data = load_checkpoint(tmp_path, 5, torch.device("cpu"))
    assert loaded_meta_data == meta_data


def test_existing_save_and_load_behavior_remains_compatible(tmp_path):
    model_data = {"weight": torch.tensor([3.0, 4.0])}
    optimizer_data = {"step": torch.tensor(9)}
    meta_data = {"step": 6, "model_config": {"value": "unchanged"}}

    save_checkpoint(tmp_path, 6, model_data, optimizer_data, meta_data)
    loaded_model, loaded_optimizer, loaded_meta = load_checkpoint(
        tmp_path, 6, torch.device("cpu"), load_optimizer=True
    )

    assert torch.equal(loaded_model["weight"], model_data["weight"])
    assert torch.equal(loaded_optimizer["step"], optimizer_data["step"])
    assert loaded_meta == meta_data
    assert not os.path.exists(dataloader_checkpoint_path(tmp_path, 6, 0))


@pytest.mark.parametrize("rank", [-1, 1.5, "1", True])
def test_dataloader_checkpoint_path_rejects_invalid_rank(tmp_path, rank):
    with pytest.raises(ValueError, match="non-negative integer"):
        dataloader_checkpoint_path(tmp_path, 1, rank)
