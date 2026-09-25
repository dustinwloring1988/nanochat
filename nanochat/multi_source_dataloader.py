"""Weighted multi-source dataloader with curriculum support."""

import json
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import pyarrow.parquet as pq
import torch

from nanochat.common import get_dist_info
from nanochat.data_registry import get_source, list_parquet_files_for_source

LOADER_STATE_SCHEMA_VERSION = 1


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _normalise_weight_map(weights, field_name: str) -> Dict[str, float]:
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{field_name} must be a non-empty mapping")
    normalised = {}
    for name, value in weights.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{field_name} names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name}[{name}] must be numeric")
        try:
            numeric_value = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}[{name}] must be numeric") from exc
        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(f"{field_name}[{name}] must be finite and non-negative")
        normalised[name] = numeric_value
    if sum(normalised.values()) <= 0:
        raise ValueError(f"{field_name} must have a positive sum")
    return normalised


def _require_positive_int(value, field_name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return int(value)


def _require_nonnegative_int(value, field_name: str) -> int:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return int(value)


def _require_name_list(value, field_name: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{field_name} must not contain duplicates")
    return list(value)


@dataclass
class SourceState:
    """Track iteration state for a single data source."""

    pq_idx: int = 0
    rg_idx: int = 0
    epoch: int = 1
    row_offset: int = 0
    chunk_offset: int = 0
    first_pass: bool = True

    def __post_init__(self):
        pq_idx = _require_nonnegative_int(self.pq_idx, "pq_idx")
        rg_idx = _require_nonnegative_int(self.rg_idx, "rg_idx")
        epoch = _require_positive_int(self.epoch, "epoch")
        row_offset = _require_nonnegative_int(self.row_offset, "row_offset")
        chunk_offset = _require_nonnegative_int(self.chunk_offset, "chunk_offset")
        if row_offset == 0 and chunk_offset != 0:
            row_offset = chunk_offset
        elif chunk_offset == 0 and row_offset != 0:
            chunk_offset = row_offset
        if row_offset != chunk_offset:
            raise ValueError("row_offset and chunk_offset must match")
        if not isinstance(self.first_pass, bool):
            raise ValueError("first_pass must be boolean")
        self.pq_idx = pq_idx
        self.rg_idx = rg_idx
        self.epoch = epoch
        self.row_offset = row_offset
        self.chunk_offset = chunk_offset


def _encode_generator_state(generator: torch.Generator) -> list[int]:
    return [int(value) for value in generator.get_state().tolist()]


def _decode_generator_state(state: list[int]) -> torch.Generator:
    if not isinstance(state, list) or not state:
        raise ValueError("generator state must be a non-empty list")
    if any(not _is_int(value) or value < 0 or value > 255 for value in state):
        raise ValueError("generator state must contain byte values")
    generator = torch.Generator()
    try:
        generator.set_state(torch.tensor(state, dtype=torch.uint8))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("invalid torch generator state") from exc
    return generator


def _normalise_tokens(tokens, field_name: str) -> List[int]:
    if isinstance(tokens, torch.Tensor):
        if tokens.ndim != 1:
            raise ValueError(f"{field_name} token entries must be one-dimensional")
        tokens = tokens.detach().cpu().tolist()
    else:
        try:
            tokens = list(tokens)
        except TypeError as exc:
            raise ValueError(f"{field_name} must contain token lists") from exc
    if any(not _is_int(token) for token in tokens):
        raise ValueError(f"{field_name} must contain integer token IDs")
    return [int(token) for token in tokens]


def _copy_token_buffer(buffer, field_name: str) -> List[List[int]]:
    if not isinstance(buffer, list):
        raise ValueError(f"{field_name} must be a list")
    documents = []
    for index, document in enumerate(buffer):
        if not isinstance(document, list):
            raise ValueError(f"{field_name}[{index}] must be a list")
        if any(not _is_int(token) for token in document):
            raise ValueError(f"{field_name}[{index}] must contain integer token IDs")
        documents.append([int(token) for token in document])
    return documents


def _source_state_to_dict(state: SourceState) -> Dict:
    return {
        "pq_idx": int(state.pq_idx),
        "rg_idx": int(state.rg_idx),
        "epoch": int(state.epoch),
        "row_offset": int(state.row_offset),
        "chunk_offset": int(state.chunk_offset),
        "first_pass": bool(state.first_pass),
    }


def _source_state_from_dict(data, field_name: str) -> SourceState:
    if not isinstance(data, dict):
        raise ValueError(f"{field_name} must be a mapping")
    required = {
        "pq_idx",
        "rg_idx",
        "epoch",
        "row_offset",
        "chunk_offset",
        "first_pass",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"{field_name} is missing fields: {missing}")
    if data["row_offset"] != data["chunk_offset"]:
        raise ValueError(f"{field_name} cursor offsets must match")
    try:
        return SourceState(
            pq_idx=data["pq_idx"],
            rg_idx=data["rg_idx"],
            epoch=data["epoch"],
            row_offset=data["row_offset"],
            chunk_offset=data["chunk_offset"],
            first_pass=data["first_pass"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid: {exc}") from exc


def _validate_pending_matrix(matrix, rows: int, columns: int, field_name: str):
    if not isinstance(matrix, list) or len(matrix) != rows:
        raise ValueError(f"{field_name} must have shape [{rows}, {columns}]")
    normalised = []
    for row in matrix:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{field_name} must have shape [{rows}, {columns}]")
        if any(not _is_int(value) for value in row):
            raise ValueError(f"{field_name} must contain integer values")
        normalised.append([int(value) for value in row])
    return normalised


def _validate_pending_batch(pending_batch, batch_size: int, sequence_length: int):
    if not isinstance(pending_batch, dict):
        raise ValueError("pending_batch must be a mapping")
    required = {"inputs", "targets"}
    missing = sorted(required - set(pending_batch))
    if missing:
        raise ValueError(f"pending_batch is missing fields: {missing}")
    inputs = _validate_pending_matrix(
        pending_batch["inputs"], batch_size, sequence_length, "pending_batch.inputs"
    )
    targets = _validate_pending_matrix(
        pending_batch["targets"], batch_size, sequence_length, "pending_batch.targets"
    )
    if "shape" in pending_batch:
        if pending_batch["shape"] != [batch_size, sequence_length]:
            raise ValueError("pending_batch shape is incompatible")
    if "dtype" in pending_batch and pending_batch["dtype"] not in {
        "torch.int64",
        "long",
    }:
        raise ValueError("pending_batch dtype is incompatible")
    try:
        return (
            torch.tensor(inputs, dtype=torch.long, device="cpu"),
            torch.tensor(targets, dtype=torch.long, device="cpu"),
        )
    except (OverflowError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("pending_batch contains invalid tensor values") from exc


def _validate_generator_state_value(value, field_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    if any(not _is_int(item) or item < 0 or item > 255 for item in value):
        raise ValueError(f"{field_name} must contain byte values")


def _validate_count_map(value, expected_names: List[str], field_name: str):
    if not isinstance(value, dict) or list(value.keys()) != expected_names:
        raise ValueError(f"{field_name} must contain the expected ordered names")
    return {
        name: _require_nonnegative_int(value[name], f"{field_name}[{name}]")
        for name in expected_names
    }


def _validate_resume_state(
    state,
    source_names: List[str],
    source_weights: Dict[str, float],
    subset_names: Dict[str, List[str]],
    subset_weights: Dict[str, Dict[str, float]],
    iterator_specs: List[Dict[str, str]],
    batch_size: int,
    sequence_length: int,
    split: str,
    tokenizer_threads: int,
    tokenizer_batch_size: int,
    buffer_size: int,
    seed: int,
    ddp_rank: int,
    ddp_world_size: int,
):
    if not isinstance(state, dict):
        raise ValueError("resume state must be a mapping")
    try:
        json.dumps(state, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resume state must be JSON-compatible") from exc
    required = {
        "schema_version",
        "source_names",
        "source_weights",
        "subset_names",
        "subset_weights",
        "iterator_names",
        "iterator_specs",
        "loader_config",
        "sampler_state",
        "subset_sampler_states",
        "source_states",
        "document_buffers",
        "source_token_counts",
        "source_document_counts",
        "pending_batch",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"resume state is missing fields: {missing}")
    if not _is_int(state["schema_version"]):
        raise ValueError("resume state schema_version must be an integer")
    if state["schema_version"] != LOADER_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported resume state schema_version: {state['schema_version']}"
        )

    saved_source_names = _require_name_list(state["source_names"], "source_names")
    if saved_source_names != source_names:
        raise ValueError("resume source order is incompatible")

    saved_source_weights = _normalise_weight_map(
        state["source_weights"], "source_weights"
    )
    if list(saved_source_weights.keys()) != source_names:
        raise ValueError("resume source weight names are incompatible")
    for name in source_names:
        if saved_source_weights[name] != source_weights[name]:
            raise ValueError(f"resume source weight is incompatible: {name}")

    if not isinstance(state["subset_names"], dict):
        raise ValueError("subset_names must be a mapping")
    if list(state["subset_names"].keys()) != source_names:
        raise ValueError("resume subset source order is incompatible")
    for name in source_names:
        if (
            _require_name_list(state["subset_names"][name], f"subset_names[{name}]")
            != subset_names[name]
        ):
            raise ValueError(f"resume subset order is incompatible: {name}")

    if not isinstance(state["subset_weights"], dict):
        raise ValueError("subset_weights must be a mapping")
    expected_subset_sources = list(subset_weights.keys())
    if list(state["subset_weights"].keys()) != expected_subset_sources:
        raise ValueError("resume subset weight sources are incompatible")
    for source_name in expected_subset_sources:
        saved_subset_weights = _normalise_weight_map(
            state["subset_weights"][source_name],
            f"subset_weights[{source_name}]",
        )
        if list(saved_subset_weights.keys()) != list(
            subset_weights[source_name].keys()
        ):
            raise ValueError(
                f"resume subset weight order is incompatible: {source_name}"
            )
        for subset_name, weight in subset_weights[source_name].items():
            if saved_subset_weights[subset_name] != weight:
                raise ValueError(
                    f"resume subset weight is incompatible: {source_name}/{subset_name}"
                )

    expected_iterator_names = [spec["key"] for spec in iterator_specs]
    saved_iterator_names = state["iterator_names"]
    if saved_iterator_names != expected_iterator_names:
        raise ValueError("resume iterator order is incompatible")
    if state["iterator_specs"] != iterator_specs:
        raise ValueError("resume iterator metadata is incompatible")

    loader_config = state["loader_config"]
    if not isinstance(loader_config, dict):
        raise ValueError("loader_config must be a mapping")
    expected_config = {
        "B": batch_size,
        "T": sequence_length,
        "split": split,
        "tokenizer_threads": tokenizer_threads,
        "tokenizer_batch_size": tokenizer_batch_size,
        "buffer_size": buffer_size,
        "seed": seed,
        "ddp_rank": ddp_rank,
        "ddp_world_size": ddp_world_size,
    }
    for key, expected_value in expected_config.items():
        if key not in loader_config:
            raise ValueError(f"loader_config is missing field: {key}")
        actual_value = loader_config[key]
        if key in {
            "B",
            "T",
            "tokenizer_threads",
            "tokenizer_batch_size",
            "buffer_size",
            "seed",
            "ddp_rank",
            "ddp_world_size",
        }:
            if not _is_int(actual_value) or actual_value != expected_value:
                raise ValueError(f"loader_config.{key} is incompatible")
        elif actual_value != expected_value:
            raise ValueError(f"loader_config.{key} is incompatible")

    _validate_generator_state_value(state["sampler_state"], "sampler_state")
    subset_sampler_states = state["subset_sampler_states"]
    if not isinstance(subset_sampler_states, dict):
        raise ValueError("subset_sampler_states must be a mapping")
    if list(subset_sampler_states.keys()) != expected_subset_sources:
        raise ValueError("resume subset sampler order is incompatible")
    for source_name in expected_subset_sources:
        _validate_generator_state_value(
            subset_sampler_states[source_name],
            f"subset_sampler_states[{source_name}]",
        )

    source_states = state["source_states"]
    if (
        not isinstance(source_states, dict)
        or list(source_states.keys()) != expected_iterator_names
    ):
        raise ValueError("source_states must contain the expected ordered iterators")
    parsed_source_states = {
        name: _source_state_from_dict(source_states[name], f"source_states[{name}]")
        for name in expected_iterator_names
    }

    document_buffers = state["document_buffers"]
    if (
        not isinstance(document_buffers, dict)
        or list(document_buffers.keys()) != expected_iterator_names
    ):
        raise ValueError("document_buffers must contain the expected ordered iterators")
    parsed_document_buffers = {
        name: _copy_token_buffer(document_buffers[name], f"document_buffers[{name}]")
        for name in expected_iterator_names
    }

    source_token_counts = _validate_count_map(
        state["source_token_counts"], source_names, "source_token_counts"
    )
    source_document_counts = _validate_count_map(
        state["source_document_counts"], source_names, "source_document_counts"
    )
    pending_inputs, pending_targets = _validate_pending_batch(
        state["pending_batch"], batch_size, sequence_length
    )
    return {
        "source_states": parsed_source_states,
        "document_buffers": parsed_document_buffers,
        "source_token_counts": source_token_counts,
        "source_document_counts": source_document_counts,
        "pending_inputs": pending_inputs,
        "pending_targets": pending_targets,
    }


class WeightedSourceSampler:
    """Samples data source names according to weights."""

    def __init__(self, source_weights: Dict[str, float], seed: int = 42):
        self.update_weights(source_weights)
        self.rng = torch.Generator().manual_seed(seed)

    def update_weights(self, source_weights: Dict[str, float]):
        """Update sampling weights for curriculum stage transitions."""
        self.source_names = list(source_weights.keys())
        if not self.source_names:
            raise ValueError("At least one source weight is required")
        values = [float(source_weights[name]) for name in self.source_names]
        if any(
            not torch.isfinite(torch.tensor(value)) or value < 0 for value in values
        ):
            raise ValueError("Source weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("Source weights must have a positive sum")
        self.weights = torch.tensor(values)
        self.weights = self.weights / self.weights.sum()

    def sample(self) -> str:
        """Sample a source name according to current weights."""
        idx = torch.multinomial(self.weights, 1, generator=self.rng).item()
        return self.source_names[idx]


class DocumentBatchIterator:
    """Infinite iterator over document batches from a single source."""

    def __init__(
        self,
        source_name: str,
        subset: str = "default",
        tokenizer_batch_size: int = 128,
        resume_state: Optional[SourceState] = None,
        split: str = "train",
    ):
        self.source_name = source_name
        self.subset = subset
        self.tokenizer_batch_size = _require_positive_int(
            tokenizer_batch_size, "tokenizer_batch_size"
        )
        self.split = split

        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        self.ddp = ddp
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        if (
            self.ddp_world_size <= 0
            or self.ddp_rank < 0
            or self.ddp_rank >= self.ddp_world_size
        ):
            raise ValueError("invalid distributed iterator configuration")

        self.parquet_paths = list_parquet_files_for_source(source_name, subset)
        assert (
            len(self.parquet_paths) > 0
        ), f"No parquet files found for {source_name}/{subset}. Run data download first."

        if split == "val":
            self.parquet_paths = self.parquet_paths[-1:]
        else:
            self.parquet_paths = (
                self.parquet_paths[:-1]
                if len(self.parquet_paths) > 1
                else self.parquet_paths
            )

        if resume_state is None:
            self.state = SourceState(rg_idx=self.ddp_rank)
        elif isinstance(resume_state, SourceState):
            self.state = resume_state
        elif isinstance(resume_state, dict):
            self.state = _source_state_from_dict(resume_state, "resume_state")
        else:
            raise ValueError("resume_state must be a SourceState or mapping")
        self.first_pass = self.state.first_pass
        if resume_state is not None:
            self._validate_resume_cursor()

    def _validate_resume_cursor(self):
        if self.state.pq_idx >= len(self.parquet_paths):
            raise ValueError("resume parquet cursor is out of range")
        parquet_file = pq.ParquetFile(self.parquet_paths[self.state.pq_idx])
        if self.state.rg_idx >= parquet_file.num_row_groups:
            if self.state.rg_idx != self.ddp_rank or self.state.row_offset != 0:
                raise ValueError("resume row-group cursor is out of range")
            return
        if self.state.rg_idx % self.ddp_world_size != self.ddp_rank:
            raise ValueError("resume row-group cursor is not assigned to this rank")
        metadata = getattr(parquet_file, "metadata", None)
        if metadata is not None:
            try:
                row_count = int(metadata.row_group(self.state.rg_idx).num_rows)
            except (AttributeError, TypeError, ValueError, OverflowError):
                row_count = None
            if row_count is not None and self.state.row_offset > row_count:
                raise ValueError("resume row offset is out of range")

    def _set_cursor(
        self,
        pq_idx: int,
        rg_idx: int,
        row_offset: int,
        epoch: int,
        first_pass: bool,
    ):
        self.state.pq_idx = int(pq_idx)
        self.state.rg_idx = int(rg_idx)
        self.state.row_offset = int(row_offset)
        self.state.chunk_offset = int(row_offset)
        self.state.epoch = int(epoch)
        self.state.first_pass = bool(first_pass)
        self.first_pass = bool(first_pass)

    def __iter__(self) -> Iterator[Tuple[List[str], SourceState]]:
        pq_idx = self.state.pq_idx
        rg_idx = self.state.rg_idx
        row_offset = self.state.row_offset
        first_pass = self.state.first_pass
        epoch = self.state.epoch

        while True:
            if pq_idx >= len(self.parquet_paths):
                epoch += 1
                pq_idx = 0
                rg_idx = self.ddp_rank
                row_offset = 0
                first_pass = False
                self._set_cursor(pq_idx, rg_idx, row_offset, epoch, first_pass)
                continue

            filepath = self.parquet_paths[pq_idx]
            parquet_file = pq.ParquetFile(filepath)
            if rg_idx >= parquet_file.num_row_groups:
                pq_idx += 1
                rg_idx = self.ddp_rank
                row_offset = 0
                first_pass = False
                self._set_cursor(pq_idx, rg_idx, row_offset, epoch, first_pass)
                continue

            row_group = parquet_file.read_row_group(rg_idx)
            source_info = get_source(self.source_name)
            text_col = source_info.text_column if source_info else "text"
            if text_col not in row_group.column_names:
                if "text" in row_group.column_names:
                    text_col = "text"
                else:
                    raise KeyError(
                        f"Dataset {self.source_name} has no text column; "
                        f"available columns: {row_group.column_names}"
                    )
            batch = row_group.column(text_col).to_pylist()
            if row_offset > len(batch):
                raise ValueError("resume row offset is out of range")

            if row_offset == len(batch):
                rg_idx += self.ddp_world_size
                row_offset = 0
                if rg_idx >= parquet_file.num_row_groups:
                    pq_idx += 1
                    rg_idx = self.ddp_rank
                    first_pass = False
                    if pq_idx >= len(self.parquet_paths):
                        epoch += 1
                        pq_idx = 0
                        rg_idx = self.ddp_rank
                        first_pass = False
                self._set_cursor(pq_idx, rg_idx, row_offset, epoch, first_pass)
                continue

            while row_offset < len(batch):
                next_offset = min(row_offset + self.tokenizer_batch_size, len(batch))
                mini_batch = batch[row_offset:next_offset]
                self._set_cursor(
                    pq_idx,
                    rg_idx,
                    next_offset,
                    epoch,
                    first_pass,
                )
                yield mini_batch, self.state
                row_offset = self.state.row_offset
                if row_offset >= len(batch):
                    rg_idx += self.ddp_world_size
                    row_offset = 0
                    if rg_idx >= parquet_file.num_row_groups:
                        pq_idx += 1
                        rg_idx = self.ddp_rank
                        first_pass = False
                        if pq_idx >= len(self.parquet_paths):
                            epoch += 1
                            pq_idx = 0
                            rg_idx = self.ddp_rank
                            first_pass = False
                    self._set_cursor(
                        pq_idx,
                        rg_idx,
                        row_offset,
                        epoch,
                        first_pass,
                    )
                    break


def _source_layout(
    source_weights: Dict[str, float],
    subset_weights: Optional[Dict[str, Dict[str, float]]],
    resume_state_dict,
):
    source_names = list(source_weights.keys())
    saved_subset_weights = None
    if isinstance(resume_state_dict, dict):
        saved_subset_weights = resume_state_dict.get("subset_weights")
    subset_names = {}
    normalised_subset_weights = {}
    has_subsets = {}
    for source_name in source_names:
        source = get_source(source_name)
        source_subsets = list(source.subsets)
        if not source_subsets:
            raise ValueError(f"Source {source_name} has no subsets")
        source_has_subsets = len(source_subsets) > 1 or source_subsets[0] != "default"
        has_subsets[source_name] = source_has_subsets
        if source_has_subsets:
            raw_subset_weights = None
            if isinstance(subset_weights, dict) and source_name in subset_weights:
                raw_subset_weights = subset_weights[source_name]
            elif (
                isinstance(saved_subset_weights, dict)
                and source_name in saved_subset_weights
            ):
                raw_subset_weights = saved_subset_weights[source_name]
            if raw_subset_weights is None:
                raise ValueError(
                    f"Subset weights are required for multi-subset source {source_name}"
                )
            normalised = _normalise_weight_map(
                raw_subset_weights, f"subset_weights[{source_name}]"
            )
            normalised_subset_weights[source_name] = normalised
            subset_names[source_name] = list(normalised.keys())
        else:
            subset_names[source_name] = [source_subsets[0]]

    iterator_specs = []
    for source_name in source_names:
        for subset_name in subset_names[source_name]:
            key = (
                f"{source_name}::{subset_name}"
                if has_subsets[source_name]
                else source_name
            )
            iterator_specs.append(
                {
                    "key": key,
                    "source_name": source_name,
                    "subset": subset_name,
                }
            )
    return (
        subset_names,
        normalised_subset_weights,
        iterator_specs,
    )


def tokenizing_distributed_data_loader_multi_source(
    tokenizer,
    source_weights: Dict[str, float],
    B: int,
    T: int,
    split: str = "train",
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict: Optional[Dict] = None,
    buffer_size: int = 1000,
    subset_weights: Optional[Dict[str, Dict[str, float]]] = None,
    seed: int = 42,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict]]:
    """Yield packed token batches and a JSON-compatible loader state."""
    if split not in ["train", "val"]:
        raise ValueError("split must be 'train' or 'val'")
    batch_size = _require_positive_int(B, "B")
    sequence_length = _require_positive_int(T, "T")
    tokenizer_thread_count = _require_positive_int(
        tokenizer_threads, "tokenizer_threads"
    )
    tokenizer_batch = _require_positive_int(
        tokenizer_batch_size, "tokenizer_batch_size"
    )
    document_buffer_size = _require_positive_int(buffer_size, "buffer_size")
    if not _is_int(seed):
        raise ValueError("seed must be an integer")
    device_object = torch.device(device)

    source_weights = _normalise_weight_map(source_weights, "source_weights")
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    ddp_rank = int(ddp_rank)
    ddp_world_size = int(ddp_world_size)
    if ddp_world_size <= 0 or ddp_rank < 0 or ddp_rank >= ddp_world_size:
        raise ValueError("invalid distributed loader configuration")

    subset_names, normalised_subset_weights, iterator_specs = _source_layout(
        source_weights, subset_weights, resume_state_dict
    )
    parsed_resume = None
    if resume_state_dict is not None:
        parsed_resume = _validate_resume_state(
            resume_state_dict,
            list(source_weights.keys()),
            source_weights,
            subset_names,
            normalised_subset_weights,
            iterator_specs,
            batch_size,
            sequence_length,
            split,
            tokenizer_thread_count,
            tokenizer_batch,
            document_buffer_size,
            int(seed),
            ddp_rank,
            ddp_world_size,
        )

    sampler = WeightedSourceSampler(source_weights, seed=seed)
    if parsed_resume is not None:
        sampler.rng.set_state(
            _decode_generator_state(resume_state_dict["sampler_state"]).get_state()
        )

    doc_iterators = {}
    source_states = {}
    subset_samplers = {}
    for source_index, source_name in enumerate(source_weights.keys()):
        source_has_subsets = source_name in normalised_subset_weights
        if source_has_subsets:
            source_subset_weights = normalised_subset_weights[source_name]
            subset_samplers[source_name] = WeightedSourceSampler(
                source_subset_weights, seed=seed + source_index + 1
            )
            if parsed_resume is not None:
                subset_samplers[source_name].rng.set_state(
                    _decode_generator_state(
                        resume_state_dict["subset_sampler_states"][source_name]
                    ).get_state()
                )
            subset_names_for_source = subset_names[source_name]
        else:
            subset_names_for_source = subset_names[source_name]

        for subset_name in subset_names_for_source:
            iterator_key = (
                f"{source_name}::{subset_name}" if source_has_subsets else source_name
            )
            resume_source_state = (
                parsed_resume["source_states"][iterator_key]
                if parsed_resume is not None
                else None
            )
            document_iterator = DocumentBatchIterator(
                source_name=source_name,
                subset=subset_name,
                tokenizer_batch_size=tokenizer_batch,
                resume_state=resume_source_state,
                split=split,
            )
            doc_iterators[iterator_key] = iter(document_iterator)
            source_states[iterator_key] = document_iterator.state

    doc_buffers = {
        key: (
            list(parsed_resume["document_buffers"][key])
            if parsed_resume is not None
            else []
        )
        for key in doc_iterators
    }
    source_token_counts = (
        dict(parsed_resume["source_token_counts"])
        if parsed_resume is not None
        else {source_name: 0 for source_name in source_weights}
    )
    source_document_counts = (
        dict(parsed_resume["source_document_counts"])
        if parsed_resume is not None
        else {source_name: 0 for source_name in source_weights}
    )

    def refill_buffer(iterator_key: str):
        doc_batch, state = next(doc_iterators[iterator_key])
        source_states[iterator_key] = state
        token_lists = tokenizer.encode(
            doc_batch,
            prepend=tokenizer.get_bos_token_id(),
            num_threads=tokenizer_thread_count,
        )
        try:
            token_lists = list(token_lists)
        except TypeError as exc:
            raise ValueError("tokenizer.encode must return an iterable") from exc
        if len(token_lists) != len(doc_batch):
            raise ValueError("tokenizer returned the wrong number of documents")
        for index, tokens in enumerate(token_lists):
            doc_buffers[iterator_key].append(
                _normalise_tokens(tokens, f"tokenizer output[{index}]")
            )

    row_capacity = sequence_length + 1
    use_cuda = device_object.type == "cuda"
    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(
        2 * batch_size * sequence_length, dtype=torch.long, pin_memory=use_cuda
    )
    device_buffer = torch.empty(
        2 * batch_size * sequence_length, dtype=torch.long, device=device_object
    )
    cpu_inputs = cpu_buffer[: batch_size * sequence_length].view(
        batch_size, sequence_length
    )
    cpu_targets = cpu_buffer[batch_size * sequence_length :].view(
        batch_size, sequence_length
    )
    inputs = device_buffer[: batch_size * sequence_length].view(
        batch_size, sequence_length
    )
    targets = device_buffer[batch_size * sequence_length :].view(
        batch_size, sequence_length
    )

    pending_inputs = None
    pending_targets = None
    if parsed_resume is not None:
        pending_inputs = parsed_resume["pending_inputs"]
        pending_targets = parsed_resume["pending_targets"]
        cpu_inputs.copy_(pending_inputs)
        cpu_targets.copy_(pending_targets)

    iterator_names = [spec["key"] for spec in iterator_specs]

    def make_state_dict():
        return {
            "schema_version": LOADER_STATE_SCHEMA_VERSION,
            "source_names": list(source_weights.keys()),
            "source_weights": dict(source_weights),
            "subset_names": {
                source_name: list(names) for source_name, names in subset_names.items()
            },
            "subset_weights": {
                source_name: dict(weights)
                for source_name, weights in normalised_subset_weights.items()
            },
            "iterator_names": list(iterator_names),
            "iterator_specs": [dict(spec) for spec in iterator_specs],
            "loader_config": {
                "B": batch_size,
                "T": sequence_length,
                "split": split,
                "tokenizer_threads": tokenizer_thread_count,
                "tokenizer_batch_size": tokenizer_batch,
                "buffer_size": document_buffer_size,
                "seed": int(seed),
                "ddp_rank": ddp_rank,
                "ddp_world_size": ddp_world_size,
            },
            "sampler_state": _encode_generator_state(sampler.rng),
            "subset_sampler_states": {
                source_name: _encode_generator_state(subset_sampler.rng)
                for source_name, subset_sampler in subset_samplers.items()
            },
            "source_states": {
                key: _source_state_to_dict(source_states[key]) for key in iterator_names
            },
            "document_buffers": {
                key: [
                    [int(token) for token in document] for document in doc_buffers[key]
                ]
                for key in iterator_names
            },
            "source_token_counts": {
                source_name: int(source_token_counts[source_name])
                for source_name in source_weights
            },
            "source_document_counts": {
                source_name: int(source_document_counts[source_name])
                for source_name in source_weights
            },
            "pending_batch": {
                "inputs": cpu_inputs.detach().to(device="cpu").clone().tolist(),
                "targets": cpu_targets.detach().to(device="cpu").clone().tolist(),
                "shape": [batch_size, sequence_length],
                "dtype": "torch.int64",
            },
        }

    if pending_inputs is not None and pending_targets is not None:
        device_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        state_dict = make_state_dict()
        pending_inputs = None
        pending_targets = None
        yield inputs, targets, state_dict

    def record_document(source_name: str, token_count: int):
        source_token_counts[source_name] = source_token_counts.get(
            source_name, 0
        ) + int(token_count)
        source_document_counts[source_name] = (
            source_document_counts.get(source_name, 0) + 1
        )

    while True:
        for row_idx in range(batch_size):
            pos = 0
            while pos < row_capacity:
                source_name = sampler.sample()
                if source_name in subset_samplers:
                    subset_name = subset_samplers[source_name].sample()
                    iterator_key = f"{source_name}::{subset_name}"
                else:
                    iterator_key = source_name

                while len(doc_buffers[iterator_key]) < document_buffer_size:
                    refill_buffer(iterator_key)

                remaining = row_capacity - pos
                buffer = doc_buffers[iterator_key]
                best_idx = -1
                best_len = 0
                for index, document in enumerate(buffer):
                    document_length = len(document)
                    if document_length <= remaining and document_length > best_len:
                        best_idx = index
                        best_len = document_length

                if best_idx >= 0:
                    document = buffer.pop(best_idx)
                    document_length = len(document)
                    record_document(source_name, document_length)
                    row_buffer[row_idx, pos : pos + document_length] = torch.tensor(
                        document, dtype=torch.long
                    )
                    pos += document_length
                else:
                    shortest_idx = min(
                        range(len(buffer)), key=lambda index: len(buffer[index])
                    )
                    document = buffer.pop(shortest_idx)
                    record_document(source_name, remaining)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        document[:remaining], dtype=torch.long
                    )
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        state_dict = make_state_dict()
        device_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_multi_source_with_curriculum(
    tokenizer,
    curriculum_scheduler,
    B: int,
    T_fn,
    split: str = "train",
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict: Optional[Dict] = None,
    buffer_size: int = 1000,
):
    """Create a curriculum-aware multi-source dataloader."""
    initial_weights = curriculum_scheduler.get_source_weights(step=0)
    return tokenizing_distributed_data_loader_multi_source(
        tokenizer=tokenizer,
        source_weights=initial_weights,
        B=B,
        T=T_fn() if callable(T_fn) else T_fn,
        split=split,
        tokenizer_threads=tokenizer_threads,
        tokenizer_batch_size=tokenizer_batch_size,
        device=device,
        resume_state_dict=resume_state_dict,
        buffer_size=buffer_size,
    )
