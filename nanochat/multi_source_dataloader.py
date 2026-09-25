"""
Weighted multi-source dataloader with curriculum support.

This dataloader extends the BOS-aligned best-fit packing approach to handle
multiple dataset sources with configurable sampling weights. It supports:
- Weighted sampling from multiple sources (document-level mixing)
- Per-source best-fit document packing
- Dynamic weight updates (for curriculum stage transitions)
- Context length scheduling
- Full DDP support with proper sharding across all sources
- Checkpoint/resume with per-source state tracking
"""

import torch
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass

from nanochat.common import get_dist_info
from nanochat.data_registry import list_parquet_files_for_source, get_source


@dataclass
class SourceState:
    """Track iteration state for a single data source."""

    pq_idx: int = 0  # Current parquet file index
    rg_idx: int = 0  # Current row group index
    epoch: int = 1  # Current epoch through this source


def _encode_generator_state(generator: torch.Generator) -> list[int]:
    return [int(value) for value in generator.get_state().tolist()]


def _decode_generator_state(state: list[int]) -> torch.Generator:
    generator = torch.Generator()
    generator.set_state(torch.tensor(state, dtype=torch.uint8))
    return generator


class WeightedSourceSampler:
    """
    Samples data source names according to weights.

    Uses reservoir sampling to handle dynamic weight updates efficiently.
    """

    def __init__(self, source_weights: Dict[str, float], seed: int = 42):
        """
        Initialize sampler with source weights.

        Args:
            source_weights: Dictionary mapping source name to sampling weight
            seed: Random seed for reproducibility
        """
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
    """
    Infinite iterator over document batches from a single source.

    Handles DDP sharding and resumption for one data source.
    """

    def __init__(
        self,
        source_name: str,
        subset: str = "default",
        tokenizer_batch_size: int = 128,
        resume_state: Optional[SourceState] = None,
        split: str = "train",
    ):
        """
        Initialize document batch iterator for a single source.

        Args:
            source_name: Name of the data source
            subset: Subset name for multi-subset datasets
            tokenizer_batch_size: Number of documents per batch
            resume_state: State to resume from (optional)
            split: train or val
        """
        self.source_name = source_name
        self.subset = subset
        self.tokenizer_batch_size = tokenizer_batch_size
        self.split = split

        # Get DDP info
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size

        # Get parquet files for this source
        self.parquet_paths = list_parquet_files_for_source(source_name, subset)
        assert (
            len(self.parquet_paths) > 0
        ), f"No parquet files found for {source_name}/{subset}. Run data download first."

        # For val split, use only the last file; for train, use all but last
        if split == "val":
            self.parquet_paths = self.parquet_paths[-1:]
        else:
            self.parquet_paths = (
                self.parquet_paths[:-1]
                if len(self.parquet_paths) > 1
                else self.parquet_paths
            )

        # Initialize state
        if resume_state is not None:
            self.state = resume_state
        else:
            self.state = SourceState()

        self.first_pass = True

    def __iter__(self) -> Iterator[Tuple[List[str], SourceState]]:
        """Iterate indefinitely, yielding document batches."""
        while True:  # Multi-epoch
            pq_idx = self.state.pq_idx if self.first_pass else 0

            while pq_idx < len(self.parquet_paths):
                filepath = self.parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)

                # Determine starting row group for DDP sharding
                if self.first_pass and pq_idx == self.state.pq_idx:
                    # Resume from saved position
                    base_idx = self.state.rg_idx // self.ddp_world_size
                    base_idx += 1  # Advance by 1 to avoid repeating data
                    rg_idx = base_idx * self.ddp_world_size + self.ddp_rank
                else:
                    # Start from beginning with DDP offset
                    rg_idx = self.ddp_rank

                # Process row groups for this file
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    # Use the text column specified in the source metadata
                    source_info = get_source(self.source_name)
                    text_col = source_info.text_column if source_info else "text"
                    if text_col not in rg.column_names:
                        if "text" in rg.column_names:
                            text_col = "text"
                        else:
                            raise KeyError(
                                f"Dataset {self.source_name} has no text column; "
                                f"available columns: {rg.column_names}"
                            )
                    batch = rg.column(text_col).to_pylist()

                    # Yield in tokenizer_batch_size chunks
                    for i in range(0, len(batch), self.tokenizer_batch_size):
                        mini_batch = batch[i : i + self.tokenizer_batch_size]
                        self.state.pq_idx = pq_idx
                        self.state.rg_idx = rg_idx
                        yield mini_batch, self.state

                    rg_idx += self.ddp_world_size

                pq_idx += 1
                self.first_pass = False

            # Completed an epoch
            self.state.epoch += 1
            self.state.pq_idx = 0
            self.state.rg_idx = 0
            self.first_pass = False


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
    """
    Multi-source weighted dataloader with BOS-aligned best-fit packing.

    This is a drop-in replacement for the single-source dataloader, but samples
    documents from multiple sources according to configurable weights.

    Args:
        tokenizer: Tokenizer instance
        source_weights: Dictionary mapping source name to sampling weight
        B: Batch size
        T: Sequence length
        split: "train" or "val"
        tokenizer_threads: Number of tokenizer threads
        tokenizer_batch_size: Batch size for tokenization
        device: Device to place tensors on
        resume_state_dict: State dictionary for resumption
        buffer_size: Number of documents to buffer per source
        subset_weights: Optional per-source subset weights

    Yields:
        Tuple of (inputs, targets, state_dict)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    bos_token = tokenizer.get_bos_token_id()

    # Initialize source sampler
    sampler = WeightedSourceSampler(source_weights, seed=seed)
    if resume_state_dict and resume_state_dict.get("sampler_state"):
        sampler.rng.set_state(
            _decode_generator_state(resume_state_dict["sampler_state"])
        )

    # Import get_source to check for subsets
    from nanochat.data_registry import get_source

    # Initialize document iterators for each source
    # For sources with subsets, we create multiple iterators (one per subset)
    # and sample from them according to subset_weights
    doc_iterators = {}
    source_states = {}
    subset_samplers = {}  # Per-source subset samplers

    for source_index, source_name in enumerate(source_weights.keys()):
        source = get_source(source_name)

        # Check if this source has multiple subsets
        has_subsets = len(source.subsets) > 1 or source.subsets[0] != "default"

        if has_subsets:
            if not subset_weights or source_name not in subset_weights:
                raise ValueError(
                    f"Subset weights are required for multi-subset source {source_name}"
                )
            source_subset_weights = subset_weights[source_name]
            subset_samplers[source_name] = WeightedSourceSampler(
                source_subset_weights, seed=seed + source_index + 1
            )
            saved_subset_state = (
                (resume_state_dict or {})
                .get("subset_sampler_states", {})
                .get(source_name)
            )
            if saved_subset_state:
                subset_samplers[source_name].rng.set_state(
                    _decode_generator_state(saved_subset_state)
                )

            for subset_name in source_subset_weights.keys():
                iterator_key = f"{source_name}::{subset_name}"

                # Resume state if available
                resume_source_state = None
                if resume_state_dict and iterator_key in resume_state_dict.get(
                    "source_states", {}
                ):
                    state_data = resume_state_dict["source_states"][iterator_key]
                    resume_source_state = SourceState(
                        pq_idx=state_data["pq_idx"],
                        rg_idx=state_data["rg_idx"],
                        epoch=state_data["epoch"],
                    )

                doc_iterators[iterator_key] = iter(
                    DocumentBatchIterator(
                        source_name=source_name,
                        subset=subset_name,
                        tokenizer_batch_size=tokenizer_batch_size,
                        resume_state=resume_source_state,
                        split=split,
                    )
                )
                source_states[iterator_key] = resume_source_state or SourceState()
        else:
            # Single-subset source: create one iterator with "default" subset
            iterator_key = source_name

            # Resume state if available
            resume_source_state = None
            if resume_state_dict and iterator_key in resume_state_dict.get(
                "source_states", {}
            ):
                state_data = resume_state_dict["source_states"][iterator_key]
                resume_source_state = SourceState(
                    pq_idx=state_data["pq_idx"],
                    rg_idx=state_data["rg_idx"],
                    epoch=state_data["epoch"],
                )

            doc_iterators[iterator_key] = iter(
                DocumentBatchIterator(
                    source_name=source_name,
                    subset=source.subsets[0],
                    tokenizer_batch_size=tokenizer_batch_size,
                    resume_state=resume_source_state,
                    split=split,
                )
            )
            source_states[iterator_key] = resume_source_state or SourceState()

    # Per-iterator document buffers (one buffer per iterator, not per source)
    doc_buffers = {key: [] for key in doc_iterators.keys()}
    source_token_counts = {
        source_name: int(
            (resume_state_dict or {}).get("source_token_counts", {}).get(source_name, 0)
        )
        for source_name in source_weights
    }
    source_document_counts = {
        source_name: int(
            (resume_state_dict or {})
            .get("source_document_counts", {})
            .get(source_name, 0)
        )
        for source_name in source_weights
    }

    def refill_buffer(iterator_key: str):
        """Refill buffer for a specific iterator."""
        doc_batch, state = next(doc_iterators[iterator_key])
        source_states[iterator_key] = state
        token_lists = tokenizer.encode(
            doc_batch, prepend=bos_token, num_threads=tokenizer_threads
        )
        for tokens in token_lists:
            doc_buffers[iterator_key].append(tokens)

    # Pre-allocate tensors
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    def record_document(source_name: str, token_count: int):
        source_token_counts[source_name] = source_token_counts.get(
            source_name, 0
        ) + int(token_count)
        source_document_counts[source_name] = (
            source_document_counts.get(source_name, 0) + 1
        )

    # Main loop
    while True:
        for row_idx in range(B):
            pos = 0

            while pos < row_capacity:
                # Sample a source
                source_name = sampler.sample()

                # Determine which iterator to use
                if source_name in subset_samplers:
                    # Multi-subset source: sample a subset
                    subset_name = subset_samplers[source_name].sample()
                    iterator_key = f"{source_name}::{subset_name}"
                else:
                    # Single-subset source
                    iterator_key = source_name

                # Ensure buffer has documents
                while len(doc_buffers[iterator_key]) < buffer_size:
                    refill_buffer(iterator_key)

                remaining = row_capacity - pos

                # Best-fit: find largest doc that fits entirely
                buffer = doc_buffers[iterator_key]
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    # Found a doc that fits
                    doc = buffer.pop(best_idx)
                    doc_len = len(doc)
                    record_document(source_name, doc_len)
                    row_buffer[row_idx, pos : pos + doc_len] = torch.tensor(
                        doc, dtype=torch.long
                    )
                    pos += doc_len
                else:
                    # No doc fits - crop shortest to fill remaining
                    shortest_idx = min(range(len(buffer)), key=lambda i: len(buffer[i]))
                    doc = buffer.pop(shortest_idx)
                    record_document(source_name, remaining)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    pos += remaining

        # Copy to CPU then GPU
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Build state dict for checkpointing
        state_dict = {
            "source_weights": source_weights,
            "source_token_counts": dict(source_token_counts),
            "source_document_counts": dict(source_document_counts),
            "sampler_state": _encode_generator_state(sampler.rng),
            "subset_sampler_states": {
                name: _encode_generator_state(subset_sampler.rng)
                for name, subset_sampler in subset_samplers.items()
            },
            "source_states": {
                name: {
                    "pq_idx": state.pq_idx,
                    "rg_idx": state.rg_idx,
                    "epoch": state.epoch,
                }
                for name, state in source_states.items()
            },
        }

        # Single HtoD copy
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_multi_source_with_curriculum(
    tokenizer,
    curriculum_scheduler,
    B: int,
    T_fn,  # Function that returns current context length: lambda step: scheduler.get_context_length(step=step)
    split: str = "train",
    tokenizer_threads: int = 4,
    tokenizer_batch_size: int = 128,
    device: str = "cuda",
    resume_state_dict: Optional[Dict] = None,
    buffer_size: int = 1000,
):
    """
    Curriculum-aware multi-source dataloader.

    This is a wrapper around the multi-source loader that updates source weights
    and context length according to the curriculum schedule.

    Note: Context length changes require re-initialization of the dataloader,
    so this is provided as a factory function that should be called when T changes.

    Args:
        tokenizer: Tokenizer instance
        curriculum_scheduler: CurriculumScheduler instance
        B: Batch size
        T_fn: Function that returns current context length given training step
        split: "train" or "val"
        tokenizer_threads: Number of tokenizer threads
        tokenizer_batch_size: Batch size for tokenization
        device: Device to place tensors on
        resume_state_dict: State dictionary for resumption
        buffer_size: Number of documents to buffer per source

    Yields:
        Tuple of (inputs, targets, state_dict)
    """
    # This is a simplified version - in practice, T changes would require
    # rebuilding the dataloader. For now, we document that T should be managed
    # externally and the dataloader recreated when T changes.

    # Get initial source weights from curriculum
    # (In practice, step would be passed in or tracked)
    initial_weights = curriculum_scheduler.get_source_weights(step=0)

    # Delegate to the base multi-source loader
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
