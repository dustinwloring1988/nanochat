from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import numbers
import os
import random
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from nanochat.research_results import sha256_file, write_json_atomic
from nanochat.sft_manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    GLOBAL_CHECKPOINT_NAMESPACES,
    SFT_MANIFEST_SCHEMA_VERSION,
    DatasetFile,
    DatasetManifest,
    ManifestValidationError,
    RunLocalPathError,
    SFTContractError,
    SFTDatasetManifest,
    SFTManifest,
    SFTManifestError,
    SFTPathError,
    ValidatedDatasetManifest,
    _absolute,
    canonical_json_hash,
    ensure_run_local_directory,
    load_dataset_manifest,
    read_jsonl_records,
    resolve_run_local_path,
    resolve_run_path,
    safe_run_local_path,
    validate_dataset_manifest,
    validate_jsonl_dataset_manifest,
    validate_local_jsonl_manifest,
    validate_sft_manifest,
    write_dataset_manifest,
    write_jsonl_dataset_manifest,
)

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None


resolve_run_local_output = resolve_run_local_path
resolve_run_local_directory = resolve_run_local_path


SFT_CONTEXT_TOKENS = 2048
SFT_CONTEXT_LENGTH = SFT_CONTEXT_TOKENS
FIXED_SFT_CONTEXT_TOKENS = SFT_CONTEXT_TOKENS
SFT_SEQUENCE_LENGTH = SFT_CONTEXT_TOKENS
SFT_RENDER_TOKEN_LIMIT = SFT_CONTEXT_TOKENS + 1
SFT_LOADER_STATE_SCHEMA_VERSION = 1
SFT_BATCH_STATE_SCHEMA_VERSION = SFT_LOADER_STATE_SCHEMA_VERSION
SFT_CHECKPOINT_SCHEMA_VERSION = 1
SFT_CHECKPOINT_ENVELOPE_SCHEMA_VERSION = SFT_CHECKPOINT_SCHEMA_VERSION
SFT_BUDGET_SCHEMA_VERSION = 1
LAST_KNOWN_GOOD_POINTER = "last_known_good.json"
LAST_KNOWN_GOOD = LAST_KNOWN_GOOD_POINTER
COMPLETION_MARKER = "COMPLETED.json"
ENVELOPE_FILE = "envelope.json"


class SFTBudgetError(SFTContractError):
    pass


class SFTLoaderError(SFTContractError):
    pass


class SFTCheckpointError(SFTContractError):
    pass


class SFTTransactionError(SFTCheckpointError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: Any, field_name: str) -> int:
    if not _is_int(value) or value <= 0:
        raise SFTBudgetError(f"{field_name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if not _is_int(value) or value < 0:
        raise SFTBudgetError(f"{field_name} must be a non-negative integer")
    return int(value)


def _fixed_context(value: Any, field_name: str = "sequence_length") -> int:
    if not _is_int(value) or value != SFT_CONTEXT_TOKENS:
        raise SFTLoaderError(
            f"{field_name} must be the fixed approved value {SFT_CONTEXT_TOKENS}"
        )
    return int(value)


def validate_fixed_context(
    sequence_length: Any = SFT_CONTEXT_TOKENS,
    *,
    context_range: Any = None,
    buckets: Any = None,
    max_seq_len: Any = None,
) -> int:
    values = [sequence_length]
    if context_range is not None:
        if not isinstance(context_range, (list, tuple)) or len(context_range) != 2:
            raise SFTLoaderError("context_range must contain exactly two values")
        values.extend(context_range)
    if buckets is not None:
        if not isinstance(buckets, (list, tuple)):
            raise SFTLoaderError("buckets must be a sequence")
        values.extend(buckets)
    if max_seq_len is not None:
        values.append(max_seq_len)
    for value in values:
        _fixed_context(value)
    return SFT_CONTEXT_TOKENS


class FixedContextProfile(dict):
    @property
    def sequence_length(self) -> int:
        return int(self["sequence_length"])

    @property
    def context_length(self) -> int:
        return int(self["context_length"])

    @property
    def buckets(self) -> tuple[int, ...]:
        return tuple(self["buckets"])

    @property
    def enabled(self) -> bool:
        return bool(self["dynamic_context_enabled"])


def validate_context_contract(*args, **kwargs) -> int:
    return validate_fixed_context(*args, **kwargs)


def fixed_context_profile() -> FixedContextProfile:
    return FixedContextProfile(fixed_context_contract())


def fixed_context_contract() -> dict:
    return {
        "schema_version": 1,
        "context_length": SFT_CONTEXT_TOKENS,
        "sequence_length": SFT_CONTEXT_TOKENS,
        "context_tokens": SFT_CONTEXT_TOKENS,
        "context_range": [SFT_CONTEXT_TOKENS, SFT_CONTEXT_TOKENS],
        "buckets": [SFT_CONTEXT_TOKENS],
        "dynamic_context_enabled": False,
        "renderer_max_tokens": SFT_RENDER_TOKEN_LIMIT,
        "padding": "right_with_bos",
        "truncation": "right_to_fixed_context",
        "loss_mask": "assistant_only_shifted",
    }


@dataclass(frozen=True)
class FixedTokenBudget:
    schema_version: int
    device_batch_size: int
    sequence_length: int
    world_size: int
    gradient_accumulation_steps: int
    effective_tokens: int
    total_batch_size: int

    def __post_init__(self) -> None:
        if self.schema_version != SFT_BUDGET_SCHEMA_VERSION:
            raise SFTBudgetError("unsupported effective token budget schema")
        _fixed_context(self.sequence_length)
        _positive_int(self.device_batch_size, "device_batch_size")
        _positive_int(self.world_size, "world_size")
        _positive_int(
            self.gradient_accumulation_steps,
            "gradient_accumulation_steps",
        )
        _positive_int(self.effective_tokens, "effective_tokens")
        _positive_int(self.total_batch_size, "total_batch_size")
        expected = (
            self.device_batch_size
            * SFT_CONTEXT_TOKENS
            * self.world_size
            * self.gradient_accumulation_steps
        )
        if self.effective_tokens != expected or self.total_batch_size != expected:
            raise SFTBudgetError("effective token budget arithmetic is invalid")

    @property
    def effective_token_budget(self) -> int:
        return self.effective_tokens

    @property
    def total_sequence_batch(self) -> int:
        return (
            self.device_batch_size * self.world_size * self.gradient_accumulation_steps
        )

    def to_dict(self) -> dict:
        return asdict(self)


def build_effective_token_budget(
    device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int = 1,
    *,
    sequence_length: int = SFT_CONTEXT_TOKENS,
    total_batch_size: int | None = None,
    total_sequence_batch: int | None = None,
) -> FixedTokenBudget:
    device_batch_size = _positive_int(device_batch_size, "device_batch_size")
    world_size = _positive_int(world_size, "world_size")
    gradient_accumulation_steps = _positive_int(
        gradient_accumulation_steps, "gradient_accumulation_steps"
    )
    _fixed_context(sequence_length)
    calculated_sequences = device_batch_size * world_size * gradient_accumulation_steps
    effective_tokens = (
        device_batch_size
        * SFT_CONTEXT_TOKENS
        * world_size
        * gradient_accumulation_steps
    )
    if total_sequence_batch is not None:
        total_sequence_batch = _positive_int(
            total_sequence_batch, "total_sequence_batch"
        )
        if total_sequence_batch != calculated_sequences:
            raise SFTBudgetError(
                "total_sequence_batch does not match device batch, world size, and accumulation"
            )
    if total_batch_size is not None:
        total_batch_size = _positive_int(total_batch_size, "total_batch_size")
        if total_batch_size != effective_tokens:
            raise SFTBudgetError(
                "total_batch_size does not match the fixed effective token budget"
            )
    return FixedTokenBudget(
        schema_version=SFT_BUDGET_SCHEMA_VERSION,
        device_batch_size=device_batch_size,
        sequence_length=SFT_CONTEXT_TOKENS,
        world_size=world_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_tokens=effective_tokens,
        total_batch_size=effective_tokens,
    )


def compute_effective_tokens(
    device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int = 1,
    *,
    sequence_length: int = SFT_CONTEXT_TOKENS,
) -> int:
    return build_effective_token_budget(
        device_batch_size,
        world_size,
        gradient_accumulation_steps,
        sequence_length=sequence_length,
    ).effective_tokens


def validate_effective_token_budget(
    device_batch_size: Any = None,
    world_size: Any = None,
    gradient_accumulation_steps: Any = 1,
    expected_effective_tokens: Any = None,
    *,
    sequence_length: Any = SFT_CONTEXT_TOKENS,
    total_batch_size: Any = None,
    total_sequence_batch: Any = None,
    token_horizon: Any = None,
    optimization_steps: Any = None,
    effective_token_budget: Any = None,
    context_length: Any = None,
    effective_token_batch: Any = None,
) -> int:
    if isinstance(device_batch_size, Mapping):
        config = dict(device_batch_size)
        device_batch_size = config.get(
            "device_batch_size", config.get("B", config.get("batch_size"))
        )
        world_size = config.get("world_size", config.get("ddp_world_size"))
        gradient_accumulation_steps = config.get(
            "gradient_accumulation_steps",
            config.get("gradient_accumulation", 1),
        )
        expected_effective_tokens = config.get(
            "expected_effective_tokens",
            config.get(
                "effective_token_budget",
                config.get("effective_token_batch", config.get("effective_tokens")),
            ),
        )
        sequence_length = config.get(
            "sequence_length",
            config.get("context_length", config.get("T", sequence_length)),
        )
        total_batch_size = config.get("total_batch_size")
        total_sequence_batch = config.get("total_sequence_batch")
        token_horizon = config.get("token_horizon")
        optimization_steps = config.get("optimization_steps", config.get("steps"))
    if context_length is not None:
        sequence_length = context_length
    if effective_token_batch is not None:
        if expected_effective_tokens is not None:
            raise SFTBudgetError("declare the effective token budget only once")
        expected_effective_tokens = effective_token_batch
    if world_size is None:
        raise SFTBudgetError("world_size is required")
    if device_batch_size is None:
        raise SFTBudgetError("device_batch_size is required")
    if expected_effective_tokens is None:
        expected_effective_tokens = effective_token_budget
    _fixed_context(sequence_length)
    budget = build_effective_token_budget(
        device_batch_size,
        world_size,
        gradient_accumulation_steps,
        sequence_length=sequence_length,
        total_batch_size=total_batch_size,
        total_sequence_batch=total_sequence_batch,
    )
    if expected_effective_tokens is not None:
        if not _is_int(expected_effective_tokens) or expected_effective_tokens <= 0:
            raise SFTBudgetError("expected effective token budget must be positive")
        if expected_effective_tokens != budget.effective_tokens:
            raise SFTBudgetError(
                "effective token budget does not match the fixed-token formula"
            )
    if token_horizon is not None:
        token_horizon = _positive_int(token_horizon, "token_horizon")
        if optimization_steps is None:
            if token_horizon % budget.effective_tokens:
                raise SFTBudgetError(
                    "token_horizon is not divisible by the fixed budget"
                )
        else:
            optimization_steps = _positive_int(optimization_steps, "optimization_steps")
            if token_horizon != budget.effective_tokens * optimization_steps:
                raise SFTBudgetError("token_horizon does not match the fixed budget")
    return int(budget.effective_tokens)


def validate_fixed_token_budget(*args, **kwargs) -> int:
    return validate_effective_token_budget(*args, **kwargs)


def validate_fixed_effective_token_budget(*args, **kwargs) -> int:
    return validate_effective_token_budget(*args, **kwargs)


def validate_token_budget(*args, **kwargs) -> int:
    return validate_effective_token_budget(*args, **kwargs)


def _json_copy(value: Any, field_name: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise SFTLoaderError(f"{field_name} must be JSON-compatible") from exc


def _tensor_values(
    value: Any,
    field_name: str,
    *,
    allow_bool: bool = False,
) -> list:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim != 1:
            raise SFTLoaderError(f"{field_name} must be one-dimensional")
        value = value.detach().cpu().tolist()
    try:
        values = list(value)
    except TypeError as exc:
        raise SFTLoaderError(f"{field_name} must be a sequence") from exc
    if any(
        (isinstance(item, bool) and not allow_bool)
        or not isinstance(item, numbers.Integral)
        for item in values
    ):
        raise SFTLoaderError(f"{field_name} must contain integer values")
    return [int(item) for item in values]


def _tensor_matrix(
    value: Any, rows: int, columns: int, field_name: str
) -> list[list[int]]:
    if torch is not None and isinstance(value, torch.Tensor):
        if tuple(value.shape) != (rows, columns):
            raise SFTLoaderError(f"{field_name} has an invalid shape")
        value = value.detach().cpu().tolist()
    if not isinstance(value, list) or len(value) != rows:
        raise SFTLoaderError(f"{field_name} has an invalid batch dimension")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise SFTLoaderError(f"{field_name} has an invalid sequence dimension")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in row):
            raise SFTLoaderError(f"{field_name} must contain integer values")
        result.append([int(item) for item in row])
    return result


def _random_state_json(state: tuple) -> list:
    gauss = None if state[2] is None else int(state[2])
    return [int(state[0]), [int(item) for item in state[1]], gauss]


def _random_state_from_json(value: Any, field_name: str) -> tuple:
    if not isinstance(value, list) or len(value) != 3:
        raise SFTLoaderError(f"{field_name} has an invalid random state")
    version = value[0]
    keys = value[1]
    gauss = value[2]
    if not _is_int(version) or not isinstance(keys, list):
        raise SFTLoaderError(f"{field_name} has an invalid random state")
    if gauss is not None and not _is_int(gauss):
        raise SFTLoaderError(f"{field_name} has an invalid random state")
    if any(not _is_int(item) or item < 0 for item in keys):
        raise SFTLoaderError(f"{field_name} has an invalid random state")
    return int(version), tuple(int(item) for item in keys), gauss


def _epoch_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"sft-loader:{seed}:{epoch}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class ConversationBatch:
    inputs: Any
    targets: Any
    loss_mask: Any
    attention_mask: Any
    state: dict
    source_names: tuple[str, ...] = ()
    metadata: tuple[dict, ...] = ()

    @property
    def x(self):
        return self.inputs

    @property
    def y(self):
        return self.targets

    @property
    def mask(self):
        return self.loss_mask

    def __iter__(self):
        yield self.inputs
        yield self.targets
        yield copy.deepcopy(self.state)

    def __getitem__(self, key: Any):
        if isinstance(key, str):
            if key == "inputs" or key == "x":
                return self.inputs
            if key == "targets" or key == "y":
                return self.targets
            if key in {"loss_mask", "mask"}:
                return self.loss_mask
            if key == "attention_mask":
                return self.attention_mask
            if key == "state":
                return self.state
            raise KeyError(key)
        return (self.inputs, self.targets, self.state)[key]

    def to_state(self) -> dict:
        return copy.deepcopy(self.state)

    def as_tuple(self) -> tuple:
        return self.inputs, self.targets, self.loss_mask, self.state

    def iter_with_mask(self):
        yield self.inputs
        yield self.targets
        yield self.loss_mask
        yield copy.deepcopy(self.state)


class SFTConversationBatchLoader:
    def __init__(
        self,
        manifest: Any = None,
        tokenizer: Any = None,
        *,
        batch_size: int = 1,
        device: Any = "cpu",
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        sequence_length: int = SFT_CONTEXT_TOKENS,
        context_length: int | None = None,
        max_seq_len: int | None = None,
        pad_token_id: int | None = None,
        resume_state: dict | None = None,
        resume_state_dict: dict | None = None,
        manifest_path: Any = None,
        run_root: Any = None,
        ddp_rank: int | None = None,
        ddp_world_size: int | None = None,
        B: int | None = None,
        T: int | None = None,
    ):
        if manifest is None:
            manifest = manifest_path
        if manifest is None:
            raise SFTLoaderError("a validated dataset manifest is required")
        if tokenizer is None or not callable(
            getattr(tokenizer, "render_conversation", None)
        ):
            raise SFTLoaderError("tokenizer must provide render_conversation")
        if ddp_rank is not None:
            rank = ddp_rank
        if ddp_world_size is not None:
            world_size = ddp_world_size
        if B is not None:
            batch_size = B
        if T is not None:
            sequence_length = T
        if context_length is not None:
            sequence_length = context_length
        if max_seq_len is not None:
            sequence_length = max_seq_len
        if resume_state is not None and resume_state_dict is not None:
            raise SFTLoaderError("provide only one resume state")
        if resume_state is None:
            resume_state = resume_state_dict
        if not _is_int(batch_size) or batch_size <= 0:
            raise SFTLoaderError("batch_size must be a positive integer")
        if not _is_int(rank) or rank < 0:
            raise SFTLoaderError("rank must be a non-negative integer")
        if not _is_int(world_size) or world_size <= 0 or rank >= world_size:
            raise SFTLoaderError("world_size and rank are inconsistent")
        if not _is_int(seed):
            raise SFTLoaderError("seed must be an integer")
        _fixed_context(sequence_length)
        if isinstance(manifest, ValidatedDatasetManifest):
            if (
                run_root is not None
                and Path(os.path.realpath(_absolute(run_root))) != manifest.run_root
            ):
                raise SFTLoaderError("manifest and loader run roots do not match")
            validated_manifest = validate_jsonl_dataset_manifest(
                manifest.manifest_path,
                run_root=manifest.run_root,
            )
        elif isinstance(manifest, (str, os.PathLike)):
            validated_manifest = validate_jsonl_dataset_manifest(
                manifest,
                run_root=run_root,
            )
        else:
            raise SFTLoaderError("manifest must be a path or ValidatedDatasetManifest")
        if torch is None:
            raise SFTLoaderError("torch is required for the SFT batch loader")
        self.tokenizer = tokenizer
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.sequence_length = SFT_CONTEXT_TOKENS
        self.manifest = validated_manifest
        self.manifest_hash = validated_manifest.manifest_hash
        self.run_root = validated_manifest.run_root
        if pad_token_id is None:
            get_bos = getattr(tokenizer, "get_bos_token_id", None)
            if callable(get_bos):
                pad_token_id = get_bos()
            else:
                pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                raise SFTLoaderError("tokenizer must provide a padding token")
        if not _is_int(pad_token_id) or pad_token_id < 0:
            raise SFTLoaderError("pad_token_id must be a non-negative integer")
        self.pad_token_id = int(pad_token_id)
        self._records: list[dict] = []
        self._file_keys: list[str] = []
        for dataset_file in validated_manifest.files:
            records = read_jsonl_records(dataset_file.path, self.run_root)
            for line_number, record in enumerate(records, 1):
                conversation = self._conversation_from_record(
                    record, dataset_file, line_number
                )
                self._records.append(
                    {
                        "conversation": conversation,
                        "source": dataset_file.source,
                        "split": dataset_file.split,
                        "license": dataset_file.license,
                        "provenance": dataset_file.provenance,
                        "file": dataset_file.relative_path,
                        "line": line_number,
                    }
                )
            self._file_keys.append(dataset_file.relative_path)
        if not self._records:
            raise SFTLoaderError("dataset manifest contains no conversation records")
        self._source_keys = tuple(
            sorted({f"{item['source']}/{item['split']}" for item in self._records})
        )
        self._counters = self._new_counters()
        self._rng = random.Random(self.seed)
        self._epoch = 1
        self._cursor = 0
        self._last_batch: ConversationBatch | None = None
        self._replay_batch: ConversationBatch | None = None
        self._order, self._order_hash = self._make_order(self._epoch)
        if resume_state is not None:
            self.load_state_dict(resume_state)
        else:
            self._rebuild_rank_order()

    def _new_counters(self) -> dict:
        return {
            "examples_seen": 0,
            "batches_seen": 0,
            "total_input_tokens": 0,
            "total_target_tokens": 0,
            "total_attention_tokens": 0,
            "total_padding_tokens": 0,
            "total_truncated_tokens": 0,
            "source_document_counts": {key: 0 for key in self._source_keys},
            "source_token_counts": {key: 0 for key in self._source_keys},
            "source_target_counts": {key: 0 for key in self._source_keys},
        }

    def _make_order(self, epoch: int) -> tuple[list[int], str]:
        order = list(range(len(self._records)))
        random.Random(_epoch_seed(self.seed, epoch)).shuffle(order)
        order_hash = canonical_json_hash(
            {
                "seed": self.seed,
                "epoch": epoch,
                "rank": self.rank,
                "world_size": self.world_size,
                "order": order,
                "manifest_hash": self.manifest_hash,
            }
        )
        return order, order_hash

    def _rebuild_rank_order(self) -> None:
        rank_order = [
            index
            for position, index in enumerate(self._order)
            if position % self.world_size == self.rank
        ]
        if not rank_order:
            rank_order = list(self._order)
        self._rank_order = rank_order

    def _source_key(self, record: dict) -> str:
        return f"{record['source']}/{record['split']}"

    def _conversation_from_record(
        self,
        record: dict,
        dataset_file: DatasetFile,
        line_number: int,
    ) -> dict:
        if "conversation" in record:
            raw = record["conversation"]
        elif "messages" in record:
            raw = {"messages": record["messages"]}
        else:
            raise SFTLoaderError(
                f"record {dataset_file.relative_path}:{line_number} has no conversation"
            )
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError) as exc:
                raise SFTLoaderError(
                    f"record {dataset_file.relative_path}:{line_number} has a non-JSON conversation"
                ) from exc
        if isinstance(raw, list):
            raw = {"messages": raw}
        if not isinstance(raw, dict) or "messages" not in raw:
            raise SFTLoaderError(
                f"record {dataset_file.relative_path}:{line_number} must contain messages"
            )
        messages = raw["messages"]
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except (TypeError, ValueError) as exc:
                raise SFTLoaderError(
                    f"record {dataset_file.relative_path}:{line_number} has invalid messages"
                ) from exc
        if not isinstance(messages, list) or not messages:
            raise SFTLoaderError(
                f"record {dataset_file.relative_path}:{line_number} has no messages"
            )
        allowed_roles = {"system", "user", "assistant", "tool"}
        for message in messages:
            if not isinstance(message, dict):
                raise SFTLoaderError(
                    f"record {dataset_file.relative_path}:{line_number} has a malformed message"
                )
            role = message.get("role")
            if not isinstance(role, str) or role not in allowed_roles:
                raise SFTLoaderError(
                    f"record {dataset_file.relative_path}:{line_number} has an invalid role"
                )
            if "content" not in message and "tool_calls" not in message:
                raise SFTLoaderError(
                    f"record {dataset_file.relative_path}:{line_number} has no message content"
                )
        conversation = copy.deepcopy(raw)
        conversation["messages"] = copy.deepcopy(messages)
        return conversation

    def _render(self, conversation: dict, record: dict) -> dict:
        try:
            render = self.tokenizer.render_conversation
            try:
                parameters = inspect.signature(render).parameters
            except (TypeError, ValueError):
                parameters = {"max_tokens": None}
            if "max_tokens" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                rendered = render(
                    copy.deepcopy(conversation),
                    max_tokens=SFT_RENDER_TOKEN_LIMIT,
                )
            else:
                rendered = render(copy.deepcopy(conversation))
        except Exception as exc:
            raise SFTLoaderError(
                f"render_conversation failed for {record['file']}:{record['line']}; fallback is forbidden"
            ) from exc
        if not isinstance(rendered, (tuple, list)) or len(rendered) != 2:
            raise SFTLoaderError("render_conversation must return ids and mask")
        ids = _tensor_values(rendered[0], "rendered ids")
        mask = _tensor_values(rendered[1], "rendered mask", allow_bool=True)
        if len(ids) != len(mask):
            raise SFTLoaderError("rendered ids and mask must have equal length")
        if not ids:
            raise SFTLoaderError("render_conversation returned no tokens")
        if any(token < 0 for token in ids):
            raise SFTLoaderError("rendered token IDs must be non-negative")
        if any(value not in (0, 1) for value in mask):
            raise SFTLoaderError("rendered mask must contain only zero and one")
        original_length = len(ids)
        if original_length > SFT_RENDER_TOKEN_LIMIT:
            ids = ids[:SFT_RENDER_TOKEN_LIMIT]
            mask = mask[:SFT_RENDER_TOKEN_LIMIT]
        truncated = max(0, original_length - SFT_RENDER_TOKEN_LIMIT)
        padding = max(0, SFT_RENDER_TOKEN_LIMIT - original_length)
        if padding:
            ids.extend([self.pad_token_id] * padding)
            mask.extend([0] * padding)
        attention = [1] * (SFT_RENDER_TOKEN_LIMIT - padding)
        attention.extend([0] * padding)
        return {
            "ids": ids,
            "mask": mask,
            "attention": attention,
            "padding": padding,
            "truncated": truncated,
            "original_length": original_length,
            "record": record,
        }

    def _capture_position(self) -> dict:
        return {
            "epoch": self._epoch,
            "cursor": self._cursor,
            "order": list(self._order),
            "order_hash": self._order_hash,
            "rank_order": list(self._rank_order),
            "rng_state": self._rng.getstate(),
            "counters": copy.deepcopy(self._counters),
        }

    def _restore_position(self, position: dict) -> None:
        self._epoch = position["epoch"]
        self._cursor = position["cursor"]
        self._order = list(position["order"])
        self._order_hash = position["order_hash"]
        self._rank_order = list(position["rank_order"])
        self._rng.setstate(position["rng_state"])
        self._counters = copy.deepcopy(position["counters"])

    def _next_record(self) -> dict:
        if self._cursor >= len(self._rank_order):
            self._epoch += 1
            self._cursor = 0
            self._order, self._order_hash = self._make_order(self._epoch)
            self._rebuild_rank_order()
        index = self._rank_order[self._cursor]
        self._cursor += 1
        return self._records[index]

    def _file_states(self) -> dict:
        counts = {key: 0 for key in self._file_keys}
        for index in self._rank_order[: min(self._cursor, len(self._rank_order))]:
            key = self._records[index]["file"]
            counts[key] += 1
        return {
            key: {"epoch": self._epoch, "cursor": value}
            for key, value in counts.items()
        }

    def _make_state(self) -> dict:
        pending = None
        if self._last_batch is not None:
            inputs = self._last_batch.inputs.detach().cpu().tolist()
            targets = self._last_batch.targets.detach().cpu().tolist()
            loss_mask = self._last_batch.loss_mask.detach().cpu().tolist()
            attention = self._last_batch.attention_mask.detach().cpu().tolist()
            pending = {
                "inputs": inputs,
                "targets": targets,
                "x": copy.deepcopy(inputs),
                "y": copy.deepcopy(targets),
                "loss_mask": loss_mask,
                "mask": copy.deepcopy(loss_mask),
                "attention_mask": attention,
                "shape": [self.batch_size, SFT_CONTEXT_TOKENS],
                "source_names": list(self._last_batch.source_names),
                "source_indices": [
                    item.get("line", 0) - 1 if isinstance(item, dict) else 0
                    for item in self._last_batch.metadata
                ],
            }
        state = {
            "schema_version": SFT_LOADER_STATE_SCHEMA_VERSION,
            "context_length": SFT_CONTEXT_TOKENS,
            "sequence_length": SFT_CONTEXT_TOKENS,
            "batch_size": self.batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "manifest_hash": self.manifest_hash,
            "order_hash": self._order_hash,
            "cursor": self._cursor,
            "epoch": self._epoch,
            "rng_state": _random_state_json(self._rng.getstate()),
            "rng": {"python": _random_state_json(self._rng.getstate())},
            "counters": copy.deepcopy(self._counters),
            "file_states": self._file_states(),
            "pending_batch": pending,
            "pending_batch_consumed": pending is not None,
            "pending_batch_replay_on_resume": pending is not None,
            "config": {
                "context_length": SFT_CONTEXT_TOKENS,
                "batch_size": self.batch_size,
                "rank": self.rank,
                "world_size": self.world_size,
                "seed": self.seed,
                "manifest_hash": self.manifest_hash,
            },
        }
        return _json_copy(state, "loader state")

    def state_dict(self) -> dict:
        return self._make_state()

    def get_state(self) -> dict:
        return self.state_dict()

    def _validate_state(self, state: Any) -> dict:
        if not isinstance(state, dict):
            raise SFTLoaderError("loader state must be a mapping")
        try:
            state = _json_copy(state, "loader state")
        except SFTLoaderError:
            raise
        required = {
            "schema_version",
            "context_length",
            "sequence_length",
            "batch_size",
            "rank",
            "world_size",
            "seed",
            "manifest_hash",
            "order_hash",
            "cursor",
            "epoch",
            "rng_state",
            "counters",
            "file_states",
            "pending_batch",
            "pending_batch_consumed",
            "pending_batch_replay_on_resume",
            "rng",
            "config",
        }
        missing = sorted(required - set(state))
        if missing:
            raise SFTLoaderError(f"loader state is missing fields: {missing}")
        if state["schema_version"] != SFT_LOADER_STATE_SCHEMA_VERSION:
            raise SFTLoaderError("loader state schema version is incompatible")
        if (
            state["context_length"] != SFT_CONTEXT_TOKENS
            or state["sequence_length"] != SFT_CONTEXT_TOKENS
        ):
            raise SFTLoaderError("loader state is not fixed at 2048 tokens")
        expected_config = {
            "batch_size": self.batch_size,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "manifest_hash": self.manifest_hash,
        }
        for field, expected in expected_config.items():
            if state[field] != expected:
                raise SFTLoaderError(f"loader state {field} is incompatible")
        config = state["config"]
        if not isinstance(config, dict):
            raise SFTLoaderError("loader state config must be a mapping")
        for field, expected in {
            "context_length": SFT_CONTEXT_TOKENS,
            **expected_config,
        }.items():
            if config.get(field) != expected:
                raise SFTLoaderError(f"loader state config {field} is incompatible")
        if not isinstance(state["file_states"], dict):
            raise SFTLoaderError("loader state file cursors are invalid")
        for key, value in state["file_states"].items():
            if key not in self._file_keys or not isinstance(value, dict):
                raise SFTLoaderError("loader state file cursors are invalid")
            if (
                value.get("epoch") != state["epoch"]
                or not _is_int(value.get("cursor"))
                or value["cursor"] < 0
            ):
                raise SFTLoaderError("loader state file cursors are invalid")
        if (
            not isinstance(state["rng"], dict)
            or state["rng"].get("python") != state["rng_state"]
        ):
            raise SFTLoaderError("loader state RNG alias is invalid")
        if not isinstance(state["pending_batch_consumed"], bool) or not isinstance(
            state["pending_batch_replay_on_resume"], bool
        ):
            raise SFTLoaderError("loader state pending flags are invalid")
        if not _is_int(state["epoch"]) or state["epoch"] <= 0:
            raise SFTLoaderError("loader state epoch is invalid")
        if not _is_int(state["cursor"]) or state["cursor"] < 0:
            raise SFTLoaderError("loader state cursor is invalid")
        if (
            sum(value["cursor"] for value in state["file_states"].values())
            != state["cursor"]
        ):
            raise SFTLoaderError("loader state file cursors are inconsistent")
        order, order_hash = self._make_order(state["epoch"])
        if state["order_hash"] != order_hash:
            raise SFTLoaderError("loader state order hash is incompatible")
        rank_order = [
            index
            for position, index in enumerate(order)
            if position % self.world_size == self.rank
        ]
        if not rank_order:
            rank_order = list(order)
        if state["cursor"] > len(rank_order):
            raise SFTLoaderError("loader state cursor is out of range")
        _random_state_from_json(state["rng_state"], "rng_state")
        counters = state["counters"]
        if not isinstance(counters, dict):
            raise SFTLoaderError("loader state counters must be a mapping")
        counter_names = (
            "examples_seen",
            "batches_seen",
            "total_input_tokens",
            "total_target_tokens",
            "total_attention_tokens",
            "total_padding_tokens",
            "total_truncated_tokens",
        )
        for name in counter_names:
            if not _is_int(counters.get(name)) or counters[name] < 0:
                raise SFTLoaderError(f"loader state counter {name} is invalid")
        if counters["examples_seen"] != counters["batches_seen"] * self.batch_size:
            raise SFTLoaderError("loader state example counters are inconsistent")
        if (
            counters["total_input_tokens"]
            != counters["batches_seen"] * self.batch_size * SFT_CONTEXT_TOKENS
        ):
            raise SFTLoaderError("loader state input counters are inconsistent")
        if counters["total_target_tokens"] > counters["total_input_tokens"]:
            raise SFTLoaderError("loader state target counters are inconsistent")
        for name in (
            "source_document_counts",
            "source_token_counts",
            "source_target_counts",
        ):
            values = counters.get(name)
            if not isinstance(values, dict) or set(values) != set(self._source_keys):
                raise SFTLoaderError(f"loader state counter {name} is invalid")
            for key, value in values.items():
                if key not in self._source_keys or not _is_int(value) or value < 0:
                    raise SFTLoaderError(f"loader state counter {name} is invalid")
        pending = state["pending_batch"]
        if pending is None and (
            state["pending_batch_consumed"] or state["pending_batch_replay_on_resume"]
        ):
            raise SFTLoaderError("loader state pending flags are inconsistent")
        if pending is not None and (
            not state["pending_batch_consumed"]
            or not state["pending_batch_replay_on_resume"]
        ):
            raise SFTLoaderError("loader state pending flags are inconsistent")
        if pending is None:
            if counters["batches_seen"] != 0:
                raise SFTLoaderError("loader state lost its pending batch")
            return {
                "state": state,
                "order": order,
                "rank_order": rank_order,
                "pending": None,
            }
        if not isinstance(pending, dict):
            raise SFTLoaderError("loader state pending_batch must be a mapping")
        for field in ("inputs", "targets", "loss_mask", "attention_mask"):
            if field not in pending:
                raise SFTLoaderError(f"pending batch is missing {field}")
        inputs = _tensor_matrix(
            pending["inputs"], self.batch_size, SFT_CONTEXT_TOKENS, "pending inputs"
        )
        targets = _tensor_matrix(
            pending["targets"], self.batch_size, SFT_CONTEXT_TOKENS, "pending targets"
        )
        loss_mask = _tensor_matrix(
            pending["loss_mask"],
            self.batch_size,
            SFT_CONTEXT_TOKENS,
            "pending loss mask",
        )
        attention = _tensor_matrix(
            pending["attention_mask"],
            self.batch_size,
            SFT_CONTEXT_TOKENS,
            "pending attention mask",
        )
        if any(value < 0 for row in inputs for value in row):
            raise SFTLoaderError("pending inputs contain invalid token IDs")
        if any(value < -1 for row in targets for value in row):
            raise SFTLoaderError("pending targets contain invalid token IDs")
        if any(value not in (0, 1) for row in loss_mask for value in row):
            raise SFTLoaderError("pending loss mask is invalid")
        if any(value not in (0, 1) for row in attention for value in row):
            raise SFTLoaderError("pending attention mask is invalid")
        if "source_names" in pending and (
            not isinstance(pending["source_names"], list)
            or len(pending["source_names"]) != self.batch_size
        ):
            raise SFTLoaderError("pending source names are invalid")
        source_indices = pending.get("source_indices")
        if source_indices is not None and (
            not isinstance(source_indices, list)
            or len(source_indices) != self.batch_size
            or any(not _is_int(index) or index < 0 for index in source_indices)
        ):
            raise SFTLoaderError("pending source indices are invalid")
        return {
            "state": state,
            "order": order,
            "rank_order": rank_order,
            "pending": {
                "inputs": inputs,
                "targets": targets,
                "loss_mask": loss_mask,
                "attention_mask": attention,
                "source_names": tuple(
                    pending.get("source_names", [""] * self.batch_size)
                ),
                "source_indices": tuple(
                    source_indices
                    if source_indices is not None
                    else [0] * self.batch_size
                ),
            },
        }

    def load_state_dict(self, state: dict) -> "SFTConversationBatchLoader":
        parsed = self._validate_state(state)
        saved = parsed["state"]
        self._epoch = saved["epoch"]
        self._cursor = saved["cursor"]
        self._order = parsed["order"]
        self._rank_order = parsed["rank_order"]
        self._order_hash = saved["order_hash"]
        self._rng.setstate(_random_state_from_json(saved["rng_state"], "rng_state"))
        self._counters = copy.deepcopy(saved["counters"])
        pending = parsed["pending"]
        if pending is None:
            self._last_batch = None
            self._replay_batch = None
            return self
        self._last_batch = self._batch_from_state(pending, saved)
        self._replay_batch = self._last_batch
        return self

    def restore_state(self, state: dict) -> "SFTConversationBatchLoader":
        return self.load_state_dict(state)

    def validate_state(self, state: dict) -> dict:
        self._validate_state(state)
        return _json_copy(state, "loader state")

    validate_loader_state = validate_state

    def _batch_from_state(self, pending: dict, state: dict) -> ConversationBatch:
        inputs = torch.tensor(pending["inputs"], dtype=torch.long, device=self.device)
        targets = torch.tensor(pending["targets"], dtype=torch.long, device=self.device)
        loss_mask = torch.tensor(
            pending["loss_mask"], dtype=torch.int8, device=self.device
        )
        attention = torch.tensor(
            pending["attention_mask"], dtype=torch.int8, device=self.device
        )
        return ConversationBatch(
            inputs=inputs,
            targets=targets,
            loss_mask=loss_mask,
            attention_mask=attention,
            state=copy.deepcopy(state),
            source_names=tuple(pending["source_names"]),
            metadata=tuple(
                {"line": int(index) + 1}
                for index in pending.get("source_indices", [0] * self.batch_size)
            ),
        )

    def __iter__(self) -> "SFTConversationBatchLoader":
        return self

    def __next__(self) -> ConversationBatch:
        if self._replay_batch is not None:
            batch = self._replay_batch
            self._replay_batch = None
            self._last_batch = batch
            batch.state = self._make_state()
            return batch
        rendered = []
        position = self._capture_position()
        try:
            for _ in range(self.batch_size):
                record = self._next_record()
                item = self._render(record["conversation"], record)
                rendered.append(item)
                self._rng.getrandbits(64)
        except Exception:
            self._restore_position(position)
            raise
        inputs = []
        targets = []
        loss_masks = []
        attentions = []
        for item in rendered:
            token_ids = item["ids"]
            item_mask = item["mask"]
            inputs.append(token_ids[:-1])
            targets.append(token_ids[1:])
            loss_masks.append(item_mask[1:])
            attentions.append(item["attention"][:-1])
            source_key = self._source_key(item["record"])
            self._counters["examples_seen"] += 1
            self._counters["source_document_counts"][source_key] += 1
            self._counters["source_token_counts"][source_key] += item["original_length"]
            self._counters["source_target_counts"][source_key] += sum(item_mask[1:])
            self._counters["total_input_tokens"] += SFT_CONTEXT_TOKENS
            self._counters["total_target_tokens"] += sum(item_mask[1:])
            self._counters["total_attention_tokens"] += sum(item["attention"])
            self._counters["total_padding_tokens"] += item["padding"]
            self._counters["total_truncated_tokens"] += item["truncated"]
        input_tensor = torch.tensor(inputs, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=self.device)
        loss_tensor = torch.tensor(loss_masks, dtype=torch.int8, device=self.device)
        attention_tensor = torch.tensor(
            attentions, dtype=torch.int8, device=self.device
        )
        target_tensor = torch.where(
            loss_tensor.bool(), target_tensor, torch.full_like(target_tensor, -1)
        )
        self._counters["batches_seen"] += 1
        batch = ConversationBatch(
            inputs=input_tensor,
            targets=target_tensor,
            loss_mask=loss_tensor,
            attention_mask=attention_tensor,
            state={},
            source_names=tuple(self._source_key(item["record"]) for item in rendered),
            metadata=tuple(
                {"file": item["record"]["file"], "line": item["record"]["line"]}
                for item in rendered
            ),
        )
        self._last_batch = batch
        batch.state = self._make_state()
        return batch

    def next_batch(self) -> ConversationBatch:
        return self.__next__()


SFTRankAwareBatchLoader = SFTConversationBatchLoader
DeterministicConversationBatchLoader = SFTConversationBatchLoader
SFTConversationLoader = SFTConversationBatchLoader
SFTBatchLoader = SFTConversationBatchLoader


def build_sft_batch_loader(*args, **kwargs) -> SFTConversationBatchLoader:
    return SFTConversationBatchLoader(*args, **kwargs)


def make_sft_batch_loader(*args, **kwargs) -> SFTConversationBatchLoader:
    return SFTConversationBatchLoader(*args, **kwargs)


def _json_safe(value: Any, field_name: str = "value") -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [_json_safe(item, field_name) for item in value]
    if isinstance(value, list):
        return [_json_safe(item, field_name) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item, field_name) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise SFTCheckpointError(f"{field_name} contains a non-finite value")
        return value
    raise SFTCheckpointError(f"{field_name} is not JSON-compatible")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _atomic_torch_save(path: Path, value: Any) -> None:
    if torch is None:
        raise SFTCheckpointError("torch is required for SFT checkpoints")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        os.close(fd)
        torch.save(value, temporary)
        with open(temporary, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _read_json_file(path: Path, field_name: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(
                handle,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SFTCheckpointError(f"{field_name} is not valid JSON: {path}") from exc


def capture_rng_state(*, include_cuda: bool = False) -> dict:
    state = {
        "schema_version": 1,
        "python": _random_state_json(random.getstate()),
    }
    try:
        import numpy as np

        numpy_state = np.random.get_state(legacy=True)
        state["numpy"] = {
            "bit_generator": str(numpy_state[0]),
            "state": _json_safe(numpy_state[1], "numpy state"),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        }
    except ImportError:
        state["numpy"] = None
    if torch is not None:
        state["torch_cpu"] = torch.get_rng_state().tolist()
        if include_cuda and torch.cuda.is_available():
            state["torch_cuda"] = [
                item.tolist() for item in torch.cuda.get_rng_state_all()
            ]
        else:
            state["torch_cuda"] = None
    else:
        state["torch_cpu"] = None
        state["torch_cuda"] = None
    return state


def _restore_torch_state(value: Any, device: Any = "cpu") -> None:
    if torch is None or value is None:
        return
    try:
        tensor = torch.tensor(value, dtype=torch.uint8, device=device)
        torch.set_rng_state(tensor)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise SFTCheckpointError("invalid PyTorch RNG state") from exc


def restore_rng_state(state: Any, *, device: Any = "cpu") -> None:
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise SFTCheckpointError("RNG state schema version is incompatible")
    python_state = state.get("python")
    random.setstate(_random_state_from_json(python_state, "python RNG state"))
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np

            if not isinstance(numpy_state, dict):
                raise ValueError
            np.random.set_state(
                (
                    numpy_state["bit_generator"],
                    tuple(numpy_state["state"]),
                    int(numpy_state["position"]),
                    int(numpy_state["has_gauss"]),
                    float(numpy_state["cached_gaussian"]),
                )
            )
        except (ImportError, KeyError, TypeError, ValueError) as exc:
            raise SFTCheckpointError("invalid NumPy RNG state") from exc
    _restore_torch_state(state.get("torch_cpu"), "cpu")
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None and torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [torch.tensor(item, dtype=torch.uint8) for item in cuda_states]
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise SFTCheckpointError("invalid CUDA RNG state") from exc


def _checkpoint_name(step: int) -> str:
    if not _is_int(step) or step < 0:
        raise SFTCheckpointError("checkpoint step must be a non-negative integer")
    return f"step_{step:06d}"


def _safe_checkpoint_id(value: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise SFTCheckpointError("checkpoint ID is invalid")
    path = Path(value)
    if path.is_absolute() or path.drive or len(path.parts) != 1 or ".." in path.parts:
        raise SFTCheckpointError("checkpoint ID must be a single relative name")
    if (
        value.casefold() in GLOBAL_CHECKPOINT_NAMESPACES
        or value.casefold() == ".staging"
    ):
        raise SFTCheckpointError("checkpoint ID is a global namespace")
    return value


def _descriptor(path: Path, relative: str) -> dict:
    return {"path": relative, "sha256": sha256_file(path)}


def _read_envelope(directory: Path) -> dict:
    envelope_path = directory / ENVELOPE_FILE
    if not envelope_path.is_file() or envelope_path.is_symlink():
        raise SFTCheckpointError("checkpoint envelope is missing or unsafe")
    envelope = _read_json_file(envelope_path, "checkpoint envelope")
    if not isinstance(envelope, dict):
        raise SFTCheckpointError("checkpoint envelope must be an object")
    return envelope


def _validate_descriptor(
    directory: Path,
    descriptor: Any,
    field_name: str,
) -> Path:
    if not isinstance(descriptor, dict):
        raise SFTCheckpointError(f"{field_name} descriptor is invalid")
    relative = descriptor.get("path")
    if not isinstance(relative, str) or not relative:
        raise SFTCheckpointError(f"{field_name} path is invalid")
    path = Path(relative)
    if path.is_absolute() or path.drive or ".." in path.parts or len(path.parts) != 1:
        raise SFTCheckpointError(f"{field_name} path is not run-local")
    try:
        resolved = resolve_run_local_path(path, directory, require_exists=True)
    except SFTPathError as exc:
        raise SFTCheckpointError(f"{field_name} path is unsafe") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise SFTCheckpointError(f"{field_name} is missing or unsafe")
    digest = descriptor.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SFTCheckpointError(f"{field_name} hash is invalid")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise SFTCheckpointError(f"{field_name} hash is invalid") from exc
    if sha256_file(resolved).casefold() != digest.casefold():
        raise SFTCheckpointError(f"{field_name} hash is stale")
    return resolved


def _normalize_rank_descriptors(
    value: Any, field_name: str, world_size: int
) -> dict[int, Any]:
    if not isinstance(value, dict) or set(value) != {
        str(rank) for rank in range(world_size)
    }:
        raise SFTCheckpointError(f"{field_name} must contain every rank")
    return {int(rank): value[str(rank)] for rank in range(world_size)}


def validate_sft_checkpoint_envelope(
    directory: str | os.PathLike,
    *,
    world_size: int | None = None,
    expected_step: int | None = None,
    require_complete: bool = True,
    allow_staging: bool = False,
) -> dict:
    directory_path = Path(directory)
    if not allow_staging and directory_path.parent.name == ".staging":
        raise SFTCheckpointError("staging directories are not loadable checkpoints")
    if not directory_path.is_dir() or directory_path.is_symlink():
        raise SFTCheckpointError("checkpoint directory is missing or unsafe")
    envelope = _read_envelope(directory_path)
    required_fields = {
        "sft_checkpoint_schema_version",
        "schema_version",
        "checkpoint_id",
        "transaction_id",
        "global_step",
        "completion_status",
        "sequence_length",
        "context_length",
        "world_size",
        "rank",
        "ranks",
        "model",
        "optimizer",
        "loader",
        "rng",
        "metadata",
    }
    missing = sorted(required_fields - set(envelope))
    if missing:
        raise SFTCheckpointError(f"checkpoint envelope is missing fields: {missing}")
    if envelope.get("sft_checkpoint_schema_version") != SFT_CHECKPOINT_SCHEMA_VERSION:
        raise SFTCheckpointError("checkpoint schema version is incompatible")
    if (
        envelope.get("schema_version", SFT_CHECKPOINT_SCHEMA_VERSION)
        != SFT_CHECKPOINT_SCHEMA_VERSION
    ):
        raise SFTCheckpointError("checkpoint schema version is incompatible")
    if envelope.get("completion_status") != "complete":
        raise SFTCheckpointError("checkpoint is not complete")
    if (
        not isinstance(envelope.get("transaction_id"), str)
        or not envelope["transaction_id"]
    ):
        raise SFTCheckpointError("checkpoint transaction ID is invalid")
    if not _is_int(envelope.get("global_step")) or envelope["global_step"] < 0:
        raise SFTCheckpointError("checkpoint global step is invalid")
    if expected_step is not None and envelope["global_step"] != expected_step:
        raise SFTCheckpointError("checkpoint step is incompatible")
    checkpoint_id = envelope.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or checkpoint_id != directory_path.name:
        raise SFTCheckpointError("checkpoint ID is incompatible")
    if (
        envelope.get("sequence_length") != SFT_CONTEXT_TOKENS
        or envelope.get("context_length") != SFT_CONTEXT_TOKENS
    ):
        raise SFTCheckpointError("checkpoint is not fixed at 2048 tokens")
    if not _is_int(envelope.get("world_size")) or envelope["world_size"] <= 0:
        raise SFTCheckpointError("checkpoint world size is invalid")
    if envelope.get("rank") != 0:
        raise SFTCheckpointError("checkpoint envelope must be rank zero owned")
    actual_world_size = envelope["world_size"]
    if world_size is not None and actual_world_size != world_size:
        raise SFTCheckpointError("checkpoint world size is incompatible")
    ranks = envelope.get("ranks")
    if ranks != list(range(actual_world_size)):
        raise SFTCheckpointError("checkpoint rank set is invalid")
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        raise SFTCheckpointError("checkpoint metadata is invalid")
    for field in ("sequence_length", "context_length", "max_seq_len"):
        if field in metadata:
            _fixed_context(metadata[field])
    if "context_range" in metadata:
        validate_fixed_context(context_range=metadata["context_range"])
    if "buckets" in metadata:
        validate_fixed_context(buckets=metadata["buckets"])
    for hash_field in ("dataset_manifest_hash", "tokenizer_hash"):
        value = envelope.get(hash_field, metadata.get(hash_field))
        if value is not None and (not isinstance(value, str) or not value):
            raise SFTCheckpointError(f"checkpoint {hash_field} is invalid")
    _validate_descriptor(directory_path, envelope.get("model"), "model")
    optimizer = _normalize_rank_descriptors(
        envelope.get("optimizer"), "optimizer", actual_world_size
    )
    loader = _normalize_rank_descriptors(
        envelope.get("loader"), "loader", actual_world_size
    )
    rng = _normalize_rank_descriptors(envelope.get("rng"), "rng", actual_world_size)
    for rank in range(actual_world_size):
        _validate_descriptor(directory_path, optimizer[rank], f"optimizer rank {rank}")
        _validate_descriptor(directory_path, loader[rank], f"loader rank {rank}")
        _validate_descriptor(directory_path, rng[rank], f"rng rank {rank}")
    completion_path = directory_path / COMPLETION_MARKER
    if require_complete:
        if not completion_path.is_file() or completion_path.is_symlink():
            raise SFTCheckpointError("checkpoint completion marker is missing")
        completion = _read_json_file(completion_path, "checkpoint completion marker")
        if not isinstance(completion, dict) or completion.get("status") != "complete":
            raise SFTCheckpointError("checkpoint completion marker is invalid")
        if completion.get("checkpoint_id") != checkpoint_id:
            raise SFTCheckpointError("completion marker checkpoint ID is invalid")
        if completion.get("envelope_sha256") != sha256_file(
            directory_path / ENVELOPE_FILE
        ):
            raise SFTCheckpointError("completion marker envelope hash is stale")
    return envelope


def _pointer_path(checkpoint_root: Path) -> Path:
    return checkpoint_root / LAST_KNOWN_GOOD_POINTER


def _write_pointer(checkpoint_root: Path, directory: Path) -> dict:
    relative = directory.relative_to(checkpoint_root).as_posix()
    pointer = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_id": directory.name,
        "relative_path": relative,
        "checkpoint_path": relative,
        "envelope_sha256": sha256_file(directory / ENVELOPE_FILE),
    }
    write_json_atomic(_pointer_path(checkpoint_root), pointer)
    return pointer


def _read_pointer(checkpoint_root: Path) -> tuple[Path, dict]:
    pointer_path = _pointer_path(checkpoint_root)
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise SFTCheckpointError("last-known-good pointer is missing")
    pointer = _read_json_file(pointer_path, "last-known-good pointer")
    if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
        raise SFTCheckpointError("last-known-good pointer schema is invalid")
    relative = pointer.get("relative_path", pointer.get("checkpoint_path"))
    if not isinstance(relative, str):
        raise SFTCheckpointError("last-known-good pointer path is invalid")
    try:
        directory = resolve_run_local_path(
            relative, checkpoint_root, require_exists=True
        )
    except SFTPathError as exc:
        raise SFTCheckpointError("last-known-good pointer path is unsafe") from exc
    if not directory.is_dir() or directory.is_symlink():
        raise SFTCheckpointError("last-known-good checkpoint is missing")
    envelope = validate_sft_checkpoint_envelope(
        directory,
        world_size=pointer.get("world_size"),
    )
    pointer_hash = pointer.get("envelope_sha256")
    if (
        not isinstance(pointer_hash, str)
        or pointer_hash.casefold() != sha256_file(directory / ENVELOPE_FILE).casefold()
    ):
        raise SFTCheckpointError("last-known-good pointer is stale")
    if pointer.get("checkpoint_id") != directory.name:
        raise SFTCheckpointError("last-known-good pointer ID is stale")
    if envelope.get("checkpoint_id") != directory.name:
        raise SFTCheckpointError("last-known-good envelope is stale")
    return directory, pointer


class SFTCheckpointTransaction:
    def __init__(
        self,
        checkpoint_root: str | os.PathLike,
        run_root: str | os.PathLike | None = None,
        step: int = 0,
        *,
        rank: int = 0,
        world_size: int = 1,
        checkpoint_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dataset_manifest_hash: str | None = None,
        tokenizer_hash: str | None = None,
        effective_tokens: int | None = None,
        stage_index: int | None = None,
        stage_name: str | None = None,
        stage_local_step: int | None = None,
        seed: int | None = None,
        device_type: str | None = None,
        transaction_id: str | None = None,
        resume: bool = False,
    ):
        if run_root is None:
            run_root = Path(checkpoint_root).parent
        self.checkpoint_root = resolve_run_local_path(checkpoint_root, run_root)
        self.run_root = Path(os.path.realpath(_absolute(run_root)))
        if not _is_int(rank) or rank < 0:
            raise SFTCheckpointError("rank must be a non-negative integer")
        if not _is_int(world_size) or world_size <= 0 or rank >= world_size:
            raise SFTCheckpointError("world_size and rank are inconsistent")
        if not _is_int(step) or step < 0:
            raise SFTCheckpointError("step must be a non-negative integer")
        self.step = int(step)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.checkpoint_id = _safe_checkpoint_id(
            checkpoint_id if checkpoint_id is not None else _checkpoint_name(step)
        )
        self.transaction_id = (
            transaction_id or f"{self.checkpoint_id}-{uuid.uuid4().hex[:12]}"
        )
        if not isinstance(self.transaction_id, str) or not self.transaction_id:
            raise SFTCheckpointError("transaction ID is invalid")
        transaction_path = Path(self.transaction_id)
        if (
            transaction_path.is_absolute()
            or transaction_path.drive
            or len(transaction_path.parts) != 1
            or ".." in transaction_path.parts
        ):
            raise SFTCheckpointError("transaction ID is invalid")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise SFTCheckpointError("checkpoint metadata must be a mapping")
        self.metadata = _json_safe(dict(metadata or {}), "checkpoint metadata")
        if not isinstance(self.metadata, dict):
            raise SFTCheckpointError("checkpoint metadata must be an object")
        if "sequence_length" in self.metadata:
            _fixed_context(self.metadata["sequence_length"])
        if "context_length" in self.metadata:
            _fixed_context(self.metadata["context_length"])
        if "max_seq_len" in self.metadata:
            _fixed_context(self.metadata["max_seq_len"])
        if "context_range" in self.metadata:
            validate_fixed_context(context_range=self.metadata["context_range"])
        if "buckets" in self.metadata:
            validate_fixed_context(buckets=self.metadata["buckets"])
        if dataset_manifest_hash is not None:
            self.metadata.setdefault("dataset_manifest_hash", dataset_manifest_hash)
        if tokenizer_hash is not None:
            self.metadata.setdefault("tokenizer_hash", tokenizer_hash)
        if effective_tokens is not None:
            if not _is_int(effective_tokens) or effective_tokens <= 0:
                raise SFTCheckpointError("effective token budget is invalid")
            self.metadata.setdefault("effective_tokens", int(effective_tokens))
        if stage_index is not None:
            if not _is_int(stage_index) or stage_index < 0:
                raise SFTCheckpointError("stage index is invalid")
            self.metadata.setdefault("stage_index", int(stage_index))
        if stage_name is not None:
            if not isinstance(stage_name, str) or not stage_name:
                raise SFTCheckpointError("stage name is invalid")
            self.metadata.setdefault("stage_name", stage_name)
        if stage_local_step is not None:
            if not _is_int(stage_local_step) or stage_local_step < 0:
                raise SFTCheckpointError("stage-local step is invalid")
            self.metadata.setdefault("stage_local_step", int(stage_local_step))
        if seed is not None:
            if not _is_int(seed):
                raise SFTCheckpointError("seed is invalid")
            self.metadata.setdefault("seed", int(seed))
        if device_type is not None:
            if not isinstance(device_type, str) or not device_type:
                raise SFTCheckpointError("device type is invalid")
            self.metadata.setdefault("device_type", device_type)
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        staging_parent = resolve_run_local_path(
            ".staging", self.checkpoint_root, require_exists=False
        )
        staging_parent.mkdir(parents=True, exist_ok=True)
        self.staging_path = resolve_run_local_path(
            f".staging/{self.checkpoint_id}", self.checkpoint_root
        )
        if self.staging_path.exists() and not resume and self.rank != 0:
            pass
        elif self.staging_path.exists() and not resume:
            raise SFTTransactionError("checkpoint staging directory already exists")
        self.staging_path.mkdir(parents=True, exist_ok=True)
        self._model_written = False
        self._rank_written: set[int] = set()
        self._descriptors: dict[str, Any] = {
            "model": None,
            "optimizer": {},
            "loader": {},
            "rng": {},
        }
        self._committed = False
        self.envelope: dict | None = None

    @property
    def stage_path(self) -> Path:
        return self.staging_path

    @property
    def committed(self) -> bool:
        return self._committed

    @property
    def checkpoint_path(self) -> Path | None:
        if not self._committed:
            return None
        return self.checkpoint_root / self.checkpoint_id

    def __getitem__(self, key: str) -> Any:
        if key == "checkpoint_path":
            return self.checkpoint_path
        if key == "envelope":
            return copy.deepcopy(self.envelope)
        if self.envelope is not None and key in self.envelope:
            return copy.deepcopy(self.envelope[key])
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        if self.envelope is None:
            return {
                "checkpoint_id": self.checkpoint_id,
                "staging_path": str(self.staging_path),
                "committed": False,
            }
        return copy.deepcopy(self.envelope)

    def _check_rank(self, rank: int) -> int:
        if not _is_int(rank) or rank < 0 or rank >= self.world_size:
            raise SFTCheckpointError("sidecar rank is outside the checkpoint world")
        return int(rank)

    def write_model(self, model_state: Any) -> None:
        if model_state is None:
            raise SFTCheckpointError("model checkpoint is required")
        if self.rank != 0:
            raise SFTCheckpointError("only rank zero owns the model checkpoint")
        if self._model_written:
            raise SFTCheckpointError("model checkpoint has already been staged")
        path = self.staging_path / "model.pt"
        _atomic_torch_save(path, model_state)
        self._descriptors["model"] = _descriptor(path, "model.pt")
        self._model_written = True

    save_model = write_model
    stage_model = write_model

    def write_rank_sidecars(
        self,
        rank: int,
        optimizer_state: Any,
        loader_state: Mapping[str, Any],
        rng_state: Mapping[str, Any] | None = None,
    ) -> None:
        rank = self._check_rank(rank)
        if rank in self._rank_written:
            raise SFTCheckpointError(f"rank {rank} sidecars are already staged")
        if optimizer_state is None:
            raise SFTCheckpointError(f"optimizer sidecar for rank {rank} is required")
        if not isinstance(loader_state, Mapping):
            raise SFTCheckpointError(f"loader state rank {rank} must be a mapping")
        loader_json = _json_safe(dict(loader_state), f"loader state rank {rank}")
        if not isinstance(loader_json, dict):
            raise SFTCheckpointError(f"loader state rank {rank} is invalid")
        if rng_state is None:
            rng_state = capture_rng_state()
        if not isinstance(rng_state, Mapping):
            raise SFTCheckpointError(f"RNG state rank {rank} must be a mapping")
        rng_json = _json_safe(dict(rng_state), f"RNG state rank {rank}")
        if not isinstance(rng_json, dict):
            raise SFTCheckpointError(f"RNG state rank {rank} is invalid")
        optimizer_path = self.staging_path / f"optimizer_rank{rank}.pt"
        loader_path = self.staging_path / f"loader_rank{rank}.json"
        rng_path = self.staging_path / f"rng_rank{rank}.json"
        _atomic_torch_save(optimizer_path, optimizer_state)
        _atomic_json(loader_path, loader_json)
        _atomic_json(rng_path, rng_json)
        self._descriptors["optimizer"][rank] = _descriptor(
            optimizer_path, optimizer_path.name
        )
        self._descriptors["loader"][rank] = _descriptor(loader_path, loader_path.name)
        self._descriptors["rng"][rank] = _descriptor(rng_path, rng_path.name)
        self._rank_written.add(rank)

    write_rank_state = write_rank_sidecars
    save_rank_state = write_rank_sidecars
    stage_rank_sidecars = write_rank_sidecars

    def _refresh_staged_sidecars(self) -> None:
        for rank in range(self.world_size):
            optimizer_path = self.staging_path / f"optimizer_rank{rank}.pt"
            loader_path = self.staging_path / f"loader_rank{rank}.json"
            rng_path = self.staging_path / f"rng_rank{rank}.json"
            if not (
                optimizer_path.is_file()
                and loader_path.is_file()
                and rng_path.is_file()
            ):
                continue
            self._descriptors["optimizer"][rank] = _descriptor(
                optimizer_path, optimizer_path.name
            )
            self._descriptors["loader"][rank] = _descriptor(
                loader_path, loader_path.name
            )
            self._descriptors["rng"][rank] = _descriptor(rng_path, rng_path.name)
            self._rank_written.add(rank)

    def _build_envelope(self) -> dict:
        self._refresh_staged_sidecars()
        if not self._model_written:
            raise SFTTransactionError("model checkpoint was not staged")
        missing = set(range(self.world_size)) - self._rank_written
        if missing:
            raise SFTTransactionError(
                f"rank-local sidecars are missing: {sorted(missing)}"
            )
        effective_tokens = self.metadata.get("effective_tokens")
        policy_fields = (
            "device_batch_size",
            "world_size",
            "gradient_accumulation_steps",
        )
        if all(field in self.metadata for field in policy_fields):
            if self.metadata["world_size"] != self.world_size:
                raise SFTTransactionError(
                    "checkpoint world size metadata is incompatible"
                )
            calculated_budget = validate_effective_token_budget(
                self.metadata["device_batch_size"],
                self.metadata["world_size"],
                self.metadata["gradient_accumulation_steps"],
                sequence_length=SFT_CONTEXT_TOKENS,
                total_batch_size=self.metadata.get("total_batch_size"),
            )
            if effective_tokens is not None and effective_tokens != calculated_budget:
                raise SFTTransactionError(
                    "checkpoint effective token budget is incompatible"
                )
            effective_tokens = calculated_budget
        if effective_tokens is not None:
            if not _is_int(effective_tokens) or effective_tokens <= 0:
                raise SFTTransactionError(
                    "checkpoint effective token budget is invalid"
                )
        return {
            "sft_checkpoint_schema_version": SFT_CHECKPOINT_SCHEMA_VERSION,
            "schema_version": SFT_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "transaction_id": self.transaction_id,
            "run_id": self.metadata.get("run_id"),
            "global_step": self.step,
            "stage_index": self.metadata.get("stage_index"),
            "stage_name": self.metadata.get("stage_name"),
            "stage_local_step": self.metadata.get("stage_local_step"),
            "completion_status": "complete",
            "sequence_length": SFT_CONTEXT_TOKENS,
            "context_length": SFT_CONTEXT_TOKENS,
            "world_size": self.world_size,
            "rank": 0,
            "ranks": list(range(self.world_size)),
            "model": copy.deepcopy(self._descriptors["model"]),
            "optimizer": {
                str(rank): copy.deepcopy(self._descriptors["optimizer"][rank])
                for rank in range(self.world_size)
            },
            "loader": {
                str(rank): copy.deepcopy(self._descriptors["loader"][rank])
                for rank in range(self.world_size)
            },
            "rng": {
                str(rank): copy.deepcopy(self._descriptors["rng"][rank])
                for rank in range(self.world_size)
            },
            "rank_local": {
                "optimizer": {
                    str(rank): copy.deepcopy(self._descriptors["optimizer"][rank])
                    for rank in range(self.world_size)
                },
                "loader": {
                    str(rank): copy.deepcopy(self._descriptors["loader"][rank])
                    for rank in range(self.world_size)
                },
                "rng": {
                    str(rank): copy.deepcopy(self._descriptors["rng"][rank])
                    for rank in range(self.world_size)
                },
            },
            "dataset_manifest_hash": self.metadata.get("dataset_manifest_hash"),
            "tokenizer_hash": self.metadata.get("tokenizer_hash"),
            "effective_tokens": effective_tokens,
            "metadata": copy.deepcopy(self.metadata),
        }

    def commit(self) -> dict:
        if self._committed:
            raise SFTTransactionError("checkpoint transaction is already committed")
        if self.rank != 0:
            raise SFTTransactionError("only rank zero can commit a checkpoint")
        try:
            envelope = self._build_envelope()
            envelope_path = self.staging_path / ENVELOPE_FILE
            _atomic_json(envelope_path, envelope)
            completion = {
                "schema_version": SFT_CHECKPOINT_SCHEMA_VERSION,
                "status": "complete",
                "checkpoint_id": self.checkpoint_id,
                "envelope_sha256": sha256_file(envelope_path),
                "transaction_id": self.transaction_id,
            }
            _atomic_json(self.staging_path / COMPLETION_MARKER, completion)
            validate_sft_checkpoint_envelope(
                self.staging_path,
                world_size=self.world_size,
                expected_step=self.step,
                allow_staging=True,
            )
            final_path = resolve_run_local_path(
                self.checkpoint_id, self.checkpoint_root
            )
            if final_path.exists():
                raise SFTTransactionError("completed checkpoint already exists")
            os.replace(self.staging_path, final_path)
            self._committed = True
            self.envelope = envelope
            try:
                _write_pointer(self.checkpoint_root, final_path)
            except Exception as exc:
                self._record_failure("pointer update failed")
                raise SFTTransactionError(
                    "last-known-good pointer was not updated"
                ) from exc
            return copy.deepcopy(envelope)
        except Exception as exc:
            self._record_failure(str(exc))
            raise

    finish = commit
    complete = commit

    def _record_failure(self, reason: str) -> None:
        try:
            safe_reason = str(reason).replace("\x00", " ")
            path = resolve_run_local_path(
                f"failed_save_{self.transaction_id}.json", self.checkpoint_root
            )
            _atomic_json(
                path,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "checkpoint_id": self.checkpoint_id,
                    "transaction_id": self.transaction_id,
                    "reason": safe_reason,
                },
            )
        except Exception:
            return

    def rollback(self) -> None:
        if self._committed:
            raise SFTTransactionError("cannot roll back a committed checkpoint")
        if self.staging_path.exists():
            shutil.rmtree(self.staging_path)
        self._model_written = False
        self._rank_written.clear()

    abort = rollback
    cleanup = rollback

    def __enter__(self) -> "SFTCheckpointTransaction":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None and not self._committed:
            self.rollback()
        return False


CheckpointTransaction = SFTCheckpointTransaction


def begin_sft_checkpoint(*args, **kwargs) -> SFTCheckpointTransaction:
    return SFTCheckpointTransaction(*args, **kwargs)


begin_checkpoint_transaction = begin_sft_checkpoint


def commit_sft_checkpoint(transaction: SFTCheckpointTransaction) -> dict:
    if not isinstance(transaction, SFTCheckpointTransaction):
        raise SFTCheckpointError("transaction is invalid")
    return transaction.commit()


def _rollback_transaction(transaction: SFTCheckpointTransaction) -> None:
    if not isinstance(transaction, SFTCheckpointTransaction):
        raise SFTCheckpointError("transaction is invalid")
    transaction.rollback()


rollback_checkpoint_transaction = _rollback_transaction


def save_sft_checkpoint(
    checkpoint_root: str | os.PathLike,
    run_root: str | os.PathLike | None = None,
    step: int = 0,
    model_state: Any = None,
    optimizer_state: Any = None,
    loader_state: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
    *,
    rank: int = 0,
    world_size: int = 1,
    metadata: Mapping[str, Any] | None = None,
    rank_states: Mapping[int, Mapping[str, Any]] | None = None,
    checkpoint_id: str | None = None,
    dataset_manifest_hash: str | None = None,
    tokenizer_hash: str | None = None,
    effective_tokens: int | None = None,
    stage_index: int | None = None,
    stage_name: str | None = None,
    stage_local_step: int | None = None,
    seed: int | None = None,
    device_type: str | None = None,
) -> SFTCheckpointTransaction:
    transaction = SFTCheckpointTransaction(
        checkpoint_root,
        run_root=run_root,
        step=step,
        rank=rank,
        world_size=world_size,
        checkpoint_id=checkpoint_id,
        metadata=metadata,
        dataset_manifest_hash=dataset_manifest_hash,
        tokenizer_hash=tokenizer_hash,
        effective_tokens=effective_tokens,
        stage_index=stage_index,
        stage_name=stage_name,
        stage_local_step=stage_local_step,
        seed=seed,
        device_type=device_type,
    )
    if rank_states is not None:
        for sidecar_rank, sidecar in rank_states.items():
            if not isinstance(sidecar, Mapping):
                raise SFTCheckpointError("rank sidecar must be a mapping")
            transaction.write_rank_sidecars(
                int(sidecar_rank),
                sidecar.get("optimizer", sidecar.get("optimizer_state")),
                sidecar.get("loader", sidecar.get("loader_state", {})),
                sidecar.get("rng", sidecar.get("rng_state")),
            )
    else:
        if loader_state is None:
            loader_state = {}
        transaction.write_rank_sidecars(
            rank,
            optimizer_state,
            loader_state,
            rng_state,
        )
    if rank == 0 and world_size == 1:
        if model_state is None:
            transaction.rollback()
            raise SFTTransactionError("model state is required for rank zero")
        transaction.write_model(model_state)
        transaction.commit()
    elif (
        rank == 0
        and rank_states is not None
        and set(range(world_size)).issubset(transaction._rank_written)
    ):
        if model_state is None:
            transaction.rollback()
            raise SFTTransactionError("model state is required for rank zero")
        transaction.write_model(model_state)
        transaction.commit()
    return transaction


write_sft_checkpoint = save_sft_checkpoint
save_checkpoint_transaction = save_sft_checkpoint


@dataclass
class LoadedSFTCheckpoint:
    envelope: dict
    model_state: Any
    optimizer_state: Any
    loader_state: dict
    rng_state: dict
    checkpoint_path: Path
    rank: int

    @property
    def model(self):
        return self.model_state

    @property
    def optimizer(self):
        return self.optimizer_state

    @property
    def loader(self):
        return self.loader_state

    @property
    def rng(self):
        return self.rng_state

    def __getitem__(self, key: str) -> Any:
        values = {
            "envelope": self.envelope,
            "model": self.model_state,
            "model_state": self.model_state,
            "optimizer": self.optimizer_state,
            "optimizer_state": self.optimizer_state,
            "loader": self.loader_state,
            "loader_state": self.loader_state,
            "rng": self.rng_state,
            "rng_state": self.rng_state,
            "checkpoint_path": self.checkpoint_path,
            "rank": self.rank,
        }
        return values[key]


def _resolve_checkpoint_directory(
    checkpoint_root: str | os.PathLike,
    run_root: str | os.PathLike | None,
    step: int | None,
    checkpoint_id: str | None,
    use_last_known_good: bool,
) -> Path:
    if run_root is None:
        run_root = Path(checkpoint_root).parent
    root = resolve_run_local_path(checkpoint_root, run_root)
    if not root.exists() or not root.is_dir():
        raise SFTCheckpointError("checkpoint root is missing")
    if (root / ENVELOPE_FILE).is_file():
        return root
    if checkpoint_id is not None:
        directory = resolve_run_local_path(
            _safe_checkpoint_id(checkpoint_id), root, require_exists=True
        )
    elif step is not None:
        directory = resolve_run_local_path(
            _checkpoint_name(step), root, require_exists=True
        )
    elif use_last_known_good:
        directory, _ = _read_pointer(root)
    else:
        raise SFTCheckpointError(
            "checkpoint step or last-known-good pointer is required"
        )
    if not directory.is_dir() or directory.is_symlink():
        raise SFTCheckpointError("checkpoint directory is missing or unsafe")
    return directory


def load_sft_checkpoint(
    checkpoint_root: str | os.PathLike,
    run_root: str | os.PathLike | None = None,
    *,
    step: int | None = None,
    checkpoint_id: str | None = None,
    rank: int = 0,
    device: Any = "cpu",
    use_last_known_good: bool = True,
) -> LoadedSFTCheckpoint:
    directory = _resolve_checkpoint_directory(
        checkpoint_root, run_root, step, checkpoint_id, use_last_known_good
    )
    envelope = validate_sft_checkpoint_envelope(directory)
    if not _is_int(rank) or rank < 0 or rank >= envelope["world_size"]:
        raise SFTCheckpointError("load rank is outside the checkpoint world")
    if torch is None:
        raise SFTCheckpointError("torch is required to load SFT checkpoints")
    model_path = _validate_descriptor(directory, envelope["model"], "model")
    try:
        try:
            model_state = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            model_state = torch.load(model_path, map_location=device)
    except Exception as exc:
        raise SFTCheckpointError("unable to load model checkpoint") from exc
    rank_key = str(rank)
    optimizer_descriptor = envelope["optimizer"][rank_key]
    loader_descriptor = envelope["loader"][rank_key]
    rng_descriptor = envelope["rng"][rank_key]
    optimizer_path = _validate_descriptor(directory, optimizer_descriptor, "optimizer")
    loader_path = _validate_descriptor(directory, loader_descriptor, "loader")
    rng_path = _validate_descriptor(directory, rng_descriptor, "rng")
    try:
        try:
            optimizer_state = torch.load(
                optimizer_path, map_location=device, weights_only=True
            )
        except TypeError:
            optimizer_state = torch.load(optimizer_path, map_location=device)
    except Exception as exc:
        raise SFTCheckpointError("unable to load optimizer checkpoint") from exc
    loader_state = _read_json_file(loader_path, "loader sidecar")
    rng_state = _read_json_file(rng_path, "RNG sidecar")
    if not isinstance(loader_state, dict) or not isinstance(rng_state, dict):
        raise SFTCheckpointError("rank-local sidecars must be JSON objects")
    return LoadedSFTCheckpoint(
        envelope=envelope,
        model_state=model_state,
        optimizer_state=optimizer_state,
        loader_state=loader_state,
        rng_state=rng_state,
        checkpoint_path=directory,
        rank=rank,
    )


def load_checkpoint_envelope(*args, **kwargs) -> dict:
    directory = _resolve_checkpoint_directory(
        args[0],
        args[1] if len(args) > 1 else kwargs.get("run_root"),
        kwargs.get("step"),
        kwargs.get("checkpoint_id"),
        kwargs.get("use_last_known_good", True),
    )
    return validate_sft_checkpoint_envelope(
        directory, world_size=kwargs.get("world_size")
    )


load_sft_checkpoint_envelope = load_checkpoint_envelope


def _rollback_checkpoint_root(
    checkpoint_root: str | os.PathLike,
    run_root: str | os.PathLike | None = None,
    *,
    checkpoint_id: str | None = None,
) -> Path:
    if run_root is None:
        run_root = Path(checkpoint_root).parent
    root = resolve_run_local_path(checkpoint_root, run_root)
    if checkpoint_id is None:
        directory, _ = _read_pointer(root)
    else:
        directory = resolve_run_local_path(
            _safe_checkpoint_id(checkpoint_id), root, require_exists=True
        )
        validate_sft_checkpoint_envelope(directory)
    _write_pointer(root, directory)
    return directory


def rollback_sft_checkpoint(
    checkpoint_root: str | os.PathLike | SFTCheckpointTransaction,
    run_root: str | os.PathLike | None = None,
    *,
    checkpoint_id: str | None = None,
) -> Path | None:
    if isinstance(checkpoint_root, SFTCheckpointTransaction):
        _rollback_transaction(checkpoint_root)
        return None
    return _rollback_checkpoint_root(
        checkpoint_root,
        run_root,
        checkpoint_id=checkpoint_id,
    )


def last_known_good_path(checkpoint_root: str | os.PathLike) -> Path:
    return Path(checkpoint_root) / LAST_KNOWN_GOOD_POINTER


__all__ = [
    "COMPLETION_MARKER",
    "DATASET_MANIFEST_SCHEMA_VERSION",
    "DatasetManifest",
    "ENVELOPE_FILE",
    "FIXED_SFT_CONTEXT_TOKENS",
    "GLOBAL_CHECKPOINT_NAMESPACES",
    "LAST_KNOWN_GOOD",
    "LAST_KNOWN_GOOD_POINTER",
    "SFT_BUDGET_SCHEMA_VERSION",
    "SFT_CHECKPOINT_SCHEMA_VERSION",
    "SFT_CHECKPOINT_ENVELOPE_SCHEMA_VERSION",
    "SFT_CONTEXT_LENGTH",
    "SFT_CONTEXT_TOKENS",
    "SFT_BATCH_STATE_SCHEMA_VERSION",
    "SFT_LOADER_STATE_SCHEMA_VERSION",
    "SFT_MANIFEST_SCHEMA_VERSION",
    "SFT_RENDER_TOKEN_LIMIT",
    "SFT_SEQUENCE_LENGTH",
    "CheckpointTransaction",
    "ConversationBatch",
    "DeterministicConversationBatchLoader",
    "FixedContextProfile",
    "FixedTokenBudget",
    "LoadedSFTCheckpoint",
    "SFTBudgetError",
    "SFTCheckpointError",
    "SFTContractError",
    "SFTConversationBatchLoader",
    "SFTConversationLoader",
    "SFTBatchLoader",
    "SFTDatasetManifest",
    "SFTManifest",
    "SFTLoaderError",
    "SFTLoaderError",
    "SFTManifestError",
    "ManifestValidationError",
    "RunLocalPathError",
    "SFTPathError",
    "SFTTransactionError",
    "SFTRankAwareBatchLoader",
    "SFTCheckpointTransaction",
    "begin_checkpoint_transaction",
    "begin_sft_checkpoint",
    "build_effective_token_budget",
    "build_sft_batch_loader",
    "canonical_json_hash",
    "compute_effective_tokens",
    "fixed_context_profile",
    "capture_rng_state",
    "commit_sft_checkpoint",
    "ensure_run_local_directory",
    "fixed_context_contract",
    "last_known_good_path",
    "load_checkpoint_envelope",
    "load_sft_checkpoint_envelope",
    "load_dataset_manifest",
    "load_sft_checkpoint",
    "make_sft_batch_loader",
    "read_jsonl_records",
    "resolve_run_local_path",
    "resolve_run_path",
    "restore_rng_state",
    "rollback_checkpoint_transaction",
    "rollback_sft_checkpoint",
    "safe_run_local_path",
    "save_checkpoint_transaction",
    "save_sft_checkpoint",
    "validate_dataset_manifest",
    "validate_context_contract",
    "validate_effective_token_budget",
    "validate_fixed_context",
    "validate_fixed_effective_token_budget",
    "validate_fixed_token_budget",
    "validate_jsonl_dataset_manifest",
    "validate_local_jsonl_manifest",
    "validate_sft_checkpoint_envelope",
    "validate_sft_manifest",
    "write_dataset_manifest",
    "write_jsonl_dataset_manifest",
    "write_sft_checkpoint",
]
