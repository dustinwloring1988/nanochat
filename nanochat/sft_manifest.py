from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nanochat.research_results import sha256_file, write_json_atomic

SFT_MANIFEST_SCHEMA_VERSION = 1
DATASET_MANIFEST_SCHEMA_VERSION = 1
JSONL_MANIFEST_SCHEMA_VERSION = SFT_MANIFEST_SCHEMA_VERSION
SFT_MANIFEST_VERSION = SFT_MANIFEST_SCHEMA_VERSION
SHA256_HEX_LENGTH = 64
TRUSTED_CACHE_ENVIRONMENT_VARIABLES = (
    "NANOCHAT_SHARED_CACHE",
    "NANOCHAT_CACHE_DIR",
    "NANOCHAT_TRUSTED_CACHE",
    "NANOCHAT_BASE_DIR",
)
GLOBAL_CHECKPOINT_ENVIRONMENT_VARIABLES = (
    "NANOCHAT_GLOBAL_CHECKPOINT_ROOT",
    "NANOCHAT_GLOBAL_CHECKPOINTS",
    "NANOCHAT_CHECKPOINT_ROOT",
)
GLOBAL_CHECKPOINT_NAMESPACES = frozenset(
    {
        "base_checkpoints",
        "chatsft_checkpoints",
        "sft_checkpoints",
        "chatrl_checkpoints",
        "rl_checkpoints",
        "global_checkpoints",
        "global_checkpoint",
    }
)


class SFTContractError(ValueError):
    pass


class SFTPathError(SFTContractError):
    pass


class SFTManifestError(SFTContractError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _json_copy(value: Any, field_name: str) -> Any:
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise SFTManifestError(f"{field_name} must be JSON-compatible") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SFTManifestError("manifest data must be JSON-compatible") from exc


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _absolute(path: str | os.PathLike) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _path_key(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _is_under(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([_path_key(path), _path_key(root)]) == _path_key(root)
    except ValueError:
        return False


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor) if path.anchor else Path()
    parts = path.parts
    if path.anchor:
        parts = parts[1:]
    for part in parts:
        current = current / part
        try:
            if os.path.lexists(current) and os.path.islink(current):
                return True
        except OSError as exc:
            raise SFTPathError(f"unable to inspect path component: {current}") from exc
    return False


def _coerce_roots(value: Any) -> tuple[Path, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, os.PathLike)):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise SFTPathError("trusted cache roots must be path-like values") from exc
    roots = []
    for item in values:
        if not isinstance(item, (str, os.PathLike)):
            raise SFTPathError("trusted cache roots must be path-like values")
        roots.append(Path(os.path.realpath(_absolute(item))))
    return tuple(roots)


def _trusted_cache_roots(extra: Any = None) -> tuple[Path, ...]:
    roots = list(_coerce_roots(extra))
    for name in TRUSTED_CACHE_ENVIRONMENT_VARIABLES:
        value = os.environ.get(name)
        if value:
            roots.append(Path(os.path.realpath(_absolute(value))))
    default_cache = Path.home() / ".cache" / "nanochat"
    roots.append(Path(os.path.realpath(default_cache)))
    return tuple(dict.fromkeys(roots))


def _global_checkpoint_roots(extra: Any = None) -> tuple[Path, ...]:
    roots = list(_coerce_roots(extra))
    for name in GLOBAL_CHECKPOINT_ENVIRONMENT_VARIABLES:
        value = os.environ.get(name)
        if value:
            roots.append(Path(os.path.realpath(_absolute(value))))
    base_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if base_dir:
        base_path = Path(os.path.realpath(_absolute(base_dir)))
        roots.extend(base_path / name for name in GLOBAL_CHECKPOINT_NAMESPACES)
    return tuple(dict.fromkeys(roots))


def _reject_reserved_namespace(path: Path, root: Path, names: frozenset[str]) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SFTPathError("resolved path escapes the run root") from exc
    for part in relative.parts:
        if part.casefold() in names:
            raise SFTPathError(f"global checkpoint namespace is not run-local: {part}")


def resolve_run_local_path(
    path: str | os.PathLike,
    run_root: str | os.PathLike,
    *,
    trusted_cache_roots: Any = None,
    global_checkpoint_roots: Any = None,
    global_namespace_names: Iterable[str] | None = None,
    require_exists: bool = False,
) -> Path:
    if path is None or run_root is None:
        raise SFTPathError("path and run_root are required")
    root_absolute = _absolute(run_root)
    if _has_symlink_component(root_absolute):
        raise SFTPathError("run root must not contain symlink components")
    try:
        root_resolved = Path(os.path.realpath(root_absolute))
    except (OSError, RuntimeError) as exc:
        raise SFTPathError("unable to resolve run root") from exc
    names = set(GLOBAL_CHECKPOINT_NAMESPACES)
    if global_namespace_names is not None:
        try:
            names.update(str(name).casefold() for name in global_namespace_names)
        except TypeError as exc:
            raise SFTPathError("global namespace names must be iterable") from exc
    names = frozenset(names)
    trusted_roots = _trusted_cache_roots(trusted_cache_roots)
    global_roots = _global_checkpoint_roots(global_checkpoint_roots)
    for protected in (*trusted_roots, *global_roots):
        if _is_under(root_resolved, protected):
            raise SFTPathError("run root is inside a trusted or global namespace")
    for part in root_resolved.parts:
        if part.casefold() in names:
            raise SFTPathError(f"run root uses a global checkpoint namespace: {part}")
    candidate_input = Path(os.path.expanduser(os.fspath(path)))
    if ".." in candidate_input.parts:
        raise SFTPathError("parent traversal is not allowed")
    if candidate_input.is_absolute():
        candidate_absolute = _absolute(candidate_input)
    else:
        candidate_absolute = _absolute(root_absolute / candidate_input)
    if _has_symlink_component(candidate_absolute):
        raise SFTPathError("run-local paths must not contain symlink components")
    try:
        candidate_resolved = Path(os.path.realpath(candidate_absolute))
    except (OSError, RuntimeError) as exc:
        raise SFTPathError("unable to resolve run-local path") from exc
    if not _is_under(candidate_resolved, root_resolved):
        raise SFTPathError("resolved path escapes the run root")
    for protected in (*trusted_roots, *global_roots):
        if _is_under(candidate_resolved, protected):
            raise SFTPathError("path is inside a trusted cache or global namespace")
    _reject_reserved_namespace(candidate_resolved, root_resolved, names)
    if require_exists and not candidate_resolved.exists():
        raise SFTPathError(f"run-local path does not exist: {candidate_resolved}")
    return candidate_resolved


def resolve_run_path(path: str | os.PathLike, run_root: str | os.PathLike, **kwargs):
    return resolve_run_local_path(path, run_root, **kwargs)


def safe_run_local_path(path: str | os.PathLike, run_root: str | os.PathLike, **kwargs):
    return resolve_run_local_path(path, run_root, **kwargs)


resolve_run_local_output = resolve_run_local_path
resolve_run_local_directory = resolve_run_local_path


def ensure_run_local_directory(
    path: str | os.PathLike,
    run_root: str | os.PathLike,
    **kwargs,
) -> Path:
    resolved = resolve_run_local_path(path, run_root, **kwargs)
    if resolved.exists() and not resolved.is_dir():
        raise SFTPathError(f"run-local path is not a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolve_run_local_path(resolved, run_root, require_exists=True)


@dataclass(frozen=True)
class DatasetFile:
    source: str
    split: str
    relative_path: str
    path: Path
    sha256: str
    byte_count: int
    record_count: int
    license: Any
    provenance: Any

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "split": self.split,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "record_count": self.record_count,
            "license": _json_copy(self.license, "license"),
            "provenance": _json_copy(self.provenance, "provenance"),
        }


@dataclass
class ValidatedDatasetManifest:
    manifest_path: Path
    run_root: Path
    schema_version: int
    payload: dict
    datasets: tuple[dict, ...]
    files: tuple[DatasetFile, ...]
    manifest_hash: str
    aggregate_sha256: str

    @property
    def path(self) -> Path:
        return self.manifest_path

    @property
    def hash(self) -> str:
        return self.manifest_hash

    @property
    def canonical_hash(self) -> str:
        return self.manifest_hash

    @property
    def raw_hash(self) -> str:
        return sha256_file(self.manifest_path)

    @property
    def aggregate_hash(self) -> str:
        return self.aggregate_sha256

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def records_for_file(self, relative_path: str) -> list[dict]:
        path = Path(relative_path).as_posix()
        for entry in self.files:
            if entry.relative_path == path:
                return read_jsonl_records(entry.path, self.run_root)
        raise SFTManifestError(f"file is not present in the manifest: {relative_path}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SFTManifestError(f"unable to read JSON manifest: {path}") from exc


def _read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise SFTManifestError(
                        f"blank JSONL record at {path}:{line_number}"
                    )
                try:
                    record = json.loads(
                        line,
                        parse_constant=_reject_json_constant,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise SFTManifestError(
                        f"invalid JSONL record at {path}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise SFTManifestError(
                        f"JSONL record must be an object at {path}:{line_number}"
                    )
                records.append(record)
    except SFTManifestError:
        raise
    except (OSError, UnicodeError) as exc:
        raise SFTManifestError(f"unable to read JSONL dataset: {path}") from exc
    return records


def read_jsonl_records(
    path: str | os.PathLike, run_root: str | os.PathLike
) -> list[dict]:
    resolved = resolve_run_local_path(path, run_root, require_exists=True)
    if not resolved.is_file() or resolved.suffix.casefold() != ".jsonl":
        raise SFTManifestError("dataset records must be a local .jsonl file")
    return _read_jsonl_records(resolved)


def _manifest_entries(payload: Any) -> tuple[dict, ...]:
    if isinstance(payload, list):
        entries = payload
    elif not isinstance(payload, dict):
        raise SFTManifestError("dataset manifest must be an object or list")
    else:
        nested = payload.get("manifest")
        if isinstance(nested, dict):
            merged = copy.deepcopy(payload)
            merged.pop("manifest")
            merged.update(nested)
            payload = merged
        raw = payload.get(
            "datasets",
            payload.get("sources", payload.get("data", payload.get("entries"))),
        )
        if raw is None and "records" in payload:
            raw = payload["records"]
        if raw is None:
            if (
                "files" in payload
                or "file" in payload
                or "file_manifest" in payload
                or "path" in payload
                or "relative_path" in payload
                or "local_path" in payload
                or "file_path" in payload
            ):
                raw = [payload]
            else:
                raise SFTManifestError("dataset manifest has no datasets")
        if isinstance(raw, dict):
            entries = []
            for source, item in raw.items():
                if not isinstance(item, dict):
                    raise SFTManifestError("dataset manifest entries must be objects")
                entry = copy.deepcopy(item)
                entry.setdefault("source", source)
                entries.append(entry)
        elif isinstance(raw, list):
            entries = raw
        else:
            raise SFTManifestError("datasets must be a list or mapping")
    normalised = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise SFTManifestError("dataset manifest entries must be objects")
        normalised.append(copy.deepcopy(entry))
    if not normalised:
        raise SFTManifestError("dataset manifest must contain at least one dataset")
    return tuple(normalised)


def _manifest_version(payload: Any) -> int:
    if not isinstance(payload, dict):
        raise SFTManifestError("dataset manifest must be a versioned object")
    if not any(
        field in payload
        for field in (
            "manifest_schema_version",
            "schema_version",
            "manifest_version",
            "version",
        )
    ):
        raise SFTManifestError("dataset manifest schema version is missing")
    value = payload.get(
        "manifest_schema_version",
        payload.get(
            "schema_version", payload.get("manifest_version", payload.get("version"))
        ),
    )
    if not _is_int(value) or value != SFT_MANIFEST_SCHEMA_VERSION:
        raise SFTManifestError("unsupported dataset manifest schema version")
    return int(value)


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SFTManifestError(f"{field_name} must be a non-empty string")
    return value


def _license_value(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        return _required_text(value, field_name)
    if isinstance(value, dict) and value:
        if "name" in value:
            _required_text(value["name"], f"{field_name}.name")
        return _json_copy(value, field_name)
    raise SFTManifestError(f"{field_name} is required")


def _provenance_value(value: Any, field_name: str) -> Any:
    if isinstance(value, str):
        return _required_text(value, field_name)
    if isinstance(value, dict) and value:
        return _json_copy(value, field_name)
    raise SFTManifestError(f"{field_name} is required")


def _file_specs(entry: dict) -> list[dict]:
    files = entry.get(
        "files",
        entry.get("file", entry.get("file_manifest", entry.get("artifacts"))),
    )
    if files is None and any(
        field in entry for field in ("path", "relative_path", "local_path", "file_path")
    ):
        files = [entry]
    if isinstance(files, str):
        files = [files]
    if isinstance(files, dict):
        converted = []
        for path, value in files.items():
            if not isinstance(value, dict):
                raise SFTManifestError("file manifest entries must be objects")
            item = copy.deepcopy(value)
            item.setdefault("path", path)
            converted.append(item)
        files = converted
    if not isinstance(files, list) or not files:
        raise SFTManifestError("each dataset must list at least one local JSONL file")
    result = []
    for item in files:
        if isinstance(item, str):
            result.append({"path": item})
            continue
        if not isinstance(item, dict):
            raise SFTManifestError("file manifest entries must be objects")
        result.append(copy.deepcopy(item))
    return result


def _relative_file_path(value: Any, field_name: str) -> str:
    path_text = _required_text(value, field_name)
    path = Path(path_text)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise SFTManifestError(f"{field_name} must be a relative run-local path")
    normalized = Path(os.path.normpath(path_text)).as_posix()
    if normalized in {"", "."} or normalized.startswith("../"):
        raise SFTManifestError(f"{field_name} is not a valid relative path")
    return normalized


def _declared_int(item: dict, names: tuple[str, ...], field_name: str) -> int | None:
    present = [name for name in names if name in item]
    if not present:
        return None
    value = item[present[0]]
    if not _is_int(value) or value < 0:
        raise SFTManifestError(f"{field_name} must be a non-negative integer")
    return int(value)


def _validate_file_record(
    item: dict,
    entry: dict,
    run_root: Path,
    seen_paths: set[str],
) -> tuple[DatasetFile, int]:
    source = _required_text(
        item.get(
            "source",
            item.get(
                "source_id",
                entry.get("source", entry.get("source_name", entry.get("source_id"))),
            ),
        ),
        "source",
    )
    split = _required_text(
        item.get(
            "split",
            item.get(
                "split_name",
                entry.get("split", entry.get("split_name", entry.get("subset"))),
            ),
        ),
        "split",
    )
    license_value = _license_value(
        item.get("license", entry.get("license", entry.get("license_name"))),
        "license",
    )
    provenance = _provenance_value(
        item.get("provenance", entry.get("provenance")),
        "provenance",
    )
    relative = _relative_file_path(
        item.get(
            "relative_path",
            item.get(
                "local_path",
                item.get("file_path", item.get("path")),
            ),
        ),
        "file path",
    )
    if relative in seen_paths:
        raise SFTManifestError(f"duplicate dataset file: {relative}")
    seen_paths.add(relative)
    try:
        file_path = resolve_run_local_path(relative, run_root, require_exists=True)
    except SFTPathError as exc:
        if "does not exist" in str(exc):
            raise SFTManifestError(f"dataset file is missing: {relative}") from exc
        raise
    if not file_path.is_file() or file_path.suffix.casefold() != ".jsonl":
        raise SFTManifestError(f"dataset file is not a local JSONL file: {relative}")
    declared_hash = item.get(
        "sha256",
        item.get(
            "sha256_hash",
            item.get(
                "file_sha256",
                item.get("sha256_digest", item.get("checksum", item.get("hash"))),
            ),
        ),
    )
    if not isinstance(declared_hash, str) or len(declared_hash) != SHA256_HEX_LENGTH:
        raise SFTManifestError(f"file SHA-256 is missing or malformed: {relative}")
    try:
        int(declared_hash, 16)
    except ValueError as exc:
        raise SFTManifestError(f"file SHA-256 is malformed: {relative}") from exc
    declared_hash = declared_hash.casefold()
    actual_hash = sha256_file(file_path)
    if actual_hash != declared_hash:
        raise SFTManifestError(f"dataset file hash is stale: {relative}")
    records = _read_jsonl_records(file_path)
    actual_bytes = file_path.stat().st_size
    declared_bytes = _declared_int(
        item,
        ("byte_count", "bytes", "size_bytes", "file_size", "size"),
        f"byte count for {relative}",
    )
    if declared_bytes is not None and declared_bytes != actual_bytes:
        raise SFTManifestError(f"dataset file byte count is stale: {relative}")
    declared_records = _declared_int(
        item,
        ("record_count", "example_count", "examples", "num_examples"),
        f"record count for {relative}",
    )
    actual_records = len(records)
    if declared_records is not None and declared_records != actual_records:
        raise SFTManifestError(f"dataset file record count is stale: {relative}")
    if "mtime_ns" in item:
        if (
            not _is_int(item["mtime_ns"])
            or item["mtime_ns"] != file_path.stat().st_mtime_ns
        ):
            raise SFTManifestError(f"dataset file timestamp is stale: {relative}")
    if "last_modified_ns" in item:
        if (
            not _is_int(item["last_modified_ns"])
            or item["last_modified_ns"] != file_path.stat().st_mtime_ns
        ):
            raise SFTManifestError(f"dataset file timestamp is stale: {relative}")
    return (
        DatasetFile(
            source=source,
            split=split,
            relative_path=relative,
            path=file_path,
            sha256=actual_hash,
            byte_count=actual_bytes,
            record_count=actual_records,
            license=license_value,
            provenance=provenance,
        ),
        actual_records,
    )


def _load_manifest_payload(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SFTManifestError(f"unable to read dataset manifest: {path}") from exc
    try:
        return json.loads(text, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError):
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                raise SFTManifestError(f"blank manifest line at {path}:{line_number}")
            try:
                value = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise SFTManifestError(
                    f"invalid JSONL dataset manifest at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise SFTManifestError("JSONL manifest records must be objects")
            records.append(value)
        if not records:
            raise SFTManifestError("dataset manifest is empty")
        if (
            len(records) > 1
            and "source" not in records[0]
            and "source_name" not in records[0]
            and "path" not in records[0]
            and "relative_path" not in records[0]
            and "files" not in records[0]
            and (
                "manifest_schema_version" in records[0]
                or "schema_version" in records[0]
                or "manifest_version" in records[0]
                or "version" in records[0]
            )
        ):
            header = records[0]
            return {**header, "datasets": records[1:]}
        common = {}
        for key in (
            "manifest_schema_version",
            "schema_version",
            "manifest_version",
            "version",
            "run_id",
            "provenance",
        ):
            if all(key in record for record in records):
                common[key] = records[0][key]
        return {**common, "datasets": records}


def _aggregate_hash(files: tuple[DatasetFile, ...]) -> str:
    payload = [entry.to_dict() for entry in files]
    return canonical_json_hash(payload)


def validate_jsonl_dataset_manifest(
    manifest_path: str | os.PathLike,
    run_root: str | os.PathLike | None = None,
    *,
    trusted_cache_roots: Any = None,
    global_checkpoint_roots: Any = None,
    global_namespace_names: Iterable[str] | None = None,
    expected_manifest_hash: str | None = None,
) -> ValidatedDatasetManifest:
    path = Path(os.path.expanduser(os.fspath(manifest_path)))
    if run_root is None:
        if not path.is_absolute():
            path = _absolute(path)
        run_root = path.parent
    path_kwargs = {
        "trusted_cache_roots": trusted_cache_roots,
        "global_checkpoint_roots": global_checkpoint_roots,
        "global_namespace_names": global_namespace_names,
    }
    resolved_path = resolve_run_local_path(
        path, run_root, require_exists=True, **path_kwargs
    )
    if not resolved_path.is_file():
        raise SFTManifestError("dataset manifest must be a regular file")
    payload = _load_manifest_payload(resolved_path)
    version = _manifest_version(payload)
    entries = _manifest_entries(payload)
    files: list[DatasetFile] = []
    seen_paths: set[str] = set()
    defaults = payload if isinstance(payload, dict) else {}
    for entry in entries:
        merged_entry = copy.deepcopy(entry)
        for field in (
            "source",
            "source_name",
            "split",
            "split_name",
            "license",
            "license_name",
            "provenance",
        ):
            if field not in merged_entry and field in defaults:
                merged_entry[field] = copy.deepcopy(defaults[field])
        for file_spec in _file_specs(merged_entry):
            file_entry, _ = _validate_file_record(
                file_spec,
                merged_entry,
                run_root,
                seen_paths,
            )
            files.append(file_entry)
    validated_files = tuple(files)
    aggregate = _aggregate_hash(validated_files)
    if isinstance(payload, dict):
        declared_values = []
        for name in (
            "aggregate_sha256",
            "aggregate_dataset_hash",
            "dataset_hash",
            "aggregate_hash",
            "files_hash",
        ):
            if name in payload:
                declared_values.append((name, payload[name]))
        for name, value in declared_values:
            if not isinstance(value, str) or value.casefold() != aggregate:
                raise SFTManifestError(f"manifest {name} is stale or mismatched")
        declared_file_count = payload.get("file_count")
        if declared_file_count is not None:
            if not _is_int(declared_file_count) or declared_file_count != len(
                validated_files
            ):
                raise SFTManifestError("manifest file count is stale")
        declared_total_bytes = payload.get(
            "total_byte_count", payload.get("total_bytes")
        )
        if declared_total_bytes is not None:
            total = sum(item.byte_count for item in validated_files)
            if not _is_int(declared_total_bytes) or declared_total_bytes != total:
                raise SFTManifestError("manifest total byte count is stale")
        declared_examples = payload.get("example_count", payload.get("total_examples"))
        if declared_examples is not None:
            total = sum(item.record_count for item in validated_files)
            if not _is_int(declared_examples) or declared_examples != total:
                raise SFTManifestError("manifest example count is stale")
        for name in ("missing_files", "stale_files"):
            if payload.get(name) not in (None, [], False):
                raise SFTManifestError(f"manifest reports {name}")
    manifest_hash_payload = copy.deepcopy(payload)
    if isinstance(manifest_hash_payload, dict):
        manifest_hash_payload.pop("manifest_hash", None)
        manifest_hash_payload.pop("manifest_sha256", None)
    manifest_hash = canonical_json_hash(manifest_hash_payload)
    if expected_manifest_hash is not None:
        if not isinstance(expected_manifest_hash, str):
            raise SFTManifestError("expected manifest hash is invalid")
        raw_hash = sha256_file(resolved_path)
        if expected_manifest_hash.casefold() not in {
            manifest_hash,
            raw_hash,
        }:
            raise SFTManifestError("dataset manifest hash is stale")
    if isinstance(payload, dict):
        for name in ("manifest_hash", "manifest_sha256"):
            if name in payload:
                declared = payload[name]
                raw_hash = sha256_file(resolved_path)
                if not isinstance(declared, str) or declared.casefold() not in {
                    manifest_hash,
                    raw_hash,
                }:
                    raise SFTManifestError(f"manifest {name} is stale or mismatched")
    return ValidatedDatasetManifest(
        manifest_path=resolved_path,
        run_root=Path(os.path.realpath(_absolute(run_root))),
        schema_version=version,
        payload=payload,
        datasets=entries,
        files=validated_files,
        manifest_hash=manifest_hash,
        aggregate_sha256=aggregate,
    )


def validate_dataset_manifest(*args, **kwargs) -> ValidatedDatasetManifest:
    return validate_jsonl_dataset_manifest(*args, **kwargs)


def load_dataset_manifest(*args, **kwargs) -> ValidatedDatasetManifest:
    return validate_jsonl_dataset_manifest(*args, **kwargs)


def validate_sft_manifest(*args, **kwargs) -> ValidatedDatasetManifest:
    return validate_jsonl_dataset_manifest(*args, **kwargs)


def validate_local_jsonl_manifest(*args, **kwargs) -> ValidatedDatasetManifest:
    return validate_jsonl_dataset_manifest(*args, **kwargs)


def write_jsonl_dataset_manifest(
    manifest_path: str | os.PathLike,
    payload: dict,
    run_root: str | os.PathLike,
) -> ValidatedDatasetManifest:
    if not isinstance(payload, dict):
        raise SFTManifestError("dataset manifest payload must be an object")
    resolved = resolve_run_local_path(manifest_path, run_root)
    write_json_atomic(resolved, payload)
    return validate_jsonl_dataset_manifest(resolved, run_root)


write_dataset_manifest = write_jsonl_dataset_manifest

DatasetManifest = ValidatedDatasetManifest
SFTDatasetManifest = ValidatedDatasetManifest
SFTManifest = ValidatedDatasetManifest
ManifestValidationError = SFTManifestError
RunLocalPathError = SFTPathError
