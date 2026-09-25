from __future__ import annotations

import base64
import json
import math
import os
import re
import stat
import subprocess
import threading
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from filelock import FileLock, Timeout

TRACE_SCHEMA = "nanochat.ai_scientist.trace"
TRACE_SCHEMA_VERSION = 1
REDACTION_VERSION = 1
TRACE_ROOT_NAME = "local-traces"
EVENTS_FILENAME = "events.jsonl"
MANIFEST_FILENAME = "manifest.json"
LOCK_FILENAME = "events.lock"
MAX_CONTENT_CHARS = 16_384
MAX_MANIFEST_BYTES = 1_048_576
ALLOWED_RECORD_TYPES = frozenset(
    {
        "run_start",
        "provider_call_start",
        "message",
        "tool_call",
        "tool_result",
        "provider_call_end",
        "stage_transition",
        "run_end",
    }
)
ALLOWED_TRUST_VALUES = frozenset(
    {
        "trusted_controller",
        "provider_metadata",
        "untrusted_provider_text",
        "untrusted_tool_output",
        "untrusted_dataset_content",
    }
)
_TRUST_BY_RECORD_TYPE = {
    "run_start": "trusted_controller",
    "provider_call_start": "provider_metadata",
    "message": "untrusted_provider_text",
    "tool_call": "untrusted_provider_text",
    "tool_result": "untrusted_tool_output",
    "provider_call_end": "provider_metadata",
    "stage_transition": "trusted_controller",
    "run_end": "trusted_controller",
}
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_DATA_URL_RE = re.compile(r"(?i)\bdata:[^\s\"'<>]+")
_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s<>\"']+")
_AUTH_SCHEME_RE = re.compile(r"(?i)\b(bearer|basic)\s+[a-z0-9+/_=.-]+")
_HEADER_RE = re.compile(
    r"(?i)(?P<label>proxy[-_\s]?authorization|proxy-authentication|authorization|set-cookie|cookie)"
    r"(?P<quote>[\"']?)\s*[:=]\s*(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\r\n]+)"
)
_ASSIGNMENT_RE = re.compile(
    r"(?P<key_quote>[\"']?)(?P<key>[A-Za-z_][A-Za-z0-9_.-]{0,127})"
    r"(?P=key_quote)\s*(?P<separator>[:=])\s*"
    r"(?P<value_quote>[\"']?)(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;}\]]+)"
    r"(?P=value_quote)"
)
_CLI_SECRET_RE = re.compile(
    r"(?i)(?P<flag>--(?:api[-_]?key|access[-_]?token|refresh[-_]?token|auth[-_]?token|"
    r"bearer[-_]?token|password|passwd|secret|credential|private[-_]?key|token|cookie))"
    r"(?:=|\s+)(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s]+)"
)
_TOKEN_PREFIX_RE = re.compile(
    r"(?i)\b(?:sk-(?:proj-|live-)?[a-z0-9_-]{8,}|"
    r"gh[pousr]_[a-z0-9_]{8,}|github_pat_[a-z0-9_]{8,}|"
    r"hf_[a-z0-9]{8,}|xox[baprs]-[a-z0-9-]{8,}|"
    r"AKIA[A-Z0-9]{12,}|AIza[a-zA-Z0-9_-]{20,})\b"
)
_SENSITIVE_ENV_CONTAINERS = frozenset(
    {"env", "envs", "environ", "environment", "environmentvariables", "envvars"}
)
_RESERVED_ENVELOPE_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "redaction_version",
        "run_id",
        "record_id",
        "sequence",
        "timestamp_utc",
    }
)
_HASH_KEYS = frozenset(
    {
        "checksum",
        "contenthash",
        "contentsha256",
        "hash",
        "md5",
        "sha1",
        "sha256",
        "sha512",
    }
)


class TraceError(RuntimeError):
    pass


class TraceConfigurationError(TraceError):
    pass


class TracePathError(TraceConfigurationError):
    pass


class TracePermissionError(TraceConfigurationError):
    pass


class TraceSerializationError(TraceError):
    pass


class TraceLimitError(TraceError):
    pass


class TraceDeletedError(TraceError):
    pass


class TraceStateError(TraceError):
    pass


class TraceNormalizationError(TraceError):
    pass


@dataclass(frozen=True)
class TraceConfig:
    run_workspace: Path | str
    run_id: str
    enabled: bool = False
    retention_seconds: int | None = None
    max_bytes: int | None = None
    cache_dir: Path | str | None = None
    lock_timeout_seconds: float = 30.0

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TraceConfigurationError("Trace enabled must be a boolean")
        if not self.enabled:
            return
        if not isinstance(self.run_id, str) or not _RUN_ID_RE.fullmatch(self.run_id):
            raise TraceConfigurationError("Trace run_id is invalid")
        if self.run_id in {".", ".."}:
            raise TraceConfigurationError("Trace run_id is invalid")
        if not _positive_int(self.retention_seconds):
            raise TraceConfigurationError("Trace retention_seconds must be positive")
        if not _positive_int(self.max_bytes):
            raise TraceConfigurationError("Trace max_bytes must be positive")
        if self.cache_dir is None:
            raise TraceConfigurationError("Trace cache_dir is required when enabled")
        if (
            isinstance(self.lock_timeout_seconds, bool)
            or not isinstance(self.lock_timeout_seconds, (int, float))
            or self.lock_timeout_seconds <= 0
        ):
            raise TraceConfigurationError("Trace lock timeout must be positive")


def create_trace_writer(
    config: TraceConfig, clock: Callable[[], datetime] | None = None
) -> TraceWriter | None:
    if not config.enabled:
        return None
    return TraceWriter(config, clock=clock)


def redact(value: Any) -> Any:
    return _redact(value, set(), None)


def _force_content_trust(value: Any, trust: str) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                trust
                if _normalized_key(key) == "trust"
                else _force_content_trust(item, trust)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_force_content_trust(item, trust) for item in value]
    return value


def utc_timestamp(value: datetime | None = None) -> str:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise TraceSerializationError("Trace timestamps must include a timezone")
    return (
        current.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def parse_utc_timestamp(value: str) -> datetime:
    if not isinstance(value, str):
        raise TraceSerializationError("Trace timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TraceSerializationError("Trace timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TraceSerializationError("Trace timestamp is invalid")
    return parsed.astimezone(timezone.utc)


def classify_base_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return "[REDACTED:base_url:invalid]"
    if not parsed.scheme or not host:
        return "[REDACTED:base_url:invalid]"
    scheme = re.sub(r"[^a-z0-9+.-]", "", parsed.scheme.casefold())
    safe_host = re.sub(r"[^a-z0-9.:_-]", "", host.casefold())
    if not safe_host:
        return "[REDACTED:base_url:invalid]"
    authority = safe_host
    if ":" in safe_host and not safe_host.startswith("["):
        authority = f"[{safe_host}]"
    if port is not None:
        authority = f"{authority}:{port}"
    return f"{scheme}://{authority}"


def new_opaque_id(prefix: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]", "_", prefix.casefold()).strip("_") or "record"
    return f"{normalized}-{uuid.uuid4().hex}"


class TraceWriter:
    def __init__(
        self, config: TraceConfig, clock: Callable[[], datetime] | None = None
    ) -> None:
        config.validate()
        if not config.enabled:
            raise TraceConfigurationError("Disabled tracing cannot create a writer")
        self.config = config
        self.run_id = config.run_id
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._thread_lock = threading.RLock()
        self._closed = False
        self._deleted = False
        self._append_failed = False
        self._record_count = 0
        self._byte_count = 0
        self._sequence = 0
        self._created_at = utc_timestamp(self._clock())
        self._updated_at = self._created_at
        self._capture_state = "active"
        self._stop_reason = None
        self._deletion_audit: list[dict] = []
        self._secure_inodes: dict[tuple[str, bool], tuple[int, int]] = {}
        self.run_workspace = _validate_run_workspace(
            config.run_workspace, config.cache_dir
        )
        self.trace_root = self.run_workspace / TRACE_ROOT_NAME
        _secure_directory(self.trace_root, self._secure_inodes)
        self.run_directory = self.trace_root / self.run_id
        _secure_directory(self.run_directory, self._secure_inodes)
        self._prune_expired_runs()
        self.events_path = self.run_directory / EVENTS_FILENAME
        self.manifest_path = self.run_directory / MANIFEST_FILENAME
        self.lock_path = self.run_directory / LOCK_FILENAME
        self._lock = FileLock(
            str(self.lock_path),
            timeout=config.lock_timeout_seconds,
        )
        needs_run_start = False
        with self._locked():
            _require_regular_private_file(
                self.lock_path, self._secure_inodes, secure=True
            )
            _reject_link_components(self.run_directory)
            events_exists = self.events_path.exists() or self.events_path.is_symlink()
            manifest_exists = (
                self.manifest_path.exists() or self.manifest_path.is_symlink()
            )
            if events_exists != manifest_exists:
                raise TraceStateError("Trace state is incomplete")
            if events_exists:
                manifest = _read_private_json(self.manifest_path, self._secure_inodes)
                self._load_existing_state(manifest)
                needs_run_start = self._record_count == 0
            else:
                if any(
                    item.name != LOCK_FILENAME for item in self.run_directory.iterdir()
                ):
                    raise TraceStateError(
                        "Trace run directory contains unexpected state"
                    )
                _create_private_file(self.events_path, self._secure_inodes)
                self._write_manifest_atomic(self._manifest_payload())
                needs_run_start = True
        if needs_run_start:
            self.write(
                "run_start",
                capture_mode="local_only",
                storage_policy="owner_only_run_workspace",
                retention_seconds=self.config.retention_seconds,
                max_bytes=self.config.max_bytes,
                creation_time_utc=self._created_at,
                upload=False,
                training=False,
                replay=False,
            )

    @property
    def record_count(self) -> int:
        return self._record_count

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def new_id(self, prefix: str) -> str:
        return new_opaque_id(prefix)

    def timestamp(self) -> str:
        return utc_timestamp(self._clock())

    def write(self, record_type: str, **fields: Any) -> str:
        if record_type not in ALLOWED_RECORD_TYPES:
            raise TraceSerializationError("Trace record type is not allowed")
        if _RESERVED_ENVELOPE_FIELDS.intersection(fields):
            raise TraceSerializationError("Trace envelope fields are controller-owned")
        trust = fields.pop("trust", _TRUST_BY_RECORD_TYPE[record_type])
        if trust not in ALLOWED_TRUST_VALUES:
            raise TraceSerializationError("Trace trust classification is invalid")
        try:
            json.dumps(fields, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TraceSerializationError(
                "Trace fields must be JSON-compatible and finite"
            ) from exc
        fields = redact(fields)
        if trust.startswith("untrusted_"):
            fields = _force_content_trust(fields, trust)
        with self._locked():
            self._ensure_writable()
            now = self._clock()
            sequence = self._sequence + 1
            record_id = f"record-{sequence:09d}"
            record = {
                **fields,
                "schema": TRACE_SCHEMA,
                "schema_version": TRACE_SCHEMA_VERSION,
                "redaction_version": REDACTION_VERSION,
                "record_type": record_type,
                "run_id": self.run_id,
                "record_id": record_id,
                "sequence": sequence,
                "timestamp_utc": utc_timestamp(now),
                "trust": trust,
            }
            _validate_record(record)
            data = _canonical_line(record)
            if self._byte_count + len(data) > self.config.max_bytes:
                self._capture_state = "stopped_max_size"
                self._stop_reason = "max_bytes_exceeded"
                self._updated_at = record["timestamp_utc"]
                self._write_manifest_atomic(self._manifest_payload())
                raise TraceLimitError("Trace maximum size reached; capture stopped")
            try:
                _append_private_line(self.events_path, data, self._secure_inodes)
            except TraceError:
                self._append_failed = True
                raise
            self._sequence = sequence
            self._record_count += 1
            self._byte_count += len(data)
            self._updated_at = record["timestamp_utc"]
            self._write_manifest_atomic(self._manifest_payload())
            return record_id

    def start_provider_call(
        self,
        *,
        provider: str,
        model: str,
        api_mode: str,
        base_url: Any,
        context: Mapping[str, Any] | None = None,
        request_parameters: Mapping[str, Any] | None = None,
        message_count: int = 0,
        max_attempts: int = 1,
        tool_definition: Any = None,
    ) -> str:
        call_id = self.new_id("call")
        fields = {
            "call_id": call_id,
            "provider": provider,
            "model": model,
            "api_mode": api_mode,
            "base_url_classification": classify_base_url(base_url),
            "context": dict(context or {}),
            "request_parameters": dict(request_parameters or {}),
            "message_count": message_count,
            "max_attempts": max_attempts,
            "attempt": 1,
            "timestamp_start_utc": self.timestamp(),
        }
        if tool_definition is not None:
            fields["tool_definition"] = tool_definition
        self.write("provider_call_start", **fields)
        return call_id

    def record_messages(
        self, call_id: str, messages: Any, direction: str
    ) -> list[dict[str, Any]]:
        if direction not in {"request", "response"}:
            raise TraceNormalizationError("Trace message direction is invalid")
        if not isinstance(messages, (list, tuple)):
            raise TraceNormalizationError("Trace messages must be an ordered list")
        references = []
        provider_tool_call_ids: dict[str, str] = {}
        for index, raw_message in enumerate(messages):
            if not isinstance(raw_message, Mapping):
                raise TraceNormalizationError("Trace message must be an object")
            role = raw_message.get("role")
            if not isinstance(role, str) or not role.strip():
                raise TraceNormalizationError("Trace message role is missing")
            role = role.strip()
            trust = (
                "untrusted_tool_output"
                if role == "tool"
                else (
                    "untrusted_dataset_content"
                    if direction == "request"
                    else "untrusted_provider_text"
                )
            )
            parts, tool_calls, tool_results = _normalize_message(raw_message)
            parts = [{**part, "trust": trust} for part in parts]
            message_id = self.new_id("message")
            reference = {
                "message_id": message_id,
                "index": index,
                "role": role,
                "part_count": len(parts),
                "trust": trust,
            }
            self.write(
                "message",
                call_id=call_id,
                message_id=message_id,
                direction=direction,
                role=role,
                parts=parts,
                trust=trust,
            )
            references.append(reference)
            for tool_call in tool_calls:
                tool_reference = self.record_tool_call(
                    call_id,
                    name=tool_call.get("name"),
                    arguments=tool_call.get("arguments"),
                    origin=tool_call.get("origin", "untrusted_provider_output"),
                    source_message_id=message_id,
                )
                reference.setdefault("tool_call_ids", []).append(
                    tool_reference["tool_call_id"]
                )
                source_call_id = tool_call.get("source_call_id")
                if isinstance(source_call_id, str) and source_call_id:
                    provider_tool_call_ids[source_call_id] = tool_reference[
                        "tool_call_id"
                    ]
            for tool_result in tool_results:
                source_call_id = tool_result.get("source_call_id")
                generated_call_id = (
                    provider_tool_call_ids.get(source_call_id)
                    if isinstance(source_call_id, str)
                    else None
                )
                result_reference = self.record_tool_result(
                    call_id,
                    result=tool_result.get("result"),
                    status=tool_result.get("status", "available"),
                    tool_call_id=generated_call_id,
                    source_message_id=message_id,
                )
                reference.setdefault("tool_result_ids", []).append(
                    result_reference["tool_result_id"]
                )
        return references

    def record_tool_call(
        self,
        call_id: str,
        *,
        name: Any,
        arguments: Any,
        origin: str = "untrusted_provider_output",
        source_message_id: str | None = None,
        schema_version: Any = None,
    ) -> dict[str, Any]:
        tool_call_id = self.new_id("toolcall")
        normalized_arguments = _normalize_tool_arguments(arguments)
        fields = {
            "tool_call_id": tool_call_id,
            "call_id": call_id,
            "name": name,
            "arguments": normalized_arguments,
            "arguments_origin": origin,
            "trust": "untrusted_provider_text",
        }
        if source_message_id is not None:
            fields["source_message_id"] = source_message_id
        if schema_version is not None:
            fields["schema_version"] = schema_version
        self.write("tool_call", **fields)
        return {"tool_call_id": tool_call_id, "name": name, "trust": fields["trust"]}

    def record_tool_result(
        self,
        call_id: str,
        *,
        result: Any,
        status: str,
        tool_call_id: str | None = None,
        source_message_id: str | None = None,
    ) -> dict[str, Any]:
        tool_result_id = self.new_id("toolresult")
        fields = {
            "tool_result_id": tool_result_id,
            "call_id": call_id,
            "status": status,
            "result": result,
            "trust": "untrusted_tool_output",
        }
        if tool_call_id is not None:
            fields["tool_call_id"] = tool_call_id
        if source_message_id is not None:
            fields["source_message_id"] = source_message_id
        self.write("tool_result", **fields)
        return {
            "tool_result_id": tool_result_id,
            "status": status,
            "trust": fields["trust"],
        }

    def finish_provider_call(
        self,
        call_id: str,
        *,
        status: str,
        reason: str,
        started_at: str,
        duration_seconds: float,
        attempts: list[dict],
        message_references: list[dict],
        fallback: dict,
        response: Mapping[str, Any],
        termination_error: Any = None,
    ) -> str:
        error_class = None
        error = None
        if termination_error is not None:
            error_class = _safe_error_class(termination_error)
            error = {
                "class": error_class,
                "message": str(termination_error),
                "trust": "untrusted_tool_output",
            }
        return self.write(
            "provider_call_end",
            call_id=call_id,
            timestamp_start_utc=started_at,
            timestamp_end_utc=self.timestamp(),
            duration_seconds=duration_seconds,
            attempts=attempts,
            retry_count=sum(
                1
                for attempt in attempts
                if isinstance(attempt.get("retry"), Mapping)
                and attempt["retry"].get("scheduled") is True
            ),
            message_references=message_references,
            fallback=fallback,
            response=dict(response),
            termination={
                "status": status,
                "reason": reason,
                "error_class": error_class,
            },
            error=error,
        )

    def request_delete(
        self, *, reason: str = "operator_request", operator: str = "local_operator"
    ) -> None:
        with self._locked():
            self._deleted = True
            self._capture_state = "deleted"
            self._stop_reason = reason
            self._deletion_audit.append(
                {
                    "timestamp_utc": self.timestamp(),
                    "reason": reason,
                    "operator": operator,
                }
            )
            if self.events_path.exists() or self.events_path.is_symlink():
                _reject_link_components(self.events_path)
                if not self.events_path.is_file():
                    raise TraceStateError("Trace event path is not a regular file")
                self.events_path.unlink()
            _fsync_directory(self.run_directory)
            self._updated_at = self._deletion_audit[-1]["timestamp_utc"]
            self._write_manifest_atomic(self._manifest_payload())
        self._closed = True

    def close(self) -> None:
        with self._thread_lock:
            if self._closed:
                return
            if self._append_failed:
                self._closed = True
                return
            final_status = "deleted" if self._deleted else "closed"
            final_reason = (
                "deleted" if self._deleted else self._stop_reason or "completed"
            )
            if not self._deleted and self._capture_state == "active":
                try:
                    self.write(
                        "run_end",
                        status=final_status,
                        reason=final_reason,
                        record_count=self._record_count + 1,
                        deletion_state="deleted" if self._deleted else "retained",
                    )
                    self._capture_state = "closed"
                except TraceLimitError:
                    self._capture_state = "stopped_max_size"
                    self._stop_reason = "max_bytes_exceeded"
                except TraceDeletedError:
                    self._capture_state = "deleted"
            self._updated_at = self.timestamp()
            self._write_manifest_atomic(self._manifest_payload())
            self._closed = True

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            try:
                self._lock.acquire()
            except Timeout as exc:
                raise TraceStateError("Trace per-run lock timed out") from exc
            try:
                yield
            finally:
                self._lock.release()

    def _ensure_writable(self) -> None:
        if self._append_failed:
            raise TraceStateError("Trace append integrity is uncertain")
        if self._closed or self._deleted:
            raise TraceDeletedError("Trace capture is closed")
        if self._capture_state != "active":
            raise TraceStateError("Trace capture is stopped")
        created = parse_utc_timestamp(self._created_at)
        now = self._clock()
        if (now - created).total_seconds() >= self.config.retention_seconds:
            self.request_delete(reason="retention_expired", operator="retention_policy")
            raise TraceDeletedError("Trace retention expired; capture stopped")

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "schema_version": TRACE_SCHEMA_VERSION,
            "redaction_version": REDACTION_VERSION,
            "run_id": self.run_id,
            "events_file": str(self.events_path),
            "record_count": self._record_count,
            "byte_count": self._byte_count,
            "created_at_utc": self._created_at,
            "updated_at_utc": self._updated_at,
            "capture_state": self._capture_state,
            "stop_reason": self._stop_reason,
            "deletion_state": "deleted" if self._deleted else "retained",
            "retention_seconds": self.config.retention_seconds,
            "max_bytes": self.config.max_bytes,
            "secure_deletion": "filesystem_dependent",
            "deletion_audit": self._deletion_audit,
        }

    def _load_existing_state(self, manifest: Mapping[str, Any]) -> None:
        required = {
            "schema",
            "schema_version",
            "redaction_version",
            "run_id",
            "events_file",
            "record_count",
            "byte_count",
            "created_at_utc",
            "capture_state",
            "retention_seconds",
            "max_bytes",
        }
        if not required.issubset(manifest):
            raise TraceStateError("Trace manifest is incomplete")
        if (
            manifest["schema"] != TRACE_SCHEMA
            or manifest["schema_version"] != TRACE_SCHEMA_VERSION
            or manifest["redaction_version"] != REDACTION_VERSION
            or manifest["run_id"] != self.run_id
        ):
            raise TraceStateError("Trace manifest version or run identity mismatch")
        if Path(manifest["events_file"]) != self.events_path:
            raise TraceStateError("Trace manifest event path mismatch")
        if manifest["retention_seconds"] != self.config.retention_seconds:
            raise TraceStateError("Trace retention policy mismatch")
        if manifest["max_bytes"] != self.config.max_bytes:
            raise TraceStateError("Trace size policy mismatch")
        record_count, byte_count, sequence = _validate_existing_events(
            self.events_path, self.run_id, self.config.max_bytes, self._secure_inodes
        )
        if (
            manifest["record_count"] != record_count
            or manifest["byte_count"] != byte_count
        ):
            raise TraceStateError("Trace manifest count mismatch")
        self._record_count = record_count
        self._byte_count = byte_count
        self._sequence = sequence
        self._created_at = utc_timestamp(
            parse_utc_timestamp(manifest["created_at_utc"])
        )
        self._updated_at = utc_timestamp(
            parse_utc_timestamp(manifest["updated_at_utc"])
        )
        state = manifest["capture_state"]
        if state not in {"active", "closed", "stopped_max_size", "deleted"}:
            raise TraceStateError("Trace manifest capture state is invalid")
        if state == "deleted":
            if self.events_path.exists():
                raise TraceStateError("Deleted trace has event data")
            self._deleted = True
            self._closed = True
            self._deletion_audit = list(manifest.get("deletion_audit", []))
        elif state == "stopped_max_size":
            self._capture_state = "stopped_max_size"
            self._stop_reason = manifest.get("stop_reason")
        else:
            self._capture_state = "active"
            self._stop_reason = manifest.get("stop_reason")

    def _write_manifest_atomic(self, payload: Mapping[str, Any]) -> None:
        _write_private_json_atomic(
            self.manifest_path,
            payload,
            self.run_directory,
            self._secure_inodes,
        )

    def _prune_expired_runs(self) -> None:
        now = self._clock()
        for candidate in self.trace_root.iterdir():
            if candidate == self.run_directory or not candidate.is_dir():
                continue
            _reject_link_components(candidate)
            manifest_path = candidate / MANIFEST_FILENAME
            if not manifest_path.exists():
                continue
            if manifest_path.is_symlink():
                raise TraceStateError("Trace retention manifest is a link")
            manifest = _read_private_json(manifest_path, self._secure_inodes)
            if (
                manifest.get("schema") != TRACE_SCHEMA
                or manifest.get("schema_version") != TRACE_SCHEMA_VERSION
                or manifest.get("redaction_version") != REDACTION_VERSION
                or manifest.get("run_id") != candidate.name
            ):
                raise TraceStateError("Trace retention manifest identity is invalid")
            if manifest.get("capture_state") == "deleted":
                continue
            retention = manifest.get("retention_seconds")
            if not _positive_int(retention):
                raise TraceStateError("Trace retention manifest is invalid")
            created = parse_utc_timestamp(manifest.get("created_at_utc"))
            if (now - created).total_seconds() < retention:
                continue
            expected_events = candidate / EVENTS_FILENAME
            if Path(manifest.get("events_file", "")) != expected_events:
                raise TraceStateError("Trace retention event path mismatch")
            lock_path = candidate / LOCK_FILENAME
            lock = FileLock(str(lock_path), timeout=self.config.lock_timeout_seconds)
            try:
                lock.acquire()
            except Timeout as exc:
                raise TraceStateError("Trace retention lock timed out") from exc
            try:
                _require_regular_private_file(
                    lock_path, self._secure_inodes, secure=True
                )
                current = _read_private_json(manifest_path, self._secure_inodes)
                if (
                    current.get("schema") != TRACE_SCHEMA
                    or current.get("schema_version") != TRACE_SCHEMA_VERSION
                    or current.get("redaction_version") != REDACTION_VERSION
                    or current.get("run_id") != candidate.name
                ):
                    raise TraceStateError(
                        "Trace retention manifest identity is invalid"
                    )
                if current.get("capture_state") == "deleted":
                    continue
                if expected_events.exists() or expected_events.is_symlink():
                    _reject_link_components(expected_events)
                    if not expected_events.is_file():
                        raise TraceStateError("Trace retention event path is invalid")
                    expected_events.unlink()
                timestamp = utc_timestamp(now)
                current["capture_state"] = "deleted"
                current["stop_reason"] = "retention_expired"
                current["deletion_state"] = "deleted"
                current["updated_at_utc"] = timestamp
                current.setdefault("deletion_audit", []).append(
                    {
                        "timestamp_utc": timestamp,
                        "reason": "retention_expired",
                        "operator": "retention_policy",
                    }
                )
                _write_private_json_atomic(
                    manifest_path, current, candidate, self._secure_inodes
                )
            finally:
                lock.release()


def _normalize_message(
    raw_message: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    content = raw_message.get("content")
    parts = _normalize_content(content)
    tool_calls = []
    raw_tool_calls = raw_message.get("tool_calls")
    if raw_tool_calls is not None:
        if not isinstance(raw_tool_calls, (list, tuple)):
            raise TraceNormalizationError("Trace tool_calls must be an ordered list")
        for item in raw_tool_calls:
            if not isinstance(item, Mapping):
                raise TraceNormalizationError("Trace tool call must be an object")
            function = item.get("function", item)
            if not isinstance(function, Mapping):
                raise TraceNormalizationError("Trace tool function must be an object")
            tool_calls.append(
                {
                    "name": function.get("name"),
                    "arguments": function.get("arguments", {}),
                    "origin": "untrusted_provider_output",
                    "source_call_id": item.get("id") or item.get("call_id"),
                }
            )
    tool_results = []
    if raw_message.get("role") == "tool":
        tool_results.append(
            {
                "result": content,
                "status": "available",
                "source_call_id": raw_message.get("tool_call_id"),
            }
        )
    return parts, tool_calls, tool_results


def _normalize_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [_text_part(content)]
    if content is None:
        return [{"type": "opaque_reference", "reference_omitted": True}]
    if isinstance(content, Mapping):
        return [{"type": "structured_json", "value": content}]
    if not isinstance(content, (list, tuple)):
        raise TraceNormalizationError("Trace message content is ambiguous")
    if len(content) > 256:
        raise TraceNormalizationError("Trace message has too many content parts")
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(_text_part(item))
            continue
        if not isinstance(item, Mapping):
            raise TraceNormalizationError("Trace content part must be an object")
        part_type = item.get("type")
        if part_type in {"text", "input_text", "output_text"}:
            parts.append(_text_part(item.get("text", "")))
        elif part_type in {"json", "structured_json"}:
            parts.append({"type": "structured_json", "value": item.get("value", item)})
        elif part_type in {"image_url", "input_image", "image"}:
            parts.append(
                {
                    "type": "image_reference",
                    "source": "omitted",
                    "bytes_omitted": True,
                    "uri_omitted": True,
                }
            )
        else:
            parts.append(
                {
                    "type": "opaque_reference",
                    "source_type": _safe_label(part_type),
                    "reference_omitted": True,
                }
            )
    return parts


def _text_part(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        raise TraceNormalizationError("Trace text part must be text")
    safe = _redact_text(value)
    truncated = len(safe) > MAX_CONTENT_CHARS
    if truncated:
        safe = safe[:MAX_CONTENT_CHARS]
    return {"type": "text", "text": safe, "truncated": truncated}


def _normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return {"unparsed_text": arguments, "normalized": False}
        return {"value": parsed, "normalized": True}
    return {"value": arguments, "normalized": arguments is not None}


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _safe_error_class(error: BaseException) -> str:
    name = re.sub(r"[^A-Za-z0-9_.]", "_", type(error).__name__)
    return name[:128] or "ProviderError"


def _safe_label(value: Any) -> str:
    if not isinstance(value, str):
        return "[REDACTED:label:type]"
    return _redact_text(value)[:128]


def _redact(value: Any, seen: set[int], key: str | None) -> Any:
    if _is_sensitive_key(key):
        return f"[REDACTED:{_key_category(key)}:field]"
    if isinstance(value, BaseException):
        return {
            "type": _safe_error_class(value),
            "message": _redact_text(str(value)),
            "trust": "untrusted_tool_output",
        }
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else "[REDACTED:non_finite_number]"
    identity = id(value)
    if identity in seen:
        return "[REDACTED:recursive_value]"
    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            if _normalized_key(
                key
            ) in _SENSITIVE_ENV_CONTAINERS or _looks_like_environment(value):
                return "[REDACTED:environment:dump]"
            result = {}
            for raw_key, item in value.items():
                safe_key = _redact_mapping_key(raw_key)
                if safe_key in result:
                    safe_key = f"{safe_key}__{len(result) + 1}"
                result[safe_key] = _redact(item, seen, str(raw_key))
            return result
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        seen.add(identity)
        try:
            return [_redact(item, seen, None) for item in value]
        finally:
            seen.remove(identity)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[REDACTED:binary:omitted]"
    return "[REDACTED:unsupported_type]"


def _is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = _normalized_key(key)
    if not normalized:
        return False
    if normalized in _HASH_KEYS or "contenthash" in normalized:
        return True
    exact = {
        "auth",
        "authentication",
        "authorization",
        "cookie",
        "cookiejar",
        "cookies",
        "credential",
        "credentials",
        "dataurl",
        "imagebase64",
        "imagebytes",
        "imagedata",
        "imageurl",
        "password",
        "passwd",
        "privatekey",
        "proxyauthorization",
        "refreshtoken",
        "accesstoken",
        "session",
        "sessionid",
        "setcookie",
        "sig",
        "sas",
        "oauthcode",
        "token",
    }
    if normalized in exact:
        return True
    suffixes = (
        "apikey",
        "authtoken",
        "bearertoken",
        "connectionstring",
        "credential",
        "csrftoken",
        "idtoken",
        "privatekey",
        "refreshtoken",
        "accesstoken",
        "password",
        "passwd",
        "secretkey",
        "signature",
        "sessionid",
        "xamzcredential",
        "xamzsecuritytoken",
        "xamzsignature",
        "xsrftoken",
    )
    if normalized.endswith(suffixes):
        return True
    if "authorization" in normalized or "proxyauthentication" in normalized:
        return True
    if "password" in normalized or "passwd" in normalized:
        return True
    if "secret" in normalized and normalized != "secretary":
        return True
    if "credential" in normalized:
        return True
    return False


def _key_category(key: Any) -> str:
    normalized = _normalized_key(key)
    if normalized in {
        "dataurl",
        "imagebase64",
        "imagebytes",
        "imagedata",
        "imageurl",
    }:
        return "image_reference"
    if "authorization" in normalized:
        return "authorization"
    if "cookie" in normalized:
        return "cookie"
    if "password" in normalized or "passwd" in normalized:
        return "password"
    if "token" in normalized:
        return "token"
    if "secret" in normalized:
        return "secret"
    if "credential" in normalized:
        return "credential"
    if "key" in normalized:
        return "api_key"
    if normalized in _HASH_KEYS or "contenthash" in normalized:
        return "content_hash"
    return "sensitive"


def _normalized_key(key: Any) -> str:
    if not isinstance(key, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", key.casefold())


def _redact_mapping_key(key: Any) -> str:
    if not isinstance(key, str):
        return "[REDACTED:key:type]"
    if _normalized_key(key) in _HASH_KEYS or "contenthash" in _normalized_key(key):
        return "[REDACTED:content_hash:field]"
    if _is_sensitive_key(key):
        return key[:128]
    safe = _redact_text(key)
    if safe == key:
        return key[:128]
    return safe[:128] or "[REDACTED:key:empty]"


def _looks_like_environment(value: Mapping[Any, Any]) -> bool:
    if len(value) < 3:
        return False
    if not all(isinstance(key, str) and _ENV_NAME_RE.fullmatch(key) for key in value):
        return False
    return sum(
        isinstance(item, (str, int, float, bool)) or item is None
        for item in value.values()
    ) == len(value)


def _redact_text(value: str) -> str:
    result = value[: MAX_CONTENT_CHARS * 2]
    for secret in _known_secret_values():
        if secret:
            result = result.replace(secret, "[REDACTED:known_credential]")
    result = _DATA_URL_RE.sub("[REDACTED:data_url:omitted]", result)
    result = _URL_RE.sub(lambda match: _sanitize_url(match.group(0)), result)
    result = _HEADER_RE.sub(
        lambda match: (
            f"{match.group('label')}{match.group('quote')}:"
            f"[REDACTED:{_key_category(match.group('label'))}:value]"
        ),
        result,
    )
    result = _ASSIGNMENT_RE.sub(_redact_assignment, result)
    result = _CLI_SECRET_RE.sub(
        lambda match: f"{match.group('flag')} [REDACTED:credential:argument]", result
    )
    result = _AUTH_SCHEME_RE.sub(
        lambda match: f"{match.group(1)} [REDACTED:authorization:scheme]", result
    )
    result = _TOKEN_PREFIX_RE.sub("[REDACTED:credential:prefix]", result)
    return result


def _redact_assignment(match: re.Match[str]) -> str:
    key = match.group("key")
    if not _is_sensitive_key(key):
        return match.group(0)
    quote = match.group("value_quote")
    return (
        f"{match.group('key_quote')}{key}{match.group('key_quote')}"
        f"{match.group('separator')}{quote}[REDACTED:{_key_category(key)}:value]{quote}"
    )


def _sanitize_url(value: str) -> str:
    trailing = ""
    while value and value[-1] in ".,;)]}":
        trailing = value[-1] + trailing
        value = value[:-1]
    if value.casefold().startswith("data:"):
        return "[REDACTED:data_url:omitted]" + trailing
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return "[REDACTED:url:invalid]" + trailing
    if not parsed.scheme or not host:
        return "[REDACTED:url:invalid]" + trailing
    safe_host = re.sub(r"[^a-z0-9.:_-]", "", host.casefold())
    authority = (
        f"[{safe_host}]"
        if ":" in safe_host and not safe_host.startswith("[")
        else safe_host
    )
    if port is not None:
        authority = f"{authority}:{port}"
    query = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        if _is_sensitive_key(key):
            item = f"[REDACTED:{_key_category(key)}:query]"
        query.append((key, item))
    return (
        urlunsplit(
            (
                re.sub(r"[^a-z0-9+.-]", "", parsed.scheme.casefold()),
                authority,
                parsed.path,
                urlencode(query, doseq=True),
                "",
            )
        )
        + trailing
    )


def _known_secret_values() -> list[str]:
    values = []
    for name, value in os.environ.items():
        if isinstance(value, str) and len(value) >= 4 and _is_sensitive_key(name):
            values.append(value)
    return sorted(set(values), key=len, reverse=True)


def _canonical_line(record: Any) -> bytes:
    _reject_non_finite(record, set())
    safe_record = redact(record)
    try:
        serialized = json.dumps(
            safe_record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return serialized.encode("utf-8") + b"\n"
    except (TypeError, ValueError, UnicodeError):
        raise TraceSerializationError("Trace record is not canonical JSON") from None


def _reject_non_finite(value: Any, seen: set[int]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TraceSerializationError("Trace numbers must be finite")
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        try:
            for item in value.values():
                _reject_non_finite(item, seen)
        finally:
            seen.remove(identity)
    elif isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        try:
            for item in value:
                _reject_non_finite(item, seen)
        finally:
            seen.remove(identity)


def _validate_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "schema_version",
        "redaction_version",
        "record_type",
        "run_id",
        "record_id",
        "sequence",
        "timestamp_utc",
        "trust",
    }
    if not required.issubset(record):
        raise TraceSerializationError("Trace envelope is incomplete")
    if record["schema"] != TRACE_SCHEMA:
        raise TraceSerializationError("Trace schema is invalid")
    if record["schema_version"] != TRACE_SCHEMA_VERSION:
        raise TraceSerializationError("Trace schema version is invalid")
    if record["redaction_version"] != REDACTION_VERSION:
        raise TraceSerializationError("Trace redaction version is invalid")
    if record["record_type"] not in ALLOWED_RECORD_TYPES:
        raise TraceSerializationError("Trace record type is invalid")
    if record["trust"] not in ALLOWED_TRUST_VALUES:
        raise TraceSerializationError("Trace trust classification is invalid")
    if not _RUN_ID_RE.fullmatch(str(record["run_id"])):
        raise TraceSerializationError("Trace run ID is invalid")
    if not isinstance(record["sequence"], int) or isinstance(record["sequence"], bool):
        raise TraceSerializationError("Trace sequence is invalid")
    if record["sequence"] <= 0:
        raise TraceSerializationError("Trace sequence is invalid")
    parse_utc_timestamp(record["timestamp_utc"])
    if record["record_type"] == "provider_call_start":
        if not {
            "call_id",
            "provider",
            "model",
            "api_mode",
            "timestamp_start_utc",
        }.issubset(record):
            raise TraceSerializationError("Provider call start is incomplete")
    elif record["record_type"] == "provider_call_end":
        if not {"call_id", "timestamp_end_utc", "termination"}.issubset(record):
            raise TraceSerializationError("Provider call end is incomplete")
        termination = record["termination"]
        if not isinstance(termination, Mapping) or termination.get("status") not in {
            "success",
            "error",
        }:
            raise TraceSerializationError("Provider termination is invalid")


def _append_private_line(
    path: Path, data: bytes, secured: dict[tuple[str, bool], tuple[int, int]]
) -> None:
    _reject_link_components(path)
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags, 0o600)
        _secure_open_file(path, descriptor, secured)
        written = os.write(descriptor, data)
        if written != len(data):
            raise TraceStateError("Trace append was not atomic")
        os.fsync(descriptor)
    except TraceError:
        raise
    except OSError as exc:
        raise TraceStateError("Trace append failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_existing_events(
    path: Path,
    run_id: str,
    max_bytes: int,
    secured: dict[tuple[str, bool], tuple[int, int]],
) -> tuple[int, int, int]:
    _require_regular_private_file(path, secured)
    size = path.stat().st_size
    if size > max_bytes:
        raise TraceStateError("Existing trace exceeds configured maximum size")
    count = 0
    byte_count = 0
    sequence = 0
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            for line in handle:
                if not line.endswith(b"\n") or b"\r" in line:
                    raise TraceStateError("Trace event file has a partial line")
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TraceStateError("Trace event file is not valid JSON") from exc
                _validate_record(record)
                if record["run_id"] != run_id:
                    raise TraceStateError("Trace event run identity mismatch")
                if record["sequence"] != sequence + 1:
                    raise TraceStateError("Trace event sequence is invalid")
                if line != _canonical_line(record):
                    raise TraceStateError("Trace event is not canonical")
                sequence = record["sequence"]
                count += 1
                byte_count += len(line)
    except OSError as exc:
        raise TraceStateError("Trace event file could not be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if byte_count != size:
        raise TraceStateError("Trace event byte count is invalid")
    return count, byte_count, sequence


def _read_private_json(
    path: Path, secured: dict[tuple[str, bool], tuple[int, int]]
) -> dict[str, Any]:
    _require_regular_private_file(path, secured)
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise TraceStateError("Trace manifest exceeds size limit")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            data = handle.read(MAX_MANIFEST_BYTES + 1)
    except OSError as exc:
        raise TraceStateError("Trace manifest could not be read") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(data) > MAX_MANIFEST_BYTES or not data.endswith(b"\n"):
        raise TraceStateError("Trace manifest encoding is invalid")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceStateError("Trace manifest is not valid JSON") from exc
    if not isinstance(payload, dict) or data != _canonical_line(payload):
        raise TraceStateError("Trace manifest is not canonical")
    return payload


def _write_private_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    parent: Path,
    secured: dict[tuple[str, bool], tuple[int, int]],
) -> None:
    data = _canonical_line(payload)
    if len(data) > MAX_MANIFEST_BYTES:
        raise TraceSerializationError("Trace manifest exceeds size limit")
    _reject_link_components(parent)
    temp_path = parent / f".{path.name}.tmp"
    if temp_path.exists() or temp_path.is_symlink():
        _reject_link_components(temp_path)
        if not temp_path.is_file():
            raise TraceStateError("Trace manifest temporary path is invalid")
        temp_path.unlink()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(temp_path, flags, 0o600)
        _secure_open_file(temp_path, descriptor, secured)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TraceStateError("Trace manifest write failed")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _reject_link_components(path)
        os.replace(temp_path, path)
        _require_regular_private_file(path, secured)
        _fsync_directory(parent)
    except TraceError:
        raise
    except OSError as exc:
        raise TraceStateError("Trace manifest update failed") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _create_private_file(
    path: Path, secured: dict[tuple[str, bool], tuple[int, int]]
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise TraceStateError("Trace file could not be created") from exc
    try:
        _secure_open_file(path, descriptor, secured)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _require_regular_private_file(
    path: Path,
    secured: dict[tuple[str, bool], tuple[int, int]],
    *,
    secure: bool = False,
) -> None:
    _reject_link_components(path)
    try:
        status = path.stat()
    except OSError as exc:
        raise TraceStateError("Trace file is unavailable") from exc
    if not stat.S_ISREG(status.st_mode):
        raise TraceStateError("Trace path is not a regular file")
    if secure:
        _secure_owner_only(path, False, secured)
    else:
        _verify_owner_only(path, False, secured)


def _secure_open_file(
    path: Path,
    descriptor: int,
    secured: dict[tuple[str, bool], tuple[int, int]],
) -> None:
    try:
        path_status = path.lstat()
        descriptor_status = os.fstat(descriptor)
    except OSError as exc:
        raise TraceStateError("Trace file identity could not be verified") from exc
    if _is_link_like(path_status) or not stat.S_ISREG(path_status.st_mode):
        raise TraceStateError("Trace file is not a regular non-link file")
    if not os.path.samestat(path_status, descriptor_status):
        raise TraceStateError("Trace file changed during open")
    _secure_owner_only(path, False, secured)


def _secure_directory(
    path: Path, secured: dict[tuple[str, bool], tuple[int, int]]
) -> None:
    if path.exists() or path.is_symlink():
        _reject_link_components(path)
        if not path.is_dir():
            raise TracePathError("Trace path is not a directory")
        _secure_owner_only(path, True, secured)
        return
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise TracePathError("Trace parent directory does not exist")
    _reject_link_components(parent)
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        _reject_link_components(path)
    except OSError as exc:
        raise TracePathError("Trace directory could not be created") from exc
    if not path.is_dir():
        raise TracePathError("Trace directory creation failed")
    _secure_owner_only(path, True, secured)
    _fsync_directory(parent)


def _secure_owner_only(
    path: Path,
    directory: bool,
    secured: dict[tuple[str, bool], tuple[int, int]],
) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise TracePermissionError(
            "Trace path permissions could not be inspected"
        ) from exc
    key = (str(path), directory)
    identity = (status.st_dev, status.st_ino)
    if secured.get(key) == identity:
        return
    if os.name == "posix":
        try:
            os.chmod(path, 0o700 if directory else 0o600)
        except OSError as exc:
            raise TracePermissionError(
                "Trace owner-only mode could not be set"
            ) from exc
        _verify_owner_only(path, directory, secured)
    elif os.name == "nt":
        _secure_windows_acl(path, directory)
        _verify_owner_only(path, directory, secured)
    else:
        raise TracePermissionError("Owner-only trace storage is unsupported")
    secured[key] = identity


def _verify_owner_only(
    path: Path,
    directory: bool,
    secured: dict[tuple[str, bool], tuple[int, int]],
) -> None:
    try:
        status = path.lstat()
    except OSError as exc:
        raise TracePermissionError(
            "Trace path permissions could not be inspected"
        ) from exc
    expected_mode = 0o700 if directory else 0o600
    if os.name == "posix":
        if status.st_uid != os.geteuid():
            raise TracePermissionError("Trace path is not owned by the current user")
        if stat.S_IMODE(status.st_mode) != expected_mode:
            raise TracePermissionError("Trace path is not owner-only")
    elif os.name == "nt":
        if not _verify_windows_acl(path, directory):
            raise TracePermissionError("Trace path does not have an owner-only ACL")
    else:
        raise TracePermissionError("Owner-only trace storage is unsupported")
    secured[(str(path), directory)] = (status.st_dev, status.st_ino)


def _secure_windows_acl(path: Path, directory: bool) -> None:
    script = _WINDOWS_OWNER_ONLY_ACL_SCRIPT
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    environment = os.environ.copy()
    environment["NANOCHAT_TRACE_ACL_PATH"] = str(path)
    environment["NANOCHAT_TRACE_ACL_DIRECTORY"] = "1" if directory else "0"
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TracePermissionError(
            "Windows owner-only ACL could not be established"
        ) from exc
    if completed.returncode != 0:
        raise TracePermissionError("Windows owner-only ACL could not be established")


def _verify_windows_acl(path: Path, directory: bool) -> bool:
    script = _WINDOWS_VERIFY_ACL_SCRIPT
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    environment = os.environ.copy()
    environment["NANOCHAT_TRACE_ACL_PATH"] = str(path)
    environment["NANOCHAT_TRACE_ACL_DIRECTORY"] = "1" if directory else "0"
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and any(
        line.strip() == "owner-only" for line in completed.stdout.splitlines()
    )


def _validate_run_workspace(
    workspace: Path | str, cache_dir: Path | str | None
) -> Path:
    workspace_path = _absolute_path(workspace)
    _reject_link_components(workspace_path)
    try:
        resolved = workspace_path.resolve(strict=True)
        status = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise TracePathError("Trace run workspace is unavailable") from exc
    if not stat.S_ISDIR(status.st_mode):
        raise TracePathError("Trace run workspace is not a directory")
    if cache_dir is None:
        return resolved
    cache_path = _absolute_path(cache_dir)
    _reject_link_components(cache_path)
    resolved_cache = cache_path.resolve(strict=False)
    trace_path = (resolved / TRACE_ROOT_NAME).resolve(strict=False)
    if _is_within(resolved, resolved_cache) or _is_within(trace_path, resolved_cache):
        raise TracePathError("Trace workspace must be outside the external cache")
    return resolved


def _absolute_path(value: Path | str) -> Path:
    try:
        path = Path(value).expanduser()
    except (TypeError, ValueError) as exc:
        raise TracePathError("Trace path is invalid") from exc
    if not path.is_absolute():
        path = Path.cwd() / path
    return Path(os.path.abspath(path))


def _reject_link_components(path: Path) -> None:
    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            status = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise TracePathError("Trace path could not be inspected") from exc
        if _is_link_like(status):
            raise TracePathError("Trace paths must not contain links or reparse points")


def _is_link_like(status: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(status, "st_file_attributes", 0)
    return stat.S_ISLNK(status.st_mode) or bool(attributes & reparse_flag)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError as exc:
        raise TraceStateError("Trace directory could not be synchronized") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


_WINDOWS_OWNER_ONLY_ACL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$path = $env:NANOCHAT_TRACE_ACL_PATH
$isDirectory = $env:NANOCHAT_TRACE_ACL_DIRECTORY -eq '1'
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
if ($isDirectory) {
    $acl = Get-Acl -LiteralPath $path
    $inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $identity,
        [System.Security.AccessControl.FileSystemRights]::FullControl,
        $inheritance,
        [System.Security.AccessControl.PropagationFlags]::None,
        [System.Security.AccessControl.AccessControlType]::Allow
    )
} else {
    $acl = Get-Acl -LiteralPath $path
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $identity,
        [System.Security.AccessControl.FileSystemRights]::FullControl,
        [System.Security.AccessControl.AccessControlType]::Allow
    )
}
$acl.SetAccessRuleProtection($true, $false)
foreach ($existingRule in @($acl.Access)) {
    $acl.RemoveAccessRuleSpecific($existingRule)
}
$acl.AddAccessRule($rule)
if ($isDirectory) {
    [System.IO.Directory]::SetAccessControl($path, $acl)
} else {
    [System.IO.File]::SetAccessControl($path, $acl)
}
"""


_WINDOWS_VERIFY_ACL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$path = $env:NANOCHAT_TRACE_ACL_PATH
$isDirectory = $env:NANOCHAT_TRACE_ACL_DIRECTORY -eq '1'
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$acl = Get-Acl -LiteralPath $path
$rules = @($acl.Access)
if (-not $acl.AreAccessRulesProtected) { throw 'not protected' }
if ($rules.Count -ne 1) { throw 'too many rules' }
$ownerSid = ([System.Security.Principal.NTAccount]::new($acl.Owner)).Translate([System.Security.Principal.SecurityIdentifier]).Value
$ruleSid = $rules[0].IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
if ($ownerSid -ne $identity.Value -or $ruleSid -ne $identity.Value) { throw 'wrong identity' }
if ($rules[0].AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) { throw 'not allow' }
if ($rules[0].IsInherited) { throw 'inherited rule' }
if (($rules[0].FileSystemRights -band [System.Security.AccessControl.FileSystemRights]::FullControl) -ne [System.Security.AccessControl.FileSystemRights]::FullControl) { throw 'not full control' }
Write-Output 'owner-only'
"""
