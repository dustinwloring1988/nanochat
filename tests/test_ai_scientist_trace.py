import base64
import hashlib
import importlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")
pytest.importorskip("httpx")
httpx = pytest.importorskip("httpx")
pytest.importorskip("yaml")
yaml = pytest.importorskip("yaml")

try:
    providers = importlib.import_module("ai_scientist.providers")
except ModuleNotFoundError as exc:
    if exc.name not in {"jsonschema", "openai"}:
        raise
    providers = None
trace_writer = pytest.importorskip("ai_scientist.trace_writer")
token_tracker = importlib.import_module(
    "ai_scientist.utils.token_tracker"
).token_tracker
ModelSpec = providers.ModelSpec if providers is not None else None
ProviderSession = providers.ProviderSession if providers is not None else None
configure_provider_tracing = (
    providers.configure_provider_tracing if providers is not None else None
)
llm_budget = providers.llm_budget if providers is not None else None
query_model = providers.query_model if providers is not None else None
TraceConfig = trace_writer.TraceConfig
TraceConfigurationError = trace_writer.TraceConfigurationError
TraceDeletedError = trace_writer.TraceDeletedError
TraceLimitError = trace_writer.TraceLimitError
TracePathError = trace_writer.TracePathError
TracePermissionError = trace_writer.TracePermissionError
TraceSerializationError = trace_writer.TraceSerializationError
TraceStateError = trace_writer.TraceStateError
TraceWriter = trace_writer.TraceWriter
classify_base_url = trace_writer.classify_base_url
create_trace_writer = trace_writer.create_trace_writer
redact = trace_writer.redact

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def reset_trace_and_budget_state():
    if providers is not None:
        configure_provider_tracing(None)
        llm_budget.reset()
    token_tracker.reset()
    yield
    if providers is not None:
        configure_provider_tracing(None)
        llm_budget.reset()
    token_tracker.reset()


def trace_config(
    workspace: Path,
    run_id: str = "fabricated-run",
    *,
    retention_seconds: int = 3600,
    max_bytes: int = 262_144,
    cache_dir: Path | None = None,
) -> TraceConfig:
    return TraceConfig(
        run_workspace=workspace,
        run_id=run_id,
        enabled=True,
        retention_seconds=retention_seconds,
        max_bytes=max_bytes,
        cache_dir=cache_dir or workspace.parent / "trusted-external-cache",
    )


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *, chat_outcomes=None, responses_outcomes=None):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(chat_outcomes or [chat_response("ok")])
        )
        self.responses = FakeResponses(responses_outcomes or [responses_response("ok")])


def usage(input_key="prompt_tokens", output_key="completion_tokens"):
    return SimpleNamespace(
        **{
            input_key: 3,
            output_key: 4,
            "prompt_tokens_details": SimpleNamespace(cached_tokens=1),
            "completion_tokens_details": SimpleNamespace(reasoning_tokens=2),
        }
    )


def chat_response(content="ok", tool_arguments=None):
    tool_calls = None
    if tool_arguments is not None:
        tool_calls = [
            SimpleNamespace(
                id="provider-tool-id",
                function=SimpleNamespace(
                    name="submit",
                    arguments=json.dumps(tool_arguments),
                ),
            )
        ]
    return SimpleNamespace(
        id="chat-fabricated",
        model="space-bunny-free",
        system_fingerprint="fingerprint-fabricated",
        usage=usage(),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ],
    )


def responses_response(text="ok", function_arguments=None):
    if function_arguments is not None:
        output = [
            SimpleNamespace(
                type="function_call",
                name="submit",
                arguments=json.dumps(function_arguments),
            )
        ]
        output_text = None
    else:
        output = [
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ]
        output_text = text
    return SimpleNamespace(
        id="responses-fabricated",
        model="gpt-test",
        system_fingerprint=None,
        usage=usage("input_tokens", "output_tokens"),
        output=output,
        output_text=output_text,
    )


def function_spec():
    return {
        "name": "submit",
        "description": "Submit a result",
        "json_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "access_token": {"type": "string"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    }


def fabricated_secrets():
    return {
        "opencode": "FABRICATED_OPENCODE_CREDENTIAL_001",
        "openai": "sk-fabricated-openai-credential-002",
        "openrouter": "FABRICATED_OPENROUTER_CREDENTIAL_003",
        "github": "ghp_fabricatedCredentialValue",
        "github_pat": "github_pat_fabricatedCredentialValue",
        "huggingface": "hf_fabricatedCredentialValue",
        "slack": "xoxb-fabricated-credential-value",
        "aws": "AKIAFABRICATED123456",
        "google": "AIzaFABRICATEDCredentialValue123456789",
        "authorization": "FABRICATED_BEARER_CREDENTIAL_004",
        "proxy": "FABRICATED_PROXY_CREDENTIAL_005",
        "cookie": "FABRICATED_COOKIE_SESSION_006",
        "session": "FABRICATED_SESSION_ID_007",
        "csrf": "FABRICATED_CSRF_TOKEN_008",
        "refresh": "FABRICATED_REFRESH_TOKEN_009",
        "password": "FABRICATED_PASSWORD_010",
        "credential": "FABRICATED_CREDENTIAL_011",
        "private_key": "FABRICATED_PRIVATE_KEY_012",
        "url_user": "FABRICATED_URL_USER_013",
        "url_password": "FABRICATED_URL_PASSWORD_014",
        "query": "FABRICATED_QUERY_SECRET_015",
        "environment": "FABRICATED_ENVIRONMENT_ASSIGNMENT_016",
        "tool_argument": "FABRICATED_TOOL_ARGUMENT_017",
        "tool_result": "FABRICATED_TOOL_RESULT_018",
        "error": "FABRICATED_EXCEPTION_CREDENTIAL_019",
    }


def nested_secret_payload():
    secrets = fabricated_secrets()
    return secrets, {
        "OPENCODE_API_KEY": secrets["opencode"],
        "openai-api-key": secrets["openai"],
        "apiKey": secrets["openrouter"],
        "opaque_credential_formats": [
            secrets["github"],
            secrets["github_pat"],
            secrets["huggingface"],
            secrets["slack"],
            secrets["aws"],
            secrets["google"],
        ],
        "Authorization": f"Bearer {secrets['authorization']}",
        "Proxy-Authorization": f"Basic {secrets['proxy']}",
        "Cookie": f"session={secrets['cookie']}; theme=fabricated",
        "Set-Cookie": f"refresh={secrets['session']}",
        "nested_auth": {
            "session_id": secrets["session"],
            "csrf_token": secrets["csrf"],
            "refresh_token": secrets["refresh"],
        },
        "credentials": {
            "username": "fabricated-user",
            "password": secrets["password"],
        },
        "private-key": secrets["private_key"],
        "connection_string": f"postgresql://{secrets['url_user']}:{secrets['url_password']}@db.invalid/db",
        "urls": [
            f"https://api.invalid/path?api_key={secrets['query']}&visible=fabricated",
        ],
        "tool": {
            "arguments": {
                "access_token": secrets["tool_argument"],
                "trust": "trusted_controller",
            },
            "result": {"authorization": f"Bearer {secrets['tool_result']}"},
            "error": f"request failed with OPENROUTER_API_KEY={secrets['error']}",
        },
        "command": (
            f"OPENAI_API_KEY={secrets['environment']} tool --api-key {secrets['tool_argument']}"
        ),
        "environment": {
            "PATH": "/fabricated/bin",
            "HOME": "/fabricated/home",
            "OPENAI_API_KEY": secrets["openai"],
        },
        "image": {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ZmFrZS1pbWFnZS1ieXRlcw=="},
        },
        "content_sha256": "f" * 64,
        "safe_field": "fabricated-safe-value",
    }


@pytest.mark.skipif(providers is None, reason="provider dependencies unavailable")
def test_default_disabled_creates_no_trace_files_and_does_not_change_query(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    config = TraceConfig(
        run_workspace=run_workspace,
        run_id="disabled-run",
        enabled=False,
        retention_seconds=None,
        max_bytes=None,
    )
    assert create_trace_writer(config) is None
    configure_provider_tracing(config)
    session = ProviderSession(
        ModelSpec.parse("opencode/space-bunny-free"),
        FakeClient(chat_outcomes=[chat_response("offline")]),
    )
    result = query_model(session, user_message="fabricated prompt")
    assert result.output == "offline"
    assert not (run_workspace / "local-traces").exists()


def test_recursive_redaction_is_canonical_deterministic_and_content_free(
    tmp_path, caplog
):
    caplog.clear()
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    secrets, payload = nested_secret_payload()
    writer = TraceWriter(trace_config(run_workspace))
    writer.write(
        "message",
        call_id="call-fabricated",
        message_id="message-fabricated",
        direction="request",
        role="user",
        parts=[{"type": "structured_json", "value": payload}],
        trust="untrusted_dataset_content",
    )
    events_path = writer.events_path
    manifest_path = writer.manifest_path
    writer.close()
    raw = events_path.read_bytes() + manifest_path.read_bytes()
    text = raw.decode("utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["redaction_version"] == 1
    assert manifest["events_file"] == str(events_path)
    assert manifest["record_count"] == len(read_events(events_path))
    assert manifest["byte_count"] == events_path.stat().st_size
    for secret in secrets.values():
        assert secret not in text
        assert base64.b64encode(secret.encode()).decode() not in text
        assert hashlib.sha256(secret.encode()).hexdigest() not in text
        assert all(secret not in record.message for record in caplog.records)
    assert "ZmFrZS1pbWFnZS1ieXRlcw==" not in text
    assert "content_sha256" not in text
    assert '"PATH"' not in text
    assert "fabricated-safe-value" in text
    assert "untrusted_dataset_content" in text
    events = read_events(events_path)
    run_start = events[0]
    assert run_start["capture_mode"] == "local_only"
    assert run_start["upload"] is False
    assert run_start["training"] is False
    assert run_start["replay"] is False
    structured = next(
        part["value"]
        for record in events
        if record["record_type"] == "message"
        for part in record["parts"]
        if part["type"] == "structured_json"
    )
    assert structured["tool"]["arguments"]["trust"] == "untrusted_dataset_content"
    for line in events_path.read_bytes().splitlines(keepends=True):
        assert line.endswith(b"\n") and b"\r" not in line
        record = json.loads(line)
        canonical = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert line == canonical + b"\n"
    first = json.dumps(redact(payload), sort_keys=True, separators=(",", ":"))
    second = json.dumps(redact(payload), sort_keys=True, separators=(",", ":"))
    assert first == second


def test_base_url_classification_drops_userinfo_query_and_fragment():
    classified = classify_base_url(
        "https://fabricated-user:FABRICATED_PASSWORD@api.invalid/v1?api_key=FABRICATED_QUERY"
        "#FABRICATED_FRAGMENT"
    )
    assert classified == "https://api.invalid"


def test_normalized_request_tool_result_links_opaque_ids_without_provider_ids(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    writer = TraceWriter(trace_config(run_workspace))
    call_id = writer.start_provider_call(
        provider="opencode",
        model="space-bunny-free",
        api_mode="chat_completions",
        base_url="https://api.invalid/v1",
    )
    writer.record_messages(
        call_id,
        [
            {
                "role": "assistant",
                "content": "call tool",
                "tool_calls": [
                    {
                        "id": "provider-fabricated-tool-id",
                        "type": "function",
                        "function": {
                            "name": "submit",
                            "arguments": json.dumps(
                                {"api_key": "FABRICATED_TOOL_ARGUMENT_024"}
                            ),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "provider-fabricated-tool-id",
                "content": "password=FABRICATED_TOOL_RESULT_025",
            },
        ],
        "request",
    )
    events_path = writer.events_path
    writer.close()
    events = read_events(events_path)
    tool_call = next(
        record for record in events if record["record_type"] == "tool_call"
    )
    tool_result = next(
        record for record in events if record["record_type"] == "tool_result"
    )
    assert tool_result["tool_call_id"] == tool_call["tool_call_id"]
    assert tool_call["tool_call_id"].startswith("toolcall-")
    raw = events_path.read_text(encoding="utf-8")
    assert "provider-fabricated-tool-id" not in raw
    assert "FABRICATED_TOOL_ARGUMENT_024" not in raw
    assert "FABRICATED_TOOL_RESULT_025" not in raw


def test_schema_rejects_unknown_records_nonfinite_numbers_and_reserved_fields(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    writer = TraceWriter(trace_config(run_workspace))
    with pytest.raises(TraceSerializationError, match="record type"):
        writer.write("unknown_record")
    with pytest.raises(TraceSerializationError, match="finite"):
        writer.write("stage_transition", value=float("nan"))
    with pytest.raises(TraceSerializationError, match="controller-owned"):
        writer.write("stage_transition", sequence=999)
    writer.close()


def test_manifest_and_existing_events_detect_noncanonical_partial_tail(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    config = trace_config(run_workspace)
    writer = TraceWriter(config)
    writer.write(
        "message",
        call_id="call-one",
        message_id="message-one",
        direction="request",
        role="user",
        parts=[{"type": "text", "text": "safe"}],
        trust="untrusted_dataset_content",
    )
    events_path = writer.events_path
    writer.close()
    with events_path.open("ab") as handle:
        handle.write(b'{"partial":')
    with pytest.raises(TraceStateError, match="canonical|state|partial"):
        TraceWriter(config)


def test_max_size_stops_before_partial_or_oversized_append(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    writer = TraceWriter(trace_config(run_workspace, max_bytes=25_000))
    payload = "fabricated-safe-" * 800
    writer.write(
        "message",
        call_id="call-size",
        message_id="message-size",
        direction="request",
        role="user",
        parts=[{"type": "text", "text": payload}],
        trust="untrusted_dataset_content",
    )
    size_before = writer.events_path.stat().st_size
    with pytest.raises(TraceLimitError):
        writer.write(
            "message",
            call_id="call-size",
            message_id="message-size-two",
            direction="request",
            role="user",
            parts=[{"type": "text", "text": payload}],
            trust="untrusted_dataset_content",
        )
    assert writer.events_path.stat().st_size == size_before
    manifest = json.loads(writer.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture_state"] == "stopped_max_size"
    assert manifest["byte_count"] == size_before
    assert writer.events_path.read_bytes().endswith(b"\n")
    writer.close()


def test_deletion_removes_only_event_data_and_records_content_free_audit(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    writer = TraceWriter(trace_config(run_workspace))
    writer.write(
        "message",
        call_id="call-delete",
        message_id="message-delete",
        direction="request",
        role="user",
        parts=[{"type": "text", "text": "safe-to-delete"}],
        trust="untrusted_dataset_content",
    )
    unrelated = writer.run_directory / "unrelated.txt"
    unrelated.write_text("fabricated-unrelated", encoding="utf-8")
    writer.request_delete(reason="operator_approved", operator="fabricated-operator")
    assert not writer.events_path.exists()
    assert unrelated.read_text(encoding="utf-8") == "fabricated-unrelated"
    manifest_text = writer.manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["capture_state"] == "deleted"
    assert manifest["deletion_audit"][-1]["reason"] == "operator_approved"
    assert "safe-to-delete" not in manifest_text
    with pytest.raises(TraceDeletedError):
        writer.write("run_end", status="closed", reason="after-delete")


def test_retention_prunes_only_expired_trace_run(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    start = datetime(2026, 9, 25, tzinfo=timezone.utc)
    current = [start]
    old = TraceWriter(
        trace_config(run_workspace, "old-run", retention_seconds=60),
        clock=lambda: current[0],
    )
    old.write(
        "message",
        call_id="call-old",
        message_id="message-old",
        direction="request",
        role="user",
        parts=[{"type": "text", "text": "expired-fabricated-content"}],
        trust="untrusted_dataset_content",
    )
    old_events = old.events_path
    old_manifest = old.manifest_path
    old.close()
    current[0] = start + timedelta(seconds=61)
    new = TraceWriter(
        trace_config(run_workspace, "new-run", retention_seconds=60),
        clock=lambda: current[0],
    )
    assert not old_events.exists()
    old_manifest_payload = json.loads(old_manifest.read_text(encoding="utf-8"))
    assert old_manifest_payload["capture_state"] == "deleted"
    assert old_manifest_payload["deletion_audit"][-1]["reason"] == "retention_expired"
    assert "expired-fabricated-content" not in old_manifest.read_text(encoding="utf-8")
    new.close()


def test_rejects_cache_escape_invalid_run_id_and_symlink_paths(tmp_path):
    cache = tmp_path / "cache"
    cache_run = cache / "run"
    cache_run.mkdir(parents=True)
    with pytest.raises(TracePathError, match="cache"):
        TraceWriter(trace_config(cache_run, cache_dir=cache))
    invalid_workspace = tmp_path / "invalid-run"
    invalid_workspace.mkdir()
    with pytest.raises(TraceConfigurationError, match="cache_dir"):
        TraceConfig(
            run_workspace=invalid_workspace,
            run_id="missing-cache",
            enabled=True,
            retention_seconds=60,
            max_bytes=1024,
        ).validate()
    with pytest.raises(TraceConfigurationError, match="run_id"):
        TraceWriter(trace_config(invalid_workspace, "../escape"))
    real_workspace = tmp_path / "real-workspace"
    real_workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_workspace = tmp_path / "linked-workspace"
    try:
        linked_workspace.symlink_to(real_workspace, target_is_directory=True)
    except OSError:
        linked_workspace = None
    if linked_workspace is not None:
        with pytest.raises(TracePathError, match="link|reparse"):
            TraceWriter(trace_config(linked_workspace))
    existing_root = invalid_workspace / "local-traces"
    existing_root.mkdir()
    escaped_run = existing_root / "escape"
    try:
        escaped_run.symlink_to(outside, target_is_directory=True)
    except OSError:
        escaped_run = None
    if escaped_run is not None:
        with pytest.raises(TracePathError, match="link|reparse"):
            TraceWriter(trace_config(invalid_workspace, "escape"))


@pytest.mark.skipif(os.name != "posix", reason="POSIX mode assertion")
def test_posix_trace_paths_are_owner_only(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    writer = TraceWriter(trace_config(run_workspace))
    assert writer.trace_root.stat().st_mode & 0o077 == 0
    assert writer.run_directory.stat().st_mode & 0o077 == 0
    assert writer.events_path.stat().st_mode & 0o077 == 0
    assert writer.manifest_path.stat().st_mode & 0o077 == 0
    assert writer.lock_path.stat().st_mode & 0o077 == 0
    writer.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows ACL assertion")
def test_windows_acl_failure_stops_before_event_append(tmp_path, monkeypatch):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()

    def fail_acl(path, directory):
        raise TracePermissionError("fabricated ACL failure")

    monkeypatch.setattr(trace_writer, "_secure_windows_acl", fail_acl)
    with pytest.raises(TracePermissionError, match="ACL"):
        TraceWriter(trace_config(run_workspace))
    assert not (
        run_workspace / "local-traces" / "fabricated-run" / "events.jsonl"
    ).exists()


def test_disabled_yaml_section_is_explicit_and_has_no_approved_policy_values():
    config = yaml.safe_load((ROOT / "bfts_config.yaml").read_text(encoding="utf-8"))
    assert config["trace"] == {
        "enabled": False,
        "run_id": None,
        "retention_seconds": None,
        "max_bytes": None,
    }


@pytest.mark.skipif(providers is None, reason="provider dependencies unavailable")
def test_provider_trace_records_lifecycle_messages_tools_and_redacts_content(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    configure_provider_tracing(trace_config(run_workspace, max_bytes=1_048_576))
    response = chat_response(
        tool_arguments={"value": 2.5, "access_token": "FABRICATED_TOOL_ARGUMENT_020"}
    )
    session = ProviderSession(
        ModelSpec.parse("opencode/space-bunny-free"),
        FakeClient(chat_outcomes=[response]),
    )
    result = query_model(
        session,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OPENAI_API_KEY=FABRICATED_OPENAI_020"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,ZmFrZS1pbWFnZS1ieXRlcw=="
                        },
                    },
                ],
            }
        ],
        func_spec=function_spec(),
        trace_context={
            "role": "code",
            "node_id": "node-fabricated",
            "stage_id": "stage-fabricated",
        },
    )
    assert result.output == {
        "value": 2.5,
        "access_token": "FABRICATED_TOOL_ARGUMENT_020",
    }
    events_path = run_workspace / "local-traces" / "fabricated-run" / "events.jsonl"
    configure_provider_tracing(None)
    events = read_events(events_path)
    types = [record["record_type"] for record in events]
    assert types[0] == "run_start"
    assert "provider_call_start" in types
    assert "message" in types
    assert "tool_call" in types
    assert "tool_result" in types
    assert any(
        record["status"] == "not_executed"
        for record in events
        if record["record_type"] == "tool_result"
    )
    assert types[-2:] == ["provider_call_end", "run_end"]
    call_start = next(r for r in events if r["record_type"] == "provider_call_start")
    assert call_start["context"]["node_id"] == "node-fabricated"
    assert call_start["message_count"] == 1
    call_end = next(r for r in events if r["record_type"] == "provider_call_end")
    assert call_end["attempts"][0]["status"] == "success"
    assert call_end["retry_count"] == 0
    assert call_end["message_references"][0]["part_count"] == 2
    assert call_end["message_references"][0]["trust"] == "untrusted_dataset_content"
    assert call_end["fallback"] == {
        "occurred": False,
        "from_mode": None,
        "to_mode": None,
        "reason": None,
    }
    assert call_end["termination"] == {
        "status": "success",
        "reason": "completed",
        "error_class": None,
    }
    assert call_end["response"]["usage"]["usage_missing"] is False
    raw = events_path.read_text(encoding="utf-8")
    assert "FABRICATED_OPENAI_020" not in raw
    assert "FABRICATED_TOOL_ARGUMENT_020" not in raw
    assert "ZmFrZS1pbWFnZS1ieXRlcw==" not in raw
    assert all(
        part["trust"] == record["trust"]
        for record in events
        if record["record_type"] == "message"
        for part in record["parts"]
    )
    assert any(
        part.get("type") == "image_reference" and part.get("bytes_omitted") is True
        for record in events
        if record["record_type"] == "message"
        for part in record["parts"]
    )


@pytest.mark.skipif(providers is None, reason="provider dependencies unavailable")
def test_provider_trace_records_retry_and_fallback_metadata(tmp_path, monkeypatch):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    configure_provider_tracing(trace_config(run_workspace, "retry-run"))
    request = httpx.Request("POST", "https://provider.invalid/v1/chat")
    retry_error = openai.APIConnectionError(
        request=request,
        message="connection failed OPENAI_API_KEY=FABRICATED_RETRY_021",
    )
    client = FakeClient(chat_outcomes=[retry_error, chat_response('{"value": 1.5}')])
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    monkeypatch.setattr(providers.time, "sleep", lambda _: None)
    monkeypatch.setattr(providers.random, "uniform", lambda *_: 0)
    result = query_model(
        session,
        user_message="submit",
        func_spec=function_spec(),
        structured_mode="json",
        max_retries=1,
    )
    assert result.output == {"value": 1.5}
    events_path = run_workspace / "local-traces" / "retry-run" / "events.jsonl"
    configure_provider_tracing(None)
    events = read_events(events_path)
    end = next(
        record for record in events if record["record_type"] == "provider_call_end"
    )
    assert end["retry_count"] == 1
    assert [attempt["status"] for attempt in end["attempts"]] == ["error", "success"]
    assert end["attempts"][0]["retry"]["scheduled"] is True
    assert end["fallback"]["occurred"] is False
    raw = events_path.read_text(encoding="utf-8")
    assert "FABRICATED_RETRY_021" not in raw

    fallback_workspace = tmp_path / "fallback-run"
    fallback_workspace.mkdir()
    configure_provider_tracing(trace_config(fallback_workspace, "fallback-run"))
    http_request = httpx.Request("POST", "https://provider.invalid/v1/chat")
    http_response = httpx.Response(400, request=http_request, json={"error": "tool"})
    fallback_error = openai.APIStatusError(
        "tool use unsupported with OPENROUTER_API_KEY=FABRICATED_FALLBACK_022",
        response=http_response,
        body=None,
    )
    fallback_client = FakeClient(
        chat_outcomes=[fallback_error, chat_response('{"value": 1.5}')]
    )
    fallback_session = ProviderSession(
        ModelSpec.parse("opencode/space-bunny-free"), fallback_client
    )
    fallback_result = query_model(
        fallback_session,
        user_message="submit",
        func_spec=function_spec(),
        max_retries=0,
    )
    assert fallback_result.output == {"value": 1.5}
    fallback_events_path = (
        fallback_workspace / "local-traces" / "fallback-run" / "events.jsonl"
    )
    configure_provider_tracing(None)
    fallback_events = read_events(fallback_events_path)
    fallback_end = next(
        record
        for record in fallback_events
        if record["record_type"] == "provider_call_end"
    )
    assert fallback_end["fallback"] == {
        "occurred": True,
        "from_mode": "auto",
        "to_mode": "json",
        "reason": "structured_tool_unsupported",
    }
    assert [attempt["status"] for attempt in fallback_end["attempts"]] == [
        "error",
        "success",
    ]
    assert fallback_end["retry_count"] == 0
    assert "FABRICATED_FALLBACK_022" not in fallback_events_path.read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(providers is None, reason="provider dependencies unavailable")
def test_provider_trace_records_sanitized_error_termination(tmp_path):
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    configure_provider_tracing(trace_config(run_workspace, "error-run"))
    request = httpx.Request("POST", "https://provider.invalid/v1/chat")
    error = openai.APIConnectionError(
        request=request,
        message="failed Authorization: Bearer FABRICATED_TERMINATION_023",
    )
    session = ProviderSession(
        ModelSpec.parse("opencode/space-bunny-free"),
        FakeClient(chat_outcomes=[error]),
    )
    with pytest.raises(openai.APIConnectionError):
        query_model(session, user_message="hello", max_retries=0)
    events_path = run_workspace / "local-traces" / "error-run" / "events.jsonl"
    configure_provider_tracing(None)
    events = read_events(events_path)
    end = next(
        record for record in events if record["record_type"] == "provider_call_end"
    )
    assert end["termination"]["status"] == "error"
    assert end["termination"]["error_class"] == "APIConnectionError"
    assert end["error"]["trust"] == "untrusted_tool_output"
    raw = events_path.read_text(encoding="utf-8")
    assert "FABRICATED_TERMINATION_023" not in raw
    assert "[REDACTED:" in raw


def test_direct_provider_paths_fail_closed_when_tracing_is_enabled(tmp_path):
    if providers is None:
        pytest.skip("provider dependencies unavailable")
    run_workspace = tmp_path / "run"
    run_workspace.mkdir()
    configure_provider_tracing(trace_config(run_workspace, "direct-guard"))
    try:
        vlm = pytest.importorskip("ai_scientist.vlm")
        with pytest.raises(RuntimeError, match="not trace-instrumented"):
            vlm.make_llm_call(
                SimpleNamespace(),
                model="gpt-test",
                temperature=0.0,
                system_message="system",
                prompt=[{"role": "user", "content": "fabricated"}],
            )
    finally:
        configure_provider_tracing(None)
