import json
import os
import random
import re
import sqlite3
import tempfile
import threading
from pathlib import Path
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jsonschema
import openai

from ai_scientist.trace_writer import TraceConfig, TraceWriter, create_trace_writer
from ai_scientist.utils.token_tracker import token_tracker

DEFAULT_MODEL = "opencode/space-bunny-free"
DEFAULT_MAX_TOKENS = 4096


class ProviderError(RuntimeError):
    pass


class ProviderBudgetError(ProviderError):
    pass


class StructuredOutputError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str | None
    api_key_env: str
    default_api_mode: str


PROVIDER_CONFIGS = {
    "opencode": ProviderConfig(
        name="opencode",
        base_url="https://opencode.ai/zen/v1",
        api_key_env="OPENCODE_API_KEY",
        default_api_mode="chat_completions",
    ),
    "openai": ProviderConfig(
        name="openai",
        base_url=None,
        api_key_env="OPENAI_API_KEY",
        default_api_mode="responses",
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        default_api_mode="chat_completions",
    ),
}

PROVIDER_ALIASES = {
    "zen": "opencode",
    "opencode-zen": "opencode",
}

OPENCODE_RESPONSES_PREFIXES = (
    "gpt-",
    "grok-",
    "muse-spark-",
)


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    provider: str
    model: str
    api_mode: str
    base_url: str | None
    api_key_env: str

    @classmethod
    def parse(cls, identifier: str, api_mode: str | None = None) -> "ModelSpec":
        if not isinstance(identifier, str) or not identifier.strip():
            raise ProviderError("A model identifier is required")
        identifier = identifier.strip()
        if "/" in identifier:
            provider, model = identifier.split("/", 1)
            provider = PROVIDER_ALIASES.get(provider.lower(), provider.lower())
        elif (
            identifier.startswith("gpt-")
            or identifier.startswith("o1")
            or identifier.startswith("o3")
        ):
            provider, model = "openai", identifier
        else:
            raise ProviderError(
                "Model identifiers must use provider/model, for example opencode/space-bunny-free"
            )
        if provider not in PROVIDER_CONFIGS:
            raise ProviderError(
                f"Unsupported provider {provider!r}; choose opencode, openai, or openrouter"
            )
        if not model:
            raise ProviderError(f"Missing model ID after provider in {identifier!r}")
        config = PROVIDER_CONFIGS[provider]
        resolved_mode = api_mode or os.environ.get("AI_SCIENTIST_API_MODE")
        if not resolved_mode:
            if provider == "opencode" and model.startswith(OPENCODE_RESPONSES_PREFIXES):
                resolved_mode = "responses"
            else:
                resolved_mode = config.default_api_mode
        resolved_mode = resolved_mode.replace("-", "_")
        if resolved_mode in {"chat", "chat_completions"}:
            resolved_mode = "chat_completions"
        if resolved_mode not in {"chat_completions", "responses"}:
            raise ProviderError(
                f"Unsupported API mode {resolved_mode!r}; use chat_completions or responses"
            )
        return cls(
            identifier=f"{provider}/{model}",
            provider=provider,
            model=model,
            api_mode=resolved_mode,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
        )


@dataclass
class ProviderSession:
    spec: ModelSpec
    client: Any


@dataclass
class QueryResult:
    output: str | dict
    request_time: float
    input_tokens: int
    output_tokens: int
    info: dict


@dataclass
class _ActiveQueryTrace:
    writer: TraceWriter
    call_id: str = ""
    started_at: str = ""
    started_perf: float = 0.0
    attempts: list[dict] = field(default_factory=list)
    message_references: list[dict] = field(default_factory=list)
    fallback: dict = field(
        default_factory=lambda: {
            "occurred": False,
            "from_mode": None,
            "to_mode": None,
            "reason": None,
        }
    )
    lock: threading.Lock = field(default_factory=threading.Lock)

    def observe_attempt(self, event: dict) -> None:
        phase = event.get("phase")
        attempt = event.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool):
            return
        with self.lock:
            if phase == "start":
                self.attempts.append(
                    {
                        "attempt": attempt,
                        "timestamp_start_utc": self.writer.timestamp(),
                        "perf_counter_start": time.perf_counter(),
                    }
                )
                return
            if phase != "end":
                return
            for record in reversed(self.attempts):
                if record["attempt"] == attempt:
                    record.update(
                        {
                            "timestamp_end_utc": self.writer.timestamp(),
                            "duration_seconds": round(
                                max(
                                    0.0,
                                    time.perf_counter() - record["perf_counter_start"],
                                ),
                                6,
                            ),
                            "status": event.get("status"),
                            "error_class": event.get("error_class"),
                            "retry": event.get("retry"),
                        }
                    )
                    record.pop("perf_counter_start", None)
                    return

    def set_fallback(self, from_mode: str, to_mode: str, reason: str) -> None:
        with self.lock:
            self.fallback = {
                "occurred": True,
                "from_mode": from_mode,
                "to_mode": to_mode,
                "reason": reason,
            }

    def snapshot(self) -> tuple[list[dict], dict, list[dict]]:
        with self.lock:
            attempts = []
            for record in self.attempts:
                snapshot = dict(record)
                if isinstance(record.get("retry"), dict):
                    snapshot["retry"] = dict(record["retry"])
                attempts.append(snapshot)
            return (
                attempts,
                dict(self.fallback),
                [dict(item) for item in self.message_references],
            )

    def finish(
        self,
        *,
        status: str,
        reason: str,
        response: dict | None = None,
        error: BaseException | None = None,
    ) -> None:
        attempts, fallback, message_references = self.snapshot()
        self.writer.finish_provider_call(
            self.call_id,
            status=status,
            reason=reason,
            started_at=self.started_at,
            duration_seconds=round(
                max(0.0, time.perf_counter() - self.started_perf), 6
            ),
            attempts=attempts,
            message_references=message_references,
            fallback=fallback,
            response=response or {},
            termination_error=error,
        )


@dataclass(frozen=True)
class FunctionContract:
    name: str
    description: str
    json_schema: dict


class LLMCallBudget:
    def __init__(self):
        self.max_calls = _optional_int_env("AI_SCIENTIST_MAX_API_CALLS")
        self.max_input_tokens = _optional_int_env("AI_SCIENTIST_MAX_INPUT_TOKENS")
        self.max_output_tokens = _optional_int_env("AI_SCIENTIST_MAX_OUTPUT_TOKENS")
        run_id = re.sub(
            r"[^A-Za-z0-9_-]+", "_", os.environ.get("AI_SCIENTIST_RUN_ID", "default")
        )
        default_budget_file = os.path.join(
            tempfile.gettempdir(), f"ai_scientist_budget_{run_id or 'default'}.sqlite3"
        )
        self.path = Path(
            os.environ.get("AI_SCIENTIST_BUDGET_FILE", default_budget_file)
        )
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS budget ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), "
                "calls INTEGER NOT NULL, input_tokens INTEGER NOT NULL, "
                "output_tokens INTEGER NOT NULL)"
            )
            connection.execute("INSERT OR IGNORE INTO budget VALUES (1, 0, 0, 0)")

    def before_call(self):
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            calls = connection.execute(
                "SELECT calls FROM budget WHERE id=1"
            ).fetchone()[0]
            if self.max_calls is not None and calls >= self.max_calls:
                raise ProviderBudgetError(
                    f"LLM call budget exhausted ({self.max_calls} calls)"
                )
            connection.execute("UPDATE budget SET calls = calls + 1 WHERE id=1")

    def add_usage(self, input_tokens: int, output_tokens: int):
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE budget SET input_tokens = input_tokens + ?, "
                "output_tokens = output_tokens + ? WHERE id=1",
                (input_tokens, output_tokens),
            )
            totals = connection.execute(
                "SELECT input_tokens, output_tokens FROM budget WHERE id=1"
            ).fetchone()
        if self.max_input_tokens is not None and totals[0] > self.max_input_tokens:
            raise ProviderBudgetError(
                f"LLM input token budget exceeded ({self.max_input_tokens} tokens)"
            )
        if self.max_output_tokens is not None and totals[1] > self.max_output_tokens:
            raise ProviderBudgetError(
                f"LLM output token budget exceeded ({self.max_output_tokens} tokens)"
            )

    def snapshot(self) -> dict:
        with self._lock, self._connect() as connection:
            calls, input_tokens, output_tokens = connection.execute(
                "SELECT calls, input_tokens, output_tokens FROM budget WHERE id=1"
            ).fetchone()
        return {
            "calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_calls": self.max_calls,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "database": str(self.path),
        }

    def reset(self):
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE budget SET calls=0, input_tokens=0, output_tokens=0 WHERE id=1"
            )


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ProviderError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ProviderError(f"{name} must be positive")
    return parsed


llm_budget = LLMCallBudget()
_trace_writer: TraceWriter | None = None
_trace_configuration_lock = threading.Lock()


def configure_provider_tracing(config: TraceConfig | None) -> None:
    global _trace_writer
    with _trace_configuration_lock:
        previous = _trace_writer
        replacement = None
        _trace_writer = None
        try:
            replacement = create_trace_writer(config) if config is not None else None
            if previous is not None and previous is not replacement:
                previous.close()
            _trace_writer = replacement
        except BaseException:
            for writer in (replacement, previous):
                if writer is None:
                    continue
                try:
                    writer.close()
                except Exception:
                    pass
            _trace_writer = None
            raise


def _active_trace_writer() -> TraceWriter | None:
    with _trace_configuration_lock:
        return _trace_writer


def provider_tracing_enabled() -> bool:
    return _active_trace_writer() is not None


def _allow_missing_usage() -> bool:
    return os.environ.get("AI_SCIENTIST_ALLOW_MISSING_USAGE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _normalize_function_spec(func_spec: Any) -> FunctionContract | None:
    if func_spec is None:
        return None
    if isinstance(func_spec, Mapping):
        schema = (
            func_spec.get("json_schema")
            or func_spec.get("parameters")
            or func_spec.get("input_schema")
        )
        normalized = FunctionContract(
            name=func_spec.get("name"),
            description=func_spec.get("description", ""),
            json_schema=schema,
        )
    else:
        normalized = FunctionContract(
            name=func_spec.name,
            description=func_spec.description,
            json_schema=func_spec.json_schema,
        )
    if not normalized.name or not isinstance(normalized.json_schema, dict):
        raise StructuredOutputError(
            "Function schema must define a name and object schema"
        )
    return normalized


def get_provider_session(
    model: str | ModelSpec,
    max_retries: int = 2,
) -> ProviderSession:
    spec = model if isinstance(model, ModelSpec) else ModelSpec.parse(model)
    api_key = os.environ.get(spec.api_key_env)
    if not api_key:
        raise ProviderError(f"{spec.provider} requires {spec.api_key_env} to be set")
    kwargs = {
        "api_key": api_key,
        "max_retries": 0,
    }
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    if spec.provider == "openrouter":
        default_headers = {}
        referer = os.environ.get("OPENROUTER_HTTP_REFERER")
        title = os.environ.get("OPENROUTER_APP_TITLE")
        if referer:
            default_headers["HTTP-Referer"] = referer
        if title:
            default_headers["X-OpenRouter-Title"] = title
        if default_headers:
            kwargs["default_headers"] = default_headers
    return ProviderSession(spec=spec, client=openai.OpenAI(**kwargs))


def list_models(model: str | ModelSpec) -> list[dict]:
    session = get_provider_session(model)
    response = _call_with_retry(session.client.models.list, max_retries=2)
    models = []
    for item in response.data:
        if hasattr(item, "model_dump"):
            models.append(item.model_dump())
        elif isinstance(item, dict):
            models.append(item)
        else:
            models.append({"id": getattr(item, "id", str(item))})
    return models


def preflight_model(model: str | ModelSpec) -> dict:
    try:
        session = get_provider_session(model)
        models = list_models(session.spec)
    except Exception as exc:
        return {
            "ok": False,
            "model": model if isinstance(model, str) else model.identifier,
            "error": f"{type(exc).__name__}: {exc}",
        }
    model_ids = {entry.get("id") for entry in models if entry.get("id")}
    model_listed = bool(model_ids) and session.spec.model in model_ids
    result = {
        "ok": False,
        "model": session.spec.identifier,
        "provider": session.spec.provider,
        "api_mode": session.spec.api_mode,
        "base_url": session.spec.base_url,
        "credential_env": session.spec.api_key_env,
        "catalog_size": len(models),
        "model_listed": model_listed,
        "structured_output": False,
        "tool_call_supported": False,
        "structured_mode_used": None,
        "fallback_used": False,
    }
    if not model_listed:
        result["error"] = "Selected model was not present in the provider catalog"
        return result
    result["preflight_attempts"] = 0
    last_error = None
    for attempt, requested_mode in enumerate(("auto", "auto", "json"), start=1):
        result["preflight_attempts"] = attempt
        result["preflight_requested_mode"] = requested_mode
        try:
            smoke = query_model(
                session,
                system_message="Perform a provider capability check.",
                user_message="Set ok to true.",
                func_spec=FunctionContract(
                    name="provider_preflight",
                    description="Confirm structured provider output",
                    json_schema={
                        "type": "object",
                        "properties": {"ok": {"type": "boolean", "const": True}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                ),
                structured_mode=requested_mode,
                max_tokens=64,
                max_retries=1,
            )
            if smoke.output != {"ok": True}:
                raise StructuredOutputError(
                    "Provider capability response did not match the preflight contract"
                )
            result["structured_output"] = True
            result["tool_call_supported"] = bool(smoke.info.get("tool_call_used"))
            result["structured_mode_used"] = smoke.info.get("structured_mode_used")
            result["fallback_used"] = bool(smoke.info.get("fallback_used"))
            result["ok"] = True
            return result
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(0.25 * attempt)
    result["error"] = f"{type(last_error).__name__}: {last_error}"
    return result


def _start_query_trace(
    *,
    spec: ModelSpec,
    messages: list[dict],
    func_spec: FunctionContract | None,
    structured_mode: str,
    max_retries: int,
    request_parameters: dict,
    trace_context: Mapping[str, Any] | None,
) -> _ActiveQueryTrace | None:
    writer = _active_trace_writer()
    if writer is None:
        return None
    active = _ActiveQueryTrace(writer=writer)
    active.started_at = writer.timestamp()
    active.started_perf = time.perf_counter()
    tool_definition = None
    if func_spec is not None:
        tool_definition = {
            "name": func_spec.name,
            "description": func_spec.description,
            "json_schema": func_spec.json_schema,
            "trust": "untrusted_provider_text",
        }
    parameters = {
        **request_parameters,
        "structured_mode": structured_mode,
        "temperature": request_parameters.get("temperature"),
        "max_tokens": request_parameters.get("max_tokens"),
    }
    context = {
        "role": "provider_query",
        "node_id": None,
        "stage_id": None,
        **dict(trace_context or {}),
    }
    try:
        active.call_id = writer.start_provider_call(
            provider=spec.provider,
            model=spec.model,
            api_mode=spec.api_mode,
            base_url=spec.base_url,
            context=context,
            request_parameters=parameters,
            message_count=len(messages),
            max_attempts=(max_retries + 1) * 2,
            tool_definition=tool_definition,
        )
        active.message_references = writer.record_messages(
            active.call_id, messages, "request"
        )
    except BaseException as exc:
        if active.call_id:
            active.finish(status="error", reason="trace_normalization_error", error=exc)
        raise
    return active


def _record_trace_response(
    active: _ActiveQueryTrace | None,
    response: Any,
    output: str | dict,
    normalized_spec: FunctionContract | None,
    *,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    cached_tokens: int,
    usage_missing: bool,
) -> dict:
    if active is None:
        return {}
    tool_calls = _trace_tool_calls(response)
    for tool_call in tool_calls:
        reference = active.writer.record_tool_call(
            active.call_id,
            name=tool_call.get("name"),
            arguments=tool_call.get("arguments", {}),
            origin="untrusted_provider_output",
        )
        active.writer.record_tool_result(
            active.call_id,
            result=None,
            status="not_executed",
            tool_call_id=reference["tool_call_id"],
        )
    if normalized_spec is not None and tool_calls:
        response_message = {
            "role": "assistant",
            "content": [{"type": "structured_json", "value": output}],
        }
    else:
        response_message = {"role": "assistant", "content": output}
    active.writer.record_messages(active.call_id, [response_message], "response")
    return {
        "model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "system_fingerprint": getattr(response, "system_fingerprint", None),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "cached_tokens": cached_tokens,
            "usage_missing": usage_missing,
        },
        "tool_call_count": len(tool_calls),
    }


def _trace_tool_calls(response: Any) -> list[dict]:
    calls = []
    choices = getattr(response, "choices", None)
    if choices:
        raw_calls = getattr(getattr(choices[0], "message", None), "tool_calls", None)
        for item in raw_calls or []:
            function = getattr(item, "function", None)
            if function is not None:
                calls.append(
                    {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", {}),
                    }
                )
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            calls.append(
                {
                    "name": getattr(item, "name", None),
                    "arguments": getattr(item, "arguments", {}),
                }
            )
    return calls


def query_model(
    model: str | ModelSpec | ProviderSession,
    system_message: Any = None,
    user_message: Any = None,
    messages: list[dict] | None = None,
    func_spec: Any = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    structured_mode: str = "auto",
    max_retries: int = 2,
    trace_context: Mapping[str, Any] | None = None,
    **model_kwargs,
) -> QueryResult:
    session = (
        model
        if isinstance(model, ProviderSession)
        else get_provider_session(model, max_retries=max_retries)
    )
    spec = session.spec
    if spec.api_mode not in {"chat_completions", "responses"}:
        raise ProviderError(f"Unsupported API mode {spec.api_mode}")
    if structured_mode not in {"auto", "tools", "json"}:
        raise ProviderError("structured_mode must be auto, tools, or json")
    normalized_messages = _normalize_messages(system_message, user_message, messages)
    normalized_spec = _normalize_function_spec(func_spec)
    request_parameters = {
        **model_kwargs,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    active_trace = _start_query_trace(
        spec=spec,
        messages=normalized_messages,
        func_spec=normalized_spec,
        structured_mode=structured_mode,
        max_retries=max_retries,
        request_parameters=request_parameters,
        trace_context=trace_context,
    )
    trace_observer = active_trace.observe_attempt if active_trace is not None else None
    fallback_used = False
    started = time.perf_counter()
    try:
        try:
            if spec.api_mode == "chat_completions":
                response = _query_chat(
                    session,
                    normalized_messages,
                    normalized_spec,
                    temperature,
                    max_tokens,
                    structured_mode,
                    max_retries,
                    model_kwargs,
                    trace_observer,
                )
            else:
                response = _query_responses(
                    session,
                    normalized_messages,
                    normalized_spec,
                    temperature,
                    max_tokens,
                    structured_mode,
                    max_retries,
                    model_kwargs,
                    trace_observer,
                )
        except openai.APIStatusError as exc:
            if (
                normalized_spec is None
                or structured_mode != "auto"
                or not _is_tool_output_unsupported(exc)
            ):
                raise
            fallback_used = True
            if active_trace is not None:
                active_trace.set_fallback("auto", "json", "structured_tool_unsupported")
            if spec.api_mode == "chat_completions":
                response = _query_chat(
                    session,
                    normalized_messages,
                    normalized_spec,
                    temperature,
                    max_tokens,
                    "json",
                    max_retries,
                    model_kwargs,
                    trace_observer,
                )
            else:
                response = _query_responses(
                    session,
                    normalized_messages,
                    normalized_spec,
                    temperature,
                    max_tokens,
                    "json",
                    max_retries,
                    model_kwargs,
                    trace_observer,
                )
        request_time = time.perf_counter() - started
        input_tokens, output_tokens = _extract_usage(response)
        reasoning_tokens = (
            _nested_value(
                response, "usage", "output_tokens_details", "reasoning_tokens"
            )
            or _nested_value(
                response, "usage", "completion_tokens_details", "reasoning_tokens"
            )
            or 0
        )
        cached_tokens = (
            _nested_value(response, "usage", "input_tokens_details", "cached_tokens")
            or _nested_value(
                response, "usage", "prompt_tokens_details", "cached_tokens"
            )
            or 0
        )
        token_tracker.add_tokens(
            spec.identifier,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cached_tokens,
        )
        usage_missing = input_tokens == 0 and output_tokens == 0
        if (
            usage_missing
            and (
                llm_budget.max_input_tokens is not None
                or llm_budget.max_output_tokens is not None
            )
            and not _allow_missing_usage()
        ):
            raise ProviderError(
                "Provider response did not include billable token usage"
            )
        llm_budget.add_usage(input_tokens, output_tokens)
        output = _extract_output(response, normalized_spec)
        tool_call_used = _has_structured_tool_call(response)
        info = {
            "model": getattr(response, "model", spec.model),
            "response_id": getattr(response, "id", None),
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "api_mode": spec.api_mode,
            "tool_call_used": tool_call_used,
            "structured_mode_used": (
                "tools" if tool_call_used else "json" if normalized_spec else "text"
            ),
            "fallback_used": fallback_used,
            "usage_missing": usage_missing,
        }
        result = QueryResult(
            output=output,
            request_time=request_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            info=info,
        )
        trace_response = _record_trace_response(
            active_trace,
            response,
            output,
            normalized_spec,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens,
            usage_missing=usage_missing,
        )
    except BaseException as exc:
        if active_trace is not None:
            active_trace.finish(
                status="error", reason="provider_termination", error=exc
            )
        raise
    if active_trace is not None:
        active_trace.finish(
            status="success",
            reason="completed",
            response=trace_response,
        )
    return result


def _query_chat(
    session: ProviderSession,
    messages: list[dict],
    func_spec: Any,
    temperature: float | None,
    max_tokens: int | None,
    structured_mode: str,
    max_retries: int,
    model_kwargs: dict,
    trace_observer: Callable[[dict], None] | None = None,
):
    kwargs = _without_none(dict(model_kwargs))
    kwargs["model"] = session.spec.model
    kwargs["messages"] = messages
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None and structured_mode != "json":
        kwargs["temperature"] = temperature
    if func_spec is not None and structured_mode in {"auto", "tools"}:
        kwargs["tools"] = [_chat_tool(func_spec)]
        kwargs["tool_choice"] = _chat_tool_choice(func_spec)
    elif func_spec is not None:
        kwargs["response_format"] = _chat_json_format(func_spec)
        messages = _append_json_instruction(messages, func_spec)
        kwargs["messages"] = messages
    return _call_with_retry(
        session.client.chat.completions.create,
        max_retries=max_retries,
        _trace_observer=trace_observer,
        **kwargs,
    )


def _supports_temperature(spec: ModelSpec) -> bool:
    if spec.provider != "openai":
        return True
    model = spec.model.lower()
    return not model.startswith(("o1", "o3", "gpt-5"))


def _query_responses(
    session: ProviderSession,
    messages: list[dict],
    func_spec: Any,
    temperature: float | None,
    max_tokens: int | None,
    structured_mode: str,
    max_retries: int,
    model_kwargs: dict,
    trace_observer: Callable[[dict], None] | None = None,
):
    system_messages = _system_messages(messages)
    input_messages = _non_system_messages(messages)
    kwargs = _without_none(dict(model_kwargs))
    kwargs["model"] = session.spec.model
    kwargs["input"] = [_response_input_message(message) for message in input_messages]
    instructions = _messages_to_text(system_messages)
    if instructions:
        kwargs["instructions"] = instructions
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if (
        temperature is not None
        and structured_mode != "json"
        and _supports_temperature(session.spec)
    ):
        kwargs["temperature"] = temperature
    if func_spec is not None and structured_mode in {"auto", "tools"}:
        kwargs["tools"] = [_responses_tool(func_spec)]
        kwargs["tool_choice"] = {"type": "function", "name": func_spec.name}
    elif func_spec is not None:
        kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": func_spec.name,
                "schema": func_spec.json_schema,
                "strict": False,
            }
        }
        kwargs["input"] = [
            *_responses_input_messages(input_messages),
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _json_instruction(func_spec),
                    }
                ],
            },
        ]
    return _call_with_retry(
        session.client.responses.create,
        max_retries=max_retries,
        _trace_observer=trace_observer,
        **kwargs,
    )


def _is_tool_output_unsupported(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code not in {400, 404, 422, 501}:
        return False
    message = str(exc).lower()
    markers = (
        "tool",
        "function calling",
        "json_schema",
        "structured output",
        "response_format",
    )
    return any(marker in message for marker in markers)


def _call_with_retry(
    create_fn,
    max_retries: int = 2,
    _trace_observer: Callable[[dict], None] | None = None,
    **kwargs,
):
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    retry_exceptions = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )
    attempts = max_retries + 1
    for attempt_index in range(attempts):
        attempt = attempt_index + 1
        llm_budget.before_call()
        if _trace_observer is not None:
            _trace_observer({"phase": "start", "attempt": attempt})
        try:
            response = create_fn(**kwargs)
        except retry_exceptions as exc:
            if attempt == attempts:
                if _trace_observer is not None:
                    _trace_observer(
                        {
                            "phase": "end",
                            "attempt": attempt,
                            "status": "error",
                            "error_class": type(exc).__name__,
                            "retry": None,
                        }
                    )
                raise
            delay = min(60.0, 2**attempt_index)
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
            retry_after = headers.get("retry-after") if headers is not None else None
            if retry_after:
                try:
                    delay = min(60.0, max(delay, float(retry_after)))
                except ValueError:
                    pass
            sleep_delay = delay + random.uniform(0, min(1.0, delay / 2))
            if _trace_observer is not None:
                _trace_observer(
                    {
                        "phase": "end",
                        "attempt": attempt,
                        "status": "error",
                        "error_class": type(exc).__name__,
                        "retry": {
                            "scheduled": True,
                            "delay_seconds": round(sleep_delay, 6),
                        },
                    }
                )
            time.sleep(sleep_delay)
        except Exception as exc:
            if _trace_observer is not None:
                _trace_observer(
                    {
                        "phase": "end",
                        "attempt": attempt,
                        "status": "error",
                        "error_class": type(exc).__name__,
                        "retry": None,
                    }
                )
            raise
        else:
            if _trace_observer is not None:
                _trace_observer(
                    {
                        "phase": "end",
                        "attempt": attempt,
                        "status": "success",
                        "error_class": None,
                        "retry": None,
                    }
                )
            return response
    raise ProviderError("Provider retry loop exited unexpectedly")


def _message_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _normalize_messages(
    system_message: Any, user_message: Any, messages: list[dict] | None
) -> list[dict]:
    if messages is not None:
        normalized = [dict(message) for message in messages]
    else:
        normalized = []
        if system_message:
            normalized.append(
                {"role": "system", "content": _message_text(system_message)}
            )
        if user_message:
            normalized.append({"role": "user", "content": _message_text(user_message)})
    if normalized and not any(message.get("role") == "user" for message in normalized):
        normalized.append(
            {"role": "user", "content": "Respond to the system instructions."}
        )
    return normalized


def _system_messages(messages: list[dict]) -> list[dict]:
    return [message for message in messages if message.get("role") == "system"]


def _non_system_messages(messages: list[dict]) -> list[dict]:
    return [message for message in messages if message.get("role") != "system"]


def _messages_to_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in {"text", "input_text"}
            )
        else:
            parts.append(str(content))
    return "\n\n".join(part for part in parts if part)


def _response_input_message(message: dict) -> dict:
    result = dict(message)
    if result.get("role") == "system":
        result["role"] = "developer"
    content = result.get("content")
    if isinstance(content, str):
        result["content"] = [{"type": "input_text", "text": content}]
        return result
    if not isinstance(content, list):
        return result
    converted = []
    for item in content:
        if not isinstance(item, dict):
            converted.append(item)
            continue
        item_type = item.get("type")
        if item_type in {"text", "input_text"}:
            converted.append({"type": "input_text", "text": item.get("text", "")})
        elif item_type in {"image_url", "input_image"}:
            image = item.get("image_url")
            if isinstance(image, dict):
                image_url = image.get("url")
            else:
                image_url = image
            converted.append({"type": "input_image", "image_url": image_url})
        else:
            converted.append(item)
    result["content"] = converted
    return result


def _responses_input_messages(messages: list[dict]) -> list[dict]:
    return [_response_input_message(message) for message in messages]


def _append_json_instruction(messages: list[dict], func_spec: Any) -> list[dict]:
    updated = [dict(message) for message in messages]
    updated.append({"role": "user", "content": _json_instruction(func_spec)})
    return updated


def _json_instruction(func_spec: Any) -> str:
    return (
        f"Return one JSON object matching the {func_spec.name} schema. "
        "Do not wrap it in Markdown. Schema: "
        + json.dumps(func_spec.json_schema, sort_keys=True)
    )


def _chat_tool(func_spec: Any) -> dict:
    return {
        "type": "function",
        "function": {
            "name": func_spec.name,
            "description": func_spec.description,
            "parameters": func_spec.json_schema,
        },
    }


def _chat_tool_choice(func_spec: Any) -> dict:
    return {"type": "function", "function": {"name": func_spec.name}}


def _chat_json_format(func_spec: Any) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": func_spec.name,
            "schema": func_spec.json_schema,
            "strict": False,
        },
    }


def _responses_tool(func_spec: Any) -> dict:
    return {
        "type": "function",
        "name": func_spec.name,
        "description": func_spec.description,
        "parameters": func_spec.json_schema,
    }


def _extract_output(response: Any, func_spec: Any) -> str | dict:
    if func_spec is None:
        output = _extract_text(response)
        if not isinstance(output, str) or not output.strip():
            raise ProviderError("Provider returned an empty text response")
        return output
    output = _extract_structured_output(response)
    try:
        jsonschema.validate(output, func_spec.json_schema)
    except (jsonschema.ValidationError, jsonschema.SchemaError) as exc:
        raise StructuredOutputError(
            "Provider response did not match the schema"
        ) from exc
    return output


def _extract_text(response: Any) -> str | None:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct:
        return direct
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
    chunks = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", None) or []:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks) if chunks else None


def _has_structured_tool_call(response: Any) -> bool:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        if getattr(message, "tool_calls", None):
            return True
    return any(
        getattr(item, "type", None) == "function_call"
        for item in getattr(response, "output", None) or []
    )


def _extract_structured_output(response: Any) -> dict:
    choices = getattr(response, "choices", None)
    if choices:
        tool_calls = getattr(getattr(choices[0], "message", None), "tool_calls", None)
        if tool_calls:
            function = tool_calls[0].function
            return _decode_json(function.arguments)
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "function_call":
            return _decode_json(getattr(item, "arguments", ""))
    text = _extract_text(response)
    if text:
        return _decode_json(text)
    raise StructuredOutputError("Provider did not return a structured response")


def _decode_json(value: str) -> dict:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, re.DOTALL)
        if not match:
            raise StructuredOutputError("Structured response was not valid JSON")
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                "Structured response was not valid JSON"
            ) from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputError("Structured response must be a JSON object")
    return parsed


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = _object_value(response, "usage")
    if usage is None:
        return 0, 0
    input_tokens = _first_value(
        _object_value(usage, "input_tokens"),
        _object_value(usage, "prompt_tokens"),
    )
    output_tokens = _first_value(
        _object_value(usage, "output_tokens"),
        _object_value(usage, "completion_tokens"),
    )
    return int(input_tokens or 0), int(output_tokens or 0)


def _first_value(*values):
    return next((value for value in values if value is not None), None)


def _object_value(obj: Any, key: str):
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _nested_value(obj: Any, *path: str):
    current = obj
    for key in path:
        current = _object_value(current, key)
        if current is None:
            return None
    return current


def _without_none(values: dict) -> dict:
    return {key: value for key, value in values.items() if value is not None}
