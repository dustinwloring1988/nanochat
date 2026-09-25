import json
import multiprocessing
import os
from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")
pytest.importorskip("jsonschema")

from ai_scientist.providers import (
    LLMCallBudget,
    ModelSpec,
    ProviderBudgetError,
    ProviderError,
    ProviderSession,
    QueryResult,
    StructuredOutputError,
    llm_budget,
    preflight_model,
    query_model,
)
from ai_scientist.treesearch.backend.utils import FunctionSpec
from ai_scientist.utils.token_tracker import token_tracker


class FakeChatCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeResponses:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, chat_response=None, responses_response=None):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(chat_response))
        self.responses = FakeResponses(responses_response)


def usage(prompt_key="prompt_tokens", completion_key="completion_tokens"):
    return SimpleNamespace(
        **{
            prompt_key: 3,
            completion_key: 4,
            "prompt_tokens_details": SimpleNamespace(cached_tokens=1),
            "completion_tokens_details": SimpleNamespace(reasoning_tokens=2),
        }
    )


def chat_response(content="ok", tool_arguments=None):
    tool_calls = None
    if tool_arguments is not None:
        tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(
                    name="submit",
                    arguments=json.dumps(tool_arguments),
                )
            )
        ]
    return SimpleNamespace(
        id="chat-1",
        model="space-bunny-free",
        system_fingerprint="fp",
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
        id="responses-1",
        model="gpt-test",
        system_fingerprint=None,
        usage=usage("input_tokens", "output_tokens"),
        output=output,
        output_text=output_text,
    )


def _consume_shared_budget(path, queue):
    os.environ["AI_SCIENTIST_BUDGET_FILE"] = str(path)
    os.environ["AI_SCIENTIST_MAX_API_CALLS"] = "2"
    budget = LLMCallBudget()
    try:
        budget.before_call()
        queue.put("ok")
    except ProviderBudgetError:
        queue.put("exhausted")


@pytest.fixture(autouse=True)
def reset_global_state():
    llm_budget.reset()
    token_tracker.reset()
    yield
    llm_budget.reset()
    token_tracker.reset()


def test_parse_space_bunny():
    spec = ModelSpec.parse("opencode/space-bunny-free")
    assert spec.provider == "opencode"
    assert spec.model == "space-bunny-free"
    assert spec.api_mode == "chat_completions"
    assert spec.base_url == "https://opencode.ai/zen/v1"
    assert spec.api_key_env == "OPENCODE_API_KEY"


def test_parse_zen_responses_model():
    spec = ModelSpec.parse("opencode/gpt-5.6")
    assert spec.api_mode == "responses"


def test_parse_openai_model():
    spec = ModelSpec.parse("openai/gpt-test")
    assert spec.provider == "openai"
    assert spec.api_mode == "responses"
    assert spec.base_url is None


def test_parse_openrouter_model_preserves_vendor_path():
    spec = ModelSpec.parse("openrouter/vendor/model-id")
    assert spec.model == "vendor/model-id"
    assert spec.api_mode == "chat_completions"


def test_missing_credential_fails(monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENCODE_API_KEY"):
        query_model("opencode/space-bunny-free", user_message="hello")


def test_chat_text_query():
    client = FakeClient(chat_response=chat_response("hello"))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    result = query_model(
        session,
        system_message="system",
        user_message="user",
        temperature=0.5,
        max_tokens=123,
    )
    assert result.output == "hello"
    assert result.input_tokens == 3
    assert result.output_tokens == 4
    call = client.chat.completions.calls[0]
    assert call["model"] == "space-bunny-free"
    assert call["temperature"] == 0.5
    assert call["max_tokens"] == 123
    assert token_tracker.get_interactions() == {}


def test_mapping_prompts_are_serialized_for_chat_completions():
    client = FakeClient(chat_response=chat_response("ok"))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    result = query_model(
        session,
        system_message={"Instructions": ["keep it concise"]},
        user_message={"task": "return ok"},
    )
    assert result.output == "ok"
    sent_messages = client.chat.completions.calls[0]["messages"]
    assert isinstance(sent_messages[0]["content"], str)
    assert isinstance(sent_messages[1]["content"], str)
    assert "keep it concise" in sent_messages[0]["content"]


def test_system_only_request_gets_a_user_turn():
    client = FakeClient(chat_response=chat_response("ok"))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    result = query_model(session, system_message="Return a short answer.")
    assert result.output == "ok"
    sent_messages = client.chat.completions.calls[0]["messages"]
    assert [message["role"] for message in sent_messages] == ["system", "user"]


def test_responses_text_query_sends_supported_temperature():
    client = FakeClient(responses_response=responses_response("hello"))
    session = ProviderSession(ModelSpec.parse("openai/gpt-test"), client)
    result = query_model(
        session,
        system_message="system",
        user_message="user",
        temperature=0.5,
        max_tokens=321,
    )
    assert result.output == "hello"
    call = client.responses.calls[0]
    assert call["max_output_tokens"] == 321
    assert call["temperature"] == 0.5
    assert call["instructions"] == "system"


def test_responses_reasoning_model_omits_temperature():
    client = FakeClient(responses_response=responses_response("hello"))
    session = ProviderSession(ModelSpec.parse("openai/gpt-5-test"), client)
    query_model(session, user_message="hello", temperature=0.5)
    assert "temperature" not in client.responses.calls[0]


def test_responses_input_normalizes_multimodal_content():
    client = FakeClient(responses_response=responses_response("seen"))
    session = ProviderSession(ModelSpec.parse("openai/gpt-test"), client)
    query_model(
        session,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ],
            }
        ],
    )
    content = client.responses.calls[0]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "inspect"}
    assert content[1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,AA==",
    }


def function_spec():
    return FunctionSpec(
        name="submit",
        description="Submit a result",
        json_schema={
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )


def test_chat_tool_call_is_validated():
    client = FakeClient(chat_response=chat_response(tool_arguments={"value": 1.5}))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    result = query_model(session, user_message="submit", func_spec=function_spec())
    assert result.output == {"value": 1.5}
    call = client.chat.completions.calls[0]
    assert call["tool_choice"]["function"]["name"] == "submit"


def test_mapping_function_spec_is_normalized():
    client = FakeClient(chat_response=chat_response(tool_arguments={"value": 3.5}))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    result = query_model(
        session,
        user_message="submit",
        func_spec={
            "name": "submit",
            "description": "Submit a result",
            "parameters": function_spec().json_schema,
        },
    )
    assert result.output == {"value": 3.5}


def test_responses_tool_call_is_validated():
    client = FakeClient(
        responses_response=responses_response(function_arguments={"value": 2.5})
    )
    session = ProviderSession(ModelSpec.parse("openai/gpt-test"), client)
    result = query_model(session, user_message="submit", func_spec=function_spec())
    assert result.output == {"value": 2.5}
    call = client.responses.calls[0]
    assert call["tool_choice"] == {"type": "function", "name": "submit"}


def test_malformed_and_schema_invalid_structured_output_fail_closed():
    client = FakeClient(chat_response=chat_response("not-json"))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    with pytest.raises(StructuredOutputError, match="valid JSON"):
        query_model(session, user_message="submit", func_spec=function_spec())

    client = FakeClient(chat_response=chat_response(tool_arguments={"value": "bad"}))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    with pytest.raises(StructuredOutputError, match="match the schema"):
        query_model(session, user_message="submit", func_spec=function_spec())


def test_structured_json_fallback_is_reported():
    client = FakeClient(chat_response=chat_response('{"value": 1.5}'))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    result = query_model(
        session,
        user_message="submit",
        func_spec=function_spec(),
        structured_mode="json",
    )
    assert result.output == {"value": 1.5}
    assert result.info["structured_mode_used"] == "json"
    assert "response_format" in client.chat.completions.calls[0]


def test_responses_json_fallback_is_reported():
    client = FakeClient(responses_response=responses_response('{"value": 1.5}'))
    session = ProviderSession(ModelSpec.parse("openai/gpt-test"), client)
    result = query_model(
        session,
        user_message="submit",
        func_spec=function_spec(),
        structured_mode="json",
    )
    assert result.output == {"value": 1.5}
    assert result.info["structured_mode_used"] == "json"
    assert client.responses.calls[0]["text"]["format"]["type"] == "json_schema"


def test_retry_count_and_budget_reservation(monkeypatch):
    class FlakyCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise openai.APIConnectionError(request=SimpleNamespace())
            return chat_response("ok")

    flaky = FlakyCompletions()
    client = FakeClient()
    client.chat.completions = flaky
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    monkeypatch.setattr("ai_scientist.providers.time.sleep", lambda _: None)
    monkeypatch.setattr("ai_scientist.providers.random.uniform", lambda *_: 0)
    result = query_model(session, user_message="hello", max_retries=2)
    assert result.output == "ok"
    assert flaky.calls == 3
    assert llm_budget.snapshot()["calls"] == 3


def test_retry_exhaustion_stops_after_configured_attempts(monkeypatch):
    class AlwaysFail:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            raise openai.APIConnectionError(request=SimpleNamespace())

    always_fail = AlwaysFail()
    client = FakeClient()
    client.chat.completions = always_fail
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    monkeypatch.setattr("ai_scientist.providers.time.sleep", lambda _: None)
    monkeypatch.setattr("ai_scientist.providers.random.uniform", lambda *_: 0)
    with pytest.raises(openai.APIConnectionError):
        query_model(session, user_message="hello", max_retries=2)
    assert always_fail.calls == 3


def test_input_token_budget_exhaustion_fails_closed(monkeypatch):
    monkeypatch.setenv("AI_SCIENTIST_MAX_INPUT_TOKENS", "1")
    budget = type(llm_budget)()
    client = FakeClient(chat_response=chat_response("ok"))
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    monkeypatch.setattr("ai_scientist.providers.llm_budget", budget)
    with pytest.raises(ProviderBudgetError, match="input token budget"):
        query_model(session, user_message="hello")


def test_preflight_reports_structured_capability(monkeypatch):
    session = ProviderSession(
        ModelSpec.parse("opencode/space-bunny-free"), FakeClient()
    )
    monkeypatch.setattr(
        "ai_scientist.providers.get_provider_session", lambda model: session
    )
    monkeypatch.setattr(
        "ai_scientist.providers.list_models", lambda spec: [{"id": "space-bunny-free"}]
    )
    monkeypatch.setattr(
        "ai_scientist.providers.query_model",
        lambda *args, **kwargs: QueryResult(
            output={"ok": True},
            request_time=0.1,
            input_tokens=1,
            output_tokens=1,
            info={
                "tool_call_used": True,
                "structured_mode_used": "tools",
                "fallback_used": False,
            },
        ),
    )
    result = preflight_model("opencode/space-bunny-free")
    assert result["ok"] is True
    assert result["tool_call_supported"] is True
    assert result["structured_mode_used"] == "tools"
    assert result["fallback_used"] is False


def test_preflight_catalog_failure_is_structured(monkeypatch):
    monkeypatch.setattr(
        "ai_scientist.providers.get_provider_session",
        lambda model: (_ for _ in ()).throw(ProviderError("catalog unavailable")),
    )
    result = preflight_model("opencode/space-bunny-free")
    assert result["ok"] is False
    assert "catalog unavailable" in result["error"]


def test_call_budget_fails_closed(monkeypatch):
    monkeypatch.setenv("AI_SCIENTIST_MAX_API_CALLS", "1")
    budget = type(llm_budget)()
    client = FakeClient(chat_response=chat_response())
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    monkeypatch.setattr("ai_scientist.providers.llm_budget", budget)
    query_model(session, user_message="first")
    with pytest.raises(ProviderBudgetError, match="budget exhausted"):
        query_model(session, user_message="second")


def test_missing_usage_fails_closed_when_budgeted(monkeypatch):
    monkeypatch.setenv("AI_SCIENTIST_MAX_INPUT_TOKENS", "10")
    budget = LLMCallBudget()
    response = chat_response()
    response.usage = None
    client = FakeClient(chat_response=response)
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    monkeypatch.setattr("ai_scientist.providers.llm_budget", budget)
    with pytest.raises(ProviderError, match="did not include"):
        query_model(session, user_message="hello")


def test_missing_usage_can_be_explicitly_allowed(monkeypatch):
    monkeypatch.setenv("AI_SCIENTIST_MAX_INPUT_TOKENS", "10")
    monkeypatch.setenv("AI_SCIENTIST_ALLOW_MISSING_USAGE", "1")
    budget = type(llm_budget)()
    response = chat_response()
    response.usage = None
    client = FakeClient(chat_response=response)
    session = ProviderSession(ModelSpec.parse("opencode/space-bunny-free"), client)
    monkeypatch.setattr("ai_scientist.providers.llm_budget", budget)
    result = query_model(session, user_message="hello")
    assert result.output == "ok"
    assert result.info["usage_missing"] is True


def test_call_budget_is_shared_across_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    budget_path = tmp_path / "budget.sqlite3"
    processes = [
        context.Process(target=_consume_shared_budget, args=(budget_path, queue))
        for _ in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = [queue.get(timeout=1) for _ in processes]
    assert outcomes.count("ok") == 2
    assert outcomes.count("exhausted") == 2
