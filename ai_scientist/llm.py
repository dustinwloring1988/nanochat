import json
import re
from typing import Any

from ai_scientist.providers import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    ModelSpec,
    ProviderSession,
    get_provider_session,
    query_model,
)

MAX_NUM_TOKENS = DEFAULT_MAX_TOKENS
AVAILABLE_LLMS = [DEFAULT_MODEL]


def _resolve_session(client: Any, model: str) -> ProviderSession:
    if isinstance(client, ProviderSession):
        return client
    return get_provider_session(model)


def make_llm_call(
    client: Any,
    model: str,
    temperature: float | None,
    system_message: Any,
    prompt: list[dict],
):
    session = _resolve_session(client, model)
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.extend(prompt)
    return query_model(
        session,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
    ).output


def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    history = list(msg_history or [])
    contents = []
    histories = []
    for _ in range(n_responses):
        content, updated_history = get_response_from_llm(
            prompt,
            client,
            model,
            system_message,
            print_debug=print_debug,
            msg_history=history,
            temperature=temperature,
        )
        contents.append(content)
        histories.append(updated_history)
    return contents, histories


def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    session = _resolve_session(client, model)
    history = [dict(message) for message in (msg_history or [])]
    messages = [
        *({"role": "system", "content": system_message} if system_message else []),
        *history,
        {"role": "user", "content": prompt},
    ]
    result = query_model(
        session,
        messages=messages,
        temperature=temperature,
        max_tokens=MAX_NUM_TOKENS,
    )
    if not isinstance(result.output, str):
        raise TypeError("Text generation returned a structured response")
    updated_history = [
        *history,
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": result.output},
    ]
    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for message in updated_history:
            print(f'{message["role"]}: {message["content"]}')
        print(result.output)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return result.output, updated_history


def extract_json_between_markers(llm_output: str) -> dict | None:
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)
    if not matches:
        matches = re.findall(r"\{.*?\}", llm_output, re.DOTALL)
    for json_string in matches:
        try:
            parsed = json.loads(json_string.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            cleaned = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return None


def create_client(model: str) -> tuple[ProviderSession, str]:
    session = get_provider_session(model)
    return session, session.spec.identifier
