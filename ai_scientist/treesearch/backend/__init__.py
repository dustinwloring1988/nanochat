from ai_scientist.providers import (
    ModelSpec,
    ProviderSession,
    get_provider_session,
    query_model,
)

from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md


def get_ai_client(model: str, **model_kwargs) -> ProviderSession:
    return get_provider_session(model, **model_kwargs)


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    compiled_system = compile_prompt_to_md(system_message) if system_message else None
    compiled_user = compile_prompt_to_md(user_message) if user_message else None
    return query_model(
        model,
        system_message=compiled_system,
        user_message=compiled_user,
        func_spec=func_spec,
        temperature=temperature,
        max_tokens=max_tokens,
        **model_kwargs,
    ).output
