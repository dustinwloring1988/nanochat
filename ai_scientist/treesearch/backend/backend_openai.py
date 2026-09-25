from ai_scientist.providers import get_provider_session, query_model

from .utils import FunctionSpec, OutputType


def get_ai_client(model: str, max_retries: int = 2):
    return get_provider_session(model, max_retries=max_retries)


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    result = query_model(
        model_kwargs.get("model"),
        system_message=system_message,
        user_message=user_message,
        func_spec=func_spec,
        temperature=model_kwargs.get("temperature"),
        max_tokens=model_kwargs.get("max_tokens"),
        **{
            key: value
            for key, value in model_kwargs.items()
            if key not in {"model", "temperature", "max_tokens"}
        },
    )
    return (
        result.output,
        result.request_time,
        result.input_tokens,
        result.output_tokens,
        result.info,
    )
