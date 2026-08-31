"""Thin OpenAI-compatible LLM provider used by retrieval nodes."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

if TYPE_CHECKING:
    from src.config import Settings


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM cannot produce a usable response."""


class LLMTransientError(LLMProviderError):
    """Raised for temporary provider failures that are safe to retry unchanged."""


class LLMResponseError(LLMProviderError):
    """Raised when the LLM response is empty or structurally unusable."""


def _is_transient_status(error: APIStatusError) -> bool:
    return error.status_code in {408, 409, 425, 429} or error.status_code >= 500


def _required_setting(name: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise LLMProviderError(f"Missing required setting: {name}")
    return value


def _normalize_base_url(value: str) -> str:
    """Accept either an OpenAI API root or a full chat-completions endpoint."""
    normalized = value.rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    if not normalized:
        raise LLMProviderError("LLM_BASE_URL is invalid")
    return normalized


@lru_cache(maxsize=8)
def get_llm_client(settings: Settings) -> OpenAI:
    """Create one lazily initialized OpenAI-compatible client."""
    try:
        return OpenAI(
            base_url=_normalize_base_url(
                _required_setting("LLM_BASE_URL", settings.llm_base_url)
            ),
            api_key=settings.llm_api_key or "dummy",
            timeout=settings.llm_timeout,
        )
    except LLMProviderError:
        raise
    except (TypeError, ValueError) as exc:
        raise LLMProviderError("Invalid LLM configuration") from exc


def get_llm_model(settings: Settings) -> str:
    """Return the model served by the configured endpoint."""
    return _required_setting("LLM_MODEL", settings.llm_model)


def generate(
    prompt: str, *, settings: Settings, system_prompt: str | None = None
) -> str:
    """Generate text without relying on provider-specific extensions."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = get_llm_client(settings).chat.completions.create(
            model=get_llm_model(settings),
            messages=messages,
            temperature=settings.llm_temperature,
        )
        content = response.choices[0].message.content
    except (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        json.JSONDecodeError,
    ) as exc:
        raise LLMTransientError("Temporary LLM request failure") from exc
    except APIStatusError as exc:
        if _is_transient_status(exc):
            raise LLMTransientError("Temporary LLM request failure") from exc
        raise LLMProviderError("LLM request failed") from exc
    except LLMProviderError:
        raise
    except Exception as exc:
        raise LLMProviderError("LLM request failed") from exc

    if not content:
        raise LLMResponseError("LLM returned an empty response")
    return content.strip()


def _native_json_completion(
    prompt: str,
    *,
    settings: Settings,
    system_prompt: str | None = None,
    json_schema: Mapping[str, Any] | None = None,
    sdk_max_retries: int | None = None,
) -> Any:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response_format: dict[str, Any] = {"type": "json_object"}
    if json_schema is not None:
        response_format = {
            "type": "json_schema",
            "json_schema": dict(json_schema),
        }
    client = get_llm_client(settings)
    if sdk_max_retries is not None:
        client = client.with_options(max_retries=sdk_max_retries)
    return client.chat.completions.create(
        model=get_llm_model(settings),
        messages=messages,
        temperature=settings.llm_temperature,
        response_format=response_format,
    )


def _structured_text_completion(
    prompt: str,
    *,
    settings: Settings,
    system_prompt: str | None = None,
    sdk_max_retries: int | None = None,
) -> Any:
    """Complete a structured prompt without a provider response format."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    client = get_llm_client(settings)
    if sdk_max_retries is not None:
        client = client.with_options(max_retries=sdk_max_retries)
    return client.chat.completions.create(
        model=get_llm_model(settings),
        messages=messages,
        temperature=settings.llm_temperature,
    )


def _generate_structured_text(
    prompt: str,
    *,
    settings: Settings,
    system_prompt: str | None = None,
    sdk_max_retries: int | None = None,
) -> str:
    """Generate prompt-constrained JSON while honoring structured-call retries."""
    try:
        response = _structured_text_completion(
            prompt,
            settings=settings,
            system_prompt=system_prompt,
            sdk_max_retries=sdk_max_retries,
        )
        content = response.choices[0].message.content
    except (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        json.JSONDecodeError,
    ) as exc:
        raise LLMTransientError("Temporary LLM request failure") from exc
    except APIStatusError as exc:
        if _is_transient_status(exc):
            raise LLMTransientError("Temporary LLM request failure") from exc
        raise LLMProviderError("LLM request failed") from exc
    except LLMProviderError:
        raise
    except Exception as exc:
        raise LLMProviderError("LLM request failed") from exc

    if not content:
        raise LLMResponseError("LLM returned an empty response")
    return content.strip()


def _response_format_is_unsupported(error: APIStatusError) -> bool:
    error_text = f"{error} {error.body}".casefold()
    return error.status_code in {400, 422} and any(
        marker in error_text
        for marker in (
            "response_format",
            "json_schema",
            "json object",
            "extra inputs are not permitted",
        )
    )


def _generate_native_json(
    prompt: str,
    *,
    settings: Settings,
    system_prompt: str | None = None,
    json_schema: Mapping[str, Any] | None = None,
    sdk_max_retries: int | None = None,
) -> str:
    try:
        response = _native_json_completion(
            prompt,
            settings=settings,
            system_prompt=system_prompt,
            json_schema=json_schema,
            sdk_max_retries=sdk_max_retries,
        )
        content = response.choices[0].message.content
    except (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        json.JSONDecodeError,
    ) as exc:
        raise LLMTransientError("Temporary LLM request failure") from exc
    except APIStatusError as exc:
        if _response_format_is_unsupported(exc):
            return _generate_structured_text(
                prompt,
                settings=settings,
                system_prompt=system_prompt,
                sdk_max_retries=sdk_max_retries,
            )
        if _is_transient_status(exc):
            raise LLMTransientError("Temporary LLM request failure") from exc
        raise LLMProviderError("LLM request failed") from exc
    except LLMProviderError:
        raise
    except Exception as exc:
        raise LLMProviderError("LLM request failed") from exc

    if not content:
        raise LLMResponseError("LLM returned an empty response")
    return content.strip()


def generate_structured(
    prompt: str,
    *,
    settings: Settings,
    system_prompt: str | None = None,
    native: bool = True,
    json_schema: Mapping[str, Any] | None = None,
    sdk_max_retries: int | None = None,
    invalid_json_parser: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Request JSON, preferring native mode with a prompt-only fallback."""
    content = (
        _generate_native_json(
            prompt,
            settings=settings,
            system_prompt=system_prompt,
            json_schema=json_schema,
            sdk_max_retries=sdk_max_retries,
        )
        if native
        else _generate_structured_text(
            prompt,
            settings=settings,
            system_prompt=system_prompt,
            sdk_max_retries=sdk_max_retries,
        )
    )
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
    if fenced:
        content = fenced.group(1)
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        if invalid_json_parser is None:
            raise LLMResponseError("LLM response is not valid JSON") from exc
        try:
            value = dict(invalid_json_parser(content))
        except (TypeError, ValueError) as repair_error:
            raise LLMResponseError("LLM response is not valid JSON") from repair_error
    if not isinstance(value, dict):
        raise LLMResponseError("LLM JSON response must be an object")
    return value
