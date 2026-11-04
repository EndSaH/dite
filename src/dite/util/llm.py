"""LLM request helpers."""

import json
from typing import Any

from openai import APIError, AsyncOpenAI, OpenAI

from dite.app.config import ChatCompletionProfileConfig


def _translate_reasoning_controls(
    client: OpenAI | AsyncOpenAI,
    profile: ChatCompletionProfileConfig,
) -> dict[str, Any]:
    """Translate generic reasoning controls into provider-specific request fields."""
    if profile.reasoning_mode == "default" and profile.thinking_budget is None:
        return {}

    base_url = str(getattr(client, "base_url", "")).casefold()
    if "siliconflow.cn" in base_url:
        extra_body: dict[str, Any] = {}
        if profile.reasoning_mode != "default":
            extra_body["enable_thinking"] = profile.reasoning_mode == "on"
        if profile.thinking_budget is not None:
            extra_body["thinking_budget"] = profile.thinking_budget
        return extra_body

    return {}


def build_chat_completion_kwargs(
    *,
    client: OpenAI | AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    profile: ChatCompletionProfileConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build chat completion kwargs from task intent plus provider adaptation."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **overrides,
    }

    if profile is None:
        return kwargs

    kwargs.setdefault("max_tokens", profile.max_tokens)

    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body.update(_translate_reasoning_controls(client, profile))
    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


def format_api_error(exc: APIError) -> str:
    """Extract the highest-signal fields from provider API errors."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("message") or getattr(exc, "message", None) or str(exc)
        details = []

        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            details.append(f"status={status_code}")

        code = body.get("code") or getattr(exc, "code", None)
        if code not in (None, ""):
            details.append(f"code={code}")

        error_type = body.get("type") or getattr(exc, "type", None)
        if error_type:
            details.append(f"type={error_type}")

        param = body.get("param") or getattr(exc, "param", None)
        if param:
            details.append(f"param={param}")

        request_id = getattr(exc, "request_id", None)
        if request_id:
            details.append(f"request_id={request_id}")

        if details:
            return f"{message} ({', '.join(details)})"
        return str(message)

    return str(exc)


def should_retry_api_error(exc: APIError) -> bool:
    """Return whether the provider error is worth retrying."""
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    error_code = (
        body.get("code") if isinstance(body, dict) else getattr(exc, "code", None)
    )

    if status_code is None:
        return True
    if status_code >= 500:
        return True
    return error_code in {50507}


def parse_json_object(content: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model reply, tolerating code fences and prose.

    Returns ``None`` when no JSON object can be recovered.
    """
    text = (content or "").strip()
    if not text:
        return None
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def parse_json_bool(value: Any) -> bool:
    """Read a model-provided boolean; the string ``"false"`` means False."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "yes", "y", "1", "是"}
    return False
