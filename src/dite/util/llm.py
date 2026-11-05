"""LLM request helpers."""

import json
import re
from typing import Any
from urllib.parse import urlsplit

from openai import APIError, AsyncOpenAI, OpenAI

from dite.app.config import ChatCompletionProfileConfig

# Hosts whose OpenAI-compatible API takes top-level ``enable_thinking`` and
# ``thinking_budget`` fields. DashScope also rejects non-streaming Qwen3
# calls unless thinking is switched off explicitly.
_ENABLE_THINKING_DOMAINS = ("siliconflow.cn", "siliconflow.com", "modelscope.cn")
_DASHSCOPE_HOST_RE = re.compile(r"dashscope(?:-[a-z0-9]+)?\.aliyuncs\.com")

_REASONING_BLOCK_RE = re.compile(
    r"<think(?:ing)?\s*>.*?</think(?:ing)?\s*>", re.IGNORECASE | re.DOTALL
)
_REASONING_OPEN_RE = re.compile(r"<think(?:ing)?\s*>", re.IGNORECASE)
_REASONING_CLOSE_RE = re.compile(r"</think(?:ing)?\s*>", re.IGNORECASE)


def strip_reasoning_text(content: str | None) -> str:
    """Return a model reply without its ``<think>`` reasoning.

    Servers without a reasoning parser (vLLM or SGLang started without
    ``--reasoning-parser``, for example) put Qwen3 / DeepSeek-R1 reasoning
    into ``content``. A reply cut off by ``max_tokens`` ends inside an
    unclosed ``<think>``, and chat templates that open the block in the
    prompt leave only ``</think>``; reasoning is dropped in every case.

    Reasoning always comes before the answer, so only tags that lead the
    reply count: an answer that quotes ``<think>`` (a topic about prompt
    formats, say) is kept as it is.
    """
    text = (content or "").lstrip()
    opened_in_reply = False
    while block := _REASONING_BLOCK_RE.match(text):
        text = text[block.end() :].lstrip()
        opened_in_reply = True
    if _REASONING_OPEN_RE.match(text):
        # Cut off by max_tokens inside the reasoning: there is no answer yet.
        return ""
    if not opened_in_reply:
        closing_tag = _REASONING_CLOSE_RE.search(text)
        if closing_tag and not _REASONING_OPEN_RE.search(text, 0, closing_tag.start()):
            text = text[closing_tag.end() :]
    return text.strip()


def _client_host(client: OpenAI | AsyncOpenAI) -> str:
    base_url = str(getattr(client, "base_url", "") or "")
    if "://" not in base_url:
        base_url = f"//{base_url}"
    try:
        return (urlsplit(base_url).hostname or "").casefold()
    except ValueError:
        return ""


def _resolve_reasoning_param_style(
    client: OpenAI | AsyncOpenAI, reasoning_param_style: str
) -> str:
    """Pick how thinking controls are sent; ``auto`` only trusts known hosts."""
    if reasoning_param_style != "auto":
        return reasoning_param_style
    # Earlier releases matched "siliconflow.cn" anywhere in base_url, so a
    # proxy URL that names it keeps getting the field.
    if "siliconflow.cn" in str(getattr(client, "base_url", "") or "").casefold():
        return "enable_thinking"
    host = _client_host(client)
    if _DASHSCOPE_HOST_RE.fullmatch(host) or any(
        host == domain or host.endswith(f".{domain}")
        for domain in _ENABLE_THINKING_DOMAINS
    ):
        return "enable_thinking"
    # Unknown fields make strict servers (api.openai.com among them) answer
    # 400, so other hosts get nothing unless the user picks a style.
    return "none"


def _translate_reasoning_controls(
    client: OpenAI | AsyncOpenAI,
    profile: ChatCompletionProfileConfig,
    reasoning_param_style: str = "auto",
) -> dict[str, Any]:
    """Translate generic reasoning controls into provider-specific request fields."""
    if profile.reasoning_mode == "default" and profile.thinking_budget is None:
        return {}

    style = _resolve_reasoning_param_style(client, reasoning_param_style)
    if style == "enable_thinking":
        extra_body: dict[str, Any] = {}
        if profile.reasoning_mode != "default":
            extra_body["enable_thinking"] = profile.reasoning_mode == "on"
        if profile.thinking_budget is not None:
            extra_body["thinking_budget"] = profile.thinking_budget
        return extra_body

    if style == "chat_template_kwargs" and profile.reasoning_mode != "default":
        # Qwen3 chat templates read enable_thinking; they have no budget knob.
        return {
            "chat_template_kwargs": {
                "enable_thinking": profile.reasoning_mode == "on",
            }
        }

    return {}


def build_chat_completion_kwargs(
    *,
    client: OpenAI | AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    profile: ChatCompletionProfileConfig | None = None,
    reasoning_param_style: str = "auto",
    **overrides: Any,
) -> dict[str, Any]:
    """Build chat completion kwargs from task intent plus provider adaptation.

    ``reasoning_param_style`` is ``config.api.reasoning_param_style``.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **overrides,
    }

    if profile is None:
        return kwargs

    kwargs.setdefault("max_tokens", profile.max_tokens)

    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body.update(
        _translate_reasoning_controls(client, profile, reasoning_param_style)
    )
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

    ``<think>`` reasoning is dropped first. Returns ``None`` when no JSON
    object can be recovered.
    """
    text = strip_reasoning_text(content)
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
