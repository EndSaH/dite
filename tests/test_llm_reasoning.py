"""Reasoning (``<think>``) handling in LLM replies and thinking controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from openai import OpenAI

from dite.app.config import (
    APIConfig,
    ChatCompletionProfileConfig,
    Config,
    load_config,
)
from dite.app.i18n import set_locale
from dite.cluster.model import ClusterMetrics, ClusterResult
from dite.doc import DocumentFeatures
from dite.util.api import ChatCompletionResult
from dite.util.llm import build_chat_completion_kwargs, parse_json_object
from dite.util.log import setup_logging

OFF_PROFILE = ChatCompletionProfileConfig(max_tokens=50, reasoning_mode="off")


@pytest.fixture(autouse=True)
def _english_quiet_logs():
    set_locale("en")
    setup_logging()
    yield
    set_locale("en")
    setup_logging()


def _flat(text: str) -> str:
    """Undo Rich line wrapping so assertions can match whole sentences."""
    return " ".join(text.split())


def _cluster_result(labels: list[int]) -> ClusterResult:
    labels_array = np.array(labels, dtype=int)
    return ClusterResult(
        labels=labels_array,
        cluster_names={},
        repaired_mask=np.zeros(labels_array.shape, dtype=bool),
        metrics=ClusterMetrics(),
    )


class _Endpoint:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


class _ReplyClient:
    """Sync chat client returning canned replies (or raising)."""

    def __init__(
        self,
        replies: list[str] | str,
        *,
        base_url: str = "http://localhost:8000/v1",
        error: Exception | None = None,
        finish_reason: str = "stop",
    ) -> None:
        self.base_url = base_url
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if error is not None:
                    raise error
                if isinstance(replies, str):
                    content = replies
                else:
                    content = replies[len(outer.calls) - 1]

                class _Message:
                    def __init__(self, text: str) -> None:
                        self.content = text

                class _Choice:
                    def __init__(self, text: str) -> None:
                        self.message = _Message(text)
                        self.finish_reason = finish_reason

                class _Response:
                    model = "dummy-model"

                    def __init__(self, text: str) -> None:
                        self.choices = [_Choice(text)]

                return _Response(content)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class _Runtime:
    def __init__(self, contents: list[str | None], error: object = None) -> None:
        self._contents = contents
        self._error = error
        self.batches: list[list] = []

    def run_cluster_naming_batch(self, requests):
        self.batches.append(list(requests))
        return [
            ChatCompletionResult(
                content=content,
                error=self._error,
                queue_wait_sec=0.0,
                request_elapsed_sec=0.0,
            )
            for content in self._contents[: len(requests)]
        ]


NAMING_CONTENTS = [
    "Matrix Rank Notes\nRank-nullity theorem.",
    "Matrix Rank Notes\nColumn space.",
    "Quarterly Budget Review\nSpending by month.",
    "Quarterly Budget Review\nIncome statement.",
    "Organic Chemistry Lab\nTitration results.",
    "Organic Chemistry Lab\nReaction yields.",
]
NAMING_FILES = [
    Path("rank-a.txt"),
    Path("rank-b.txt"),
    Path("budget-a.txt"),
    Path("budget-b.txt"),
    Path("lab-a.txt"),
    Path("lab-b.txt"),
]


# --- strip_reasoning_text -------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "<think>\nThe files cover matrices.\n</think>\n\nLinear Algebra",
            "Linear Algebra",
        ),
        ("<think>\nOkay, the user wants a name for", ""),
        ("<think>\n\n</think>\n\nFinancial Reports", "Financial Reports"),
        ("The files cover matrices.\n</think>\nLinear Algebra", "Linear Algebra"),
        ("<THINK>plan</THINK>  Budget", "Budget"),
        ("<think>a</think>Budget<think>b", "Budget"),
        ("Linear Algebra", "Linear Algebra"),
        ("", ""),
        (None, ""),
    ],
)
def test_strip_reasoning_text_removes_closed_unclosed_and_orphan_blocks(
    reply, expected
) -> None:
    from dite.util.llm import strip_reasoning_text

    assert strip_reasoning_text(reply) == expected


def test_parse_json_object_ignores_braces_inside_reasoning() -> None:
    reply = (
        '<think>\nMaybe {"should_merge": false}? No, they share a course.\n'
        "</think>\n"
        '{"should_merge": true, "confidence": 0.8, "reason": "same course"}'
    )

    assert parse_json_object(reply) == {
        "should_merge": True,
        "confidence": 0.8,
        "reason": "same course",
    }


def test_parse_json_object_does_not_read_truncated_reasoning_as_answer() -> None:
    reply = '<think>\nA draft answer could be {"should_merge": true} but wait'

    assert parse_json_object(reply) is None


def test_adjudication_and_representation_parsers_strip_reasoning() -> None:
    from dite.cluster.rule import _parse_llm_adjudication
    from dite.cluster.view import _parse_llm_cluster_representation

    adjudication = _parse_llm_adjudication(
        '<think>{"should_merge": false}</think>'
        '{"should_merge": true, "confidence": 0.9, "reason": "same project"}'
    )
    representation = _parse_llm_cluster_representation(
        '<think>{"topic": "wrong"}</think>{"topic": "linear algebra"}'
    )

    assert adjudication is not None
    assert adjudication["should_merge"] is True
    assert representation is not None
    assert representation["topic"] == "linear algebra"


def test_analyze_document_strips_reasoning_before_json_parsing() -> None:
    from dite.doc.analyze import analyze_document

    client = _ReplyClient(
        "<think>\nLet me look at the document.\n</think>\n\n"
        '{"content": {"topic": "finance", "keywords": ["profit"]}}'
    )

    analysis = analyze_document(client, "Quarterly profit", config=Config())

    assert analysis.content.topic == "finance"
    assert analysis.content.keywords == ["profit"]
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "reply",
    [
        "<think>\nLet me analyze this document. The topic seems",
        "<think>\nThe document is about profit.\n</think>",
        "   ",
    ],
)
def test_analyze_document_retries_and_warns_when_reply_has_no_answer(
    reply: str, capsys
) -> None:
    from dite.doc.analyze import analyze_document

    client = _ReplyClient(reply, finish_reason="length")

    analysis = analyze_document(
        client, "Quarterly profit", config=Config(), max_retries=2
    )

    assert analysis.content.topic == ""
    assert len(client.calls) == 2
    assert "Document analysis failed, using defaults" in _flat(capsys.readouterr().out)


# --- cluster-name validation ----------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["<think>", "</think>", "<think> Okay, the user wants", "<|im_end|>", "<answer>"],
)
def test_tag_like_cluster_names_are_invalid(name: str) -> None:
    from dite.cluster.name import _display_cluster_name, _is_invalid_cluster_name

    assert _is_invalid_cluster_name(name) is True
    assert _display_cluster_name(3, name) == "Unnamed-3"


def test_ordinary_cluster_names_stay_valid() -> None:
    from dite.cluster.name import _is_invalid_cluster_name

    assert _is_invalid_cluster_name("Linear Algebra") is False
    assert _is_invalid_cluster_name("HTML <canvas> Basics") is False


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (
            "<think>\nThe files cover matrices.\n</think>\n\nLinear Algebra",
            "Linear Algebra",
        ),
        ("**Financial Reports**", "Financial Reports"),
        ("Name: Machine Learning", "Machine Learning"),
        ("**Name:** Linear Algebra", "Linear Algebra"),
        ("Category name: Tax Filings", "Tax Filings"),
        ("名称：财务报表", "财务报表"),
        ("类别名称: 线性代数", "线性代数"),
    ],
)
def test_generate_cluster_name_cleans_model_reply(reply: str, expected: str) -> None:
    from dite.cluster.api import generate_cluster_name

    name = generate_cluster_name(
        _ReplyClient(reply),
        None,
        ["Matrix Rank Notes\nRank-nullity theorem."],
        ["rank.txt"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == expected


def test_generate_cluster_name_falls_back_on_unclosed_think_reply() -> None:
    from dite.cluster.api import generate_cluster_name

    name = generate_cluster_name(
        _ReplyClient("<think>\nOkay, the user wants me to name this category"),
        None,
        ["Matrix Rank Notes\nRank-nullity theorem."],
        ["rank.txt"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "Matrix Rank Notes"


@pytest.mark.parametrize("module_name", ["dite.cluster.api", "dite.cluster.name"])
def test_unclosed_think_replies_do_not_collapse_clusters_sync(
    module_name: str,
) -> None:
    import importlib

    module = importlib.import_module(module_name)
    named = module.generate_all_cluster_names(
        _ReplyClient("<think>\nOkay, the user wants me to name"),
        _cluster_result([0, 0, 1, 1, 2, 2]),
        NAMING_CONTENTS,
        NAMING_FILES,
        config=Config(),
        merge_same_name=True,
        llm_model="dummy-model",
    )

    assert named.cluster_names == {
        0: "Matrix Rank Notes",
        1: "Quarterly Budget Review",
        2: "Organic Chemistry Lab",
    }
    assert named.name_clusters_merged == 0
    assert np.array_equal(named.labels, np.array([0, 0, 1, 1, 2, 2]))


@pytest.mark.parametrize("module_name", ["dite.cluster.api", "dite.cluster.name"])
def test_think_replies_are_cleaned_in_async_naming(module_name: str) -> None:
    import importlib

    module = importlib.import_module(module_name)
    runtime = _Runtime(
        [
            "<think>\nOkay, the user wants me to name",
            "<think>\nBudget words.\n</think>\n\nFinancial Reports",
            "<think>",
        ]
    )
    named = module.generate_all_cluster_names(
        _Endpoint("http://localhost:8000/v1"),
        _cluster_result([0, 0, 1, 1, 2, 2]),
        NAMING_CONTENTS,
        NAMING_FILES,
        config=Config(),
        merge_same_name=True,
        llm_model="dummy-model",
        request_runtime=runtime,
    )

    assert named.cluster_names == {
        0: "Matrix Rank Notes",
        1: "Financial Reports",
        2: "Organic Chemistry Lab",
    }
    assert named.name_clusters_merged == 0


# --- thinking controls ----------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "https://api-inference.modelscope.cn/v1",
        "https://api.siliconflow.cn/v1",
    ],
)
def test_reasoning_off_is_sent_to_known_enable_thinking_hosts(base_url: str) -> None:
    kwargs = build_chat_completion_kwargs(
        client=_Endpoint(base_url),
        model="qwen3-32b",
        messages=[],
        profile=OFF_PROFILE,
    )

    assert kwargs["extra_body"] == {"enable_thinking": False}


def test_reasoning_controls_read_host_from_real_openai_client() -> None:
    client = OpenAI(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    profile = ChatCompletionProfileConfig(reasoning_mode="on", thinking_budget=256)

    kwargs = build_chat_completion_kwargs(
        client=client, model="qwen3-32b", messages=[], profile=profile
    )

    assert kwargs["extra_body"] == {"enable_thinking": True, "thinking_budget": 256}


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "http://localhost:8000/v1",
        "https://openrouter.ai/api/v1",
        "https://proxy.example.com/dashscope.aliyuncs.com/v1",
    ],
)
def test_reasoning_off_sends_nothing_to_other_hosts_by_default(base_url: str) -> None:
    kwargs = build_chat_completion_kwargs(
        client=_Endpoint(base_url),
        model="gpt-4o-mini",
        messages=[],
        profile=OFF_PROFILE,
    )

    assert "extra_body" not in kwargs
    assert "enable_thinking" not in kwargs
    assert "chat_template_kwargs" not in kwargs


@pytest.mark.parametrize(
    ("style", "base_url", "expected"),
    [
        (
            "chat_template_kwargs",
            "http://localhost:8000/v1",
            {"chat_template_kwargs": {"enable_thinking": False}},
        ),
        ("enable_thinking", "http://localhost:8000/v1", {"enable_thinking": False}),
        ("none", "https://dashscope.aliyuncs.com/compatible-mode/v1", None),
        (
            "auto",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            {"enable_thinking": False},
        ),
    ],
)
def test_reasoning_param_style_overrides_host_detection(
    style: str, base_url: str, expected: dict | None
) -> None:
    kwargs = build_chat_completion_kwargs(
        client=_Endpoint(base_url),
        model="Qwen/Qwen3-32B",
        messages=[],
        profile=OFF_PROFILE,
        reasoning_param_style=style,
    )

    assert kwargs.get("extra_body") == expected


def test_default_reasoning_mode_sends_nothing_with_explicit_style() -> None:
    kwargs = build_chat_completion_kwargs(
        client=_Endpoint("http://localhost:8000/v1"),
        model="Qwen/Qwen3-32B",
        messages=[],
        profile=ChatCompletionProfileConfig(reasoning_mode="default"),
        reasoning_param_style="chat_template_kwargs",
    )

    assert "extra_body" not in kwargs


def test_cluster_naming_requests_use_configured_reasoning_param_style() -> None:
    from dite.cluster.api import generate_all_cluster_names, generate_cluster_name

    config = Config(api=APIConfig(reasoning_param_style="chat_template_kwargs"))
    expected = {"chat_template_kwargs": {"enable_thinking": False}}

    client = _ReplyClient("Linear Algebra")
    generate_cluster_name(
        client, None, ["x"], ["x.txt"], config=config, llm_model="dummy-model"
    )
    assert client.calls[0]["extra_body"] == expected

    runtime = _Runtime(["Linear Algebra"])
    generate_all_cluster_names(
        _Endpoint("http://localhost:8000/v1"),
        _cluster_result([0, 0]),
        NAMING_CONTENTS[:2],
        NAMING_FILES[:2],
        config=config,
        llm_model="dummy-model",
        request_runtime=runtime,
    )
    assert runtime.batches[0][0].kwargs["extra_body"] == expected


def test_adjudication_and_representation_requests_use_configured_style() -> None:
    from dite.cluster.model import AdjudicationRequest
    from dite.cluster.rule import apply_llm_adjudication
    from dite.cluster.view import build_cluster_representations

    config = Config(api=APIConfig(reasoning_param_style="chat_template_kwargs"))
    config.cluster_representation.mode = "llm_enhanced"
    expected = {"chat_template_kwargs": {"enable_thinking": False}}
    client = _Endpoint("http://localhost:8000/v1")

    adjudication_runtime = _Runtime(
        ['<think>hm</think>{"should_merge": true, "confidence": 0.9, "reason": "x"}']
    )
    decisions = apply_llm_adjudication(
        [
            AdjudicationRequest(
                request_id="r1", request_type="edge_review", subjects=["a", "b"]
            )
        ],
        client=client,
        config=config,
        request_runtime=adjudication_runtime,
    )
    assert adjudication_runtime.batches[0][0].kwargs["extra_body"] == expected
    assert decisions[0].decision == "merge_edge"
    assert decisions[0].fallback_used is False

    representation_runtime = _Runtime(['<think>hm</think>{"topic": "algebra"}'])
    representations = build_cluster_representations(
        labels=np.array([0], dtype=int),
        cluster_names={0: "Linear Algebra"},
        document_features=[
            DocumentFeatures(
                file_id="a.txt",
                path=Path("a.txt"),
                name="a.txt",
                stem="a",
                extension=".txt",
                content_text="Matrices",
            )
        ],
        config=config,
        client=client,
        request_runtime=representation_runtime,
    )
    assert representation_runtime.batches[0][0].kwargs["extra_body"] == expected
    assert representations[0].topic == "algebra"


def test_reasoning_param_style_config_default_and_parsing(
    tmp_path: Path, monkeypatch
) -> None:
    from dite.app.config import _dict_to_config

    assert Config().api.reasoning_param_style == "auto"

    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "api:\n  base_url: http://localhost:8000/v1\n"
        "  reasoning_param_style: chat_template_kwargs\n",
        encoding="utf-8",
    )
    assert load_config().api.reasoning_param_style == "chat_template_kwargs"

    with pytest.raises(ValueError, match="reasoning_param_style"):
        _dict_to_config({"api": {"reasoning_param_style": "vllm"}})


# --- user-facing warning when naming degrades -----------------------------


def test_all_failed_naming_requests_warn_with_count_and_error(capsys) -> None:
    from dite.cluster.api import generate_all_cluster_names

    client = _ReplyClient(
        [], error=RuntimeError("parameter.enable_thinking must be set to false")
    )
    generate_all_cluster_names(
        client,
        _cluster_result([0, 0, 1, 1]),
        NAMING_CONTENTS[:4],
        NAMING_FILES[:4],
        config=Config(),
        llm_model="dummy-model",
    )

    output = _flat(capsys.readouterr().out)
    assert "WARNING" in output
    assert "heuristic names for 2 of 2 clusters" in output
    assert "request failed: 2" in output
    assert "parameter.enable_thinking must be set to false" in output


def test_reasoning_only_naming_replies_warn_with_hint(capsys) -> None:
    from dite.cluster.api import generate_all_cluster_names

    runtime = _Runtime(["<think>\nOkay", "", "<think>\nHmm"])
    generate_all_cluster_names(
        _Endpoint("http://localhost:8000/v1"),
        _cluster_result([0, 0, 1, 1, 2, 2]),
        NAMING_CONTENTS,
        NAMING_FILES,
        config=Config(),
        llm_model="dummy-model",
        request_runtime=runtime,
    )

    output = _flat(capsys.readouterr().out)
    assert "heuristic names for 3 of 3 clusters" in output
    assert "empty or reasoning-only reply: 3" in output
    assert "api.reasoning_param_style" in output


def test_naming_fallback_warning_follows_locale(capsys) -> None:
    from dite.cluster.name import generate_all_cluster_names

    set_locale("zh-CN")
    runtime = _Runtime([None, None], error="boom")
    generate_all_cluster_names(
        _Endpoint("http://localhost:8000/v1"),
        _cluster_result([0, 0, 1, 1]),
        NAMING_CONTENTS[:4],
        NAMING_FILES[:4],
        config=Config(),
        llm_model="dummy-model",
        request_runtime=runtime,
    )

    # Rich folds CJK text at any character, so compare without whitespace.
    output = "".join(capsys.readouterr().out.split())
    assert "2/2个簇退回了启发式名称" in output
    assert "请求失败2个" in output
    assert "boom" in output
    assert "heuristicnames" not in output


def test_occasional_naming_fallback_does_not_warn(capsys) -> None:
    from dite.cluster.api import generate_all_cluster_names

    runtime = _Runtime(["Linear Algebra", "Financial Reports", "<think>"])
    generate_all_cluster_names(
        _Endpoint("http://localhost:8000/v1"),
        _cluster_result([0, 0, 1, 1, 2, 2]),
        NAMING_CONTENTS,
        NAMING_FILES,
        config=Config(),
        llm_model="dummy-model",
        request_runtime=runtime,
    )

    output = _flat(capsys.readouterr().out)
    assert "heuristic names" not in output
