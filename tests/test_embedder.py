import types

import httpx
import numpy as np
import pytest
from openai import BadRequestError

from dite.app.config import Config
from dite.doc.embed import (
    ContentTruncator,
    get_embedding_cache_version,
    get_embeddings,
)


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Item:
            def __init__(self) -> None:
                self.embedding = [0.1, 0.2, 0.3]

        class _Usage:
            total_tokens = 3

        class _Response:
            data = [_Item()]
            usage = _Usage()

        return _Response()


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


def test_get_embeddings_includes_file_name_for_non_empty_text() -> None:
    client = _FakeClient()

    result = get_embeddings(
        client,
        ["This document explains Rust ownership and borrowing."],
        config=Config(),
        file_names=["Rust 程序设计.pdf"],
        embedding_model="embed-model",
    )

    assert result.shape == (1, 3)
    request_input = client.embeddings.calls[0]["input"][0]
    assert request_input.startswith("File name: Rust 程序设计.pdf")
    assert "Content:\nThis document explains Rust ownership" in request_input
    assert client.embeddings.calls[0]["model"] == "embed-model"
    np.testing.assert_allclose(
        result,
        np.array([[0.26726124, 0.53452248, 0.80178373]], dtype=np.float32),
        rtol=1e-6,
    )


def test_get_embedding_cache_version_tracks_input_format() -> None:
    assert get_embedding_cache_version("embed-model") == (
        "embed-model|input=filename-smart-content-normalized-v2"
    )
    assert get_embedding_cache_version("embed-model", "content_only") == (
        "embed-model|input=content-only-normalized-v1"
    )


def test_get_embeddings_content_only_omits_file_name() -> None:
    client = _FakeClient()

    get_embeddings(
        client,
        ["This document explains Rust ownership and borrowing."],
        config=Config(),
        file_names=["Rust 程序设计.pdf"],
        embedding_model="embed-model",
        input_mode="content_only",
    )

    request_input = client.embeddings.calls[0]["input"][0]
    assert request_input == "This document explains Rust ownership and borrowing."


def test_get_embeddings_content_only_falls_back_to_placeholder_for_empty_text() -> None:
    client = _FakeClient()

    get_embeddings(
        client,
        [""],
        config=Config(),
        file_names=["empty.txt"],
        embedding_model="embed-model",
        input_mode="content_only",
    )

    request_input = client.embeddings.calls[0]["input"][0]
    assert request_input == "file_0"


def test_content_truncator_keeps_head_middle_tail_within_limit() -> None:
    content = ("HEAD-" * 30) + ("MIDDLE-" * 30) + ("TAIL-" * 30)

    truncated = ContentTruncator.truncate_smart(content, max_chars=120)

    assert len(truncated) <= 120
    assert truncated.startswith("HEAD-")
    assert "MIDDLE-" in truncated
    assert truncated.endswith("TAIL-")
    assert "middle omitted" in truncated


class _BatchLimitedEmbeddings:
    """Fake provider that rejects batches above a limit and shuffles order."""

    def __init__(self, max_inputs: int, *, reverse_order: bool = False) -> None:
        self.max_inputs = max_inputs
        self.reverse_order = reverse_order
        self.calls: list[list[str]] = []

    def create(self, **kwargs):
        inputs = list(kwargs["input"])
        self.calls.append(inputs)
        if len(inputs) > self.max_inputs:
            request = httpx.Request("POST", "https://api.example.com/v1/embeddings")
            raise BadRequestError(
                "too many inputs",
                response=httpx.Response(400, request=request),
                body=None,
            )
        items = []
        for index, text in enumerate(inputs):
            number = float(text.rsplit("#", 1)[-1])
            items.append(types.SimpleNamespace(index=index, embedding=[1.0, number]))
        if self.reverse_order:
            items.reverse()
        return types.SimpleNamespace(
            data=items,
            usage=types.SimpleNamespace(total_tokens=len(inputs)),
        )


def _numbered_texts(count: int) -> list[str]:
    return [f"document body long enough #{index}" for index in range(count)]


def _second_components(result: np.ndarray) -> list[int]:
    # Vectors are [1, n] normalized; recover n from the component ratio.
    return [round(float(row[1] / row[0])) for row in result]


def test_get_embeddings_sends_configured_batches_in_order() -> None:
    config = Config()
    config.processing.embedding_batch_size = 4
    client = types.SimpleNamespace(embeddings=_BatchLimitedEmbeddings(100))

    result = get_embeddings(
        client,
        _numbered_texts(10),
        config=config,
        input_mode="content_only",
    )

    assert [len(call) for call in client.embeddings.calls] == [4, 4, 2]
    assert _second_components(result) == list(range(10))


def test_get_embeddings_halves_batches_the_provider_rejects() -> None:
    config = Config()
    config.processing.embedding_batch_size = 8
    client = types.SimpleNamespace(embeddings=_BatchLimitedEmbeddings(3))

    result = get_embeddings(
        client,
        _numbered_texts(8),
        config=config,
        input_mode="content_only",
    )

    assert _second_components(result) == list(range(8))
    accepted = [len(call) for call in client.embeddings.calls if len(call) <= 3]
    assert sum(accepted) == 8


def test_get_embeddings_reorders_vectors_by_response_index() -> None:
    config = Config()
    client = types.SimpleNamespace(
        embeddings=_BatchLimitedEmbeddings(100, reverse_order=True)
    )

    result = get_embeddings(
        client,
        _numbered_texts(5),
        config=config,
        input_mode="content_only",
    )

    assert _second_components(result) == list(range(5))


def test_get_embeddings_raises_when_even_single_input_is_rejected() -> None:
    config = Config()
    config.processing.embedding_batch_size = 16
    client = types.SimpleNamespace(embeddings=_BatchLimitedEmbeddings(0))

    with pytest.raises(BadRequestError):
        get_embeddings(
            client,
            _numbered_texts(16),
            config=config,
            input_mode="content_only",
        )

    # Only the first half is descended: 16, 8, 4, 2, 1.
    assert [len(call) for call in client.embeddings.calls] == [16, 8, 4, 2, 1]


def test_get_embeddings_rejects_response_with_missing_vectors() -> None:
    class _ShortEmbeddings:
        def create(self, **kwargs):
            del kwargs
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(index=0, embedding=[1.0, 0.0])],
                usage=None,
            )

    client = types.SimpleNamespace(embeddings=_ShortEmbeddings())

    with pytest.raises(ValueError, match="size mismatch"):
        get_embeddings(
            client,
            _numbered_texts(2),
            config=Config(),
            input_mode="content_only",
        )
