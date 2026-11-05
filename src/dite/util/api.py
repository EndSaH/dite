"""Shared OpenAI client configuration and async request runtime."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from dataclasses import dataclass
from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI

from dite.app.config import Config
from dite.app.i18n import t
from dite.util.cancel import (
    ExtractionCancelledError,
    current_cancel_event,
    raise_if_cancelled,
)
from dite.util.log import get_logger


@dataclass(frozen=True)
class ChatCompletionRequest:
    """Single chat completion request payload."""

    kwargs: dict[str, Any]


@dataclass(frozen=True)
class ChatCompletionResult:
    """Single chat completion response plus runtime timings."""

    content: str | None
    error: BaseException | None
    queue_wait_sec: float
    request_elapsed_sec: float
    finish_reason: str | None = None


def build_httpx_timeout(config: Config) -> httpx.Timeout:
    """Build a shared timeout object from config."""
    api = config.api
    return httpx.Timeout(
        timeout=api.read_timeout_sec,
        connect=api.connect_timeout_sec,
        read=api.read_timeout_sec,
        write=api.write_timeout_sec,
        pool=api.pool_timeout_sec,
    )


def build_httpx_limits(config: Config) -> httpx.Limits:
    """Build shared connection pool limits from config."""
    api = config.api
    return httpx.Limits(
        max_connections=api.max_connections,
        max_keepalive_connections=api.max_keepalive_connections,
        keepalive_expiry=api.keepalive_expiry_sec,
    )


def build_sync_openai_client(config: Config) -> OpenAI:
    """Build the sync OpenAI client with explicit pool and timeout settings."""
    timeout = build_httpx_timeout(config)
    limits = build_httpx_limits(config)
    http_client = httpx.Client(timeout=timeout, limits=limits)
    return OpenAI(
        api_key=config.api.api_key,
        base_url=config.api.base_url,
        timeout=timeout,
        max_retries=config.api.max_retries,
        http_client=http_client,
    )


def _build_async_openai_client(config: Config) -> AsyncOpenAI:
    timeout = build_httpx_timeout(config)
    limits = build_httpx_limits(config)
    http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
    return AsyncOpenAI(
        api_key=config.api.api_key,
        base_url=config.api.base_url,
        timeout=timeout,
        max_retries=config.api.max_retries,
        http_client=http_client,
    )


def _effective_concurrency(requested: int, *, max_connections: int) -> int:
    requested = max(1, requested)
    return min(requested, max_connections)


def _is_cancelled(cancel_event: threading.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _cancelled_result(queued_at: float) -> ChatCompletionResult:
    """Result for a request skipped because the run was interrupted."""
    now = asyncio.get_running_loop().time()
    return ChatCompletionResult(
        content=None,
        error=ExtractionCancelledError(),
        queue_wait_sec=now - queued_at,
        request_elapsed_sec=0.0,
    )


class AsyncRequestRuntime:
    """Run async OpenAI chat requests from sync code with shared semaphores."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._logger = get_logger()
        self._closed = False
        self._started = False
        self._effective_cluster_workers = _effective_concurrency(
            config.processing.cluster_naming_workers,
            max_connections=config.api.max_connections,
        )
        self._effective_vlm_workers = _effective_concurrency(
            config.processing.vlm_api_workers,
            max_connections=config.api.max_connections,
        )
        self._effective_vlm_pages_per_document = min(
            _effective_concurrency(
                config.processing.vlm_pages_per_document,
                max_connections=config.api.max_connections,
            ),
            self._effective_vlm_workers,
        )
        self._cluster_naming_sem = threading.BoundedSemaphore(
            self._effective_cluster_workers
        )
        self._vlm_global_sem = threading.BoundedSemaphore(self._effective_vlm_workers)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client: AsyncOpenAI | None = None

    def __enter__(self) -> AsyncRequestRuntime:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()

    @property
    def effective_cluster_workers(self) -> int:
        return self._effective_cluster_workers

    @property
    def effective_vlm_workers(self) -> int:
        return self._effective_vlm_workers

    @property
    def effective_vlm_pages_per_document(self) -> int:
        return self._effective_vlm_pages_per_document

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("async request runtime is already closed")
        if self._started:
            return
        loop_ready = threading.Event()

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._client = _build_async_openai_client(self._config)
            loop_ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, name="dite-async-runtime")
        self._thread.start()
        loop_ready.wait()
        self._started = True
        self._log_effective_limits()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._started:
            return
        assert self._loop is not None
        self._submit_coroutine(self._shutdown_runtime()).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        assert self._thread is not None
        self._thread.join()
        self._loop.close()
        self._loop = None
        self._thread = None
        self._client = None

    def run_cluster_naming_batch(
        self,
        requests: list[ChatCompletionRequest],
    ) -> list[ChatCompletionResult]:
        self.start()
        return self._submit_coroutine(
            self._run_batch(
                requests=requests,
                global_sem=self._cluster_naming_sem,
                local_limit=None,
            )
        ).result()

    def run_vlm_page_batch(
        self,
        requests: list[ChatCompletionRequest],
        *,
        per_document_limit: int | None = None,
    ) -> list[ChatCompletionResult]:
        self.start()
        local_limit = self._effective_vlm_pages_per_document
        if per_document_limit is not None:
            local_limit = min(
                max(1, per_document_limit),
                self._effective_vlm_pages_per_document,
            )
        # Called on an extraction worker. Once the run is interrupted, pages
        # still waiting for a request slot are not sent, and the partial
        # batch is not handed back as if it were the document's content.
        cancel_event = current_cancel_event()
        results = self._submit_coroutine(
            self._run_batch(
                requests=requests,
                global_sem=self._vlm_global_sem,
                local_limit=local_limit,
                cancel_event=cancel_event,
            )
        ).result()
        raise_if_cancelled()
        return results

    def run_image_vlm(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        return self.run_vlm_page_batch([request], per_document_limit=1)[0]

    async def _run_batch(
        self,
        *,
        requests: list[ChatCompletionRequest],
        global_sem: threading.BoundedSemaphore,
        local_limit: int | None,
        cancel_event: threading.Event | None = None,
    ) -> list[ChatCompletionResult]:
        assert self._client is not None
        local_sem = asyncio.Semaphore(local_limit) if local_limit is not None else None
        tasks = [
            self._run_single_request(
                client=self._client,
                request=request,
                global_sem=global_sem,
                local_sem=local_sem,
                cancel_event=cancel_event,
            )
            for request in requests
        ]
        return await asyncio.gather(*tasks)

    async def _run_single_request(
        self,
        *,
        client: AsyncOpenAI,
        request: ChatCompletionRequest,
        global_sem: threading.BoundedSemaphore,
        local_sem: asyncio.Semaphore | None,
        cancel_event: threading.Event | None = None,
    ) -> ChatCompletionResult:
        queued_at = asyncio.get_running_loop().time()
        while not global_sem.acquire(blocking=False):
            if _is_cancelled(cancel_event):
                return _cancelled_result(queued_at)
            await asyncio.sleep(0.001)
        try:
            if local_sem is not None:
                async with local_sem:
                    if _is_cancelled(cancel_event):
                        return _cancelled_result(queued_at)
                    return await self._perform_request(client, request, queued_at)
            if _is_cancelled(cancel_event):
                return _cancelled_result(queued_at)
            return await self._perform_request(client, request, queued_at)
        finally:
            global_sem.release()

    async def _perform_request(
        self,
        client: AsyncOpenAI,
        request: ChatCompletionRequest,
        queued_at: float,
    ) -> ChatCompletionResult:
        started_at = asyncio.get_running_loop().time()
        try:
            response = await client.chat.completions.create(**request.kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""
            return ChatCompletionResult(
                content=content,
                error=None,
                queue_wait_sec=started_at - queued_at,
                request_elapsed_sec=asyncio.get_running_loop().time() - started_at,
                finish_reason=getattr(choice, "finish_reason", None),
            )
        except BaseException as exc:
            return ChatCompletionResult(
                content=None,
                error=exc,
                queue_wait_sec=started_at - queued_at,
                request_elapsed_sec=asyncio.get_running_loop().time() - started_at,
            )

    def _log_effective_limits(self) -> None:
        self._logger.debug(
            t(
                "debug_async_runtime_limits",
                cluster_workers=self._effective_cluster_workers,
                vlm_workers=self._effective_vlm_workers,
                vlm_pages_per_document=self._effective_vlm_pages_per_document,
                max_connections=self._config.api.max_connections,
            )
        )

    def _submit_coroutine(
        self,
        coro: asyncio.Future | asyncio.coroutines.Coroutine[Any, Any, Any],
    ) -> concurrent.futures.Future:
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _shutdown_runtime(self) -> None:
        if self._client is not None:
            await self._client.close()
        loop = asyncio.get_running_loop()
        await loop.shutdown_asyncgens()
        await loop.shutdown_default_executor(timeout=5.0)
