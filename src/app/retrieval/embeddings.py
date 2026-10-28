"""Embeddings: text -> vectors with OpenAI's text-embedding-3-small (1536 dims).

Used at ingestion (clause chunks, batched) and at query time (one query).
The same model must be used for both, or vectors aren't comparable.
"""

from __future__ import annotations

import time

from openai import AsyncOpenAI

from app.config import Settings
from app.core.exceptions import LLMProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

_BATCH_SIZE = 64


class Embedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY is required for embeddings.")
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self._model = settings.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            started = time.perf_counter()
            try:
                response = await self._client.embeddings.create(model=self._model, input=batch)
            except Exception as exc:
                raise LLMProviderError(f"Embedding call failed: {exc}") from exc
            vectors.extend(item.embedding for item in response.data)
            logger.info(
                "embedding_call",
                extra={
                    "ctx_model": self._model,
                    "ctx_inputs": len(batch),
                    "ctx_tokens": response.usage.prompt_tokens,
                    "ctx_latency_ms": round((time.perf_counter() - started) * 1000),
                },
            )
        return vectors
