"""Embeddings: the OpenAI-compatible wire shape, and a fake that behaves like it.

Anthropic has no embeddings API, so the embedding vendor is always a different
one -- Voyage, OpenAI, Jina, or something we host on Modal. What they share is a
single request shape (`POST /embeddings` with `{"model", "input": [...]}`
returning `{"data": [{"index", "embedding"}]}`), which is why one client covers
all of them and `EMBEDDING_BASE_URL` is the only thing that changes.

Two rules in here are worth more than they look.

**Batch strictly to `EMBEDDING_BATCH_SIZE`.** Providers cap inputs per request
and cap total tokens per request, and the failure when you exceed either is a
400 for the whole batch -- so one oversized call loses every chunk in it, not
the last one. Chunking on our side makes the loss granular and the retries
cheap.

**Check the vector width on arrival.** A model swap that changes dimensionality
(1536 -> 1024 is the common one) produces vectors that look perfectly fine right
up until the Qdrant upsert, which is *after* the spend and after the pipeline
has declared the Embedding stage successful. Checking at the boundary turns a
confusing write failure hours later into an immediate error naming both numbers.

The error taxonomy is `services/llm/provider.py`'s, deliberately. An enrichment
stage catches one family and degrades (`docs/signal-model.md` §5.2); giving
embeddings a parallel hierarchy would mean every caller catching two, and the
one that forgets the second is the one that drops Signals.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from typing import Any, Final, Protocol, runtime_checkable

import httpx

from backend.core.config import EmbeddingProvider as EmbeddingBackend
from backend.core.config import EmbeddingSettings, get_settings
from backend.core.exceptions import ConfigurationError
from services.llm.provider import LLMError, LLMRateLimited, LLMTimeout

__all__ = [
    "EmbeddingDimensionMismatch",
    "EmbeddingProvider",
    "FakeEmbeddingProvider",
    "OpenAICompatibleEmbeddingProvider",
]

_DEFAULT_BASE_URLS: Final[dict[EmbeddingBackend, str]] = {
    EmbeddingBackend.OPENAI: "https://api.openai.com/v1",
    EmbeddingBackend.VOYAGE: "https://api.voyageai.com/v1",
    EmbeddingBackend.JINA: "https://api.jina.ai/v1",
}
"""Known hosted endpoints. Anything else must set `EMBEDDING_BASE_URL` itself.

Self-hosted backends (`modal`, `local`, `huggingface`) have no canonical URL to
guess, and guessing wrong there means embedding against a *different corpus's*
model with no error to show for it.
"""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What the Embedding stage and the retrieval layer depend on.

    `model` and `dimensions` are attributes rather than methods because they are
    recorded per Signal: reproducing a vector later requires knowing which model
    produced it (`docs/signal-model.md` §5.1), and a collection's width is fixed
    at creation, so both belong to the provider's identity rather than to a call.
    """

    model: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, in order. `len(result) == len(texts)`."""
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP client. Idempotent."""
        ...


class EmbeddingDimensionMismatch(LLMError):  # noqa: N818 -- matches LLMTimeout/LLMRateLimited
    """The provider returned a vector of the wrong width.

    Its own class because the remedy is unique and structural: either
    `EMBEDDING_DIMENSIONS` is wrong, or the model changed underneath us and the
    entire corpus needs re-embedding (`docs/signal-model.md` §9). Neither is
    something a retry can help with, so it must not look like a provider blip.
    """

    code = "embedding_dimension_mismatch"
    default_message = "The embedding provider returned an unexpected vector width."


class OpenAICompatibleEmbeddingProvider:
    """HTTP embedding client for any provider speaking the OpenAI shape."""

    def __init__(
        self,
        *,
        settings: EmbeddingSettings | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        resolved = settings if settings is not None else get_settings().embedding

        if not resolved.model:
            # Boot-time. Every other failure mode here is recoverable; a missing
            # model name means no Signal can ever be embedded.
            raise ConfigurationError(
                "EMBEDDING_MODEL is not set; the embedding provider cannot be built.",
                details={"setting": "EMBEDDING_MODEL"},
            )
        base_url = resolved.base_url or _DEFAULT_BASE_URLS.get(
            resolved.provider or EmbeddingBackend.OPENAI
        )
        if not base_url:
            raise ConfigurationError(
                f"EMBEDDING_BASE_URL is required for provider "
                f"{(resolved.provider or 'unset')!r}; there is no default endpoint.",
                details={"setting": "EMBEDDING_BASE_URL"},
            )

        self.model = resolved.model
        self.dimensions = resolved.dimensions
        self._batch_size = resolved.batch_size
        self._base_url = base_url.rstrip("/")
        # Optional: a self-hosted endpoint on Modal or in-cluster needs no
        # credential, and demanding one would make the local path unusable.
        self._api_key = (
            resolved.api_key.get_secret_value() if resolved.api_key is not None else None
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed `texts`, one request per `EMBEDDING_BATCH_SIZE` inputs."""
        if not texts:
            # Not a request with an empty array -- most providers answer that
            # with a 400, and "nothing to embed" is a legitimate call from a
            # chunker that found no usable text.
            return []
        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(
                    f"texts[{index}] is empty; an empty string has no meaningful "
                    "embedding and most providers reject it. Filter before calling."
                )

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def aclose(self) -> None:
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------ internals --

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model, "input": batch}
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        try:
            response = await self._client.post(
                f"{self._base_url}/embeddings", json=payload, headers=headers
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeout("the embedding provider did not respond in time.", cause=exc) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"transport failure talking to the embedding provider ({type(exc).__name__}).",
                cause=exc,
            ) from exc

        if response.status_code == 429:
            raise LLMRateLimited("the embedding provider rate limited the request.")
        if response.status_code >= 400:
            raise LLMError(
                f"the embedding provider returned HTTP {response.status_code}.",
                provider_status=response.status_code,
            )

        return self._vectors_of(response, expected=len(batch))

    def _vectors_of(self, response: httpx.Response, *, expected: int) -> list[list[float]]:
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError("the embedding provider returned non-JSON.", cause=exc) from exc

        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            raise LLMError("the embedding response had no 'data' array.")
        if len(data) != expected:
            # Silently short results would shift every later vector onto the
            # wrong Signal, which is unrecoverable and undetectable downstream.
            raise LLMError(
                f"the embedding provider returned {len(data)} vectors for {expected} inputs."
            )

        # Sort by `index` rather than trusting arrival order. The field exists
        # precisely because the order is not guaranteed, and a provider that
        # reorders under load would pair each vector with the wrong text -- a
        # corruption that no test of ours would catch and no user would report
        # as anything but "search got worse".
        ordered = sorted(data, key=lambda item: _as_int(item.get("index")))
        vectors: list[list[float]] = []
        for position, item in enumerate(ordered):
            raw = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(raw, list):
                raise LLMError(f"embedding {position} was missing or not a list.")
            if len(raw) != self.dimensions:
                raise EmbeddingDimensionMismatch(
                    f"the embedding provider returned {len(raw)} dimensions but "
                    f"EMBEDDING_DIMENSIONS is {self.dimensions} (model "
                    f"{self.model!r}). Qdrant would reject the upsert after the "
                    "spend; fix the setting or re-embed the corpus.",
                    details={
                        "returned": len(raw),
                        "expected": self.dimensions,
                        "model": self.model,
                    },
                )
            vectors.append([float(value) for value in raw])
        return vectors


class FakeEmbeddingProvider:
    """Deterministic offline `EmbeddingProvider`.

    Same text in, same unit vector out, on every machine and every run -- which
    is what lets a retrieval test assert on ordering at all. Vectors are
    normalized so cosine similarity behaves the way the real ones do: identical
    texts score 1.0, unrelated texts hover near zero, and a similarity threshold
    tuned against this fake means something.

    `batches` records the exact grouping it was called with, so a caller's
    batching can be asserted without a mock.
    """

    def __init__(self, *, dimensions: int = 8, model: str = "fake-embed-v1") -> None:
        self.model = model
        self.dimensions = dimensions
        self.batches: list[list[str]] = []
        self.closed = False

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.batches.append(batch)
        return [self.vector_for(text) for text in batch]

    async def aclose(self) -> None:
        self.closed = True

    def vector_for(self, text: str) -> list[float]:
        """The vector this fake will always produce for `text`.

        Exposed so a test can assert on a stored vector without re-deriving the
        hashing scheme, which would couple the test to this implementation.

        Seeded from a sha256 digest rather than from `hash()`: `hash()` on `str`
        is salted per process, so a fixture written on Monday would fail on
        Tuesday for no visible reason.
        """
        seed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(seed)
        raw = [rng.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
        norm = math.sqrt(sum(value * value for value in raw))
        if norm == 0.0:  # pragma: no cover -- unreachable for any real digest
            return [0.0] * self.dimensions
        return [value / norm for value in raw]


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
