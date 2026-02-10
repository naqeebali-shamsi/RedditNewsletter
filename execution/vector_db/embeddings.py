"""
OpenAI Embedding Client with Sync and Batch Modes.

Provides EmbeddingClient for converting text to 1536-dimensional vectors
using OpenAI's text-embedding-3-small model. Supports two modes:

- **Synchronous**: For small batches and interactive use (up to 100 texts).
- **Batch API**: For bulk ingestion with 50% cost savings (24h completion window).

Token usage is tracked via TokenTracker with daily budget enforcement.

Usage:
    from execution.vector_db.embeddings import EmbeddingClient, embed_texts

    # Quick single-text embedding
    client = EmbeddingClient()
    vector = client.embed_text("Hello world")

    # Batch mode for bulk ingestion
    vectors = client.batch_embed_texts(large_text_list)
"""

import json
import logging
import time
from typing import List, Optional

import openai
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from execution.vector_db.token_tracking import TokenTracker, estimate_tokens

logger = logging.getLogger(__name__)

# Maximum texts per synchronous embedding request (OpenAI limit)
_MAX_SYNC_BATCH = 100

# Approximate character limit per text (~8191 tokens * 4 chars/token)
_MAX_TEXT_CHARS = 8191 * 4


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class EmbeddingError(Exception):
    """General embedding failure."""
    pass


class TokenBudgetExceeded(EmbeddingError):
    """Daily token limit reached -- embedding blocked."""
    pass


class BatchEmbeddingFailed(EmbeddingError):
    """OpenAI Batch API returned a failure status."""
    pass


# ---------------------------------------------------------------------------
# Transient error detection (mirrors base_agent.py pattern)
# ---------------------------------------------------------------------------

def _is_transient(exc: Exception) -> bool:
    """Check if an OpenAI exception is transient and worth retrying."""
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APITimeoutError):
        return True
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.InternalServerError):
        return True
    return False


def _log_retry(retry_state) -> None:
    """Log each retry attempt."""
    exc = retry_state.outcome.exception()
    attempt = retry_state.attempt_number
    logger.warning("[embedding retry] Attempt %d failed (%s), retrying...", attempt, exc)


# ---------------------------------------------------------------------------
# EmbeddingClient
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """OpenAI embedding client with sync and batch modes.

    Args:
        model: Embedding model name. Defaults to config value.
        api_key: OpenAI API key. Defaults to config value.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        from execution.config import config

        self.model = model or config.vector_db.EMBEDDING_MODEL
        self._api_key = api_key or config.api.OPENAI_API_KEY
        self._client: Optional[openai.OpenAI] = None
        self.tracker = TokenTracker()

    @property
    def client(self) -> openai.OpenAI:
        """Lazy-initialize the OpenAI client on first use."""
        if self._client is None:
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    @staticmethod
    def _truncate_texts(texts: List[str]) -> List[str]:
        """Truncate texts exceeding the model's token limit.

        Args:
            texts: Raw input texts.

        Returns:
            Texts truncated to _MAX_TEXT_CHARS characters.
        """
        result = []
        for text in texts:
            if len(text) > _MAX_TEXT_CHARS:
                logger.warning(
                    "Truncating text from %d to %d chars (model limit)",
                    len(text),
                    _MAX_TEXT_CHARS,
                )
                result.append(text[:_MAX_TEXT_CHARS])
            else:
                result.append(text)
        return result

    def embed_texts(
        self,
        texts: List[str],
        check_budget: bool = True,
    ) -> List[List[float]]:
        """Embed texts synchronously via the OpenAI embeddings endpoint.

        Processes up to 100 texts per call. For larger batches, use
        batch_embed_texts() which provides 50% cost savings.

        Args:
            texts: List of text strings to embed (max 100).
            check_budget: If True, verify daily token budget before calling API.

        Returns:
            List of embedding vectors (each a list of floats).

        Raises:
            TokenBudgetExceeded: Daily token limit would be exceeded.
            EmbeddingError: API call failed after retries.
            ValueError: More than 100 texts provided.
        """
        if len(texts) > _MAX_SYNC_BATCH:
            raise ValueError(
                f"embed_texts supports up to {_MAX_SYNC_BATCH} texts. "
                f"Got {len(texts)}. Use batch_embed_texts for larger batches."
            )

        if not texts:
            return []

        texts = self._truncate_texts(texts)

        if check_budget:
            estimated = estimate_tokens(texts)
            if not self.tracker.check_budget(estimated):
                raise TokenBudgetExceeded(
                    f"Estimated {estimated} tokens exceeds remaining daily budget"
                )

        embeddings = self._call_embeddings_api(texts)
        return embeddings

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string.

        Convenience wrapper around embed_texts.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        return self.embed_texts([text])[0]

    def _call_embeddings_api(self, texts: List[str]) -> List[List[float]]:
        """Call the OpenAI embeddings API with retry logic.

        Args:
            texts: Pre-validated and truncated texts.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: API call failed after retries.
        """
        retrying = retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(0, 2),
            retry=retry_if_exception(_is_transient),
            before_sleep=_log_retry,
            reraise=True,
        )

        def _api_call():
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            # Record actual usage
            actual_tokens = response.usage.total_tokens
            self.tracker.record_usage(actual_tokens)
            logger.info(
                "Embedded %d texts (%d tokens, model=%s)",
                len(texts),
                actual_tokens,
                self.model,
            )
            # Return embeddings in the same order as input
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [d.embedding for d in sorted_data]

        try:
            return retrying(_api_call)()
        except Exception as exc:
            if _is_transient(exc):
                raise EmbeddingError(f"Embedding API failed after retries: {exc}") from exc
            raise EmbeddingError(f"Embedding API error: {exc}") from exc

    def batch_embed_texts(
        self,
        texts: List[str],
        poll_interval: int = 30,
    ) -> List[List[float]]:
        """Embed texts via the OpenAI Batch API for 50% cost savings.

        Creates a JSONL batch file, uploads it, and polls for completion.
        Suitable for bulk ingestion where 10-20 minute latency is acceptable.

        Args:
            texts: List of text strings to embed.
            poll_interval: Seconds between status polls.

        Returns:
            List of embedding vectors in the same order as input.

        Raises:
            TokenBudgetExceeded: Daily token limit would be exceeded.
            BatchEmbeddingFailed: Batch API returned failure status.
            EmbeddingError: General failure during batch processing.
        """
        if not texts:
            return []

        texts = self._truncate_texts(texts)

        # Budget check
        estimated = estimate_tokens(texts)
        if not self.tracker.check_budget(estimated):
            raise TokenBudgetExceeded(
                f"Estimated {estimated} tokens exceeds remaining daily budget"
            )

        try:
            # Build JSONL batch request
            batch_lines = []
            for i, text in enumerate(texts):
                request_obj = {
                    "custom_id": f"emb-{i}",
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "model": self.model,
                        "input": text,
                    },
                }
                batch_lines.append(json.dumps(request_obj))

            jsonl_content = "\n".join(batch_lines)
            logger.info("Uploading batch file with %d embedding requests", len(texts))

            # Upload batch file
            batch_file = self.client.files.create(
                file=jsonl_content.encode("utf-8"),
                purpose="batch",
            )

            # Create batch job
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
            )
            logger.info("Created batch %s (%d texts)", batch.id, len(texts))

            # Poll for completion
            while batch.status not in ("completed", "failed", "cancelled", "expired"):
                time.sleep(poll_interval)
                batch = self.client.batches.retrieve(batch.id)
                completed = batch.request_counts.completed if batch.request_counts else 0
                total = batch.request_counts.total if batch.request_counts else len(texts)
                logger.info(
                    "Batch %s: %s (%d/%d complete)",
                    batch.id,
                    batch.status,
                    completed,
                    total,
                )

            if batch.status != "completed":
                error_detail = ""
                if batch.errors and batch.errors.data:
                    error_detail = "; ".join(
                        e.message for e in batch.errors.data[:5]
                    )
                raise BatchEmbeddingFailed(
                    f"Batch {batch.id} ended with status '{batch.status}': {error_detail}"
                )

            # Download and parse results
            result_content = self.client.files.content(batch.output_file_id)
            result_lines = [
                line for line in result_content.text.split("\n") if line.strip()
            ]

            results = []
            total_tokens = 0
            for line in result_lines:
                obj = json.loads(line)
                results.append(obj)
                # Accumulate token usage from each result
                usage = obj.get("response", {}).get("body", {}).get("usage", {})
                total_tokens += usage.get("total_tokens", 0)

            # Record total token usage
            self.tracker.record_usage(total_tokens)
            logger.info(
                "Batch %s completed: %d embeddings, %d tokens",
                batch.id,
                len(results),
                total_tokens,
            )

            # Sort by custom_id to maintain original order
            results.sort(key=lambda r: int(r["custom_id"].split("-")[1]))
            embeddings = [
                r["response"]["body"]["data"][0]["embedding"] for r in results
            ]
            return embeddings

        except (TokenBudgetExceeded, BatchEmbeddingFailed):
            raise
        except Exception as exc:
            raise EmbeddingError(f"Batch embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_client: Optional[EmbeddingClient] = None


def _get_client() -> EmbeddingClient:
    """Get or create the module-level singleton EmbeddingClient."""
    global _client
    if _client is None:
        _client = EmbeddingClient()
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts synchronously using the singleton client.

    Args:
        texts: List of text strings to embed (max 100).

    Returns:
        List of embedding vectors.
    """
    return _get_client().embed_texts(texts)


def batch_embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts via Batch API using the singleton client.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors.
    """
    return _get_client().batch_embed_texts(texts)
