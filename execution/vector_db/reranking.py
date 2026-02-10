"""
CrossEncoder Reranking Module.

Implements precision reranking using sentence-transformers CrossEncoder models.
Reranks retrieval candidates by query-document relevance to improve top-K results.

Two-stage retrieval pattern:
1. Hybrid retrieval (BM25 + vector) returns 50-100 candidates
2. CrossEncoder reranks to top-10 by fine-grained relevance

Usage:
    from execution.vector_db.reranking import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_profile="balanced")
    results = reranker.rerank(
        query="How does pgvector handle vector indexing?",
        candidates=[{"id": 1, "content": "..."},...],
        top_k=10
    )
"""

import logging
import time
from typing import Dict, List

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieval candidates using CrossEncoder models.

    Provides three model profiles trading off speed vs accuracy:
    - fast: TinyBERT-L2 (9000 docs/sec, 69.84 NDCG)
    - balanced: MiniLM-L6 (1800 docs/sec, 74.30 NDCG) [DEFAULT]
    - accurate: MiniLM-L12 (960 docs/sec, 74.31 NDCG)

    Lazy loading: Model only loaded on first rerank() call to avoid
    slow import during module initialization.
    """

    # MS-MARCO pre-trained CrossEncoder models
    MODELS = {
        "fast": "cross-encoder/ms-marco-TinyBERT-L2-v2",      # 9000 docs/sec, 69.84 NDCG
        "balanced": "cross-encoder/ms-marco-MiniLM-L6-v2",    # 1800 docs/sec, 74.30 NDCG
        "accurate": "cross-encoder/ms-marco-MiniLM-L12-v2",   # 960 docs/sec, 74.31 NDCG
    }

    def __init__(self, model_profile: str = "balanced"):
        """Initialize reranker with model profile.

        Model is NOT loaded until first rerank() call (lazy loading).

        Args:
            model_profile: One of "fast", "balanced", or "accurate".
        """
        if model_profile not in self.MODELS:
            logger.warning(
                f"Unknown model profile '{model_profile}', "
                f"using 'balanced'. Valid options: {list(self.MODELS.keys())}"
            )
            model_profile = "balanced"

        self.model_profile = model_profile
        self.model_name = self.MODELS[model_profile]
        self._model = None  # Lazy loading

    def _ensure_model(self):
        """Load CrossEncoder model if not yet loaded (lazy loading)."""
        if self._model is not None:
            return

        # Import inside method to avoid slow module-level import
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading CrossEncoder: {self.model_name} (profile: {self.model_profile})")
        self._model = CrossEncoder(self.model_name)
        logger.info(f"CrossEncoder loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 10,
        batch_size: int = 32,
        timeout_ms: int = 5000
    ) -> List[Dict]:
        """Rerank candidates by query-document relevance.

        Args:
            query: User query string.
            candidates: List of dicts with "content" key (str). Other keys preserved.
            top_k: Number of top results to return after reranking.
            batch_size: Batch size for scoring (default 32).
            timeout_ms: Log warning if reranking exceeds this time (default 5000ms).

        Returns:
            Top-K reranked results with "rerank_score" (float) added to each dict.
            Sorted by rerank_score descending.
        """
        self._ensure_model()  # Lazy load model

        if not candidates:
            return []

        # Cap candidates at 100 to control latency
        max_candidates = 100
        if len(candidates) > max_candidates:
            logger.warning(
                f"Limiting reranking from {len(candidates)} to {max_candidates} candidates"
            )
            candidates = candidates[:max_candidates]

        # Build query-document pairs, truncate content to 2000 chars
        pairs = [(query, candidate["content"][:2000]) for candidate in candidates]

        # Score all pairs
        start = time.time()
        scores = self._model.predict(pairs, batch_size=batch_size)
        elapsed_ms = (time.time() - start) * 1000

        # Log performance
        docs_per_sec = len(candidates) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        logger.info(
            f"CrossEncoder reranked {len(candidates)} candidates in {elapsed_ms:.0f}ms "
            f"({docs_per_sec:.0f} docs/sec)"
        )

        # Warn if exceeded timeout
        if elapsed_ms > timeout_ms:
            logger.warning(
                f"Reranking exceeded timeout ({elapsed_ms:.0f}ms > {timeout_ms}ms)"
            )

        # Add rerank scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        # Sort by rerank score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]

    def rank_passages(self, query: str, passages: List[str]) -> List[Dict]:
        """Rank passages by relevance to query (simplified interface).

        Useful for ad-hoc ranking and testing, not part of main retrieval pipeline.

        Args:
            query: User query string.
            passages: List of passage strings.

        Returns:
            List of dicts with keys: corpus_id (int), score (float).
            Sorted by score descending.
        """
        self._ensure_model()  # Lazy load model

        if not passages:
            return []

        # Build pairs
        pairs = [(query, passage) for passage in passages]

        # Score
        scores = self._model.predict(pairs)

        # Build result list
        results = [
            {"corpus_id": i, "score": float(score)}
            for i, score in enumerate(scores)
        ]

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results
