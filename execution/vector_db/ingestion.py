"""
Ingestion Orchestrator for the Vector Knowledge Base.

Wires the full pipeline: document -> chunk -> tag -> embed -> store.
Supports single-document and batch ingestion with per-item error
isolation, duplicate detection, and incremental re-processing.

Usage:
    from execution.vector_db.ingestion import IngestionPipeline, ingest_document

    # Single document
    doc = ingest_document(
        title="My Article",
        content="Full text...",
        source_type="rss",
        source_id="article-123",
    )

    # Batch
    pipeline = IngestionPipeline()
    result = pipeline.ingest_batch(documents_list)
"""

import logging
import time
from datetime import datetime
from typing import Optional

from sqlalchemy import and_

from execution.utils.datetime_utils import utc_now
from execution.vector_db.chunking import SemanticChunker
from execution.vector_db.connection import get_session
from execution.vector_db.embeddings import EmbeddingClient, TokenBudgetExceeded
from execution.vector_db.models import Document, IngestionLog, KnowledgeChunk
from execution.vector_db.tagging import AutoTagger

logger = logging.getLogger(__name__)

# Maximum retries for embedding failures (non-budget)
_EMBED_RETRIES = 3
_EMBED_RETRY_DELAY = 5  # seconds


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline: chunk -> tag -> embed -> store.

    Args:
        tenant_id: Tenant identifier. Defaults to config default.
    """

    def __init__(self, tenant_id: Optional[str] = None) -> None:
        from execution.config import config

        self.tenant_id = tenant_id or config.vector_db.DEFAULT_TENANT
        self.chunker = SemanticChunker()
        self.embedder = EmbeddingClient()
        self.tagger = AutoTagger()

    def ingest_document(
        self,
        title: str,
        content: str,
        source_type: str,
        source_id: str,
        url: Optional[str] = None,
        date_published: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> Document:
        """Ingest a single document through the full pipeline.

        Steps: check duplicates -> create record -> chunk -> tag -> embed -> store.

        Args:
            title: Document title.
            content: Full text content.
            source_type: Origin type (email, rss, paper, manual).
            source_id: Unique identifier within tenant + source_type.
            url: Source URL if available.
            date_published: Original publication date.
            metadata: Arbitrary metadata dict.

        Returns:
            The Document ORM instance (committed to database).

        Raises:
            Exception: On unrecoverable database errors after rollback.
        """
        with get_session() as session:
            # 1. Check for existing document
            existing = session.query(Document).filter(
                and_(
                    Document.tenant_id == self.tenant_id,
                    Document.source_type == source_type,
                    Document.source_id == source_id,
                )
            ).first()

            if existing:
                if existing.processing_status == "embedded":
                    logger.info("Skipping already-embedded document: %s", title)
                    return existing
                # Reprocess: delete old chunks
                logger.info(
                    "Reprocessing document (status=%s): %s",
                    existing.processing_status,
                    title,
                )
                session.query(KnowledgeChunk).filter(
                    KnowledgeChunk.document_id == existing.id
                ).delete()
                doc = existing
                doc.content = content
                doc.title = title
                doc.url = url
                doc.date_published = date_published
                doc.metadata_ = metadata or {}
                doc.processing_status = "pending"
                doc.error_message = None
            else:
                # 2. Create Document record
                doc = Document(
                    tenant_id=self.tenant_id,
                    source_type=source_type,
                    source_id=source_id,
                    title=title,
                    content=content,
                    url=url,
                    date_published=date_published,
                    metadata_=metadata or {},
                    processing_status="pending",
                )
                session.add(doc)
                session.flush()  # Get doc.id

            # 3. Chunk the content
            try:
                chunks = self.chunker.chunk_content(content, source_type)
            except Exception as exc:
                doc.processing_status = "failed"
                doc.error_message = f"Chunking failed: {exc}"
                logger.error("Chunking failed for '%s': %s", title, exc)
                return doc

            if not chunks:
                doc.processing_status = "failed"
                doc.error_message = "No chunks produced"
                logger.warning("No chunks produced for '%s'", title)
                return doc

            logger.info("Chunked '%s' into %d chunks", title, len(chunks))

            # 4. Tag the full content (non-critical)
            topic_tags = []
            entities = []
            try:
                tags = self.tagger.tag_content(content, source_type)
                topic_tags = tags.topic_tags
                entities = [e for e in tags.entities]
            except Exception as exc:
                logger.warning("Tagging failed for '%s' (non-critical): %s", title, exc)

            # 5. Create KnowledgeChunk records
            chunk_records = []
            for chunk in chunks:
                kc = KnowledgeChunk(
                    tenant_id=self.tenant_id,
                    document_id=doc.id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.text,
                    topic_tags=topic_tags,
                    entities=entities,
                )
                session.add(kc)
                chunk_records.append(kc)

            session.flush()  # Get chunk IDs

            # 6. Embed all chunk texts
            chunk_texts = [c.text for c in chunks]
            try:
                embeddings = self._embed_with_retry(chunk_texts)
            except TokenBudgetExceeded as exc:
                doc.processing_status = "pending"
                doc.error_message = f"Token budget exceeded, will retry: {exc}"
                logger.warning("Token budget exceeded for '%s': %s", title, exc)
                return doc
            except Exception as exc:
                doc.processing_status = "failed"
                doc.error_message = f"Embedding failed: {exc}"
                logger.error("Embedding failed for '%s': %s", title, exc)
                return doc

            # 7. Update chunks with embeddings
            for kc, embedding in zip(chunk_records, embeddings):
                kc.embedding = embedding

            # 8. Mark as embedded
            doc.processing_status = "embedded"
            logger.info(
                "Ingested '%s': %d chunks embedded",
                title,
                len(chunk_records),
            )
            return doc

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with retry on non-budget failures.

        Args:
            texts: Chunk texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            TokenBudgetExceeded: Immediately (no retry).
            Exception: After all retries exhausted.
        """
        last_exc = None
        for attempt in range(1, _EMBED_RETRIES + 1):
            try:
                return self.embedder.embed_texts(texts)
            except TokenBudgetExceeded:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < _EMBED_RETRIES:
                    logger.warning(
                        "Embedding attempt %d/%d failed: %s. Retrying in %ds...",
                        attempt,
                        _EMBED_RETRIES,
                        exc,
                        _EMBED_RETRY_DELAY,
                    )
                    time.sleep(_EMBED_RETRY_DELAY)
        raise last_exc

    def ingest_batch(
        self,
        documents: list[dict],
        use_batch_api: bool = False,
    ) -> dict:
        """Ingest multiple documents with per-item error isolation.

        Args:
            documents: List of dicts with keys matching ingest_document params.
            use_batch_api: If True, use OpenAI Batch API for embedding (50% savings).

        Returns:
            Summary dict: {processed, skipped, failed, errors}.
        """
        summary = {"processed": 0, "skipped": 0, "failed": 0, "errors": []}

        # Create IngestionLog
        with get_session() as session:
            log = IngestionLog(
                tenant_id=self.tenant_id,
                source_type=None,
                status="running",
            )
            session.add(log)
            session.flush()
            log_id = log.id

        if not use_batch_api:
            # Sequential processing
            for i, doc_dict in enumerate(documents, 1):
                logger.info("Processing document %d/%d: %s", i, len(documents), doc_dict.get("title", "?"))
                try:
                    doc = self.ingest_document(**doc_dict)
                    if doc.processing_status == "embedded":
                        summary["processed"] += 1
                    elif doc.processing_status == "pending":
                        summary["skipped"] += 1
                    else:
                        summary["failed"] += 1
                        if doc.error_message:
                            summary["errors"].append(doc.error_message)
                except Exception as exc:
                    summary["failed"] += 1
                    summary["errors"].append(f"{doc_dict.get('title', '?')}: {exc}")
                    logger.error("Failed to ingest '%s': %s", doc_dict.get("title", "?"), exc)
        else:
            # Batch API mode: chunk and tag all first, then embed in one batch
            all_chunks = []
            doc_chunk_map = []  # (doc_record, chunk_records, chunk_texts)

            for doc_dict in documents:
                with get_session() as session:
                    try:
                        doc = self._prepare_document(session, doc_dict)
                        if doc is None:
                            summary["skipped"] += 1
                            continue
                        if doc.processing_status == "failed":
                            summary["failed"] += 1
                            if doc.error_message:
                                summary["errors"].append(doc.error_message)
                            continue

                        chunk_texts = [kc.content for kc in doc.chunks]
                        doc_chunk_map.append((doc.id, chunk_texts))
                        all_chunks.extend(chunk_texts)
                    except Exception as exc:
                        summary["failed"] += 1
                        summary["errors"].append(f"{doc_dict.get('title', '?')}: {exc}")
                        logger.error("Batch prep failed: %s", exc)

            # Embed all chunks at once via Batch API
            if all_chunks:
                try:
                    all_embeddings = self.embedder.batch_embed_texts(all_chunks)
                    # Distribute embeddings back to documents
                    offset = 0
                    for doc_id, chunk_texts in doc_chunk_map:
                        n = len(chunk_texts)
                        doc_embeddings = all_embeddings[offset:offset + n]
                        offset += n
                        self._apply_embeddings(doc_id, doc_embeddings)
                        summary["processed"] += 1
                except Exception as exc:
                    summary["failed"] += len(doc_chunk_map)
                    summary["errors"].append(f"Batch embedding failed: {exc}")
                    logger.error("Batch embedding failed: %s", exc)

        # Update IngestionLog
        with get_session() as session:
            log = session.query(IngestionLog).get(log_id)
            if log:
                log.completed_at = utc_now()
                log.status = "completed" if not summary["errors"] else "failed"
                log.documents_processed = summary["processed"]
                log.chunks_created = summary["processed"]  # Approximate
                log.errors = summary["errors"]

        logger.info(
            "Batch complete: %d processed, %d skipped, %d failed",
            summary["processed"],
            summary["skipped"],
            summary["failed"],
        )
        return summary

    def _prepare_document(self, session, doc_dict: dict) -> Optional[Document]:
        """Prepare a document (chunk + tag) without embedding. For batch mode.

        Returns None if document already embedded (skip).
        """
        title = doc_dict["title"]
        content = doc_dict["content"]
        source_type = doc_dict["source_type"]
        source_id = doc_dict["source_id"]

        existing = session.query(Document).filter(
            and_(
                Document.tenant_id == self.tenant_id,
                Document.source_type == source_type,
                Document.source_id == source_id,
            )
        ).first()

        if existing and existing.processing_status == "embedded":
            logger.info("Skipping already-embedded: %s", title)
            return None

        if existing:
            session.query(KnowledgeChunk).filter(
                KnowledgeChunk.document_id == existing.id
            ).delete()
            doc = existing
            doc.content = content
            doc.title = title
            doc.processing_status = "pending"
            doc.error_message = None
        else:
            doc = Document(
                tenant_id=self.tenant_id,
                source_type=source_type,
                source_id=source_id,
                title=title,
                content=content,
                url=doc_dict.get("url"),
                date_published=doc_dict.get("date_published"),
                metadata_=doc_dict.get("metadata") or {},
                processing_status="pending",
            )
            session.add(doc)
            session.flush()

        # Chunk
        try:
            chunks = self.chunker.chunk_content(content, source_type)
        except Exception as exc:
            doc.processing_status = "failed"
            doc.error_message = f"Chunking failed: {exc}"
            return doc

        if not chunks:
            doc.processing_status = "failed"
            doc.error_message = "No chunks produced"
            return doc

        # Tag
        topic_tags = []
        entities = []
        try:
            tags = self.tagger.tag_content(content, source_type)
            topic_tags = tags.topic_tags
            entities = [e for e in tags.entities]
        except Exception as exc:
            logger.warning("Tagging failed for '%s': %s", title, exc)

        # Create chunk records (without embeddings)
        for chunk in chunks:
            kc = KnowledgeChunk(
                tenant_id=self.tenant_id,
                document_id=doc.id,
                chunk_index=chunk.chunk_index,
                content=chunk.text,
                topic_tags=topic_tags,
                entities=entities,
            )
            session.add(kc)

        doc.processing_status = "chunked"
        session.flush()
        return doc

    def _apply_embeddings(self, doc_id: int, embeddings: list[list[float]]) -> None:
        """Apply embeddings to a document's chunks and mark as embedded."""
        with get_session() as session:
            chunks = (
                session.query(KnowledgeChunk)
                .filter(KnowledgeChunk.document_id == doc_id)
                .order_by(KnowledgeChunk.chunk_index)
                .all()
            )
            for kc, emb in zip(chunks, embeddings):
                kc.embedding = emb

            doc = session.query(Document).get(doc_id)
            if doc:
                doc.processing_status = "embedded"

    def get_pending_documents(
        self, since: Optional[datetime] = None
    ) -> list[Document]:
        """Return documents awaiting processing.

        Args:
            since: Only return documents created after this timestamp.

        Returns:
            List of Document objects with status 'pending' or 'failed'.
        """
        with get_session() as session:
            query = session.query(Document).filter(
                and_(
                    Document.tenant_id == self.tenant_id,
                    Document.processing_status.in_(["pending", "failed"]),
                )
            )
            if since:
                query = query.filter(Document.date_ingested >= since)
            return query.all()

    def reprocess_failed(self) -> dict:
        """Re-run ingestion for all failed documents.

        Returns:
            Summary dict matching ingest_batch format.
        """
        with get_session() as session:
            failed_docs = session.query(Document).filter(
                and_(
                    Document.tenant_id == self.tenant_id,
                    Document.processing_status == "failed",
                )
            ).all()

            doc_dicts = [
                {
                    "title": d.title,
                    "content": d.content,
                    "source_type": d.source_type,
                    "source_id": d.source_id,
                    "url": d.url,
                    "date_published": d.date_published,
                    "metadata": d.metadata_,
                }
                for d in failed_docs
            ]

        if not doc_dicts:
            return {"processed": 0, "skipped": 0, "failed": 0, "errors": []}

        logger.info("Reprocessing %d failed documents", len(doc_dicts))
        return self.ingest_batch(doc_dicts)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_pipeline: Optional[IngestionPipeline] = None


def _get_pipeline() -> IngestionPipeline:
    """Get or create the module-level singleton IngestionPipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = IngestionPipeline()
    return _pipeline


def ingest_document(**kwargs) -> Document:
    """Ingest a single document through the full pipeline.

    Convenience wrapper around the singleton IngestionPipeline.
    See IngestionPipeline.ingest_document for accepted keyword arguments.
    """
    return _get_pipeline().ingest_document(**kwargs)


def ingest_batch(documents: list[dict], **kwargs) -> dict:
    """Ingest multiple documents with per-item error isolation.

    Convenience wrapper around the singleton IngestionPipeline.
    See IngestionPipeline.ingest_batch for accepted keyword arguments.
    """
    return _get_pipeline().ingest_batch(documents, **kwargs)
