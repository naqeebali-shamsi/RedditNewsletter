"""
Semantic Chunking Pipeline for the Vector Knowledge Base.

Splits raw documents (emails, papers, RSS items) into semantically
meaningful chunks with content-type-aware strategies. Rule-based
chunking using structural heuristics — deterministic and testable.

Usage:
    from execution.vector_db.chunking import chunk_content, SemanticChunker

    chunks = chunk_content("Some long article text...", content_type="rss")
    for c in chunks:
        print(c.chunk_index, c.token_count, c.text[:80])
"""

import logging
import re
from dataclasses import dataclass, field

import html2text
import pysbd

from execution.config import config

logger = logging.getLogger(__name__)

# Module-level sentence segmenter (reused across calls)
_segmenter = pysbd.Segmenter(language="en", clean=False)


@dataclass
class Chunk:
    """A single chunk of text extracted from a document.

    Attributes:
        text: The chunk content.
        chunk_index: Order within the parent document.
        token_count: Estimated token count (1 token per 4 chars).
        topic_hint: Optional topic label from boundary detection.
        metadata: Optional extra metadata.
    """
    text: str
    chunk_index: int
    token_count: int
    topic_hint: str = ""
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """Content-type-aware semantic chunker.

    Routes to content-type-specific strategies with appropriate
    chunk sizes, boundary detection, and overlap handling.
    """

    # Content-type to chunk size mapping (tokens)
    _CHUNK_SIZES = {
        "email": config.vector_db.CHUNK_SIZE_EMAIL,
        "paper": config.vector_db.CHUNK_SIZE_PAPER,
        "rss": config.vector_db.CHUNK_SIZE_RSS,
        "default": config.vector_db.CHUNK_SIZE_DEFAULT,
    }

    # Content-type to overlap percentage
    _OVERLAP_PCTS = {
        "email": 0.12,
        "paper": config.vector_db.CHUNK_OVERLAP_PERCENT,
        "rss": 0.10,
        "default": config.vector_db.CHUNK_OVERLAP_PERCENT,
    }

    def __init__(
        self,
        default_chunk_size: int | None = None,
        overlap_percent: float | None = None,
    ):
        self._default_chunk_size = default_chunk_size or config.vector_db.CHUNK_SIZE_DEFAULT
        self._default_overlap = overlap_percent or config.vector_db.CHUNK_OVERLAP_PERCENT

    def chunk_content(self, content: str, content_type: str = "default") -> list[Chunk]:
        """Split content into semantic chunks based on content type.

        Args:
            content: Raw text to chunk.
            content_type: One of "email", "paper", "rss", "default".

        Returns:
            Ordered list of Chunk objects with sequential chunk_index.
        """
        if not content or not content.strip():
            return []

        strategy = {
            "email": self._chunk_email,
            "paper": self._chunk_paper,
            "rss": self._chunk_rss,
        }.get(content_type, self._chunk_default)

        chunks = strategy(content)

        # Re-index sequentially
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    # ------------------------------------------------------------------
    # Content-type strategies
    # ------------------------------------------------------------------

    def _chunk_email(self, content: str) -> list[Chunk]:
        """Chunk email content. Target: 400 tokens."""
        target = self._CHUNK_SIZES["email"]
        overlap_pct = self._OVERLAP_PCTS["email"]

        # Convert HTML to text if content looks like HTML
        if self._looks_like_html(content):
            content = self._html_to_text(content)

        # Strip email boilerplate
        content = self._strip_email_boilerplate(content)

        chunks = self._split_by_paragraphs_then_sentences(content, target)
        chunks = self._merge_small_chunks(chunks)
        chunks = self._add_overlap(chunks, overlap_pct)
        return chunks

    def _chunk_paper(self, content: str) -> list[Chunk]:
        """Chunk academic paper content. Target: 512 tokens."""
        target = self._CHUNK_SIZES["paper"]
        overlap_pct = self._OVERLAP_PCTS["paper"]

        sections = self._split_into_sections(content)

        chunks: list[Chunk] = []
        for section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            token_count = self._estimate_tokens(section_text)
            if token_count <= target:
                chunks.append(Chunk(
                    text=section_text,
                    chunk_index=0,
                    token_count=token_count,
                ))
            else:
                chunks.extend(
                    self._split_by_paragraphs_then_sentences(section_text, target)
                )

        chunks = self._merge_small_chunks(chunks)
        chunks = self._add_overlap(chunks, overlap_pct)
        return chunks

    def _chunk_rss(self, content: str) -> list[Chunk]:
        """Chunk RSS item content. Target: 256 tokens."""
        target = self._CHUNK_SIZES["rss"]
        overlap_pct = self._OVERLAP_PCTS["rss"]

        # RSS items are typically short
        token_count = self._estimate_tokens(content)
        if token_count <= target:
            return [Chunk(text=content.strip(), chunk_index=0, token_count=token_count)]

        chunks = self._split_by_paragraphs_then_sentences(content, target)
        chunks = self._merge_small_chunks(chunks)
        chunks = self._add_overlap(chunks, overlap_pct)
        return chunks

    def _chunk_default(self, content: str) -> list[Chunk]:
        """Chunk generic content. Target: 400 tokens, 15% overlap."""
        target = self._CHUNK_SIZES["default"]
        overlap_pct = self._OVERLAP_PCTS["default"]

        chunks = self._split_by_paragraphs_then_sentences(content, target)
        chunks = self._merge_small_chunks(chunks)
        chunks = self._add_overlap(chunks, overlap_pct)
        return chunks

    # ------------------------------------------------------------------
    # Core splitting logic
    # ------------------------------------------------------------------

    def _split_by_paragraphs_then_sentences(
        self, text: str, target_tokens: int
    ) -> list[Chunk]:
        """Split text on paragraph boundaries, then sentences if paragraphs exceed target."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        chunks: list[Chunk] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            token_count = self._estimate_tokens(para)
            if token_count <= target_tokens:
                chunks.append(Chunk(text=para, chunk_index=0, token_count=token_count))
            else:
                # Paragraph too long -- split on sentences
                sentences = self._split_into_sentences(para)
                current_text = ""
                current_tokens = 0

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    sent_tokens = self._estimate_tokens(sentence)

                    if current_tokens + sent_tokens > target_tokens and current_text:
                        chunks.append(Chunk(
                            text=current_text.strip(),
                            chunk_index=0,
                            token_count=current_tokens,
                        ))
                        current_text = sentence
                        current_tokens = sent_tokens
                    else:
                        current_text = (current_text + " " + sentence).strip()
                        current_tokens += sent_tokens

                if current_text.strip():
                    chunks.append(Chunk(
                        text=current_text.strip(),
                        chunk_index=0,
                        token_count=self._estimate_tokens(current_text),
                    ))

        return chunks

    def _split_into_sections(self, text: str) -> list[str]:
        """Split paper-like content on section headers.

        Recognizes:
        - Markdown headers (# ...)
        - ALL CAPS lines (e.g., ABSTRACT, INTRODUCTION)
        - Numbered sections (1. ..., 1.1 ...)
        """
        # Pattern: line that looks like a section header
        header_pattern = re.compile(
            r"^(?:"
            r"#{1,6}\s+.+"           # Markdown headers
            r"|[A-Z][A-Z\s]{3,}$"   # ALL CAPS lines (4+ chars)
            r"|\d+\.[\d.]*\s+\S+"   # Numbered sections
            r")",
            re.MULTILINE,
        )

        matches = list(header_pattern.finditer(text))
        if not matches:
            return [text]

        sections: list[str] = []
        # Content before first header
        if matches[0].start() > 0:
            sections.append(text[: matches[0].start()])

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sections.append(text[start:end])

        return sections

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences using pysbd.

        Handles abbreviations (Dr., Mr.), decimal numbers ($3.50),
        URLs, and other edge cases with 97.92% accuracy.
        """
        return _segmenter.segment(text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count: ~1 token per 4 characters."""
        return max(1, len(text) // 4)

    @staticmethod
    def _looks_like_html(content: str) -> bool:
        """Check if content appears to be HTML."""
        return bool(re.search(r"<(?:html|body|div|p|table|br|a\s)[>\s/]", content, re.IGNORECASE))

    @staticmethod
    def _html_to_text(html_content: str) -> str:
        """Convert HTML to clean text using html2text."""
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.body_width = 0  # Don't wrap lines
        return h.handle(html_content)

    @staticmethod
    def _merge_small_chunks(
        chunks: list[Chunk], min_tokens: int = 50
    ) -> list[Chunk]:
        """Merge chunks smaller than min_tokens with adjacent chunks."""
        if len(chunks) <= 1:
            return chunks

        merged: list[Chunk] = []
        for chunk in chunks:
            if merged and chunk.token_count < min_tokens:
                # Merge with previous chunk
                prev = merged[-1]
                combined_text = prev.text + "\n\n" + chunk.text
                merged[-1] = Chunk(
                    text=combined_text,
                    chunk_index=0,
                    token_count=SemanticChunker._estimate_tokens(combined_text),
                    topic_hint=prev.topic_hint,
                    metadata=prev.metadata,
                )
            elif not merged and chunk.token_count < min_tokens and len(chunks) > 1:
                # First chunk is small -- hold it, will merge with next
                merged.append(chunk)
            else:
                merged.append(chunk)

        # If last chunk became too small after processing, merge with previous
        if len(merged) > 1 and merged[-1].token_count < min_tokens:
            last = merged.pop()
            prev = merged[-1]
            combined_text = prev.text + "\n\n" + last.text
            merged[-1] = Chunk(
                text=combined_text,
                chunk_index=0,
                token_count=SemanticChunker._estimate_tokens(combined_text),
                topic_hint=prev.topic_hint,
                metadata=prev.metadata,
            )

        return merged

    @staticmethod
    def _add_overlap(chunks: list[Chunk], overlap_pct: float) -> list[Chunk]:
        """Prepend overlap from previous chunk to each chunk (except first).

        Args:
            chunks: List of chunks to add overlap to.
            overlap_pct: Fraction of previous chunk to prepend (0.0-1.0).
        """
        if len(chunks) <= 1 or overlap_pct <= 0:
            return chunks

        result: list[Chunk] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            overlap_chars = int(len(prev_text) * overlap_pct)

            if overlap_chars > 0:
                # Find a word boundary near the overlap point
                overlap_start = max(0, len(prev_text) - overlap_chars)
                # Snap forward to next space to avoid splitting words
                space_pos = prev_text.find(" ", overlap_start)
                if space_pos != -1 and space_pos < len(prev_text):
                    overlap_start = space_pos + 1

                overlap_text = prev_text[overlap_start:]
                combined = overlap_text + " " + chunks[i].text
                result.append(Chunk(
                    text=combined,
                    chunk_index=0,
                    token_count=SemanticChunker._estimate_tokens(combined),
                    topic_hint=chunks[i].topic_hint,
                    metadata=chunks[i].metadata,
                ))
            else:
                result.append(chunks[i])

        return result

    @staticmethod
    def _strip_email_boilerplate(content: str) -> str:
        """Remove common email headers, footers, and unsubscribe blocks.

        Strips:
        - "View in browser" / "View online" links
        - Unsubscribe blocks
        - Email signatures (after ---)
        - Forward/reply headers
        - Common footer patterns
        """
        lines = content.split("\n")
        cleaned: list[str] = []
        skip_rest = False

        for line in lines:
            stripped = line.strip().lower()

            # Skip common header boilerplate
            if stripped in ("view in browser", "view online", "view in your browser"):
                continue

            # Stop at signature markers or unsubscribe blocks
            if stripped == "---" or stripped == "-- ":
                skip_rest = True
                continue
            if any(kw in stripped for kw in [
                "unsubscribe",
                "manage your preferences",
                "email preferences",
                "update your preferences",
                "opt out",
                "no longer wish to receive",
                "sent to you because",
                "you are receiving this",
                "this email was sent",
                "copyright ©",
                "all rights reserved",
            ]):
                skip_rest = True
                continue

            if skip_rest:
                continue

            cleaned.append(line)

        result = "\n".join(cleaned).strip()
        return result if result else content.strip()


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------

_singleton: SemanticChunker | None = None


def chunk_content(content: str, content_type: str = "default") -> list[Chunk]:
    """Chunk content using a shared SemanticChunker instance.

    Args:
        content: Raw text to chunk.
        content_type: One of "email", "paper", "rss", "default".

    Returns:
        Ordered list of Chunk objects.
    """
    global _singleton
    if _singleton is None:
        _singleton = SemanticChunker()
    return _singleton.chunk_content(content, content_type)
