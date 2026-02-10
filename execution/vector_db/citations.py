"""
Citation Extraction for Sentence-Level Source Attribution.

Extracts fine-grained citations from retrieved chunks to enable
sentence-level source attribution in generated content. Splits
chunks into sentences using pysbd, assigns unique citation IDs,
and provides formatting utilities for LLM prompts and markdown.

Key concepts:
- Citation ID format: "{chunk_id}.{sentence_idx}" (e.g., "42.3")
- Sentence splitting: Uses pysbd for 97.92% accuracy
- Context blocks: Format citations for LLM prompts
- Citation maps: Lookup structures for resolving references

Usage:
    from execution.vector_db.citations import CitationExtractor, Citation

    extractor = CitationExtractor()

    # Extract citations from a chunk
    chunk = {
        'id': 42,
        'content': 'PostgreSQL is great. pgvector adds vector support.',
        'title': 'pgvector Guide',
        'url': 'https://example.com'
    }
    citations = extractor.extract_citations(chunk)
    # [Citation(citation_id='42.0', ...), Citation(citation_id='42.1', ...)]

    # Format for LLM prompt
    context = extractor.format_context_block(citations)
    # "[42.0] PostgreSQL is great.\n[42.1] pgvector adds vector support."

    # Build lookup map
    citation_map = extractor.build_citation_map(citations)
    # {'42.0': Citation(...), '42.1': Citation(...)}
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional

import pysbd


# Module-level sentence segmenter (reused across calls, matching chunking.py pattern)
_segmenter = pysbd.Segmenter(language="en", clean=False)


@dataclass
class Citation:
    """A single sentence with citation metadata for source attribution.

    Attributes:
        text: The sentence text.
        citation_id: Unique identifier "{chunk_id}.{sentence_idx}" (e.g., "42.3").
        chunk_id: Parent chunk ID.
        sentence_idx: Sentence index within chunk (0-based).
        title: Document title (optional).
        url: Document URL (optional).
        source_type: Source type (email, rss, paper) (optional).
        date_published: Document publication date (optional).

    Example:
        citation = Citation(
            text="PostgreSQL handles vector indexing well.",
            citation_id="42.0",
            chunk_id=42,
            sentence_idx=0,
            title="pgvector Guide",
            url="https://example.com/pgvector",
            source_type="rss",
            date_published=datetime(2026, 1, 15)
        )
    """
    text: str
    citation_id: str
    chunk_id: int
    sentence_idx: int
    title: Optional[str] = None
    url: Optional[str] = None
    source_type: Optional[str] = None
    date_published: Optional[datetime] = None


class CitationExtractor:
    """Extract sentence-level citations from retrieved chunks.

    Uses pysbd for sentence splitting (97.92% accuracy, handles edge cases
    like abbreviations, decimals, URLs). Assigns unique citation IDs in
    format "{chunk_id}.{sentence_idx}".

    Example:
        extractor = CitationExtractor()

        # From single chunk
        citations = extractor.extract_citations(chunk_dict)

        # From search results
        citations = extractor.extract_from_results(results_list)

        # Format for LLM
        context = extractor.format_context_block(citations)
    """

    def __init__(self):
        """Initialize CitationExtractor with shared sentence segmenter."""
        # Use module-level segmenter for efficiency
        self.segmenter = _segmenter

    def extract_citations(self, chunk: Dict) -> List[Citation]:
        """Split chunk into sentences and create Citation objects.

        Args:
            chunk: Dict with keys:
                - id (int): Chunk ID
                - content (str): Chunk text
                - title (str, optional): Document title
                - url (str, optional): Document URL
                - source_type (str, optional): Source type
                - date_published (datetime, optional): Publication date

        Returns:
            List of Citation objects, one per sentence.
            Empty list if content is empty or missing.

        Example:
            chunk = {
                'id': 42,
                'content': 'First sentence. Second sentence.',
                'title': 'Example',
                'url': 'https://example.com'
            }
            citations = extractor.extract_citations(chunk)
            # [Citation(citation_id='42.0', ...), Citation(citation_id='42.1', ...)]
        """
        chunk_id = chunk.get("id")
        content = chunk.get("content", "")

        # Handle empty content
        if not content or not content.strip():
            return []

        # Extract metadata
        title = chunk.get("title")
        url = chunk.get("url")
        source_type = chunk.get("source_type")
        date_published = chunk.get("date_published")

        # Split into sentences using pysbd
        sentences = self.segmenter.segment(content)

        # Create Citation objects
        citations = []
        for idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                # Skip empty sentences
                continue

            citation = Citation(
                text=sentence,
                citation_id=f"{chunk_id}.{idx}",
                chunk_id=chunk_id,
                sentence_idx=idx,
                title=title,
                url=url,
                source_type=source_type,
                date_published=date_published
            )
            citations.append(citation)

        return citations

    def extract_from_results(self, results: List[Dict]) -> List[Citation]:
        """Extract citations from multiple search result chunks.

        Convenience method that calls extract_citations() for each result
        and returns a flat list preserving ordering (chunk order, then
        sentence order within each chunk).

        Args:
            results: List of chunk dicts (same format as extract_citations).

        Returns:
            Flat list of all citations from all chunks.

        Example:
            results = [chunk1_dict, chunk2_dict, chunk3_dict]
            citations = extractor.extract_from_results(results)
            # All sentences from all chunks, in order
        """
        all_citations = []
        for chunk in results:
            citations = self.extract_citations(chunk)
            all_citations.extend(citations)

        return all_citations

    def format_citation_markdown(self, citation: Citation) -> str:
        """Format citation as markdown link.

        Format: [Title](URL)
        - If title is None, uses "Source"
        - If URL is None, uses "#"

        Args:
            citation: Citation object to format.

        Returns:
            Markdown link string.

        Example:
            citation = Citation(..., title="Example", url="https://example.com")
            link = extractor.format_citation_markdown(citation)
            # "[Example](https://example.com)"
        """
        title = citation.title if citation.title else "Source"
        url = citation.url if citation.url else "#"
        return f"[{title}]({url})"

    def format_context_block(self, citations: List[Citation]) -> str:
        """Build context block for LLM prompts with citation IDs.

        Formats each citation as "[{citation_id}] {text}" separated by newlines.
        This is what gets injected into RAG prompts so the LLM can reference
        specific citation IDs in its response.

        Args:
            citations: List of Citation objects.

        Returns:
            Multi-line string with numbered citations.

        Example:
            citations = [
                Citation(citation_id='42.0', text='First sentence.'),
                Citation(citation_id='42.1', text='Second sentence.'),
            ]
            block = extractor.format_context_block(citations)
            # "[42.0] First sentence.\n[42.1] Second sentence."

        Usage in LLM prompt:
            prompt = f'''Answer using the context below. Cite sources with [citation_id].

            Context:
            {context_block}

            Question: {query}
            Answer (with citations):'''
        """
        lines = []
        for citation in citations:
            lines.append(f"[{citation.citation_id}] {citation.text}")

        return "\n".join(lines)

    def build_citation_map(self, citations: List[Citation]) -> Dict[str, Citation]:
        """Build lookup dictionary from citation_id to Citation object.

        Used for resolving citation references in LLM output back to
        source metadata (url, title, sentence).

        Args:
            citations: List of Citation objects.

        Returns:
            Dictionary mapping citation_id (str) to Citation object.

        Example:
            citations = [Citation(citation_id='42.0', ...), ...]
            citation_map = extractor.build_citation_map(citations)
            # {'42.0': Citation(...), '42.1': Citation(...)}

            # Later, resolve LLM citation reference
            if '[42.0]' in llm_response:
                source = citation_map['42.0']
                print(source.url, source.title)
        """
        return {citation.citation_id: citation for citation in citations}
