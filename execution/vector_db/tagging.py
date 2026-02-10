"""
AI Auto-Tagging for the Vector Knowledge Base.

Extracts topic tags, named entities, content type classification,
and confidence scores from text content using LLM-powered analysis.
Uses BaseAgent for multi-provider LLM routing.

Usage:
    from execution.vector_db.tagging import auto_tag, AutoTagger

    result = auto_tag("Article about PostgreSQL and vector search...", source_type="rss")
    print(result.topic_tags)   # ["PostgreSQL", "vector search", "databases"]
    print(result.entities)     # [{"type": "TECH", "value": "PostgreSQL"}]
"""

import json
import logging
import re
from dataclasses import dataclass, field

from execution.config import config

logger = logging.getLogger(__name__)

# Maximum characters of content sent to LLM for tagging
_MAX_CONTENT_CHARS = 4000


@dataclass
class TagResult:
    """Structured result from AI auto-tagging.

    Attributes:
        topic_tags: 3-7 topic keywords (e.g., ["AI", "databases", "Python"]).
        entities: Named entities as [{"type": "ORG|PERSON|TECH|CONCEPT", "value": "name"}].
        source_type_label: Inferred content category (newsletter, research, news, tutorial, opinion).
        confidence: 0-1, how confident the tagger is in its classification.
    """
    topic_tags: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    source_type_label: str = "unknown"
    confidence: float = 0.0


class AutoTagger:
    """LLM-powered content tagger using BaseAgent for multi-provider routing.

    Extracts structured topic tags, named entities, and content type
    classification from text. Gracefully degrades on LLM failures.
    """

    def __init__(self, model: str | None = None, provider: str | None = None):
        from execution.agents.base_agent import BaseAgent

        self._agent = BaseAgent(
            role="Content Tagger",
            persona=(
                "You are a content classification expert. "
                "Extract topics and entities from text. "
                "Always respond with valid JSON only."
            ),
            model=model or config.models.DEFAULT_FAST_MODEL,
            provider=provider,
        )

    def tag_content(self, content: str, source_type: str = "unknown") -> TagResult:
        """Tag a single piece of content with topics, entities, and classification.

        Args:
            content: Text content to analyze.
            source_type: Hint about content origin (email, rss, paper, unknown).

        Returns:
            TagResult with extracted information. Returns empty TagResult on failure.
        """
        if not content or not content.strip():
            return TagResult()

        truncated = content[:_MAX_CONTENT_CHARS]

        prompt = (
            f"Analyze this {source_type} content and extract:\n"
            f"1. Topic tags: 3-7 keywords (e.g., AI, databases, Python, cloud computing)\n"
            f"2. Named entities: organizations, people, technologies mentioned\n"
            f"3. Content type: newsletter, research, news, tutorial, or opinion\n"
            f"4. Confidence: 0-1 how confident you are\n"
            f"\n"
            f'Return ONLY valid JSON:\n'
            f'{{"topics": ["tag1", "tag2"], "entities": [{{"type": "ORG|PERSON|TECH|CONCEPT", "value": "name"}}], "content_type": "category", "confidence": 0.9}}\n'
            f"\n"
            f"Content:\n"
            f"{truncated}"
        )

        try:
            response = self._agent.call_llm(prompt, temperature=0.3)
            return self._parse_response(response)
        except Exception as e:
            logger.warning("Auto-tagging failed: %s", e)
            return self._fallback_parse(str(e))

    def tag_batch(
        self, contents: list[tuple[str, str]]
    ) -> list[TagResult]:
        """Tag multiple content items sequentially.

        Args:
            contents: List of (content, source_type) tuples.

        Returns:
            List of TagResults in same order. Failed items get empty TagResult.
        """
        results: list[TagResult] = []
        total = len(contents)

        for i, (content, source_type) in enumerate(contents, 1):
            logger.info("Tagging %d/%d...", i, total)
            try:
                result = self.tag_content(content, source_type)
                results.append(result)
            except Exception as e:
                logger.error("Tagging item %d/%d failed: %s", i, total, e)
                results.append(TagResult())

        return results

    def _parse_response(self, response: str) -> TagResult:
        """Parse LLM JSON response into TagResult.

        Strips markdown code fences and attempts JSON parsing.
        Falls back to regex extraction on malformed JSON.
        """
        # Strip markdown code fences
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            return self._dict_to_tag_result(data)
        except json.JSONDecodeError:
            pass

        # Fallback: use the json_parser utility
        from execution.utils.json_parser import extract_json_from_llm

        data = extract_json_from_llm(response, default=None)
        if data and isinstance(data, dict):
            return self._dict_to_tag_result(data)

        # Last resort: regex extraction
        return self._regex_extract(response)

    @staticmethod
    def _dict_to_tag_result(data: dict) -> TagResult:
        """Convert a parsed JSON dict to TagResult with validation."""
        topics = data.get("topics", data.get("topic_tags", []))
        if not isinstance(topics, list):
            topics = []
        topics = [str(t) for t in topics if t]

        entities = data.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        # Validate entity structure
        valid_entities = []
        for ent in entities:
            if isinstance(ent, dict) and "type" in ent and "value" in ent:
                valid_entities.append({
                    "type": str(ent["type"]),
                    "value": str(ent["value"]),
                })

        content_type = str(data.get("content_type", data.get("source_type_label", "unknown")))

        confidence = data.get("confidence", 0.0)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.0

        return TagResult(
            topic_tags=topics,
            entities=valid_entities,
            source_type_label=content_type,
            confidence=confidence,
        )

    @staticmethod
    def _regex_extract(response: str) -> TagResult:
        """Best-effort extraction using regex when JSON parsing fails."""
        topics: list[str] = []
        # Look for array-like patterns
        array_match = re.search(r'"topics"\s*:\s*\[(.*?)\]', response, re.DOTALL)
        if array_match:
            raw = array_match.group(1)
            topics = re.findall(r'"([^"]+)"', raw)

        return TagResult(
            topic_tags=topics,
            confidence=0.3 if topics else 0.0,
        )

    @staticmethod
    def _fallback_parse(error_msg: str) -> TagResult:
        """Return empty TagResult on complete failure."""
        return TagResult()


# ------------------------------------------------------------------
# Module-level convenience function
# ------------------------------------------------------------------

_singleton: AutoTagger | None = None


def auto_tag(content: str, source_type: str = "unknown") -> TagResult:
    """Tag content using a shared AutoTagger instance.

    Args:
        content: Text content to analyze.
        source_type: Hint about content origin.

    Returns:
        TagResult with extracted information.
    """
    global _singleton
    if _singleton is None:
        _singleton = AutoTagger()
    return _singleton.tag_content(content, source_type)
