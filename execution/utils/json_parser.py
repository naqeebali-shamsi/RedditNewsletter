"""Robust JSON extraction from LLM responses."""
import json
import re
from typing import Any, Optional


def extract_json_from_llm(response: str, default: Any = None) -> Any:
    """Extract JSON from an LLM response that may contain markdown fences or extra text.

    Handles:
    - ```json ... ``` fenced blocks
    - ``` ... ``` generic fenced blocks
    - Plain JSON responses
    - Nested braces/brackets in JSON values
    - Multiple JSON objects (returns first valid one)

    Args:
        response: Raw LLM response string
        default: Value to return if no valid JSON found

    Returns:
        Parsed JSON object, or default if parsing fails
    """
    if not response or not isinstance(response, str):
        return default

    # Strategy 1: Extract from markdown code fences
    # Try ```json ... ``` first, then bare ``` ... ```
    for fence_pattern in [r'```json\s*([\s\S]*?)```', r'```\s*([\s\S]*?)```']:
        for match in re.finditer(fence_pattern, response):
            candidate = match.group(1).strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

    # Strategy 2: Use JSONDecoder.raw_decode to find JSON objects/arrays
    # This correctly handles nested braces unlike rfind("}")
    decoder = json.JSONDecoder()
    text = response.strip()
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            try:
                obj, _ = decoder.raw_decode(text, i)
                return obj
            except json.JSONDecodeError:
                continue

    # Strategy 3: Last resort - try the entire string as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return default
