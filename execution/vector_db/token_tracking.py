"""
Token Usage Tracking and Daily Limit Enforcement.

Tracks daily embedding token usage via a JSON file in the temp directory.
Provides budget checking to prevent runaway API costs during bulk ingestion.
Counters reset automatically at the start of each new day (UTC).

Usage:
    from execution.vector_db.token_tracking import (
        TokenTracker, check_daily_limit, record_usage, estimate_tokens,
    )

    # Check budget before embedding
    if check_daily_limit(estimated_tokens=5000):
        # ... call embedding API ...
        record_usage(actual_tokens)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from execution.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)

# Module-level singleton
_tracker: Optional["TokenTracker"] = None


class TokenTracker:
    """Tracks daily embedding token usage with configurable limits.

    Persists usage counters to a JSON file that resets each day (UTC).

    Args:
        daily_limit: Maximum tokens per day. Reads from config if None.
        usage_file: Path to the JSON usage file. Uses TEMP_DIR default if None.
    """

    def __init__(
        self,
        daily_limit: Optional[int] = None,
        usage_file: Optional[Path] = None,
    ) -> None:
        if daily_limit is None:
            from execution.config import config
            daily_limit = config.vector_db.DAILY_TOKEN_LIMIT

        self.daily_limit = daily_limit

        if usage_file is None:
            from execution.config import config
            self._usage_file = config.paths.TEMP_DIR / "token_usage.json"
        else:
            self._usage_file = usage_file

    def _read_usage(self) -> dict:
        """Read current usage from file, resetting if date has changed.

        Returns:
            Dict with keys: date, tokens_used, requests, last_updated.
        """
        today = utc_now().date().isoformat()
        default = {
            "date": today,
            "tokens_used": 0,
            "requests": 0,
            "last_updated": utc_now().isoformat(),
        }

        if not self._usage_file.exists():
            return default

        try:
            data = json.loads(self._usage_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt token usage file, resetting counters")
            return default

        # Reset if the stored date is not today
        if data.get("date") != today:
            logger.info("New day detected, resetting token counters")
            return default

        return data

    def _write_usage(self, data: dict) -> None:
        """Write usage data to file.

        Args:
            data: Usage dict to persist.
        """
        data["last_updated"] = utc_now().isoformat()
        try:
            self._usage_file.parent.mkdir(parents=True, exist_ok=True)
            self._usage_file.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to write token usage file: %s", exc)

    def get_usage_today(self) -> dict:
        """Return today's token usage summary.

        Returns:
            Dict with tokens_used, requests, remaining, and limit.
        """
        data = self._read_usage()
        tokens_used = data.get("tokens_used", 0)
        return {
            "tokens_used": tokens_used,
            "requests": data.get("requests", 0),
            "remaining": max(0, self.daily_limit - tokens_used),
            "limit": self.daily_limit,
        }

    def check_budget(self, estimated_tokens: int) -> bool:
        """Check whether estimated_tokens fits within the remaining daily budget.

        Logs a warning when usage is within 10% of the limit.

        Args:
            estimated_tokens: Number of tokens the next operation will consume.

        Returns:
            True if the budget allows the operation, False otherwise.
        """
        usage = self.get_usage_today()
        remaining = usage["remaining"]

        if remaining <= 0:
            logger.error(
                "Daily token limit exceeded (%d/%d). Embedding blocked.",
                usage["tokens_used"],
                self.daily_limit,
            )
            return False

        if estimated_tokens > remaining:
            logger.error(
                "Estimated tokens (%d) exceed remaining budget (%d/%d). Embedding blocked.",
                estimated_tokens,
                remaining,
                self.daily_limit,
            )
            return False

        # Warn when within 10% of limit
        threshold = self.daily_limit * 0.1
        if remaining - estimated_tokens < threshold:
            logger.warning(
                "Token budget nearly exhausted: %d remaining after this operation (limit %d)",
                remaining - estimated_tokens,
                self.daily_limit,
            )

        return True

    def record_usage(self, tokens: int) -> None:
        """Add tokens to today's counter and persist to file.

        Args:
            tokens: Number of tokens consumed.
        """
        data = self._read_usage()
        data["tokens_used"] = data.get("tokens_used", 0) + tokens
        data["requests"] = data.get("requests", 0) + 1
        self._write_usage(data)

        logger.info(
            "Recorded %d tokens (total today: %d/%d)",
            tokens,
            data["tokens_used"],
            self.daily_limit,
        )

    def reset(self) -> None:
        """Force reset counters to zero. Intended for testing."""
        self._write_usage({
            "date": utc_now().date().isoformat(),
            "tokens_used": 0,
            "requests": 0,
        })


def _get_tracker() -> "TokenTracker":
    """Get or create the module-level singleton TokenTracker."""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker


def check_daily_limit(estimated_tokens: int) -> bool:
    """Check if estimated_tokens fits within the daily budget.

    Convenience wrapper around the singleton TokenTracker.

    Args:
        estimated_tokens: Number of tokens the next operation will consume.

    Returns:
        True if within budget, False if limit would be exceeded.
    """
    return _get_tracker().check_budget(estimated_tokens)


def record_usage(tokens: int) -> None:
    """Record token usage to today's counter.

    Convenience wrapper around the singleton TokenTracker.

    Args:
        tokens: Number of tokens consumed.
    """
    _get_tracker().record_usage(tokens)


def estimate_tokens(texts: list[str]) -> int:
    """Estimate token count for a list of texts.

    Uses tiktoken (cl100k_base encoding) for accurate counts.
    Falls back to 1 token per 4 characters if tiktoken is unavailable.

    Args:
        texts: List of text strings to estimate.

    Returns:
        Estimated total token count.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        return sum(len(enc.encode(t)) for t in texts)
    except (ImportError, Exception):
        logger.debug("tiktoken unavailable, using 4-char approximation")
        return sum(len(t) // 4 for t in texts)
