"""Timezone-aware datetime utilities.

Replaces deprecated datetime.utcnow() with timezone-aware alternatives.
Uses stdlib only (no arrow/pendulum dependencies).
"""
from datetime import datetime, timezone, timedelta


def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime.

    Replaces deprecated datetime.utcnow() which returns naive datetime.
    """
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    """Get current UTC time as ISO 8601 string."""
    return utc_now().isoformat()


def parse_iso(s: str) -> datetime:
    """Parse ISO 8601 string to timezone-aware datetime.

    If the string has no timezone info, assumes UTC.
    Returns utc_now() for empty or unparseable strings.
    """
    if not s:
        return utc_now()
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return utc_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"
