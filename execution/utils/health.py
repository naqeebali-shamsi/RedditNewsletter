"""Health check for GhostWriter system dependencies."""

import os
from pathlib import Path

from execution.utils.datetime_utils import utc_iso


def check_health() -> dict:
    """Run health checks on all system dependencies.

    Returns dict with status ("healthy"/"degraded"/"unhealthy"),
    individual check results, and timestamp.
    """
    checks = {
        "database": _check_database(),
        "llm_providers": _check_providers(),
        "filesystem": _check_filesystem(),
        "optional_deps": _check_optional_deps(),
    }

    failed = [k for k, v in checks.items() if v.get("status") == "unhealthy"]
    degraded = [k for k, v in checks.items() if v.get("status") == "degraded"]

    if failed:
        status = "unhealthy"
    elif degraded:
        status = "degraded"
    else:
        status = "healthy"

    return {"status": status, "checks": checks, "timestamp": utc_iso()}


def _check_database() -> dict:
    """Check SQLite database connectivity."""
    try:
        import sqlalchemy as sa
        from execution.sources.database import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        return {"status": "healthy", "detail": "Database connected"}
    except Exception as e:
        return {"status": "unhealthy", "detail": str(e)}


def _check_providers() -> dict:
    """Check that at least one LLM provider API key is configured."""
    providers = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "google": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
    }
    configured = [k for k, v in providers.items() if v]
    if not configured:
        return {
            "status": "unhealthy",
            "detail": "No LLM API keys configured",
            "providers": providers,
        }
    return {
        "status": "healthy",
        "detail": f"{len(configured)} provider(s) configured",
        "providers": providers,
    }


def _check_filesystem() -> dict:
    """Check required directories exist and are writable."""
    dirs_to_check = [".tmp", "output"]
    issues = []
    for d in dirs_to_check:
        p = Path(d)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                issues.append(f"Cannot create {d}: {e}")
    if issues:
        return {"status": "degraded", "detail": "; ".join(issues)}
    return {"status": "healthy", "detail": "All directories accessible"}


def _check_optional_deps() -> dict:
    """Check which optional packages are installed."""
    deps = {}
    for name in ["sklearn", "vaderSentiment", "structlog", "praw", "numpy"]:
        try:
            __import__(name)
            deps[name] = True
        except ImportError:
            deps[name] = False
    installed = sum(v for v in deps.values())
    total = len(deps)
    status = "healthy" if installed >= 3 else "degraded"
    return {
        "status": status,
        "detail": f"{installed}/{total} optional deps installed",
        "packages": deps,
    }
