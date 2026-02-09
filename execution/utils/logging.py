"""Structured logging for GhostWriter pipeline.

Uses structlog for JSON-formatted, machine-readable logs.
Replaces SQLite-based audit logging for better observability.
Falls back to stdlib logging if structlog is not installed.
"""
import logging
import logging.handlers
from pathlib import Path

try:
    import structlog
    _HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    _HAS_STRUCTLOG = False


def configure_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    """Configure structured logging with JSON renderer and file output.

    Args:
        log_dir: Directory for log files. Created if it doesn't exist.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # File handler with rotation (10MB, keep 5)
    file_handler = logging.handlers.RotatingFileHandler(
        log_path / "ghostwriter.jsonl",
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[file_handler],
    )

    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str):
    """Get a named logger.

    Returns structlog BoundLogger if available, otherwise stdlib logger.

    Args:
        name: Logger name, typically the module or component name.

    Returns:
        A logger instance (structlog or stdlib).
    """
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)
