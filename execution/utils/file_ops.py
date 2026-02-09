"""Atomic file I/O operations for critical data.

Prevents corruption from crashes mid-write using temp file + atomic rename.
Use for provenance logs, config, and final article output.
Do NOT use for draft outputs or temp files (unnecessary overhead).
"""
import os
import json
import tempfile
from pathlib import Path
from typing import Any, Union


def atomic_write(filepath: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """Write content atomically using temp file + rename.

    Creates temp file in same directory (ensures same filesystem for atomic rename).
    On failure, cleans up temp file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, str(filepath))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(filepath: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Atomically write JSON data to file."""
    content = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    atomic_write(filepath, content)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_read_json(filepath: Union[str, Path], default: Any = None) -> Any:
    """Read JSON file safely, returning default on any error."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default if default is not None else {}
