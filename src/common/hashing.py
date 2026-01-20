import hashlib
from pathlib import Path


def file_hash(path: str) -> str:
    """SHA-256 (16-char prefix) of any file - used for change detection and ID derivation."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
