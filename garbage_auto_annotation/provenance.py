from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_record(
    name: str,
    doi: str,
    url: str,
    files: list[Path],
    license_name: str,
) -> dict:
    """Build reproducibility metadata for downloaded source files."""
    file_records = []
    for file_path in files:
        resolved = file_path.resolve()
        file_records.append(
            {
                "name": resolved.name,
                "path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )

    return {
        "name": name,
        "doi": doi,
        "url": url,
        "license": license_name,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_records,
    }
