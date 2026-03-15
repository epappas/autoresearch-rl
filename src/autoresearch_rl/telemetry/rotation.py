from __future__ import annotations

from pathlib import Path


def rotate_if_needed(
    path: str,
    max_size_bytes: int,
    max_rotated: int,
) -> None:
    """Rotate file if it exceeds max_size_bytes.

    Rotation scheme: path -> path.1 -> path.2 -> ... -> path.N (oldest deleted).
    """
    p = Path(path)
    if not p.exists():
        return
    if p.stat().st_size <= max_size_bytes:
        return

    # Delete the oldest rotated file if it exists
    oldest = Path(f"{path}.{max_rotated}")
    if oldest.exists():
        oldest.unlink()

    # Shift existing rotated files up by one
    for i in range(max_rotated - 1, 0, -1):
        src = Path(f"{path}.{i}")
        dst = Path(f"{path}.{i + 1}")
        if src.exists():
            src.rename(dst)

    # Move the current file to .1
    p.rename(Path(f"{path}.1"))
