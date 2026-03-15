from __future__ import annotations

from pathlib import Path

from autoresearch_rl.telemetry.rotation import rotate_if_needed


def test_no_rotation_when_under_threshold(tmp_path: Path) -> None:
    f = tmp_path / "events.jsonl"
    f.write_text("small content\n")
    rotate_if_needed(str(f), max_size_bytes=1024, max_rotated=3)
    assert f.exists()
    assert not (tmp_path / "events.jsonl.1").exists()


def test_no_rotation_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    rotate_if_needed(str(missing), max_size_bytes=10, max_rotated=3)
    assert not missing.exists()


def test_rotation_triggers_at_threshold(tmp_path: Path) -> None:
    f = tmp_path / "events.jsonl"
    f.write_text("x" * 200)
    rotate_if_needed(str(f), max_size_bytes=100, max_rotated=3)

    assert not f.exists()
    rotated = tmp_path / "events.jsonl.1"
    assert rotated.exists()
    assert rotated.read_text() == "x" * 200


def test_rotation_chain(tmp_path: Path) -> None:
    f = tmp_path / "events.jsonl"

    f.write_text("content_a")
    rotate_if_needed(str(f), max_size_bytes=5, max_rotated=5)
    assert not f.exists()
    assert (tmp_path / "events.jsonl.1").read_text() == "content_a"

    f.write_text("content_b")
    rotate_if_needed(str(f), max_size_bytes=5, max_rotated=5)
    assert (tmp_path / "events.jsonl.1").read_text() == "content_b"
    assert (tmp_path / "events.jsonl.2").read_text() == "content_a"

    f.write_text("content_c")
    rotate_if_needed(str(f), max_size_bytes=5, max_rotated=5)
    assert (tmp_path / "events.jsonl.1").read_text() == "content_c"
    assert (tmp_path / "events.jsonl.2").read_text() == "content_b"
    assert (tmp_path / "events.jsonl.3").read_text() == "content_a"


def test_max_rotated_files_limit(tmp_path: Path) -> None:
    f = tmp_path / "events.jsonl"
    max_rotated = 2

    f.write_text("oldest")
    rotate_if_needed(str(f), max_size_bytes=1, max_rotated=max_rotated)

    f.write_text("middle")
    rotate_if_needed(str(f), max_size_bytes=1, max_rotated=max_rotated)

    f.write_text("newest")
    rotate_if_needed(str(f), max_size_bytes=1, max_rotated=max_rotated)

    assert (tmp_path / "events.jsonl.1").read_text() == "newest"
    assert (tmp_path / "events.jsonl.2").read_text() == "middle"
    assert not (tmp_path / "events.jsonl.3").exists()


def test_rotation_exact_threshold(tmp_path: Path) -> None:
    """File exactly at threshold should not rotate (only strictly over)."""
    f = tmp_path / "events.jsonl"
    f.write_text("x" * 100)
    rotate_if_needed(str(f), max_size_bytes=100, max_rotated=3)
    assert f.exists()
    assert not (tmp_path / "events.jsonl.1").exists()
