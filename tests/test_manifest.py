from pathlib import Path

from autoresearch_rl.telemetry.manifest import write_manifest


def test_manifest_written(tmp_path: Path):
    p = write_manifest(str(tmp_path), {"type": "x"})
    assert p.exists()
