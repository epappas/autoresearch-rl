import json
import time
import uuid
from pathlib import Path

from autoresearch_rl.telemetry.rotation import rotate_if_needed


def emit(
    path: str,
    event: dict,
    *,
    run_id: str | None = None,
    schema: str = "v1",
    max_file_size_bytes: int | None = None,
    max_rotated_files: int = 5,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if max_file_size_bytes is not None:
        rotate_if_needed(path, max_file_size_bytes, max_rotated_files)

    enriched = {
        "schema": schema,
        "run_id": run_id,
        "event_id": uuid.uuid4().hex[:12],
        "ts": int(time.time()),
        **event,
    }

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
