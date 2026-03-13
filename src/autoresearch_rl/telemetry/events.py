import json
import time
import uuid
from pathlib import Path


def emit(path: str, event: dict, *, run_id: str | None = None, schema: str = "v1") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    enriched = {
        "schema": schema,
        "run_id": run_id,
        "event_id": uuid.uuid4().hex[:12],
        "ts": int(time.time()),
        **event,
    }

    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
