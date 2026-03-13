from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib import request

from autoresearch_rl.target.interface import RunOutcome, TargetAdapter


@dataclass
class HttpTarget(TargetAdapter):
    url: str
    headers: dict[str, str] | None = None
    timeout_s: int = 3600

    def _post(self, payload: dict[str, object]) -> RunOutcome:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self.url, data=data, method="POST")
        for k, v in (self.headers or {}).items():
            req.add_header(k, v)
        req.add_header("Content-Type", "application/json")

        start = time.monotonic()
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        elapsed = time.monotonic() - start
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {}

        metrics = parsed.get("metrics") if isinstance(parsed, dict) else {}
        status = parsed.get("status", "ok") if isinstance(parsed, dict) else "ok"
        return RunOutcome(status=status, metrics=metrics or {}, stdout=body, stderr="", elapsed_s=elapsed, run_dir=parsed.get("run_dir", ""))

    def run(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        return self._post({"mode": "train", "run_dir": run_dir, "params": params})

    def eval(self, *, run_dir: str, params: dict[str, object]) -> RunOutcome:
        return self._post({"mode": "eval", "run_dir": run_dir, "params": params})
