from __future__ import annotations


def extract_touched_files_from_diff(diff: str) -> list[str]:
    touched: list[str] = []
    seen: set[str] = set()
    for raw in diff.splitlines():
        line = raw.strip()
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:]
            if path == "/dev/null" or not path:
                continue
            if path not in seen:
                seen.add(path)
                touched.append(path)
    return touched
