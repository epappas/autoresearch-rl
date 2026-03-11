# Errors

## [ERR-20260311-001] pip install (PEP 668 externally-managed-environment)

**Logged**: 2026-03-11T10:02:00Z
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
`pip install -e .[dev]` failed because system Python is externally managed.

### Error
```
error: externally-managed-environment
```

### Context
- Command: `python3 -m pip install -e .[dev]`
- Environment: Debian/Ubuntu managed Python (PEP 668)

### Suggested Fix
Use project virtualenv (`python3 -m venv .venv && . .venv/bin/activate`) before pip installs.

### Metadata
- Reproducible: yes
- Related Files: README.md

---

## [ERR-20260311-002] python venv creation failed (ensurepip unavailable)

**Logged**: 2026-03-11T10:03:00Z
**Priority**: medium
**Status**: pending
**Area**: infra

### Summary
`python3 -m venv .venv` failed because `python3-venv` / ensurepip is missing on host.

### Error
```
The virtual environment was not created successfully because ensurepip is not available.
```

### Context
- Command: `python3 -m venv .venv`
- Host lacks `python3.12-venv` package

### Suggested Fix
Install `python3-venv` on host, or run tests in CI container.

### Metadata
- Reproducible: yes
- Related Files: README.md

---
