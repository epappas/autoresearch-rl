# Default Program Policy

Objective: improve `val_bpb` under fixed wall-clock budget while minimizing unnecessary complexity.

Rules:
1. Edit only the configured mutable file.
2. Never modify frozen environment/evaluation files.
3. Run bounded experiments only.
4. Keep changes iff objective improves (or ties with simplification).
5. Log every trial in the canonical results ledger.
