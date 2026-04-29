# Contributing

## Pre-merge checklist

Before claiming a feature, fix, or refactor is **done**, every item below
must be true. This list exists because of a real pattern: contributors
(human and AI) shipped work that passed unit tests but broke under
realistic configurations. The fix is not "more careful coding" — it's
exercising the change end-to-end before declaring done.

### Always

- [ ] `make check` is green (lint + typecheck + full pytest, ~95 s).
- [ ] `make smoke` is green (~30 s — runs every in-tree example through
      either a full 2-iter loop or `autoresearch-rl validate`).
- [ ] No new ledger / trace files were committed by accident
      (`git status` clean for `artifacts/` and `traces/`).
- [ ] Any new `*.py` test file uses the `test_*.py` naming pattern so
      pytest auto-discovers it. Files in `tests/eval/` that aren't named
      `test_*` will silently NOT run in CI.

### When you change configuration shape (`config.py`, `config_validate.py`)

- [ ] At least one example's `config.yaml` has been re-validated:
      `make validate CONFIG=examples/<name>/config.yaml`.
- [ ] If you added a new field, the `dev` extra still picks it up:
      `uv sync --extra dev` then `make smoke`.

### When you change the diff / contract path (`controller/diff_executor.py`,
`controller/contract.py`, `sandbox/validator.py`)

- [ ] **Fixtures cover both basename-only and workdir-prefixed paths.**
      The contract bug fixed in `fef66d1` lived for weeks because every
      test fixture used `mutable_file="train.py"` while every real
      example uses `mutable_file="examples/foo/train.py"`. Whenever you
      touch path-comparison logic, write at least one test that mirrors
      a realistic config.

### When you change LLM-policy plumbing (`policy/llm_*.py`,
`policy/_prompt_fragments.py`)

- [ ] If `MOONSHOT_API_KEY` is set in your env: `make real-llm` passes.
      This is the only test that exercises the actual chat-completions
      payload format and assertion that the prompt steers behavior.
- [ ] If you modified a prompt fragment, the structural assertions in
      `tests/eval/test_prompt_eval.py` still pass (these run in regular
      CI).

### When you change parallel-engine code (`controller/parallel_engine.py`,
`controller/intra_iteration.py`, `controller/resource_pool.py`)

- [ ] `tests/test_showcase_determinism.py` passes (~65 s — runs the
      showcase twice, asserts bit-identical strict-mode behavior and
      stable best_value with cancel enabled).
- [ ] `make showcase` produces a result with `best_value != null` and
      a non-empty cancellation set.

### When you change Basilica integration (`target/basilica.py`)

- [ ] The bootstrap script still parses: `uv run pytest
      tests/test_basilica_unit.py::TestBuildBootstrapCmd -q`.
- [ ] `_propagate_control` round-trip tests still pass.
- [ ] If you have a real Basilica account: a 1-iter run completes
      against the live API. (CI cannot do this; it's on you.)

## Hard rule

**Do not call a feature "done" without a realistic-config end-to-end
run on the same day you wrote it.** Unit tests, lint, and type-check
prove the code compiles and the tests you happened to write pass.
They do not prove the feature works.

The arc that produced this checklist surfaced ~15 instances where
"done" was reversed by the next end-to-end run. Each one was preventable
by spending five minutes running the actual CLI against a real example
config before commit. The cost of that habit is small; the cost of
working around its absence is large (sometimes weeks of silent
regression, as with the contract path bug).

If you're an LLM agent contributing to this codebase: re-read this
section before declaring any work complete. The user's trust depends
on "done" meaning done.

## Local development setup

```bash
uv sync --extra dev          # pytest, ruff, mypy, basilica-sdk for tests
uv sync --extra dev --extra chart   # adds matplotlib for progress charts
make help                    # list all targets
```

## Common workflows

```bash
make test            # full suite (~95 s)
make test-fast       # skip the slow showcase determinism (~30 s)
make smoke           # end-to-end examples (~30 s)
make showcase        # run the parallel-cancel-showcase demo
make showcase-chart  # regenerate the showcase progress.png
make validate CONFIG=examples/foo/config.yaml
make real-llm        # only with MOONSHOT_API_KEY set
```

## Adding a new example

1. Create `examples/<name>/{config.yaml, prepare.py, train.py, program.md, run.sh, README.md}`.
2. Add the example dir to `tests/test_examples_smoke.py`:
   - If it runs CPU-only with a stub API key: add to `TIER1_FULL_RUN`.
   - If it needs GPU / heavy ML deps / real LLM: add to
     `TIER2_VALIDATE_ONLY`.
3. Run `make smoke` and confirm your new example passes.
4. Commit the example + the updated smoke test in the same PR.

Skipping step 2 is the most common way regressions land in this repo.
Don't.
