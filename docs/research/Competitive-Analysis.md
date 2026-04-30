# Competitive Analysis — Where autoresearch-rl Sits

Written 2026-04-30 as Phase A.1 of issue
[#31](https://github.com/epappas/autoresearch-rl/issues/31). Pinned commit
of autoresearch-rl: `a3bdb57` (`main` at this date).

The wedge claim under review:

> autoresearch-rl is the only open-source framework that lets an LLM
> autonomously search BOTH hyperparameters AND training code, on cloud
> GPUs, with cooperative trial cancellation and a frozen evaluation
> contract.

This document checks that claim against ten adjacent tools. Every yes/no
is tied to a primary-source citation: an official-doc URL, a code path,
a docstring, or a paper section. If the docs use marketing language and
the source code disagrees, the source code wins.

## How to read the matrix

The six axes:

1. **LLM-driven HP search** — does the tool ask an LLM to propose
   hyperparameter values for the next trial? An LLM choosing
   prompts/demos counts as "partial". Bayesian/TPE/PBT/evolutionary
   samplers do not count even if they're clever.
2. **LLM-driven code mutation** — does the tool ask an LLM to propose
   source-code changes (diffs, function bodies, new modules) that get
   executed in the next trial?
3. **Cooperative trial cancellation** — does the tool deliver a
   stop-now signal to a *running* trial that the trial reads
   cooperatively, vs `kill -9`?
4. **Frozen-evaluation contract** — does the tool enforce a boundary
   between mutable training code (which the LLM can change) and the
   evaluation/scoring code (which it cannot), so the LLM cannot game
   the metric by editing the scorer?
5. **Cloud-GPU target as first-class adapter** — does the tool ship a
   built-in adapter that provisions a remote GPU container per trial?
   "You can run it on a Ray cluster you stood up yourself" is a "no"
   here; "you set `gpu='A100'` and we get you one" is a "yes".
6. **License** — the OSS license (or "proprietary" / "paper").

A "partial" verdict means the feature exists in some constrained form
or in an experimental subdirectory; the entry explains the
constraint.

## The matrix

| Tool | LLM HP | LLM code | Coop cancel | Frozen-eval | Cloud-GPU | License |
|---|---|---|---|---|---|---|
| Ray Tune | no | no | partial | no | partial | Apache-2.0 |
| Optuna | no | no | **yes** | no | no | MIT |
| HF AutoTrain | no | no | no | no | partial | Apache-2.0 (archived) |
| DSPy | partial (prompts) | no | no | no | no | MIT |
| FunSearch | no | **yes** | no | **yes** | no | Apache-2.0 |
| OPRO | partial (prompts/scalars) | no | n/a | n/a | n/a | paper |
| Modal | no | no | no | no | **yes** | Apache-2.0 (client) |
| SkyPilot | no | no | no | no | **yes** | Apache-2.0 |
| verl | no | no | no | no | no | Apache-2.0 |
| Vega | no | no | no | no | no | Apache-2.0 |
| **autoresearch-rl** | **yes** | **yes** | **yes** | **yes** | **yes** | unlicensed (see #32) |

Citations follow.

---

## Ray Tune

Distributed hyperparameter-tuning library on top of Ray. Samplers are
TPE/Bayes/PBT/ASHA — non-LLM.

- **LLM HP search — no.** Search algorithms in
  `python/ray/tune/search/` are HEBO, Optuna, Ax, BOHB, HyperOpt, etc.
  No LLM proposer exists in the search module. Source:
  [ray/python/ray/tune/search/](https://github.com/ray-project/ray/tree/master/python/ray/tune/search).
- **LLM code mutation — no.** No evidence of source-mutation in Tune;
  it is a hyperparameter library by design.
- **Cooperative cancellation — partial.** Run-level graceful shutdown
  works (`SIGUSR1`/`SIGINT` gives "Ray Tune shuts down training
  gracefully and saves the final experiment state"
  — [tune-stopping.html](https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html)).
  But per-trial scheduler decisions are hard `TrialScheduler.STOP`
  actions; `async_hyperband.py`'s ASHA returns `STOP` with no cooperative
  signal back to the running trial. The trial never sees a "you should
  stop" packet.
- **Frozen-eval contract — no.** The user's `Trainable` / function
  trainable owns both training and metric reporting; nothing prevents
  trial code from editing the scorer.
- **Cloud-GPU per trial — partial.** Ray clusters can scale on cloud
  GPUs, but there is no built-in "per-trial container on a remote
  GPU provider" adapter. Trials run as Ray actors on whatever cluster
  you provisioned. Tune is portable, not cloud-aware.
- **License:** Apache-2.0
  ([LICENSE](https://github.com/ray-project/ray/blob/master/LICENSE)).

**Gap:** Ray Tune does dense HP-sampling-as-a-service across Ray
clusters but has no LLM in the loop, no source-mutation, and no
frozen-eval contract.

## Optuna

Define-by-run Bayesian/TPE hyperparameter optimizer.

- **LLM HP search — no.** Samplers are TPE / CMA-ES / NSGA-II.
- **LLM code mutation — no.**
- **Cooperative cancellation — yes.** This is the canonical primitive:
  the user's objective calls `trial.report(intermediate_value, step)`
  and `if trial.should_prune(): raise optuna.TrialPruned()`. The
  *user code* raises rather than Optuna killing it.
  [tutorial 003](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).
- **Frozen-eval contract — no.** The objective owns everything.
- **Cloud-GPU per trial — no.** Optuna is in-process / RDB-backed;
  cloud orchestration is out of scope.
- **License:** MIT
  ([LICENSE](https://github.com/optuna/optuna/blob/master/LICENSE)).

**Gap:** Optuna is the cleanest cooperative-cancellation contract in
the field, but the proposer is a sampler, not an LLM, and the trial
boundary is the user's responsibility.

## HuggingFace AutoTrain (Advanced)

No-code fine-tuning UI. The repo
[`huggingface/autotrain-advanced`](https://github.com/huggingface/autotrain-advanced)
is **archived / unmaintained** as of 2026 — README banner: *"This
project is no longer maintained."*

- **LLM HP search — no evidence found.** AutoTrain runs preset
  recipes; HP search is via configuration, not an LLM proposer.
- **LLM code mutation — no.**
- **Cooperative cancellation — no evidence found.**
- **Frozen-eval contract — no.**
- **Cloud-GPU per trial — partial.** SpaceRunner / DGX Cloud paths
  exist for *training jobs*, not per-trial provisioning of a search
  loop. Source:
  [HF AutoTrain docs](https://huggingface.co/docs/autotrain/index).
- **License:** Apache-2.0
  ([LICENSE](https://github.com/huggingface/autotrain-advanced/blob/main/LICENSE)).

**Gap:** AutoTrain solves "I want to fine-tune a model with no code".
It does not solve "I want a search loop". Also, archived.

## DSPy

Framework for compiling LM programs. Optimizers (MIPROv2, GEPA, etc.)
let an LLM propose prompt instructions and few-shot demonstrations.

- **LLM HP search — partial (prompts only).** `MIPROv2._propose_instructions`
  instantiates `GroundedProposer(... prompt_model=self.prompt_model ...)`
  to "draft many potential instructions for every prompt"
  — [`mipro_optimizer_v2.py`](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/mipro_optimizer_v2.py).
  Final selection is via Bayesian optimization over Optuna. The
  optimizer's search space is **prompt instructions and few-shot
  demos**, not learning rates / epochs / batch sizes. Marked partial
  because the LLM-as-proposer pattern matches, but the search axis
  is prompt strings, not training hyperparameters.
- **LLM code mutation — no.** GEPA "uses reflection to evolve **text
  components** of complex systems"
  ([gepa.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/gepa/gepa.py)).
  No Python source mutation.
- **Cooperative cancellation — no evidence found.**
- **Frozen-eval contract — no.** The metric function and the program
  live in the same Python process; no enforced boundary preventing the
  optimizer from reading or modifying eval internals. (In practice
  DSPy users do not modify metric code, but DSPy does not *enforce* it.)
- **Cloud-GPU per trial — no.** DSPy is not a cloud orchestrator.
- **License:** MIT
  ([LICENSE](https://github.com/stanfordnlp/dspy/blob/main/LICENSE)).

**Gap:** DSPy optimizes prompts and demos using an LLM proposer; it
does not touch hyperparameters, source code, or trial-level
infrastructure. The closest thing to autoresearch-rl in spirit, but
mostly orthogonal in scope.

## FunSearch (DeepMind)

Evolutionary code search with an LLM as the program-mutator. Used in
the Nature paper to discover new constructions for the cap set
problem.

- **LLM HP search — no.** Optimizes the **body** of a Python function,
  not hyperparameters.
- **LLM code mutation — yes.** README explicitly: this implementation
  "does not contain language models for generating new programs, the
  sandbox for executing untrusted code, nor the infrastructure for
  running FunSearch on our distributed system" — confirming the
  LLM-program-generation is the design centerpiece (the public
  release ships only the orchestration shell). Source:
  [funsearch README](https://github.com/google-deepmind/funsearch).
- **Cooperative cancellation — no evidence found.**
- **Frozen-eval contract — yes.** `evaluator.py` deepcopies a frozen
  template and splices the evolved function body in:

  ```python
  program = copy.deepcopy(template)
  evolved_function = program.get_function(function_to_evolve)
  evolved_function.body = body
  ```

  Then runs in a `Sandbox`. Source:
  [`implementation/evaluator.py`](https://github.com/google-deepmind/funsearch/blob/main/implementation/evaluator.py)
  ~lines 109–112. The template is preserved across iterations; the
  mutable region is fenced by markers.
- **Cloud-GPU per trial — no.** The omitted "infrastructure for
  running on our distributed system" is exactly this; the public repo
  evaluates on CPU.
- **License:** Apache-2.0 (code) plus CC-BY-4.0 (other materials)
  ([README](https://github.com/google-deepmind/funsearch)).

**Gap:** FunSearch is the closest design ancestor for the LLM-as-code-
proposer pattern. It does not do hyperparameter search, does not run
on cloud GPUs in the public release, and does not implement
cooperative cancellation.

## OPRO ("Large Language Models as Optimizers")

Paper ([arXiv 2309.03409](https://arxiv.org/abs/2309.03409)) plus a
reference implementation in
[`google-deepmind/opro`](https://github.com/google-deepmind/opro).

- **LLM HP search — partial (scalars and prompts).** "the LLM
  generates new solutions from the prompt that contains previously
  generated solutions with their values" — applied to linear
  regression coefficients, TSP solutions, and prompt strings. The
  pattern "LLM-as-optimizer over a search space" matches, but the
  canonical applications are prompts and tiny scalar problems, not
  training hyperparameters for ML jobs.
- **LLM code mutation — no.** Solutions are scalars or strings, not
  source diffs.
- **Cooperative cancellation — n/a.** No production loop.
- **Frozen-eval contract — n/a.** No production loop.
- **Cloud-GPU per trial — n/a.** Reference impl is a notebook-grade
  script.
- **License:** Apache-2.0 reference impl.

**Gap:** OPRO is a paper plus reference code. The pattern lives on in
DSPy and FunSearch; OPRO itself is not a framework.

## Modal

Serverless compute platform. `@app.function(gpu='A100')` gives you a
GPU container per function call; the platform handles cold-start,
images, secrets, scaling.

- **LLM HP search — no.** Not its purpose.
- **LLM code mutation — no.**
- **Cooperative cancellation — no.** Modal cancels at the function-
  call level (a `Function.cancel` call), not at a trial-search
  abstraction. There is no Modal-native concept of "trial" or "stop
  the trial that's losing".
- **Frozen-eval contract — no.**
- **Cloud-GPU per trial — yes.** This is the product:
  `@app.function(gpu='A100')` … `gpu='H100:8'` for eight H100 GPUs on
  a single container. Source:
  [modal.com/docs/guide/gpu](https://modal.com/docs/guide/gpu).
- **License:** Apache-2.0 (client SDK)
  ([LICENSE](https://github.com/modal-labs/modal-client/blob/main/LICENSE));
  backend is proprietary SaaS.

**Gap:** Modal is a per-function-container compute layer. It does not
have a search loop, an LLM in the loop, or a frozen-eval contract.
It is the natural runtime *target* for one, which is exactly the
adapter pattern autoresearch-rl uses with Basilica.

## SkyPilot

Multi-cloud compute orchestrator for AI workloads.

- **LLM HP search — no.**
- **LLM code mutation — no.**
- **Cooperative cancellation — no.** `sky cancel` is job-level
  termination, not trial-loop cooperative signaling.
- **Frozen-eval contract — no.**
- **Cloud-GPU per trial — yes.** YAML
  `resources: { accelerators: A100:8 }` — "selects an appropriate
  cloud and VM based on the specified resource constraints; provisions
  (or reuses) a cluster". Source:
  [SkyPilot quickstart](https://docs.skypilot.co/en/latest/getting-started/quickstart.html).
- **License:** Apache-2.0
  ([LICENSE](https://github.com/skypilot-org/skypilot/blob/master/LICENSE)).

**Gap:** SkyPilot solves "give me the cheapest GPU across N clouds"
once. It does not solve search, LLM proposal, or per-trial
cancellation. Like Modal, it's a runtime target, not a competitor.

## verl (Volcengine / ByteDance)

Production RL training library for LLMs (PPO/GRPO/DPO) using Ray,
FSDP/Megatron, and vLLM/SGLang.

- **LLM HP search — no.** Single-job RL framework.
- **LLM code mutation — no.**
- **Cooperative cancellation — no evidence found** at trial level.
  verl operates at the level of "run one big RL training job", not
  "run many trials and cancel doomed ones".
- **Frozen-eval contract — no.**
- **Cloud-GPU per trial — no.** Scales one job across nodes via Ray;
  not per-trial provisioning.
- **License:** Apache-2.0 ([README](https://github.com/volcengine/verl)).

**Gap:** verl is the algorithm-and-distribution layer (the thing you'd
*put inside* an autoresearch-rl trial). Composable, not competitive.

## Vega (Huawei Noah)

Classical AutoML — NAS, HPO, model compression. Last meaningful
release ~2022.

- **LLM HP search — no.** Evolutionary + Bayesian.
- **LLM code mutation — no.**
- **Cooperative cancellation — no evidence found.**
- **Frozen-eval contract — no.**
- **Cloud-GPU per trial — no evidence found** of a per-trial cloud
  adapter.
- **License:** Apache-2.0
  ([LICENSE](https://github.com/huawei-noah/vega/blob/master/LICENSE)).

**Gap:** Pre-LLM AutoML, focused on NAS and classical HPO; the design
predates the "LLM-as-proposer" pattern entirely.

## autoresearch-rl

Defending each "yes" with a code path on commit `a3bdb57`:

- **LLM HP search — yes.** `LLMParamPolicy.propose` calls an
  OpenAI-compatible chat-completions endpoint to propose params from
  the search space + experiment history.
  Source:
  [`src/autoresearch_rl/policy/llm_search.py:341`](../../src/autoresearch_rl/policy/llm_search.py#L341)
  (class `LLMParamPolicy`),
  [`policy/llm_search.py:110`](../../src/autoresearch_rl/policy/llm_search.py#L110)
  (`_call_chat_api_messages`). Validated against real Kimi K2.6
  responses captured under
  `tests/eval/fixtures/real_responses/`.
- **LLM code mutation — yes.** `LLMDiffPolicy.propose` returns a
  `DiffProposal` (unified diff) against the mutable file. Source:
  [`policy/llm_diff.py:140`](../../src/autoresearch_rl/policy/llm_diff.py#L140).
  The diff is applied via `DiffExecutor.execute`
  ([`controller/diff_executor.py:105`](../../src/autoresearch_rl/controller/diff_executor.py#L105)),
  validated, and run.
- **Cooperative cancellation — yes.** The trial-side helper
  `emit_progress(...)` writes one JSON line and reads the cancel
  control file on each call; on
  `{"action": "cancel"}` it `sys.exit(CANCEL_EXIT_CODE)` with code
  42. Source:
  [`target/progress.py:48`](../../src/autoresearch_rl/target/progress.py#L48),
  [`target/progress.py:28`](../../src/autoresearch_rl/target/progress.py#L28)
  (`CANCEL_EXIT_CODE = 42`). The controller-side decision is made by
  `IntraIterationGuard` in
  [`controller/intra_iteration.py`](../../src/autoresearch_rl/controller/intra_iteration.py)
  using the live `ProgressReader` series and the power-law forecaster.
  For Basilica, the controller propagates the local control file to
  the running container's `/control` endpoint
  (`BasilicaTarget._propagate_control`,
  [`target/basilica.py:479`](../../src/autoresearch_rl/target/basilica.py#L479)).
  Validated against CPU showcase: 10/16 trials cooperatively cancel
  in `make showcase`. Note: cooperative cancel against a *live
  Basilica container* still has only mock unit tests, not a live
  round-trip; this is the weakest "yes" in the row.
- **Frozen-eval contract — yes.** `validate_diff_against_contract`
  rejects diffs that touch frozen or program files (basename
  comparison after the
  [`fef66d1`](https://github.com/epappas/autoresearch-rl/commit/fef66d1)
  fix). Source:
  [`controller/contract.py:26`](../../src/autoresearch_rl/controller/contract.py#L26).
  Wired in [`diff_executor.py:117`](../../src/autoresearch_rl/controller/diff_executor.py#L117).
  The contract is enforced at *diff-validation time*, not by an OS
  sandbox: the LLM cannot ship a diff that touches `prepare.py`, but
  could in principle write code in `train.py` that opens
  `prepare.py` at runtime. See "Weakest claims" below.
- **Cloud-GPU per trial — yes.** `BasilicaTarget` deploys each trial
  as its own GPU container on Basilica cloud, with bootstrap HTTP
  endpoints (`/progress`, `/control`, `/model/files`,
  `/model/download/<path>`). Source:
  [`target/basilica.py:167`](../../src/autoresearch_rl/target/basilica.py#L167).
  Examples consume it via `target.type: basilica` in
  [`examples/security-judge/config.yaml:10`](../../examples/security-judge/config.yaml#L10)
  and `examples/basilica-grpo/config.yaml`. Validated end-to-end on
  real A100s in probe 6 (`eval_score=[0.41, 0.11, 0.55, 0.62]`,
  4 LoRA adapters downloaded; see `RLix-Adoption-Outcomes.md`).
  Caveat: only one cloud is wired up. Modal/SkyPilot adapters do not
  yet exist as first-class targets.
- **License:** No LICENSE file at HEAD `a3bdb57` — `gh repo view
  epappas/autoresearch-rl --json licenseInfo` returns `null`. Filed
  as [#32](https://github.com/epappas/autoresearch-rl/issues/32);
  intent is open-source (the wedge claim and README treat the project
  as such), but the file is missing and the matrix has to reflect
  that.

---

## Where autoresearch-rl actually sits

**Paragraph 1 — what nobody else combines.** Across the ten tools
above, no single competitor combines (LLM HP search) + (LLM code
mutation) + (cooperative cancellation) + (frozen-eval contract) +
(cloud-GPU per-trial adapter). FunSearch is closest on the
LLM-code-proposer axis, with a real frozen-eval contract via
template + sandbox, but it ships no hyperparameter search, no
cooperative cancel, and no cloud-GPU adapter — its public release
explicitly omits the distributed runtime. DSPy is closest on the
LLM-as-proposer axis, but it optimizes prompt instructions and
few-shot demos, not training hyperparameters or source code, and it
runs in-process. Optuna is the gold standard for cooperative
cancellation but does not use an LLM and does not provision GPUs.
Modal and SkyPilot are runtime *targets* that an HP-search loop can
sit on top of, not loops in their own right. Ray Tune is the
table-stakes parallel HP-sampling library, but every sampler is a
non-LLM optimizer. The wedge survives.

**Paragraph 2 — the boundary the wedge actually traces.** The
useful frame is to read the matrix as two axes orthogonal to
"who runs the GPU". On the **proposer** axis, autoresearch-rl shares
"LLM proposes the next thing" with DSPy and FunSearch. On the
**runtime** axis, it shares "spin up a per-trial GPU container"
with Modal and SkyPilot. No one else lives on both axes at once.
Composing those two — proposing diffs *and* hyperparameters from an
LLM, against a frozen evaluator, with a power-law-driven cancel
signal that survives a containerized round-trip — is the actual
wedge. autoresearch-rl is not "Optuna plus LLM" or "FunSearch on
GPUs"; it is a search loop whose unit of proposal is either a
hyperparameter dict or a unified diff, and whose unit of execution
is a contained, observable, cancellable trial wherever it runs.

**Paragraph 3 — the three weakest claims.** First, the frozen-eval
contract is enforced at *diff-validation time*, not by an OS-level
sandbox. We compare basenames in `validate_diff_against_contract` and
reject anything that names `prepare.py`. A diff that does not name
`prepare.py` but writes Python code in `train.py` that opens
`prepare.py` at runtime, parses out the gold answers, and returns a
trivially-perfect evaluation — that is *not* caught by the contract,
and the diff would also pass the AST guardrail because no forbidden
imports or calls are involved. FunSearch's actual sandboxed
execution is the stronger contract, even though they did not ship
the sandbox in the public release. Second, cooperative cancellation
against a live Basilica container has only mocked round-trip tests;
the unit-test green light is necessary, not sufficient. The
`POST /control` to a running deployment has not been exercised
end-to-end yet against an example whose `train.py` actually calls
`emit_progress` (security-judge predates it; we would need to patch
or write a new example). Third, "cloud GPU per-trial adapter" rests
entirely on the Basilica integration. There is no Modal, SkyPilot,
or Vast.ai adapter today; the abstraction is plausible but
unproven on a second cloud. If a future real run shows the
`TargetAdapter` protocol breaks down on Modal's per-function model
or SkyPilot's per-cluster model, the matrix entry collapses to
"partial" until a second adapter ships.

The wedge survives this round of scrutiny — but the survival is on
narrower ground than the marketing version of the claim. If Phase B
ships a blog post, it should lead with the composition (search +
contract + per-trial container), not with any single axis, and it
should disclose the three weak points above.
