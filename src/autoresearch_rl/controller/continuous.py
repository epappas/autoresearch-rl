from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from autoresearch_rl.config import (
    ComparabilityConfig,
    ControllerConfig,
    ObjectiveConfig,
    PolicyConfig,
    TelemetryConfig,
)
from autoresearch_rl.controller.contract import ContractConfig
from autoresearch_rl.controller.engine import IterationCallback, run_experiment
from autoresearch_rl.controller.executor import MetricEvaluator, TargetExecutor
from autoresearch_rl.controller.types import LoopResult
from autoresearch_rl.policy.interface import DiffProposal, ParamProposal, Proposal
from autoresearch_rl.policy.learned_search import LearnedParamPolicy, LearnedSearchConfig
from autoresearch_rl.policy.llm_search import LLMParamPolicy
from autoresearch_rl.policy.search import GridPolicy, RandomPolicy, StaticPolicy
from autoresearch_rl.target.interface import TargetAdapter

logger = logging.getLogger(__name__)


def _build_llm_param_policy(
    policy_cfg: PolicyConfig, objective: ObjectiveConfig | None,
) -> LLMParamPolicy:
    assert policy_cfg.llm_api_url is not None
    assert policy_cfg.llm_model is not None
    return LLMParamPolicy(
        policy_cfg.params,
        api_url=policy_cfg.llm_api_url,
        model=policy_cfg.llm_model,
        api_key_env=policy_cfg.llm_api_key_env,
        timeout_s=policy_cfg.llm_timeout_s,
        metric=objective.metric if objective else "val_bpb",
        direction=objective.direction if objective else "min",
        seed=policy_cfg.seed,
    )


def _build_llm_diff_policy(
    policy_cfg: PolicyConfig, objective: ObjectiveConfig | None,
):
    from autoresearch_rl.policy.llm_diff import LLMDiffPolicy

    assert policy_cfg.mutable_file is not None
    assert policy_cfg.llm_api_url is not None
    assert policy_cfg.llm_model is not None
    return LLMDiffPolicy(
        mutable_file=policy_cfg.mutable_file,
        api_url=policy_cfg.llm_api_url,
        model=policy_cfg.llm_model,
        api_key_env=policy_cfg.llm_api_key_env,
        timeout_s=policy_cfg.llm_timeout_s,
        metric=objective.metric if objective else "val_bpb",
        direction=objective.direction if objective else "min",
        seed=policy_cfg.seed,
    )


def _policy_from_config(policy_cfg, objective: ObjectiveConfig | None = None):
    if policy_cfg.type == "grid":
        return GridPolicy(policy_cfg.params)
    if policy_cfg.type == "random":
        return RandomPolicy(policy_cfg.params, seed=policy_cfg.seed)
    if policy_cfg.type == "learned":
        return LearnedParamPolicy(policy_cfg.params, LearnedSearchConfig())
    if policy_cfg.type == "llm":
        return _build_llm_param_policy(policy_cfg, objective)
    if policy_cfg.type == "llm_diff":
        return _build_llm_diff_policy(policy_cfg, objective)
    if policy_cfg.type == "hybrid":
        from autoresearch_rl.policy.hybrid import HybridPolicy

        param_policy = _build_llm_param_policy(policy_cfg, objective)
        diff_policy = _build_llm_diff_policy(policy_cfg, objective)
        return HybridPolicy(
            param_policy,
            diff_policy,
            param_explore_iters=policy_cfg.hybrid_param_explore_iters,
            stall_threshold=policy_cfg.hybrid_stall_threshold,
            diff_failure_limit=policy_cfg.hybrid_diff_failure_limit,
        )
    return StaticPolicy()


def _param_state_builder(history: list[dict], program: str) -> dict:
    return {"history": history, "program": program}


def _param_extractor(proposal: Proposal) -> dict:
    assert isinstance(proposal, ParamProposal)
    return proposal.params


def _make_diff_state_builder(
    mutable_file: str,
) -> Callable[[list[dict], str], dict]:
    from autoresearch_rl.policy.llm_context import (
        extract_recent_errors,
        extract_recent_logs,
    )

    def builder(history: list[dict], program: str) -> dict:
        source = Path(mutable_file).read_text(encoding="utf-8")
        return {
            "history": history,
            "program": program,
            "source": source,
            "mutable_file": mutable_file,
            "recent_errors": extract_recent_errors(history),
            "recent_logs": extract_recent_logs(history),
        }

    return builder


def _diff_extractor(proposal: Proposal) -> dict:
    assert isinstance(proposal, DiffProposal)
    return {"diff": proposal.diff[:200]}


def _hybrid_extractor(proposal: Proposal) -> dict:
    if isinstance(proposal, DiffProposal):
        return {"diff": proposal.diff[:200], "_type": "diff"}
    assert isinstance(proposal, ParamProposal)
    return {**proposal.params, "_type": "param"}


def _build_contract(policy_cfg: PolicyConfig) -> ContractConfig | None:
    if not policy_cfg.frozen_file or not policy_cfg.program_file:
        return None
    assert policy_cfg.mutable_file is not None
    return ContractConfig(
        frozen_file=policy_cfg.frozen_file,
        mutable_file=policy_cfg.mutable_file,
        program_file=policy_cfg.program_file,
        strict=policy_cfg.contract_strict,
    )


def _make_on_keep_callback(mutable_file: str) -> IterationCallback:
    from autoresearch_rl.controller.diff_executor import _persist_diff

    def on_keep(
        iter_idx: int,
        proposal: Proposal,
        outcome: object,
        value: float | None,
        decision: str,
    ) -> None:
        if decision == "keep" and isinstance(proposal, DiffProposal):
            _persist_diff(mutable_file, proposal.diff)

    return on_keep


def _run_diff_mode(
    *,
    target: TargetAdapter,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    telemetry: TelemetryConfig,
    policy_cfg: PolicyConfig,
    comparability_cfg: ComparabilityConfig,
    program: str,
    manifest_config: dict,
) -> LoopResult:
    from autoresearch_rl.controller.diff_executor import DiffExecutor

    assert policy_cfg.mutable_file is not None
    mutable_file = policy_cfg.mutable_file
    policy = _build_llm_diff_policy(policy_cfg, objective)
    contract = _build_contract(policy_cfg)
    executor = DiffExecutor(target, mutable_file, contract)
    state_builder = _make_diff_state_builder(mutable_file)
    on_keep = _make_on_keep_callback(mutable_file)

    return run_experiment(
        executor=executor,
        evaluator=MetricEvaluator(),
        policy=policy,
        objective=objective,
        controller=controller,
        telemetry=telemetry,
        comparability_cfg=comparability_cfg,
        proposal_state_builder=state_builder,
        proposal_params_extractor=_diff_extractor,
        program=program,
        description_label="continuous-diff",
        on_iteration=on_keep,
        manifest_config=manifest_config,
    )


def _run_hybrid_mode(
    *,
    target: TargetAdapter,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    telemetry: TelemetryConfig,
    policy_cfg: PolicyConfig,
    comparability_cfg: ComparabilityConfig,
    program: str,
    manifest_config: dict,
) -> LoopResult:
    from autoresearch_rl.controller.diff_executor import DiffExecutor, HybridExecutor

    assert policy_cfg.mutable_file is not None
    mutable_file = policy_cfg.mutable_file
    policy = _policy_from_config(policy_cfg, objective)
    contract = _build_contract(policy_cfg)
    target_exec = TargetExecutor(target)
    diff_exec = DiffExecutor(target, mutable_file, contract)
    executor = HybridExecutor(target_exec, diff_exec)
    state_builder = _make_diff_state_builder(mutable_file)
    on_keep = _make_on_keep_callback(mutable_file)

    return run_experiment(
        executor=executor,
        evaluator=MetricEvaluator(),
        policy=policy,
        objective=objective,
        controller=controller,
        telemetry=telemetry,
        comparability_cfg=comparability_cfg,
        proposal_state_builder=state_builder,
        proposal_params_extractor=_hybrid_extractor,
        program=program,
        description_label="continuous-hybrid",
        on_iteration=on_keep,
        manifest_config=manifest_config,
    )


def run_continuous(
    *,
    target: TargetAdapter,
    objective: ObjectiveConfig,
    controller: ControllerConfig,
    telemetry: TelemetryConfig,
    policy_cfg: PolicyConfig,
    comparability_cfg: ComparabilityConfig,
    program: str = "",
) -> LoopResult:
    manifest_config: dict = {
        "objective": objective.model_dump(),
        "controller": controller.model_dump(),
        "telemetry": telemetry.model_dump(),
        "policy": policy_cfg.model_dump(),
        "comparability": comparability_cfg.model_dump(),
    }
    if program:
        manifest_config["program"] = program

    if policy_cfg.type == "llm_diff":
        return _run_diff_mode(
            target=target,
            objective=objective,
            controller=controller,
            telemetry=telemetry,
            policy_cfg=policy_cfg,
            comparability_cfg=comparability_cfg,
            program=program,
            manifest_config=manifest_config,
        )

    if policy_cfg.type == "hybrid":
        return _run_hybrid_mode(
            target=target,
            objective=objective,
            controller=controller,
            telemetry=telemetry,
            policy_cfg=policy_cfg,
            comparability_cfg=comparability_cfg,
            program=program,
            manifest_config=manifest_config,
        )

    policy = _policy_from_config(policy_cfg, objective)
    return run_experiment(
        executor=TargetExecutor(target),
        evaluator=MetricEvaluator(),
        policy=policy,
        objective=objective,
        controller=controller,
        telemetry=telemetry,
        comparability_cfg=comparability_cfg,
        proposal_state_builder=_param_state_builder,
        proposal_params_extractor=_param_extractor,
        program=program,
        description_label="continuous",
        manifest_config=manifest_config,
    )
