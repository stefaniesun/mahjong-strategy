from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from console.storage import sha256_file
from learning.eval.arena import ArenaConfig, ArenaReport, run_arena
from policies.rule_policy import RulePolicy
from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig
from rl.opponent_resolver import ModelPolicy
from rl.rollout import FrozenBeliefProvider


def load_candidate(
    checkpoint: Path, *, s4_policy_path: Path | None = None, s4_belief_path: Path | None = None,
) -> tuple[PolicyValueNet, FrozenBeliefProvider, dict[str, object]]:

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state_dict"), dict):
        raise ValueError("checkpoint is not a complete S5 checkpoint")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint config is missing")
    policy_path = s4_policy_path or Path(str(config.get("frozen_s4_policy_path", "")))
    belief_path = s4_belief_path or Path(str(config.get("frozen_s4_belief_path", "")))
    if not policy_path.is_file() or not belief_path.is_file():
        raise FileNotFoundError("configured S4 evaluation assets are unavailable")
    frozen = torch.load(policy_path, map_location="cpu", weights_only=True)
    model_config = frozen.get("model_config") if isinstance(frozen, dict) else None
    if not isinstance(model_config, dict):
        raise ValueError("frozen S4 policy model_config is missing")
    model = PolicyValueNet(PolicyValueNetConfig(**model_config))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, FrozenBeliefProvider.from_checkpoint(str(belief_path)), payload


def summarize_report(report: ArenaReport, checkpoint: Path, payload: dict[str, object], *, games: int, seed: int, kind: str) -> dict[str, object]:
    candidate_scores = [result.scores[0] for result in report.results]
    mean = sum(candidate_scores) / games if games else 0.0
    ci95 = report.score_confidence95[0] if games else 0.0
    metrics = payload.get("metrics", {})
    return {
        "checkpoint": checkpoint.name,
        "sha256": sha256_file(checkpoint),
        "kind": kind,
        "games": games,
        "seed": seed,
        "global_step": payload.get("global_step", 0),
        "total_episodes": metrics.get("total_episodes", 0) if isinstance(metrics, dict) else 0,
        "average_score_difference": mean,
        "score_difference_ci95": ci95,
        "candidate_average_score": report.average_scores[0],
        "win_rate_vs_s3": report.win_rate_by_seat[0],
        "finish_rate": 1 - report.unfinished / games if games else 0.0,
        "illegal_actions": report.illegal_actions,
        "zero_sum_failures": report.zero_sum_violations,
    }


def evaluate_checkpoint(
    checkpoint: Path, *, games: int, seed: int, kind: str = "quick", max_steps: int = 1000,
    s4_policy_path: Path | None = None, s4_belief_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    arena_runner: Callable = run_arena,
) -> dict[str, object]:

    if games <= 0:
        raise ValueError("games must be positive")
    model, belief, payload = load_candidate(
        checkpoint, s4_policy_path=s4_policy_path, s4_belief_path=s4_belief_path,
    )
    report = arena_runner(
        (ModelPolicy(model, belief_provider=belief, seed=seed), RulePolicy(), RulePolicy(), RulePolicy()),
        ArenaConfig(games, seed, max_steps, progress_callback),
    )
    return summarize_report(report, checkpoint, payload, games=games, seed=seed, kind=kind)
