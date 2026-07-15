"""Spawn-safe production adapters for complete S5 rollout and arena games."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Callable, Protocol

import torch
from torch import nn

from learning.eval.arena import ArenaConfig, run_arena
from rl.curriculum import DegradationProfile, ObservationCurriculum
from rl.league import DualArenaMetrics, OpponentEntry, OpponentLeague
from rl.opponent_resolver import ModelPolicy
from rl.rollout import (
    FrozenBeliefProvider,
    LearnerDecision,
    LearnerView,
    RolloutConfig,
    RolloutInvariantError,
    RolloutResult,
    run_rollout_game,
)


class Resolver(Protocol):
    def resolve(self, entry: OpponentEntry, *, seed: int): ...


LearnerDecider = Callable[[LearnerView], LearnerDecision]


@dataclass(frozen=True, slots=True)
class ProductionGameTask:
    """One immutable complete-game task produced by the rollout scheduler."""

    round_id: int
    game_index: int
    policy_generation: str
    policy_checksum: str
    environment_seed: int
    learner_seed: int
    opponent_seed: int
    learner_seat: int

    def __post_init__(self) -> None:
        for name, value in (
            ("round_id", self.round_id),
            ("game_index", self.game_index),
            ("environment_seed", self.environment_seed),
            ("learner_seed", self.learner_seed),
            ("opponent_seed", self.opponent_seed),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in (
            ("policy_generation", self.policy_generation),
            ("policy_checksum", self.policy_checksum),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.learner_seat != self.game_index % 4:
            raise ValueError("learner_seat must rotate by game_index")


@dataclass(frozen=True, slots=True)
class RolloutRuntimeState:
    """Current restored state plus worker-local immutable inference adapters."""

    league: OpponentLeague
    curriculum: ObservationCurriculum
    policy_generation: str
    policy_checksum: str
    learner_decider: LearnerDecider | None = None
    opponent_resolver: Resolver | None = None
    belief_provider: FrozenBeliefProvider | object | None = None
    arena_seed: int = 0
    max_game_steps: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.league, OpponentLeague):
            raise TypeError("league must be OpponentLeague")
        if not isinstance(self.curriculum, ObservationCurriculum):
            raise TypeError("curriculum must be ObservationCurriculum")
        for name, value in (
            ("policy_generation", self.policy_generation),
            ("policy_checksum", self.policy_checksum),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.arena_seed, int) or isinstance(self.arena_seed, bool) or self.arena_seed < 0:
            raise ValueError("arena_seed must be a non-negative integer")
        if not isinstance(self.max_game_steps, int) or isinstance(self.max_game_steps, bool) or self.max_game_steps <= 0:
            raise ValueError("max_game_steps must be a positive integer")


def run_production_game(task: ProductionGameTask, runtime: RolloutRuntimeState) -> RolloutResult:
    """Resolve three current-League opponents and run one complete learner game."""
    if not isinstance(task, ProductionGameTask):
        raise TypeError("task must be ProductionGameTask")
    if not isinstance(runtime, RolloutRuntimeState):
        raise TypeError("runtime must be RolloutRuntimeState")
    if task.policy_generation != runtime.policy_generation:
        raise ValueError("task policy generation does not match runtime generation")
    if task.policy_checksum != runtime.policy_checksum:
        raise ValueError("task policy checksum does not match runtime checksum")
    if runtime.learner_decider is None or not callable(runtime.learner_decider):
        raise ValueError("runtime learner_decider is required")
    if runtime.opponent_resolver is None:
        raise ValueError("runtime opponent_resolver is required")
    if not isinstance(runtime.belief_provider, FrozenBeliefProvider):
        raise TypeError("runtime belief_provider must be FrozenBeliefProvider")

    entries = runtime.league.sample(3, seed=task.opponent_seed)
    opponents = tuple(
        runtime.opponent_resolver.resolve(entry, seed=task.opponent_seed + index)
        for index, entry in enumerate(entries)
    )
    result = run_rollout_game(
        runtime.learner_decider,
        opponents,
        config=RolloutConfig(
            seed=task.environment_seed,
            learner_rng_seed=task.learner_seed,
            learner_seat=task.learner_seat,
            max_steps=runtime.max_game_steps,
            observation_noise=runtime.curriculum.learner_profile.vision_miss_rate,
        ),
        policy_version=runtime.policy_generation,
        belief_provider=runtime.belief_provider,
        observation_degrader=_degrader(runtime.curriculum.learner_profile),
    )
    _validate_rollout_result(result, runtime.policy_generation)
    return result


def run_production_arena(
    candidate: nn.Module,
    entry: OpponentEntry,
    runtime: RolloutRuntimeState,
    games_per_track: int,
) -> DualArenaMetrics:
    """Evaluate one candidate/opponent pair under perfect and current degraded tracks."""
    if not isinstance(candidate, nn.Module):
        raise TypeError("candidate must be a torch module")
    if not isinstance(entry, OpponentEntry):
        raise TypeError("entry must be OpponentEntry")
    if not isinstance(runtime, RolloutRuntimeState):
        raise TypeError("runtime must be RolloutRuntimeState")
    if entry not in runtime.league.entries:
        raise ValueError("arena entry is not present in the current runtime league")

    if not isinstance(games_per_track, int) or isinstance(games_per_track, bool) or games_per_track <= 0:
        raise ValueError("games_per_track must be a positive integer")
    if runtime.opponent_resolver is None:
        raise ValueError("runtime opponent_resolver is required")
    if runtime.belief_provider is None or not callable(getattr(runtime.belief_provider, "apply", None)):
        raise ValueError("runtime belief_provider is required")

    reports = []
    for track_index, degradation in enumerate((None, runtime.curriculum.learner_profile)):
        seed = runtime.arena_seed + track_index * 100_000
        learner = ModelPolicy(
            copy.deepcopy(candidate),
            belief_provider=runtime.belief_provider,
            degradation=degradation,
            seed=seed,
        )
        opponents = tuple(
            runtime.opponent_resolver.resolve(entry, seed=seed + index + 1)
            for index in range(3)
        )
        reports.append(
            run_arena(
                (learner, *opponents),
                ArenaConfig(games=games_per_track, seed=seed, max_steps=runtime.max_game_steps),
            )
        )

    perfect, degraded = reports
    return DualArenaMetrics(
        perfect_win_rate=float(perfect.win_rate_by_seat[0]),
        degraded_win_rate=float(degraded.win_rate_by_seat[0]),
        perfect_games=perfect.games - perfect.unfinished,
        degraded_games=degraded.games - degraded.unfinished,
        illegal_actions=perfect.illegal_actions + degraded.illegal_actions,
        zero_sum_failures=perfect.zero_sum_violations + degraded.zero_sum_violations,
    )


def _degrader(profile: DegradationProfile):
    from state.observation_degradation import DegradationPipeline, MidGameSnapshot, VisionNoise

    def degrade(state, seed: int):
        operators = [VisionNoise(miss_rate=profile.vision_miss_rate, seed=seed)]
        if profile.mid_game_ratio > 0.0:
            operators.insert(0, MidGameSnapshot(int(108 * profile.mid_game_ratio)))
        return DegradationPipeline(operators).apply(state)

    return degrade


def _validate_rollout_result(result: RolloutResult, generation: str) -> None:
    if not result.finished:
        raise RolloutInvariantError("production rollout returned a non-terminal episode")
    if result.illegal_action:
        raise RolloutInvariantError("production rollout returned an illegal action")
    if not result.steps:
        raise RolloutInvariantError("production rollout returned an empty trajectory")
    if sum(result.scores) != 0:
        raise RolloutInvariantError("production rollout terminal scores are not zero-sum")
    for index, step in enumerate(result.steps):
        numeric = (*step.feature_values, step.old_log_prob, step.value, step.reward)
        if any(not math.isfinite(float(value)) for value in numeric):
            raise RolloutInvariantError("production rollout trajectory contains NaN or Inf")
        if not 0 <= step.action < len(step.legal_mask) or not step.legal_mask[step.action]:
            raise RolloutInvariantError("production rollout trajectory contains an illegal action")
        if step.policy_version != generation:
            raise RolloutInvariantError("production rollout trajectory generation mismatch")
        terminal = index == len(result.steps) - 1
        if step.done != terminal or (not terminal and step.reward != 0.0):
            raise RolloutInvariantError("production rollout reward/done boundary is invalid")
