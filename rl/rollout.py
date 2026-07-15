"""Learner-visible S5 game rollouts with a strict S2 information boundary."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Callable, Protocol, Sequence

from engine.game import Game
from engine.settlement import assert_zero_sum
from policies.base_policy import BasePolicy
from policies.decision_boundary import choose_policy_action
from policies.protocol_actions import action_from_protocol, validate_legal_mask
from rl.reward import assign_terminal_reward
from rl.types import TrajectoryStep
from state.action_space import index_to_action, legal_mask
from state.adapters.from_engine import from_engine
from state.encoder import encode_state
from state.observation_degradation import DegradationPipeline, VisionNoise
from state.protocol import S2ProtocolState
from state.tile_belief import LearnedBelief, with_prior_beliefs


class RolloutInvariantError(RuntimeError):
    """A full rollout violated a required game or learner-boundary invariant."""


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    # This root is intentionally required.  Deriving it from ``seed`` would let
    # a learner enumerate S1 environment seeds and reconstruct hidden state.
    learner_rng_seed: int
    # ``seed`` is exclusively the S1 environment seed.  It must never cross the
    # learner information boundary, because a learner that knows it can recreate
    # hidden wall order and concealed hands.
    seed: int = 0
    learner_seat: int = 0
    max_steps: int = 1000
    score_scale: float = 48.0
    observation_noise: float = 0.03

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        if not isinstance(self.learner_rng_seed, int) or isinstance(self.learner_rng_seed, bool):
            raise TypeError("learner_rng_seed must be an integer")
        if not 0 <= self.learner_seat < 4:
            raise ValueError("learner_seat must be in [0, 3]")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not self.score_scale > 0:
            raise ValueError("score_scale must be positive")
        if not 0.0 <= self.observation_noise <= 1.0:
            raise ValueError("observation_noise must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class LearnerView:
    """The complete information boundary passed to a learner decision function."""

    observation: S2ProtocolState
    feature_values: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    # Every learner callback receives an independent, deterministic random seed.
    # A stochastic learner must derive all rollout sampling from this field rather
    # than global RNG state, so replaying a RolloutConfig reproduces its actions.
    decision_seed: int


@dataclass(frozen=True, slots=True)
class LearnerDecision:
    """An action sampled by a learner policy, with its historical PPO values."""

    action: int
    old_log_prob: float
    value: float


class LearnerDecider(Protocol):
    def __call__(self, view: LearnerView) -> LearnerDecision: ...


class FeatureExtractor(Protocol):
    def __call__(self, state: S2ProtocolState) -> Sequence[float]: ...


class ObservationDegrader(Protocol):
    def __call__(self, state: S2ProtocolState, seed: int) -> S2ProtocolState: ...


@dataclass(frozen=True, slots=True)
class FrozenBeliefProvider:
    """Read-only, checkpoint-backed S4 adapter for learner-visible S2 state.

    This intentionally accepts only :class:`LearnedBelief`: ``PriorBelief`` is a
    diagnostic baseline, not a permitted silent fallback for S5 self-play.
    Construction puts the model in eval mode and disables parameter gradients, so
    a rollout cannot accidentally update the frozen S4 asset.
    """

    belief: LearnedBelief

    def __post_init__(self) -> None:
        if not isinstance(self.belief, LearnedBelief):
            raise TypeError("FrozenBeliefProvider requires a LearnedBelief from the frozen S4 checkpoint")
        if self.belief.source != "learned":
            raise ValueError("frozen S4 belief source must be 'learned'")
        model = self.belief.model
        if model is None:
            raise ValueError("frozen S4 belief model is unavailable")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "FrozenBeliefProvider":
        """Load an S4 belief checkpoint once and expose it as a frozen adapter."""
        if not checkpoint_path:
            raise ValueError("S4 belief checkpoint path must be non-empty")
        return cls(LearnedBelief(model_path=checkpoint_path))

    def apply(self, state: S2ProtocolState) -> S2ProtocolState:
        if not isinstance(state, S2ProtocolState):
            raise TypeError("frozen S4 belief requires an S2ProtocolState")
        result = with_prior_beliefs(state, self.belief)
        if not isinstance(result, S2ProtocolState):
            raise TypeError("frozen S4 belief provider must return S2ProtocolState")
        if result.beliefs.source.value != "learned":
            raise RolloutInvariantError("frozen S4 belief did not emit learned beliefs")
        return result


@dataclass(frozen=True, slots=True)
class PolicyValueAgent:
    """Small adapter that makes an injected policy/value sampler a learner agent."""

    decide: LearnerDecider

    def __call__(self, view: LearnerView) -> LearnerDecision:
        return self.decide(view)


@dataclass(frozen=True, slots=True)
class RolloutResult:
    seed: int
    learner_seat: int
    steps: tuple[TrajectoryStep, ...]
    scores: tuple[int, int, int, int]
    finished: bool
    game_steps: int
    # Invalid actions abort with RolloutInvariantError; completed results are
    # therefore always zero-illegal-action evidence, never partial failures.
    illegal_action: bool = False

    def __post_init__(self) -> None:
        if self.illegal_action:
            raise ValueError("a completed rollout never returns an illegal action")

    @property
    def trajectory(self) -> tuple[TrajectoryStep, ...]:
        return self.steps


def run_rollout_game(
    learner: LearnerDecider,
    opponents: Sequence[BasePolicy],
    *,
    config: RolloutConfig | None = None,
    policy_version: str,
    belief_provider: FrozenBeliefProvider | None = None,
    feature_extractor: FeatureExtractor | None = None,
    observation_degrader: ObservationDegrader | None = None,
) -> RolloutResult:
    """Play one complete S1 game while recording only learner decisions.

    The learner callback never receives ``Game`` or ``GameState``.  Its only input
    is an immutable S2 snapshot after deterministic degradation and frozen belief
    inference.  Opponents continue through the normal, non-degraded policy path.
    """
    if config is None:
        raise ValueError("rollout requires an explicit RolloutConfig with learner_rng_seed")
    cfg = config
    if len(opponents) != 3:
        raise ValueError("rollout requires exactly three BasePolicy opponents")
    if any(not isinstance(opponent, BasePolicy) for opponent in opponents):
        raise TypeError("rollout opponents must implement BasePolicy")
    if not policy_version:
        raise ValueError("policy_version must be non-empty")

    if belief_provider is None:
        raise ValueError("rollout requires an explicit frozen S4 belief provider")
    if not isinstance(belief_provider, FrozenBeliefProvider):
        raise TypeError("rollout belief_provider must be a FrozenBeliefProvider")
    provider = belief_provider
    extractor = feature_extractor or _encoded_features
    degrader = observation_degrader or _default_degrader(cfg.observation_noise)
    policies = _policies_by_seat(opponents, cfg.learner_seat)
    game = Game(seed=cfg.seed)
    state = game.reset()
    raw_steps: list[TrajectoryStep] = []
    game_steps = 0
    learner_turn = 0

    def step_player(player: int) -> None:
        nonlocal learner_turn
        if game.state is None:
            raise RolloutInvariantError("game state is unavailable")
        if player == cfg.learner_seat:
            view = _learner_view(
                game.state,
                player,
                decision_seed=_decision_seed(cfg.learner_rng_seed, learner_turn),
                degradation_seed=_degradation_seed(cfg.seed, learner_turn),
                belief_provider=provider,
                feature_extractor=extractor,
                observation_degrader=degrader,
            )
            learner_turn += 1
            decision = learner(view)
            action = _action_from_learner_decision(decision, view.legal_mask)
            raw_steps.append(
                TrajectoryStep(
                    feature_values=view.feature_values,
                    legal_mask=view.legal_mask,
                    action=decision.action,
                    old_log_prob=float(decision.old_log_prob),
                    value=float(decision.value),
                    reward=0.0,
                    done=False,
                    policy_version=policy_version,
                )
            )
            game.step(player, action)
            return
        opponent = policies[player]
        if opponent is None:
            raise RolloutInvariantError("learner seat cannot be routed to an opponent policy")
        try:
            decision = choose_policy_action(game.state, player, opponent)
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise RolloutInvariantError(f"opponent selected an illegal action: {exc}") from exc
        game.step(player, decision.action)

    try:
        while not state.finished and game_steps < cfg.max_steps:
            if state.phase in {"swap_three", "declare_void"}:
                for player in range(4):
                    if state.finished or state.phase not in {"swap_three", "declare_void"}:
                        break
                    if state.phase == "swap_three" and state.swap_choices[player] is not None:
                        continue
                    if state.phase == "declare_void" and state.void_suits[player] is not None:
                        continue
                    step_player(player)
                    game_steps += 1
                    if game_steps >= cfg.max_steps:
                        break
                continue

            if state.pending_rob_kong is not None:
                step_player(state.pending_rob_kong.winners[0])
                game_steps += 1
                continue

            if state.pending_discard is not None:
                resolved, decisions = _resolve_pending_discard(game, step_player, cfg.max_steps - game_steps)
                game_steps += decisions
                if game_steps >= cfg.max_steps:
                    continue
                if not resolved and state.pending_discard is not None:
                    raise RolloutInvariantError("pending discard responses made no progress")
                continue

            if state.phase == "play":
                step_player(state.current_player)
                game_steps += 1
                continue
            break
    except (TypeError, ValueError, IndexError) as exc:
        raise RolloutInvariantError(f"illegal rollout action: {exc}") from exc

    if not state.finished:
        raise RolloutInvariantError(f"rollout did not finish within {cfg.max_steps} actions")
    if not raw_steps:
        raise RolloutInvariantError("rollout recorded no learner decisions")
    try:
        assert_zero_sum(state.scores)
    except AssertionError as exc:
        raise RolloutInvariantError("rollout terminal scores are not zero-sum") from exc

    return RolloutResult(
        seed=cfg.seed,
        learner_seat=cfg.learner_seat,
        steps=assign_terminal_reward(raw_steps, state.scores[cfg.learner_seat], cfg.score_scale),
        scores=tuple(state.scores),
        finished=True,
        game_steps=game_steps,
    )


def run_rollout(
    learner: LearnerDecider,
    opponents: Sequence[BasePolicy],
    *,
    config: RolloutConfig | None = None,
    policy_version: str,
    belief_provider: FrozenBeliefProvider | None = None,
    feature_extractor: FeatureExtractor | None = None,
    observation_degrader: ObservationDegrader | None = None,
) -> RolloutResult:
    """Alias for :func:`run_rollout_game` used by training callers."""
    return run_rollout_game(
        learner,
        opponents,
        config=config,
        policy_version=policy_version,
        belief_provider=belief_provider,
        feature_extractor=feature_extractor,
        observation_degrader=observation_degrader,
    )


def _policies_by_seat(opponents: Sequence[BasePolicy], learner_seat: int) -> tuple[BasePolicy | None, ...]:
    iterator = iter(opponents)
    return tuple(None if player == learner_seat else next(iterator) for player in range(4))


def _learner_view(
    state,
    player: int,
    *,
    decision_seed: int,
    degradation_seed: int,
    belief_provider: FrozenBeliefProvider,
    feature_extractor: FeatureExtractor,
    observation_degrader: ObservationDegrader,
) -> LearnerView:
    try:
        perfect = from_engine(state, player_id=player)
        mask = tuple(legal_mask(perfect))
    except Exception as exc:
        raise RolloutInvariantError(f"observation construction failure: {exc}") from exc
    try:
        degraded = observation_degrader(perfect, degradation_seed)
    except Exception as exc:
        raise RolloutInvariantError(f"observation degradation failure: {exc}") from exc
    if not isinstance(degraded, S2ProtocolState):
        raise RolloutInvariantError("observation degradation failure: observation_degrader must return S2ProtocolState")
    # Legal feedback is supplied by the S1 environment, rather than inferred from
    # noisy fields, so a learner can never sample a conditionally legal action.
    degraded = replace(degraded, legal_actions=perfect.legal_actions)
    try:
        with_beliefs = belief_provider.apply(degraded)
    except Exception as exc:
        raise RolloutInvariantError(f"belief inference failure: {exc}") from exc
    if not isinstance(with_beliefs, S2ProtocolState):
        raise RolloutInvariantError("belief inference failure: belief_provider must return S2ProtocolState")
    snapshot = S2ProtocolState.from_dict(with_beliefs.to_dict())
    try:
        features = tuple(float(value) for value in feature_extractor(snapshot))
    except Exception as exc:
        raise RolloutInvariantError(f"feature extraction failure: {exc}") from exc
    return LearnerView(
        observation=snapshot,
        feature_values=features,
        legal_mask=mask,
        decision_seed=decision_seed,
    )


def _resolve_pending_discard(
    game: Game,
    step_player: Callable[[int], None],
    max_decisions: int,
) -> tuple[bool, int]:
    state = game.state
    if state is None or state.pending_discard is None or max_decisions <= 0:
        return False, 0
    if state.pending_winners:
        step_player(state.pending_winners[0])
        return True, 1

    decisions = 0
    discarder = state.pending_discard.discarder
    for offset in range(1, 4):
        if decisions >= max_decisions:
            break
        player = (discarder + offset) % 4
        if state.won[player] or player in state.pending_passers:
            continue
        step_player(player)
        decisions += 1
        if state.pending_discard is None:
            return True, decisions
    return False, decisions


def _action_from_learner_decision(decision: LearnerDecision, mask: Sequence[bool]):
    if not isinstance(decision, LearnerDecision):
        raise TypeError("learner must return LearnerDecision")
    validate_legal_mask(mask)
    if not isinstance(decision.action, int) or isinstance(decision.action, bool):
        raise TypeError("learner action must be an integer action index")
    if not 0 <= decision.action < len(mask) or not mask[decision.action]:
        raise ValueError("learner selected an action outside the legal mask")
    if not math.isfinite(decision.old_log_prob) or not math.isfinite(decision.value):
        raise ValueError("learner log probability and value must be finite")
    return action_from_protocol(index_to_action(decision.action))


def _encoded_features(state: S2ProtocolState) -> tuple[float, ...]:
    return encode_state(state).values


def _default_degrader(noise: float) -> ObservationDegrader:
    def degrade(state: S2ProtocolState, seed: int) -> S2ProtocolState:
        return DegradationPipeline([VisionNoise(miss_rate=noise, seed=seed)]).apply(state)

    return degrade


def _decision_seed(learner_rng_seed: int, decision_index: int) -> int:
    """Derive a learner-only decision seed from the learner RNG root."""
    return _derive_seed("s5/learner-decision/v1", learner_rng_seed, decision_index)


def _degradation_seed(environment_seed: int, decision_index: int) -> int:
    """Derive observation-noise randomness from the environment stream only."""
    return _derive_seed("s5/observation-degradation/v1", environment_seed, decision_index)


def _derive_seed(domain: str, *values: int) -> int:
    payload = domain.encode("utf-8")
    for value in values:
        encoded = str(value).encode("ascii")
        payload += len(encoded).to_bytes(4, byteorder="big") + encoded
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")
