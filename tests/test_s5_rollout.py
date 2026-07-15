from __future__ import annotations

from dataclasses import dataclass, replace
import random
from types import SimpleNamespace
from typing import Sequence

import pytest

from policies.rule_policy import RulePolicy
from state.adapters.from_engine import from_engine
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState


class UnsafeBeliefPayload:
    pass


@dataclass
class FirstLegalLearner:
    views: list[object]

    def __init__(self) -> None:
        self.views = []

    def __call__(self, view):
        self.views.append(view)
        return _decision_for_first_legal(view.legal_mask)


@dataclass
class SeededStochasticLearner:
    """A learner whose sampled action is determined only by the supplied decision seed."""

    actions: list[int]
    views: list[object]

    def __init__(self) -> None:
        self.actions = []
        self.views = []

    def __call__(self, view):
        self.views.append(view)
        legal_actions = [index for index, legal in enumerate(view.legal_mask) if legal]
        action = random.Random(view.decision_seed).choice(legal_actions)
        self.actions.append(action)
        return replace(_decision_for_first_legal(view.legal_mask), action=action)


def _decision_for_first_legal(legal_mask: Sequence[bool]):
    from rl.rollout import LearnerDecision

    return LearnerDecision(
        action=next(index for index, legal in enumerate(legal_mask) if legal),
        old_log_prob=-0.25,
        value=0.5,
    )


@pytest.fixture(scope="module")
def frozen_belief_provider():
    torch = pytest.importorskip("torch")
    from engine.game import Game
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from rl.rollout import FrozenBeliefProvider
    from state.encoder import encode_state
    from state.tile_belief import LearnedBelief

    state = Game(seed=3).reset()
    model = BeliefNet(BeliefNetConfig(input_size=encode_state(from_engine(state, 0)).size, hidden_size=8, residual_blocks=1))
    return FrozenBeliefProvider(LearnedBelief(model=model))


def _run(seed: int = 17, learner=None, *, belief_provider, opponents=None, learner_rng_seed: int = 101, **kwargs):
    from rl.rollout import RolloutConfig, run_rollout

    return run_rollout(
        learner=learner or FirstLegalLearner(),
        opponents=opponents or [RulePolicy(), RulePolicy(), RulePolicy()],
        config=RolloutConfig(
            seed=seed,
            learner_rng_seed=learner_rng_seed,
            learner_seat=0,
            max_steps=600,
            score_scale=48.0,
        ),
        policy_version="test-policy-v1",
        belief_provider=belief_provider,
        **kwargs,
    )


def test_rollout_records_only_learner_trajectory_steps(frozen_belief_provider) -> None:
    learner = FirstLegalLearner()

    result = _run(learner=learner, belief_provider=frozen_belief_provider)

    assert result.finished
    assert len(result.steps) == len(learner.views)
    assert result.steps
    assert all(step.policy_version == "test-policy-v1" for step in result.steps)
    assert all(step.old_log_prob == -0.25 and step.value == 0.5 for step in result.steps)


def test_rollout_assigns_reward_only_to_terminal_learner_step(frozen_belief_provider) -> None:
    result = _run(belief_provider=frozen_belief_provider)

    assert [step.reward for step in result.steps[:-1]] == [0.0] * (len(result.steps) - 1)
    assert [step.done for step in result.steps[:-1]] == [False] * (len(result.steps) - 1)
    assert result.steps[-1].done is True
    assert result.steps[-1].reward == result.scores[0] / 48.0


def test_rollout_actions_are_legal_and_terminal_scores_are_zero_sum(frozen_belief_provider) -> None:
    result = _run(belief_provider=frozen_belief_provider)

    assert result.finished
    assert result.illegal_action is False
    assert sum(result.scores) == 0
    assert all(step.legal_mask[step.action] for step in result.steps)


def test_rollout_is_reproducible_for_same_seed(frozen_belief_provider) -> None:
    first = _run(seed=29, belief_provider=frozen_belief_provider)
    second = _run(seed=29, belief_provider=frozen_belief_provider)

    assert first == second


def test_rollout_config_requires_an_explicit_independent_learner_rng_seed() -> None:
    from rl.rollout import RolloutConfig

    with pytest.raises(TypeError, match="learner_rng_seed"):
        RolloutConfig()
    with pytest.raises(TypeError, match="learner_rng_seed"):
        RolloutConfig(learner_rng_seed=None)
    with pytest.raises(TypeError, match="learner_rng_seed"):
        RolloutConfig(learner_rng_seed=True)


def test_environment_and_learner_seeds_reproduce_stochastic_trajectory(frozen_belief_provider) -> None:
    first_learner = SeededStochasticLearner()
    second_learner = SeededStochasticLearner()

    first = _run(seed=29, learner_rng_seed=101, learner=first_learner, belief_provider=frozen_belief_provider)
    second = _run(seed=29, learner_rng_seed=101, learner=second_learner, belief_provider=frozen_belief_provider)

    assert first == second
    assert first_learner.actions == second_learner.actions


def test_learner_rng_seed_changes_sampling_without_changing_initial_game_observation(frozen_belief_provider) -> None:
    first_learner = SeededStochasticLearner()
    changed_learner = SeededStochasticLearner()

    _run(seed=29, learner_rng_seed=101, learner=first_learner, belief_provider=frozen_belief_provider)
    _run(seed=29, learner_rng_seed=202, learner=changed_learner, belief_provider=frozen_belief_provider)

    first_view = first_learner.views[0]
    changed_view = changed_learner.views[0]
    assert first_view.observation == changed_view.observation
    assert first_view.feature_values == changed_view.feature_values
    assert first_view.legal_mask == changed_view.legal_mask
    assert first_view.decision_seed != changed_view.decision_seed
    assert first_learner.actions != changed_learner.actions


def test_decision_seed_stream_is_independent_of_environment_seed(frozen_belief_provider) -> None:
    first_learner = FirstLegalLearner()
    second_learner = FirstLegalLearner()

    _run(seed=29, learner_rng_seed=101, learner=first_learner, belief_provider=frozen_belief_provider)
    _run(seed=30, learner_rng_seed=101, learner=second_learner, belief_provider=frozen_belief_provider)

    first_seeds = [view.decision_seed for view in first_learner.views]
    second_seeds = [view.decision_seed for view in second_learner.views]
    shared_count = min(len(first_seeds), len(second_seeds))
    assert shared_count > 0
    assert first_seeds[:shared_count] == second_seeds[:shared_count]
    assert first_learner.views[0].observation != second_learner.views[0].observation
    assert first_learner.views[0].feature_values != second_learner.views[0].feature_values


def test_explicit_learner_rng_seed_cannot_match_the_removed_environment_seed_fallback(frozen_belief_provider) -> None:
    """A finite enumeration of environment seeds cannot recreate the supplied RNG root."""
    import rl.rollout as rollout

    learner = FirstLegalLearner()
    _run(seed=29, learner_rng_seed=101, learner=learner, belief_provider=frozen_belief_provider)

    observed = [view.decision_seed for view in learner.views]
    old_first_decision_candidates = [
        rollout._derive_seed("s5/learner-decision/v1", candidate, 0)
        for candidate in range(64)
    ]
    assert observed[0] not in old_first_decision_candidates
    assert observed[0] != rollout._derive_seed("s5/learner-decision/v1", 29, 0)
    assert observed[0] == rollout._decision_seed(101, 0)


def test_learner_boundary_redacts_hidden_information_and_exposes_only_s2_view(frozen_belief_provider) -> None:
    learner = FirstLegalLearner()

    result = _run(seed=37, learner=learner, belief_provider=frozen_belief_provider)

    assert result.finished
    assert learner.views
    for view in learner.views:
        assert set(view.__dataclass_fields__) == {"observation", "feature_values", "legal_mask", "decision_seed"}
        assert isinstance(view.observation, S2ProtocolState)
        assert isinstance(view.feature_values, tuple)
        assert isinstance(view.legal_mask, tuple)
        players = view.observation.facts.players.value
        assert players is not None
        for player in players:
            if player["relative_position"]:
                assert player["concealed_hand"].status is ObservationStatus.UNKNOWN


def test_rollout_requires_explicit_frozen_s4_belief_provider() -> None:
    from rl.rollout import RolloutConfig, run_rollout

    with pytest.raises(ValueError, match="frozen S4 belief provider"):
        run_rollout(
            learner=FirstLegalLearner(),
            opponents=[RulePolicy(), RulePolicy(), RulePolicy()],
            config=RolloutConfig(learner_rng_seed=101, max_steps=1),
            policy_version="test-policy-v1",
        )


def test_frozen_belief_provider_freezes_learned_model_and_uses_learned_source(frozen_belief_provider) -> None:
    model = frozen_belief_provider.belief.model

    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    learner = FirstLegalLearner()
    _run(learner=learner, belief_provider=frozen_belief_provider)
    assert all(view.observation.beliefs.source.value == "learned" for view in learner.views)


def test_frozen_belief_provider_rejects_unsafe_checkpoint_payload(tmp_path) -> None:
    import pickle

    torch = pytest.importorskip("torch")
    from rl.rollout import FrozenBeliefProvider

    checkpoint = tmp_path / "unsafe-belief.pt"
    torch.save(UnsafeBeliefPayload(), checkpoint)

    with pytest.raises(pickle.UnpicklingError, match="weights_only"):
        FrozenBeliefProvider.from_checkpoint(str(checkpoint))


def test_frozen_belief_provider_loads_the_versioned_s4_checkpoint(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    from engine.game import Game
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from rl.rollout import FrozenBeliefProvider
    from state.encoder import ENCODER_VERSION, encode_state

    state = Game(seed=5).reset()
    config = BeliefNetConfig(input_size=encode_state(from_engine(state, 0)).size, hidden_size=8, residual_blocks=1)
    checkpoint = tmp_path / "belief_s4.pt"
    torch.save(
        {"model_config": config.__dict__, "encoder_version": ENCODER_VERSION, "state_dict": BeliefNet(config).state_dict()},
        checkpoint,
    )

    provider = FrozenBeliefProvider.from_checkpoint(str(checkpoint))

    assert provider.belief.source == "learned"
    assert provider.belief.model.training is False
    assert all(parameter.requires_grad is False for parameter in provider.belief.model.parameters())


def test_learner_boundary_orders_degrader_then_belief_then_features_and_never_degrades_opponents(monkeypatch, frozen_belief_provider) -> None:
    trace: list[tuple[str, int]] = []
    learner = FirstLegalLearner()
    from policies.base_policy import BasePolicy
    from policies.protocol_actions import actions_from_mask

    class PerfectObservationOpponent(BasePolicy):
        def __init__(self) -> None:
            self.wall_statuses = []

        def choose_action(self, protocol_state, legal_mask):
            self.wall_statuses.append(protocol_state.facts.wall_count.status)
            return actions_from_mask(legal_mask)[0]

    opponent = PerfectObservationOpponent()

    def degrader(state: S2ProtocolState, seed: int) -> S2ProtocolState:
        trace.append(("degrade", state.perspective_player))
        return replace(state, facts=replace(state.facts, wall_count=ObservedValue.unknown()))

    original_apply = type(frozen_belief_provider).apply

    def tracked_apply(self, state: S2ProtocolState) -> S2ProtocolState:
        assert state.facts.wall_count.status is ObservationStatus.UNKNOWN
        trace.append(("belief", state.perspective_player))
        return original_apply(self, state)

    monkeypatch.setattr(type(frozen_belief_provider), "apply", tracked_apply)
    def features(state: S2ProtocolState) -> Sequence[float]:
        assert state.facts.wall_count.status is ObservationStatus.UNKNOWN
        assert state.beliefs.source.value == "learned"
        trace.append(("features", state.perspective_player))
        return (1.0,)

    _run(
        learner=learner,
        belief_provider=frozen_belief_provider,
        observation_degrader=degrader,
        feature_extractor=features,
        opponents=[opponent, opponent, opponent],
    )

    assert trace
    assert all(player == 0 for _, player in trace)
    assert [name for name, _ in trace] == [item for _ in learner.views for item in ("degrade", "belief", "features")]
    assert opponent.wall_statuses
    assert all(status is ObservationStatus.OBSERVED for status in opponent.wall_statuses)


def test_injected_degradation_deterministically_changes_learner_visible_observation(frozen_belief_provider) -> None:
    learner = FirstLegalLearner()

    def degrader(state: S2ProtocolState, seed: int) -> S2ProtocolState:
        return replace(state, facts=replace(state.facts, wall_count=ObservedValue.unknown()))

    _run(learner=learner, belief_provider=frozen_belief_provider, observation_degrader=degrader)
    assert learner.views
    assert all(view.observation.facts.wall_count.status is ObservationStatus.UNKNOWN for view in learner.views)


def test_rollout_rejects_invalid_learner_action(frozen_belief_provider) -> None:
    def invalid(view):
        return replace(_decision_for_first_legal(view.legal_mask), action=len(view.legal_mask))

    from rl.rollout import RolloutInvariantError
    with pytest.raises(RolloutInvariantError, match="illegal rollout action: learner selected an action outside the legal mask"):
        _run(learner=invalid, belief_provider=frozen_belief_provider)


def test_rollout_rejects_invalid_opponent_action(frozen_belief_provider) -> None:
    from policies.base_policy import BasePolicy
    from rl.rollout import RolloutInvariantError

    class InvalidOpponent(BasePolicy):
        def choose_action(self, protocol_state, legal_mask):
            return {"kind": "not-an-action"}

    with pytest.raises(RolloutInvariantError, match="opponent selected an illegal action"):
        _run(belief_provider=frozen_belief_provider, opponents=[InvalidOpponent(), RulePolicy(), RulePolicy()])


def test_rollout_rejects_incomplete_game_at_max_steps(frozen_belief_provider) -> None:
    from rl.rollout import RolloutConfig, RolloutInvariantError, run_rollout

    with pytest.raises(RolloutInvariantError, match="did not finish"):
        run_rollout(
            FirstLegalLearner(),
            [RulePolicy(), RulePolicy(), RulePolicy()],
            config=RolloutConfig(learner_rng_seed=101, max_steps=1),
            policy_version="test-policy-v1",
            belief_provider=frozen_belief_provider,
        )


def test_rollout_rejects_malformed_belief_provider() -> None:
    from rl.rollout import RolloutConfig, RolloutInvariantError, run_rollout

    with pytest.raises(TypeError, match="FrozenBeliefProvider"):
        run_rollout(
            FirstLegalLearner(),
            [RulePolicy(), RulePolicy(), RulePolicy()],
            config=RolloutConfig(learner_rng_seed=101, max_steps=1),
            policy_version="test-policy-v1",
            belief_provider=object(),
        )


def test_rollout_result_cannot_report_a_completed_illegal_action() -> None:
    from rl.rollout import RolloutResult

    with pytest.raises(ValueError, match="never returns an illegal action"):
        RolloutResult(
            seed=1,
            learner_seat=0,
            steps=(),
            scores=(0, 0, 0, 0),
            finished=True,
            game_steps=1,
            illegal_action=True,
        )


@pytest.mark.parametrize("bad_degrader", [lambda state, seed: object(), lambda state, seed: None])
def test_rollout_rejects_malformed_degrader(frozen_belief_provider, bad_degrader) -> None:
    from rl.rollout import RolloutInvariantError

    with pytest.raises(RolloutInvariantError, match="observation_degrader"):
        _run(belief_provider=frozen_belief_provider, observation_degrader=bad_degrader)


def test_rollout_rejects_malformed_frozen_belief_output(monkeypatch, frozen_belief_provider) -> None:
    import rl.rollout as rollout

    monkeypatch.setattr(rollout, "with_prior_beliefs", lambda state, belief: object())
    with pytest.raises(rollout.RolloutInvariantError, match="belief inference failure: frozen S4 belief provider must return S2ProtocolState"):
        _run(belief_provider=frozen_belief_provider)


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ("degrader", "observation degradation failure: degrade failed"),
        ("belief", "belief inference failure: belief failed"),
        ("features", "feature extraction failure: feature failed"),
    ],
)
def test_rollout_classifies_observation_pipeline_failures_by_stage(monkeypatch, frozen_belief_provider, component, expected) -> None:
    import rl.rollout as rollout

    kwargs = {}
    if component == "degrader":
        def bad_degrader(state, seed):
            raise RuntimeError("degrade failed")
        kwargs["observation_degrader"] = bad_degrader
    elif component == "belief":
        def bad_apply(self, state):
            raise RuntimeError("belief failed")
        monkeypatch.setattr(type(frozen_belief_provider), "apply", bad_apply)
    else:
        def bad_features(state):
            raise RuntimeError("feature failed")
        kwargs["feature_extractor"] = bad_features

    with pytest.raises(rollout.RolloutInvariantError, match=expected):
        _run(belief_provider=frozen_belief_provider, **kwargs)


def test_rollout_drives_swap_declare_discard_rob_kong_and_settlement(monkeypatch, frozen_belief_provider) -> None:
    """The driver must handle every S1 control-flow branch before terminal settlement."""
    import rl.rollout as rollout

    events: list[tuple[str, int]] = []

    class ScriptedGame:
        def __init__(self, seed):
            self.state = None

        def reset(self):
            self.state = SimpleNamespace(
                phase="swap_three", finished=False, swap_choices=[None] * 4, void_suits=[None] * 4,
                pending_discard=None, pending_rob_kong=None, pending_winners=[], current_player=0,
                won=[False] * 4, scores=[0, 0, 0, 0], pending_passers=[]
            )
            return self.state

        def step(self, player, action):
            state = self.state
            stage = "rob_kong" if state.pending_rob_kong is not None else "pending_discard" if state.pending_discard is not None else state.phase
            events.append((stage, player))
            if state.phase == "swap_three":
                state.swap_choices[player] = True
                if all(choice is not None for choice in state.swap_choices):
                    state.phase = "declare_void"
            elif state.phase == "declare_void":
                state.void_suits[player] = True
                if all(suit is not None for suit in state.void_suits):
                    state.phase = "play"
                    state.current_player = 0
            elif state.pending_rob_kong is not None:
                state.pending_rob_kong = None
                state.won[player] = True
                state.current_player = 1
            elif state.pending_discard is not None:
                state.pending_discard = None
                state.pending_passers = []
                state.current_player = 0
                state.pending_rob_kong = SimpleNamespace(winners=[0])
            elif state.current_player == 0:
                state.pending_discard = SimpleNamespace(discarder=0)
            else:
                state.finished = True
                state.phase = "finished"
            return state

    state = from_engine(__import__("engine.game", fromlist=["Game"]).Game(seed=1).reset(), 0)
    monkeypatch.setattr(rollout, "Game", ScriptedGame)
    from state.action_space import action_space_size
    mask = (True,) + (False,) * (action_space_size() - 1)
    monkeypatch.setattr(rollout, "_learner_view", lambda *args, **kwargs: rollout.LearnerView(state, (1.0,), mask, 0))
    monkeypatch.setattr(rollout, "choose_policy_action", lambda *args, **kwargs: SimpleNamespace(action=object()))
    result = rollout.run_rollout(
        FirstLegalLearner(), [RulePolicy(), RulePolicy(), RulePolicy()],
        config=rollout.RolloutConfig(learner_rng_seed=101, max_steps=20), policy_version="test-policy-v1", belief_provider=frozen_belief_provider,
    )

    assert result.finished and result.illegal_action is False
    assert [phase for phase, _ in events].count("swap_three") == 4
    assert [phase for phase, _ in events].count("declare_void") == 4
    assert ("play", 0) in events  # discard opens the normal response branch
    assert ("pending_discard", 1) in events
    assert ("rob_kong", 0) in events
    assert events[-1] == ("play", 1)  # play continues after a win until final settlement
