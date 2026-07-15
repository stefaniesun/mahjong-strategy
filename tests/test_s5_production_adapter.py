from __future__ import annotations

from dataclasses import dataclass, replace

from types import SimpleNamespace

import pytest

from policies.base_policy import BasePolicy
from policies.protocol_actions import actions_from_mask
from rl.curriculum import (
    CurriculumConfig,
    CurriculumStage,
    DegradationProfile,
    ObservationCurriculum,
)
from rl.league import LeagueConfig, OpponentLeague, SnapshotMetadata
from rl.production_adapter import (
    ProductionGameTask,
    RolloutRuntimeState,
    run_production_arena,
    run_production_game,
)
from rl.rollout import LearnerDecision


class FirstLegalPolicy(BasePolicy):
    def choose_action(self, protocol_state, legal_mask):
        return actions_from_mask(legal_mask)[0]


@dataclass
class RecordingResolver:
    resolved_keys: list[str]

    def resolve(self, entry, *, seed: int):
        self.resolved_keys.append(entry.key)
        return FirstLegalPolicy()


def _curriculum(profile: DegradationProfile) -> ObservationCurriculum:
    return ObservationCurriculum(
        CurriculumConfig((CurriculumStage(profile.name, profile, 0.0, 0.0),))
    )


def _league(snapshot_id: str, *, only_current: bool = False) -> OpponentLeague:
    config = None
    if only_current:
        config = LeagueConfig(
            latest_weight=1.0,
            history_weight=0.0,
            s3_weight=0.0,
            greedy_random_weight=0.0,
            arena_games_per_track=2,
        )
    return OpponentLeague(
        config,
        current_policy=SnapshotMetadata(
            snapshot_id,
            f"{snapshot_id}-generation",
            f"{snapshot_id}.pt",
            7,
            f"{snapshot_id}-checksum",
        ),
    )


def _first_legal(view) -> LearnerDecision:
    action = next(index for index, legal in enumerate(view.legal_mask) if legal)
    return LearnerDecision(action, -0.25, 0.5)


@pytest.fixture(scope="module")
def frozen_belief_provider():
    torch = pytest.importorskip("torch")
    from engine.game import Game
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from rl.rollout import FrozenBeliefProvider
    from state.adapters.from_engine import from_engine
    from state.encoder import encode_state
    from state.tile_belief import LearnedBelief

    state = Game(seed=3).reset()
    size = encode_state(from_engine(state, 0)).size
    model = BeliefNet(BeliefNetConfig(input_size=size, hidden_size=8, residual_blocks=0))
    return FrozenBeliefProvider(LearnedBelief(model=model))


def test_production_game_uses_runtime_league_curriculum_and_rotating_seat(
    frozen_belief_provider,
) -> None:
    restored_league = _league("restored", only_current=True)
    restored_profile = DegradationProfile("restored-noise", vision_miss_rate=0.2)
    resolver = RecordingResolver([])
    runtime = RolloutRuntimeState(
        league=restored_league,
        curriculum=_curriculum(restored_profile),
        policy_generation="generation-9",
        policy_checksum="checksum-9",
        learner_decider=_first_legal,
        opponent_resolver=resolver,
        belief_provider=frozen_belief_provider,
        max_game_steps=600,
    )
    task = ProductionGameTask(
        round_id=4,
        game_index=5,
        policy_generation="generation-9",
        policy_checksum="checksum-9",
        environment_seed=17,
        learner_seed=101,
        opponent_seed=303,
        learner_seat=1,
    )

    result = run_production_game(task, runtime)

    assert result.finished
    assert result.learner_seat == task.game_index % 4 == 1
    assert resolver.resolved_keys == ["current", "current", "current"]
    assert result.steps
    assert all(step.policy_version == "generation-9" for step in result.steps)
    assert all(step.reward == 0.0 and not step.done for step in result.steps[:-1])
    assert result.steps[-1].done
    assert sum(result.scores) == 0


def test_production_game_rejects_generation_or_checksum_mismatch(
    frozen_belief_provider,
) -> None:
    runtime = RolloutRuntimeState(
        league=_league("restored", only_current=True),
        curriculum=_curriculum(DegradationProfile.perfect()),
        policy_generation="generation-9",
        policy_checksum="checksum-9",
        learner_decider=_first_legal,
        opponent_resolver=RecordingResolver([]),
        belief_provider=frozen_belief_provider,
    )
    common = dict(
        round_id=4,
        game_index=0,
        environment_seed=17,
        learner_seed=101,
        opponent_seed=303,
        learner_seat=0,
    )

    with pytest.raises(ValueError, match="generation"):
        run_production_game(
            ProductionGameTask(**common, policy_generation="stale", policy_checksum="checksum-9"),
            runtime,
        )
    with pytest.raises(ValueError, match="checksum"):
        run_production_game(
            ProductionGameTask(**common, policy_generation="generation-9", policy_checksum="stale"),
            runtime,
        )


def test_production_arena_reports_real_track_completion_and_invariants(monkeypatch) -> None:
    import torch
    import rl.production_adapter as adapter
    from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig

    reports = iter(
        (
            SimpleNamespace(
                games=3,
                unfinished=1,
                illegal_actions=1,
                zero_sum_violations=0,
                win_rate_by_seat=[2 / 3, 0.0, 0.0, 0.0],
            ),
            SimpleNamespace(
                games=3,
                unfinished=0,
                illegal_actions=0,
                zero_sum_violations=1,
                win_rate_by_seat=[1 / 3, 0.0, 0.0, 0.0],
            ),
        )
    )
    tracks: list[object] = []

    def fake_run_arena(policies, config):
        tracks.append(policies[0].degradation)
        assert len(policies) == 4
        return next(reports)

    monkeypatch.setattr(adapter, "run_arena", fake_run_arena)
    model = PolicyValueNet(
        PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0)
    )
    runtime = RolloutRuntimeState(
        league=_league("restored"),
        curriculum=_curriculum(DegradationProfile("noise", vision_miss_rate=0.2)),
        policy_generation="generation-9",
        policy_checksum="checksum-9",
        learner_decider=_first_legal,
        opponent_resolver=RecordingResolver([]),
        belief_provider=SimpleNamespace(apply=lambda state: state),
        arena_seed=77,
        max_game_steps=50,
    )

    metric = run_production_arena(
        model,
        runtime.league.current_entry,
        runtime,
        games_per_track=3,
    )

    assert metric.perfect_win_rate == pytest.approx(2 / 3)
    assert metric.degraded_win_rate == pytest.approx(1 / 3)
    assert metric.perfect_games == 2
    assert metric.degraded_games == 3
    assert metric.illegal_actions == 1
    assert metric.zero_sum_failures == 1
    assert tracks[0] is None
    assert tracks[1] == runtime.curriculum.learner_profile
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_production_arena_rejects_same_key_entry_with_different_snapshot() -> None:
    import torch
    from rl.models.value_net import PolicyValueNet, PolicyValueNetConfig

    league = _league("restored")
    forged = replace(
        league.current_entry,
        snapshot=SnapshotMetadata(
            "forged", "forged-generation", "forged.pt", 9, "forged-checksum"
        ),
    )
    runtime = RolloutRuntimeState(
        league=league,
        curriculum=_curriculum(DegradationProfile.perfect()),
        policy_generation="generation-9",
        policy_checksum="checksum-9",
        learner_decider=_first_legal,
        opponent_resolver=RecordingResolver([]),
        belief_provider=SimpleNamespace(apply=lambda state: state),
    )
    model = PolicyValueNet(
        PolicyValueNetConfig(input_size=3, action_size=4, hidden_size=8, residual_blocks=0)
    )

    with pytest.raises(ValueError, match="current runtime league"):
        run_production_arena(model, forged, runtime, games_per_track=1)

