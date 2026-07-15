import random
from copy import deepcopy
import json

import pytest

from rl.league import (
    DualArenaMetrics,
    LeagueConfig,
    OpponentEntry,
    OpponentKind,
    OpponentLeague,
    SnapshotMetadata,
)


def _snapshot(name: str, step: int) -> SnapshotMetadata:
    return SnapshotMetadata(
        snapshot_id=name,
        policy_version=f"policy-{step}",
        checkpoint_path=f"checkpoints/{name}.pt",
        training_step=step,
    )


def _passing_report(league: OpponentLeague) -> dict[str, DualArenaMetrics]:
    return {
        entry.key: DualArenaMetrics(
            perfect_win_rate=0.6,
            degraded_win_rate=0.55,
            perfect_games=league.config.arena_games_per_track,
            degraded_games=league.config.arena_games_per_track,
            illegal_actions=0,
            zero_sum_failures=0,
        )
        for entry in league.entries
    }


def test_dual_arena_metrics_require_exact_completed_game_counts_and_clean_invariants():
    league = OpponentLeague(LeagueConfig(arena_games_per_track=1000), current_policy=_snapshot("latest", 10))
    candidate = _snapshot("candidate", 11)

    for replacement in (
        DualArenaMetrics(0.6, 0.6, 999, 1000, 0, 0),
        DualArenaMetrics(0.6, 0.6, 1000, 999, 0, 0),
        DualArenaMetrics(0.6, 0.6, 1000, 1000, 1, 0),
        DualArenaMetrics(0.6, 0.6, 1000, 1000, 0, 1),
    ):
        report = _passing_report(league)
        report["s3"] = replacement
        before = league.to_json()
        assert league.promote_candidate(candidate, report) is False
        assert league.to_json() == before


def test_rejected_candidate_leaves_current_and_history_unchanged():
    league = OpponentLeague(LeagueConfig(), current_policy=_snapshot("latest", 10))
    candidate = _snapshot("candidate", 11)

    for metric in (
        DualArenaMetrics(0.49, 0.8, 1000, 1000, 0, 0),
        DualArenaMetrics(0.8, 0.49, 1000, 1000, 0, 0),
    ):
        report = _passing_report(league)
        report["current"] = metric
        before = league.to_json()
        assert league.promote_candidate(candidate, report) is False
        assert league.to_json() == before


def test_promotion_requires_exact_opponent_metric_keys():
    league = OpponentLeague(LeagueConfig(), current_policy=_snapshot("latest", 10))
    candidate = _snapshot("candidate", 11)

    missing = _passing_report(league)
    del missing["random"]
    with pytest.raises(ValueError, match="missing"):
        league.promote_candidate(candidate, missing)

    extra = _passing_report(league)
    extra["unexpected"] = DualArenaMetrics(0.6, 0.6, 1000, 1000, 0, 0)
    with pytest.raises(ValueError, match="unexpected"):
        league.promote_candidate(candidate, extra)


def test_successful_promotion_preserves_snapshot_identities():
    old_current = _snapshot("latest", 10)
    candidate = _snapshot("candidate", 11)
    league = OpponentLeague(LeagueConfig(), current_policy=old_current)

    assert league.promote_candidate(candidate, _passing_report(league)) is True

    assert league.current_entry.key == "current"
    assert league.current_entry.snapshot == candidate
    assert league.historical_entries[0].key == old_current.snapshot_id
    assert league.historical_entries[0].snapshot == old_current


def test_legacy_config_without_arena_games_restores_with_production_default():
    state = OpponentLeague(LeagueConfig(), current_policy=_snapshot("latest", 10)).to_dict()
    del state["config"]["arena_games_per_track"]

    restored = OpponentLeague.from_dict(state)

    assert restored.config.arena_games_per_track == 1000


def test_history_key_must_match_snapshot_identity_on_restore():
    league = OpponentLeague(LeagueConfig(), current_policy=_snapshot("latest", 10))
    assert league.admit_snapshot(_snapshot("historical", 4), _passing_report(league))
    state = league.to_dict()
    state["history"][0]["key"] = "wrong-key"

    with pytest.raises(ValueError, match="snapshot_id"):
        OpponentLeague.from_dict(state)


def test_seeded_weighted_sampling_is_repeatable_and_uses_configured_groups():
    league = OpponentLeague(
        LeagueConfig(history_capacity=4),
        current_policy=_snapshot("latest", 10),
    )
    assert league.admit_snapshot(_snapshot("hist-a", 1), _passing_report(league))
    assert league.admit_snapshot(_snapshot("hist-b", 2), _passing_report(league))

    first = [entry.kind for entry in league.sample(20, seed=91)]
    second = [entry.kind for entry in league.sample(20, seed=91)]

    assert first == second
    # This sequence pins both group weights and the seeded weighted sampler.
    assert first == [
        OpponentKind.CURRENT, OpponentKind.CURRENT, OpponentKind.SNAPSHOT,
        OpponentKind.CURRENT, OpponentKind.GREEDY, OpponentKind.S3,
        OpponentKind.S3, OpponentKind.SNAPSHOT, OpponentKind.S3,
        OpponentKind.CURRENT, OpponentKind.SNAPSHOT, OpponentKind.SNAPSHOT,
        OpponentKind.SNAPSHOT, OpponentKind.RANDOM, OpponentKind.CURRENT,
        OpponentKind.S3, OpponentKind.S3, OpponentKind.CURRENT,
        OpponentKind.RANDOM, OpponentKind.SNAPSHOT,
    ]


def test_capacity_evicts_oldest_non_milestone_but_retains_milestones():
    league = OpponentLeague(LeagueConfig(history_capacity=2), current_policy=_snapshot("latest", 10))
    assert league.admit_snapshot(_snapshot("milestone", 1), _passing_report(league), milestone=True)
    assert league.admit_snapshot(_snapshot("old", 2), _passing_report(league))
    assert league.admit_snapshot(_snapshot("new", 3), _passing_report(league))

    assert [entry.key for entry in league.historical_entries] == ["milestone", "new"]
    assert league.historical_entries[0].milestone is True


def test_snapshot_admission_requires_dual_arena_metrics_for_every_existing_entry():
    league = OpponentLeague(LeagueConfig(), current_policy=_snapshot("latest", 10))
    report = _passing_report(league)
    report["s3"] = DualArenaMetrics(perfect_win_rate=0.9, degraded_win_rate=0.1, perfect_games=1000, degraded_games=1000, illegal_actions=0, zero_sum_failures=0)

    assert league.admit_snapshot(_snapshot("candidate", 11), report) is False
    assert league.historical_entries == ()
    with pytest.raises(ValueError, match="missing"):
        league.admit_snapshot(_snapshot("other", 12), {})


def test_json_round_trip_keeps_portable_metadata_and_state_without_live_models():
    league = OpponentLeague(LeagueConfig(history_capacity=3), current_policy=_snapshot("latest", 10))
    assert league.admit_snapshot(_snapshot("frozen", 4), _passing_report(league), milestone=True)

    restored = OpponentLeague.from_json(league.to_json())

    assert restored.to_dict() == league.to_dict()
    assert all(not isinstance(value, random.Random) for value in restored.to_dict().values())
    assert restored.historical_entries[0].snapshot is not None
    assert restored.historical_entries[0].snapshot.checkpoint_path == "checkpoints/frozen.pt"


def _valid_serialized_league_state() -> dict[str, object]:
    """Return a complete portable state so each failure isolates one scalar."""
    league = OpponentLeague(
        LeagueConfig(history_capacity=3),
        current_policy=_snapshot("latest", 10),
    )
    assert league.admit_snapshot(_snapshot("historical", 4), _passing_report(league))
    return league.to_dict()


def _replace_scalar(state: dict[str, object], path: tuple[object, ...], value: object) -> None:
    target: object = state
    for segment in path[:-1]:
        target = target[segment]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]


@pytest.mark.parametrize(
    ("path", "value", "use_json", "message"),
    [
        (("config", "history_capacity"), True, False, "positive integer"),
        (("config", "latest_weight"), "0.35", False, "sampling weights"),
        (("config", "latest_weight"), float("nan"), True, "finite"),
        (("current", "snapshot", "training_step"), True, False, "training_step"),
        (("history", 0, "milestone"), 1, False, "milestone must be a bool"),
    ],
)
def test_league_checkpoint_rejects_one_malformed_scalar_from_valid_state(path, value, use_json, message):
    state = deepcopy(_valid_serialized_league_state())
    _replace_scalar(state, path, value)
    loader = OpponentLeague.from_json if use_json else OpponentLeague.from_dict
    payload = json.dumps(state) if use_json else state

    with pytest.raises((TypeError, ValueError), match=message):
        loader(payload)


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -float("inf")])
def test_league_config_rejects_non_finite_sampling_weights(weight):
    with pytest.raises(ValueError, match="finite"):
        LeagueConfig(latest_weight=weight)


def test_league_config_rejects_history_only_initial_sampling_mass():
    with pytest.raises(ValueError, match="initial"):
        LeagueConfig(latest_weight=0.0, history_weight=1.0, s3_weight=0.0, greedy_random_weight=0.0)


def test_league_public_constructors_reject_untyped_metadata_and_entries():
    with pytest.raises(TypeError):
        OpponentLeague(current_policy="latest")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OpponentEntry("bad", "s3")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OpponentEntry("bad", OpponentKind.S3, milestone="false")  # type: ignore[arg-type]


def test_league_checkpoint_requires_all_serialized_config_and_entry_fields():
    state = OpponentLeague(LeagueConfig(), current_policy=_snapshot("latest", 1)).to_dict()
    del state["config"]["s3_weight"]
    with pytest.raises(ValueError, match="missing"):
        OpponentLeague.from_dict(state)

    state = OpponentLeague(LeagueConfig(), current_policy=_snapshot("latest", 1)).to_dict()
    del state["current"]["milestone"]
    with pytest.raises(ValueError, match="missing"):
        OpponentLeague.from_dict(state)
