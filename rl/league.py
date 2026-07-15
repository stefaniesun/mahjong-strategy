"""Portable S5 opponent-league state and anti-forgetting admission gate.

The league deliberately stores checkpoint *metadata*, not policy objects.  Live
models are reconstructed by the training entry point after a checkpoint resume;
this makes the pool portable and deterministic to serialize.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence


class OpponentKind(str, Enum):
    """The policy family represented by a league entry."""

    CURRENT = "current"
    SNAPSHOT = "snapshot"
    S3 = "s3"
    GREEDY = "greedy"
    RANDOM = "random"


@dataclass(frozen=True, slots=True)
class SnapshotMetadata:
    """Portable reference to one policy checkpoint, never a live model."""

    snapshot_id: str
    policy_version: str
    checkpoint_path: str
    training_step: int
    checksum: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("snapshot_id", self.snapshot_id),
            ("policy_version", self.policy_version),
            ("checkpoint_path", self.checkpoint_path),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.checksum is not None and not isinstance(self.checksum, str):
            raise TypeError("checksum must be a string or None")
        if not isinstance(self.training_step, int) or isinstance(self.training_step, bool) or self.training_step < 0:
            raise ValueError("training_step must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class DualArenaMetrics:
    """Complete candidate results under both required S5 evaluation tracks."""

    perfect_win_rate: float
    degraded_win_rate: float
    perfect_games: int
    degraded_games: int
    illegal_actions: int
    zero_sum_failures: int

    def __post_init__(self) -> None:
        for name, value in (
            ("perfect_win_rate", self.perfect_win_rate),
            ("degraded_win_rate", self.degraded_win_rate),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in (
            ("perfect_games", self.perfect_games),
            ("degraded_games", self.degraded_games),
            ("illegal_actions", self.illegal_actions),
            ("zero_sum_failures", self.zero_sum_failures),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class LeagueConfig:
    """Capacity, sampling distribution, and dual-track admission thresholds."""

    history_capacity: int = 16
    latest_weight: float = 0.35
    history_weight: float = 0.35
    s3_weight: float = 0.20
    greedy_random_weight: float = 0.10
    min_perfect_win_rate: float = 0.50
    min_degraded_win_rate: float = 0.50
    arena_games_per_track: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.history_capacity, int) or isinstance(self.history_capacity, bool) or self.history_capacity < 1:
            raise ValueError("history_capacity must be a positive integer")
        if (
            not isinstance(self.arena_games_per_track, int)
            or isinstance(self.arena_games_per_track, bool)
            or self.arena_games_per_track < 1
        ):
            raise ValueError("arena_games_per_track must be a positive integer")
        weights = (self.latest_weight, self.history_weight, self.s3_weight, self.greedy_random_weight)
        if any(
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight < 0.0
            for weight in weights
        ):
            raise ValueError("sampling weights must be finite non-negative numbers")
        if sum(weights) <= 0.0:
            raise ValueError("at least one sampling weight must be positive")
        if self.latest_weight + self.s3_weight + self.greedy_random_weight <= 0.0:
            raise ValueError("initial league sampling mass cannot be history-only")
        for name, value in (
            ("min_perfect_win_rate", self.min_perfect_win_rate),
            ("min_degraded_win_rate", self.min_degraded_win_rate),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class OpponentEntry:
    """One portable pool entry.  Only current/history entries carry snapshots."""

    key: str
    kind: OpponentKind
    snapshot: SnapshotMetadata | None = None
    milestone: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.key, str):
            raise TypeError("league entry key must be a string")
        if not self.key:
            raise ValueError("league entry key must be non-empty")
        if not isinstance(self.kind, OpponentKind):
            raise TypeError("league entry kind must be OpponentKind")
        if self.snapshot is not None and not isinstance(self.snapshot, SnapshotMetadata):
            raise TypeError("league entry snapshot must be SnapshotMetadata or None")
        if not isinstance(self.milestone, bool):
            raise TypeError("league entry milestone must be a bool")
        if self.kind in (OpponentKind.CURRENT, OpponentKind.SNAPSHOT) and self.snapshot is None:
            raise ValueError(f"{self.kind.value} entries require snapshot metadata")
        if self.kind not in (OpponentKind.CURRENT, OpponentKind.SNAPSHOT) and self.snapshot is not None:
            raise ValueError("fixed baseline entries cannot carry snapshot metadata")
        if self.kind is not OpponentKind.SNAPSHOT and self.milestone:
            raise ValueError("only historical snapshots can be milestones")


class OpponentLeague:
    """Current policy, historical snapshots, and fixed S3/greedy/random baselines."""

    _FIXED_ENTRIES = (
        OpponentEntry("s3", OpponentKind.S3),
        OpponentEntry("greedy", OpponentKind.GREEDY),
        OpponentEntry("random", OpponentKind.RANDOM),
    )

    def __init__(self, config: LeagueConfig | None = None, *, current_policy: SnapshotMetadata) -> None:
        if config is not None and not isinstance(config, LeagueConfig):
            raise TypeError("config must be LeagueConfig or None")
        if not isinstance(current_policy, SnapshotMetadata):
            raise TypeError("current_policy must be SnapshotMetadata")
        self.config = config or LeagueConfig()
        self._current = OpponentEntry("current", OpponentKind.CURRENT, current_policy)
        self._history: list[OpponentEntry] = []

    @property
    def current_entry(self) -> OpponentEntry:
        return self._current

    @property
    def historical_entries(self) -> tuple[OpponentEntry, ...]:
        return tuple(self._history)

    @property
    def fixed_entries(self) -> tuple[OpponentEntry, ...]:
        return self._FIXED_ENTRIES

    @property
    def entries(self) -> tuple[OpponentEntry, ...]:
        return (self._current, *self._history, *self._FIXED_ENTRIES)

    def set_current_policy(self, metadata: SnapshotMetadata) -> None:
        """Replace only the live/latest policy reference, never history."""
        if not isinstance(metadata, SnapshotMetadata):
            raise TypeError("metadata must be SnapshotMetadata")
        self._current = OpponentEntry("current", OpponentKind.CURRENT, metadata)

    def sample(self, count: int, *, seed: int) -> tuple[OpponentEntry, ...]:
        """Draw pool entries deterministically with the configured group weights."""
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        entries, weights = self._weighted_entries()
        return tuple(random.Random(seed).choices(entries, weights=weights, k=count))

    def admit_snapshot(
        self,
        candidate: SnapshotMetadata,
        metrics_by_opponent: Mapping[str, DualArenaMetrics],
        *,
        milestone: bool = False,
    ) -> bool:
        """Admit a historical snapshot without changing the current entry."""
        if not isinstance(candidate, SnapshotMetadata):
            raise TypeError("candidate must be SnapshotMetadata")
        if not isinstance(milestone, bool):
            raise TypeError("milestone must be a bool")
        if any(entry.key == candidate.snapshot_id for entry in self.entries):
            raise ValueError(f"duplicate league snapshot key: {candidate.snapshot_id}")
        self._validate_metrics(metrics_by_opponent)
        if not self._passes_gate(metrics_by_opponent):
            return False
        next_history = self._history_with_candidate(candidate, milestone)
        if next_history is None:
            return False
        self._history = next_history
        return True

    def promote_candidate(
        self,
        candidate: SnapshotMetadata,
        metrics_by_opponent: Mapping[str, DualArenaMetrics],
        *,
        milestone: bool = False,
    ) -> bool:
        """Atomically archive the old current policy and install a passing candidate."""
        if not isinstance(candidate, SnapshotMetadata):
            raise TypeError("candidate must be SnapshotMetadata")
        if not isinstance(milestone, bool):
            raise TypeError("milestone must be a bool")
        if any(entry.key == candidate.snapshot_id for entry in self.entries):
            raise ValueError(f"duplicate league snapshot key: {candidate.snapshot_id}")
        self._validate_metrics(metrics_by_opponent)
        if not self._passes_gate(metrics_by_opponent):
            return False

        archived = self._current.snapshot
        assert archived is not None
        next_history = self._history_with_candidate(archived, milestone)
        if next_history is None:
            return False
        next_current = OpponentEntry("current", OpponentKind.CURRENT, candidate)
        self._history, self._current = next_history, next_current
        return True

    def _history_with_candidate(
        self, candidate: SnapshotMetadata, milestone: bool
    ) -> list[OpponentEntry] | None:
        next_history = list(self._history)
        if len(next_history) >= self.config.history_capacity:
            eviction_index = next((i for i, entry in enumerate(next_history) if not entry.milestone), None)
            if eviction_index is None:
                return None
            del next_history[eviction_index]
        next_history.append(OpponentEntry(candidate.snapshot_id, OpponentKind.SNAPSHOT, candidate, milestone))
        return next_history

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "current": _entry_to_dict(self._current),
            "history": [_entry_to_dict(entry) for entry in self._history],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "OpponentLeague":
        if not isinstance(data, Mapping):
            raise TypeError("league state must be a mapping")
        state = _exact_mapping(data, "league state", {"config", "current", "history"})
        raw_config = _mapping(state["config"], "config")
        if "arena_games_per_track" not in raw_config:
            raw_config["arena_games_per_track"] = 1000
        config_data = _exact_mapping(raw_config, "config", _LEAGUE_CONFIG_FIELDS)
        current = _entry_from_dict(_exact_mapping(state["current"], "current", _ENTRY_FIELDS))
        if current.key != "current" or current.kind is not OpponentKind.CURRENT or current.snapshot is None:
            raise ValueError("league current entry must use key 'current' and carry a current snapshot")
        league = cls(LeagueConfig(**config_data), current_policy=current.snapshot)
        raw_history = state["history"]
        if not isinstance(raw_history, list):
            raise TypeError("history must be a list")
        for raw_entry in raw_history:
            entry = _entry_from_dict(_exact_mapping(raw_entry, "history entry", _ENTRY_FIELDS))
            if entry.kind is not OpponentKind.SNAPSHOT or entry.snapshot is None:
                raise ValueError("history may contain only snapshot entries")
            if entry.key != entry.snapshot.snapshot_id:
                raise ValueError("history entry key must match snapshot_id")
            if len(league._history) >= league.config.history_capacity:
                raise ValueError("serialized history exceeds configured capacity")
            if any(existing.key == entry.key for existing in league.entries):
                raise ValueError(f"duplicate league entry key: {entry.key}")
            league._history.append(entry)
        return league

    @classmethod
    def from_json(cls, payload: str) -> "OpponentLeague":
        if not isinstance(payload, str):
            raise TypeError("league JSON must be a string")
        return cls.from_dict(json.loads(payload))

    def _weighted_entries(self) -> tuple[tuple[OpponentEntry, ...], tuple[float, ...]]:
        entries: list[OpponentEntry] = [self._current]
        weights: list[float] = [float(self.config.latest_weight)]
        if self._history:
            each_history_weight = float(self.config.history_weight) / len(self._history)
            entries.extend(self._history)
            weights.extend([each_history_weight] * len(self._history))
        # If no historical policy exists its configured share is deliberately
        # unavailable; random.choices normalizes the remaining live mass.
        entries.extend(self._FIXED_ENTRIES)
        weights.extend(
            [
                float(self.config.s3_weight),
                float(self.config.greedy_random_weight) / 2.0,
                float(self.config.greedy_random_weight) / 2.0,
            ]
        )
        return tuple(entries), tuple(weights)

    def _validate_metrics(self, metrics_by_opponent: Mapping[str, DualArenaMetrics]) -> None:
        if not isinstance(metrics_by_opponent, Mapping):
            raise TypeError("metrics_by_opponent must be a mapping")
        expected = {entry.key for entry in self.entries}
        if any(not isinstance(key, str) or not key for key in metrics_by_opponent):
            raise TypeError("metrics_by_opponent keys must be non-empty strings")
        actual = set(metrics_by_opponent)
        missing = expected.difference(actual)
        if missing:
            raise ValueError(f"missing dual-arena metrics for: {', '.join(sorted(missing))}")
        unexpected = actual.difference(expected)
        if unexpected:
            raise ValueError(f"unexpected dual-arena metrics for: {', '.join(sorted(unexpected))}")
        for key in expected:
            if not isinstance(metrics_by_opponent[key], DualArenaMetrics):
                raise TypeError(f"metrics for {key} must be DualArenaMetrics")

    def _passes_gate(self, metrics_by_opponent: Mapping[str, DualArenaMetrics]) -> bool:
        required_games = self.config.arena_games_per_track
        return all(
            metric.perfect_games == required_games
            and metric.degraded_games == required_games
            and metric.illegal_actions == 0
            and metric.zero_sum_failures == 0
            and metric.perfect_win_rate >= self.config.min_perfect_win_rate
            and metric.degraded_win_rate >= self.config.min_degraded_win_rate
            for metric in metrics_by_opponent.values()
        )


def _entry_to_dict(entry: OpponentEntry) -> dict[str, object]:
    return {
        "key": entry.key,
        "kind": entry.kind.value,
        "snapshot": asdict(entry.snapshot) if entry.snapshot is not None else None,
        "milestone": entry.milestone,
    }


def _entry_from_dict(data: Mapping[str, object]) -> OpponentEntry:
    key = _required(data, "key")
    kind = _required(data, "kind")
    snapshot_data = _required(data, "snapshot")
    milestone = _required(data, "milestone")
    if not isinstance(key, str):
        raise TypeError("key must be a string")
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    if not isinstance(milestone, bool):
        raise TypeError("milestone must be a bool")
    snapshot = (
        SnapshotMetadata(**_exact_mapping(snapshot_data, "snapshot", _SNAPSHOT_FIELDS))
        if snapshot_data is not None
        else None
    )
    return OpponentEntry(
        key=key,
        kind=OpponentKind(kind),
        snapshot=snapshot,
        milestone=milestone,
    )


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


_LEAGUE_CONFIG_FIELDS = frozenset(
    {
        "history_capacity",
        "latest_weight",
        "history_weight",
        "s3_weight",
        "greedy_random_weight",
        "min_perfect_win_rate",
        "min_degraded_win_rate",
        "arena_games_per_track",
    }
)
_SNAPSHOT_FIELDS = frozenset({"snapshot_id", "policy_version", "checkpoint_path", "training_step", "checksum"})
_ENTRY_FIELDS = frozenset({"key", "kind", "snapshot", "milestone"})


def _exact_mapping(value: object, name: str, expected_fields: frozenset[str]) -> dict[str, object]:
    data = _mapping(value, name)
    missing = expected_fields.difference(data)
    if missing:
        raise ValueError(f"{name} missing required field: {sorted(missing)[0]}")
    unexpected = set(data).difference(expected_fields)
    if unexpected:
        raise ValueError(f"{name} has unexpected field: {sorted(unexpected)[0]}")
    return data


def _required(data: Mapping[str, object], field: str) -> object:
    if field not in data:
        raise ValueError(f"missing required field: {field}")
    return data[field]
