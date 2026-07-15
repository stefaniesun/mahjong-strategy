from __future__ import annotations

import copy
import random
from dataclasses import replace

from typing import Any, Protocol

from state.legality import legal_actions
from state.protocol import Facts, ObservationStatus, ObservedValue, S2ProtocolState
from state.tile_counting import compute_seen_counts, compute_tile_statistics


class DegradationOperator(Protocol):
    def apply(self, state: S2ProtocolState) -> S2ProtocolState:
        ...


class MidGameSnapshot:
    def __init__(self, k: int) -> None:
        self.k = k

    def apply(self, state: S2ProtocolState) -> S2ProtocolState:
        cloned = _clone_state(state)
        facts = replace(
            cloned.facts,
            exchange_tracking=ObservedValue.unknown(),
            event_history=ObservedValue.observed(_events_from_start(cloned.facts.event_history, self.k)),
        )
        return _with_recomputed_counts(replace(cloned, facts=facts, observation_start=ObservedValue.observed(self.k)))


class MaskExchange:
    def __init__(self, p: float, seed: int | None = None) -> None:
        self.p = p
        self.seed = seed

    def apply(self, state: S2ProtocolState) -> S2ProtocolState:
        cloned = _clone_state(state)
        if random.Random(self.seed).random() >= self.p:
            return cloned
        return replace(cloned, facts=replace(cloned.facts, exchange_tracking=ObservedValue.unknown()))


class VisionNoise:
    def __init__(
        self,
        confusion_matrix: dict[str, dict[str, float]] | None = None,
        miss_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.confusion_matrix = confusion_matrix or {}
        self.miss_rate = miss_rate
        self.seed = seed

    def apply(self, state: S2ProtocolState) -> S2ProtocolState:
        cloned = _clone_state(state)
        rng = random.Random(self.seed)
        players = []
        changed = False
        for player in cloned.facts.players.value or []:
            degraded_player = dict(player)
            rivers = player.get("rivers")
            if isinstance(rivers, ObservedValue) and rivers.status is not ObservationStatus.UNKNOWN:
                tiles, field_changed = self._degrade_tiles(list(rivers.value or []), rng)
                if field_changed:
                    degraded_player["rivers"] = ObservedValue.estimated(tiles, _confidence(self.miss_rate, changed=True))
                    changed = True
            melds = player.get("melds")
            if isinstance(melds, ObservedValue) and melds.status is not ObservationStatus.UNKNOWN:
                degraded_melds, field_changed = self._degrade_melds(list(melds.value or []), rng)
                if field_changed:
                    if degraded_melds is None:
                        degraded_player["melds"] = ObservedValue.unknown()
                    else:
                        degraded_player["melds"] = ObservedValue.estimated(degraded_melds, _confidence(self.miss_rate, changed=True))
                    changed = True
            players.append(degraded_player)


        if not changed:
            return cloned
        facts = replace(cloned.facts, players=ObservedValue.observed(players))
        return _with_recomputed_counts(replace(cloned, facts=facts))

    def _degrade_melds(self, melds: list[dict[str, Any]], rng: random.Random) -> tuple[list[dict[str, Any]] | None, bool]:
        degraded: list[dict[str, Any]] = []
        changed = False
        for meld in melds:
            original_tiles = list(meld.get("tiles", []))
            tiles, field_changed = self._degrade_tiles(original_tiles, rng)
            changed = changed or field_changed
            if len(tiles) != len(original_tiles):
                return None, True
            copied = dict(meld)
            copied["tiles"] = tiles
            degraded.append(copied)
        return degraded, changed


    def _degrade_tiles(self, tiles: list[str], rng: random.Random) -> tuple[list[str], bool]:
        degraded: list[str] = []
        changed = False
        for tile in tiles:
            if rng.random() < self.miss_rate:
                changed = True
                continue
            replacement = self._sample_replacement(tile, rng)
            changed = changed or replacement != tile
            degraded.append(replacement)
        return degraded, changed

    def _sample_replacement(self, tile: str, rng: random.Random) -> str:
        alternatives = self.confusion_matrix.get(tile)
        if not alternatives:
            return tile
        threshold = rng.random()
        cumulative = 0.0
        for replacement, probability in alternatives.items():
            cumulative += probability
            if threshold <= cumulative:
                return replacement
        return tile


class MaskField:
    def __init__(self, field: str, p: float, seed: int | None = None) -> None:
        self.field = field
        self.p = p
        self.seed = seed

    def apply(self, state: S2ProtocolState) -> S2ProtocolState:
        cloned = _clone_state(state)
        if random.Random(self.seed).random() >= self.p:
            return cloned
        if self.field in Facts.__dataclass_fields__:
            return replace(cloned, facts=replace(cloned.facts, **{self.field: ObservedValue.unknown()}))
        if self.field == "phase":
            return replace(cloned, phase=ObservedValue.unknown())
        if self.field == "current_player":
            return replace(cloned, current_player=ObservedValue.unknown(), current_player_relative=ObservedValue.unknown())
        if self.field == "rule_config":
            return replace(cloned, rule_config=ObservedValue.unknown())
        raise ValueError(f"unsupported mask field: {self.field}")


class DegradationPipeline:
    def __init__(self, operators: list[DegradationOperator], recompute_statistics: bool = True) -> None:
        self.operators = operators
        self.recompute_statistics = recompute_statistics

    def apply(self, state: S2ProtocolState) -> S2ProtocolState:
        degraded = _clone_state(state)
        for operator in self.operators:
            degraded = operator.apply(degraded)
        if self.recompute_statistics:
            degraded = _with_recomputed_counts(degraded)
        return _with_legal_actions(degraded)


def _clone_state(state: S2ProtocolState) -> S2ProtocolState:
    return copy.deepcopy(state)



def _events_from_start(event_history: ObservedValue[list[dict[str, Any]]], start: int) -> list[dict[str, Any]]:
    if event_history.status is ObservationStatus.UNKNOWN:
        return []
    return [event for event in event_history.value or [] if int(event.get("seq", start)) >= start]


def _with_recomputed_counts(state: S2ProtocolState) -> S2ProtocolState:
    count_status = _count_status(state)
    facts = replace(state.facts, seen_counts=ObservedValue(compute_seen_counts(state), count_status, _status_confidence(count_status)))
    counted = replace(state, facts=facts)
    return replace(counted, statistics=compute_tile_statistics(counted))


def _with_legal_actions(state: S2ProtocolState) -> S2ProtocolState:
    return replace(state, legal_actions=ObservedValue.observed(legal_actions(state)))


def _count_status(state: S2ProtocolState) -> ObservationStatus:
    if state.facts.players.status is ObservationStatus.UNKNOWN:
        return ObservationStatus.ESTIMATED
    for player in state.facts.players.value or []:
        for key in ("concealed_hand", "rivers", "melds"):
            value = player.get(key)
            if isinstance(value, ObservedValue) and value.status is ObservationStatus.ESTIMATED:
                return ObservationStatus.ESTIMATED
    if state.facts.revealed_win_hands.status is ObservationStatus.ESTIMATED:
        return ObservationStatus.ESTIMATED
    return ObservationStatus.OBSERVED


def _status_confidence(status: ObservationStatus) -> float:
    return 1.0 if status is ObservationStatus.OBSERVED else 0.8


def _confidence(miss_rate: float, *, changed: bool) -> float:
    if not changed:
        return 1.0
    return min(0.99, max(0.0, 1.0 - miss_rate))
