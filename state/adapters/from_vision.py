from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from state.legality import legal_actions
from state.protocol import Facts, ObservedValue, S2ProtocolState
from state.tile_belief import PriorBelief
from state.tile_counting import compute_seen_counts, compute_tile_statistics


@dataclass(frozen=True)
class VisionEvent:
    seq: int
    timestamp: float
    event_type: str
    player: int
    tile: str | None = None
    confidence: float = 1.0
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "player": self.player,
            "tile": self.tile,
            "confidence": self.confidence,
            "alternatives": [[tile, confidence] for tile, confidence in self.alternatives],
        }


@dataclass(frozen=True)
class VisionSnapshot:
    seq: int
    timestamp: float
    rivers: dict[int, list[tuple[str, float]]] = field(default_factory=dict)
    melds: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    void_suits: dict[int, tuple[str, float]] = field(default_factory=dict)
    wall_count: tuple[int, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event_type": "snapshot",
            "rivers": {str(player): [[tile, confidence] for tile, confidence in tiles] for player, tiles in self.rivers.items()},
            "melds": {str(player): [dict(meld) for meld in melds] for player, melds in self.melds.items()},
            "void_suits": {str(player): [suit, confidence] for player, (suit, confidence) in self.void_suits.items()},
            "wall_count": None if self.wall_count is None else [self.wall_count[0], self.wall_count[1]],
        }


@dataclass(frozen=True)
class VisionReconciliationReport:
    contradictions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _PlayerAccumulator:
    rivers: list[str] = field(default_factory=list)
    river_confidences: list[float] = field(default_factory=list)
    melds: list[dict[str, Any]] = field(default_factory=list)
    meld_confidences: list[float] = field(default_factory=list)
    void_suit: str | None = None
    void_suit_confidence: float = 0.0


def from_vision_events(
    events: list[VisionEvent | VisionSnapshot],
    perspective_player: int,
    current_player: int | None = None,
    wall_count: int | None = None,
) -> tuple[S2ProtocolState, VisionReconciliationReport]:
    sorted_events = sorted(events, key=lambda event: event.seq)
    observation_start = sorted_events[0].seq if sorted_events else 0
    players = [_PlayerAccumulator() for _ in range(4)]
    event_history: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    estimated_wall_count = (wall_count, 1.0) if wall_count is not None else None

    for event in sorted_events:
        if isinstance(event, VisionSnapshot):
            if _snapshot_has_contradiction(event, players):
                contradictions.append({"seq": event.seq, "reason": "tile_count_exceeds_four"})
                continue
            _apply_snapshot(event, players)
            if event.wall_count is not None:
                estimated_wall_count = event.wall_count
            event_history.append(event.to_dict())
            continue

        if _event_has_contradiction(event, players):
            contradictions.append({"seq": event.seq, "reason": "tile_count_exceeds_four"})
            continue
        _apply_event(event, players)
        event_history.append(event.to_dict())

    state = _build_state(
        perspective_player=perspective_player,
        current_player=current_player,
        wall_count=estimated_wall_count,
        observation_start=observation_start,
        players=players,
        event_history=event_history,
    )
    return state, VisionReconciliationReport(contradictions=contradictions)


def _apply_snapshot(snapshot: VisionSnapshot, players: list[_PlayerAccumulator]) -> None:
    for player, tiles in snapshot.rivers.items():
        players[player].rivers = [tile for tile, _ in tiles]
        players[player].river_confidences = [confidence for _, confidence in tiles]
    for player, melds in snapshot.melds.items():
        players[player].melds = [_normalize_meld(meld) for meld in melds]
        players[player].meld_confidences = [float(meld.get("confidence", 1.0)) for meld in melds]
    for player, (suit, confidence) in snapshot.void_suits.items():
        players[player].void_suit = suit
        players[player].void_suit_confidence = confidence


def _apply_event(event: VisionEvent, players: list[_PlayerAccumulator]) -> None:
    player = players[event.player]
    if event.event_type == "discard" and event.tile is not None:
        player.rivers.append(event.tile)
        player.river_confidences.append(event.confidence)
    elif event.event_type == "pong" and event.tile is not None:
        player.melds.append({"kind": "pong", "tiles": [event.tile] * 3, "from_player": None})
        player.meld_confidences.append(event.confidence)
    elif event.event_type == "kong" and event.tile is not None:
        player.melds.append({"kind": "kong", "tiles": [event.tile] * 4, "from_player": None})
        player.meld_confidences.append(event.confidence)
    elif event.event_type == "hu" and event.tile is not None:
        player.rivers.append(event.tile)
        player.river_confidences.append(event.confidence)
    elif event.event_type == "dingque" and event.tile is not None:
        player.void_suit = event.tile
        player.void_suit_confidence = event.confidence
    elif event.event_type == "deal" and event.tile is not None:
        pass


def _event_has_contradiction(event: VisionEvent, players: list[_PlayerAccumulator]) -> bool:
    if event.tile is None or event.event_type not in {"discard", "pong", "kong", "hu"}:
        return False
    added = 3 if event.event_type == "pong" else 4 if event.event_type == "kong" else 1
    return _tile_count(players, event.tile) + added > 4


def _snapshot_has_contradiction(snapshot: VisionSnapshot, players: list[_PlayerAccumulator]) -> bool:
    counts: dict[str, int] = {}
    for player in players:
        for tile in player.rivers:
            counts[tile] = counts.get(tile, 0) + 1
        for meld in player.melds:
            for tile in meld.get("tiles", []):
                counts[tile] = counts.get(tile, 0) + 1
    for tiles in snapshot.rivers.values():
        for tile, _ in tiles:
            counts[tile] = counts.get(tile, 0) + 1
    for melds in snapshot.melds.values():
        for meld in melds:
            for tile in meld.get("tiles", []):
                counts[tile] = counts.get(tile, 0) + 1
    return any(count > 4 for count in counts.values())


def _build_state(
    perspective_player: int,
    current_player: int | None,
    wall_count: tuple[int, float] | None,
    observation_start: int,
    players: list[_PlayerAccumulator],
    event_history: list[dict[str, Any]],
) -> S2ProtocolState:
    facts = Facts(
        players=ObservedValue.observed([_player_dict(index, player, perspective_player) for index, player in enumerate(players)]),
        dealer=ObservedValue.unknown(),
        dealer_relative_position=ObservedValue.unknown(),
        is_dealer=ObservedValue.unknown(),
        wall_count=_wall_count_value(wall_count),
        is_last_tile=ObservedValue.unknown(),
        pending_discard=ObservedValue.unknown(),
        pending_rob_kong=ObservedValue.unknown(),
        exchange_tracking=ObservedValue.unknown(),
        event_history=ObservedValue.observed(event_history),
        revealed_win_hands=ObservedValue.observed({}),
        seen_counts=ObservedValue.observed([0] * 27),
    )
    base = S2ProtocolState(
        perspective_player=perspective_player,
        phase=ObservedValue.estimated("play", 0.5),
        current_player=ObservedValue.unknown() if current_player is None else ObservedValue.estimated(current_player, 0.5),
        current_player_relative=ObservedValue.unknown() if current_player is None else ObservedValue.estimated((current_player - perspective_player) % 4, 0.5),
        facts=facts,
        observation_start=ObservedValue.observed(observation_start),
        rule_config=ObservedValue.unknown(),
    )
    facts = Facts(
        players=base.facts.players,
        dealer=base.facts.dealer,
        dealer_relative_position=base.facts.dealer_relative_position,
        is_dealer=base.facts.is_dealer,
        wall_count=base.facts.wall_count,
        is_last_tile=base.facts.is_last_tile,
        pending_discard=base.facts.pending_discard,
        pending_rob_kong=base.facts.pending_rob_kong,
        exchange_tracking=base.facts.exchange_tracking,
        event_history=base.facts.event_history,
        revealed_win_hands=base.facts.revealed_win_hands,
        seen_counts=ObservedValue.estimated(compute_seen_counts(base), _state_confidence(players)),
    )
    counted = S2ProtocolState(
        perspective_player=base.perspective_player,
        phase=base.phase,
        current_player=base.current_player,
        current_player_relative=base.current_player_relative,
        facts=facts,
        observation_start=base.observation_start,
        rule_config=base.rule_config,
    )
    with_statistics = S2ProtocolState(
        perspective_player=counted.perspective_player,
        phase=counted.phase,
        current_player=counted.current_player,
        current_player_relative=counted.current_player_relative,
        facts=counted.facts,
        statistics=compute_tile_statistics(counted),
        beliefs=PriorBelief().infer(counted),
        legal_actions=ObservedValue.observed([]),
        observation_start=counted.observation_start,
        rule_config=counted.rule_config,
    )
    return S2ProtocolState(
        perspective_player=with_statistics.perspective_player,
        phase=with_statistics.phase,
        current_player=with_statistics.current_player,
        current_player_relative=with_statistics.current_player_relative,
        facts=with_statistics.facts,
        statistics=with_statistics.statistics,
        beliefs=with_statistics.beliefs,
        legal_actions=ObservedValue.observed(legal_actions(with_statistics)),
        observation_start=with_statistics.observation_start,
        rule_config=with_statistics.rule_config,
    )


def _player_dict(index: int, player: _PlayerAccumulator, perspective_player: int) -> dict[str, Any]:
    return {
        "player_id": index,
        "relative_position": (index - perspective_player) % 4,
        "concealed_hand": ObservedValue.unknown(),
        "hand_count": ObservedValue.unknown(),
        "melds": ObservedValue.estimated(player.melds, _average_confidence(player.meld_confidences)),
        "rivers": ObservedValue.estimated(player.rivers, _average_confidence(player.river_confidences)),
        "void_suit": ObservedValue.unknown() if player.void_suit is None else ObservedValue.estimated(player.void_suit, player.void_suit_confidence),
        "won": ObservedValue.unknown(),
        "passed_hu_lock": ObservedValue.unknown(),
        "passed_fan": ObservedValue.unknown(),
    }


def _wall_count_value(wall_count: tuple[int, float] | None) -> ObservedValue[int]:
    if wall_count is None:
        return ObservedValue.unknown()
    value, confidence = wall_count
    return ObservedValue.estimated(value, confidence)


def _normalize_meld(meld: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": meld.get("kind"),
        "tiles": list(meld.get("tiles", [])),
        "from_player": meld.get("from_player"),
    }


def _tile_count(players: list[_PlayerAccumulator], tile: str) -> int:
    count = 0
    for player in players:
        count += player.rivers.count(tile)
        for meld in player.melds:
            count += list(meld.get("tiles", [])).count(tile)
    return count


def _average_confidence(confidences: list[float]) -> float:
    if not confidences:
        return 1.0
    return sum(confidences) / len(confidences)


def _state_confidence(players: list[_PlayerAccumulator]) -> float:
    confidences: list[float] = []
    for player in players:
        confidences.extend(player.river_confidences)
        confidences.extend(player.meld_confidences)
        if player.void_suit is not None:
            confidences.append(player.void_suit_confidence)
    return _average_confidence(confidences)
