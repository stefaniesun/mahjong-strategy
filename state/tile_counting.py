from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.tiles import Tile, parse_tile, tile_to_str
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState, Statistics


@dataclass(frozen=True)
class SoftCountReconciliation:
    seen_counts: list[float]
    remaining_counts: list[float]
    unknown_pool_total: float
    contradiction: bool


def compute_seen_counts(state: S2ProtocolState) -> list[float]:
    counts = [0.0] * 27
    if state.facts.players.status is not ObservationStatus.OBSERVED:
        return counts

    for player in state.facts.players.value:
        concealed = player.get("concealed_hand")
        if isinstance(concealed, ObservedValue) and concealed.status is ObservationStatus.OBSERVED:
            for tile_text in concealed.value or []:
                counts[_tile_index(tile_text)] += 1.0

        rivers = player.get("rivers")
        if isinstance(rivers, ObservedValue) and rivers.status in {ObservationStatus.OBSERVED, ObservationStatus.ESTIMATED}:
            confidence = rivers.confidence if rivers.status is ObservationStatus.ESTIMATED else 1.0
            for tile_text in rivers.value or []:
                counts[_tile_index(tile_text)] += confidence

        melds = player.get("melds")
        if isinstance(melds, ObservedValue) and melds.status in {ObservationStatus.OBSERVED, ObservationStatus.ESTIMATED}:
            confidence = melds.confidence if melds.status is ObservationStatus.ESTIMATED else 1.0
            for meld in melds.value or []:
                for tile_text in meld.get("tiles", []):
                    counts[_tile_index(tile_text)] += confidence

    if state.facts.revealed_win_hands.status in {ObservationStatus.OBSERVED, ObservationStatus.ESTIMATED}:
        confidence = state.facts.revealed_win_hands.confidence if state.facts.revealed_win_hands.status is ObservationStatus.ESTIMATED else 1.0
        revealed_players = set(_revealed_player_ids(state.facts.revealed_win_hands.value or {}))
        for player_id, hand_tiles in (state.facts.revealed_win_hands.value or {}).items():
            if _player_has_observed_concealed_hand(state, int(player_id)):
                continue
            for tile_text in hand_tiles:
                counts[_tile_index(tile_text)] += confidence

    if (
        state.phase.status is ObservationStatus.OBSERVED
        and state.phase.value == "swap_three"
        and state.facts.exchange_tracking.status is ObservationStatus.OBSERVED
    ):
        own_swap_out = (state.facts.exchange_tracking.value or {}).get("own_swap_out")
        for tile_text in own_swap_out or []:
            counts[_tile_index(tile_text)] += 1.0

    return [min(4.0, count) for count in counts]


def compute_tile_statistics(state: S2ProtocolState) -> Statistics:
    seen_counts = compute_seen_counts(state)
    status = _combined_count_status(state)
    unknown_pool = _unknown_pool_breakdown(state)
    unknown_total = _unknown_pool_total(unknown_pool)
    reconciled = reconcile_soft_counts(seen_counts, unknown_total)
    remaining = ObservedValue(
        value=reconciled.remaining_counts,
        status=status,
        confidence=1.0 if status is ObservationStatus.OBSERVED else 0.8,
    )
    return Statistics(
        remaining_tile_counts=remaining,
        unknown_pool_breakdown=ObservedValue.observed(unknown_pool),
        own_hand_analysis=state.statistics.own_hand_analysis,
        candidate_action_features=state.statistics.candidate_action_features,
        dingque_constraints=state.statistics.dingque_constraints,
    )


def reconcile_soft_counts(seen_counts: list[float], unknown_pool_total: float) -> SoftCountReconciliation:
    contradiction = any(count < 0.0 or count > 4.0 for count in seen_counts)
    clamped = [min(4.0, max(0.0, float(count))) for count in seen_counts]
    target_seen_total = max(0.0, 108.0 - float(unknown_pool_total))
    current_seen_total = sum(clamped)

    if current_seen_total > 0.0 and abs(current_seen_total - target_seen_total) > 1e-9:
        scale = target_seen_total / current_seen_total
        scaled = [count * scale for count in clamped]
        if all(count <= 4.0 for count in scaled):
            clamped = scaled
        else:
            contradiction = True
    elif current_seen_total == 0.0 and target_seen_total > 0.0:
        contradiction = True

    clamped = [min(4.0, max(0.0, count)) for count in clamped]
    remaining = [max(0.0, 4.0 - count) for count in clamped]
    adjusted_unknown = max(0.0, 108.0 - sum(clamped))
    if abs(adjusted_unknown - unknown_pool_total) > 1e-9:
        contradiction = True
    return SoftCountReconciliation(
        seen_counts=clamped,
        remaining_counts=remaining,
        unknown_pool_total=adjusted_unknown,
        contradiction=contradiction,
    )


def hypergeometric_location_prior(
    remaining_counts: list[float],
    unknown_pool_breakdown: ObservedValue[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    weights = _location_weights(unknown_pool_breakdown)
    total = sum(weights.values())
    if total <= 0.0:
        weights = {"wall": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}
        total = 4.0

    distribution = {location: weight / total for location, weight in weights.items()}
    priors: dict[str, dict[str, float]] = {}
    for index, remaining in enumerate(remaining_counts):
        if remaining > 0:
            priors[tile_to_str(Tile.from_index(index))] = dict(distribution)
    return priors


def _combined_count_status(state: S2ProtocolState) -> ObservationStatus:
    if state.facts.players.status is ObservationStatus.UNKNOWN:
        return ObservationStatus.ESTIMATED
    if state.facts.players.status is ObservationStatus.ESTIMATED:
        return ObservationStatus.ESTIMATED
    for player in state.facts.players.value or []:
        for key in ("rivers", "melds", "concealed_hand"):
            value = player.get(key)
            if isinstance(value, ObservedValue) and value.status is ObservationStatus.ESTIMATED:
                return ObservationStatus.ESTIMATED
    if state.facts.revealed_win_hands.status is ObservationStatus.ESTIMATED:
        return ObservationStatus.ESTIMATED
    return ObservationStatus.OBSERVED


def _unknown_pool_breakdown(state: S2ProtocolState) -> dict[str, Any]:
    opponents: dict[str, int] = {}
    for player in state.facts.players.value or []:
        if player.get("relative_position") == 0:
            continue
        won = player.get("won")
        if isinstance(won, ObservedValue) and won.status is ObservationStatus.OBSERVED and won.value:
            continue
        hand_count = player.get("hand_count")
        if isinstance(hand_count, ObservedValue) and hand_count.status is ObservationStatus.OBSERVED:
            opponents[str(player["relative_position"])] = hand_count.value

    wall = state.facts.wall_count.value if state.facts.wall_count.status is ObservationStatus.OBSERVED else None
    return {"wall": wall, "opponents": opponents}


def _unknown_pool_total(breakdown: dict[str, Any]) -> float:
    wall = breakdown.get("wall")
    return float(wall or 0) + sum(float(count) for count in breakdown.get("opponents", {}).values())


def _location_weights(unknown_pool_breakdown: ObservedValue[dict[str, Any]]) -> dict[str, float]:
    if unknown_pool_breakdown.status is ObservationStatus.UNKNOWN or unknown_pool_breakdown.value is None:
        return {"wall": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}
    value = unknown_pool_breakdown.value
    weights: dict[str, float] = {}
    wall = value.get("wall")
    if wall is None:
        return {"wall": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}
    weights["wall"] = float(wall)
    weights.update({str(relative): float(count) for relative, count in value.get("opponents", {}).items()})
    return weights


def _revealed_player_ids(revealed: dict[Any, list[str]]) -> list[int]:
    return [int(player_id) for player_id in revealed]


def _player_has_observed_concealed_hand(state: S2ProtocolState, player_id: int) -> bool:
    for player in state.facts.players.value or []:
        if player.get("player_id") != player_id:
            continue
        concealed = player.get("concealed_hand")
        return isinstance(concealed, ObservedValue) and concealed.status is ObservationStatus.OBSERVED
    return False


def _tile_index(tile_text: str) -> int:
    return parse_tile(tile_text).index
