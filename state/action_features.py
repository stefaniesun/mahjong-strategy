from __future__ import annotations

from typing import Any

from engine.hand import Hand
from engine.tiles import Suit, parse_tile

from engine.ting_check import ting_tiles
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState


def compute_candidate_action_features(state: S2ProtocolState) -> ObservedValue[list[dict[str, Any]]]:
    if state.legal_actions.status is ObservationStatus.UNKNOWN:
        return ObservedValue.unknown()

    player = _self_player(state)
    hand = _hand_from_player(player)
    void_suit = _void_suit(player)
    remaining = state.statistics.remaining_tile_counts
    estimated_inputs: list[str] = []
    if remaining.status is ObservationStatus.ESTIMATED:
        estimated_inputs.append("remaining_tile_counts")
    if state.observation_start.status is not ObservationStatus.UNKNOWN and int(state.observation_start.value or 0) > 0:
        estimated_inputs.append("observation_start")


    features: list[dict[str, Any]] = []
    for action in state.legal_actions.value or []:
        item = dict(action)
        tile_text = item.get("tile")
        if tile_text is not None:
            tile = parse_tile(tile_text)
            item["tile_remaining_count"] = _remaining_count(remaining, tile.index)
            item["is_void_suit"] = void_suit is not None and tile.suit is void_suit
        else:
            item["tile_remaining_count"] = None
            item["is_void_suit"] = None
        item["keeps_ting"] = _keeps_ting_after_discard(hand, tile_text, void_suit) if item.get("kind") == "discard" else None
        item["estimated_inputs"] = list(estimated_inputs)
        features.append(item)

    status = ObservationStatus.ESTIMATED if estimated_inputs or state.legal_actions.status is ObservationStatus.ESTIMATED else ObservationStatus.OBSERVED
    if status is ObservationStatus.ESTIMATED:
        confidences = [state.legal_actions.confidence]
        if remaining.status is ObservationStatus.ESTIMATED:
            confidences.append(remaining.confidence)
        return ObservedValue.estimated(features, min(confidences))
    return ObservedValue.observed(features)


def _self_player(state: S2ProtocolState) -> dict[str, Any]:
    return state.facts.players.value[0]


def _hand_from_player(player: dict[str, Any]) -> Hand | None:
    concealed = player.get("concealed_hand")
    if not isinstance(concealed, ObservedValue) or concealed.status is ObservationStatus.UNKNOWN:
        return None
    return Hand.from_strings(concealed.value or [])


def _void_suit(player: dict[str, Any]) -> Suit | None:
    value = player.get("void_suit")
    if not isinstance(value, ObservedValue) or value.status is ObservationStatus.UNKNOWN or value.value is None:
        return None
    return Suit(value.value)


def _remaining_count(remaining: ObservedValue[list[float]], index: int) -> float | None:
    if remaining.status is ObservationStatus.UNKNOWN or remaining.value is None:
        return None
    return remaining.value[index]


def _keeps_ting_after_discard(hand: Hand | None, tile_text: str | None, void_suit: Suit | None) -> bool | None:
    if hand is None or tile_text is None:
        return None
    tile = parse_tile(tile_text)
    if hand.count(tile) <= 0:
        return None
    trial = Hand(counts=list(hand.counts), melds=list(hand.melds))
    trial.remove(tile)
    if void_suit is not None and any(candidate.suit is void_suit for candidate in trial.tiles()):
        return False
    return bool(ting_tiles(trial, void_suit))

