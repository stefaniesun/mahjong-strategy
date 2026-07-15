from __future__ import annotations

from itertools import combinations
from typing import Any

from engine.fan_calc import WinContext, calculate_fan
from engine.gang import GangKind
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import SUITS, Suit, parse_tile, tile_to_str
from engine.win_check import can_win
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState


ActionDict = dict[str, Any]


def legal_actions(state: S2ProtocolState) -> list[ActionDict]:
    player = _self_player(state)
    won = _observed_value(player.get("won"))
    if won is True:
        return []

    phase = state.phase.value
    if phase == "swap_three":
        return _swap_actions(player)
    if phase == "declare_void":
        return [{"kind": "declare_void", "suit": suit.value} for suit in SUITS]
    if phase != "play":
        return []

    pending_rob = state.facts.pending_rob_kong
    if pending_rob.status is not ObservationStatus.UNKNOWN and pending_rob.value is not None:
        return _rob_kong_actions(state, player, pending_rob.value)

    pending_discard = state.facts.pending_discard
    if pending_discard.status is not ObservationStatus.UNKNOWN and pending_discard.value is not None:
        return _pending_discard_actions(state, player, pending_discard.value)

    if state.current_player_relative.status is ObservationStatus.OBSERVED and state.current_player_relative.value != 0:
        return []

    return _turn_actions(state, player)


def _swap_actions(player: dict[str, Any]) -> list[ActionDict]:
    hand = _hand_from_player(player)
    if hand is None:
        return []
    actions: list[ActionDict] = []
    for suit in SUITS:
        suited_tiles = [tile for tile in hand.tiles() if tile.suit is suit]
        for combo in sorted(set(combinations(suited_tiles, 3))):
            actions.append({"kind": "swap_three", "tiles": [tile_to_str(tile) for tile in combo]})
    return actions


def _rob_kong_actions(state: S2ProtocolState, player: dict[str, Any], pending: dict[str, Any]) -> list[ActionDict]:
    tile = parse_tile(pending["tile"])

    hand = _hand_from_player(player)
    if hand is None:
        return []
    trial = _copy_hand(hand)
    try:
        trial.add(tile)
    except ValueError:
        return [{"kind": "pass"}]
    if not can_win(trial, _void_suit(player)):
        return [{"kind": "pass"}]
    win_action = _win_action_with_lock(state, player, trial, robbing_kong=True, kind="rob_kong_win")
    return ([win_action] if win_action is not None else []) + [{"kind": "pass"}]



def _pending_discard_actions(state: S2ProtocolState, player: dict[str, Any], pending: dict[str, Any]) -> list[ActionDict]:
    if pending.get("discarder_relative") == 0:
        return [{"kind": "pass"}]

    tile = parse_tile(pending["tile"])
    hand = _hand_from_player(player)
    if hand is None:
        return [{"kind": "pass"}]

    win_action = _discard_win_action_if_legal(state, player, hand, tile)
    if win_action is not None:
        return [win_action, {"kind": "pass"}]



    actions: list[ActionDict] = []

    if hand.count(tile) >= 2:
        actions.append({"kind": "pong", "tile": tile_to_str(tile)})
    wall_count = state.facts.wall_count
    wall_available = wall_count.status is ObservationStatus.UNKNOWN or int(wall_count.value) > 1
    if wall_available and hand.count(tile) >= 3:

        action: ActionDict = {"kind": "kong", "tile": tile_to_str(tile), "kong_kind": GangKind.EXPOSED.value}
        if wall_count.status is ObservationStatus.UNKNOWN:
            action = _conditional(action, "wall_count_unknown")
        actions.append(action)
    actions.append({"kind": "pass"})
    return actions


def _turn_actions(state: S2ProtocolState, player: dict[str, Any]) -> list[ActionDict]:
    hand = _hand_from_player(player)
    if hand is None:
        return []
    actions = [{"kind": "discard", "tile": tile_to_str(tile)} for tile in sorted(set(hand.tiles()))]
    if can_win(hand, _void_suit(player)):
        actions.append({"kind": "self_win"})

    wall_count = state.facts.wall_count
    wall_available = wall_count.status is ObservationStatus.UNKNOWN or int(wall_count.value) > 1
    if wall_available:

        for tile in sorted(set(hand.tiles())):
            if hand.count(tile) == 4:
                action: ActionDict = {"kind": "kong", "tile": tile_to_str(tile), "kong_kind": GangKind.CONCEALED.value}
                actions.append(_conditional(action, "wall_count_unknown") if wall_count.status is ObservationStatus.UNKNOWN else action)
        for meld in hand.melds:
            if meld.kind is MeldKind.PONG and hand.count(meld.tiles[0]) >= 1:
                action = {"kind": "kong", "tile": tile_to_str(meld.tiles[0]), "kong_kind": GangKind.ADDED.value}
                actions.append(_conditional(action, "wall_count_unknown") if wall_count.status is ObservationStatus.UNKNOWN else action)
    return actions


def _discard_win_action_if_legal(state: S2ProtocolState, player: dict[str, Any], hand: Hand, tile: Any) -> ActionDict | None:
    trial = _copy_hand(hand)
    try:
        trial.add(tile)
    except ValueError:
        return None
    if not can_win(trial, _void_suit(player)):
        return None
    return _win_action_with_lock(
        state,
        player,
        trial,
        after_kong=_pending_discard_context(state, "after_kong"),
        haidi=_pending_discard_context(state, "haidi"),
        kind="win",
    )



def _win_action_with_lock(
    state: S2ProtocolState,
    player: dict[str, Any],
    hand: Hand,
    *,
    kind: str,
    after_kong: bool = False,
    robbing_kong: bool = False,
    haidi: bool = False,
) -> ActionDict | None:

    action: ActionDict = {"kind": kind}

    lock = player.get("passed_hu_lock")
    fan = player.get("passed_fan")
    if not isinstance(lock, ObservedValue) or lock.status is ObservationStatus.UNKNOWN:
        return _conditional(action, "passed_hu_lock_unknown")
    if lock.value is False:
        return action
    current_fan = calculate_fan(hand, WinContext(after_kong=after_kong, robbing_kong=robbing_kong, haidi=haidi)).fan

    if not isinstance(fan, ObservedValue) or fan.status is ObservationStatus.UNKNOWN:
        return _conditional(action, "passed_fan_unknown")
    if current_fan > int(fan.value):
        return action
    return None



def _self_player(state: S2ProtocolState) -> dict[str, Any]:
    return state.facts.players.value[0]


def _hand_from_player(player: dict[str, Any]) -> Hand | None:
    concealed = player.get("concealed_hand")
    if not isinstance(concealed, ObservedValue) or concealed.status is ObservationStatus.UNKNOWN:
        return None
    hand = Hand.from_strings(concealed.value or [])
    melds = player.get("melds")
    if isinstance(melds, ObservedValue) and melds.status is not ObservationStatus.UNKNOWN:
        for meld in melds.value or []:
            hand.add_meld(
                Meld(
                    MeldKind(meld["kind"]),
                    tuple(parse_tile(tile) for tile in meld["tiles"]),
                    exposed=bool(meld.get("exposed", False)),
                    from_player=meld.get("from_player"),
                )
            )
    return hand


def _void_suit(player: dict[str, Any]) -> Suit | None:
    value = player.get("void_suit")
    if not isinstance(value, ObservedValue) or value.status is ObservationStatus.UNKNOWN or value.value is None:
        return None
    return Suit(value.value)


def _copy_hand(hand: Hand) -> Hand:
    return Hand(counts=list(hand.counts), melds=list(hand.melds))


def _observed_value(value: Any) -> Any:
    if isinstance(value, ObservedValue) and value.status is not ObservationStatus.UNKNOWN:
        return value.value
    return None


def _pending_discard_context(state: S2ProtocolState, key: str) -> bool:
    pending = state.facts.pending_discard.value if state.facts.pending_discard.status is not ObservationStatus.UNKNOWN else None
    return bool(pending is not None and pending.get(key, False))



def _conditional(action: ActionDict, reason: str) -> ActionDict:
    result = dict(action)
    result["conditionally_legal"] = True
    result["depends_on"] = [reason]
    return result
