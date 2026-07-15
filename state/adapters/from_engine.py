from __future__ import annotations

from dataclasses import asdict

from engine.config import RuleConfig
from engine.state import GameState
from engine.tiles import Suit, tile_to_str
from state.legality import legal_actions
from state.protocol import Facts, ObservedValue, S2ProtocolState
from state.tile_belief import PriorBelief
from state.tile_counting import compute_seen_counts, compute_tile_statistics



from state.visibility import (
    hand_tiles_to_strings,
    meld_to_dict,
    players_in_relative_order,
    relative_position,
    tiles_to_strings,
    visible_concealed_hand,
)


def from_engine(engine_state: GameState, player_id: int, rule_config: RuleConfig | None = None) -> S2ProtocolState:
    config = rule_config or RuleConfig()
    players = [_player_view(engine_state, player, player_id) for player in players_in_relative_order(player_id)]
    facts = Facts(
        players=ObservedValue.observed(players),
        dealer=ObservedValue.observed(engine_state.dealer),
        dealer_relative_position=ObservedValue.observed(relative_position(engine_state.dealer, player_id)),
        is_dealer=ObservedValue.observed(engine_state.dealer == player_id),
        wall_count=ObservedValue.observed(len(engine_state.wall)),
        is_last_tile=ObservedValue.observed(len(engine_state.wall) == 0),
        pending_discard=_pending_discard(engine_state, player_id),
        pending_rob_kong=_pending_rob_kong(engine_state, player_id),
        exchange_tracking=_exchange_tracking(engine_state, player_id),
        event_history=ObservedValue.observed(_event_history(engine_state)),

        revealed_win_hands=ObservedValue.observed(_revealed_win_hands(engine_state)),
        seen_counts=ObservedValue.observed([0] * 27),
    )
    counting_state = S2ProtocolState(
        perspective_player=player_id,
        phase=ObservedValue.observed(engine_state.phase),
        current_player=ObservedValue.observed(engine_state.current_player),
        current_player_relative=ObservedValue.observed(relative_position(engine_state.current_player, player_id)),
        facts=facts,
        observation_start=ObservedValue.observed(0),
        rule_config=ObservedValue.observed(_rule_config_dict(config)),
    )
    facts = Facts(
        players=facts.players,
        dealer=facts.dealer,
        dealer_relative_position=facts.dealer_relative_position,
        is_dealer=facts.is_dealer,
        wall_count=facts.wall_count,
        is_last_tile=facts.is_last_tile,
        pending_discard=facts.pending_discard,
        pending_rob_kong=facts.pending_rob_kong,
        exchange_tracking=facts.exchange_tracking,
        event_history=facts.event_history,
        revealed_win_hands=facts.revealed_win_hands,
        seen_counts=ObservedValue.observed(compute_seen_counts(counting_state)),
    )
    counting_state = S2ProtocolState(
        perspective_player=player_id,
        phase=counting_state.phase,
        current_player=counting_state.current_player,
        current_player_relative=counting_state.current_player_relative,
        facts=facts,
        observation_start=counting_state.observation_start,
        rule_config=counting_state.rule_config,
    )
    statistics = compute_tile_statistics(counting_state)
    protocol_state = S2ProtocolState(

        perspective_player=player_id,
        phase=ObservedValue.observed(engine_state.phase),
        current_player=ObservedValue.observed(engine_state.current_player),
        current_player_relative=ObservedValue.observed(relative_position(engine_state.current_player, player_id)),
        facts=facts,
        statistics=statistics,
        beliefs=PriorBelief().infer(counting_state),
        legal_actions=ObservedValue.observed([]),

        observation_start=ObservedValue.observed(0),
        rule_config=ObservedValue.observed(_rule_config_dict(config)),
    )
    return S2ProtocolState(
        perspective_player=protocol_state.perspective_player,
        phase=protocol_state.phase,
        current_player=protocol_state.current_player,
        current_player_relative=protocol_state.current_player_relative,
        facts=protocol_state.facts,
        statistics=protocol_state.statistics,
        beliefs=protocol_state.beliefs,
        legal_actions=ObservedValue.observed(legal_actions(protocol_state)),
        observation_start=protocol_state.observation_start,
        rule_config=protocol_state.rule_config,
    )



def _player_view(engine_state: GameState, player: int, perspective_player: int) -> dict:
    hand = engine_state.hands[player]
    is_self = player == perspective_player
    return {
        "player_id": player,
        "relative_position": relative_position(player, perspective_player),
        "concealed_hand": visible_concealed_hand(hand, is_self=is_self, has_won=engine_state.won[player]),
        "hand_count": ObservedValue.observed(hand.size),
        "melds": ObservedValue.observed([meld_to_dict(meld) for meld in hand.melds]),
        "rivers": ObservedValue.observed(tiles_to_strings(engine_state.rivers[player])),
        "void_suit": ObservedValue.observed(_suit_value(engine_state.void_suits[player])),
        "won": ObservedValue.observed(engine_state.won[player]),
        "passed_hu_lock": ObservedValue.observed(engine_state.passed_hu_lock[player] if is_self else None),
        "passed_fan": ObservedValue.observed(engine_state.passed_fan[player] if is_self else None),
    }


def _pending_discard(engine_state: GameState, perspective_player: int) -> ObservedValue[dict | None]:
    pending = engine_state.pending_discard
    if pending is None:
        return ObservedValue.observed(None)
    return ObservedValue.observed(
        {
            "discarder": pending.discarder,
            "discarder_relative": relative_position(pending.discarder, perspective_player),
            "tile": tile_to_str(pending.tile),
            "after_kong": engine_state.pending_discard_after_kong,
            "haidi": engine_state.pending_discard_last_wall,


        }

    )


def _pending_rob_kong(engine_state: GameState, perspective_player: int) -> ObservedValue[dict | None]:
    pending = engine_state.pending_rob_kong
    if pending is None:
        return ObservedValue.observed(None)
    return ObservedValue.observed(
        {
            "kong_player": pending.kong_player,
            "kong_player_relative": relative_position(pending.kong_player, perspective_player),
            "tile": tile_to_str(pending.tile),

        }
    )


def _exchange_tracking(engine_state: GameState, perspective_player: int) -> ObservedValue[dict]:
    own_choice = engine_state.swap_choices[perspective_player]
    return ObservedValue.observed(
        {
            "swap_direction": engine_state.swap_direction,
            "own_swap_out": None if own_choice is None else tiles_to_strings(own_choice),
        }
    )


def _event_history(engine_state: GameState) -> list[dict]:
    event_log = getattr(engine_state, "event_log", None)
    if event_log is not None:
        return [dict(event) for event in event_log]
    if engine_state.pending_discard is None:
        return []
    if engine_state.after_kong_discard_player != engine_state.pending_discard.discarder:
        return []
    return [{"type": "after_kong_discard", "player": engine_state.pending_discard.discarder}]



def _revealed_win_hands(engine_state: GameState) -> dict[int, list[str]]:
    return {
        player: hand_tiles_to_strings(engine_state.hands[player])
        for player, won in enumerate(engine_state.won)
        if won
    }





def _seen_counts(engine_state: GameState, perspective_player: int) -> list[int]:
    counts = [0] * 27
    for tile in engine_state.hands[perspective_player].tiles():
        counts[tile.index] += 1
    for river in engine_state.rivers:
        for tile in river:
            counts[tile.index] += 1
    for hand in engine_state.hands:
        for meld in hand.melds:
            for tile in meld.tiles:
                counts[tile.index] += 1
    for player, won in enumerate(engine_state.won):
        if won:
            for tile in engine_state.hands[player].tiles():
                counts[tile.index] += 1
    return counts


def _unknown_pool_breakdown(engine_state: GameState, perspective_player: int) -> dict:
    opponents = {
        str(relative_position(player, perspective_player)): engine_state.hands[player].size
        for player in range(4)
        if player != perspective_player and not engine_state.won[player]
    }
    return {"wall": len(engine_state.wall), "opponents": opponents}


def _rule_config_dict(config: RuleConfig) -> dict:
    return asdict(config)


def _suit_value(suit: Suit | None) -> str | None:
    return None if suit is None else suit.value
