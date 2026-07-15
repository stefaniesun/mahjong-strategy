from engine.config import RuleConfig
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.state import GameState, PendingDiscard

from engine.tiles import parse_tile
from state.adapters.from_engine import from_engine
from state.protocol import ObservationStatus


def _tiles(texts):
    return [parse_tile(text) for text in texts]


def _player(protocol_state, relative_position):
    return protocol_state.facts.players.value[relative_position]


def test_from_engine_builds_relative_player_view_and_hides_opponent_concealed_hands():
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "1W", "2W", "3W"]),
            Hand.from_strings(["4W", "5W", "6W"]),
            Hand.from_strings(["7W", "8W", "9W"]),
            Hand.from_strings(["1T", "2T", "3T"]),
        ],
        wall=_tiles(["9B", "8B", "7B"]),
        dealer=1,
        current_player=2,
        phase="play",
    )
    state.rivers = [[parse_tile("9W")], [], [], []]
    state.void_suits[2] = parse_tile("1B").suit

    protocol_state = from_engine(state, player_id=2, rule_config=RuleConfig(max_fan=5, self_draw_mode="add_fan"))

    assert protocol_state.version == "s2.v4"
    assert protocol_state.perspective_player == 2
    assert protocol_state.current_player_relative.value == 0
    assert [player["player_id"] for player in protocol_state.facts.players.value] == [2, 3, 0, 1]
    assert _player(protocol_state, 0)["concealed_hand"].value == ["7W", "8W", "9W"]
    assert _player(protocol_state, 1)["concealed_hand"].status is ObservationStatus.UNKNOWN
    assert _player(protocol_state, 2)["concealed_hand"].status is ObservationStatus.UNKNOWN
    assert _player(protocol_state, 3)["concealed_hand"].status is ObservationStatus.UNKNOWN
    assert _player(protocol_state, 1)["hand_count"].value == 3
    assert protocol_state.rule_config.value["max_fan"] == 5
    assert protocol_state.rule_config.value["self_draw_mode"] == "add_fan"


def test_from_engine_counts_seen_tiles_from_own_hand_discards_melds_and_revealed_win_hands():
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "2W", "3W"]),
            Hand.from_strings(["4W", "4W"]),
            Hand.from_strings(["5W", "6W"]),
            Hand.from_strings(["7W", "8W"]),
        ],
        wall=[],
        dealer=0,
        current_player=0,
        phase="play",
    )
    state.hands[0].add_meld(Meld(MeldKind.KONG, tuple(_tiles(["9W", "9W", "9W", "9W"])), exposed=False))
    state.hands[1].add_meld(Meld(MeldKind.PONG, tuple(_tiles(["1T", "1T", "1T"])), exposed=True, from_player=0))
    state.rivers = [[parse_tile("2T")], [parse_tile("3T")], [], []]
    state.won[1] = True
    state.win_order = [1]

    protocol_state = from_engine(state, player_id=0)
    seen_counts = protocol_state.facts.seen_counts.value
    revealed = protocol_state.facts.revealed_win_hands.value

    assert _player(protocol_state, 0)["melds"].value[0]["kind"] == "kong"
    assert _player(protocol_state, 0)["melds"].value[0]["tiles"] == ["9W", "9W", "9W", "9W"]
    assert revealed[1] == ["4W", "4W"]
    assert seen_counts[parse_tile("1W").index] == 1
    assert seen_counts[parse_tile("2T").index] == 1
    assert seen_counts[parse_tile("1T").index] == 3
    assert seen_counts[parse_tile("4W").index] == 2
    assert seen_counts[parse_tile("7W").index] == 0


def test_from_engine_exposes_pending_discard_win_context():
    state = GameState(
        hands=[Hand() for _ in range(4)],
        wall=_tiles(["9B", "8B"]),
        dealer=0,
        current_player=0,
        phase="play",
    )
    state.pending_discard = PendingDiscard(discarder=0, tile=parse_tile("9W"))
    state.pending_discard_after_kong = True
    state.pending_discard_last_wall = True

    pending = from_engine(state, player_id=1).facts.pending_discard.value

    assert pending["discarder"] == 0
    assert pending["tile"] == "9W"
    assert pending["after_kong"] is True
    assert pending["haidi"] is True

