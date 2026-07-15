from engine.actions import Action, ActionKind
from engine.game import Game
from engine.gang import GangKind
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.state import GameState, PendingDiscard, PendingRobKong
from engine.tiles import Suit, parse_tile
from state.action_space import action_space_size, action_to_index, index_to_action, legal_mask

from state.adapters.from_engine import from_engine
from state.legality import legal_actions
from state.protocol import ObservedValue


def _kinds(actions):
    return {(action["kind"], action.get("tile"), action.get("kong_kind")) for action in actions}


def test_legal_actions_match_engine_for_discard_self_win_and_kongs():
    game = Game(seed=1)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.hands[0] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W".split())
    state.hands[0].add_meld(Meld(MeldKind.PONG, (parse_tile("7W"),) * 3, exposed=True, from_player=1))
    state.hands[0].add(parse_tile("7W"))
    state.hands[0].add(parse_tile("8W"))
    state.hands[0].add(parse_tile("8W"))
    state.hands[0].add(parse_tile("8W"))
    state.hands[0].add(parse_tile("8W"))

    protocol_state = from_engine(state, player_id=0)

    assert _kinds(legal_actions(protocol_state)) == _kinds(_action_to_dict(action) for action in game.legal_actions(0))


def test_legal_actions_handle_pending_discard_win_priority_and_pass_lock():
    game = Game(seed=2)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.passed_hu_lock[1] = True
    state.passed_fan[1] = 2
    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())
    state.hands[2] = Hand.from_strings("9W 9W 9W 1T 2T 3T 4T 5T 6T 7T 8T 1B 2B".split())
    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))

    assert _kinds(legal_actions(from_engine(state, player_id=1))) == _kinds(_action_to_dict(action) for action in game.legal_actions(1))
    assert _kinds(legal_actions(from_engine(state, player_id=2))) == _kinds(_action_to_dict(action) for action in game.legal_actions(2))


def test_pending_discard_response_includes_pass_with_pong_and_kong():
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "2W", "3W"]),
            Hand.from_strings("5W 5W 5W 1T 2T 3T 4T 5T 6T 7T 8T 1B 2B".split()),
            Hand.from_strings(["4W", "5W", "6W"]),
            Hand.from_strings(["7W", "8W", "9W"]),
        ],
        wall=[parse_tile("1B"), parse_tile("2B")],
        phase="play",
        current_player=0,
        pending_discard=PendingDiscard(discarder=0, tile=parse_tile("5W")),
    )

    actions = legal_actions(from_engine(state, player_id=1))

    assert _kinds(actions) == {
        ("pong", "5W", None),
        ("kong", "5W", GangKind.EXPOSED.value),
        ("pass", None, None),
    }


def test_legal_actions_handle_rob_kong_and_unknown_lock_is_conditional():

    state = GameState(
        hands=[
            Hand.from_strings(["1W", "2W", "3W"]),
            Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split()),
            Hand.from_strings(["4W", "5W", "6W"]),
            Hand.from_strings(["7W", "8W", "9W"]),
        ],
        wall=[parse_tile("1B")],
        phase="play",
        current_player=0,
        pending_rob_kong=PendingRobKong(kong_player=0, tile=parse_tile("9W"), winners=[1]),
    )
    protocol_state = from_engine(state, player_id=1)

    assert _kinds(legal_actions(protocol_state)) == {("rob_kong_win", None, None), ("pass", None, None)}

    unlocked = from_engine(state, player_id=1)
    player = unlocked.facts.players.value[0]
    player["passed_hu_lock"] = ObservedValue.unknown()
    player["passed_fan"] = ObservedValue.unknown()
    actions = legal_actions(unlocked)

    assert any(action["kind"] == "rob_kong_win" and action.get("conditionally_legal") for action in actions)


def test_action_space_round_trip_and_legal_mask():
    action = {"kind": "kong", "tile": "5W", "kong_kind": GangKind.CONCEALED.value}
    assert index_to_action(action_to_index(action)) == action

    state = GameState(
        hands=[Hand.from_strings(["1W", "2W", "3W", "4W"]), Hand(), Hand(), Hand()],
        wall=[parse_tile("1B")],
        phase="play",
        current_player=0,
    )
    protocol_state = from_engine(state, player_id=0)
    mask = legal_mask(protocol_state)

    assert len(mask) == action_space_size()

    assert mask[action_to_index({"kind": "discard", "tile": "1W"})] is True
    assert mask[action_to_index({"kind": "discard", "tile": "9B"})] is False


def _action_to_dict(action: Action) -> dict:
    result = {"kind": action.kind.value}
    if action.tile is not None:
        result["tile"] = f"{action.tile.rank}{action.tile.suit.value}"
    if action.tiles:
        result["tiles"] = [f"{tile.rank}{tile.suit.value}" for tile in action.tiles]
    if action.suit is not None:
        result["suit"] = action.suit.value
    if action.kong_kind is not None:
        result["kong_kind"] = action.kong_kind.value
    return result
