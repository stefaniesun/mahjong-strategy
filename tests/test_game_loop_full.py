import pytest

from engine.actions import Action, ActionKind
from engine.game import Game
from engine.gang import GangKind
from engine.hand import Hand
from engine.settlement import record_kong_payment
from engine.state import GameState, PendingDiscard


from engine.meld import Meld, MeldKind
from engine.tiles import Suit, parse_tile




def choose_first_legal(game: Game, player: int, kind: ActionKind | None = None) -> Action:
    actions = game.legal_actions(player)
    if kind is None:
        return actions[0]
    return next(action for action in actions if action.kind is kind)


def complete_opening(game: Game) -> None:
    for player in range(4):
        game.step(player, choose_first_legal(game, player, ActionKind.SWAP_THREE))
    assert game.state is not None
    assert game.state.phase == "declare_void"
    for player in range(4):
        game.step(player, Action(ActionKind.DECLARE_VOID, suit=Suit.WAN))
    assert game.state.phase == "play"


def test_state_serializes_full_loop_fields():
    game = Game(seed=11)
    state = game.reset()

    data = state.to_dict()

    assert data["phase"] == "swap_three"
    assert data["rivers"] == [[], [], [], []]
    assert data["swap_direction"] in {-1, 1, 2}
    assert data["pending_discard"] is None
    assert data["gang_records"] == []
    assert data["next_gang_id"] == 1
    assert data["last_transferable_gang_id"] is None
    assert data["after_kong_discard_player"] is None
    assert data["current_draw_after_kong"] is False
    assert data["current_draw_last_wall"] is False
    assert data["pending_discard_after_kong"] is False
    assert data["pending_discard_last_wall"] is False
    assert data["finished"] is False

    assert data["next_dealer"] is None



def test_play_legal_actions_include_discard_and_self_win_when_available():
    game = Game(seed=1)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.hands[0] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W".split())


    actions = game.legal_actions(0)

    assert any(action.kind is ActionKind.DISCARD for action in actions)
    assert Action(ActionKind.SELF_WIN) in actions


def test_opening_step_applies_all_swaps_and_void_declarations():
    game = Game(seed=2)
    state = game.reset()
    original_tiles = [hand.size for hand in state.hands]

    complete_opening(game)

    assert [hand.size for hand in state.hands] == original_tiles
    assert state.phase == "play"
    assert state.current_player == state.dealer
    assert state.void_suits == [Suit.WAN, Suit.WAN, Suit.WAN, Suit.WAN]


def test_swap_three_downstream_dice_10_player_0_receives_player_1_tiles():
    game = Game(seed=1)
    state = game.reset()
    assert state.dice == (5, 5)
    assert state.swap_direction == 1

    choices = []
    for player in range(4):
        action = choose_first_legal(game, player, ActionKind.SWAP_THREE)
        choices.append(action.tiles)
        game.step(player, action)

    for tile in choices[1]:
        assert state.hands[0].count(tile) > 0


def _pass_all_discard_responses(game: Game) -> None:
    state = game.state
    assert state is not None
    while state.pending_discard is not None:
        responder = next(
            player
            for player in range(4)
            if Action(ActionKind.PASS) in game.legal_actions(player)
        )
        game.step(responder, Action(ActionKind.PASS))


def test_discard_removes_tile_and_pass_draws_next_player():

    game = Game(seed=3)
    state = game.reset()
    complete_opening(game)
    player = state.current_player
    tile = state.hands[player].tiles()[0]
    before_wall = len(state.wall)

    game.step(player, Action(ActionKind.DISCARD, tile=tile))
    assert state.pending_discard is not None
    assert state.rivers[player][-1] == tile
    assert state.hands[player].count(tile) == 0 or tile not in state.hands[player].tiles()

    _pass_all_discard_responses(game)

    assert state.pending_discard is None
    assert state.current_player == 1
    assert len(state.wall) == before_wall - 1
    assert state.hands[1].size == 14


def test_self_draw_win_settles_and_marks_winner():
    game = Game(seed=4)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, Suit.BING, Suit.BING, Suit.BING]
    state.hands[0] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W".split())

    game.step(0, Action(ActionKind.SELF_WIN))


    assert state.won[0] is True
    assert state.win_order == [0]
    assert sum(state.scores) == 0


def test_discard_win_response_settles_and_marks_winner():
    game = Game(seed=5)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())


    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    assert 1 in state.pending_winners
    game.step(1, Action(ActionKind.WIN))

    assert state.won[1] is True
    assert state.win_order == [1]
    assert sum(state.scores) == 0
    assert state.pending_discard is None


def test_discard_win_continues_from_winner_next_active_player():
    game = Game(seed=5)
    state = game.reset()
    state.phase = "play"
    state.current_player = 2
    state.void_suits = [None, None, None, None]
    state.hands[2] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[0] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())
    before_wall = len(state.wall)

    game.step(2, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    assert state.pending_winners == [0]
    game.step(0, Action(ActionKind.WIN))

    assert state.won[0] is True
    assert state.current_player == 1
    assert len(state.wall) == before_wall - 1
    assert state.hands[1].size == 14



def test_pass_on_discard_win_sets_pass_lock_and_fan():
    game = Game(seed=6)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())
    state.hands[2] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    game.step(1, Action(ActionKind.PASS))

    assert state.passed_hu_lock[1] is True
    assert state.passed_fan[1] == 2
    assert 2 in state.pending_winners



def test_passing_discard_win_keeps_lower_priority_pong_response_open():
    game = Game(seed=61)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    tile = parse_tile("4W")
    state.hands[0] = Hand.from_strings(
        "4W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B 2B 3B 4B".split()
    )
    state.hands[1] = Hand.from_strings(
        "1W 1W 2W 3W 1T 2T 3T 4T 5T 6T 1B 2B 3B".split()
    )
    state.hands[2] = Hand.from_strings(
        "4W 4W 5W 6W 7W 1T 2T 3T 4T 5T 6T 7T 8T".split()
    )

    game.step(0, Action(ActionKind.DISCARD, tile=tile))
    assert state.pending_winners == [1]
    assert game.legal_actions(2) == []

    game.step(1, Action(ActionKind.PASS))

    assert state.pending_discard is not None
    assert Action(ActionKind.PONG, tile=tile) in game.legal_actions(2)


def test_discarder_cannot_pass_before_other_players_respond():
    game = Game(seed=62)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    tile = parse_tile("5W")
    state.hands[0] = Hand.from_strings(
        "5W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B 2B 3B 4B".split()
    )
    state.hands[1] = Hand.from_strings(
        "5W 5W 1W 2W 3W 4W 6W 7W 8W 1T 2T 3T 4T".split()
    )

    game.step(0, Action(ActionKind.DISCARD, tile=tile))
    before = state.to_dict()

    assert Action(ActionKind.PASS) not in game.legal_actions(0)
    with pytest.raises(ValueError, match="illegal action"):
        game.step(0, Action(ActionKind.PASS))
    assert state.to_dict() == before
    assert Action(ActionKind.PONG, tile=tile) in game.legal_actions(1)


def test_pass_lock_filters_same_or_lower_fan_discard_win():

    game = Game(seed=7)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.passed_hu_lock[1] = True
    state.passed_fan[1] = 2
    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))

    assert 1 not in state.pending_winners
    assert Action(ActionKind.WIN) not in game.legal_actions(1)


def test_pass_lock_allows_bigger_fan_discard_win():
    game = Game(seed=8)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.passed_hu_lock[1] = True
    state.passed_fan[1] = 1
    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))

    assert 1 in state.pending_winners
    assert Action(ActionKind.WIN) in game.legal_actions(1)


def test_pass_lock_clears_when_player_draws():
    game = Game(seed=9)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.passed_hu_lock[1] = True
    state.passed_fan[1] = 2
    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    _pass_all_discard_responses(game)

    assert state.current_player == 1
    assert state.passed_hu_lock[1] is False
    assert state.passed_fan[1] == 0


def test_after_kong_discard_win_adds_after_kong_fan():
    game = Game(seed=10)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.current_draw_after_kong = True

    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    game.step(1, Action(ActionKind.WIN))

    assert state.scores == [-8, 8, 0, 0]
    assert state.after_kong_discard_player is None


def test_paid_kong_after_discard_win_transfers_full_kong_money():
    game = Game(seed=12)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    record = record_kong_payment(state.scores, gang_id=1, kong_player=0, kind=GangKind.CONCEALED)
    state.gang_records.append(record)
    state.last_transferable_gang_id = 1
    state.after_kong_discard_player = 0
    state.current_draw_after_kong = True

    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    game.step(1, Action(ActionKind.WIN))

    assert state.scores == [-8, 12, -2, -2]
    assert state.gang_records[0].transferred_to == (1,)
    assert state.gang_records[0].transfer_count == 1


def test_added_kong_after_discard_win_has_no_transfer():
    game = Game(seed=13)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    record = record_kong_payment(state.scores, gang_id=1, kong_player=0, kind=GangKind.ADDED)
    state.gang_records.append(record)
    state.last_transferable_gang_id = 1
    state.after_kong_discard_player = 0
    state.current_draw_after_kong = True
    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    game.step(1, Action(ActionKind.WIN))

    assert state.scores == [-8, 8, 0, 0]
    assert state.gang_records[0].transferred_to == ()



def test_after_kong_multi_win_gets_copied_kong_transfer():
    game = Game(seed=14)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    record = record_kong_payment(state.scores, gang_id=1, kong_player=0, kind=GangKind.CONCEALED)
    state.gang_records.append(record)
    state.last_transferable_gang_id = 1
    state.after_kong_discard_player = 0
    state.current_draw_after_kong = True

    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())
    state.hands[2] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    game.step(1, Action(ActionKind.WIN))
    game.step(2, Action(ActionKind.WIN))

    assert state.scores == [-22, 12, 12, -2]
    assert state.gang_records[0].transferred_to == (1, 2)
    assert state.gang_records[0].transfer_count == 2
    assert state.after_kong_discard_player is None


def test_discard_response_can_pong_and_take_turn():
    game = Game(seed=15)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    tile = parse_tile("5W")
    state.hands[0] = Hand.from_strings("5W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B 2B 3B 4B".split())
    state.hands[1] = Hand.from_strings("5W 5W 1W 2W 3W 4W 6W 7W 8W 1T 2T 3T 4T".split())

    game.step(0, Action(ActionKind.DISCARD, tile=tile))

    assert Action(ActionKind.PONG, tile=tile) in game.legal_actions(1)

    game.step(1, Action(ActionKind.PONG, tile=tile))

    assert state.pending_discard is None
    assert state.current_player == 1
    assert state.rivers[0] == []
    assert state.hands[1].count(tile) == 0
    assert state.hands[1].melds[-1].kind is MeldKind.PONG
    assert state.hands[1].melds[-1].tiles == (tile, tile, tile)
    assert state.hands[1].melds[-1].from_player == 0


def test_discard_response_can_exposed_kong_settle_and_draw_replacement():
    game = Game(seed=16)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    tile = parse_tile("5W")
    replacement = parse_tile("9B")
    state.wall = [parse_tile("8B"), replacement]
    state.hands[0] = Hand.from_strings("5W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B 2B 3B 4B".split())

    state.hands[1] = Hand.from_strings("5W 5W 5W 1W 2W 3W 4W 6W 7W 8W 1T 2T 3T".split())

    game.step(0, Action(ActionKind.DISCARD, tile=tile))

    assert Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.EXPOSED) in game.legal_actions(1)

    game.step(1, Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.EXPOSED))

    assert state.scores == [-2, 2, 0, 0]
    assert state.current_player == 1
    assert state.pending_discard is None
    assert state.rivers[0] == []
    assert state.hands[1].count(tile) == 0
    assert state.hands[1].count(replacement) == 1
    assert state.hands[1].melds[-1].kind is MeldKind.KONG
    assert state.gang_records[-1].gang_type is GangKind.EXPOSED
    assert state.last_transferable_gang_id == state.gang_records[-1].gang_id
    assert state.after_kong_discard_player == 1


def test_current_player_can_concealed_kong_settle_and_draw_replacement():
    game = Game(seed=17)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    tile = parse_tile("5W")
    replacement = parse_tile("9B")
    state.wall = [parse_tile("8B"), replacement]
    state.hands[0] = Hand.from_strings("5W 5W 5W 5W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B".split())


    assert Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.CONCEALED) in game.legal_actions(0)

    game.step(0, Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.CONCEALED))

    assert state.scores == [6, -2, -2, -2]
    assert state.current_player == 0
    assert state.hands[0].count(tile) == 0
    assert state.hands[0].count(replacement) == 1
    assert state.hands[0].melds[-1].kind is MeldKind.KONG
    assert state.hands[0].melds[-1].exposed is False
    assert state.gang_records[-1].gang_type is GangKind.CONCEALED
    assert state.after_kong_discard_player == 0


def test_current_player_can_added_kong_without_payment_and_draw_replacement():
    game = Game(seed=18)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    tile = parse_tile("5W")
    replacement = parse_tile("9B")
    state.wall = [parse_tile("8B"), replacement]
    state.hands[0] = Hand.from_strings("5W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B".split())

    state.hands[0].melds.append(Meld(MeldKind.PONG, (tile, tile, tile), exposed=True, from_player=2))

    assert Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.ADDED) in game.legal_actions(0)

    game.step(0, Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.ADDED))

    assert state.scores == [0, 0, 0, 0]
    assert state.current_player == 0
    assert state.hands[0].count(tile) == 0
    assert state.hands[0].count(replacement) == 1
    assert state.hands[0].melds[-1].kind is MeldKind.KONG
    assert state.hands[0].melds[-1].tiles == (tile, tile, tile, tile)
    assert state.gang_records[-1].gang_type is GangKind.ADDED
    assert state.gang_records[-1].total_amount == 0
    assert state.after_kong_discard_player == 0


def test_kong_is_not_legal_without_replacement_tile():
    game = Game(seed=19)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    tile = parse_tile("5W")
    state.wall = []
    state.hands[0] = Hand.from_strings("5W 5W 5W 5W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B".split())

    assert Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.CONCEALED) not in game.legal_actions(0)


def test_added_kong_can_be_robbed_and_kong_is_cancelled():
    game = Game(seed=20)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    tile = parse_tile("9W")
    state.wall = [parse_tile("8B"), parse_tile("9B")]
    state.hands[0] = Hand.from_strings("9W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B".split())

    state.hands[0].melds.append(Meld(MeldKind.PONG, (tile, tile, tile), exposed=True, from_player=2))
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.KONG, tile=tile, kong_kind=GangKind.ADDED))

    assert Action(ActionKind.ROB_KONG_WIN) in game.legal_actions(1)

    game.step(1, Action(ActionKind.ROB_KONG_WIN))

    assert state.scores == [-8, 8, 0, 0]
    assert state.won[1] is True
    assert state.hands[0].count(tile) == 1
    assert state.hands[0].melds[-1].kind is MeldKind.PONG
    assert state.gang_records == []
    assert state.pending_rob_kong is None


def test_rob_kong_pass_lock_uses_robbing_kong_fan_context():
    game = Game(seed=63)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.passed_hu_lock[1] = True
    state.passed_fan[1] = 2
    tile = parse_tile("9W")
    state.wall = [parse_tile("8B"), parse_tile("9B")]
    state.hands[0] = Hand.from_strings(
        "9W 1T 2T 3T 4T 5T 6T 7T 8T 9T 1B".split()
    )
    state.hands[0].melds.append(
        Meld(
            MeldKind.PONG,
            (tile, tile, tile),
            exposed=True,
            from_player=2,
        )
    )
    state.hands[1] = Hand.from_strings(
        "1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split()
    )

    game.step(
        0,
        Action(
            ActionKind.KONG,
            tile=tile,
            kong_kind=GangKind.ADDED,
        ),
    )

    assert state.pending_rob_kong is not None
    assert Action(ActionKind.ROB_KONG_WIN) in game.legal_actions(1)


def test_wall_exhaustion_settles_drawn_game_and_keeps_dealer_when_no_winner():

    game = Game(seed=21, dealer=2)
    state = GameState(
        hands=[
            Hand.from_strings("1B 2B 3B 4W 5W 6W 7W 8W 9W 2T 3T 4T 5T".split()),
            Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split()),
            Hand.from_strings("1W 2W 3W 4W 5W 6W 7W 8W 1T 2T 3T 4T 5T".split()),
            Hand.from_strings("1W 2W 3W 4W 5W 6W 7W 8W 1T 2T 3T 4T 5T".split()),
        ],
        wall=[],
        dealer=2,
        current_player=0,
        phase="play",
        void_suits=[Suit.BING, None, None, None],
    )
    state.gang_records.append(record_kong_payment(state.scores, gang_id=1, kong_player=0, kind=GangKind.CONCEALED))
    game.state = state

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("5T")))
    _pass_all_discard_responses(game)

    assert state.finished is True
    assert state.phase == "finished"
    assert state.next_dealer == 2
    assert state.scores == [-4, 12, -4, -4]
    assert state.gang_records[0].refunded is True


def test_first_win_multi_discard_makes_discarder_next_dealer():
    game = Game(seed=22, dealer=3)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [None, None, None, None]
    state.hands[0] = Hand.from_strings("5W 6W 7W 8W 9W 1T 2T 3T 4T 5T 6T 7T 8T 9T".split())
    state.hands[1] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())
    state.hands[2] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())
    state.hands[3] = Hand.from_strings("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W".split())

    game.step(0, Action(ActionKind.DISCARD, tile=parse_tile("9W")))
    game.step(1, Action(ActionKind.WIN))
    game.step(2, Action(ActionKind.WIN))
    game.step(3, Action(ActionKind.WIN))

    assert state.finished is True
    assert state.win_order == [1, 2, 3]
    assert state.next_dealer == 0













