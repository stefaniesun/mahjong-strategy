from engine.actions import Action, ActionKind, legal_swap_actions, swap_direction_from_dice_sum
from engine.game import Game
from engine.hand import Hand
from engine.tiles import Suit, parse_tile


def test_swap_direction_table():
    assert swap_direction_from_dice_sum(2) == 1
    assert swap_direction_from_dice_sum(3) == 2
    assert swap_direction_from_dice_sum(4) == -1
    assert swap_direction_from_dice_sum(11) == 2
    assert swap_direction_from_dice_sum(12) == -1


def test_legal_swap_actions_are_three_same_suit():
    hand = Hand.from_strings(["1W", "2W", "3W", "4T", "5T", "6B"])

    actions = legal_swap_actions(hand)

    assert Action(ActionKind.SWAP_THREE, tiles=(parse_tile("1W"), parse_tile("2W"), parse_tile("3W"))) in actions
    assert all(len({tile.suit for tile in action.tiles}) == 1 for action in actions)
    assert all(len(action.tiles) == 3 for action in actions)


def test_game_reset_is_deterministic_and_deals_correct_counts():
    g1 = Game(seed=123)
    g2 = Game(seed=123)

    s1 = g1.reset()
    s2 = g2.reset()

    assert [h.tiles() for h in s1.hands] == [h.tiles() for h in s2.hands]
    assert s1.hands[0].size == 14
    assert [s1.hands[i].size for i in range(1, 4)] == [13, 13, 13]
    assert len(s1.wall) == 55

    assert s1.phase == "swap_three"


def test_declare_void_actions():
    game = Game(seed=1)
    state = game.reset()
    state.phase = "declare_void"

    actions = game.legal_actions(0)

    assert set(actions) == {
        Action(ActionKind.DECLARE_VOID, suit=Suit.WAN),
        Action(ActionKind.DECLARE_VOID, suit=Suit.TIAO),
        Action(ActionKind.DECLARE_VOID, suit=Suit.BING),
    }
