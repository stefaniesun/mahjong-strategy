from engine.actions import Action, ActionKind
from engine.gang import GangKind
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.state import GameState, PendingDiscard, PendingRobKong

from engine.tiles import Suit, parse_tile, tile_to_str
from policies.heuristics import choose_discard, choose_swap_tiles, choose_void_suit, should_pong, visible_hand_and_void_suit

from policies.protocol_actions import (
    action_from_protocol,
    action_to_protocol,
    actions_from_mask,
    validate_legal_mask,
    validate_policy_action,
)
from policies.rule_policy import RulePolicy
from state.action_space import action_space_size, action_to_index, index_to_action
from state.adapters.from_engine import from_engine


def hand(tiles: list[str]) -> Hand:

    return Hand.from_strings(tiles)


def tile_texts(tiles) -> list[str]:
    return [tile_to_str(tile) for tile in tiles]


def policy_input(state: GameState, player: int, actions: list[Action]):
    protocol_state = from_engine(state, player)
    mask = [False] * action_space_size()
    for action in actions:
        mask[action_to_index(action_to_protocol(action))] = True
    return protocol_state, mask


def test_protocol_action_conversion_covers_fixed_action_families():

    actions = [
        Action(ActionKind.DISCARD, tile=parse_tile("1W")),
        Action(ActionKind.PONG, tile=parse_tile("2T")),
        Action(ActionKind.KONG, tile=parse_tile("3B"), kong_kind=GangKind.EXPOSED),
        Action(ActionKind.KONG, tile=parse_tile("4W"), kong_kind=GangKind.CONCEALED),
        Action(ActionKind.KONG, tile=parse_tile("5T"), kong_kind=GangKind.ADDED),
        Action(ActionKind.WIN),
        Action(ActionKind.SELF_WIN),
        Action(ActionKind.ROB_KONG_WIN),
        Action(ActionKind.PASS),
        Action(ActionKind.DECLARE_VOID, suit=Suit.BING),
        Action(ActionKind.SWAP_THREE, tiles=(parse_tile("1W"), parse_tile("2W"), parse_tile("3W"))),
    ]

    for action in actions:
        protocol_action = action_to_protocol(action)
        assert action_from_protocol(protocol_action) == action
        assert action_to_index(protocol_action) >= 0

    for index in range(action_space_size()):
        protocol_action = index_to_action(index)
        assert action_to_protocol(action_from_protocol(protocol_action)) == protocol_action


def test_protocol_action_conversion_rejects_noncanonical_fields():
    try:
        action_from_protocol({"kind": "pass", "tile": "1W"})
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("protocol action with extra fields must be rejected")

    mask = [False] * action_space_size()
    mask[action_to_index({"kind": "win"})] = True
    try:
        validate_policy_action(Action(ActionKind.WIN, tile=parse_tile("1W")), mask)
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("engine action with extra fields must be rejected")


def test_legal_mask_validation_and_candidate_enumeration_are_strict():

    mask = [False] * action_space_size()
    expected = Action(ActionKind.PASS)
    mask[action_to_index(action_to_protocol(expected))] = True

    assert validate_legal_mask(mask) is None
    assert actions_from_mask(mask) == [expected]
    assert validate_policy_action(expected, mask) == expected

    try:
        validate_legal_mask(mask[:-1])
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("short mask must be rejected")

    try:
        validate_legal_mask([False] * action_space_size())
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty mask must be rejected")

    invalid_mask = list(mask)
    invalid_mask[0] = 1
    try:
        validate_legal_mask(invalid_mask)
    except TypeError as exc:
        assert "boolean" in str(exc)
    else:
        raise AssertionError("non-boolean mask entries must be rejected")

    try:
        validate_policy_action(Action(ActionKind.DRAW), mask)

    except ValueError as exc:
        assert "fixed action space" in str(exc)
    else:
        raise AssertionError("unmapped policy action must be rejected")

    try:
        validate_policy_action(Action(ActionKind.WIN), mask)
    except ValueError as exc:
        assert "legal mask" in str(exc)
    else:
        raise AssertionError("masked policy action must be rejected")


def test_choose_void_suit_uses_fewest_suit():

    candidate = hand(["1W", "2W", "3W", "4W", "1T", "2T", "3T", "4T", "5T", "6T", "1B", "9B", "9B"])

    assert choose_void_suit(candidate) is Suit.BING


def test_choose_swap_tiles_prefers_void_suit_isolated_tiles_without_breaking_pair():
    candidate = hand(["1W", "1W", "4W", "8W", "2T", "3T", "4T", "5T", "6T", "2B", "3B", "4B", "5B"])

    chosen = choose_swap_tiles(candidate)

    assert set(tile_texts(chosen)) == {"1W", "4W", "8W"}



def test_choose_discard_first_discards_void_suit_tile():
    candidate = hand(["1W", "9W", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "2B", "3B", "4B", "8B", "8B"])

    chosen = choose_discard(candidate, void_suit=Suit.WAN)

    assert chosen.suit is Suit.WAN


def test_choose_discard_keeps_pair_when_shanten_is_tied():
    candidate = hand(["1W", "2W", "3W", "2T", "3T", "4T", "5B", "6B", "7B", "7W", "8W", "5T", "9B", "9B"])

    chosen = choose_discard(candidate)

    assert tile_to_str(chosen) in {"2T", "5T"}
    assert tile_to_str(chosen) != "9B"



def test_should_pong_only_when_it_reduces_shanten():
    useful = hand(["1W", "2W", "6W", "9W", "2T", "3T", "4T", "5T", "6T", "6T", "8T", "8T", "9T"])
    not_useful = hand(["2W", "3W", "3W", "6W", "7W", "9W", "1T", "3T", "2B", "4B", "6B", "8B", "9B"])

    assert should_pong(useful, parse_tile("6T")) is True
    assert should_pong(not_useful, parse_tile("3W")) is False



def test_visible_hand_extraction_preserves_public_own_melds():
    own_hand = hand(["1W", "2W", "3W"])
    meld = Meld(MeldKind.PONG, (parse_tile("5T"),) * 3, exposed=True, from_player=1)
    own_hand.add_meld(meld)
    state = GameState(hands=[own_hand, Hand(), Hand(), Hand()], wall=[], phase="play")

    visible_hand, _ = visible_hand_and_void_suit(from_engine(state, 0))

    assert visible_hand.melds == [meld]


def test_rule_policy_declares_void_and_swaps_with_heuristics():

    policy = RulePolicy()
    state = GameState(hands=[hand(["1W", "1W", "4W", "8W", "2T", "3T", "4T", "5T", "6T", "2B", "3B", "4B", "5B"])] + [Hand(), Hand(), Hand()], wall=[])


    void_actions = [Action(ActionKind.DECLARE_VOID, suit=suit) for suit in Suit]
    swap_actions = [
        Action(ActionKind.SWAP_THREE, tiles=tuple(parse_tile(text) for text in ["1W", "4W", "8W"])),
        Action(ActionKind.SWAP_THREE, tiles=tuple(parse_tile(text) for text in ["2T", "3T", "4T"])),
    ]
    void_action = policy.choose_action(*policy_input(state, 0, void_actions))
    swap_action = policy.choose_action(*policy_input(state, 0, swap_actions))


    assert void_action == Action(ActionKind.DECLARE_VOID, suit=Suit.WAN)
    assert set(tile_texts(swap_action.tiles)) == {"1W", "4W", "8W"}


def test_rule_policy_always_wins_when_legal():
    policy = RulePolicy()
    state = GameState(hands=[hand(["1W", "2W", "3W"]), Hand(), Hand(), Hand()], wall=[], phase="play")

    for expected, actions in (
        (ActionKind.SELF_WIN, [Action(ActionKind.DISCARD, tile=parse_tile("1W")), Action(ActionKind.SELF_WIN)]),
        (ActionKind.WIN, [Action(ActionKind.PASS), Action(ActionKind.WIN)]),
        (ActionKind.ROB_KONG_WIN, [Action(ActionKind.PASS), Action(ActionKind.ROB_KONG_WIN)]),
    ):
        assert policy.choose_action(*policy_input(state, 0, actions)).kind is expected



def test_rule_policy_discards_only_legal_chosen_tile():
    policy = RulePolicy()
    state = GameState(
        hands=[hand(["1W", "9W", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "2B", "3B", "4B", "8B", "8B"]), Hand(), Hand(), Hand()],
        wall=[],
        phase="play",
        void_suits=[Suit.WAN, None, None, None],
    )

    actions = [Action(ActionKind.DISCARD, tile=parse_tile("1W")), Action(ActionKind.DISCARD, tile=parse_tile("1T"))]
    action = policy.choose_action(*policy_input(state, 0, actions))


    assert action == Action(ActionKind.DISCARD, tile=parse_tile("1W"))


def test_rule_policy_pongs_only_when_helpful_and_passes_otherwise():
    policy = RulePolicy()
    state = GameState(
        hands=[Hand(), hand(["1W", "2W", "6W", "9W", "2T", "3T", "4T", "5T", "6T", "6T", "8T", "8T", "9T"]), Hand(), Hand()],
        wall=[],
        phase="play",
        pending_discard=PendingDiscard(discarder=0, tile=parse_tile("6T")),
    )

    useful_actions = [Action(ActionKind.PONG, tile=parse_tile("6T"))]
    action = policy.choose_action(*policy_input(state, 1, useful_actions))


    assert action == Action(ActionKind.PONG, tile=parse_tile("6T"))

    state.hands[1] = hand(["2W", "3W", "3W", "6W", "7W", "9W", "1T", "3T", "2B", "4B", "6B", "8B", "9B"])
    state.pending_discard = PendingDiscard(discarder=0, tile=parse_tile("3W"))
    unhelpful_actions = [Action(ActionKind.PONG, tile=parse_tile("3W")), Action(ActionKind.PASS)]
    action = policy.choose_action(*policy_input(state, 1, unhelpful_actions))


    assert action == Action(ActionKind.PASS)



def test_rule_policy_takes_legal_kong_before_pass():
    policy = RulePolicy()
    state = GameState(hands=[hand(["1W", "1W", "1W", "1W"]), Hand(), Hand(), Hand()], wall=[parse_tile("9B")], phase="play")

    actions = [Action(ActionKind.PASS), Action(ActionKind.KONG, tile=parse_tile("1W"), kong_kind=GangKind.CONCEALED)]
    action = policy.choose_action(*policy_input(state, 0, actions))


    assert action == Action(ActionKind.KONG, tile=parse_tile("1W"), kong_kind=GangKind.CONCEALED)


def test_rule_policy_takes_self_kong_before_discard():
    policy = RulePolicy()
    state = GameState(hands=[hand(["1W", "1W", "1W", "1W", "2T", "3T", "4T", "5T", "6T", "7T", "8T", "9T", "1B", "2B"]), Hand(), Hand(), Hand()], wall=[parse_tile("9B")], phase="play")

    actions = [
        Action(ActionKind.DISCARD, tile=parse_tile("1W")),
        Action(ActionKind.DISCARD, tile=parse_tile("2T")),
        Action(ActionKind.KONG, tile=parse_tile("1W"), kong_kind=GangKind.CONCEALED),
    ]
    action = policy.choose_action(*policy_input(state, 0, actions))


    assert action == Action(ActionKind.KONG, tile=parse_tile("1W"), kong_kind=GangKind.CONCEALED)





def test_rule_policy_takes_exposed_kong_before_pong_or_pass():
    policy = RulePolicy()
    state = GameState(
        hands=[Hand(), hand(["1W", "1W", "1W", "4W", "4W", "7W", "8W", "8W", "9W", "2T", "4T", "5T", "7T"]), Hand(), Hand()],
        wall=[parse_tile("9B")],
        phase="play",
        pending_discard=PendingDiscard(discarder=2, tile=parse_tile("1W")),
        void_suits=[None, Suit.BING, None, None],
    )

    actions = [
        Action(ActionKind.PONG, tile=parse_tile("1W")),
        Action(ActionKind.KONG, tile=parse_tile("1W"), kong_kind=GangKind.EXPOSED),
        Action(ActionKind.PASS),
    ]
    action = policy.choose_action(*policy_input(state, 1, actions))


    assert action == Action(ActionKind.KONG, tile=parse_tile("1W"), kong_kind=GangKind.EXPOSED)


def test_rule_policy_falls_back_to_first_legal_action():

    policy = RulePolicy()
    state = GameState(hands=[Hand(), Hand(), Hand(), Hand()], wall=[], phase="play", pending_rob_kong=PendingRobKong(0, parse_tile("1W"), [1]))

    actions = [Action(ActionKind.PASS)]
    assert policy.choose_action(*policy_input(state, 1, actions)) == Action(ActionKind.PASS)

