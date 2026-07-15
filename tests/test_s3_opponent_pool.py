from engine.actions import Action, ActionKind
from engine.gang import GangKind
from engine.hand import Hand

from engine.state import GameState
from engine.tiles import Suit, parse_tile
from policies.base_policy import BasePolicy
from policies.opponent_pool import OpponentSpec, create_standard_opponents, make_policy, sample_opponents
from policies.protocol_actions import action_to_protocol
from policies.rule_policy import RulePolicy
from state.action_space import action_space_size, action_to_index, legal_mask

from state.adapters.from_engine import from_engine



def hand(tiles: list[str]) -> Hand:
    return Hand.from_strings(tiles)


def policy_input(state: GameState, player: int, actions: list[Action]):
    protocol_state = from_engine(state, player)
    mask = [False] * action_space_size()
    for action in actions:
        mask[action_to_index(action_to_protocol(action))] = True
    return protocol_state, mask


def test_standard_opponent_pool_exposes_three_strength_tiers():

    pool = create_standard_opponents()

    assert [opponent.key for opponent in pool] == ["random", "greedy", "s3_rule"]
    assert [opponent.strength for opponent in pool] == ["weak", "medium", "baseline"]
    assert all(isinstance(opponent, OpponentSpec) for opponent in pool)
    assert all(issubclass(opponent.policy_cls, BasePolicy) for opponent in pool)


def test_make_policy_builds_named_baseline_instances_with_reproducible_random_policy():
    first = make_policy("random", seed=7)
    second = make_policy("random", seed=7)
    state = GameState(hands=[Hand(), Hand(), Hand(), Hand()], wall=[], phase="play")
    actions = [Action(ActionKind.PASS), Action(ActionKind.KONG, tile=parse_tile("1W"), kong_kind=GangKind.CONCEALED)]


    protocol_state, mask = policy_input(state, 0, actions)
    assert first.choose_action(protocol_state, mask) == second.choose_action(protocol_state, mask)

    assert isinstance(make_policy("s3_rule"), RulePolicy)


def test_greedy_policy_prefers_immediate_win_then_void_tile_discard():
    greedy = make_policy("greedy")
    state = GameState(
        hands=[hand(["1W", "9W", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "2B", "3B", "4B", "8B", "8B"]), Hand(), Hand(), Hand()],
        wall=[],
        phase="play",
        void_suits=[Suit.WAN, None, None, None],
    )

    win_actions = [Action(ActionKind.DISCARD, tile=parse_tile("1W")), Action(ActionKind.SELF_WIN)]
    assert greedy.choose_action(*policy_input(state, 0, win_actions)).kind is ActionKind.SELF_WIN

    discard_actions = [Action(ActionKind.DISCARD, tile=parse_tile("1T")), Action(ActionKind.DISCARD, tile=parse_tile("9W"))]
    action = greedy.choose_action(*policy_input(state, 0, discard_actions))

    assert action == Action(ActionKind.DISCARD, tile=parse_tile("9W"))


def test_deterministic_policies_are_independent_of_hidden_hands_and_wall_order():
    own = hand(["1W", "9W", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "2B", "3B", "4B", "8B", "8B"])
    first_state = GameState(
        hands=[own, hand(["1W"]), hand(["2W"]), hand(["3W"])],
        wall=[parse_tile("4W"), parse_tile("5W")],
        phase="play",
        void_suits=[Suit.WAN, None, None, None],
    )
    second_state = GameState(
        hands=[hand(["1W", "9W", "1T", "2T", "3T", "4T", "5T", "6T", "7T", "2B", "3B", "4B", "8B", "8B"]), hand(["9B"]), hand(["8B"]), hand(["7B"])],
        wall=[parse_tile("5W"), parse_tile("4W")],
        phase="play",
        void_suits=[Suit.WAN, None, None, None],
    )

    first_protocol = from_engine(first_state, 0)
    second_protocol = from_engine(second_state, 0)
    first_mask = legal_mask(first_protocol)
    second_mask = legal_mask(second_protocol)

    assert first_protocol.to_dict() == second_protocol.to_dict()
    assert first_mask == second_mask
    for policy in (RulePolicy(), make_policy("greedy")):
        assert policy.choose_action(first_protocol, first_mask) == policy.choose_action(second_protocol, second_mask)


def test_sample_opponents_is_reproducible():

    first = sample_opponents(size=4, seed=11)
    second = sample_opponents(size=4, seed=11)

    assert [type(policy) for policy in first] == [type(policy) for policy in second]

