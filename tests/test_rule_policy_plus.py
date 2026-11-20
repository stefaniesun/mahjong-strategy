"""S3+ 的开关:全关时必须和 S3 一模一样,每个开关只改它该改的决策。"""
import pytest

from assistant.driver import run_game
from engine.actions import Action, ActionKind
from engine.game import Game
from engine.gang import GangKind
from engine.hand import Hand
from engine.tiles import Suit, parse_tile
from policies.decision_boundary import choose_policy_action
from policies.rule_policy import RulePolicy
from policies.rule_policy_plus import PlusConfig, RulePlusPolicy
from tools.eval_paired import play, summarize


def _turn(hand: str, *, rivers=None, voids=None, seed: int = 5) -> Game:
    game = Game(seed=seed)
    state = game.reset()
    state.phase = "play"
    state.current_player = 0
    state.void_suits = [Suit(v) if v else None for v in (voids or ["B", "T", "T", "T"])]
    state.hands[0] = Hand.from_strings(hand.split())
    state.rivers = [[parse_tile(t) for t in r.split()] for r in (rivers or ["", "", "", ""])]
    return game


def _act(game: Game, config: PlusConfig, player: int = 0) -> Action:
    return choose_policy_action(game.state, player, RulePlusPolicy(config)).action


def test_all_switches_off_plays_exactly_like_s3():
    for seed in range(800, 808):
        base, _ = run_game(seed, policies=[RulePolicy() for _ in range(4)])
        plus, _ = run_game(seed, policies=[RulePlusPolicy() for _ in range(4)])
        assert base.state.to_dict() == plus.state.to_dict()


def test_parse_switches():
    assert PlusConfig.parse("none") == PlusConfig()
    assert PlusConfig.parse("smart_kong,void_order") == PlusConfig(smart_kong=True, void_order=True)
    assert PlusConfig.parse("all").safe_tiebreak is True
    with pytest.raises(ValueError):
        PlusConfig.parse("not_a_switch")


def test_void_order_discards_unseen_tile_first_then_edge_before_middle():
    # 缺筒,手里 5筒、9筒:都没现过时先打边张 9筒
    game = _turn("5B 9B 1W 2W 3W 4W 5W 6W 7W 8W 9W 1T 2T 3T")
    assert _act(game, PlusConfig(void_order=True)).tile == parse_tile("9B")
    # 9筒已经现过,5筒没现过:先打没现过的 5筒
    game = _turn("5B 9B 1W 2W 3W 4W 5W 6W 7W 8W 9W 1T 2T 3T", rivers=["", "9B", "", ""])
    assert _act(game, PlusConfig(void_order=True)).tile == parse_tile("5B")


def test_smart_kong_keeps_a_wide_wait_instead_of_breaking_it():
    # 听牌:2222 3 + 其余完整,杠了 2万只剩孤张 3万,会把好叫杠坏
    game = _turn("2W 2W 2W 2W 3W 4W 5W 6W 7W 8W 1T 2T 3T 9T")
    kong = Action(ActionKind.KONG, tile=parse_tile("2W"), kong_kind=GangKind.CONCEALED)
    assert kong in game.legal_actions(0)
    assert _act(game, PlusConfig()) == kong  # S3 能杠就杠
    assert _act(game, PlusConfig(smart_kong=True)) != kong


def test_smart_kong_still_kongs_when_far_from_tenpai():
    game = _turn("5W 5W 5W 5W 1W 9W 2T 7T 9T 1B 4B 8B 3T 6T")
    kong = Action(ActionKind.KONG, tile=parse_tile("5W"), kong_kind=GangKind.CONCEALED)
    assert _act(game, PlusConfig(smart_kong=True)) == kong


def test_paired_self_check_gives_zero_difference():
    seeds = range(900, 920)
    base = [play(s, None) for s in seeds]
    cand = [play(s, "none") for s in seeds]
    summary = summarize(base, cand)
    assert summary["diff_mean"] == 0 and summary["changed_games"] == 0 and summary["bad_games"] == 0
