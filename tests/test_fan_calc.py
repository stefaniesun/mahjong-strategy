from engine.config import RuleConfig
from engine.fan_calc import WinContext, calculate_fan
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import parse_tile


def hand(text: str) -> Hand:
    return Hand.from_strings(text.split())


def test_ping_hu_scores_one_point():
    result = calculate_fan(hand("1W 2W 3W 2W 3W 4W 3T 4T 5T 7B 8B 9B 9W 9W"), WinContext())

    assert result.fan == 0
    assert result.score_per_payer == 1
    assert result.patterns == ["ping_hu"]


def test_seven_pairs_scores_two_fan():
    result = calculate_fan(hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W"), WinContext())

    assert result.fan == 2
    assert result.score_per_payer == 4
    assert "seven_pairs" in result.patterns


def test_pure_suit_adds_two_fan():
    result = calculate_fan(hand("1W 2W 3W 2W 3W 4W 4W 5W 6W 7W 8W 9W 9W 9W"), WinContext())

    assert result.fan == 2
    assert result.score_per_payer == 4
    assert "pure_suit" in result.patterns


def test_all_pongs_adds_one_fan():
    result = calculate_fan(hand("1W 1W 1W 2T 2T 2T 3B 3B 3B 4W 4W 4W 9B 9B"), WinContext())

    assert result.fan == 1
    assert result.score_per_payer == 2
    assert "all_pongs" in result.patterns


def test_single_wait_with_four_exposed_melds_is_jin_gou_diao_not_all_pongs():
    h = hand("9B 9B")
    h.melds.extend([
        Meld(MeldKind.PONG, (parse_tile("1W"),) * 3, exposed=True),
        Meld(MeldKind.PONG, (parse_tile("2W"),) * 3, exposed=True),
        Meld(MeldKind.KONG, (parse_tile("3T"),) * 4, exposed=True),
        Meld(MeldKind.PONG, (parse_tile("4B"),) * 3, exposed=True),
    ])

    result = calculate_fan(h, WinContext())

    assert result.fan == 3
    assert result.score_per_payer == 8
    assert "jin_gou_diao" in result.patterns
    assert "root" in result.patterns
    assert "all_pongs" not in result.patterns



def test_roots_and_bonus_fan_and_cap():
    result = calculate_fan(
        hand("1W 1W 1W 1W 2W 2W 3W 3W 4W 4W 5W 5W 6W 6W"),
        WinContext(after_kong=True, robbing_kong=True),
        RuleConfig(max_fan=4),
    )

    assert result.fan == 4
    assert result.score_per_payer == 16
    assert "root" in result.patterns
    assert "after_kong" in result.patterns
    assert "robbing_kong" in result.patterns


def test_self_draw_add_di_mode():
    result = calculate_fan(
        hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W"),
        WinContext(self_draw=True),
        RuleConfig(self_draw_mode="add_di"),
    )

    assert result.fan == 2
    assert result.score_per_payer == 5


def test_self_draw_add_fan_mode():
    result = calculate_fan(
        hand("1W 1W 2W 2W 3T 3T 4T 4T 5B 5B 6B 6B 9W 9W"),
        WinContext(self_draw=True),
        RuleConfig(self_draw_mode="add_fan"),
    )

    assert result.fan == 3
    assert result.score_per_payer == 8


def test_two_identical_chows_are_not_all_pongs():
    # 清一色两副相同顺子 234W 234W + 将 99W + 两副碰,是平胡+清一色(2番),
    # 不是对对胡。回归 _is_all_pongs 的奇偶误判 bug。
    h = hand("2W 3W 4W 2W 3W 4W 9W 9W")
    h.melds.extend([
        Meld(MeldKind.PONG, (parse_tile("1W"),) * 3, exposed=True),
        Meld(MeldKind.PONG, (parse_tile("5W"),) * 3, exposed=True),
    ])

    result = calculate_fan(h, WinContext())

    assert result.fan == 2
    assert result.patterns == ["ping_hu", "pure_suit"]
    assert "all_pongs" not in result.patterns


def test_capped_hand_self_draw_add_fan_falls_back_to_add_di():
    # 已顶番(max_fan=4)的牌,加番模式自摸不能白摸:番仍封 4,但分数额外加一个底分。
    capped_hand = hand("1W 1W 1W 1W 2W 2W 3W 3W 4W 4W 5W 5W 6W 6W")
    context = WinContext(self_draw=True, after_kong=True, robbing_kong=True)

    result = calculate_fan(capped_hand, context, RuleConfig(max_fan=4, self_draw_mode="add_fan"))

    assert result.fan == 4
    assert result.score_per_payer == 16 + 1  # 2^4 底分 + 1 个底分(加底)
