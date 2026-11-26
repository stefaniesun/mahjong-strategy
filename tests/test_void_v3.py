"""测试 void_v3 换后定缺校准特性、受控抽样与 assistant.opening 接入。"""
from __future__ import annotations

import random

import pytest

from assistant.opening import analyze_opening
from assistant.state import SwapInfo
from assistant.void_v3 import (
    choose_void_v3,
    sample_void_v3,
    score_void_candidates_v3,
    void_v3_sampling_probs,
)
from engine.hand import Hand
from engine.tiles import Suit
from assistant_fixtures import make_state


def test_v3_prefers_wider_live_tiles_on_same_shanten_and_count():
    # 4/4/5 闲家: 万4张(1379W散), 条4张(3456T连), 筒5张(23478B)
    # 缺万留条筒显然比缺条留万筒有效进张宽得多
    h = Hand.from_strings("1W 3W 7W 9W 3T 4T 5T 6T 2B 3B 4B 7B 8B".split())
    ranked = score_void_candidates_v3(h.counts)
    assert ranked[0]["suit_code"] == "W"
    assert ranked[0]["live"] > ranked[1]["live"]


def test_v3_avoids_crash_suit_on_true_445_tie_for_non_dealer():
    # 对称 4/4/5 牌型: 万与条牌面完全镜像一致(1479W vs 1479T), 筒5张(23467B)
    # 无换牌信息时默认按花色顺序选万; 但若万为闲家撞车门(换出万且换回万), void_v3 应避开万改缺条防单行道
    h = Hand.from_strings("1W 4W 7W 9W 1T 4T 7T 9T 2B 3B 4B 6B 7B".split())
    assert choose_void_v3(h, is_dealer=False) is Suit.WAN
    assert (
        choose_void_v3(h, is_dealer=False, swap_out_suit="W", swap_in_suit="W", swap_direction=2)
        is Suit.TIAO
    )
    assert (
        choose_void_v3(h, is_dealer=False, swap_out_suit="W", swap_in_suit="W", swap_direction=1)
        is Suit.TIAO
    )


def test_v3_prefers_voiding_out_suit_over_in_suit_on_noncrash_tie():
    # 对称 4/4/5: 万=换进门, 条=换出门 -> 非撞车平手时优先缺换出门(条)避开与给牌方撞缺
    h = Hand.from_strings("1W 4W 7W 9W 1T 4T 7T 9T 2B 3B 4B 6B 7B".split())
    pick_dir2 = choose_void_v3(h, is_dealer=False, swap_out_suit="T", swap_in_suit="W", swap_direction=2)
    pick_dir1 = choose_void_v3(h, is_dealer=False, swap_out_suit="T", swap_in_suit="W", swap_direction=1)
    assert pick_dir2 is Suit.TIAO
    assert pick_dir1 is Suit.TIAO


def test_v3_context_never_overrides_clear_shanten_or_live_advantage():
    # 即使万是撞车门, 但万是烂散牌(1479W)、条是好连张(4567T), 绝不因避撞车而弃好条留烂万
    h = Hand.from_strings("1W 4W 7W 9W 4T 5T 6T 7T 2B 3B 4B 7B 8B".split())
    assert (
        choose_void_v3(h, is_dealer=False, swap_out_suit="W", swap_in_suit="W", swap_direction=2)
        is Suit.WAN
    )


def test_v3_respects_3_close_boundary():
    # 3/5/5: 万3张散牌(149W), 条5张(23467T), 筒5张(23478B) -> 绝不硬缺5张留3张
    h_355 = Hand.from_strings("1W 4W 9W 2T 3T 4T 6T 7T 2B 3B 4B 7B 8B".split())
    assert choose_void_v3(h_355, is_dealer=False) is Suit.WAN

    # 3/4/6: 万3张成顺(456W), 条4张全散(1479T), 筒6张(234568B) -> 缺4张条向听明显更快且进张更宽, 值得破例缺条
    h_346 = Hand.from_strings("4W 5W 6W 1T 4T 7T 9T 2B 3B 4B 5B 6B 8B".split())
    assert choose_void_v3(h_346, is_dealer=False) is Suit.TIAO

    # 3/4/6 跨向听防虚高回归 (#16): 3689W(4,撞车) + 133469T(6) + 345B(3成顺)
    # 缺万(4张)=4向听(live=30), 缺筒(3张)=5向听(live=62虚高); 不得因5向听进张虚高或单门撞车而反拆345B顺子
    h_case16 = Hand.from_strings("3W 6W 8W 9W 1T 3T 3T 4T 6T 9T 3B 4B 5B".split())
    ranked16 = score_void_candidates_v3(
        h_case16.counts, is_dealer=False, swap_out_suit="W", swap_in_suit="W", swap_direction=2
    )
    assert ranked16[0]["suit_code"] == "W"
    assert ranked16[0]["ctx_cost"] == 0.0


def test_v3_sampling_locks_clear_hand_and_explores_tie():
    # 明显 1 张缺门 -> 100% 锁定 #1
    h_clear = Hand.from_strings("9W 1T 2T 3T 4T 5T 6T 2B 3B 4B 6B 7B 8B".split())
    ranked_clear = score_void_candidates_v3(h_clear.counts)
    assert void_v3_sampling_probs(ranked_clear) == [1.0]
    rng = random.Random(7)
    for _ in range(10):
        assert sample_void_v3(h_clear, rng=rng) is Suit.WAN

    # 镜像平手 4/4/5 -> 探索两门且概率和为 1
    h_tie = Hand.from_strings("1W 4W 7W 9W 1T 4T 7T 9T 2B 3B 4B 6B 7B".split())
    ranked_tie = score_void_candidates_v3(h_tie.counts)
    probs_tie = void_v3_sampling_probs(ranked_tie)
    assert len(probs_tie) == 2
    assert probs_tie[0] == pytest.approx(0.5)
    assert sum(probs_tie) == pytest.approx(1.0)

    # 接近但不完全平手的撞车 4/4/5 -> #1 明显占优(>=65%), 同时保留探索 #2 的非零概率
    ranked_crash_tie = score_void_candidates_v3(
        h_tie.counts, is_dealer=False, swap_out_suit="W", swap_in_suit="W", swap_direction=2
    )
    probs_ct = void_v3_sampling_probs(ranked_crash_tie)
    assert len(probs_ct) == 2
    assert 0.65 <= probs_ct[0] <= 0.85
    assert sum(probs_ct) == pytest.approx(1.0)


def test_opening_surfaces_void_v3_recommendation_with_swap_context():
    own = "1W 4W 7W 9W 1T 4T 7T 9T 2B 3B 4B 6B 7B".split()
    swap = SwapInfo(given=("2W", "5W", "8W"), given_to=2, received=("1W", "4W", "9W"), received_from=2)
    state = make_state(own, phase="declare_void", swap=swap, swap_direction=2, dealer=1)
    info, findings = analyze_opening(state)
    assert info["void_v3_pick"] == "条"
    assert len(info["void_v3_options"]) == 3
    v3_findings = [f for f in findings if "void_v3" in f.title]
    assert v3_findings and v3_findings[0].tier == 3
    assert "撞车门" in "\n".join(v3_findings[0].basis)
