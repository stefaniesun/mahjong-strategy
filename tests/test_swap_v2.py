"""测试 swap_v2 校准特性与 assistant.opening 接入。"""
from __future__ import annotations

import pytest

from assistant.adapters.from_engine import build_state
from assistant.analyzer import analyze
from assistant.opening import analyze_opening
from assistant.recorder import PublicRecorder
from assistant.swap_v2 import (
    choose_swap_v2,
    empirical_incoming_suit_weights,
    sample_swap_v2,
    score_swap_candidates_v2,
    swap_v2_sampling_probs,
)
from engine.game import Game
from engine.hand import Hand
from engine.tiles import tile_to_str
from assistant_fixtures import make_state


def test_empirical_reverse_siphon_weights_increase_with_own_suit_count():
    # 1/5/7 牌型下, 持有 7 张的那门进牌概率应显著高于持有 1 张的那门
    w = empirical_incoming_suit_weights((1, 5, 7))
    assert w[0] < w[1] < w[2]
    assert round(w[0], 3) == 0.296
    assert round(w[2], 3) == 0.360


def test_v2_fixes_v1_overdefense_blind_spots():
    # 1) 3/4/6 双顺子: 234568W + 2456T(含顺子) + 234B(3张顺子) -> 应直接清零 234B, 不再拆 2456T 留废牌
    h1 = Hand.from_strings("2W 3W 4W 5W 6W 8W 2T 4T 5T 6T 2B 3B 4B".split())
    assert [tile_to_str(t) for t in choose_swap_v2(h1)] == ["2B", "3B", "4B"]

    # 2) 4/4/5 保顺子允许送边幺对子: 13459W + 1225T + 1899B -> 不再拆 345W 顺子凑 149W, 而是换 199B(或 189B)
    h2 = Hand.from_strings("1W 3W 4W 5W 9W 1T 2T 2T 5T 1B 8B 9B 9B".split())
    c2 = [tile_to_str(t) for t in choose_swap_v2(h2)]
    assert all(x.endswith("B") for x in c2)

    # 3) 庄家 4/5/5 首打清 1 张缺门红利: 2368W(4) + 26899T(5) + 56799B(5) -> 应拆 4 张万子留 1 张缺门
    h3 = Hand.from_strings("2W 3W 6W 8W 2T 6T 8T 9T 9T 5B 6B 7B 9B 9B".split())
    c3 = [tile_to_str(t) for t in choose_swap_v2(h3, is_dealer=True)]
    assert all(x.endswith("W") for x in c3)

    # 4) 经典 3/4/6 (3张顺子 vs 4张纯散牌):
    # (a) 已有将牌 55B 时: 保 456W 顺子, 换 149T 留中张 7T 靠搭子
    h4_paired = Hand.from_strings("4W 5W 6W 1T 4T 7T 9T 2B 3B 4B 5B 5B 8B".split())
    assert [tile_to_str(t) for t in choose_swap_v2(h4_paired)] == ["1T", "4T", "9T"]
    # (b) 三副顺子无将牌时: 同样坚定保 456W 顺子、从条子换出三不靠散牌
    h4_nopair = Hand.from_strings("4W 5W 6W 1T 4T 7T 9T 2B 3B 4B 6B 7B 8B".split())
    c4 = [tile_to_str(t) for t in choose_swap_v2(h4_nopair)]
    assert all(x.endswith("T") for x in c4)

    # 5) 无将手牌保留真对子 (Seed #5002): 124799W + 46T + 12357B -> 换万子时不拆 99W 对子
    h5 = Hand.from_strings("1W 2W 4W 7W 9W 9W 4T 6T 1B 2B 3B 5B 7B".split())
    c5 = [tile_to_str(t) for t in choose_swap_v2(h5)]
    assert all(x.endswith("W") for x in c5) and c5.count("9W") == 0

    # 6) 1/5/7 中 5 张门是双对+两面强牌 (Seed #5003): 6W + 1445689T + 11667B -> 不强拆强5张, 出 189T
    h6 = Hand.from_strings("6W 1T 4T 4T 5T 6T 8T 9T 1B 1B 6B 6B 7B".split())
    c6 = [tile_to_str(t) for t in choose_swap_v2(h6, swap_direction=-1)]
    assert c6 == ["1T", "8T", "9T"]

    # 7) 2/4/7 短门带暗刻 (Seed #5065): 29W + 1255789T + 7999B -> 保住 999B 暗刻留杠机会, 从条子换出 3 张
    h7 = Hand.from_strings("2W 9W 1T 2T 5T 5T 7T 8T 9T 7B 9B 9B 9B".split())
    c7 = [tile_to_str(t) for t in choose_swap_v2(h7, swap_direction=2)]
    assert all(x.endswith("T") for x in c7)

    # 8) 3/4/6 3张门就是暗刻 (Seed #7058): 777B + 1447T + 113358W -> 留 777B 暗刻, 改拆 4 张条子留单张
    h8 = Hand.from_strings("7B 7B 7B 1T 4T 4T 7T 1W 1W 3W 3W 5W 8W".split())
    c8 = [tile_to_str(t) for t in choose_swap_v2(h8)]
    assert all(x.endswith("T") for x in c8)

    # 9) 2/5/6 中 5 张门带双对 (Seed #8018): 122347W + 46T + 11889B -> 保留 11889B 双对子, 从 6 张万子换出 147W
    h9 = Hand.from_strings("1W 2W 2W 3W 4W 7W 4T 6T 1B 1B 8B 8B 9B".split())
    c9 = [tile_to_str(t) for t in choose_swap_v2(h9, swap_direction=1)]
    assert c9 == ["1W", "4W", "7W"]

    # 10) 1/5/7 中 5 张门带双对 (Seed #8111): 5W + 22399B + 1467778T -> 不拆 22399B 双对, 从 7 张条子换出 148T
    h10 = Hand.from_strings("5W 2B 2B 3B 9B 9B 1T 4T 6T 7T 7T 7T 8T".split())
    c10 = [tile_to_str(t) for t in choose_swap_v2(h10, swap_direction=-1)]
    assert c10 == ["1T", "4T", "8T"]

    # 11) 2/4/7 中 4 张门选留单张 (Seed #8039): 2456889W + 12T + 2467B -> 出更分散的 247B 留中张 6B, 不喂 246 双嵌边搭
    h11 = Hand.from_strings("2W 4W 5W 6W 8W 8W 9W 1T 2T 2B 4B 6B 7B".split())
    c11 = [tile_to_str(t) for t in choose_swap_v2(h11, swap_direction=2)]
    assert c11 == ["2B", "4B", "7B"]

    # 12) 2/5/6 5张门含单对+两面 (Seed #10007): 56679W + 124699T + 79B -> 换 579W 保 66W 对子凑双对碰杠潜能
    h12 = Hand.from_strings("5W 6W 6W 7W 9W 1T 2T 4T 6T 9T 9T 7B 9B".split())
    c12 = [tile_to_str(t) for t in choose_swap_v2(h12, swap_direction=2)]
    assert c12 == ["5W", "7W", "9W"]


def test_opening_panel_surfaces_v2_recommendation_and_flush_conflict():
    # 1/5/7 中 5 张门是 1顺1将好牌: 9W + 34577T + 1246789B -> 存在“求稳保两门 vs 拆条冲筒清一色”路线权衡
    own = "9W 3T 4T 5T 7T 7T 1B 2B 4B 6B 7B 8B 9B".split()
    state = make_state(own, phase="swap_three")
    info, findings = analyze_opening(state)
    assert info["swap_options"]
    # 存在第 3 档期望结论且写明假设
    tier3 = [f for f in findings if f.tier == 3]
    assert tier3 and all(f.assumption for f in tier3)
    # 无论综合第一是求稳还是冲清一色, 若综合第一不是清一色则应触发 flush 路线提示
    if info["swap_options"][0]["exp_flush_rate"] < 0.15:
        assert info["swap_conflict"] is not None
        assert info["swap_conflict"]["type"] == "flush"


def test_engine_adapter_passes_swap_direction_and_dealer_to_opening():
    game = Game(seed=42, dealer=1)
    game.reset()
    rec = PublicRecorder(game)
    st = build_state(game.state, rec, perspective=0)
    assert st.swap_direction == game.state.swap_direction
    assert st.dealer == 1
    rep = analyze(st).to_dict()
    assert rep["opening_info"]["swap_target"] in ("下家", "对家", "上家")
    assert rep["elapsed_ms"] < 200


def test_sample_swap_v2_respects_gap_cutoff_and_prefers_top_choice():
    import random

    # 1) 分差过大(Seed #7058: 147T 远甩 #2 拆刻子 >= 0.20) -> 严格 100% 锁定 #1, 绝不误选劣解
    h_clear = Hand.from_strings("7B 7B 7B 1T 4T 4T 7T 1W 1W 3W 3W 5W 8W".split())
    ranked_clear = score_swap_candidates_v2(h_clear.counts)
    probs_clear = swap_v2_sampling_probs(ranked_clear)
    assert probs_clear == [1.0]
    rng = random.Random(123)
    for _ in range(10):
        assert [tile_to_str(t) for t in sample_swap_v2(h_clear, rng=rng)] == ranked_clear[0]["tiles"]

    # 2) 评分相近(Seed #8039: 247B vs 246B 等仅差 ~0.01~0.08) -> 按权重探索 1~3 名且 #1 占优
    h_close = Hand.from_strings("2W 4W 5W 6W 8W 8W 9W 1T 2T 2B 4B 6B 7B".split())
    ranked_close = score_swap_candidates_v2(h_close.counts, swap_direction=2)
    probs_close = swap_v2_sampling_probs(ranked_close)
    assert 2 <= len(probs_close) <= 3
    assert probs_close[0] >= probs_close[1]
    assert sum(probs_close) == pytest.approx(1.0)

    # 3) 大样本下第 1 选择期望占比保持在 75% 以上, 同时约 15%~25% 探索第 2/3 选择
    p1_sum = 0.0
    p2_sum = 0.0
    p3_sum = 0.0
    n_hands = 60
    for seed in range(1000, 1000 + n_hands):
        g = Game(seed=seed, dealer=seed % 4)
        st = g.reset()
        r = score_swap_candidates_v2(st.hands[0].counts, is_dealer=(seed % 4 == 0), swap_direction=st.swap_direction)
        ps = swap_v2_sampling_probs(r)
        p1_sum += ps[0]
        if len(ps) >= 2:
            p2_sum += ps[1]
        if len(ps) >= 3:
            p3_sum += ps[2]
    assert p1_sum / n_hands >= 0.75
    assert (p2_sum + p3_sum) / n_hands >= 0.10
