"""测试 discard_v2 攻防一体出牌引擎的牌理合理性、残局避炮与受控随机抽样。"""
from __future__ import annotations

import random

import pytest

from assistant.analyzer import analyze
from assistant.discard_v2 import discard_v2_sampling_probs, sample_discard_v2_index
from engine.tiles import full_wall, tile_to_str
from assistant_fixtures import make_state, pong


def test_discard_v2_always_clears_void_tiles_first():
    own = "9W 1T 2T 3T 4T 5T 6T 7T 8T 9T 2B 3B 4B 5B".split()
    st = make_state(own, voids={0: "W", 1: "T", 2: "B", 3: "W"})
    rep = analyze(st)
    assert rep.own_info["discard_v2_pick"] == "9万"


def test_discard_v2_prefers_high_fan_flush_wait_over_slightly_wider_small_wait():
    # 手牌 14 张(缺万): 条子 12 张 + 筒子 2 张 (打筒子听清一色大番 vs 拆条听混一色小番)
    # 11123456789T (11张) + 5T + 2B 8B -> 若只有 1 张 2B: 11123456789T + 5T + 2B(单张) -> 打 2B 听纯清一色多口!
    # 构造精准听牌对比: 234567899T + 22B + 1T + 8T -> 14张:
    # 简明 100% 明确案例: 11123456789T + 2B + 3B + 7T (14张, 打2B或3B退1向听? 构造两张都能听牌的局!)
    # 111234567T(9) + 8T 8T + 5T + 2B 2B -> 打 5T 听 8T/2B 对倒(平胡 0番, 4张) vs 打 2B 不听.
    # 更好: 111234567T(9) + 5T + 6T + 7T + 9T + 2B(14张):
    # 打 2B -> 111 234 567 567 9T (清一色单钓 9T, 2番+清一色 = 4分+, 剩3张)
    # 打 9T -> 111 234 567 567 2B (单钓 2B, 平胡 0番 = 1分, 剩3张)
    own = "1T 1T 1T 2T 3T 4T 5T 5T 6T 6T 7T 7T 9T 2B".split()
    st = make_state(own, voids={0: "W", 1: "W", 2: "W", 3: "W"})
    rep = analyze(st)
    assert not rep.errors
    assert rep.own_info["discard_v2_pick"] == "2筒"
    ev_opts = {c["tile"]: c for c in rep.own_info["discard_v2_options"]}
    assert ev_opts["2筒"]["ev_score"] > ev_opts["9条"]["ev_score"]


def test_discard_v2_avoids_high_danger_flush_tile_in_endgame():
    # 残局(牌墙剩 10 张), 自己 2 向听几乎无望上听; 下家强清一色条且快听, 刚打过 5筒(近3巡现物)
    # 我方打 5条(命中清一色条高危) vs 打 5筒(近3巡现物极安全) 向听相同 -> discard_v2 必须选打 5筒避炮!
    own = "1T 4T 5T 8T 2B 5B 8B 1B 2B 4B 6B 7B 9B 3T".split()
    rem = [tile_to_str(t) for t in full_wall()]
    for t in own:
        rem.remove(t)
    # 下家 2 副露(2T, 8T) -> 占 6 张条子, 剩 7 张暗牌
    for t in ["2T"] * 3 + ["8T"] * 3 + ["1W", "9W", "1B", "3B", "5B"]:
        if t in rem:
            rem.remove(t)
    # 留 10 张牌墙 + (7 + 13 + 13) = 43 张未现
    extra_river = rem[: len(rem) - 43]
    r2, r3 = extra_river[: len(extra_river) // 2], extra_river[len(extra_river) // 2 :]
    st = make_state(
        own,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        melds={1: [pong("2T"), pong("8T")]},
        rivers={1: ["1W", "9W", "1B", "3B", "5B"], 2: r2, 3: r3},
        hidden={1: 7, 2: 13, 3: 13},
    )
    assert st.wall_count == 10
    rep = analyze(st)
    assert not rep.errors
    ev_opts = {c["tile"]: c for c in rep.own_info["discard_v2_options"]}
    assert ev_opts["5筒"]["ev_score"] > ev_opts["5条"]["ev_score"]
    assert ev_opts["5筒"]["ev_defense"] > ev_opts["5条"]["ev_defense"]


def test_discard_v2_sampling_locks_clear_choice_and_explores_close_ties():
    # 清晰清缺门(9W 唯一缺门牌) -> 100% 锁定 #1
    own_clear = "9W 1T 2T 3T 4T 5T 6T 7T 8T 9T 2B 3B 4B 5B".split()
    rep_c = analyze(make_state(own_clear, voids={0: "W", 1: "T", 2: "B", 3: "W"}))
    probs_c = discard_v2_sampling_probs(rep_c.own_info["discard_v2_options"])
    assert probs_c == [1.0]

    # 镜像平手(1T 与 9T 对称孤张, 其余成牌) -> 按 Softmax 探索多个相近解
    own_tie = "1T 9T 3T 4T 5T 2B 3B 4B 5B 6B 7B 8B 8B 8B".split()
    rep_t = analyze(make_state(own_tie, voids={0: "W", 1: "W", 2: "W", 3: "W"}))
    probs_t = discard_v2_sampling_probs(rep_t.own_info["discard_v2_options"])
    assert len(probs_t) >= 2
    assert sum(probs_t) == pytest.approx(1.0)
    rng = random.Random(42)
    picks = {sample_discard_v2_index(rep_t.own_info["discard_v2_options"], rng=rng) for _ in range(30)}
    assert len(picks) >= 2


def test_discard_v2_1shanten_good_wait_and_pong_speedup():
    from assistant.discard_v2 import _one_shanten_good_wait_quality, _pong_speedup_outs
    from assistant.own_hand import fast_shanten_hand, fast_useful_indices
    from engine.hand import Hand
    from engine.tiles import parse_tile

    # 1) 1向听两步好型率: 含两面 45T+45B 的听牌宽度期望显著高于含边嵌 12T+13B 的同向听手牌
    h_good = Hand.from_tiles([parse_tile(t) for t in "7T 8T 9T 7B 8B 9B 2B 2B 4T 5T 4B 5B 1T".split()])
    h_bad = Hand.from_tiles([parse_tile(t) for t in "7T 8T 9T 7B 8B 9B 2B 2B 1T 2T 1B 3B 5T".split()])
    unseen_full = tuple(4 - c for c in h_good.counts)
    u_good = [{"index": i, "remaining": unseen_full[i]} for i in fast_useful_indices(h_good.counts, 0, 0, 1)]
    unseen_bad = tuple(4 - c for c in h_bad.counts)
    u_bad = [{"index": i, "remaining": unseen_bad[i]} for i in fast_useful_indices(h_bad.counts, 0, 0, 1)]
    assert fast_shanten_hand(h_good.counts, 0, 0) == 1
    assert fast_shanten_hand(h_bad.counts, 0, 0) == 1
    q_good = _one_shanten_good_wait_quality(list(h_good.counts), 0, 0, u_good, unseen_full)
    q_bad = _one_shanten_good_wait_quality(list(h_bad.counts), 0, 0, u_bad, unseen_bad)
    assert q_good > q_bad + 0.3

    # 2) 三家碰牌提速进张: 两对子(2B2B + 5B5B)1向听手牌, 碰任意一对均可立刻听牌 -> pong_speedup_outs == 4
    h_pair = Hand.from_tiles([parse_tile(t) for t in "7T 8T 9T 7B 8B 9B 2B 2B 5B 5B 4T 5T 1T".split()])
    unseen_p = tuple(4 - c for c in h_pair.counts)
    outs = _pong_speedup_outs(list(h_pair.counts), 0, 0, 1, unseen_p)
    assert outs == 4


def test_discard_v2_prefers_waiting_on_opponents_void_suit_and_terminal():
    # 同 0 番单钓(各剩 3 张): 两家对手缺条(T), 无人缺筒(B) -> 打 5筒 听 9条(双家缺门+幺九) 应显著优于打 9条 听 5筒
    own = "1T 2T 3T 4T 5T 6T 1B 2B 3B 7B 8B 9B 9T 5B".split()
    st = make_state(own, voids={0: "W", 1: "T", 2: "T", 3: "W"})
    rep = analyze(st)
    assert not rep.errors
    ev_opts = {c["tile"]: c for c in rep.own_info["discard_v2_options"]}
    assert ev_opts["5筒"]["ev_score"] > ev_opts["9条"]["ev_score"]
    assert rep.own_info["discard_v2_pick"] == "5筒"
