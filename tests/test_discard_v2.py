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


def test_v5_fast_tenpai_fan_mult_distinguishes_big_hands_from_pinghu():
    from assistant.discard_v2 import _fast_tenpai_fan_mult
    from engine.hand import Hand
    from engine.tiles import parse_tile

    # 清一色听牌 (2番 -> 4.0x)
    h_flush = Hand.from_tiles([parse_tile(t) for t in "1T 2T 3T 4T 5T 6T 7T 8T 9T 2T 3T 5T 5T".split()])
    assert _fast_tenpai_fan_mult(list(h_flush.counts), h_flush.melds, [parse_tile("1T").index, parse_tile("4T").index]) >= 4.0

    # 七对听牌 (2番 -> 4.0x)
    h_7p = Hand.from_tiles([parse_tile(t) for t in "1T 1T 3T 3T 5T 5T 7T 7T 2B 2B 6B 6B 9B".split()])
    assert _fast_tenpai_fan_mult(list(h_7p.counts), h_7p.melds, [parse_tile("9B").index]) >= 4.0

    # 普通 0 番平胡 (1.0x)
    h_plain = Hand.from_tiles([parse_tile(t) for t in "1T 2T 3T 4T 5T 6T 2B 3B 4B 6B 7B 9B 9B".split()])
    assert _fast_tenpai_fan_mult(list(h_plain.counts), h_plain.melds, [parse_tile("5B").index, parse_tile("8B").index]) == 1.0


def test_discard_v2_void_clear_order_prefers_unseen_mid_and_pair_over_edge_and_seen():
    # 1) 全是生张单张 (1B/5B/9B): 低危早巡应先打 5筒(中张), 把 1筒/9筒 留后
    own1 = "1W 7W 7W 8W 8W 9W 2T 2T 6T 6T 7T 1B 5B 9B".split()
    st1 = make_state(own1, voids={0: "B", 1: "T", 2: "B", 3: "T"})
    rep1 = analyze(st1)
    assert rep1.own_info["discard_v2_pick"] == "5筒"

    # 2) 若 5筒 牌池已现 1 张而 9筒 仍是生张: 低危早巡应先清生张 9筒, 把熟张 5筒 留后兜底
    st2 = make_state(own1, voids={0: "B", 1: "T", 2: "B", 3: "T"}, rivers={2: ["5B"]})
    rep2 = analyze(st2)
    assert rep2.own_info["discard_v2_pick"] in ("1筒", "9筒")

    # 3) 自己握生张对子 5B 5B 与生张单张 5B/4B: 对子 5B 优先先出1张
    own3 = "1W 7W 7W 8W 8W 9W 2T 2T 6T 6T 7T 4B 5B 5B".split()
    st3 = make_state(own3, voids={0: "B", 1: "T", 2: "B", 3: "T"})
    rep3 = analyze(st3)
    assert rep3.own_info["discard_v2_pick"] == "5筒"


def test_discard_v2_prefers_safe_4out_pair_wait_with_5B_upgrade_over_3out_kanchan():
    # 实盘截图复现: [888B] + 345678W + 1B 3B 3B 4B 4B (余38张, 缺条, 对家缺筒)
    # 打1筒(极安全0.6%, 听3/4筒4张, 摸5筒打4筒变3345筒听3/6筒5~6张两面) 应严格优于 打3筒(嵌张2筒3张, 6.9%风险),
    # 即使6筒已现1张(摸5筒4->5仅+1张改良)也不能因+2门槛断崖被嵌张反超!
    from assistant.state import AssistantState, MeldView, PlayerView, WinRecord

    for s2_river in (("1B", "5B", "7B", "6B", "9W"), ("1B", "5B", "7B", "8T", "9W")):
        players = (
            PlayerView(0, (MeldView("pong", "8B"),), ("7T", "7T", "3W", "7B"), ("7T", "7T", "3W", "7B"), "T", False, 0),
            PlayerView(1, (), ("4T", "6T", "5W", "7W"), ("4T", "6T", "5W", "7W"), "B", True, 13),
            PlayerView(2, (MeldView("pong", "1W"), MeldView("pong", "9T")), s2_river, s2_river, "B", False, 7),
            PlayerView(3, (), ("1B", "5T", "7B", "2B"), ("1B", "5T", "7B", "2B"), "T", False, 13),
        )
        st = AssistantState(
            perspective=0,
            phase="play",
            own_hand=("3W", "4W", "5W", "6W", "7W", "8W", "1B", "3B", "3B", "4B", "4B"),
            players=players,
            wall_count=38,
            wins=(WinRecord(1, "6T", False, 2),),
        )
        rep = analyze(st)
        assert not rep.errors
        assert rep.own_info["discard_v2_pick"] == "1筒"


def test_discard_v2_wall_expected_live_and_p_next_reflect_two_opp_void_suit():
    # 实盘截图复现: 自己缺条(T), 9张筒子(123 345 6 89B) + 5张万子(1W 34W 79W), 余30张, 对家与上家均缺筒(B)
    # 牌墙40%为筒子(未现筒子在墙占比71% vs 万子34%), 打1万的真实墙内进张、下摸率与进攻分均显著高于打9筒!
    from assistant.state import AssistantState, MeldView, PlayerView, WinRecord

    players = (
        PlayerView(0, (), ("6T", "3T", "9T", "9T", "8T", "8T"), ("6T", "3T", "9T", "9T", "8T", "8T"), "T", False, 0),
        PlayerView(1, (), ("1B", "1T", "7T", "1W", "4T"), ("1B", "1T", "7T", "1W", "4T"), "T", True, 13),
        PlayerView(2, (), ("8B", "7B", "8B", "1B", "7B", "4T", "7T"), ("8B", "7B", "8B", "1B", "7B", "4T", "5T", "7T"), "B", False, 13),
        PlayerView(3, (MeldView("pong", "5T"),), ("2B", "8T", "1B", "6B", "7T", "5B", "4T"), ("2B", "8T", "1B", "6B", "7T", "5B", "4T"), "B", False, 10),
    )
    st = AssistantState(
        perspective=0,
        phase="play",
        own_hand=("1W", "3W", "4W", "7W", "9W", "1B", "2B", "3B", "3B", "4B", "5B", "6B", "8B", "9B"),
        players=players,
        wall_count=30,
        wins=(WinRecord(1, "7B", False, 2),),
    )
    rep = analyze(st)
    assert not rep.errors
    ev_opts = {c["tile"]: c for c in rep.own_info["discard_v2_options"]}
    # 打1万的真实牌墙进张与下摸率严格高于打9筒(两家缺筒导致筒子墙内密度约为万子2.1倍)
    assert ev_opts["1万"]["wall_expected_live"] > ev_opts["9筒"]["wall_expected_live"] + 1.0
    assert ev_opts["1万"]["p_next"] > ev_opts["9筒"]["p_next"] + 0.04
    assert ev_opts["1万"]["ev_offense"] > ev_opts["9筒"]["ev_offense"]


def test_discard_v2_breaks_weak_0fan_2out_tenpai_for_12_tile_pure_flush():
    # 实盘截图复现: [碰5T] + 11W + 2T 444T 77899T (12张条子清一色胚 + 11W, 余30张, 上家缺条)
    # 打1万(1向听冲12张清一色条, 250副模拟均分+3.98) 应严格优于 打2条(0番仅2张8条平胡, 均分+3.47)!
    from assistant.state import AssistantState, MeldView, PlayerView, WinRecord

    players = (
        PlayerView(0, (MeldView("pong", "5T"),), ("8B", "9B", "2B", "7W", "9W", "2W"), ("8B", "9B", "2B", "7W", "9W", "2W"), "B", False, 0),
        PlayerView(1, (MeldView("pong", "6B"), MeldView("pong", "4W")), ("3T", "3T", "9B", "5W"), ("3T", "3T", "9B", "5W"), "T", True, 7),
        PlayerView(2, (MeldView("pong", "6T"), MeldView("pong", "5B")), ("1T", "8T", "8W", "1T", "5W", "3T", "5W"), ("1T", "8T", "8W", "1T", "5W", "3T", "5W"), "W", False, 7),
        PlayerView(3, (MeldView("pong", "9W"),), ("1T", "3T", "9B", "9T", "2T", "8B", "7T"), ("1T", "3T", "9B", "9T", "2T", "8B", "7T"), "T", False, 10),
    )
    st = AssistantState(
        perspective=0,
        phase="play",
        own_hand=("1W", "1W", "2T", "4T", "4T", "4T", "7T", "7T", "8T", "9T", "9T"),
        players=players,
        wall_count=30,
        wins=(WinRecord(1, "1B", True, None),),
    )
    rep = analyze(st)
    assert not rep.errors
    assert rep.own_info["discard_v2_pick"] == "1万"
    ev_opts = {c["tile"]: c for c in rep.own_info["discard_v2_options"]}
    assert ev_opts["1万"]["ev_score"] > ev_opts["2条"]["ev_score"]
