"""测试中后盘实战参谋(碰/杠建议、残局查叫研判)与高速向听/进张算法的 100% 数学等价性。"""
from __future__ import annotations

import random

from assistant.analyzer import analyze
from assistant.own_hand import fast_shanten_hand, fast_useful_indices
from assistant.state import MeldView
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import SUITS, Suit, Tile, full_wall, tile_to_str
from state.hand_analysis import shanten, useful_tiles
from assistant_fixtures import make_state, pong


def test_fast_shanten_and_useful_exact_match_across_random_hands():
    rng = random.Random(2026)
    wall = list(full_wall())
    for melds_cnt in (0, 1, 2, 3, 4):
        hand_size = 13 - 3 * melds_cnt
        for _ in range(120):
            rng.shuffle(wall)
            tiles = wall[:hand_size]
            h = Hand.from_tiles(tiles)
            for k in range(melds_cnt):
                mt = Tile(SUITS[k % 3], (k % 9) + 1)
                h.melds.append(Meld(MeldKind.PONG, (mt,) * 3, exposed=True))
            for void_idx, void_suit in ((None, None), (0, Suit.WAN), (1, Suit.TIAO), (2, Suit.BING)):
                sh_ref = shanten(h, void_suit)
                sh_fast = fast_shanten_hand(h.counts, melds_cnt, void_idx)
                assert sh_fast == sh_ref, (h.counts, melds_cnt, void_suit)
                u_ref = useful_tiles(h, void_suit)
                u_fast = [tile_to_str(Tile.from_index(i)) for i in fast_useful_indices(h.counts, melds_cnt, void_idx, sh_fast)]
                assert u_fast == u_ref, (h.counts, melds_cnt, void_suit, u_fast, u_ref)


def test_pong_advisor_warns_against_breaking_seven_pairs_and_recommends_real_speedup():
    # 1) 6 对子(七对听牌): 别人打 1T -> 碰会毁掉七对, 应提示"建议不碰"
    own_7p = "1T 1T 3T 3T 5T 5T 7T 7T 2B 2B 6B 6B 9B".split()
    st_7p = make_state(own_7p, voids={0: "W", 1: "W", 2: "T", 3: "B"}, pending_discard=(2, "1T"), rivers={2: ["1T"]})
    rep_7p = analyze(st_7p)
    pong_f = next(f for f in rep_7p.sections["alerts"] if f.title.startswith("碰1条"))
    assert "建议不碰" in pong_f.title
    assert "七对" in "\n".join(pong_f.basis)

    # 2) 普通 1 向听手牌碰完直接听牌 -> 应提示"建议碰"
    own_speed = "2T 3T 4T 5T 5T 7T 8T 9T 2B 3B 4B 6B 9B".split()
    st_sp = make_state(own_speed, voids={0: "W", 1: "W", 2: "T", 3: "B"}, pending_discard=(2, "5T"), rivers={2: ["5T"]})
    rep_sp = analyze(st_sp)
    pong_sp = next(f for f in rep_sp.sections["alerts"] if f.title.startswith("碰5条"))
    assert "建议碰" in pong_sp.title


def test_endgame_chajiao_advisor_distinguishes_ready_push_and_fold():
    # 构造残局(牌墙剩 10 张):
    # 1) 自己已听牌 -> 提示残局守听策略
    own_ready = "1T 2T 3T 4T 5T 6T 7T 8T 9T 2B 3B 4B 5B".split()
    rem = [tile_to_str(t) for t in full_wall()]
    for t in own_ready:
        rem.remove(t)
    # 留 10 张牌墙 + 3 家暗手 39 张 = 49 张未现 -> 牌河放 len(rem) - 49 张
    river_tiles = rem[: len(rem) - 49]
    r1, r2, r3 = river_tiles[:16], river_tiles[16:32], river_tiles[32:]
    st_ready = make_state(
        own_ready,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        rivers={1: r1, 2: r2, 3: r3},
        hidden={1: 13, 2: 13, 3: 13},
    )
    assert st_ready.wall_count == 10
    rep_ready = analyze(st_ready)
    titles_r = [f.title for f in rep_ready.sections["alerts"]]
    assert any("残局守听策略" in t for t in titles_r), titles_r

    # 2) 自己 2 向听 + 收过 6 分暗杠钱 + 仅剩 6 张牌墙 -> 提示防守止损并列明退税风险
    own_far = "1T 3T 6T 9T 1B 4B 7B 9B 2B 2B".split()
    kong_m = MeldView(kind="kong", tile="5T", concealed=True, income=6)
    rem2 = [tile_to_str(t) for t in full_wall()]
    for t in own_far + ["5T"] * 4:
        rem2.remove(t)
    rv2 = rem2[: len(rem2) - 45]  # 6 张牌墙 + 39 张暗牌
    st_far = make_state(
        own_far,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        melds={0: [kong_m]},
        rivers={1: rv2[:17], 2: rv2[17:34], 3: rv2[34:]},
        hidden={1: 13, 2: 13, 3: 13},
    )
    assert st_far.wall_count == 6
    rep_far = analyze(st_far)
    chajiao_f = next(f for f in rep_far.sections["alerts"] if "残局查叫研判" in f.title)
    assert "防守止损" in chajiao_f.title
    assert "退回" in "\n".join(chajiao_f.basis)


def test_kong_advisor_reclaims_midgame_kong_and_warns_late_far():
    from assistant.tactics import advise_kong
    from engine.tiles import parse_tile

    own = "1T 1T 1T 1T 3T 4T 5T 2B 3B 4B 6B 8B 9B 5B".split()
    rem = [tile_to_str(t) for t in full_wall()]
    for t in own:
        rem.remove(t)
    # 牌墙剩 15 张 (sh=1): 不应再保守判"慎杠", 应果断"建议暗杠"
    rv15 = rem[: len(rem) - 54]  # 15 墙 + 39 暗牌
    st15 = make_state(
        own,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        rivers={1: rv15[:18], 2: rv15[18:36], 3: rv15[36:]},
        hidden={1: 13, 2: 13, 3: 13},
    )
    assert st15.wall_count == 15
    h = Hand.from_tiles([parse_tile(t) for t in own])
    res_mid = advise_kong(st15, h, parse_tile("1T"), "concealed", sh_before=1, sh_after=1, opp_reads={})
    assert res_mid["verdict"] == "建议暗杠"

    # 牌墙剩 10 张且仍 2 向听: 才提示"慎杠"防查叫退税
    rv10 = rem[: len(rem) - 49]
    st10 = make_state(
        own,
        voids={0: "W", 1: "W", 2: "T", 3: "B"},
        rivers={1: rv10[:17], 2: rv10[17:34], 3: rv10[34:]},
        hidden={1: 13, 2: 13, 3: 13},
    )
    assert st10.wall_count == 10
    res_late = advise_kong(st10, h, parse_tile("1T"), "concealed", sh_before=2, sh_after=2, opp_reads={})
    assert res_late["verdict"] == "慎杠"

    # 中盘(牌墙>20且3家在场)1->2向听暗杠: 立收6分+1根+补牌, 应果断"建议暗杠"
    st_mid = make_state(own, voids={0: "W", 1: "W", 2: "T", 3: "B"})
    assert st_mid.wall_count >= 20
    res_1to2 = advise_kong(st_mid, h, parse_tile("1T"), "concealed", sh_before=1, sh_after=2, opp_reads={})
    assert res_1to2["verdict"] == "建议暗杠"


def test_v5_unified_ev_pong_exposes_ev_breakdown_and_avoids_degrading_good_tenpai():
    # 1) 已听两面好口(45T 听 3T/6T 共 8 张), 对手打 9T -> 碰 9T 会拆雀头退回 1 向听, 统一 EV 应算得 delta_ev < 0 并拒碰
    own = "1B 2B 3B 4B 5B 6B 7B 8B 9B 4T 5T 9T 9T".split()
    st = make_state(
        own,
        voids={0: "W", 1: "W", 2: "W", 3: "W"},
        pending_discard=(2, "9T"),
        rivers={2: ["9T"]},
        hidden={1: 13, 2: 13, 3: 13},
    )
    rep = analyze(st)
    assert not rep.errors
    pong_f = next(f for f in rep.sections["alerts"] if f.title.startswith("碰9条"))
    assert "EV" in "\n".join(pong_f.basis)
    assert "建议不碰" in pong_f.title

    # 2) 中盘碰完只剩死听口 -> 统一 EV 自动避开死听并给出"建议不碰"或"慎碰"
    own_dead = "1T 2T 3T 4T 5T 6T 5B 6B 7B 9B 9B 1B 8T".split()
    st_dead = make_state(
        own_dead,
        voids={0: "W", 1: "W", 2: "W", 3: "W"},
        pending_discard=(2, "9B"),
        rivers={1: ["1B", "1B", "1B"], 2: ["9B"], 3: ["8T", "8T", "8T"]},
        hidden={1: 13, 2: 13, 3: 13},
    )
    rep_dead = analyze(st_dead)
    pong_d = next(f for f in rep_dead.sections["alerts"] if f.title.startswith("碰9筒"))
    assert "建议不碰" in pong_d.title or "慎碰" in pong_d.title
