"""测试中后盘实战参谋(碰/杠建议、残局查叫研判)与高速向听/进张算法的 100% 数学等价性。"""
from __future__ import annotations

import random

from assistant.analyzer import analyze
from assistant.own_hand import fast_shanten_hand, fast_useful_indices
from assistant.state import MeldView
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import SUITS, Suit, Tile, full_wall, tile_to_str
from engine.ting_check import ting_tiles
from engine.win_check import can_win
from state.hand_analysis import _effective_counts, _standard_shanten
from assistant_fixtures import make_state, pong


def _sichuan_shanten_ref(hand: Hand, void_suit: Suit | None) -> int:
    counts = tuple(_effective_counts(hand, void_suit))
    std = _standard_shanten(counts, len(hand.melds))
    if hand.melds:
        return std
    sp_pairs = sum(c // 2 for c in counts)
    sp_slots = sum((c + 1) // 2 for c in counts)
    return min(std, 6 - sp_pairs + max(0, 7 - sp_slots))


def _sichuan_useful_ref(hand: Hand, void_suit: Suit | None) -> list[str]:
    cur = _sichuan_shanten_ref(hand, void_suit)
    out = []
    for s in SUITS:
        if void_suit is not None and s is void_suit:
            continue
        for r in range(1, 10):
            t = Tile(s, r)
            if hand.count(t) >= 4:
                continue
            tr = Hand(counts=list(hand.counts), melds=list(hand.melds))
            tr.add(t)
            if _sichuan_shanten_ref(tr, void_suit) < cur:
                out.append(tile_to_str(t))
    return out


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
                sh_ref = _sichuan_shanten_ref(h, void_suit)
                sh_fast = fast_shanten_hand(h.counts, melds_cnt, void_idx)
                assert sh_fast == sh_ref, (h.counts, melds_cnt, void_suit)
                u_ref = _sichuan_useful_ref(h, void_suit)
                u_fast = [tile_to_str(Tile.from_index(i)) for i in fast_useful_indices(h.counts, melds_cnt, void_idx, sh_fast)]
                assert u_fast == u_ref, (h.counts, melds_cnt, void_suit, u_fast, u_ref)
                # 0 向听时必须与 engine.ting_check.ting_tiles 100% 一致
                if sh_fast == 0 and (void_suit is None or all(t.suit is not void_suit for t in h.tiles())):
                    engine_waits = [tile_to_str(t) for t in sorted(ting_tiles(h, void_suit), key=lambda x: x.index)]
                    assert u_fast == engine_waits, (h.counts, void_suit, u_fast, engine_waits)


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

    # 3) 4 对子(七对 2 向听 & 正常 3 向听, 实测碰牌净赢 +1.58 分): 应识别标准牌型提速并果断建议碰!
    own_4p = "3W 6W 7W 9W 9W 1B 1B 4B 4B 6B 8B 9B 9B".split()
    st_4p = make_state(
        own_4p,
        voids={0: "T", 1: "B", 2: "W", 3: "T"},
        pending_discard=(1, "9W"),
        rivers={1: ["9W"]},
        hidden={1: 13, 2: 13, 3: 13},
    )
    rep_4p = analyze(st_4p)
    pong_4p = next(f for f in rep_4p.sections["alerts"] if f.title.startswith("碰9万"))
    assert "建议碰" in pong_4p.title

    # 4) 听牌换听改良潜力(截图原手牌: [111T] + 333T 89T 456B 77B, 碰 7B 后应选打 9T/8T 留暗刻3T+活单钓改良, 绝不拆 3T 守死边张!)
    own_upg = "3T 3T 3T 8T 9T 4B 5B 6B 7B 7B".split()
    st_upg = make_state(
        own_upg,
        voids={0: "W", 1: "W", 2: "B", 3: "T"},
        melds={0: [pong("1T")]},
        pending_discard=(2, "7B"),
        rivers={2: ["7B", "6B", "5B"], 3: ["7T", "7T"]},  # 7条已现2张只剩2张 -> 更应果断碰7筒打8/9条换活听!
        hidden={1: 13, 2: 13, 3: 13},
    )
    rep_upg = analyze(st_upg)
    pong_u = next(f for f in rep_upg.sections["alerts"] if f.title.startswith("碰7筒"))
    assert "打3条" not in pong_u.title
    assert "打9条" in pong_u.title or "打8条" in pong_u.title
    assert "可碰" in pong_u.title or "建议碰" in pong_u.title


def test_long_seven_pairs_shanten_and_endgame_fan_consistency():
    # 1) 纯龙七对听牌(含暗刻待第4张 / 已握4张根待单钓)必须精确识别为 0 向听, 并与 engine.ting_tiles / can_win 100% 一致
    h_l7_tri = Hand.from_strings("2W 2W 4W 4W 6W 6W 8W 8W 2T 2T 9T 9T 9T".split())
    assert fast_shanten_hand(h_l7_tri.counts, 0, 2) == 0
    u_tri = [tile_to_str(Tile.from_index(i)) for i in fast_useful_indices(h_l7_tri.counts, 0, 2, 0)]
    assert u_tri == ["9T"]
    h_l7_tri_win = Hand.from_strings("2W 2W 4W 4W 6W 6W 8W 8W 2T 2T 9T 9T 9T 9T".split())
    assert can_win(h_l7_tri_win, Suit.BING)
    assert fast_shanten_hand(h_l7_tri_win.counts, 0, 2) == -1

    h_l7_quad = Hand.from_strings("2W 2W 4W 4W 6W 6W 2T 2T 5T 9T 9T 9T 9T".split())
    assert fast_shanten_hand(h_l7_quad.counts, 0, 2) == 0
    u_quad = [tile_to_str(Tile.from_index(i)) for i in fast_useful_indices(h_l7_quad.counts, 0, 2, 0)]
    assert u_quad == ["5T"]

    # 2) 实盘截图复现: 22 33 44 66W 22 3 999T (缺筒, 余15张):
    #    - 打 3条 的有效进张必须包含 9条(3番龙七对), 共 3 种 4 张(6W×2, 2T×1, 9T×1), 不能漏成 2 种 3 张;
    #    - 残局守听策略的最高番数必须与综合推荐弃牌(2番/3番)一致, 绝不能矛盾显示"最高 0 番"!
    own_14 = "2W 2W 3W 3W 4W 4W 6W 6W 2T 2T 3T 9T 9T 9T".split()
    st = make_state(
        own_14,
        voids={0: "B", 1: "W", 2: "B", 3: "T"},
        rivers={
            0: ["4B", "1B", "5B", "9W", "5T", "8B", "6B", "8T", "5T", "7B"],
            1: ["1T", "1B", "7B", "5B", "2W", "1W", "2W"],
            2: ["2B", "3B", "9B", "6B", "1W", "3W", "8B", "5B", "8T", "7T", "5T"],
            3: ["2T", "6T", "1W", "9B", "8B", "1W", "1T", "6B", "7W", "4W", "1B", "7T"],
        },
        melds={2: [pong("5W")], 3: [pong("9W")]},
        won={1: True},
        hidden={1: 13, 2: 10, 3: 10},
    )
    rep = analyze(st)
    assert not rep.errors
    c_3t = next(c for c in rep.own_info["discard_v2_options"] if c["tile"] == "3条")
    assert c_3t["useful_kinds"] == 3
    assert c_3t["live"] == 4
    assert {u["tile"] for u in c_3t["useful"]} == {"6万", "2条", "9条"}
    endgame_alerts = [f for f in rep.sections["alerts"] if f.title.startswith("残局守听策略")]
    assert endgame_alerts
    assert "最高 0 番" not in endgame_alerts[0].title
    top_ev_fan = rep.own_info["discard_v2_options"][0]["max_fan"]
    assert f"最高 {top_ev_fan} 番" in endgame_alerts[0].title


def test_void_aware_wall_expected_and_suit_inference():
    # 实盘场景: 2家缺筒(下家已胡 + 上家缺筒且已出多张非筒牌), 对家自摸3筒(缺万, 碰2筒)
    # 剩余未现筒子应大幅向牌墙集中(而非按旧均匀公式被错分一半给缺筒的上家), 且 suit_inference 产出完整归属与已胡线索
    from assistant.state import WinRecord

    own_14 = "7W 8W 9W 1B 3B 4B 5B 6B 7B 7B 8B 8B 8B 9B".split()
    st = make_state(
        own_14,
        voids={0: "T", 1: "B", 2: "W", 3: "B"},
        rivers={
            0: ["1T", "2T", "3T", "4T", "6T", "7T", "8T", "9T", "1W", "7W", "4B", "6B", "9B", "9B"],
            1: ["1B", "4B", "8T", "9T", "1T", "2T", "3T"],
            2: ["1W", "2W", "3W", "3W", "6W", "1B", "2T", "9T"],
            3: ["1B", "9B", "1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "2T", "3T"],
        },
        melds={2: [pong("2B")]},
        won={1: True, 2: True},
        wins=(
            WinRecord(seat=1, tile="3W", self_draw=False, from_seat=2),
            WinRecord(seat=2, tile="3B", self_draw=True, from_seat=None),
        ),
        hidden={1: 13, 2: 10, 3: 13},
    )
    rep = analyze(st)
    assert not rep.errors
    assert len(rep.suit_inference) == 3
    bing = next(s for s in rep.suit_inference if s["suit"] == "B")
    assert bing["opp_void_count"] == 2
    assert bing["active_opp_expected"] < 0.8  # 上家缺筒已清门, 极少可能再握筒子
    # 张数守恒: 三门 wall_expected 之和 == state.wall_count
    total_wall_est = sum(s["wall_expected"] for s in rep.suit_inference)
    assert abs(total_wall_est - st.wall_count) < 0.2
    # 每张筒子未现 = 墙 + 已胡扣 + 在场对手
    for t in bing["tiles"]:
        assert abs(t["wall_expected"] + t["winner_expected"] + t["active_opp_expected"] - t["unseen"]) < 0.1
    # 5筒牌面未现3张, 扣除对家后剩余部分几乎全在墙内(明显高于旧均匀比例 wall/(wall+13))
    t5 = next(t for t in bing["tiles"] if t["rank"] == 5)
    assert t5["unseen"] == 3
    assert t5["wall_expected"] > 1.35


def test_exhausted_win_tile_deduces_certain_sequence_even_on_too_many():
    # 断张1筒胡牌: 外面3张1筒全现身 + 对家点炮第4张1筒给上家 -> 即使组合数超过 enum_limit(STATUS_TOO_MANY),
    # 仍必须 100% 确定档反推出上家手里必有 2筒 和 3筒, 并同步更新全局 effective_unseen / certain_locked!
    from assistant.counting import compute_counts
    from assistant.state import WinRecord
    from assistant.winner_inference import STATUS_TOO_MANY, infer_winners

    own = "1B 2T 3T 4T 5T 6T 7T 8T 8T 8T 4B 5B 8B 8B".split()
    st = make_state(
        own,
        voids={0: "W", 1: "B", 2: "B", 3: "T"},
        rivers={0: ["7W", "2W"], 1: ["1B", "5B", "1B"], 2: ["1B", "9B", "2B", "2B", "2B"], 3: ["7T", "3T"]},
        won={3: True},
        wins=(WinRecord(seat=3, tile="1B", self_draw=False, from_seat=1),),
        hidden={1: 13, 2: 13, 3: 13},
    )
    # 1) 强行把 enum_limit 设极小触发 STATUS_TOO_MANY -> 依然必须推导出 2筒(19) 和 3筒(20) 为 certain=1
    counts = compute_counts(st)
    infs = infer_winners(st, counts, enum_limit=5)
    assert len(infs) == 1
    assert infs[0].status == STATUS_TOO_MANY
    assert infs[0].certain[19] == 1  # 2B
    assert infs[0].certain[20] == 1  # 3B

    # 2) 完整报告里 winners.certain_tiles、remaining.effective_unseen 和 suit_inference 均同步更新
    rep = analyze(st)
    assert not rep.errors
    w3 = rep.winner_items[0]
    sure_codes = {c["code"] for c in w3["certain_tiles"]}
    assert {"2B", "3B"} <= sure_codes
    # 2筒外面原本只剩1张未现, 被上家100%锁死 -> effective_unseen 必须归 0 (锁绝)!
    rem_2b = next(r for r in rep.remaining_rows if r["tile"] == "2筒")
    assert rem_2b["unseen"] == 1
    assert rem_2b["certain_locked"] == 1
    assert rem_2b["effective_unseen"] == 0
    assert rem_2b["exhausted"] is True


def test_flush_building_pong_recommends_pong_instead_of_fake_speedup():
    # 实盘截图复现: 5667W + 22 33 4 77 99T (9张条子含4对, 对家缺条打出2T, 牌墙44张)
    # 600局反事实模拟验证: 碰2条冲清一色条期望分 +5.69~+6.56 (不碰仅 +2.82~+2.96, 收益翻倍!)
    # 辅助器必须识别"清一色成势碰"并给出"建议碰", 绝不能误判成"假提速建议不碰"!
    own = "5W 6W 6W 7W 2T 2T 3T 3T 4T 7T 7T 9T 9T".split()
    st = make_state(
        own,
        voids={0: "B", 1: "B", 2: "T", 3: "W"},
        melds={1: [pong("3W")]},
        pending_discard=(2, "2T"),
        rivers={
            0: ["8W", "7B"],
            1: ["9W", "6W", "8T", "8W"],
            2: ["5T", "1T", "5T", "2T"],
            3: ["8T", "9B"],
        },
        hidden={1: 10, 2: 13, 3: 13},
    )
    rep = analyze(st)
    assert not rep.errors
    pong_f = next(f for f in rep.sections["alerts"] if f.title.startswith("碰2条"))
    assert "建议碰" in pong_f.title
    assert "清一色" in "\n".join(pong_f.basis)
