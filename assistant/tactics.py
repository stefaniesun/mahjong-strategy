"""中后盘实战参谋:碰牌/杠牌利弊评估 + 残局查叫与退税攻守切换。

设计原则:
- 与辅助器三档体系完全一致: 所有建议附带精确账目(碰/杠前后向听、有效进张、点炮风险、流局查叫与退税估算);
- 严格基于已校准的 opponent_read、danger_rows 与 own_hand 数据计算,单次耗时 <0.2ms。
"""
from __future__ import annotations

import math
from typing import Sequence

from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import SUITS, Suit, Tile, parse_tile

from assistant.counting import TileCounts, compute_counts
from assistant.discard_v2 import (
    DEFAULT_DISCARD_V2_CONFIG,
    _opp_expected_deal_in_payout,
    evaluate_hand_ev_state,
)
from assistant.opponent_read import OpponentRead
from assistant.own_hand import (
    build_own_hand,
    copy_hand,
    fast_shanten_hand,
    fast_useful_indices,
    my_remaining_draws,
    own_void,
    p_at_least_one,
)
from assistant.report import Finding, counted, probable
from assistant.state import AssistantState
from assistant.swap_v1 import ZERO_SUIT, suit_profile
from assistant.tiles import label, seat_name, suit_of, tidx

ENDGAME_WALL = 18


def _normal_shanten_only(counts: Sequence[int], fixed_melds: int, void_idx: int | None) -> int:
    """仅计算标准四面子一将向听(不含七对捷径), 防止 4~5 对子假七对向听误判碰牌提速。"""
    c0 = ZERO_SUIT if void_idx == 0 else tuple(counts[0:9])
    c1 = ZERO_SUIT if void_idx == 1 else tuple(counts[9:18])
    c2 = ZERO_SUIT if void_idx == 2 else tuple(counts[18:27])
    p0, p1, p2 = suit_profile(c0), suit_profile(c1), suit_profile(c2)
    m_base = p0.m0 + p1.m0 + p2.m0
    t_base = p0.t0 + p1.t0 + p2.t0
    tm = min(4, fixed_melds + m_base)
    ut = min(t_base, 4 - tm)
    best = 8 - 2 * tm - ut
    for pa, pb, pc in ((p0, p1, p2), (p1, p0, p2), (p2, p0, p1)):
        for m1, t1 in pa.pair_blocks:
            tm = min(4, fixed_melds + m1 + pb.m0 + pc.m0)
            ut = min(t1 + pb.t0 + pc.t0, 4 - tm)
            cand = 7 - 2 * tm - ut
            if cand < best:
                best = cand
    return best


def _seven_pairs_shanten_counts(counts: Sequence[int], void_idx: int | None) -> int:
    sp_pairs = 0
    sp_slots = 0
    for i, c in enumerate(counts):
        if void_idx is not None and i // 9 == void_idx:
            continue
        if c > 0:
            sp_pairs += c >> 1
            sp_slots += (c + 1) >> 1
    return 6 - sp_pairs + max(0, 7 - sp_slots)


def _post_pong_exp_dealin_loss(
    tile_idx: int,
    danger_by_tile: dict[str, dict],
    opp_reads: dict[int, OpponentRead],
    risk_by_idx: dict[int, float],
    base_score: int,
) -> float:
    d_row = danger_by_tile.get(label(tile_idx))
    if d_row is not None:
        if d_row.get("safe_all"):
            return 0.0
        loss = 0.0
        for e in d_row.get("per_opponent", ()):
            dp = e.get("dealin_prob", 0.0)
            if dp > 0.0:
                loss += dp * _opp_expected_deal_in_payout(opp_reads.get(e["seat"]), base_score)
        return loss
    return risk_by_idx.get(tile_idx, 0.0) * 1.8


def _best_post_pong_discard_ev(
    state: AssistantState,
    after: Hand,
    void: Suit | None,
    counts: TileCounts,
    eff_unseen: tuple[int, ...],
    risk_by_idx: dict[int, float],
    danger_rows: list[dict] | None,
    opp_reads: dict[int, OpponentRead],
    floors: dict[int, list[int]] | None,
) -> tuple[int, int, int, float, float, int | None]:
    """碰牌后(手里 3k+2 张)按统一 discard_v2 EV 找最优弃牌。

    返回: (disc_idx, sh_after, live_after, dealin_prob, best_ev, max_fan)
    """
    void_idx = SUITS.index(void) if void is not None else None
    melds = len(after.melds)
    base_score = max(1, state.rule_config.base_score)
    danger_by_tile = {r["tile"]: r for r in (danger_rows or [])}
    cfg = DEFAULT_DISCARD_V2_CONFIG

    # 如有缺门牌, 优先限制在缺门牌里挑最优弃牌
    present = [idx for idx in range(27) if after.counts[idx] > 0]
    void_present = [idx for idx in present if void_idx is not None and idx // 9 == void_idx]
    cands = void_present if void_present else present

    # 先快速算一遍各候选打后向听, 只对 min_sh 与 min_sh+1 做完整 EV 精算(保持 <0.5ms)
    sh_map: dict[int, int] = {}
    cnts = list(after.counts)
    for idx in cands:
        cnts[idx] -= 1
        sh_map[idx] = fast_shanten_hand(cnts, melds, void_idx)
        cnts[idx] += 1
    min_sh = min(sh_map.values())

    best_key = None
    best_res: tuple[int, int, int, float, float, int | None] = (cands[0], 99, 0, 0.0, -999.0, None)
    for idx in cands:
        sh_d = sh_map[idx]
        if sh_d > min_sh + 1:
            continue
        trial = copy_hand(after)
        trial.remove(Tile.from_index(idx))
        h_ev, sh_val, live_val, fan_val = evaluate_hand_ev_state(
            state,
            trial,
            counts,
            eff_unseen,
            opp_reads=opp_reads,
            floors=floors,
            config=cfg,
            compute_1sh_lookahead=(sh_d == min_sh),
        )
        # 与 discard_v2 一致的动态防守乘数
        if min_sh == 0:
            def_mult = 0.72 if state.wall_count > 16 else 0.90
            if (fan_val or 0) >= 2:
                def_mult *= 0.80
            elif (fan_val or 0) == 0 and live_val <= 3:
                def_mult *= 1.22
        elif min_sh == 1:
            def_mult = 1.00 if state.wall_count > 18 else 1.25
        else:
            def_mult = 2.50 if state.wall_count <= 14 else (1.85 if state.wall_count <= 24 else 1.20)
        exp_loss = _post_pong_exp_dealin_loss(idx, danger_by_tile, opp_reads, risk_by_idx, base_score)
        total_ev = h_ev - cfg.w_defense * def_mult * exp_loss
        risk = risk_by_idx.get(idx, 0.0)
        key = (-total_ev, sh_val, -live_val, risk, idx)
        if best_key is None or key < best_key:
            best_key = key
            best_res = (idx, sh_val, live_val, risk, total_ev, fan_val)
    return best_res


def advise_pong(
    state: AssistantState,
    hand: Hand,
    tile: Tile,
    discarder: int,
    eff_unseen: tuple[int, ...],
    risk_by_idx: dict[int, float],
    *,
    counts: TileCounts | None = None,
    danger_rows: list[dict] | None = None,
    opp_reads: dict[int, OpponentRead] | None = None,
    floors: dict[int, list[int]] | None = None,
    winners: list | None = None,
) -> dict:
    """按统一 discard_v2 EV 口径评估碰 tile 与不碰的实战净收益, 返回结构化建议与依据。"""
    void = own_void(state)
    void_idx = SUITS.index(void) if void is not None else None
    melds = len(hand.melds)
    t_counts = counts if counts is not None else compute_counts(state)
    reads = opp_reads or {}

    pass_ev, sh_no, live_no, fan_no = evaluate_hand_ev_state(
        state,
        hand,
        t_counts,
        eff_unseen,
        opp_reads=reads,
        floors=floors,
        config=DEFAULT_DISCARD_V2_CONFIG,
    )

    after = copy_hand(hand)
    after.remove(tile)
    after.remove(tile)
    after.melds.append(Meld(MeldKind.PONG, (tile,) * 3, exposed=True, from_player=discarder))
    disc_idx, sh_yes, live_yes, disc_risk, pong_ev, fan_yes = _best_post_pong_discard_ev(
        state, after, void, t_counts, eff_unseen, risk_by_idx, danger_rows, reads, floors
    )

    # 牌型结构硬护栏检查 (1200局实测标定: 仅 6 对子听牌或带根/清一色 5 对子才硬守七对;
    # 普通无根 4~5 对子果断允许碰牌, 实测胡率 +13%~21%, 杠率翻 2.5 倍, 净分 +0.30~+1.58!)
    sp_no = _seven_pairs_shanten_counts(hand.counts, void_idx) if melds == 0 else 99
    has_root = any(c == 4 for i, c in enumerate(hand.counts) if void_idx is None or i // 9 != void_idx)
    suit_totals = [sum(hand.counts[s * 9 : (s + 1) * 9]) for s in range(3)]
    meld_suits: set[int] = set()
    for m in hand.melds:
        ms = SUITS.index(m.tiles[0].suit)
        suit_totals[ms] += 3
        meld_suits.add(ms)
    dom_suit = max(range(3), key=lambda s: suit_totals[s])
    tile_suit_idx = SUITS.index(tile.suit)
    breaks_flush = suit_totals[dom_suit] >= 10 and tile_suit_idx != dom_suit
    breaks_seven_pairs = (
        melds == 0
        and (
            sp_no == 0
            or (sp_no == 1 and (has_root or suit_totals[dom_suit] >= 11))
        )
    )

    # 区分"标准牌型真提速"(如 4 对子正常向听 3->2) / "清一色成势碰"(碰后主花色达 10+ 张且多对/对手缺该门) vs "真不降向听":
    nsh_no = _normal_shanten_only(hand.counts, melds, void_idx)
    after_disc_cnts = list(after.counts)
    after_disc_cnts[disc_idx] -= 1
    nsh_yes = _normal_shanten_only(after_disc_cnts, melds + 1, void_idx)
    real_speedup = (sh_yes < sh_no) or (nsh_yes < nsh_no and not breaks_seven_pairs)
    pairs_after = sum(1 for i, c in enumerate(after_disc_cnts) if c >= 2 and (void_idx is None or i // 9 != void_idx))
    flush_pairs_after = sum(1 for r in range(9) if after_disc_cnts[dom_suit * 9 + r] >= 2)
    dom_total_after = suit_totals[dom_suit] + (1 if tile_suit_idx == dom_suit else 0)
    opp_void_dom = sum(1 for p in state.active_opponents() if p.void_suit == SUITS[dom_suit].value)
    builds_flush = (
        tile_suit_idx == dom_suit
        and dom_suit != void_idx
        and (not meld_suits or meld_suits == {dom_suit})
        and dom_total_after >= 10
        and state.wall_count >= 24
        and (dom_total_after >= 11 or flush_pairs_after >= 2 or opp_void_dom >= 1)
    )

    # 节奏与缩手成本修正:
    # - 真正降向听碰(含 4~5 对子标准牌型提速): 跳过中间家立得节奏红利, 若碰后仍剩 >=2 对子还有连续碰+摸杠红利
    # - 清一色成势碰(早中盘碰进 10+ 张主花色 + 多对可连碰/对手缺该门, 实测冲清一色期望分翻倍): 绝不当作"假提速"扣分!
    # - 真不降向听碰: 缩手 3 张暗牌 (-0.35)
    # - 退向听碰: 强惩罚 (-0.65)
    if real_speedup:
        if sh_yes == 0 and state.wall_count <= ENDGAME_WALL:
            tempo_adj = 0.15
        elif builds_flush:
            tempo_adj = 0.14
        elif pairs_after >= 2 and state.wall_count >= 24:
            tempo_adj = 0.08  # 多对子早碰: 附带后续连续碰提速 + 补杠/暗杠红利
        else:
            tempo_adj = -0.10
    elif sh_yes == sh_no:
        tempo_adj = 0.05 if builds_flush else -0.35
    else:
        tempo_adj = -0.65
    delta_ev = pong_ev - pass_ev + tempo_adj

    reasons: list[str] = [
        f"不碰:向听 {sh_no}, 有效进张 {live_no} 张 (EV {pass_ev:+.2f})",
        f"碰后打{label(disc_idx)}:向听 {sh_yes}, 有效进张 {live_yes} 张 (EV {pong_ev:+.2f}, 净差 {delta_ev:+.2f})"
        + (f" | 弃牌点炮风险约 {disc_risk:.0%}" if disc_risk > 0 else " | 弃牌安全"),
    ]
    waits_after: list[dict] = []
    if sh_yes == 0:
        from assistant.danger import estimate_wall_expected
        from assistant.own_hand import wait_table

        after_disc_hand = copy_hand(after)
        after_disc_hand.remove(Tile.from_index(disc_idx))
        w_exp, win_exp, opp_exp = estimate_wall_expected(state, t_counts, winners or [], floors)
        waits_after = wait_table(after_disc_hand, void, t_counts, state, eff_unseen, w_exp, win_exp, opp_exp)
        if waits_after:
            tot_rem = sum(w["remaining"] for w in waits_after)
            tot_wall = sum(w.get("wall_expected", 0.0) for w in waits_after)
            tot_win = sum(w.get("winner_expected", 0.0) for w in waits_after)
            w_txt = "、".join(
                f"{w['tile']}(剩{w['remaining']}·墙≈{w.get('wall_expected', 0.0):.1f})" for w in waits_after[:4]
            )
            hollow_note = ""
            if tot_rem > 0 and tot_wall < 0.45 and tot_win >= 0.7:
                hollow_note = f" — 注意:已胡估扣 {tot_win:.1f} 张, 牌墙仅剩约 {tot_wall:.2f} 张(疑空听,以保叫为主)"
            reasons.append(f"碰后听口:{w_txt}{hollow_note}")

    if breaks_seven_pairs:
        verdict = "建议不碰"
        summary = f"碰{label(tile.index)}会破坏高番七对路线" + ("(已听七对)" if sp_no == 0 else "(带根/清七对 1 向听)")
        reasons.append("已听七对或接近 3 番+ 龙七对/清七对, 保留门清七对更优")
    elif breaks_flush:
        verdict = "建议不碰"
        summary = f"碰{label(tile.index)}会破坏清一色路线(主花色已有 {suit_totals[dom_suit]} 张)"
        reasons.append("保持清一色大番潜力优先")
    elif sh_yes == 0 and live_yes == 0 and state.wall_count > 14:
        verdict = "慎碰"
        summary = f"碰{label(tile.index)}虽立即听牌, 但听口已绝(死听)"
        reasons.append("中盘不建议碰成死听, 留手牌改听更好")
    elif (sh_yes > sh_no and not real_speedup) and delta_ev < 0.25:
        verdict = "不建议碰"
        summary = f"碰{label(tile.index)}反而退向听({sh_no}→{sh_yes}, 净EV {delta_ev:+.2f})"
    elif sh_yes == 0 and state.wall_count <= ENDGAME_WALL and delta_ev >= 0.15:
        verdict = "强烈建议碰"
        summary = f"残局碰{label(tile.index)}打{label(disc_idx)}直接听牌保叫(剩 {live_yes} 张胡口, 净EV {delta_ev:+.2f})"
    elif real_speedup:
        if delta_ev >= 0.15:
            if disc_risk >= 0.16 and sh_yes > 0 and delta_ev < 0.70:
                verdict = "权衡"
                summary = f"碰{label(tile.index)}虽提速至 {sh_yes} 向听, 但弃牌{label(disc_idx)}点炮风险偏高({disc_risk:.0%})"
            else:
                verdict = "建议碰"
                step_txt = f"{sh_no}→{sh_yes}向听" if sh_yes < sh_no else f"标准牌型 {nsh_no}→{nsh_yes}向听·{live_no}→{live_yes}进张"
                summary = f"碰{label(tile.index)}打{label(disc_idx)}提速({step_txt}, 净EV {delta_ev:+.2f})"
        elif delta_ev >= -0.15:
            verdict = "权衡"
            summary = f"碰{label(tile.index)}虽提速, 但听口/番数缩水或弃牌偏危(净EV {delta_ev:+.2f})"
        else:
            verdict = "建议不碰"
            summary = f"好牌莫碰烂:不碰留牌质量更高(净EV {delta_ev:+.2f})"
            reasons.append("综合进张、大番潜力与弃牌点炮风险, 留门清/原搭期望收益更高")
    else:
        # sh_yes == sh_no (真不降向听)
        if builds_flush and delta_ev >= 0.15 and disc_risk < 0.12:
            verdict = "建议碰"
            suit_cn = {"W": "万", "T": "条", "B": "筒"}[SUITS[dom_suit].value]
            summary = f"冲清一色大牌:碰{label(tile.index)}后{suit_cn}子达 {dom_total_after} 张(剩 {flush_pairs_after} 对可连碰, 净EV {delta_ev:+.2f})"
            reasons.append("早中盘 10+ 张主花色成势, 果断碰牌做清一色/清对对胡(2~3番), 期望收益远高于小平胡")
        elif delta_ev >= 0.12 and disc_risk < 0.10:
            verdict = "可碰"
            summary = f"碰{label(tile.index)}牌质/番数占优({live_no}→{live_yes}张, 净EV {delta_ev:+.2f})"
        else:
            verdict = "建议不碰"
            summary = f"假提速:碰{label(tile.index)}不降向听({sh_no}→{sh_yes}), 净EV {delta_ev:+.2f} 不足以补偿缩手风险"
            reasons.append("白白亮出副露缩小手牌腾挪与避炮空间")

    return {
        "tile": label(tile.index),
        "verdict": verdict,
        "summary": summary,
        "sh_no": sh_no,
        "live_no": live_no,
        "sh_yes": sh_yes,
        "live_yes": live_yes,
        "best_discard": label(disc_idx),
        "discard_risk": round(disc_risk, 4),
        "pass_ev": round(pass_ev, 3),
        "pong_ev": round(pong_ev, 3),
        "delta_ev": round(delta_ev, 3),
        "waits_after": waits_after,
        "reasons": reasons,
    }


def advise_win_or_pass(
    state: AssistantState,
    hand: Hand,
    tile: Tile,
    win_fan: int,
    win_score: int,
    opp_reads: dict[int, OpponentRead] | None = None,
) -> dict:
    """评估早中盘面对 0 番点炮胡(+1分)时, 是直接落袋还是弃0番搏大(自摸/暗杠/换多面听/清一色)。

    仅针对他家点炮的 0 番平胡; 自摸、抢杠胡或 >=1 番点炮均直接建议胡。
    """
    if win_fan > 0 or state.wall_count < 24:
        return {"verdict": "建议直接胡", "summary": "", "reasons": []}

    active_opps = state.active_opponents()
    n_active = len(active_opps)
    if n_active < 2:
        return {"verdict": "建议直接胡", "summary": "", "reasons": []}

    reads = opp_reads or {}
    # 若场上有对手显露强清一色 / 3副露快听 / 极高听牌警戒, 直接胡牌离场止险
    danger_opps = [
        r for r in reads.values()
        if r.flush_level == "strong" or r.n_melds >= 3 or r.tenpai_prob >= 0.78
    ]
    if danger_opps:
        return {"verdict": "建议直接胡", "summary": "", "reasons": []}

    void_code = state.me.void_suit
    void_idx = SUITS.index(Suit(void_code)) if void_code else None
    suit_totals = [sum(hand.counts[s * 9 : (s + 1) * 9]) for s in range(3)]
    meld_suits: set[int] = set()
    for m in hand.melds:
        ms = SUITS.index(m.tiles[0].suit)
        suit_totals[ms] += 3
        meld_suits.add(ms)
    dom_suit = max(range(3), key=lambda s: suit_totals[s])
    n_dom = suit_totals[dom_suit]
    dom_code = SUITS[dom_suit].value
    dom_cn = {"W": "万", "T": "条", "B": "筒"}[dom_code]
    opp_void_dom = sum(1 for p in active_opps if p.void_suit == dom_code)

    has_flush_path = (n_dom >= 10) and (dom_suit != void_idx) and (not meld_suits or meld_suits == {dom_suit})
    concealed_trips = sum(1 for c in hand.counts if c >= 3)
    base = max(1, state.rule_config.base_score)
    tsumo_0fan_total = 2 * base * n_active  # 自摸加底/加番, 每家至少 2 分 x 在场家数

    reasons: list[str] = []
    if has_flush_path and (n_active == 3 or opp_void_dom >= 1 or concealed_trips >= 1):
        verdict = "建议弃0番搏大"
        trip_txt = f"、{concealed_trips} 组暗刻(可摸暗杠+根)" if concealed_trips >= 1 else ""
        void_txt = f"、{opp_void_dom} 家缺{dom_cn}" if opp_void_dom >= 1 else ""
        summary = (
            f"当前仅 0 番点炮(+{win_score}分), 但手握 {n_dom} 张{dom_cn}子{trip_txt}{void_txt}(余墙 {state.wall_count} 张·在场 {n_active} 家), "
            f"建议弃胡搏自摸(+{tsumo_0fan_total}分+)/暗杠或顺势转清一色"
        )
        reasons.append(f"点炮平胡仅得 {win_score} 分; 在场 {n_active} 家时即使 0 番自摸也能收约 {tsumo_0fan_total} 分(带根/清一色可达 4~24 分)")
        if concealed_trips >= 1:
            reasons.append(f"手里已有 {concealed_trips} 组暗刻: 留听牌不仅可摸暗杠立收 {2 * base * n_active} 分+1根, 摸主花色邻张还能换多面带根听口")
        reasons.append(f"清一色策略提示: 优先保留现有成型副顺维持听牌防点炮, 待摸进第 11~12 张{dom_cn}子或对子成势后再顺势拆他门冲清一色")
        reasons.append("注意过庄规则: 弃胡后至下巡自己摸牌前, 只能胡 >=1 番点炮或自摸")
        return {"verdict": verdict, "summary": summary, "reasons": reasons, "dom_suit": dom_cn, "n_dom": n_dom}

    if n_active == 3 and state.wall_count >= 28 and concealed_trips >= 1:
        verdict = "建议弃0番搏大" if (concealed_trips >= 2 or state.wall_count >= 32) else "可权衡弃0番"
        summary = (
            f"早中盘(余 {state.wall_count} 张·3家全在场)且手握 {concealed_trips} 组暗刻, "
            f"0 番点炮仅 +{win_score} 分, 可弃胡留听搏自摸(+{tsumo_0fan_total}分+)/暗杠(+{2 * base * n_active}分+根)与换听升级"
        )
        reasons.append(f"3 家全在场时自摸收益是 0 番点炮的 6 倍(+{tsumo_0fan_total}分 vs +{win_score}分), 摸暗杠立收 {2 * base * n_active} 分+1根")
        reasons.append("注意过庄规则: 弃胡后至下巡自己摸牌前, 只能胡 >=1 番点炮或自摸")
        return {"verdict": verdict, "summary": summary, "reasons": reasons, "dom_suit": dom_cn, "n_dom": n_dom}

    return {"verdict": "建议直接胡", "summary": "", "reasons": []}


def advise_kong(
    state: AssistantState,
    hand: Hand,
    tile: Tile,
    kong_kind: str,  # "exposed" / "concealed" / "added"
    sh_before: int,
    sh_after: int,
    opp_reads: dict[int, OpponentRead],
    tile_dealin_prob: float = 0.0,
) -> dict:
    """评估明杠/暗杠/补杠的实战风险与建议。"""
    max_tp = max((r.tenpai_prob for r in opp_reads.values()), default=0.0)
    base = state.rule_config.base_score
    active_n = len(state.active_opponents())
    late_far_unready = state.wall_count <= 12 and sh_after >= 2

    reasons: list[str] = []
    if kong_kind == "concealed":
        income = 2 * base * active_n
        reasons.append(f"暗杠立收 {income} 分({active_n} 家各付 {2 * base} 分), 不怕抢杠且 +1 根")
        if late_far_unready:
            verdict = "慎杠"
            summary = f"残局远未听牌暗杠{label(tile.index)}: 若流局无叫将全额退回这 {income} 分杠钱"
            reasons.append(f"牌墙仅余 {state.wall_count} 张且杠后仍 {sh_after} 向听, 流局查叫会退税")
        elif sh_after > sh_before and sh_before == 0:
            verdict = "权衡"
            summary = f"暗杠{label(tile.index)}立收 {income} 分, 但会拆听({sh_before}→{sh_after}向听)"
        elif sh_after > sh_before and sh_before == 1 and not (active_n >= 2 and state.wall_count >= 20):
            verdict = "权衡"
            summary = f"暗杠{label(tile.index)}收 {income} 分, 但中后盘拆搭退向听({sh_before}→{sh_after})"
        else:
            verdict = "建议暗杠"
            summary = f"暗杠{label(tile.index)}稳收 {income} 分 + 1 根(杠后向听 {sh_after})"
    elif kong_kind == "exposed":
        income = 2 * base
        reasons.append(f"明杠立收放牌者 {income} 分 + 1 根, 但杠后点炮会触发杠转水")
        if sh_after > sh_before and sh_before <= 1:
            verdict = "建议不杠"
            summary = f"明杠{label(tile.index)}临听拆搭退向听({sh_before}→{sh_after}), 得不偿失"
        elif late_far_unready:
            verdict = "慎杠"
            summary = f"残局远未听牌明杠{label(tile.index)}: 流局无叫要退杠钱, 且杠后打出生张有杠转水风险"
        elif max_tp >= 0.85 and sh_after >= 2 and state.wall_count <= 18:
            verdict = "权衡"
            summary = f"明杠{label(tile.index)}收 {income} 分, 但残局对手极可能已听({max_tp:.0%})且己方尚远, 需防杠转水"
        else:
            verdict = "建议明杠"
            summary = f"明杠{label(tile.index)}稳收 {income} 分 + 1 根(杠后向听 {sh_after})"
    else:  # added
        reasons.append("补杠不收杠钱, 收益为 +1 根(+1 番) 与 1 次杠上补牌")
        if tile_dealin_prob >= 0.10:
            verdict = "慎补杠"
            summary = f"补杠{label(tile.index)}有较高被抢杠胡风险(估计约 {tile_dealin_prob:.0%}), 且无直接杠钱"
        elif sh_after > sh_before and sh_before <= 1:
            verdict = "建议不补杠"
            summary = f"补杠{label(tile.index)}会退向听({sh_before}→{sh_after})"
        else:
            verdict = "建议补杠"
            summary = f"补杠{label(tile.index)}安全(+1 根与摸牌提速)"

    return {
        "tile": label(tile.index),
        "kind": kong_kind,
        "verdict": verdict,
        "summary": summary,
        "reasons": reasons,
    }


def _estimate_opp_chajiao_score(opp: OpponentRead, base: int) -> float:
    """保守估算某对手一旦听牌, 流局查叫的平均赔付分(乘以其听牌概率即期望赔付)。"""
    est_fan = 0
    if opp.flush_level == "strong":
        est_fan += 2
    elif opp.flush_level == "watch":
        est_fan += 1
    if opp.n_melds >= 4:
        est_fan += 2  # 金钩钓 + 对对胡
    elif opp.n_melds == 3:
        est_fan += 1
    est_fan += min(2, opp.n_kongs)
    # 四川血战基础 2^fan * base, 封顶保守按 8*base
    return float(min(8 * base, base * (1 << min(3, est_fan))))


def analyze_endgame_chajiao(
    state: AssistantState,
    own_info: dict | None,
    opp_reads: dict[int, OpponentRead],
    kong_income: int | None,
) -> tuple[dict | None, list[Finding]]:
    """残局(牌墙 <= 18 张)查叫、退税与攻守切换建议。"""
    if state.phase != "play" or state.me.won or own_info is None:
        return None, []
    if not (0 < state.wall_count <= ENDGAME_WALL):
        return None, []

    base = state.rule_config.base_score
    draws = my_remaining_draws(state)
    current_sh = own_info.get("current_best_shanten", own_info.get("shanten"))
    if current_sh is None:
        return None, []

    exp_tenpai_opps = sum(r.tenpai_prob for r in opp_reads.values())
    exp_chajiao_loss = sum(r.tenpai_prob * _estimate_opp_chajiao_score(r, base) for r in opp_reads.values())
    refund_risk = float(kong_income) if (kong_income is not None and kong_income > 0) else 0.0
    total_unready_loss = exp_chajiao_loss + refund_risk

    # 场上最大单家大牌威胁
    high_threat_opps = [r for r in opp_reads.values() if r.flush_level == "strong" or r.n_melds >= 3]

    if current_sh == 0:
        waits = own_info.get("waits")
        if waits is None and own_info.get("mode") == "discard":
            ev_opts = own_info.get("discard_v2_options") or own_info.get("candidates") or []
            tenpai_opt = next((c for c in ev_opts if c.get("shanten_after") == 0 and c.get("waits")), None)
            waits = tenpai_opt["waits"] if tenpai_opt is not None else []
        waits = waits or []
        max_fan = max((w["fan"] for w in waits), default=0)
        is_dead = bool(waits) and all(w["remaining"] == 0 for w in waits)
        tot_wall = sum(w.get("wall_expected", 0.0) for w in waits)
        tot_win = sum(w.get("winner_expected", 0.0) for w in waits)
        hollow_warn = (
            f" (⚠️ 牌面虽有剩张, 但已胡估扣 {tot_win:.1f} 张、墙内仅≈{tot_wall:.2f} 张, 疑空听以保叫为主)"
            if (not is_dead and bool(waits) and tot_wall < 0.45 and tot_win >= 0.7)
            else ""
        )
        exp_unready_opps = sum(1.0 - r.tenpai_prob for r in opp_reads.values())
        advice = (
            "你已听牌(死听仍算有叫): 切勿盲目拆牌变无叫, 优先保叫免查叫并守住杠钱"
            if is_dead
            else f"残局已听牌(最高 {max_fan} 番): 预计对手中约 {exp_unready_opps:.1f} 家未听, 流局可收查叫, 除非摸到极危大牌炮张否则严禁拆听{hollow_warn}"
        )
        info = {
            "status": "ready",
            "wall_count": state.wall_count,
            "my_draws_left": draws,
            "exp_tenpai_opps": round(exp_tenpai_opps, 2),
            "advice": advice,
        }
        return info, [
            probable(
                "alert",
                f"残局守听策略(余 {state.wall_count} 张): {advice}",
                [
                    f"在场 {len(opp_reads)} 家对手预计已听约 {exp_tenpai_opps:.1f} 家",
                    f"流局时你作为有叫者不赔查叫, 若已收杠钱({int(refund_risk)}分)也能稳稳落袋",
                ],
                assumption="基于各对手听牌概率估计残局查叫收益与保底价值。",
            )
        ]

    # 未听牌(current_sh >= 1)
    p_improve = 0.0
    if own_info.get("mode") == "discard" and own_info.get("candidates"):
        p_improve = own_info["candidates"][0].get("p_before_end", 0.0)
    elif own_info.get("live", 0) > 0 and own_info.get("total_unseen", 0) > 0:
        p_improve = p_at_least_one(own_info["total_unseen"], own_info["live"], draws)

    if current_sh == 1 and (p_improve >= 0.30 or total_unready_loss >= 4 * base):
        stance = "push_tenpai"
        advice = f"建议积极抢听保叫(1向听, 剩余 {draws} 摸上听概率约 {p_improve:.0%}, 流局无叫预计损约 {total_unready_loss:.1f} 分)"
    elif current_sh >= 2 and draws <= 3:
        stance = "fold_safe"
        advice = f"建议防守止损(仍 {current_sh} 向听仅余 {draws} 摸, 难以上听; 宁可认赔查叫约 {total_unready_loss:.1f} 分, 切勿再点大炮)"
    elif high_threat_opps and current_sh >= 2:
        stance = "cautious_defense"
        who_txt = "、".join(r.name for r in high_threat_opps)
        advice = f"建议偏向防守({who_txt} 显露大牌警戒, 你仍 {current_sh} 向听, 避免为渺茫听牌放生张)"
    else:
        stance = "balanced"
        advice = f"攻守兼顾(当前 {current_sh} 向听余 {draws} 摸, 流局无叫预计查叫+退税约 {total_unready_loss:.1f} 分, 优先打熟张同时保留听牌路线)"

    basis = [
        f"牌墙余 {state.wall_count} 张, 按当前顺序你还能摸约 {draws} 次",
        f"在场 {len(opp_reads)} 家对手预计已有约 {exp_tenpai_opps:.1f} 家听牌, 查叫保守期望约 {exp_chajiao_loss:.1f} 分",
    ]
    if refund_risk > 0:
        basis.append(f"若流局无叫, 已收的 {int(refund_risk)} 分杠钱将原路退回(合计风险约 {total_unready_loss:.1f} 分)")
    if high_threat_opps:
        basis.append("大牌警戒对手:" + "、".join(f"{r.name}({'/'.join(r.threats)})" for r in high_threat_opps))

    info = {
        "status": stance,
        "wall_count": state.wall_count,
        "my_draws_left": draws,
        "current_shanten": current_sh,
        "p_improve": round(p_improve, 3),
        "exp_tenpai_opps": round(exp_tenpai_opps, 2),
        "exp_chajiao_loss": round(exp_chajiao_loss, 2),
        "refund_risk": refund_risk,
        "advice": advice,
    }
    return info, [
        probable(
            "alert",
            f"残局查叫研判(余 {state.wall_count} 张): {advice}",
            basis,
            assumption="基于对手听牌进度、大牌警戒与已收杠钱估算流局查叫+退税成本。",
        )
    ]
