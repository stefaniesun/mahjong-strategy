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

from .counting import TileCounts, compute_counts
from .discard_v2 import (
    DEFAULT_DISCARD_V2_CONFIG,
    _opp_expected_deal_in_payout,
    evaluate_hand_ev_state,
)
from .opponent_read import OpponentRead
from .own_hand import (
    build_own_hand,
    copy_hand,
    fast_shanten_hand,
    fast_useful_indices,
    my_remaining_draws,
    own_void,
    p_at_least_one,
)
from .report import Finding, counted, probable
from .state import AssistantState
from .tiles import label, seat_name, suit_of, tidx

ENDGAME_WALL = 18


def _seven_pairs_shanten_counts(counts: Sequence[int], void_idx: int | None) -> int:
    pairs = 0
    unique = 0
    for i, c in enumerate(counts):
        if void_idx is not None and i // 9 == void_idx:
            continue
        if c >= 2:
            pairs += 1
        if c > 0:
            unique += 1
    return 6 - pairs + max(0, 7 - unique)


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

    # 牌型结构硬护栏检查
    breaks_seven_pairs = (
        melds == 0
        and _seven_pairs_shanten_counts(hand.counts, void_idx) <= 1
        and _seven_pairs_shanten_counts(hand.counts, void_idx) <= sh_no
    )
    suit_totals = [sum(hand.counts[s * 9 : (s + 1) * 9]) for s in range(3)]
    for m in hand.melds:
        suit_totals[SUITS.index(m.tiles[0].suit)] += 3
    dom_suit = max(range(3), key=lambda s: suit_totals[s])
    breaks_flush = suit_totals[dom_suit] >= 10 and SUITS.index(tile.suit) != dom_suit

    # 节奏与缩手成本修正:
    # - 降向听碰: 跳过中间家立得节奏红利, 扣除亮副露缩手成本后净微调 (-0.10; 残局直接听牌 +0.15)
    # - 不降向听碰: 无速度增益却缩手 3 张暗牌 (-0.35), 仅当牌质/进张/对对胡净增益足以覆盖才碰
    # - 退向听碰: 强惩罚 (-0.65)
    if sh_yes < sh_no:
        tempo_adj = 0.15 if (sh_yes == 0 and state.wall_count <= ENDGAME_WALL) else -0.10
    elif sh_yes == sh_no:
        tempo_adj = -0.35
    else:
        tempo_adj = -0.65
    delta_ev = pong_ev - pass_ev + tempo_adj

    reasons: list[str] = [
        f"不碰:向听 {sh_no}, 有效进张 {live_no} 张 (EV {pass_ev:+.2f})",
        f"碰后打{label(disc_idx)}:向听 {sh_yes}, 有效进张 {live_yes} 张 (EV {pong_ev:+.2f}, 净差 {delta_ev:+.2f})"
        + (f" | 弃牌点炮风险约 {disc_risk:.0%}" if disc_risk > 0 else " | 弃牌安全"),
    ]

    if breaks_seven_pairs:
        verdict = "建议不碰"
        summary = f"碰{label(tile.index)}会破坏七对路线(当前七对已 ≤1 向听)"
        reasons.append("已接近七对成牌, 副露后失去七对(2番+)资格")
    elif breaks_flush:
        verdict = "建议不碰"
        summary = f"碰{label(tile.index)}会破坏清一色路线(主花色已有 {suit_totals[dom_suit]} 张)"
        reasons.append("保持清一色大番潜力优先")
    elif sh_yes == 0 and live_yes == 0 and state.wall_count > 14:
        verdict = "慎碰"
        summary = f"碰{label(tile.index)}虽立即听牌, 但听口已绝(死听)"
        reasons.append("中盘不建议碰成死听, 留手牌改听更好")
    elif sh_yes > sh_no and delta_ev < 0.25:
        verdict = "不建议碰"
        summary = f"碰{label(tile.index)}反而退向听({sh_no}→{sh_yes}, 净EV {delta_ev:+.2f})"
    elif sh_yes == 0 and state.wall_count <= ENDGAME_WALL and delta_ev >= 0.15:
        verdict = "强烈建议碰"
        summary = f"残局碰{label(tile.index)}打{label(disc_idx)}直接听牌保叫(剩 {live_yes} 张胡口, 净EV {delta_ev:+.2f})"
    elif sh_yes < sh_no:
        if delta_ev >= 0.15:
            if disc_risk >= 0.16 and sh_yes > 0 and delta_ev < 0.70:
                verdict = "权衡"
                summary = f"碰{label(tile.index)}虽提速至 {sh_yes} 向听, 但弃牌{label(disc_idx)}点炮风险偏高({disc_risk:.0%})"
            else:
                verdict = "建议碰"
                summary = f"碰{label(tile.index)}打{label(disc_idx)}提速({sh_no}→{sh_yes}向听, 净EV {delta_ev:+.2f})"
        elif delta_ev >= -0.15:
            verdict = "权衡"
            summary = f"碰{label(tile.index)}虽提速({sh_no}→{sh_yes}), 但听口/番数缩水或弃牌偏危(净EV {delta_ev:+.2f})"
        else:
            verdict = "建议不碰"
            summary = f"好牌莫碰烂:不碰留牌质量更高(净EV {delta_ev:+.2f})"
            reasons.append("综合进张、大番潜力与弃牌点炮风险, 留门清/原搭期望收益更高")
    else:
        # sh_yes == sh_no (或极少数退向听但大番净赚 >= 0.25)
        if delta_ev >= 0.12 and disc_risk < 0.10:
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
        "reasons": reasons,
    }


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
        waits = own_info.get("waits") or (
            own_info["candidates"][0]["waits"] if own_info.get("mode") == "discard" and own_info.get("candidates") else []
        )
        max_fan = max((w["fan"] for w in waits), default=0)
        is_dead = bool(waits) and all(w["remaining"] == 0 for w in waits)
        exp_unready_opps = sum(1.0 - r.tenpai_prob for r in opp_reads.values())
        advice = (
            "你已听牌(死听仍算有叫): 切勿盲目拆牌变无叫, 优先保叫免查叫并守住杠钱"
            if is_dead
            else f"残局已听牌(最高 {max_fan} 番): 预计对手中约 {exp_unready_opps:.1f} 家未听, 流局可收查叫, 除非摸到极危大牌炮张否则严禁拆听"
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
