"""中后盘实战参谋:碰牌/杠牌利弊评估 + 残局查叫与退税攻守切换。

设计原则:
- 与辅助器三档体系完全一致: 所有建议附带精确账目(碰/杠前后向听、有效进张、点炮风险、流局查叫与退税估算);
- 严格基于已校准的 opponent_read、danger_rows 与 own_hand 数据计算,单次耗时 <0.2ms。
"""
from __future__ import annotations

from typing import Sequence

from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import SUITS, Suit, Tile, parse_tile

from assistant.counting import TileCounts
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
from assistant.tiles import label, seat_name, suit_of, tidx

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


def _best_post_pong_discard(
    hand: Hand,
    void: Suit | None,
    eff_unseen: tuple[int, ...],
    risk_by_idx: dict[int, float],
) -> tuple[int, int, int, float]:
    """碰牌后(手里 3k+2 张)找最优弃牌, 返回 (disc_idx, sh_after, live_after, dealin_prob)。"""
    best_key = None
    best_res = (0, 99, 0, 0.0)
    void_idx = SUITS.index(void) if void is not None else None
    melds = len(hand.melds)
    counts = list(hand.counts)
    for idx in range(27):
        if counts[idx] <= 0:
            continue
        counts[idx] -= 1
        sh = fast_shanten_hand(counts, melds, void_idx)
        useful = fast_useful_indices(counts, melds, void_idx, sh)
        live = sum(eff_unseen[u] for u in useful)
        risk = risk_by_idx.get(idx, 0.0)
        key = (sh, -live, risk, idx)
        if best_key is None or key < best_key:
            best_key = key
            best_res = (idx, sh, live, risk)
        counts[idx] += 1
    return best_res


def advise_pong(
    state: AssistantState,
    hand: Hand,
    tile: Tile,
    discarder: int,
    eff_unseen: tuple[int, ...],
    risk_by_idx: dict[int, float],
) -> dict:
    """评估碰 tile 的实战利弊, 返回结构化建议与依据。"""
    void = own_void(state)
    void_idx = SUITS.index(void) if void is not None else None
    melds = len(hand.melds)

    sh_no = fast_shanten_hand(hand.counts, melds, void_idx)
    useful_no = fast_useful_indices(hand.counts, melds, void_idx, sh_no)
    live_no = sum(eff_unseen[u] for u in useful_no)

    after = copy_hand(hand)
    after.remove(tile)
    after.remove(tile)
    after.melds.append(Meld(MeldKind.PONG, (tile,) * 3, exposed=True, from_player=discarder))
    disc_idx, sh_yes, live_yes, disc_risk = _best_post_pong_discard(after, void, eff_unseen, risk_by_idx)

    # 牌型结构影响检查
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

    reasons: list[str] = [
        f"不碰:向听 {sh_no}, 有效进张 {live_no} 张",
        f"碰后打{label(disc_idx)}:向听 {sh_yes}, 有效进张 {live_yes} 张"
        + (f"(点炮风险约 {disc_risk:.0%})" if disc_risk > 0 else "(安全)"),
    ]

    if breaks_seven_pairs:
        verdict = "建议不碰"
        summary = f"碰{label(tile.index)}会破坏七对路线(当前七对已 ≤1 向听)"
        reasons.append("已接近七对成牌, 副露后失去七对(2番+)资格")
    elif breaks_flush:
        verdict = "建议不碰"
        summary = f"碰{label(tile.index)}会破坏清一色路线(主花色已有 {suit_totals[dom_suit]} 张)"
        reasons.append("保持清一色大番潜力优先")
    elif sh_yes < sh_no:
        if sh_yes == 0 and live_yes == 0:
            verdict = "慎碰"
            summary = f"碰{label(tile.index)}虽立即听牌, 但听口已绝(死听)"
            reasons.append("除非残局急需保叫免查叫, 否则不建议碰成死听")
        elif sh_yes == 0 and state.wall_count <= ENDGAME_WALL:
            verdict = "强烈建议碰"
            summary = f"残局碰{label(tile.index)}打{label(disc_idx)}直接听牌保叫(剩 {live_yes} 张胡口)"
        elif disc_risk >= 0.16 and sh_yes > 0:
            verdict = "权衡"
            summary = f"碰{label(tile.index)}虽提速至 {sh_yes} 向听, 但碰后最优弃牌{label(disc_idx)}点炮风险偏高({disc_risk:.0%})"
        else:
            verdict = "建议碰"
            summary = f"碰{label(tile.index)}打{label(disc_idx)}提速 1 步({sh_no}→{sh_yes}向听, 进张 {live_yes} 张)"
    elif sh_yes == sh_no:
        # 四川血战允许不退向听碰牌: 进张不亏或做对对胡/残局冲听时可碰提节奏
        toitoi_push = melds >= 2 and live_yes >= max(1, live_no - 3)
        endgame_push = state.wall_count <= ENDGAME_WALL and sh_yes <= 1 and live_yes >= live_no - 2 and disc_risk < 0.14
        if (live_yes >= live_no + 2 and disc_risk < 0.12) or toitoi_push or endgame_push:
            verdict = "可碰"
            summary = f"碰{label(tile.index)}向听保持 {sh_yes}, 进张/节奏占优({live_no}→{live_yes}张)"
        else:
            verdict = "建议不碰"
            summary = f"假提速:碰{label(tile.index)}不降向听({sh_no}→{sh_yes}), 且进张缩水({live_no}→{live_yes}张)"
            reasons.append("白白亮出副露缩小手牌腾挪空间")
    else:
        verdict = "不建议碰"
        summary = f"碰{label(tile.index)}反而退向听({sh_no}→{sh_yes})"

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
        if sh_after > sh_before and sh_before <= 1:
            verdict = "权衡"
            summary = f"暗杠{label(tile.index)}立收 {income} 分, 但临听拆顺子退向听({sh_before}→{sh_after})"
        elif late_far_unready:
            verdict = "慎杠"
            summary = f"残局远未听牌暗杠{label(tile.index)}: 若流局无叫将全额退回这 {income} 分杠钱"
            reasons.append(f"牌墙仅余 {state.wall_count} 张且杠后仍 {sh_after} 向听, 流局查叫会退税")
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
        elif max_tp >= 0.85 and sh_after >= 2:
            verdict = "权衡"
            summary = f"明杠{label(tile.index)}收 {income} 分, 但对手极可能已听({max_tp:.0%})且己方尚远, 需防杠转水"
        else:
            verdict = "建议明杠"
            summary = f"明杠{label(tile.index)}稳收 {income} 分 + 1 根(杠后向听 {sh_after})"
    else:  # added
        reasons.append("补杠不收杠钱, 收益为 +1 根(+1 番) 与 1 次杠上补牌")
        if tile_dealin_prob >= 0.12:
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
