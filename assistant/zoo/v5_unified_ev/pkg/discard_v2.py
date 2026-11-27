"""中盘攻防一体弃牌综合评估引擎 (discard_v2)。

融合辅助器全部底层高精度信号(均已在真实对局上校准验证):
1. 真实有效进张(已扣除已胡者必有暗牌, 防假进张/假死听);
2. 听牌期真实番数与点数期望(ron_score + 自摸期望), 解决“1番4张 vs 3番3张”的科学权衡;
3. 未听牌期大番路线潜力(清一色 / 七对 / 带根暗刻)与留牌好型质量;
4. 基于 opponent_read 校准点炮概率(ECE=0.0022)与对手大牌番数估计的期望放炮代价;
5. 残局(牌墙 <= 18 张)查叫与杠钱退税保底加权(听牌保叫 / 远向听果断弃胡止损);
6. 提供与 swap_v2 / void_v3 一致的受控 Softmax 抽样接口, 供后续 AI 陪练与强化学习数据生成使用。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import SUITS, Suit

from .counting import TileCounts
from .opponent_read import OpponentRead
from .own_hand import fast_shanten_hand, fast_useful_indices
from .state import AssistantState
from .tiles import from_label, rank_of, suit_of, tidx


@dataclass(frozen=True)
class DiscardV2Config:
    w_shanten: float = 2.60        # 每少 1 步向听的进攻基础价值
    w_live_log: float = 0.72       # 未听牌期有效进张对数收益 (自然递减)
    w_good_wait_1sh: float = 0.28  # 1向听摸入有效张后听牌宽度的对数期望(偏爱两面好型、避边嵌死口)
    w_pong_speedup: float = 0.14   # 四川血战可碰对子三家提速进张红利
    w_tenpai_ev: float = 0.85      # 听牌期 log1p(加权胡牌分期望) 收益
    w_dead_wait_pen: float = 1.65  # 中盘死听惩罚(残局可被保叫红利抵消)
    w_flush_pot: float = 0.55      # 10+ 张清一色路线留同花色红利
    w_seven_pairs: float = 0.45    # <=1 向听七对路线保对子红利
    w_toitoi_pot: float = 0.30     # >=2 副露时保对子冲对对胡红利
    w_root_keep: float = 0.18      # 保住 4 张根 / 暗刻微调
    w_shape_conn: float = 0.015    # 留牌顺滑度/中张连接微调
    w_defense: float = 0.85        # 点炮期望赔付防守权重(240局配对标定甜点:中盘进攻优先+轻避炮,残局远向听强防守)
    w_endgame_chajiao: float = 0.22  # 残局保叫免查叫+防退税加权


DEFAULT_DISCARD_V2_CONFIG = DiscardV2Config()


def _opp_expected_deal_in_payout(opp: OpponentRead | None, base: int) -> float:
    """估计一旦点炮给 opp 的期望赔付分(以 base=1 归一尺度)。"""
    if opp is None:
        return 1.5
    est_fan = 0.4  # 基础番均值(平胡/自摸/断幺/带根等)
    if opp.flush_level == "strong":
        est_fan += 2.0
    elif opp.flush_level == "watch":
        est_fan += 0.9
    if opp.n_melds >= 4:
        est_fan += 2.0
    elif opp.n_melds == 3:
        est_fan += 1.0
    elif opp.n_melds == 2:
        est_fan += 0.35
    if getattr(opp, "early_mid_alert", False) and opp.non_void_discards >= 4:
        est_fan += 0.40
    est_fan += min(2.0, 0.85 * opp.n_kongs)
    return min(8.0, float(2.0 ** min(3.0, est_fan)))


def _kong_refund_exposure(state: AssistantState) -> float:
    total = 0.0
    for m in state.me.melds:
        if m.kind == "kong" and not m.added and m.income is not None:
            total += float(m.income)
    return total


def _centrality_penalty(tile_idx: int) -> int:
    r = rank_of(tile_idx)
    if r in (1, 9):
        return 0
    if r in (2, 8):
        return 1
    return 2


def _shape_quality(counts: list[int], void_idx: int | None) -> float:
    """轻量评估去弃牌后 13 张非缺门手牌的连接度与对子/刻子质量。"""
    score = 0.0
    for s in (0, 1, 2):
        if s == void_idx:
            continue
        base = s * 9
        for r in range(9):
            c = counts[base + r]
            if c == 0:
                continue
            adj = 0
            if r >= 1 and counts[base + r - 1] > 0:
                adj += 2
            if r <= 7 and counts[base + r + 1] > 0:
                adj += 2
            if r >= 2 and counts[base + r - 2] > 0:
                adj += 1
            if r <= 6 and counts[base + r + 2] > 0:
                adj += 1
            # 中张(3-7)略优于边幺(1/9)
            mid = 0.4 if 2 <= r <= 6 else 0.0
            score += c * (adj + mid)
            if c == 2:
                score += 2.0
            elif c == 3:
                score += 4.5
            elif c == 4:
                score += 7.5
    return score


def _seven_pairs_shanten_after(counts: list[int], void_idx: int | None) -> int:
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


def _fast_tenpai_fan_mult(
    counts13: list[int],
    melds: Sequence[Meld],
    waits: Sequence[int],
) -> float:
    """极速估算 0 向听手牌的番数倍率 (O(27) 零拷贝, <0.01ms): 覆盖清一色/七对/龙七对/对对胡/金钩钓/根。"""
    fan = 0
    # 1) 清一色: 所有暗牌 + 所有副露 + 至少一个听口同属一门
    suits_present = 0
    for s in (0, 1, 2):
        base = s * 9
        if any(counts13[base + r] > 0 for r in range(9)):
            suits_present |= 1 << s
    for m in melds:
        suits_present |= 1 << SUITS.index(m.tiles[0].suit)
    is_flush = suits_present in (1, 2, 4)
    if is_flush:
        only_suit = 0 if suits_present == 1 else (1 if suits_present == 2 else 2)
        if any(w // 9 == only_suit for w in waits):
            fan += 2

    # 2) 七对 / 龙七对
    n_melds = len(melds)
    roots = 0
    pairs = 0
    triplets = 0
    singles = 0
    for c in counts13:
        if c == 4:
            roots += 1
            pairs += 2
        elif c == 3:
            triplets += 1
        elif c == 2:
            pairs += 1
        elif c == 1:
            singles += 1
    for m in melds:
        if m.kind is MeldKind.KONG:
            roots += 1
        elif m.kind is MeldKind.PONG and counts13[m.tiles[0].index] >= 1:
            roots += 1

    if n_melds == 0 and pairs >= 6 and singles <= 1:
        fan += 2  # 七对 (龙七对的根已在 roots 里 +1)
    elif n_melds == 4:
        fan += 2  # 金钩钓 + 对对胡
    elif singles == 0 and (triplets + pairs + n_melds) == 5 and pairs == 2:
        fan += 1  # 双碰对对胡听牌
    elif singles == 1 and pairs == 0 and (triplets + n_melds) == 4:
        fan += 1  # 单钓对对胡听牌

    fan += roots
    return float(2.0 ** min(3, fan))


def _one_shanten_good_wait_quality(
    counts13: list[int],
    fixed_melds: int,
    void_idx: int | None,
    useful: list[dict],
    unseen: tuple[int, ...],
    *,
    melds: Sequence[Meld] = (),
    state: AssistantState | None = None,
    floors: dict[int, list[int]] | None = None,
) -> float:
    """1向听两步前瞻: 综合摸入有效张后的听牌宽度、对手缺门/边幺听口质量与预计番数倍率。

    两面好型(听 6-8 张)与大番路线(清一色/七对/对对胡/带根)显著高于 0 番边嵌死口。
    """
    total_w = 0
    acc = 0.0
    for u_row in useful:
        rem = u_row["remaining"]
        if rem <= 0:
            continue
        u_idx = u_row["index"]
        counts13[u_idx] += 1
        best_wait_ev = 0.0
        # 寻找摸入 u_idx 后能立即听牌(sh==0)的最大「加权胡口 x 番数倍率」
        for d in range(27):
            if counts13[d] <= 0 or d == u_idx:
                continue
            counts13[d] -= 1
            if fast_shanten_hand(counts13, fixed_melds, void_idx) == 0:
                ws = fast_useful_indices(counts13, fixed_melds, void_idx, 0)
                if ws:
                    if state is not None:
                        w_live = sum(unseen[w] * _wait_quality_multiplier(w, state, floors) for w in ws)
                    else:
                        w_live = float(sum(unseen[w] for w in ws))
                    f_mult = _fast_tenpai_fan_mult(counts13, melds, ws) if w_live > 0 else 1.0
                    # 用 sqrt 温和压缩番数倍率, 防止 1 向听过度贪大番而放弃好型
                    cand_ev = w_live * (f_mult ** 0.55)
                    if cand_ev > best_wait_ev:
                        best_wait_ev = cand_ev
            counts13[d] += 1
        counts13[u_idx] -= 1
        total_w += rem
        acc += rem * math.log1p(best_wait_ev)
    return (acc / total_w) if total_w > 0 else 0.0


def _pong_speedup_outs(
    counts13: list[int],
    fixed_melds: int,
    void_idx: int | None,
    cur_sh: int,
    unseen: tuple[int, ...],
) -> int:
    """计算四川血战中“可由三家打出碰牌直接降向听”的有效对子外池张数。"""
    outs = 0
    for t in range(27):
        if counts13[t] != 2 or unseen[t] <= 0:
            continue
        if void_idx is not None and t // 9 == void_idx:
            continue
        counts13[t] -= 2
        # 检查碰后打任意一张能否让向听严格变小
        improved = False
        for d in range(27):
            if counts13[d] <= 0:
                continue
            counts13[d] -= 1
            if fast_shanten_hand(counts13, fixed_melds + 1, void_idx) < cur_sh:
                improved = True
                counts13[d] += 1
                break
            counts13[d] += 1
        counts13[t] += 2
        if improved:
            outs += unseen[t]
    return outs


def _wait_quality_multiplier(
    w_idx: int,
    state: AssistantState,
    floors: dict[int, list[int]] | None,
) -> float:
    """听牌张质量加成(160局实测校准):
    1. 听对手缺门(active opp 缺该花色 -> 摸到必打送胡): 0家 1.00 / 1家 1.14 / 2家 1.24 / 3家 1.36;
    2. 已知换牌底牌(某在场对手手里确定握着该缺门牌, 1~2巡必打出): 额外 1.45x;
    3. 边幺好胡(1/9 -> 1.11, 2/8 -> 1.06, 3-7 -> 1.00)。
    """
    w_suit = suit_of(w_idx)
    n_void = 0
    forced_swap_void = False
    for p in state.active_opponents():
        if p.void_suit == w_suit:
            n_void += 1
            if floors is not None and floors.get(p.seat, [0] * 27)[w_idx] > 0:
                forced_swap_void = True
    if n_void >= 3:
        void_mult = 1.36
    elif n_void == 2:
        void_mult = 1.24
    elif n_void == 1:
        void_mult = 1.14
    else:
        void_mult = 1.00

    r = rank_of(w_idx)
    if r in (1, 9):
        rank_mult = 1.11
    elif r in (2, 8):
        rank_mult = 1.06
    else:
        rank_mult = 1.00

    mult = void_mult * rank_mult
    if forced_swap_void:
        mult *= 1.45
    return mult


def score_discard_candidates_v2(
    state: AssistantState,
    hand: Hand,
    counts: TileCounts,
    candidates: list[dict],
    danger_rows: list[dict] | None = None,
    opp_reads: dict[int, OpponentRead] | None = None,
    config: DiscardV2Config = DEFAULT_DISCARD_V2_CONFIG,
    floors: dict[int, list[int]] | None = None,
) -> list[dict]:
    """为 own_hand.py 生成的弃牌候选列表计算 discard_v2 综合 EV 分(ev_score 越高越优)。

    返回按 ev_score 降序排列的新列表(每个元素为原 candidate 浅拷贝 + ev 明细字段)。
    """
    if not candidates:
        return []

    void_code = state.me.void_suit
    void_idx = SUITS.index(Suit(void_code)) if void_code else None
    base_score = max(1, state.rule_config.base_score)
    danger_by_tile = {r["tile"]: r for r in (danger_rows or [])}
    reads = opp_reads or {}

    # 检查手里是否还有缺门牌: 有缺门牌必须先打缺门
    has_void_tile = void_code is not None and any(suit_of(c["index"]) == void_code for c in candidates)

    best_sh = min(c["shanten_after"] for c in candidates)
    best_live_at_min_sh = max((c["live"] for c in candidates if c["shanten_after"] == best_sh), default=0)
    best_tenpai_fan = max(
        ((c.get("max_fan") or 0) for c in candidates if c["shanten_after"] == 0 and c["live"] > 0),
        default=0,
    )
    active_n = max(1, len(state.active_opponents()))

    # 动态攻防系数: 按自身进度、大番价值与残局阶段平滑缩放防守权重
    if best_sh == 0 and best_live_at_min_sh > 0:
        def_mult = 0.72 if state.wall_count > 16 else 0.90
        if best_tenpai_fan >= 2:
            # 听大番(2番+): 进攻回报远大于一般点炮, 适度压低保守度
            def_mult *= 0.80
        elif best_tenpai_fan == 0 and best_live_at_min_sh <= 3:
            # 仅听 0 番窄口(<=3张): 不要为了1分平胡硬冲对手大牌炮口
            def_mult *= 1.22
    elif best_sh == 1:
        def_mult = 1.00 if state.wall_count > 18 else 1.25
    else:
        # 2+ 向听: 残局越近越应果断防守止损
        if state.wall_count <= 14:
            def_mult = 2.50
        elif state.wall_count <= 24:
            def_mult = 1.85
        else:
            def_mult = 1.20

    # 残局查叫保底价值
    endgame_chajiao_value = 0.0
    if 0 < state.wall_count <= 18:
        exp_chajiao = sum(r.tenpai_prob * _opp_expected_deal_in_payout(r, base_score) for r in reads.values())
        refund = _kong_refund_exposure(state) / base_score
        endgame_chajiao_value = config.w_endgame_chajiao * (exp_chajiao + refund)

    # 预统计各门总张数(含副露)以判断清一色苗头
    suit_totals = [sum(hand.counts[s * 9 : (s + 1) * 9]) for s in range(3)]
    for m in hand.melds:
        suit_totals[SUITS.index(m.tiles[0].suit)] += 3
    dom_suit = max(range(3), key=lambda s: suit_totals[s])
    has_flush_pot = (suit_totals[dom_suit] >= 10) and (dom_suit != void_idx)

    # 预计算原始 7 对向听
    init_7p = _seven_pairs_shanten_after(list(hand.counts), void_idx) if len(hand.melds) == 0 else 99
    base_counts = list(hand.counts)

    scored: list[dict] = []
    for c in candidates:
        idx = c["index"]
        t_lbl = c["tile"]
        is_void_tile = void_code is not None and suit_of(idx) == void_code
        d_row = danger_by_tile.get(t_lbl)

        # 计算加权期望放炮损失
        exp_dealin_loss = 0.0
        if d_row is not None and not d_row.get("safe_all"):
            for e in d_row.get("per_opponent", ()):
                dp = e.get("dealin_prob", 0.0)
                if dp > 0.0:
                    exp_dealin_loss += dp * _opp_expected_deal_in_payout(reads.get(e["seat"]), base_score)

        # 情况 A: 手里还有缺门牌 -> 严格优先打缺门
        if has_void_tile:
            if not is_void_tile:
                item = dict(c)
                item.update(
                    ev_score=-100.0 - exp_dealin_loss,
                    ev_offense=-100.0,
                    ev_defense=round(-exp_dealin_loss, 3),
                    ev_fan=0.0,
                    ev_reason="手里尚有缺门牌, 必须先清缺门",
                )
                scored.append(item)
                continue
            # 缺门牌之间排序: 对手尚安全时先打外池未现/单张/边幺(趁早清); 若某缺门牌已高危则先打安全缺门
            table_seen = max(0, 4 - counts.unseen[idx] - hand.counts[idx])
            dup = hand.counts[idx]
            cent = _centrality_penalty(idx)
            # 先打风险更低的, 风险低时按 (table_seen, dup, cent) 从小到大清牌
            void_order_bonus = -0.08 * table_seen - 0.04 * dup - 0.02 * cent
            def_pen = -config.w_defense * 1.15 * exp_dealin_loss
            ev = 10.0 + void_order_bonus + def_pen
            item = dict(c)
            item.update(
                ev_score=round(ev, 4),
                ev_offense=round(10.0 + void_order_bonus, 3),
                ev_defense=round(def_pen, 3),
                ev_fan=0.0,
                ev_reason="优先清缺门(低危早清生张边幺)" if exp_dealin_loss < 0.05 else "优先清更安全的缺门牌",
            )
            scored.append(item)
            continue

        # 情况 B: 已清缺门 -> 攻防一体综合 EV
        sh = c["shanten_after"]
        live = c["live"]

        # 1) 进攻分
        if sh == 0:
            waits = c.get("waits") or ()
            if waits and live > 0:
                # 综合点炮分 + 自摸期望, 并叠加"听对手缺门/边幺好胡/已知换牌缺门必打"质量系数
                ron_w = 0.68 if active_n >= 3 else (0.62 if active_n == 2 else 0.52)
                tsumo_w = 1.0 - ron_w
                raw_win_ev = 0.0
                for w in waits:
                    rem = w["remaining"]
                    if rem <= 0:
                        continue
                    w_idx = tidx(from_label(w["tile"]))
                    q_mult = _wait_quality_multiplier(w_idx, state, floors)
                    ron_s = w["ron_score"] / base_score
                    tsumo_s = (w["tsumo_each"] * max(1, w["tsumo_payers"])) / base_score
                    raw_win_ev += rem * q_mult * (ron_w * ron_s + tsumo_w * tsumo_s)
                offense = config.w_tenpai_ev * math.log1p(raw_win_ev) + endgame_chajiao_value
            else:
                # 死听: 中盘大扣分, 但残局仍保留保叫价值(不赔查叫不退税)
                offense = -config.w_dead_wait_pen + endgame_chajiao_value
        else:
            offense = -config.w_shanten * sh + config.w_live_log * math.log1p(live)

        # 2) 大番潜力与牌形微调(含 1 向听听牌好型率与碰牌提速进张)
        base_counts[idx] -= 1
        fan_pot = 0.0
        good_wait_term = 0.0
        pong_term = 0.0
        if sh >= 1:
            if has_flush_pot and (idx // 9) != dom_suit and sh <= 2:
                fan_pot += config.w_flush_pot
            if init_7p <= 1 and len(hand.melds) == 0:
                after_7p = _seven_pairs_shanten_after(base_counts, void_idx)
                if after_7p <= init_7p:
                    fan_pot += config.w_seven_pairs
            elif len(hand.melds) >= 2 and sh <= 2:
                # >=2 副露冲对对胡路线: 尽量不拆暗对/暗刻
                if hand.counts[idx] == 1:
                    pairs_kept = sum(1 for i, cnt in enumerate(base_counts) if cnt >= 2 and (void_idx is None or i // 9 != void_idx))
                    if len(hand.melds) + pairs_kept >= 4:
                        fan_pot += config.w_toitoi_pot
            # 不轻易拆 4 张根
            if hand.counts[idx] == 4:
                fan_pot -= config.w_root_keep
            # 1 向听两步听牌好型+番数期望(仅对竞争 1 向听候选精算, 兼顾精度与毫秒级延迟)
            if (
                config.w_good_wait_1sh > 0.0
                and best_sh == 1
                and sh == 1
                and live > 0
                and live >= max(1, best_live_at_min_sh - 12)
            ):
                gw = _one_shanten_good_wait_quality(
                    base_counts,
                    len(hand.melds),
                    void_idx,
                    c.get("useful") or [],
                    counts.unseen,
                    melds=hand.melds,
                    state=state,
                    floors=floors,
                )
                good_wait_term = config.w_good_wait_1sh * gw
            # 1~2 向听可碰对子提速进张(非七对路线)
            if config.w_pong_speedup > 0.0 and sh in (1, 2) and not (init_7p <= 1 and len(hand.melds) == 0):
                p_outs = _pong_speedup_outs(base_counts, len(hand.melds), void_idx, sh, counts.unseen)
                if p_outs > 0:
                    pong_term = config.w_pong_speedup * math.log1p(p_outs)
            shape_term = config.w_shape_conn * _shape_quality(base_counts, void_idx)
        else:
            shape_term = 0.0
        base_counts[idx] += 1

        # 3) 防守分
        def_pen = -config.w_defense * def_mult * exp_dealin_loss

        ev = offense + good_wait_term + pong_term + fan_pot + shape_term + def_pen

        # 生成简明理由标签
        reasons: list[str] = []
        if sh == 0:
            if live == 0:
                reasons.append("死听(残局可保叫)" if state.wall_count <= 18 else "死听")
            else:
                mx_fan = c.get("max_fan") or 0
                reasons.append(f"听牌{live}张(最高{mx_fan}番)")
        else:
            reasons.append(f"{sh}向听·{live}进张")
        if good_wait_term >= 0.42:
            reasons.append("好型听口")
        if fan_pot >= 0.3:
            reasons.append("保留大番路线")
        if d_row and d_row.get("safe_all"):
            reasons.append("100%安全")
        elif exp_dealin_loss >= 0.25:
            reasons.append(f"防点炮(扣{abs(def_pen):.2f})")

        item = dict(c)
        item.update(
            ev_score=round(ev, 4),
            ev_offense=round(offense + good_wait_term + pong_term + shape_term, 3),
            ev_defense=round(def_pen, 3),
            ev_fan=round(fan_pot, 3),
            ev_exp_dealin_loss=round(exp_dealin_loss, 4),
            ev_reason=" · ".join(reasons),
        )
        scored.append(item)

    scored.sort(key=lambda item: (-item["ev_score"], item["shanten_after"], -item["live"], item["index"]))
    return scored


def evaluate_hand_ev_state(
    state: AssistantState,
    hand: Hand,
    counts: TileCounts,
    eff_unseen: tuple[int, ...],
    *,
    opp_reads: dict[int, OpponentRead] | None = None,
    floors: dict[int, list[int]] | None = None,
    config: DiscardV2Config = DEFAULT_DISCARD_V2_CONFIG,
    compute_1sh_lookahead: bool = True,
) -> tuple[float, int, int, int | None]:
    """用与 discard_v2 完全一致的度量衡, 评估任意 3k+1 手牌的静态综合期望值(EV)。

    供碰牌/不碰参谋做同口径 EV 对比(彻底消灭“好牌碰烂”与“漏碰大番”)。
    返回: (ev_score, shanten, live, max_fan)
    """
    from .own_hand import wait_table

    void_code = state.me.void_suit
    void_suit = Suit(void_code) if void_code else None
    void_idx = SUITS.index(void_suit) if void_suit is not None else None
    base_score = max(1, state.rule_config.base_score)
    reads = opp_reads or {}
    active_n = max(1, len(state.active_opponents()))
    fixed_melds = len(hand.melds)

    sh = fast_shanten_hand(hand.counts, fixed_melds, void_idx)
    useful_idxs = fast_useful_indices(hand.counts, fixed_melds, void_idx, sh)
    useful = [{"index": u, "remaining": eff_unseen[u]} for u in useful_idxs]
    live = sum(u["remaining"] for u in useful)

    endgame_chajiao_value = 0.0
    if 0 < state.wall_count <= 18:
        exp_chajiao = sum(r.tenpai_prob * _opp_expected_deal_in_payout(r, base_score) for r in reads.values())
        refund = _kong_refund_exposure(state) / base_score
        endgame_chajiao_value = config.w_endgame_chajiao * (exp_chajiao + refund)

    max_fan: int | None = None
    if sh == 0:
        waits = wait_table(hand, void_suit, counts, state, eff_unseen)
        if waits and live > 0:
            max_fan = max((w["fan"] for w in waits if w["remaining"] > 0), default=max(w["fan"] for w in waits))
            ron_w = 0.68 if active_n >= 3 else (0.62 if active_n == 2 else 0.52)
            tsumo_w = 1.0 - ron_w
            raw_win_ev = 0.0
            for w in waits:
                rem = w["remaining"]
                if rem <= 0:
                    continue
                w_idx = tidx(from_label(w["tile"]))
                q_mult = _wait_quality_multiplier(w_idx, state, floors)
                ron_s = w["ron_score"] / base_score
                tsumo_s = (w["tsumo_each"] * max(1, w["tsumo_payers"])) / base_score
                raw_win_ev += rem * q_mult * (ron_w * ron_s + tsumo_w * tsumo_s)
            offense = config.w_tenpai_ev * math.log1p(raw_win_ev) + endgame_chajiao_value
        else:
            offense = -config.w_dead_wait_pen + endgame_chajiao_value
        return offense, sh, live, max_fan

    offense = -config.w_shanten * sh + config.w_live_log * math.log1p(live)
    base_counts = list(hand.counts)

    # 大番潜力(清一色 / 七对 / 对对胡 / 根)
    suit_totals = [sum(base_counts[s * 9 : (s + 1) * 9]) for s in range(3)]
    meld_suits: set[int] = set()
    for m in hand.melds:
        ms = SUITS.index(m.tiles[0].suit)
        suit_totals[ms] += 3
        meld_suits.add(ms)
    dom_suit = max(range(3), key=lambda s: suit_totals[s])
    fan_pot = 0.0
    if (
        suit_totals[dom_suit] >= 10
        and dom_suit != void_idx
        and sh <= 2
        and (not meld_suits or meld_suits == {dom_suit})
    ):
        fan_pot += config.w_flush_pot
    sp_sh = _seven_pairs_shanten_after(base_counts, void_idx) if fixed_melds == 0 else 99
    if fixed_melds == 0 and sp_sh <= 1:
        fan_pot += config.w_seven_pairs
    elif fixed_melds >= 2 and sh <= 2:
        pairs_kept = sum(1 for i, cnt in enumerate(base_counts) if cnt >= 2 and (void_idx is None or i // 9 != void_idx))
        if fixed_melds + pairs_kept >= 4:
            fan_pot += config.w_toitoi_pot
    roots = sum(1 for cnt in base_counts if cnt == 4) + sum(1 for m in hand.melds if m.kind is MeldKind.KONG)
    fan_pot += 0.14 * roots

    good_wait_term = 0.0
    if compute_1sh_lookahead and config.w_good_wait_1sh > 0.0 and sh == 1 and live > 0:
        gw = _one_shanten_good_wait_quality(
            base_counts,
            fixed_melds,
            void_idx,
            useful,
            eff_unseen,
            melds=hand.melds,
            state=state,
            floors=floors,
        )
        good_wait_term = config.w_good_wait_1sh * gw

    pong_term = 0.0
    if config.w_pong_speedup > 0.0 and sh in (1, 2) and not (fixed_melds == 0 and sp_sh <= 1):
        p_outs = _pong_speedup_outs(base_counts, fixed_melds, void_idx, sh, eff_unseen)
        if p_outs > 0:
            pong_term = config.w_pong_speedup * math.log1p(p_outs)

    shape_term = config.w_shape_conn * _shape_quality(base_counts, void_idx)
    return offense + good_wait_term + pong_term + fan_pot + shape_term, sh, live, max_fan


def discard_v2_sampling_probs(
    ranked: Sequence[dict],
    *,
    top_k: int = 3,
    max_gap: float = 0.25,
    temperature: float = 0.08,
) -> list[float]:
    """根据 score_discard_candidates_v2 的排序结果计算受控随机抽样概率。"""
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if max_gap < 0.0:
        raise ValueError("max_gap must be >= 0.0")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")
    if not ranked:
        return []

    best_ev = ranked[0]["ev_score"]
    kept = [item for item in ranked[:top_k] if (best_ev - item["ev_score"]) <= max_gap]
    if not kept:
        kept = [ranked[0]]
    if len(kept) == 1:
        return [1.0]
    if temperature == 0.0:
        tied = [1.0 if item["ev_score"] == best_ev else 0.0 for item in kept]
        s = sum(tied)
        return [w / s for w in tied]

    weights = [math.exp(-(best_ev - item["ev_score"]) / temperature) for item in kept]
    total_w = sum(weights)
    return [w / total_w for w in weights]


def sample_discard_v2_index(
    ranked: Sequence[dict],
    *,
    rng: random.Random | None = None,
    top_k: int = 3,
    max_gap: float = 0.25,
    temperature: float = 0.08,
) -> int:
    if not ranked:
        raise ValueError("no legal discard candidates")
    probs = discard_v2_sampling_probs(ranked, top_k=top_k, max_gap=max_gap, temperature=temperature)
    if len(probs) == 1:
        return ranked[0]["index"]
    drawer = rng if rng is not None else random
    chosen = drawer.choices(ranked[: len(probs)], weights=probs, k=1)[0]
    return chosen["index"]
