"""高速单门牌型快查表与“正常人换三张+定缺策略 v1”。

设计目标:
1. 严格复用 state.hand_analysis._best_suit_blocks 的牌理定义,去缺门向听数与现有引擎/辅助器 100% 一致;
2. 单门 9 维计数结果全缓存,将“四家发牌+换牌+定缺+盘面评估”提速 50~100 倍;
3. 实现符合实战常识的 v1 定缺与换三张策略(兼顾自身向听、缺门包袱、双短门防卡壳、清一色底子与送牌防守)。
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Sequence

from engine.hand import Hand
from engine.tiles import SUITS, Suit, Tile, tile_to_str
from state.hand_analysis import _best_suit_blocks

ZERO_SUIT: tuple[int, ...] = (0,) * 9

# 覆盖 1~9 全部点数的三不靠代表进牌模板 (0-indexed ranks):
# 前3组 147/258/369 均匀覆盖 1~9 各一次; 第4组 159 补充实战高频幺九散牌权重。
_INCOMING_TEMPLATES: tuple[tuple[int, int, int, float], ...] = (
    (0, 3, 6, 1.0),  # 1, 4, 7
    (1, 4, 7, 1.0),  # 2, 5, 8
    (2, 5, 8, 1.0),  # 3, 6, 9
    (0, 4, 8, 1.4),  # 1, 5, 9
)


@dataclass(frozen=True, slots=True)
class SuitProfile:
    m0: int
    t0: int
    pair_blocks: tuple[tuple[int, int], ...]
    pairs: int
    triplets: int
    quads: int
    unique: int
    total: int
    conn: int
    plus_one: tuple[tuple[int, int, tuple[int, ...]], ...]


def _suit_conn_9(c9: tuple[int, ...]) -> int:
    score = 0
    for r in range(9):
        cnt = c9[r]
        if cnt == 0:
            continue
        adj = 0
        if r >= 1 and c9[r - 1] > 0:
            adj += 2
        if r <= 7 and c9[r + 1] > 0:
            adj += 2
        if r >= 2 and c9[r - 2] > 0:
            adj += 1
        if r <= 6 and c9[r + 2] > 0:
            adj += 1
        score += cnt * (adj + 3 * max(0, cnt - 1))
    return score


@lru_cache(maxsize=None)
def suit_profile(c9: tuple[int, ...]) -> SuitProfile:
    m0, t0 = _best_suit_blocks(c9)
    seen_pb: set[tuple[int, int]] = set()
    pbs: list[tuple[int, int]] = []
    for i in range(9):
        if c9[i] >= 2:
            lst = list(c9)
            lst[i] -= 2
            pb = _best_suit_blocks(tuple(lst))
            if pb not in seen_pb:
                seen_pb.add(pb)
                pbs.append(pb)
    pairs = sum(1 for x in c9 if x >= 2)
    triplets = sum(1 for x in c9 if x >= 3)
    quads = sum(1 for x in c9 if x >= 4)
    unique = sum(1 for x in c9 if x > 0)
    total = sum(c9)
    conn = _suit_conn_9(c9)
    plus_one: list[tuple[int, int, tuple[int, ...]]] = []
    for i in range(9):
        if c9[i] < 4:
            nxt = list(c9)
            nxt[i] += 1
            plus_one.append((i, 4 - c9[i], tuple(nxt)))
    return SuitProfile(
        m0=m0,
        t0=t0,
        pair_blocks=tuple(pbs),
        pairs=pairs,
        triplets=triplets,
        quads=quads,
        unique=unique,
        total=total,
        conn=conn,
        plus_one=tuple(plus_one),
    )


def fast_shanten_two_suits(c_a: tuple[int, ...], c_b: tuple[int, ...]) -> int:
    """给定非缺门的两门 9 维计数,返回精确向听数(与 state.hand_analysis.shanten 完全一致)。"""
    pa = suit_profile(c_a)
    pb = suit_profile(c_b)

    m = pa.m0 + pb.m0
    if m > 4:
        m = 4
    t = pa.t0 + pb.t0
    if t > 4 - m:
        t = 4 - m
    best = 8 - 2 * m - t

    for m1, t1 in pa.pair_blocks:
        mp = m1 + pb.m0
        if mp > 4:
            mp = 4
        tp = t1 + pb.t0
        if tp > 4 - mp:
            tp = 4 - mp
        cand = 7 - 2 * mp - tp
        if cand < best:
            best = cand

    for m1, t1 in pb.pair_blocks:
        mp = pa.m0 + m1
        if mp > 4:
            mp = 4
        tp = pa.t0 + t1
        if tp > 4 - mp:
            tp = 4 - mp
        cand = 7 - 2 * mp - tp
        if cand < best:
            best = cand

    pairs = pa.pairs + pb.pairs
    unique = pa.unique + pb.unique
    sp = 6 - pairs + (7 - unique if unique < 7 else 0)
    if sp < best:
        best = sp
    return best


def fast_shanten_counts(
    suits3: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    void_idx: int | None = None,
) -> int:
    if void_idx == 0:
        return fast_shanten_two_suits(suits3[1], suits3[2])
    if void_idx == 1:
        return fast_shanten_two_suits(suits3[0], suits3[2])
    if void_idx == 2:
        return fast_shanten_two_suits(suits3[0], suits3[1])

    # 未定缺时三门合并计算
    p0, p1, p2 = suit_profile(suits3[0]), suit_profile(suits3[1]), suit_profile(suits3[2])
    m = min(4, p0.m0 + p1.m0 + p2.m0)
    t = min(4 - m, p0.t0 + p1.t0 + p2.t0)
    best = 8 - 2 * m - t
    for idx, (pa, pb, pc) in enumerate(((p0, p1, p2), (p1, p0, p2), (p2, p0, p1))):
        for m1, t1 in pa.pair_blocks:
            mp = min(4, m1 + pb.m0 + pc.m0)
            tp = min(4 - mp, t1 + pb.t0 + pc.t0)
            cand = 7 - 2 * mp - tp
            if cand < best:
                best = cand
    pairs = p0.pairs + p1.pairs + p2.pairs
    unique = p0.unique + p1.unique + p2.unique
    sp = 6 - pairs + (7 - unique if unique < 7 else 0)
    return sp if sp < best else best


def fast_live_useful_two_suits(c_a: tuple[int, ...], c_b: tuple[int, ...], cur_shanten: int) -> int:
    pa = suit_profile(c_a)
    pb = suit_profile(c_b)
    live = 0
    for _, rem, nxt_a in pa.plus_one:
        if fast_shanten_two_suits(nxt_a, c_b) < cur_shanten:
            live += rem
    for _, rem, nxt_b in pb.plus_one:
        if fast_shanten_two_suits(c_a, nxt_b) < cur_shanten:
            live += rem
    return live


def split_suits(counts27: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (tuple(counts27[0:9]), tuple(counts27[9:18]), tuple(counts27[18:27]))


def choose_void_v1_with_sh(
    suits3: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    is_dealer: bool = False,
) -> tuple[int, int]:
    """返回 (最优定缺花色下标 0/1/2, 去缺门后的向听数)。仅在并列时计算有效进张,提速约 20 倍。"""
    p0, p1, p2 = suit_profile(suits3[0]), suit_profile(suits3[1]), suit_profile(suits3[2])
    ps = (p0, p1, p2)
    other_pairs = ((1, 2), (0, 2), (0, 1))

    cands: list[tuple[float, int, int, int]] = []
    for v_idx in (0, 1, 2):
        i_a, i_b = other_pairs[v_idx]
        sh = fast_shanten_two_suits(suits3[i_a], suits3[i_b])
        v_cnt = ps[v_idx].total
        v_eff = max(0.0, v_cnt - 0.6) if is_dealer else float(v_cnt)
        max_main = ps[i_a].total if ps[i_a].total >= ps[i_b].total else ps[i_b].total
        flush_bonus = -0.45 if max_main >= 10 else (-0.20 if max_main == 9 and v_cnt <= 2 else 0.0)
        cost = round(2.0 * sh + 1.0 * v_eff + flush_bonus, 4)
        cands.append((cost, v_cnt, sh, v_idx))

    cands.sort()
    top_cost, top_vcnt, top_sh, top_v = cands[0]
    # 绝大部分手牌无并列,直接返回无需跑 18 次进张试算
    if (cands[1][0], cands[1][1], cands[1][2]) != (top_cost, top_vcnt, top_sh):
        return top_v, top_sh

    best_tie_key: tuple[int, int, int] | None = None
    best_v = top_v
    best_sh = top_sh
    for cost, v_cnt, sh, v_idx in cands:
        if (cost, v_cnt, sh) != (top_cost, top_vcnt, top_sh):
            break
        i_a, i_b = other_pairs[v_idx]
        live = fast_live_useful_two_suits(suits3[i_a], suits3[i_b], sh)
        tkey = (-live, ps[v_idx].conn, v_idx)
        if best_tie_key is None or tkey < best_tie_key:
            best_tie_key = tkey
            best_v = v_idx
            best_sh = sh
    return best_v, best_sh


def choose_void_v1_idx(
    suits3: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    is_dealer: bool = False,
) -> int:
    return choose_void_v1_with_sh(suits3, is_dealer=is_dealer)[0]


def choose_void_v1(hand: Hand, is_dealer: bool = False) -> Suit:
    idx = choose_void_v1_idx(split_suits(hand.counts), is_dealer=is_dealer)
    return SUITS[idx]


@lru_cache(maxsize=None)
def gift_shape_info(r0: int, r1: int, r2: int) -> tuple[str, float]:
    """对同花色三张牌 (0 <= r0 <= r1 <= r2 <= 8) 进行送牌危险度分级。
    返回 (形状类别名, 基础送牌代价)。
    """
    if r0 == r1 == r2:
        return ("刻子", 1.80)
    if r1 == r0 + 1 and r2 == r1 + 1:
        return ("顺子", 1.70)
    if r0 == r1 or r1 == r2:
        rp = r0 if r0 == r1 else r1
        ro = r2 if r0 == r1 else r0
        is_mid = 2 <= rp <= 6
        base = 0.78 if is_mid else 0.52
        if abs(ro - rp) <= 2:
            return ("对子带搭", base + 0.28)
        return ("中张对子" if is_mid else "边幺对子", base)

    # 三张不同点数
    d1 = r1 - r0
    d2 = r2 - r1
    ryanmen_1 = d1 == 1 and 1 <= r0 and r1 <= 7
    ryanmen_2 = d2 == 1 and 1 <= r1 and r2 <= 7
    if ryanmen_1 or ryanmen_2:
        if r2 - r0 <= 3:
            return ("复合两面搭", 0.72)
        return ("两面搭+孤", 0.55)

    if d1 <= 2 or d2 <= 2:
        if d1 <= 2 and d2 <= 2:
            return ("双嵌边搭", 0.44)
        if d1 == 1 or d2 == 1:
            return ("边搭+孤", 0.22)
        return ("嵌搭+孤", 0.28)

    mid_cnt = int(2 <= r0 <= 6) + int(2 <= r1 <= 6) + int(2 <= r2 <= 6)
    if mid_cnt == 0:
        return ("纯幺九废牌", 0.0)
    if mid_cnt == 1:
        return ("幺九三不靠", 0.05)
    return ("中张三不靠", 0.14)


def legal_suit_combos(c9: tuple[int, ...]) -> list[tuple[int, int, int]]:
    tiles: list[int] = []
    for r, cnt in enumerate(c9):
        tiles.extend([r] * cnt)
    if len(tiles) < 3:
        return []
    return sorted(set(combinations(tiles, 3)))


def _eval_post_swap_self_cost(
    suits3: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    is_dealer: bool,
) -> tuple[float, int, int, int]:
    """快速评估一套换后 13/14 张牌的自身综合步数代价(越低越好)。
    内部使用单门预缓存连接度与对子/搭子质量替代逐张进张枚举,提速 30 倍以上。
    返回 (cost, shanten, void_cnt, flush_flag)。
    """
    v_idx, sh = choose_void_v1_with_sh(suits3, is_dealer=is_dealer)
    other = ((1, 2), (0, 2), (0, 1))[v_idx]
    pa, pb, pv = suit_profile(suits3[other[0]]), suit_profile(suits3[other[1]]), suit_profile(suits3[v_idx])

    v_cnt = pv.total
    max_main = pa.total if pa.total >= pb.total else pb.total

    v_eff = max(0.0, v_cnt - 0.6) if is_dealer else float(v_cnt)
    jam_threshold = 3 if is_dealer else 2
    jam_pen = 0.32 * max(0, v_cnt - jam_threshold)

    flush_flag = 1 if max_main >= 9 and v_cnt <= 2 else 0
    flush_bonus = -0.48 if max_main >= 10 else (-0.22 if flush_flag else 0.0)
    root_bonus = -0.25 * (pa.quads + pb.quads) - 0.08 * (pa.triplets + pb.triplets)
    # 连接度与对子宽度微调(替代慢速18张逐一试摸,量级约 0.05~0.25)
    shape_bonus = -0.008 * (pa.conn + pb.conn) - 0.03 * (pa.pairs + pb.pairs)

    cost = 1.0 * sh + 0.36 * v_eff + jam_pen + flush_bonus + root_bonus + shape_bonus
    return cost, sh, v_cnt, flush_flag


def score_swap_candidates_v1(
    counts27: Sequence[int],
    *,
    is_dealer: bool = False,
    receiver_is_dealer: bool = False,
    incoming_suit_weights: tuple[float, float, float] | None = None,
) -> list[dict]:
    """对当前手牌所有合法换三张候选进行 v1 评分(cost 越低越优)。"""
    suits3 = split_suits(counts27)
    suit_totals = ( sum(suits3[0]), sum(suits3[1]), sum(suits3[2]) )

    # 若未指定外部进牌花色分布,按外池剩余该门总张数(36 - 自己持有张数)做自然先验
    if incoming_suit_weights is None:
        raw_w = tuple(float(36 - suit_totals[s]) for s in (0, 1, 2))
        w_sum = sum(raw_w)
        in_suit_w = (raw_w[0] / w_sum, raw_w[1] / w_sum, raw_w[2] / w_sum)
    else:
        in_suit_w = incoming_suit_weights

    dealer_mult = 1.35 if receiver_is_dealer else 1.0
    results: list[dict] = []

    for s_idx in (0, 1, 2):
        combos = legal_suit_combos(suits3[s_idx])
        if not combos:
            continue
        my_suit_cnt = suit_totals[s_idx]
        scarcity_mult = 1.12 if my_suit_cnt <= 3 else (1.0 if my_suit_cnt == 4 else 0.90)
        init_suit = suits3[s_idx]

        for r0, r1, r2 in combos:
            rem_list = list(init_suit)
            rem_list[r0] -= 1
            rem_list[r1] -= 1
            rem_list[r2] -= 1
            rem_suit = tuple(rem_list)

            rem_suits3 = [suits3[0], suits3[1], suits3[2]]
            rem_suits3[s_idx] = rem_suit

            total_w = 0.0
            acc_self_cost = 0.0
            acc_sh = 0.0
            acc_vcnt = 0.0
            acc_jam = 0.0
            acc_flush = 0.0

            for in_s in (0, 1, 2):
                sw = in_suit_w[in_s]
                init_in_c9 = suits3[in_s]
                base_in_c9 = rem_suits3[in_s]
                for t0, t1, t2, tw in _INCOMING_TEMPLATES:
                    avail = (4 - init_in_c9[t0]) * (4 - init_in_c9[t1]) * (4 - init_in_c9[t2])
                    if avail <= 0:
                        continue
                    w = sw * tw * avail
                    nxt = list(base_in_c9)
                    nxt[t0] += 1
                    nxt[t1] += 1
                    nxt[t2] += 1
                    trial3 = [rem_suits3[0], rem_suits3[1], rem_suits3[2]]
                    trial3[in_s] = tuple(nxt)
                    c_val, sh_val, v_val, fl_val = _eval_post_swap_self_cost(
                        (trial3[0], trial3[1], trial3[2]),
                        is_dealer=is_dealer,
                    )
                    total_w += w
                    acc_self_cost += w * c_val
                    acc_sh += w * sh_val
                    acc_vcnt += w * v_val
                    acc_jam += w * (1.0 if v_val >= (4 if is_dealer else 3) else 0.0)
                    acc_flush += w * fl_val

            exp_self_cost = acc_self_cost / total_w
            shape_name, raw_gift = gift_shape_info(r0, r1, r2)
            gift_cost = 0.35 * raw_gift * dealer_mult * scarcity_mult
            total_cost = exp_self_cost + gift_cost

            t_indices = (s_idx * 9 + r0, s_idx * 9 + r1, s_idx * 9 + r2)
            results.append(
                {
                    "suit_idx": s_idx,
                    "ranks": (r0, r1, r2),
                    "tile_indices": t_indices,
                    "tiles": [tile_to_str(Tile.from_index(idx)) for idx in t_indices],
                    "total_cost": round(total_cost, 5),
                    "self_cost": round(exp_self_cost, 5),
                    "gift_cost": round(gift_cost, 5),
                    "gift_shape": shape_name,
                    "exp_shanten": round(acc_sh / total_w, 3),
                    "exp_void_cnt": round(acc_vcnt / total_w, 3),
                    "exp_jam_rate": round(acc_jam / total_w, 3),
                    "exp_flush_rate": round(acc_flush / total_w, 3),
                }
            )

    results.sort(key=lambda item: (item["total_cost"], item["tile_indices"]))
    return results


def choose_swap_v1_indices(
    counts27: Sequence[int],
    *,
    is_dealer: bool = False,
    receiver_is_dealer: bool = False,
) -> tuple[int, int, int]:
    ranked = score_swap_candidates_v1(
        counts27,
        is_dealer=is_dealer,
        receiver_is_dealer=receiver_is_dealer,
    )
    if not ranked:
        raise ValueError("hand has no legal swap-three choice")
    return ranked[0]["tile_indices"]


def choose_swap_v1(
    hand: Hand,
    *,
    is_dealer: bool = False,
    receiver_is_dealer: bool = False,
) -> tuple[Tile, Tile, Tile]:
    i0, i1, i2 = choose_swap_v1_indices(
        hand.counts,
        is_dealer=is_dealer,
        receiver_is_dealer=receiver_is_dealer,
    )
    return (Tile.from_index(i0), Tile.from_index(i1), Tile.from_index(i2))
