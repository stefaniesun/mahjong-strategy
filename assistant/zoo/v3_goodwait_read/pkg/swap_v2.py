"""基于 280 万次四家发牌大样本校准的换三张与定缺策略 (swap_v2)。

相对 swap_v1 的四大核心升级(全部源自 outputs/swap_v1_eval_large.json 实证数据):
1. 进牌花色“反向虹吸”真实先验: 我方持有张数越多(外池人均越少),对手越容易将其作为短门换出(1张 29.4% -> 7张 35.8%);
2. 真实对手进牌牌型模板: 融合三不靠/嵌边搭(约68%)、两面搭(约18%)与对子(约14%)的真实换出分布;
3. 精确缓存有效进张(_cached_live): 消除 v1 连接度近似对“445粘连对子/暗刻带搭”的重复加分;
4. 校准一打三送牌权重与庄家首打清1张缺门红利: 防止为了不送对子而自拆顺子或多背废牌,补齐庄家剩1张缺门的开局清零优势。
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from typing import Sequence

from .swap_v1 import (
    fast_live_useful_two_suits,
    fast_shanten_counts,
    fast_shanten_two_suits,
    gift_shape_info,
    legal_suit_combos,
    split_suits,
    suit_profile,
)
from engine.hand import Hand
from engine.tiles import SUITS, Suit, Tile, tile_to_str

# 280 万局实测: 我方某门初始张数 k (0..9+) 对应的对手换入该花色相对倾向
_EMPIRICAL_IN_SUIT_PROPENSITY: tuple[float, ...] = (
    0.280,  # 0 张
    0.294,  # 1 张
    0.308,  # 2 张
    0.319,  # 3 张
    0.331,  # 4 张
    0.342,  # 5 张
    0.350,  # 6 张
    0.358,  # 7 张
    0.365,  # 8 张
    0.372,  # >=9 张
)

# 12 组完全镜像对称(1<->9, 2<->8, 3<->7, 4<->6)的真实进牌模板:
# 覆盖三不靠、嵌搭+孤、边搭+孤、两面+孤, 消除纯隔张模板对边幺牌的虚假对子偏置
_V2_INCOMING_TEMPLATES: tuple[tuple[int, int, int, float], ...] = (
    (0, 3, 6, 1.00),  # 1,4,7
    (2, 5, 8, 1.00),  # 3,6,9
    (1, 4, 7, 1.00),  # 2,5,8
    (0, 4, 8, 1.10),  # 1,5,9
    (0, 2, 7, 0.70),  # 1,3,8
    (1, 6, 8, 0.70),  # 2,7,9
    (1, 3, 8, 0.70),  # 2,4,9
    (0, 5, 7, 0.70),  # 1,6,8
    (0, 1, 6, 0.65),  # 1,2,7
    (2, 7, 8, 0.65),  # 3,8,9
    (1, 2, 7, 0.55),  # 2,3,8
    (1, 6, 7, 0.55),  # 2,7,8
)


@lru_cache(maxsize=1 << 17)
def cached_live_two_suits(ca: tuple[int, ...], cb: tuple[int, ...], sh: int) -> int:
    return fast_live_useful_two_suits(ca, cb, sh)


def void_burden_cost(v_cnt: int, is_dealer: bool) -> float:
    """缺门废牌张数的阶梯步数代价(按完整牌局实战节奏标定):
    - 庄家14张首打可直接扔掉1张缺门, 1张近乎无损(0.08), 2~4张约 0.40/0.74/1.08;
    - 闲家13张 1~4张约 0.24/0.52/0.84/1.16(打缺门期间仍可正常摸牌与碰牌, 避免高估4缺门导致强拆双对好牌)。
    """
    if v_cnt <= 0:
        return 0.0
    if is_dealer:
        table = (0.0, 0.08, 0.40, 0.74, 1.08)
        return table[v_cnt] if v_cnt < len(table) else 1.08 + 0.38 * (v_cnt - 4)
    table = (0.0, 0.24, 0.52, 0.84, 1.16)
    return table[v_cnt] if v_cnt < len(table) else 1.16 + 0.38 * (v_cnt - 4)


def choose_void_v2_with_sh(
    suits3: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    is_dealer: bool = False,
) -> tuple[int, int]:
    """定缺策略 v2: 返回 (最优定缺花色下标 0/1/2, 去缺门后的向听数)。"""
    p0, p1, p2 = suit_profile(suits3[0]), suit_profile(suits3[1]), suit_profile(suits3[2])
    ps = (p0, p1, p2)
    other_pairs = ((1, 2), (0, 2), (0, 1))

    cands: list[tuple[float, int, int, int]] = []
    for v_idx in (0, 1, 2):
        i_a, i_b = other_pairs[v_idx]
        sh = fast_shanten_two_suits(suits3[i_a], suits3[i_b])
        v_cnt = ps[v_idx].total
        max_main = ps[i_a].total if ps[i_a].total >= ps[i_b].total else ps[i_b].total
        flush_bonus = -0.32 if max_main >= 10 and v_cnt <= 2 else (-0.15 if max_main == 9 and v_cnt <= 2 else 0.0)
        trips = ps[i_a].triplets + ps[i_b].triplets
        quads = ps[i_a].quads + ps[i_b].quads
        meld_bonus = -0.55 * (trips - quads) - 0.85 * quads
        # 1步向听对应 1.0 代价, void_burden_cost 已按同尺度标定
        cost = round(1.0 * sh + void_burden_cost(v_cnt, is_dealer) + flush_bonus + meld_bonus, 4)
        cands.append((cost, v_cnt, sh, v_idx))

    cands.sort()
    top_cost, top_vcnt, top_sh, top_v = cands[0]
    if (cands[1][0], cands[1][1], cands[1][2]) != (top_cost, top_vcnt, top_sh):
        return top_v, top_sh

    # 仅在完全并列时用精确有效进张决胜, 避免内层循环重复扫描 18 张牌
    best_tie_key: tuple[int, int, int] | None = None
    best_v = top_v
    best_sh = top_sh
    for cost, v_cnt, sh, v_idx in cands:
        if (cost, v_cnt, sh) != (top_cost, top_vcnt, top_sh):
            break
        i_a, i_b = other_pairs[v_idx]
        live = cached_live_two_suits(suits3[i_a], suits3[i_b], sh)
        tkey = (-live, ps[v_idx].conn, v_idx)
        if best_tie_key is None or tkey < best_tie_key:
            best_tie_key = tkey
            best_v = v_idx
            best_sh = sh
    return best_v, best_sh


def choose_void_v2_idx(
    suits3: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    is_dealer: bool = False,
) -> int:
    return choose_void_v2_with_sh(suits3, is_dealer=is_dealer)[0]


def choose_void_v2(hand: Hand, is_dealer: bool = False) -> Suit:
    return SUITS[choose_void_v2_idx(split_suits(hand.counts), is_dealer=is_dealer)]


def _eval_post_swap_self_cost_v2(
    suits3: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    is_dealer: bool,
    *,
    init_trip_ranks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = ((), (), ()),
    exact_live: bool = True,
) -> tuple[float, int, int, int, int]:
    """评估换后 13/14 张牌的自身综合步数代价(越小越优)。
    exact_live=True 时计算精确有效进张张数(单手仅需 ~3ms, 供玩家推荐与评测使用);
    exact_live=False 时供大批量模拟中的三家陪练快速选牌(~0.15ms)。
    返回 (self_cost, shanten, void_cnt, live, flush_flag)。
    """
    v_idx, sh = choose_void_v2_with_sh(suits3, is_dealer=is_dealer)
    other = ((1, 2), (0, 2), (0, 1))[v_idx]
    ca, cb = suits3[other[0]], suits3[other[1]]
    pa, pb, pv = suit_profile(ca), suit_profile(cb), suit_profile(suits3[v_idx])

    v_cnt = pv.total
    max_main = pa.total if pa.total >= pb.total else pb.total
    sec_main = pb.total if pa.total >= pb.total else pa.total

    v_cost = void_burden_cost(v_cnt, is_dealer)
    flush_flag = 1 if max_main >= 9 and v_cnt <= (3 if is_dealer else 2) else 0
    # 清一色潜力按“杂牌总张数(缺门+副门)”区分纯度, 防止双副门散牌时高估 10 张清一色
    if max_main >= 10:
        flush_bonus = -0.32 if (v_cnt + sec_main) <= 2 else -0.18
    elif flush_flag:
        flush_bonus = -0.14 if (v_cnt + sec_main) <= 3 else -0.07
    else:
        flush_bonus = 0.0
    kept_init_trips = 0
    for si, c_suit in ((other[0], ca), (other[1], cb)):
        for rr in init_trip_ranks[si]:
            if c_suit[rr] >= 3:
                kept_init_trips += 1
    all_trips = pa.triplets + pb.triplets
    all_quads = pa.quads + pb.quads
    new_trips = max(0, all_trips - all_quads - kept_init_trips)
    # 四川血战起手暗刻含金量极高(锁定面子 + 根翻倍潜力 + 暗杠6分/明杠2分), 保住初始暗刻给高权重
    root_bonus = -0.85 * all_quads - 1.05 * kept_init_trips - 0.25 * new_trips
    # 四川麻将可碰对子, 且真将牌不占用面子位, 多对子具有碰牌提速与成杠/带根翻倍价值
    npairs = pa.pairs + pb.pairs
    pair_bonus = (
        -0.16 if npairs == 1
        else (-0.33 if npairs == 2
              else (-0.47 if npairs == 3
                    else (-0.57 if npairs >= 4 else 0.0)))
    )

    if exact_live:
        live = cached_live_two_suits(ca, cb, sh)
        live_term = -0.018 * live
    else:
        live = 0
        live_term = -0.005 * (pa.conn + pb.conn)

    cost = 1.0 * sh + v_cost + live_term + flush_bonus + root_bonus + pair_bonus
    return cost, sh, v_cnt, live, flush_flag


def empirical_incoming_suit_weights(suit_totals: tuple[int, int, int]) -> tuple[float, float, float]:
    raw = tuple(_EMPIRICAL_IN_SUIT_PROPENSITY[min(9, c)] for c in suit_totals)
    s = raw[0] + raw[1] + raw[2]
    return (raw[0] / s, raw[1] / s, raw[2] / s)


def _template_avail(c9: tuple[int, ...], r0: int, r1: int, r2: int) -> float:
    """计算外池剩余牌能凑出模板 (r0, r1, r2) 的相对组合可用度。"""
    if r0 == r1:  # (r0, r0, r2) 对子模板
        rem_pair = 4 - c9[r0]
        rem_single = 4 - c9[r2]
        if rem_pair < 2 or rem_single < 1:
            return 0.0
        # C(rem_pair, 2) * rem_single, 归一化使满额 C(4,2)*4=24 与单张 4*4*4=64 同量级
        return (rem_pair * (rem_pair - 1) * 0.5 * rem_single) * (64.0 / 24.0)
    if r1 == r2:  # (r0, r1, r1) 对子模板
        rem_pair = 4 - c9[r1]
        rem_single = 4 - c9[r0]
        if rem_pair < 2 or rem_single < 1:
            return 0.0
        return (rem_pair * (rem_pair - 1) * 0.5 * rem_single) * (64.0 / 24.0)
    a0, a1, a2 = 4 - c9[r0], 4 - c9[r1], 4 - c9[r2]
    if a0 <= 0 or a1 <= 0 or a2 <= 0:
        return 0.0
    return float(a0 * a1 * a2)


def score_swap_candidates_v2(
    counts27: Sequence[int],
    *,
    is_dealer: bool = False,
    receiver_is_dealer: bool = False,
    swap_direction: int | None = None,
    exact_live: bool = True,
) -> list[dict]:
    """对当前手牌所有合法换三张候选进行 v2 综合评分(total_cost 越小越优)。

    采用两阶段评估: 先对全部候选做毫秒级骨架评估, 再对各门前茅与全局前 10 候选做精确进张重排,
    兼顾全精度与首次冷启动 <40ms 响应。
    """
    suits3 = split_suits(counts27)
    suit_profiles_init = (suit_profile(suits3[0]), suit_profile(suits3[1]), suit_profile(suits3[2]))
    suit_totals = (suit_profiles_init[0].total, suit_profiles_init[1].total, suit_profiles_init[2].total)
    max_suit_cnt = max(suit_totals)
    in_suit_w = empirical_incoming_suit_weights(suit_totals)
    init_trip_ranks = tuple(tuple(r for r, c in enumerate(suits3[si]) if c >= 3) for si in (0, 1, 2))

    dealer_mult = 1.45 if receiver_is_dealer else 1.0
    dir_mult = 0.94 if swap_direction == 2 else 1.0

    def _eval_combo(s_idx: int, r0: int, r1: int, r2: int, use_exact: bool) -> dict:
        init_sp = suit_profiles_init[s_idx]
        my_suit_cnt = init_sp.total
        scarcity_mult = 1.12 if my_suit_cnt <= 3 else (1.0 if my_suit_cnt == 4 else 0.88)
        init_suit = suits3[s_idx]

        rem_list = list(init_suit)
        rem_list[r0] -= 1
        rem_list[r1] -= 1
        rem_list[r2] -= 1
        rem_suit = tuple(rem_list)
        rem_sp = suit_profile(rem_suit)
        breaks_pair = rem_sp.pairs < init_sp.pairs
        # 存在 6+ 张长门可健康出牌时, 避免强行拆碎 4/5 张门里的双对子根基
        pair_shatter_pen = (
            0.40
            if (
                max_suit_cnt >= 6
                and my_suit_cnt in (4, 5)
                and init_sp.pairs >= 2
                and not (is_dealer and max_suit_cnt >= 7)
            )
            else 0.0
        )

        rem_suits3 = (
            rem_suit if s_idx == 0 else suits3[0],
            rem_suit if s_idx == 1 else suits3[1],
            rem_suit if s_idx == 2 else suits3[2],
        )
        sh_rest = fast_shanten_counts(rem_suits3, None)
        suit_left = my_suit_cnt - 3
        rem_v, _ = choose_void_v2_with_sh(rem_suits3, is_dealer=is_dealer)
        rem_other = ((1, 2), (0, 2), (0, 1))[rem_v]
        rem_pairs = suit_profile(rem_suits3[rem_other[0]]).pairs + suit_profile(rem_suits3[rem_other[1]]).pairs
        rem_no_pair_pen = 0.05 if rem_pairs == 0 else 0.0

        total_w = 0.0
        acc_self_cost = 0.0
        acc_sh = 0.0
        acc_vcnt = 0.0
        acc_live = 0.0
        acc_jam = 0.0
        acc_flush = 0.0

        templates = _V2_INCOMING_TEMPLATES if use_exact else _V2_INCOMING_TEMPLATES[:4]
        for in_s in (0, 1, 2):
            sw = in_suit_w[in_s]
            init_in_c9 = suits3[in_s]
            base_in_c9 = rem_suits3[in_s]
            for t0, t1, t2, tw in templates:
                avail = _template_avail(init_in_c9, t0, t1, t2)
                if avail <= 0.0:
                    continue
                w = sw * tw * avail
                nxt = list(base_in_c9)
                nxt[t0] += 1
                nxt[t1] += 1
                nxt[t2] += 1
                nxt_t = tuple(nxt)
                trial3 = (
                    nxt_t if in_s == 0 else rem_suits3[0],
                    nxt_t if in_s == 1 else rem_suits3[1],
                    nxt_t if in_s == 2 else rem_suits3[2],
                )
                c_val, sh_val, v_val, lv_val, fl_val = _eval_post_swap_self_cost_v2(
                    trial3,
                    is_dealer=is_dealer,
                    init_trip_ranks=init_trip_ranks,
                    exact_live=use_exact,
                )
                total_w += w
                acc_self_cost += w * c_val
                acc_sh += w * sh_val
                acc_vcnt += w * v_val
                acc_live += w * lv_val
                acc_jam += w * (1.0 if v_val >= (4 if is_dealer else 3) else 0.0)
                acc_flush += w * fl_val

        exp_self_cost = acc_self_cost / total_w + rem_no_pair_pen + pair_shatter_pen
        shape_name, raw_gift = gift_shape_info(r0, r1, r2)
        # 双嵌边搭(如 246/357)提供两个相邻嵌张口(3/5), 实战喂搭能力接近两面搭+孤
        eff_raw_gift = raw_gift + (0.11 if shape_name == "双嵌边搭" else 0.0)
        lane_risk_pen = 0.03 if (my_suit_cnt == 3 and shape_name in ("中张对子", "对子带搭")) else 0.0
        pair_gift_pen = (
            0.08
            if (breaks_pair and suit_left in (1, 2, 3) and init_sp.pairs <= 2 and shape_name in ("中张对子", "边幺对子", "对子带搭"))
            else 0.0
        )
        gift_cost = 0.24 * eff_raw_gift * dealer_mult * dir_mult * scarcity_mult + lane_risk_pen + pair_gift_pen
        total_cost = exp_self_cost + gift_cost

        t_indices = (s_idx * 9 + r0, s_idx * 9 + r1, s_idx * 9 + r2)
        return {
            "suit_idx": s_idx,
            "suit_code": SUITS[s_idx].value,
            "ranks": (r0, r1, r2),
            "tile_indices": t_indices,
            "tiles": [tile_to_str(Tile.from_index(idx)) for idx in t_indices],
            "shanten_rest": sh_rest,
            "suit_left": suit_left,
            "total_cost": round(total_cost, 5),
            "self_cost": round(exp_self_cost, 5),
            "gift_cost": round(gift_cost, 5),
            "gift_shape": shape_name,
            "exp_shanten": round(acc_sh / total_w, 2),
            "exp_void_cnt": round(acc_vcnt / total_w, 2),
            "exp_live": round(acc_live / total_w, 1),
            "exp_jam_rate": round(acc_jam / total_w, 3),
            "exp_flush_rate": round(acc_flush / total_w, 3),
        }

    # 第一遍: 骨架粗排 (~0.2ms)
    coarse: list[dict] = []
    for s_idx in (0, 1, 2):
        for r0, r1, r2 in legal_suit_combos(suits3[s_idx]):
            coarse.append(_eval_combo(s_idx, r0, r1, r2, False))

    coarse.sort(key=lambda item: (item["total_cost"], item["shanten_rest"], item["suit_left"], item["tile_indices"]))
    if not exact_live or not coarse:
        return coarse

    # 第二遍: 对全局前 8 名及每门花色前 3 名做精确有效进张与 12 模板重算, 其余候选对齐基准偏移
    refine_keys: set[tuple[int, int, int]] = {item["tile_indices"] for item in coarse[:8]}
    for s_idx in (0, 1, 2):
        suit_cands = [item for item in coarse if item["suit_idx"] == s_idx]
        for item in suit_cands[:3]:
            refine_keys.add(item["tile_indices"])

    refined_map: dict[tuple[int, int, int], dict] = {}
    shifts: list[float] = []
    for item in coarse:
        key = item["tile_indices"]
        if key in refine_keys:
            r0, r1, r2 = item["ranks"]
            exact_item = _eval_combo(item["suit_idx"], r0, r1, r2, True)
            refined_map[key] = exact_item
            shifts.append(exact_item["total_cost"] - item["total_cost"])

    avg_shift = (sum(shifts) / len(shifts)) if shifts else 0.0
    results: list[dict] = []
    for item in coarse:
        key = item["tile_indices"]
        if key in refined_map:
            results.append(refined_map[key])
        else:
            adj = dict(item)
            # 未进入前列的劣质候选保持同量级成本并确保排在精算候选之后
            adj["total_cost"] = round(item["total_cost"] + avg_shift + 0.05, 5)
            adj["self_cost"] = round(item["self_cost"] + avg_shift + 0.05, 5)
            results.append(adj)

    results.sort(key=lambda item: (item["total_cost"], item["shanten_rest"], item["suit_left"], item["tile_indices"]))
    return results


def choose_swap_v2_indices(
    counts27: Sequence[int],
    *,
    is_dealer: bool = False,
    receiver_is_dealer: bool = False,
    swap_direction: int | None = None,
    exact_live: bool = True,
) -> tuple[int, int, int]:
    ranked = score_swap_candidates_v2(
        counts27,
        is_dealer=is_dealer,
        receiver_is_dealer=receiver_is_dealer,
        swap_direction=swap_direction,
        exact_live=exact_live,
    )
    if not ranked:
        raise ValueError("hand has no legal swap-three choice")
    return ranked[0]["tile_indices"]


def choose_swap_v2(
    hand: Hand,
    *,
    is_dealer: bool = False,
    receiver_is_dealer: bool = False,
    swap_direction: int | None = None,
) -> tuple[Tile, Tile, Tile]:
    i0, i1, i2 = choose_swap_v2_indices(
        hand.counts,
        is_dealer=is_dealer,
        receiver_is_dealer=receiver_is_dealer,
        swap_direction=swap_direction,
    )
    return (Tile.from_index(i0), Tile.from_index(i1), Tile.from_index(i2))


def swap_v2_sampling_probs(
    ranked: Sequence[dict],
    *,
    top_k: int = 3,
    max_gap: float = 0.12,
    temperature: float = 0.04,
) -> list[float]:
    """根据 score_swap_candidates_v2 的排序结果计算受控随机抽样概率。

    仅在排名前 top_k 且与最优分差 <= max_gap 的相近候选中做 Softmax 抽样;
    分差过大时直接截断为 0 概率, 确保大多数情况选第一选择且绝不抽到明显劣解。
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if max_gap < 0.0:
        raise ValueError("max_gap must be >= 0.0")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0.0")
    if not ranked:
        return []

    best_cost = ranked[0]["total_cost"]
    kept = [item for item in ranked[:top_k] if (item["total_cost"] - best_cost) <= max_gap]
    if not kept:
        kept = [ranked[0]]
    if len(kept) == 1:
        return [1.0]
    if temperature == 0.0:
        tied = [1.0 if item["total_cost"] == best_cost else 0.0 for item in kept]
        total_t = sum(tied)
        return [w / total_t for w in tied]

    weights = [math.exp(-(item["total_cost"] - best_cost) / temperature) for item in kept]
    total_w = sum(weights)
    return [w / total_w for w in weights]


def sample_swap_v2_indices(
    counts27: Sequence[int],
    *,
    rng: random.Random | None = None,
    top_k: int = 3,
    max_gap: float = 0.12,
    temperature: float = 0.04,
    is_dealer: bool = False,
    receiver_is_dealer: bool = False,
    swap_direction: int | None = None,
    exact_live: bool = True,
) -> tuple[int, int, int]:
    """带受控随机性的换三张选牌(供 AI 陪练与多局数据生成防过拟合)。"""
    ranked = score_swap_candidates_v2(
        counts27,
        is_dealer=is_dealer,
        receiver_is_dealer=receiver_is_dealer,
        swap_direction=swap_direction,
        exact_live=exact_live,
    )
    if not ranked:
        raise ValueError("hand has no legal swap-three choice")
    probs = swap_v2_sampling_probs(ranked, top_k=top_k, max_gap=max_gap, temperature=temperature)
    if len(probs) == 1:
        return ranked[0]["tile_indices"]
    drawer = rng if rng is not None else random
    chosen = drawer.choices(ranked[: len(probs)], weights=probs, k=1)[0]
    return chosen["tile_indices"]


def sample_swap_v2(
    hand: Hand,
    *,
    rng: random.Random | None = None,
    top_k: int = 3,
    max_gap: float = 0.12,
    temperature: float = 0.04,
    is_dealer: bool = False,
    receiver_is_dealer: bool = False,
    swap_direction: int | None = None,
    exact_live: bool = True,
) -> tuple[Tile, Tile, Tile]:
    """带受控随机性的换三张选牌, 返回排序后的三张 Tile。"""
    i0, i1, i2 = sample_swap_v2_indices(
        hand.counts,
        rng=rng,
        top_k=top_k,
        max_gap=max_gap,
        temperature=temperature,
        is_dealer=is_dealer,
        receiver_is_dealer=receiver_is_dealer,
        swap_direction=swap_direction,
        exact_live=exact_live,
    )
    return (Tile.from_index(i0), Tile.from_index(i1), Tile.from_index(i2))
