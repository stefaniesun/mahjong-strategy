"""基于 4,000 局定缺瞬间物理供需 + 1,750 局配对全打大样本校准的换后定缺策略 (void_v3)。

相对 void_v2 的四大核心升级(全部源自 outputs/void_moment_large.json 与 outputs/void_full_large.json):
1. 有效进张数(live)正式进入主评分(-0.020 * live): 大样本证实同向听同张数下进张更宽可显著提升 +0.543 分;
2. 适度保连张与暗刻/暗杠(void_conn +0.004, trips -0.20, quads -0.45): 避免只看一步向听而丢掉顺滑中张或杠牌潜能;
3. 收紧 5+ 张长门缺门惩罚(>=5张每多1张额外 +0.25): 守住 3/4/6、3/5/5 底线, 只在确能提速 1 步向听时破例缺 4 张, 避免乱缺 5 张;
4. 受限作用域的换牌流向防单行道修正(仅在同向听、同张数且有效进张差 <= 2 张的真正胶着局面生效):
   - 闲家撞车(换出 == 换入)平手时避开缺撞车门(+0.06), 降低放出对面单行道概率;
   - 非撞车平手时轻度避开缺换进门(+0.05)、对家互换时倾向缺换出门(-0.06), 兼顾供牌充裕与避免独缺过度喂牌;
   - 绝不因换牌流向而牺牲明显向听、张数或显著进张优势(特别保护庄家 14 张 4/4/6 牌形质量)。
"""
from __future__ import annotations

import math
import random
from typing import Sequence

from assistant.swap_v1 import fast_shanten_two_suits, split_suits, suit_profile
from assistant.swap_v2 import cached_live_two_suits, void_burden_cost
from engine.hand import Hand
from engine.tiles import SUITS, Suit

_OTHER_PAIRS: tuple[tuple[int, int], ...] = ((1, 2), (0, 2), (0, 1))
_SUIT_CODE_TO_IDX: dict[str, int] = {"W": 0, "T": 1, "B": 2}
_SUIT_CN: tuple[str, ...] = ("万", "条", "筒")
# 4,000 局 x 3 门实测各向听档位(sh=0..6)的平均有效进张基线:
# 用于消除“高向听烂牌随便摸什么都算进张”导致的跨向听虚高偏差, 同时保持同向听内 live 差值完全等价
_EXP_LIVE_BY_SH: tuple[float, ...] = (3.0, 5.0, 9.0, 20.0, 36.0, 51.0, 61.0)


def _normalize_suit_idx(suit: int | str | Suit | None) -> int | None:
    if suit is None:
        return None
    if isinstance(suit, int):
        if suit not in (0, 1, 2):
            raise ValueError(f"invalid suit index: {suit}")
        return suit
    if isinstance(suit, Suit):
        return SUITS.index(suit)
    code = str(suit).strip().upper()
    if code in _SUIT_CODE_TO_IDX:
        return _SUIT_CODE_TO_IDX[code]
    raise ValueError(f"invalid suit code: {suit}")


def score_void_candidates_v3(
    counts27: Sequence[int],
    *,
    is_dealer: bool = False,
    swap_out_suit: int | str | Suit | None = None,
    swap_in_suit: int | str | Suit | None = None,
    swap_direction: int | None = None,
) -> list[dict]:
    """对万/条/筒三种定缺候选进行 void_v3 综合评分(total_cost 越小越优)。"""
    suits3 = split_suits(counts27)
    profiles = (suit_profile(suits3[0]), suit_profile(suits3[1]), suit_profile(suits3[2]))
    out_idx = _normalize_suit_idx(swap_out_suit)
    in_idx = _normalize_suit_idx(swap_in_suit)
    has_swap_ctx = out_idx is not None and in_idx is not None
    is_crash = has_swap_ctx and (out_idx == in_idx)

    raw_rows: list[dict] = []
    for v_idx in (0, 1, 2):
        i_a, i_b = _OTHER_PAIRS[v_idx]
        pa, pb, pv = profiles[i_a], profiles[i_b], profiles[v_idx]
        sh = fast_shanten_two_suits(suits3[i_a], suits3[i_b])
        live = cached_live_two_suits(suits3[i_a], suits3[i_b], sh)
        v_cnt = pv.total
        max_main = pa.total if pa.total >= pb.total else pb.total

        # 1) 缺门张数包袱 + 5+ 张长门额外防硬拗惩罚
        burden = void_burden_cost(v_cnt, is_dealer) + 0.25 * max(0, v_cnt - 4)
        # 2) 清一色苗头红利(与 swap_v2 对齐)
        flush_bonus = -0.32 if max_main >= 10 and v_cnt <= 2 else (-0.15 if max_main == 9 and v_cnt <= 2 else 0.0)
        # 3) 保留面子根基(暗刻/暗杠)与留牌有效进张/连张质量(按向听档位去基线, 防止高向听虚高)
        kept_trips = pa.triplets + pb.triplets
        kept_quads = pa.quads + pb.quads
        kept_pairs = pa.pairs + pb.pairs
        meld_bonus = -0.20 * kept_trips - 0.45 * kept_quads
        ref_live = _EXP_LIVE_BY_SH[sh] if sh < len(_EXP_LIVE_BY_SH) else _EXP_LIVE_BY_SH[-1]
        live_term = -0.022 * (live - ref_live)
        conn_term = 0.004 * pv.conn

        base_cost = 1.0 * sh + burden + live_term + meld_bonus + conn_term + flush_bonus

        raw_rows.append(
            {
                "suit_idx": v_idx,
                "suit_code": SUITS[v_idx].value,
                "suit": _SUIT_CN[v_idx],
                "void_cnt": v_cnt,
                "shanten": sh,
                "live": live,
                "kept_pairs": kept_pairs,
                "kept_trips": kept_trips,
                "kept_quads": kept_quads,
                "void_conn": pv.conn,
                "max_main": max_main,
                "base_cost": round(base_cost, 6),
                "ctx_cost": 0.0,
                "ctx_note": "",
                "is_out_suit": (out_idx == v_idx) if out_idx is not None else False,
                "is_in_suit": (in_idx == v_idx) if in_idx is not None else False,
                "is_crash": is_crash,
            }
        )

    # 先找出纯手牌维度最优项; 换牌流向上下文仅在确实存在至少两门同向听、同张数且有效进张接近(<=2张)的竞争门时介入
    best_base = min(
        raw_rows,
        key=lambda r: (r["base_cost"], r["void_cnt"], r["shanten"], -r["live"], r["void_conn"], r["suit_idx"]),
    )
    tier_peer_idxs = {
        r["suit_idx"]
        for r in raw_rows
        if r["shanten"] == best_base["shanten"]
        and r["void_cnt"] == best_base["void_cnt"]
        and (best_base["live"] - r["live"] <= 2)
    }
    has_close_peer = len(tier_peer_idxs) >= 2

    for r in raw_rows:
        ctx = 0.0
        note = ""
        if has_swap_ctx and has_close_peer and r["suit_idx"] in tier_peer_idxs:
            if is_crash:
                if r["is_out_suit"] and not is_dealer:
                    ctx = 0.06
                    note = "撞车门(同条件避放单行道)"
                elif r["is_out_suit"] and is_dealer:
                    note = "撞车门(庄家14张优先保牌形)"
            else:
                if r["is_in_suit"]:
                    ctx = 0.05
                    note = "换进门(给牌方大概率同缺,略防撞缺)"
                elif r["is_out_suit"]:
                    if swap_direction == 2:
                        ctx = -0.06
                        note = "换出门(对家互换优先缺换出,供牌充裕)"
                    else:
                        note = "换出门(收牌方多留该门,供牌较好但需防喂牌)"
        elif has_swap_ctx:
            if is_crash and r["is_out_suit"]:
                note = "撞车门"
            elif r["is_in_suit"]:
                note = "换进门"
            elif r["is_out_suit"]:
                note = "换出门"

        r["ctx_cost"] = round(ctx, 6)
        r["ctx_note"] = note
        r["total_cost"] = round(r["base_cost"] + ctx, 6)

    raw_rows.sort(
        key=lambda r: (r["total_cost"], r["void_cnt"], r["shanten"], -r["live"], r["void_conn"], r["suit_idx"])
    )
    return raw_rows


def choose_void_v3_idx(
    counts27: Sequence[int],
    *,
    is_dealer: bool = False,
    swap_out_suit: int | str | Suit | None = None,
    swap_in_suit: int | str | Suit | None = None,
    swap_direction: int | None = None,
) -> int:
    ranked = score_void_candidates_v3(
        counts27,
        is_dealer=is_dealer,
        swap_out_suit=swap_out_suit,
        swap_in_suit=swap_in_suit,
        swap_direction=swap_direction,
    )
    return ranked[0]["suit_idx"]


def choose_void_v3(
    hand: Hand,
    *,
    is_dealer: bool = False,
    swap_out_suit: int | str | Suit | None = None,
    swap_in_suit: int | str | Suit | None = None,
    swap_direction: int | None = None,
) -> Suit:
    idx = choose_void_v3_idx(
        hand.counts,
        is_dealer=is_dealer,
        swap_out_suit=swap_out_suit,
        swap_in_suit=swap_in_suit,
        swap_direction=swap_direction,
    )
    return SUITS[idx]


def void_v3_sampling_probs(
    ranked: Sequence[dict],
    *,
    top_k: int = 2,
    max_gap: float = 0.18,
    temperature: float = 0.06,
) -> list[float]:
    """根据 score_void_candidates_v3 的排序结果计算受控随机抽样概率。

    仅在两门定缺代价极其接近(<= max_gap, 默认 0.18)时按 Softmax 探索次优门;
    实测在所有明显手牌上 100% 锁定 #1, 在真正胶着局面中约 81% 选 #1 / 19% 探索 #2,
    平均期望代价损失仅 0.0007, 兼顾高水平与 AI 训练样本多样性。
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
        s = sum(tied)
        return [w / s for w in tied]

    weights = [math.exp(-(item["total_cost"] - best_cost) / temperature) for item in kept]
    total_w = sum(weights)
    return [w / total_w for w in weights]


def sample_void_v3_idx(
    counts27: Sequence[int],
    *,
    rng: random.Random | None = None,
    top_k: int = 2,
    max_gap: float = 0.18,
    temperature: float = 0.06,
    is_dealer: bool = False,
    swap_out_suit: int | str | Suit | None = None,
    swap_in_suit: int | str | Suit | None = None,
    swap_direction: int | None = None,
) -> int:
    ranked = score_void_candidates_v3(
        counts27,
        is_dealer=is_dealer,
        swap_out_suit=swap_out_suit,
        swap_in_suit=swap_in_suit,
        swap_direction=swap_direction,
    )
    probs = void_v3_sampling_probs(ranked, top_k=top_k, max_gap=max_gap, temperature=temperature)
    if len(probs) == 1:
        return ranked[0]["suit_idx"]
    drawer = rng if rng is not None else random
    chosen = drawer.choices(ranked[: len(probs)], weights=probs, k=1)[0]
    return chosen["suit_idx"]


def sample_void_v3(
    hand: Hand,
    *,
    rng: random.Random | None = None,
    top_k: int = 2,
    max_gap: float = 0.18,
    temperature: float = 0.06,
    is_dealer: bool = False,
    swap_out_suit: int | str | Suit | None = None,
    swap_in_suit: int | str | Suit | None = None,
    swap_direction: int | None = None,
) -> Suit:
    idx = sample_void_v3_idx(
        hand.counts,
        rng=rng,
        top_k=top_k,
        max_gap=max_gap,
        temperature=temperature,
        is_dealer=is_dealer,
        swap_out_suit=swap_out_suit,
        swap_in_suit=swap_in_suit,
        swap_direction=swap_direction,
    )
    return SUITS[idx]
