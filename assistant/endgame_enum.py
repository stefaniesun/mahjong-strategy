"""残局对手精确听牌枚举引擎 (endgame_enum)。

当在场对手副露 >= 2 组(暗手牌 <= 7 张: 7 张 / 4 张 / 1 张金钩钓) 且进入中后盘(牌墙 <= 20 张或暗牌 <= 4 张)时，
直接复用 winner_inference 的单门拆分缓存，在毫秒级(<5ms)内精确枚举该对手全部合法听牌手牌组合：
1. 计算该对手「清一色」的贝叶斯后验概率（同时融合已胡反推后的真实外池余张供给 + 牌河出牌偏向）；
2. 精确区分每张手牌的真实点炮概率（解决形状启发式把 4万 和 5万 粗略估成同风险的盲区）；
3. 输出可展示在主界面和侧边栏的结构化残局推理结论。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from assistant.counting import TileCounts
from assistant.danger import Floors, _comb_real, holding_caps
from assistant.opponent_read import OpponentRead
from assistant.state import AssistantState, PlayerView
from assistant.tiles import SUIT_ORDER, label, seat_name, suit_label, tidx
from assistant.winner_inference import (
    STATUS_COMPLETE,
    WinnerInference,
    _POW5,
    enumerate_hidden_hands,
)

PER_WAIT_ENUM_LIMIT = 2500
MAX_TOTAL_UNIQUE_TENPAI = 6000


@dataclass(frozen=True)
class EndgameOppEnum:
    seat: int
    name: str
    hidden_count: int
    n_melds: int
    tenpai_hand_count: int
    flush_suit: str | None
    flush_hand_count: int
    flush_prob: float  # 条件概率 P(清一色 | 已听)
    exact_wait_prob: tuple[float, ...]  # 长度 27: P(对手听 t | 对手已听)
    top_waits: tuple[dict[str, Any], ...]
    supply_note: str
    complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "name": self.name,
            "hidden_count": self.hidden_count,
            "n_melds": self.n_melds,
            "tenpai_hand_count": self.tenpai_hand_count,
            "flush_suit": self.flush_suit,
            "flush_suit_label": suit_label(self.flush_suit) if self.flush_suit else None,
            "flush_hand_count": self.flush_hand_count,
            "flush_prob": round(self.flush_prob, 3),
            "top_waits": list(self.top_waits),
            "supply_note": self.supply_note,
            "complete": self.complete,
        }


def _river_flush_boost(opp: PlayerView, meld_suit: str | None) -> float:
    """根据对手副露同花色 + 牌河弃牌花色偏向，给出纯清一色手牌的贝叶斯行为似然倍率。"""
    if meld_suit is None or opp.void_suit is None:
        return 1.0
    other_suit = next((s for s in SUIT_ORDER if s != opp.void_suit and s != meld_suit), None)
    if other_suit is None:
        return 1.0
    other_disc = sum(1 for d in opp.discards if d[-1] == other_suit)
    meld_disc = sum(1 for d in opp.discards if d[-1] == meld_suit)
    if meld_disc >= 3:
        return 0.35
    if meld_disc == 2:
        return 0.70
    excess = max(0, other_disc - 2 * meld_disc)
    base_meld = 2.6 if len(opp.melds) >= 3 else 1.6
    cap = 30.0 if meld_disc == 0 else 16.0
    return min(cap, base_meld * (1.55 ** min(6, excess)))


def _enumerate_single_opponent(
    state: AssistantState,
    counts: TileCounts,
    winners: list[WinnerInference],
    opp: PlayerView,
    floors: Floors,
    opp_read: OpponentRead | None,
) -> EndgameOppEnum | None:
    n_melds = len(opp.melds)
    hidden = counts.hidden.get(opp.seat, opp.hidden_count)
    if n_melds < 2 or hidden > 7 or hidden <= 0:
        return None
    if state.wall_count > 20 and hidden > 4:
        return None

    caps_list = holding_caps(state, counts, winners, opp, floors)
    caps = tuple(caps_list)
    own_floor = floors.get(opp.seat)
    own_floor_arr = own_floor if own_floor is not None else (0,) * 27

    # 1) 枚举每个非缺门胡牌张 w 对应的全部合法暗手牌 H (H <= caps, H + w 胡牌)
    hand_by_key: dict[int, tuple[np.ndarray, int]] = {}
    all_complete = True
    for w in range(27):
        if opp.void_suit is not None and SUIT_ORDER[w // 9] == opp.void_suit:
            continue
        status, cands = enumerate_hidden_hands(
            w, n_melds, opp.void_suit, caps, limit=PER_WAIT_ENUM_LIMIT, floor=own_floor
        )
        if status != STATUS_COMPLETE:
            if status != "no_candidate":
                all_complete = False
            continue
        if len(cands) == 0:
            continue
        keys = cands @ _POW5
        bit = 1 << w
        for row, k_val in zip(cands, keys.tolist()):
            prev = hand_by_key.get(k_val)
            if prev is None:
                hand_by_key[k_val] = (row, bit)
            else:
                hand_by_key[k_val] = (prev[0], prev[1] | bit)
        if len(hand_by_key) > MAX_TOTAL_UNIQUE_TENPAI:
            all_complete = False
            break

    if not hand_by_key:
        return None

    # 2) 判断所有副露是否同属一门(清一色候选门)
    meld_suits = {m.tile[-1] for m in opp.melds}
    flush_suit = next(iter(meld_suits)) if len(meld_suits) == 1 and next(iter(meld_suits)) != opp.void_suit else None
    flush_suit_idx = SUIT_ORDER.index(flush_suit) if flush_suit is not None else None
    flush_boost = _river_flush_boost(opp, flush_suit)

    # 扣除已胡者期望持有 + 他家换牌底牌后的真实可摸池
    winner_exp = [sum(inf.expected[t] for inf in winners) for t in range(27)]
    other_floor = [0] * 27
    winner_seats = {inf.seat for inf in winners}
    for s, fl in floors.items():
        if s != opp.seat and s not in winner_seats:
            for t in range(27):
                other_floor[t] += fl[t]
    avail_pool = [max(0.0, counts.unseen[t] - winner_exp[t] - other_floor[t] - own_floor_arr[t]) for t in range(27)]

    # 行为过滤集合: 刚打出的最后一张 / 现物与手牌未变放过牌
    last_disc_idx = tidx(opp.discards[-1]) if opp.discards else None
    passed_set = {tidx(t) for t in opp.passed_since_tedashi}
    river_set = {tidx(t) for t in opp.river}

    # 3) 对每种唯一听牌手牌计算精确后验权重
    total_w = 0.0
    flush_w = 0.0
    flush_cnt = 0
    wait_w = [0.0] * 27

    for _, (h_vec, w_mask) in hand_by_key.items():
        # 供给组合权重: 按扣完已胡者后的 avail_pool 平滑算 C(avail_pool[t], need)
        wt = 1.0
        nz = np.nonzero(h_vec)[0]
        is_pure_flush = flush_suit_idx is not None
        for t in nz.tolist():
            cnt = int(h_vec[t])
            if is_pure_flush and (t // 9) != flush_suit_idx:
                is_pure_flush = False
            need = max(0, cnt - own_floor_arr[t])
            if need <= 0:
                continue
            p_t = avail_pool[t]
            if p_t >= need:
                wt *= _comb_real(p_t, need)
            elif p_t > 1e-6:
                wt *= max(0.08, (p_t / need) ** need)
            else:
                wt *= 0.04

        # 提取听口列表并做行为过滤
        wait_list = [w for w in range(27) if (w_mask >> w) & 1]
        if is_pure_flush:
            # 纯清一色听牌要求全部听口也落在清一色门内
            if not all((w // 9) == flush_suit_idx for w in wait_list):
                is_pure_flush = False

        if last_disc_idx is not None and ((w_mask >> last_disc_idx) & 1):
            wt *= 0.04  # 刚打出的牌极少恰好是自己正在听的牌
        elif all((w in passed_set or w in river_set) for w in wait_list):
            wt *= 0.05  # 全部听口均为现物或过庄放过牌

        if is_pure_flush:
            wt *= flush_boost
            if last_disc_idx is None or not ((w_mask >> last_disc_idx) & 1):
                flush_cnt += 1
            flush_w += wt

        total_w += wt
        for w in wait_list:
            # 若该手牌有多个听口, 只要摸/点任意一个都胡, 所以每个听口累加上该手牌的完整后验权重
            wait_w[w] += wt

    if total_w <= 1e-12:
        return None

    exact_wait_prob = tuple(min(1.0, wait_w[t] / total_w) for t in range(27))
    flush_prob = min(1.0, flush_w / total_w) if flush_suit is not None else 0.0

    # 整理最危听口 TOP 6(排除自身现物极低概率项, 按真实命中率降序)
    tp_scale = opp_read.tenpai_prob if opp_read is not None else 1.0
    top_waits = tuple(
        {
            "tile": label(t),
            "cond_prob": round(exact_wait_prob[t], 3),
            "dealin_prob": round(exact_wait_prob[t] * tp_scale, 3),
        }
        for t in sorted(range(27), key=lambda i: -exact_wait_prob[i])[:6]
        if exact_wait_prob[t] >= 0.05
    )

    # 简洁供给说明(例: 扣已胡后外池: 万≈5.2张 · 筒≈8.4张)
    suit_parts = []
    for s_idx, s_code in enumerate(SUIT_ORDER):
        if s_code == opp.void_suit:
            continue
        s_sum = sum(avail_pool[s_idx * 9 : (s_idx + 1) * 9])
        suit_parts.append(f"{suit_label(s_code)}≈{s_sum:.1f}张")
    supply_note = f"扣已胡外池: {' · '.join(suit_parts)}"
    if flush_suit is not None:
        supply_note += f" | 纯{suit_label(flush_suit)}听牌仅 {flush_cnt} 种"

    return EndgameOppEnum(
        seat=opp.seat,
        name=seat_name(opp.seat, state.perspective),
        hidden_count=hidden,
        n_melds=n_melds,
        tenpai_hand_count=len(hand_by_key),
        flush_suit=flush_suit,
        flush_hand_count=flush_cnt,
        flush_prob=flush_prob,
        exact_wait_prob=exact_wait_prob,
        top_waits=top_waits,
        supply_note=supply_note,
        complete=all_complete,
    )


def enumerate_endgame_opponents(
    state: AssistantState,
    counts: TileCounts,
    winners: list[WinnerInference],
    floors: Floors | None = None,
    opp_reads: dict[int, OpponentRead] | None = None,
) -> dict[int, EndgameOppEnum]:
    """为所有符合残局精确枚举条件的在场对手计算听牌后验分布。"""
    if state.phase != "play":
        return {}
    fl = floors or {}
    reads = opp_reads or {}
    out: dict[int, EndgameOppEnum] = {}
    for opp in state.active_opponents():
        res = _enumerate_single_opponent(state, counts, winners, opp, fl, reads.get(opp.seat))
        if res is not None:
            out[opp.seat] = res
    return out
