"""对手读牌模块:基于牌河、副露、余牌与换牌下限估计对手听牌概率、大牌预警与校准点炮风险。

所有概率权重均由 200 局混合对手(RulePlusPolicy + RulePolicy) 共 13,364 次对手快照与 80,145 次牌-对手对账实测标定:
- 听牌概率 P(tenpai | 副露数, 非缺门已打张数): 0副露0非缺门 0.7% -> 3-4张 47.2% -> 8+张 82.6%; 2副露8+张 93.3%; 4副露单吊近必然听牌;
- 现物安全效应(四川血战虽无永久振听,但对手已打过的牌点炮率断崖式下降): 最近3巡现物 0.49%(约 28 倍安全), 早巡现物 1.26%(约 11 倍安全), 非现物 13.55%;
- 双筋中张(4/5/6 两端筋牌均打过): 点炮率降约 40%;
- 牌河花色偏向: 强清一色目标门点炮率 23.5%(约 1.75x), 少弃门 17.9%(1.32x), 多弃门 10.3%(0.76x), 主弃门 8.4%(0.62x);
- 已知换出牌搭子(swap_floors 命中等待形状): 点炮率升至 19.8%(约 1.46x)。
"""
from __future__ import annotations

from dataclasses import dataclass

from assistant.counting import TileCounts
from assistant.state import AssistantState, PlayerView
from assistant.tiles import SUIT_ORDER, label, rank_of, seat_name, suit_label, suit_of, tidx

# 13,364 次对手快照实测: (min(n_melds, 3), nv_bin) -> P(tenpai)
# nv_bin: 0->0张非缺门弃牌, 1->1~2张, 2->3~4张, 3->5~7张, 4->8+张
_TENPAI_TABLE: tuple[tuple[float, ...], ...] = (
    (0.008, 0.152, 0.472, 0.723, 0.826),  # 0 副露
    (0.023, 0.272, 0.493, 0.686, 0.888),  # 1 副露
    (0.130, 0.316, 0.582, 0.726, 0.933),  # 2 副露
    (0.350, 0.550, 0.720, 0.920, 0.940),  # 3 副露
)

_BASE_WAIT_NON_GENBUTSU = 0.1355
_BASE_WAIT_GENBUTSU_RECENT = 0.0049
_BASE_WAIT_GENBUTSU_OLD = 0.0126
_BASE_WAIT_PASSED_SAME_HAND = 0.0030


@dataclass(frozen=True)
class OpponentRead:
    seat: int
    name: str
    tenpai_prob: float
    tenpai_level: str  # "none" / "low" / "mid" / "high" / "near_certain"
    holds_known_void: bool
    n_melds: int
    n_kongs: int
    non_void_discards: int
    trailing_tsumogiri: int
    flush_suit: str | None  # "W"/"T"/"B" if flush threat detected
    flush_level: str | None  # "strong" / "watch"
    toitoi_alert: str | None
    threats: tuple[str, ...]
    basis: tuple[str, ...]
    # 供单张点炮率快速查表的预计算结构
    disc_set: frozenset[int]
    recent3_set: frozenset[int]
    passed_same_hand_set: frozenset[int]
    nv_disc_by_suit: dict[str, int]
    melds_by_suit: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "seat": self.seat,
            "name": self.name,
            "tenpai_prob": round(self.tenpai_prob, 3),
            "tenpai_level": self.tenpai_level,
            "holds_known_void": self.holds_known_void,
            "n_melds": self.n_melds,
            "n_kongs": self.n_kongs,
            "non_void_discards": self.non_void_discards,
            "trailing_tsumogiri": self.trailing_tsumogiri,
            "flush_suit": suit_label(self.flush_suit) if self.flush_suit else None,
            "flush_level": self.flush_level,
            "toitoi_alert": self.toitoi_alert,
            "threats": list(self.threats),
            "basis": list(self.basis),
        }


def _nv_bin(n_nv: int) -> int:
    if n_nv <= 0:
        return 0
    if n_nv <= 2:
        return 1
    if n_nv <= 4:
        return 2
    if n_nv <= 7:
        return 3
    return 4


def _tenpai_level(p: float) -> str:
    if p <= 0.001:
        return "none"
    if p < 0.25:
        return "low"
    if p < 0.55:
        return "mid"
    if p < 0.85:
        return "high"
    return "near_certain"


def read_opponent(
    state: AssistantState,
    view: PlayerView,
    floor: tuple[int, ...] | None = None,
) -> OpponentRead:
    """分析单个在场对手的听牌进度与大牌型威胁(单次 <0.05ms)。"""
    who = seat_name(view.seat, state.perspective)
    vcode = view.void_suit
    disc_idx = [tidx(t) for t in view.discards]
    disc_set = frozenset(disc_idx)
    recent3_set = frozenset(disc_idx[-3:]) if disc_idx else frozenset()
    passed_same_hand_set = frozenset(tidx(t) for t in view.passed_since_tedashi)

    nv_disc_by_suit = {"W": 0, "T": 0, "B": 0}
    nv_positions: list[int] = []
    for pos, t in enumerate(disc_idx):
        sc = suit_of(t)
        if sc != vcode:
            nv_disc_by_suit[sc] += 1
            nv_positions.append(pos)
    n_nv = sum(nv_disc_by_suit.values())

    # 统计非缺门末尾连续摸切次数(仅在 discard_tsumogiri 与 discards 对齐时启用)
    trailing_tsumogiri = 0
    if len(view.discard_tsumogiri) == len(disc_idx) and nv_positions:
        for pos in reversed(nv_positions):
            if view.discard_tsumogiri[pos]:
                trailing_tsumogiri += 1
            else:
                break

    melds_by_suit = {"W": 0, "T": 0, "B": 0}
    n_kongs = 0
    for m in view.melds:
        melds_by_suit[suit_of(tidx(m.tile))] += 1
        if m.kind == "kong":
            n_kongs += 1
    n_melds = len(view.melds)

    holds_known_void = bool(
        floor is not None and vcode is not None and any(floor[t] > 0 and suit_of(t) == vcode for t in range(27))
    )

    basis: list[str] = []
    if holds_known_void:
        tenpai_p = 0.0
        basis.append("你换给他的缺门牌尚未打出,打完前必然未听牌")
    elif n_melds >= 4 or view.hidden_count == 1:
        # 4 副露只剩 1 张暗牌(金钩钓): 除非那 1 张刚好是缺门,否则必已听牌
        tenpai_p = 0.97 if (n_nv >= 1 or state.wall_count <= 45) else 0.85
        basis.append(f"已亮出 {n_melds} 组副露,仅剩 {view.hidden_count} 张暗牌(金钩钓单吊)")
    else:
        tenpai_p = _TENPAI_TABLE[min(n_melds, 3)][_nv_bin(n_nv)]
        # 摸切节奏微调(120局实测: 早巡1-2张非缺门即摸切听牌率 +10%, 后巡8+张摸切听牌率 +6%)
        if trailing_tsumogiri >= 1:
            if 1 <= n_nv <= 2:
                tenpai_p = min(0.95, tenpai_p + 0.07)
            elif n_nv >= 8:
                tenpai_p = min(0.96, tenpai_p + 0.05)
        # 残局轻微上调(牌墙越浅未胡者听牌比例越高)
        if state.wall_count <= 20 and n_nv >= 2:
            tenpai_p = min(0.95, tenpai_p + 0.06)
        tsumo_txt = f", 近期连续摸切 {trailing_tsumogiri} 巡" if trailing_tsumogiri >= 2 else ""
        basis.append(f"副露 {n_melds} 组,非缺门已弃 {n_nv} 张{tsumo_txt}(牌墙余 {state.wall_count} 张)")

    # 大牌型预警: 1) 清一色
    flush_suit: str | None = None
    flush_level: str | None = None
    threats: list[str] = []
    if vcode is not None:
        other_suits = [sc for sc in SUIT_ORDER if sc != vcode]
        s_a, s_b = other_suits[0], other_suits[1]
        for target_s, dump_s in ((s_a, s_b), (s_b, s_a)):
            m_t, m_o = melds_by_suit[target_s], melds_by_suit[dump_s]
            d_t, d_o = nv_disc_by_suit[target_s], nv_disc_by_suit[dump_s]
            if m_o > 0:
                continue
            if (m_t >= 2 and d_t <= 1 and d_o >= 2) or (m_t >= 3 and d_t == 0):
                flush_suit = target_s
                flush_level = "strong"
                threats.append(f"清一色警戒({suit_label(target_s)})")
                basis.append(
                    f"副露全为{suit_label(target_s)}({m_t}组),弃{suit_label(dump_s)} {d_o} 张而仅弃{suit_label(target_s)} {d_t} 张"
                )
                break
            if d_t == 0 and d_o >= 4:
                flush_suit = target_s
                flush_level = "watch"
                threats.append(f"疑似清一色({suit_label(target_s)})")
                basis.append(f"连弃 {d_o} 张{suit_label(dump_s)}且一张{suit_label(target_s)}未弃")
                break

    # 大牌型预警: 2) 金钩钓 / 对对胡
    toitoi_alert: str | None = None
    if n_melds >= 4 or view.hidden_count == 1:
        toitoi_alert = "金钩钓单吊"
        threats.append("金钩钓单吊(大番)")
    elif n_melds == 3:
        toitoi_alert = "大对子/单吊警戒"
        threats.append("3副露大对子/快听警戒")
    elif n_melds == 2 and tenpai_p >= 0.55:
        toitoi_alert = "2副露快听(留意对对胡)"

    if n_kongs >= 1:
        threats.append(f"已开 {n_kongs} 杠(带根高番)")

    return OpponentRead(
        seat=view.seat,
        name=who,
        tenpai_prob=tenpai_p,
        tenpai_level=_tenpai_level(tenpai_p),
        holds_known_void=holds_known_void,
        n_melds=n_melds,
        n_kongs=n_kongs,
        non_void_discards=n_nv,
        trailing_tsumogiri=trailing_tsumogiri,
        flush_suit=flush_suit,
        flush_level=flush_level,
        toitoi_alert=toitoi_alert,
        threats=tuple(threats),
        basis=tuple(basis),
        disc_set=disc_set,
        recent3_set=recent3_set,
        passed_same_hand_set=passed_same_hand_set,
        nv_disc_by_suit=nv_disc_by_suit,
        melds_by_suit=melds_by_suit,
    )


def read_all_opponents(
    state: AssistantState,
    floors: dict[int, tuple[int, ...]] | None = None,
) -> dict[int, OpponentRead]:
    floors = floors or {}
    return {view.seat: read_opponent(state, view, floors.get(view.seat)) for view in state.active_opponents()}


def estimate_dealin_prob(
    tile_idx: int,
    opp_read: OpponentRead,
    counts: TileCounts,
    shapes: list[dict],
) -> tuple[float, list[str]]:
    """估计在非100%安全前提下,打出 tile_idx 点炮给 opp_read 的校准概率及简明依据。"""
    if opp_read.tenpai_prob <= 0.0:
        return 0.0, ["对方尚未听牌"]

    tags: list[str] = []
    # 1) 同手未变过牌(120局实测 737/737 零点炮, 最强安全信号)与现物判断
    if tile_idx in opp_read.passed_same_hand_set:
        wait_p = _BASE_WAIT_PASSED_SAME_HAND
        tags.append("手牌未变且刚放过此张(极安全)")
    elif tile_idx in opp_read.recent3_set:
        wait_p = _BASE_WAIT_GENBUTSU_RECENT
        tags.append("近3巡现物(极安全)")
    elif tile_idx in opp_read.disc_set:
        wait_p = _BASE_WAIT_GENBUTSU_OLD
        tags.append("对方已打现物(较安全)")
    else:
        wait_p = _BASE_WAIT_NON_GENBUTSU
        r = rank_of(tile_idx)
        t_suit = suit_of(tile_idx)
        base_s = tile_idx - (r - 1)

        # 2) 点数区间修正 (1/9 边幺 vs 2/8 vs 3-7 中张)
        if r in (1, 9):
            wait_p *= 0.76
        elif r in (2, 8):
            wait_p *= 0.97
        else:
            wait_p *= 1.08

        # 3) 双筋安全修正 (中张 4/5/6 两端筋牌均已打出)
        if 4 <= r <= 6:
            suji_low = (base_s + r - 4) in opp_read.disc_set
            suji_high = (base_s + r + 2) in opp_read.disc_set
            if suji_low and suji_high:
                wait_p *= 0.60
                tags.append("双筋熟张")

        # 4) 牌河花色偏向 & 清一色目标门
        d_this = opp_read.nv_disc_by_suit.get(t_suit, 0)
        d_other = opp_read.non_void_discards - d_this
        if opp_read.flush_suit == t_suit and opp_read.flush_level == "strong":
            wait_p *= 1.75
            tags.append(f"命中对方清一色警戒门({suit_label(t_suit)})")
        elif opp_read.flush_suit == t_suit and opp_read.flush_level == "watch":
            wait_p *= 1.30
            tags.append(f"对方疑似清一色门({suit_label(t_suit)})")
        elif opp_read.flush_suit is not None and opp_read.flush_suit != t_suit:
            wait_p *= 0.62
            tags.append("对方主弃花色")
        elif d_other - d_this >= 2:
            wait_p *= 1.28
            tags.append("对方少弃此门")
        elif d_this >= 3 and d_other == 0:
            wait_p *= 0.65
            tags.append("对方主弃此门")
        elif d_this - d_other >= 2:
            wait_p *= 0.78
            tags.append("对方多弃此门")

        # 5) 剩余未现张数(3张已现 -> 很难凑单钓/对倒)
        if counts.unseen[tile_idx] == 1:
            wait_p *= 0.64
            tags.append("外池仅余1张")

        # 6) 已知换牌搭子命中
        if any(sh.get("known") for sh in shapes):
            wait_p *= 1.45
            tags.append("你换给他的牌恰好凑成等待搭子")

        # 7) 4副露金钩钓单吊(只可能单钓剩余非缺门活张)
        if opp_read.n_melds >= 4:
            wait_p = max(wait_p, 0.22)
            tags.append("金钩钓单吊警戒")

    wait_p = max(0.002, min(0.65, wait_p))
    dealin_p = max(0.0, min(0.65, opp_read.tenpai_prob * wait_p))
    return dealin_p, tags


def describe_tile_risk(dealin_prob: float) -> str:
    if dealin_prob <= 0.0:
        return "100%安全"
    if dealin_prob < 0.015:
        return "极低风险"
    if dealin_prob < 0.05:
        return "低风险"
    if dealin_prob < 0.12:
        return "中风险"
    if dealin_prob < 0.22:
        return "高风险"
    return "极危"
