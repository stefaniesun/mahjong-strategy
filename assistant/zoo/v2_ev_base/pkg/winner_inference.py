"""已胡者手牌反推。

胡牌者只亮出胡的那一张,手牌扣着。已知:胡牌张、副露、定缺、暗手牌张数、全场每种牌还剩几张没出现。
满足"暗手牌 H + 胡牌张 能构成胡牌(标准型或七对)、H 不含缺门牌、每种牌用量不超过未现张数"的 H,
就是他手里可能的全部组合(计数档)。每个组合按"这些牌还剩几张"加权,得到每种牌在他手里的概率(概率档)。

枚举按花色拆开:面子和对子都不跨花色,每门花色内的组合数很小且可缓存,再用 numpy 做跨花色的笛卡尔积。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .counting import TileCounts
from .state import AssistantState
from .tiles import SUIT_ORDER, tidx

ENUM_LIMIT = 20_000  # 单人可能组合超过这个数就不展开(信息量也很低)
JOINT_LIMIT = 400_000  # 两名胡牌者联合过滤的组合对上限

STATUS_COMPLETE = "complete"
STATUS_TOO_MANY = "too_many"
STATUS_UNKNOWN_TILE = "unknown_tile"
STATUS_NO_CANDIDATE = "no_candidate"

_EMPTY = np.zeros((0, 27), dtype=np.int64)

# 单门花色内的面子选项:9 种刻子 + 7 种顺子(按点数从 0 计)
_MELD_OPTIONS: tuple[tuple[str, int], ...] = tuple([("pong", r) for r in range(9)] + [("chow", r) for r in range(7)])

# C(n, k) 查表,n、k 都不超过 4(每种牌最多 4 张)
_COMB = np.array([[math.comb(n, k) for k in range(5)] for n in range(5)], dtype=np.float64)
# 去重用:每行 27 个 0..4 的数按 5 进制编码成一个整数(5^27 < 2^63,int64 放得下)
_POW5 = np.array([5**i for i in range(27)], dtype=np.int64)


@dataclass(frozen=True, eq=False)
class WinnerInference:
    seat: int
    tile: str | None
    hidden_count: int
    status: str
    candidates: np.ndarray = field(default_factory=lambda: _EMPTY)  # (N, 27):每行一种可能的暗手牌
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))  # 与 candidates 对应,已归一化
    joint: bool = False  # 是否和另一名胡牌者做过联合约束
    certain: tuple[int, ...] = (0,) * 27  # 所有可能组合里都有的最少张数(确定档)
    expected: tuple[float, ...] = (0.0,) * 27  # 每种牌的期望张数(概率档)
    hold_prob: tuple[float, ...] | None = None  # 至少持有 1 张的概率(概率档);未展开时为 None

    @property
    def complete(self) -> bool:
        return self.status == STATUS_COMPLETE


@lru_cache(maxsize=131072)
def _suit_standard_configs(caps: tuple[int, ...], n_melds: int, with_pair: bool) -> tuple[tuple[int, ...], ...]:
    """一门花色内,恰好 n_melds 组面子(可再加一对将)、且不超过 caps 的所有张数分布。"""
    results: set[tuple[int, ...]] = set()
    counts = [0] * 9

    def finish() -> None:
        if not with_pair:
            results.add(tuple(counts))
            return
        for r in range(9):
            if counts[r] + 2 <= caps[r]:
                counts[r] += 2
                results.add(tuple(counts))
                counts[r] -= 2

    def rec(start: int, remaining: int) -> None:
        if remaining == 0:
            finish()
            return
        for i in range(start, len(_MELD_OPTIONS)):
            kind, r = _MELD_OPTIONS[i]
            if kind == "pong":
                if counts[r] + 3 <= caps[r]:
                    counts[r] += 3
                    rec(i, remaining - 1)
                    counts[r] -= 3
            elif counts[r] < caps[r] and counts[r + 1] < caps[r + 1] and counts[r + 2] < caps[r + 2]:
                counts[r] += 1
                counts[r + 1] += 1
                counts[r + 2] += 1
                rec(i, remaining - 1)
                counts[r] -= 1
                counts[r + 1] -= 1
                counts[r + 2] -= 1

    rec(0, n_melds)
    return tuple(sorted(results))


@lru_cache(maxsize=131072)
def _suit_pair_configs(caps: tuple[int, ...], n_pairs: int) -> tuple[tuple[int, ...], ...]:
    """一门花色内恰好 n_pairs 个对子(同一张最多当两对,即龙七对的四张)的所有张数分布。"""
    results: set[tuple[int, ...]] = set()
    counts = [0] * 9

    def rec(start: int, remaining: int) -> None:
        if remaining == 0:
            results.add(tuple(counts))
            return
        for r in range(start, 9):
            if counts[r] + 2 <= min(caps[r], 4):
                counts[r] += 2
                rec(r, remaining - 1)
                counts[r] -= 2

    rec(0, n_pairs)
    return tuple(sorted(results))


@lru_cache(maxsize=131072)
def _suit_standard_array(caps: tuple[int, ...], n_melds: int, with_pair: bool) -> np.ndarray:
    return np.asarray(_suit_standard_configs(caps, n_melds, with_pair), dtype=np.int64).reshape(-1, 9)


@lru_cache(maxsize=131072)
def _suit_pair_array(caps: tuple[int, ...], n_pairs: int) -> np.ndarray:
    return np.asarray(_suit_pair_configs(caps, n_pairs), dtype=np.int64).reshape(-1, 9)


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _cartesian(allowed: list[int], arrays: list[np.ndarray]) -> np.ndarray:
    sizes = [a.shape[0] for a in arrays]
    picks = np.indices(sizes).reshape(len(sizes), -1)
    out = np.zeros((picks.shape[1], 27), dtype=np.int64)
    for j, s in enumerate(allowed):
        out[:, s * 9 : (s + 1) * 9] = arrays[j][picks[j]]
    return out


@lru_cache(maxsize=4096)
def _enumerate_hidden_hands_cached(
    win_index: int,
    fixed_melds: int,
    void_suit: str | None,
    allowed_caps: tuple[tuple[int, tuple[int, ...]], ...],
    limit: int,
    allowed_need: tuple[tuple[int, tuple[int, ...]], ...],
) -> tuple[str, np.ndarray]:
    allowed = [s for s, _ in allowed_caps]
    suit_caps = dict(allowed_caps)
    suit_need = {s: np.asarray(n, dtype=np.int64) for s, n in allowed_need}

    def suit_lists(raw):
        lists = []
        for s, configs in zip(allowed, raw):
            low = suit_need[s]
            if low.any() and len(configs):
                configs = configs[(configs >= low).all(axis=1)]
            if not len(configs):
                return None
            lists.append(configs)
        return lists

    groups = []
    for meld_split in _compositions(4 - fixed_melds, len(allowed)):
        for pair_pos in range(len(allowed)):
            lists = suit_lists([_suit_standard_array(suit_caps[s], meld_split[j], j == pair_pos) for j, s in enumerate(allowed)])
            if lists is not None:
                groups.append(lists)
    if fixed_melds == 0:
        for pair_split in _compositions(7, len(allowed)):
            lists = suit_lists([_suit_pair_array(suit_caps[s], pair_split[j]) for j, s in enumerate(allowed)])
            if lists is not None:
                groups.append(lists)
    total = sum(math.prod(len(lst) for lst in lists) for lists in groups)
    if total > limit:
        return STATUS_TOO_MANY, _EMPTY
    if total == 0:
        return STATUS_NO_CANDIDATE, _EMPTY
    hands = np.concatenate([_cartesian(allowed, lists) for lists in groups])
    _, first = np.unique(hands @ _POW5, return_index=True)
    hands = hands[np.sort(first)]
    hands[:, win_index] -= 1
    hands.setflags(write=False)
    return STATUS_COMPLETE, hands


def enumerate_hidden_hands(
    win_index: int,
    fixed_melds: int,
    void_suit: str | None,
    unseen: tuple[int, ...],
    limit: int = ENUM_LIMIT,
    floor: tuple[int, ...] | None = None,
) -> tuple[str, np.ndarray]:
    """枚举胡牌者可能的暗手牌(不含胡牌张)。返回 (状态, 候选矩阵 (N,27))。

    unseen 是他每种牌最多能拿几张;floor 是他一定有的最少张数(如你换给他、还没出手的牌)。
    """
    allowed = [s for s in range(3) if void_suit is None or SUIT_ORDER[s] != void_suit]
    win_suit, _ = divmod(win_index, 9)
    if win_suit not in allowed or not 0 <= fixed_melds <= 4:
        return STATUS_NO_CANDIDATE, _EMPTY
    caps = list(unseen)
    caps[win_index] += 1  # 完整暗牌 = H + 亮出的胡牌张
    need = list(floor) if floor is not None else [0] * 27
    need[win_index] += 1  # 胡牌张所在花色必须含这张
    if any(need[t] for s in range(3) if s not in allowed for t in range(s * 9, (s + 1) * 9)):
        return STATUS_NO_CANDIDATE, _EMPTY  # 一定有的牌落在他的缺门里,不可能胡

    allowed_caps = tuple((s, tuple(caps[s * 9 : (s + 1) * 9])) for s in allowed)
    allowed_need = tuple((s, tuple(need[s * 9 : (s + 1) * 9])) for s in allowed)
    return _enumerate_hidden_hands_cached(win_index, fixed_melds, void_suit, allowed_caps, limit, allowed_need)


def _ways(unseen, hands: np.ndarray) -> np.ndarray:
    """从未现的牌里挑出每手牌的方法数:每种牌 C(未现张数, 用量) 相乘。"""
    u = np.asarray(unseen, dtype=np.int64)
    return _COMB[u[None, :], hands].prod(axis=1)


def _build(seat, tile, hidden_count, cands: np.ndarray, raw_weights: np.ndarray, joint: bool) -> WinnerInference:
    keep = raw_weights > 0
    if not keep.any():
        return WinnerInference(seat, tile, hidden_count, STATUS_NO_CANDIDATE)
    cands = cands[keep]
    weights = raw_weights[keep] / raw_weights[keep].sum()
    return WinnerInference(
        seat,
        tile,
        hidden_count,
        STATUS_COMPLETE,
        cands,
        weights,
        joint,
        tuple(int(x) for x in cands.min(axis=0)),
        tuple(float(x) for x in weights @ cands),
        tuple(float(x) for x in weights @ (cands >= 1)),
    )


def _proportional_expected(
    hidden_count: int, void_suit: str | None, pool: np.ndarray, floor: np.ndarray
) -> tuple[float, ...]:
    """组合未展开时的退路:一定有的牌照算,其余按随机部分的张数均摊(只在非缺门花色里摊)。"""
    allowed = [0 if void_suit is not None and SUIT_ORDER[t // 9] == void_suit else float(pool[t]) for t in range(27)]
    total = sum(allowed)
    free = max(0, hidden_count - int(floor.sum()))
    if total <= 0:
        return tuple(float(x) for x in floor)
    return tuple(float(floor[t]) + free * count / total for t, count in enumerate(allowed))


def infer_winners(
    state: AssistantState,
    counts: TileCounts,
    enum_limit: int = ENUM_LIMIT,
    joint_limit: int = JOINT_LIMIT,
    floors: dict[int, tuple[int, ...]] | None = None,
) -> list[WinnerInference]:
    """floors:各座位一定有的最少张数(来自换三张)。这些牌不参与随机分布,权重只算其余部分。"""
    unseen = np.asarray(counts.unseen, dtype=np.int64)
    floors = floors or {}
    zero = np.zeros(27, dtype=np.int64)
    floor_of = {seat: np.asarray(f, dtype=np.int64) for seat, f in floors.items()}
    total_floor = sum(floor_of.values(), zero)
    pool = np.maximum(unseen - total_floor, 0)  # 随机分布的部分
    pending = []
    for record in state.wins:
        if record.seat == state.perspective:
            continue
        view = state.players[record.seat]
        own_floor = floor_of.get(record.seat, zero)
        if record.tile is None:
            pending.append((record, view, STATUS_UNKNOWN_TILE, _EMPTY, own_floor))
            continue
        caps = tuple(int(x) for x in np.maximum(unseen - (total_floor - own_floor), 0))
        floor_arg = tuple(int(x) for x in own_floor) if own_floor.any() else None
        status, cands = enumerate_hidden_hands(tidx(record.tile), len(view.melds), view.void_suit, caps, enum_limit, floor_arg)
        pending.append((record, view, status, cands, own_floor))

    complete = [item for item in pending if item[2] == STATUS_COMPLETE]
    joint_weights: dict[int, np.ndarray] = {}
    if len(complete) == 2 and len(complete[0][3]) * len(complete[1][3]) <= joint_limit:
        (rec_a, _, _, a_hands, a_floor), (rec_b, _, _, b_hands, b_floor) = sorted(complete, key=lambda item: len(item[3]))
        # 只有"一定有"之外的部分是随机来的;仅保留至少一方可能用到的列以减少2-3倍内存带宽
        a_arr = a_hands - a_floor
        b_arr = b_hands - b_floor
        cols = np.where((a_arr.max(axis=0) > 0) | (b_arr.max(axis=0) > 0))[0]
        a_sub = a_arr[:, cols]
        b_sub = b_arr[:, cols]
        u_sub = pool[cols]
        base = _COMB[u_sub[None, :], a_sub].prod(axis=1)
        wa = np.zeros(len(a_sub))
        wb = np.zeros(len(b_sub))
        for i in range(len(a_sub)):
            rest = u_sub - a_sub[i]  # 第一个人拿走后剩下的
            ok = (b_sub <= rest).all(axis=1)
            if not ok.any():
                continue
            weight = base[i] * _COMB[rest[None, :], b_sub[ok]].prod(axis=1)
            wa[i] = weight.sum()
            wb[ok] += weight
        joint_weights[rec_a.seat] = wa
        joint_weights[rec_b.seat] = wb

    results: list[WinnerInference] = []
    for record, view, status, cands, own_floor in pending:
        if status == STATUS_COMPLETE:
            joint = record.seat in joint_weights
            weights = joint_weights[record.seat] if joint else _ways(pool, cands - own_floor)
            results.append(_build(record.seat, record.tile, view.hidden_count, cands, weights, joint))
        else:
            results.append(
                WinnerInference(
                    record.seat,
                    record.tile,
                    view.hidden_count,
                    status,
                    certain=tuple(int(x) for x in own_floor),
                    expected=_proportional_expected(view.hidden_count, view.void_suit, pool, own_floor),
                )
            )
    return results
