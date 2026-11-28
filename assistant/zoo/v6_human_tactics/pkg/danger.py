"""安全度与等待形状。

一张牌 t 要让对手 P 胡,P 胡牌后的整手牌里 t 只能落在三处之一:
  · 对子/刻子里:P 手里至少要有 1 张 t(单钓、对倒、七对);
  · 顺子里:P 手里要有 (t-2,t-1)、(t-1,t+1)、(t+1,t+2) 三种搭子之一。
所以只要这些需要的牌在 P 手里都"不可能有",t 对 P 就 100% 安全。这个判断是完备的,不依赖任何习惯推测。

P 对某种牌的可持有上限 = 未现张数 − 别人一定有的张数(已胡者必然持有、你换给别家还没出手的牌);
P 缺门花色的牌按 0 算(P 胡牌时手里不能有缺门牌)。
另外,如果确定 P 手里还拿着缺门牌(你换给他、他还没打出),他在打出之前不能胡,任何牌对他都 100% 安全。
"""
from __future__ import annotations

import math

from .counting import TileCounts
from .opponent_read import OpponentRead, describe_tile_risk, estimate_dealin_prob, read_all_opponents
from .report import Finding, certain, counted, probable
from .state import AssistantState, PlayerView
from .tiles import label, rank_of, seat_name, suit_label, suit_of, tidx
from .winner_inference import WinnerInference


def _comb_real(n: float, k: int) -> float:
    if k < 0 or n < k - 1e-9:
        return 0.0
    if k == 0:
        return 1.0
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _p_none(pool: float, count: int, k_absent: float) -> float:
    """从 pool 张里随机拿 count 张,一张都不属于某 k_absent 张的概率(允许非整数,作平滑近似)。"""
    denominator = _comb_real(pool, count)
    if denominator <= 0:
        return 1.0
    return min(1.0, _comb_real(pool - k_absent, count) / denominator)


def wait_shapes(index: int) -> list[tuple[str, tuple[int, ...]]]:
    """能以这张牌为胡牌张的等待形状,以及 P 手里需要的牌。"""
    r = rank_of(index)
    base = index - (r - 1)  # 该花色 1 的编号
    shapes: list[tuple[str, tuple[int, ...]]] = [("单钓/对倒/七对", (index,))]
    if r >= 3:
        shapes.append(("边张" if r == 3 else "两面", (base + r - 3, base + r - 2)))
    if 2 <= r <= 8:
        shapes.append(("嵌张", (base + r - 2, base + r)))
    if r <= 7:
        shapes.append(("边张" if r == 7 else "两面", (base + r, base + r + 1)))
    return shapes


Floors = dict[int, tuple[int, ...]]


def holding_caps(
    state: AssistantState,
    counts: TileCounts,
    winners: list[WinnerInference],
    target: PlayerView,
    floors: Floors | None = None,
) -> list[int]:
    locked = [0] * 27
    for inference in winners:  # 已胡者的 certain 已包含他的换牌下限
        for t in range(27):
            locked[t] += inference.certain[t]
    winner_seats = {inference.seat for inference in winners}
    for seat, floor in (floors or {}).items():
        if seat != target.seat and seat not in winner_seats:
            for t in range(27):
                locked[t] += floor[t]
    caps = [max(0, counts.unseen[t] - locked[t]) for t in range(27)]
    if target.void_suit is not None:
        for t in range(27):
            if suit_of(t) == target.void_suit:
                caps[t] = 0
    return caps


def random_pool(counts: TileCounts, winners: list[WinnerInference], floors: Floors | None = None) -> tuple[list[float], float]:
    """随机假设下,在牌墙和在场对手暗手牌之间分布的牌:扣掉已胡者的期望持有,以及在场对手一定有的牌。"""
    expected = [0.0] * 27
    for inference in winners:
        for t in range(27):
            expected[t] += inference.expected[t]
    winner_seats = {inference.seat for inference in winners}
    for seat, floor in (floors or {}).items():
        if seat not in winner_seats:
            for t in range(27):
                expected[t] += floor[t]
    pool = [max(0.0, counts.unseen[t] - expected[t]) for t in range(27)]
    return pool, sum(pool)


def estimate_wall_expected(
    state: AssistantState,
    counts: TileCounts,
    winners: list[WinnerInference],
    floors: Floors | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """结合已胡者反推、换牌已知底牌与在场对手定缺清门进度，用比例迭代(IPF)精确估算 27 种牌的去向。

    返回 (wall_exp, winner_exp, active_opp_exp)，三者均为长度 27 的列表，满足张数守恒：
    - wall_exp[t] + winner_exp[t] + active_opp_exp[t] ≈ counts.unseen[t]
    - sum(wall_exp) ≈ counts.wall
    当有在场对手定缺某门且已打出非缺门牌（大概率已清完缺门）时，该门未现牌会更真实地汇聚到牌墙中。
    """
    winner_exp = [0.0] * 27
    for inf in winners:
        for t in range(27):
            winner_exp[t] += inf.expected[t]

    winner_seats = {inf.seat for inf in winners}
    locked_floor = [0.0] * 27
    opp_floor_sum: dict[int, int] = {}
    for seat, fl in (floors or {}).items():
        if seat not in winner_seats:
            s_fl = 0
            for t in range(27):
                if fl[t]:
                    locked_floor[t] += fl[t]
                    s_fl += fl[t]
            opp_floor_sum[seat] = s_fl

    for t in range(27):
        cap_w = max(0.0, float(counts.unseen[t]) - locked_floor[t])
        if winner_exp[t] > cap_w:
            winner_exp[t] = cap_w

    pool = [max(0.0, counts.unseen[t] - winner_exp[t] - locked_floor[t]) for t in range(27)]
    pool_total = sum(pool)
    if pool_total <= 1e-9 or counts.wall <= 0:
        wall_exp = [0.0] * 27
        active_opp_exp = [max(0.0, counts.unseen[t] - winner_exp[t]) for t in range(27)]
        return wall_exp, winner_exp, active_opp_exp

    # 收集牌墙 + 各在场对手(扣除已知底牌后)的未知张数与定缺先验权重
    row_targets: list[float] = [float(counts.wall)]
    row_priors: list[list[float]] = [[1.0] * 27]
    has_void_bias = False

    for p in state.active_opponents():
        free_h = max(0, counts.hidden.get(p.seat, 0) - opp_floor_sum.get(p.seat, 0))
        if free_h <= 0:
            continue
        vs = p.void_suit
        alpha_void = 1.0
        if vs is not None and state.phase == "play":
            non_void_disc = sum(1 for d in p.discards if d[-1] != vs)
            if non_void_disc >= 2:
                alpha_void = 0.02
            elif non_void_disc == 1:
                alpha_void = 0.08
            elif len(p.discards) >= 1:
                alpha_void = 0.35
        if alpha_void < 1.0:
            has_void_bias = True
        row_targets.append(float(free_h))
        row_priors.append([alpha_void if (vs is not None and suit_of(t) == vs) else 1.0 for t in range(27)])

    if len(row_targets) == 1:
        # 场上已无其他在场对手的自由暗牌 -> 剩余 pool 全在牌墙
        return list(pool), winner_exp, list(locked_floor)

    if not has_void_bias:
        ratio = counts.wall / pool_total
        wall_exp = [pool[t] * ratio for t in range(27)]
        active_opp_exp = [locked_floor[t] + pool[t] * (1.0 - ratio) for t in range(27)]
        return wall_exp, winner_exp, active_opp_exp

    # 归一化行目标总和对齐 pool_total(防浮点微小误差)
    t_sum = sum(row_targets)
    if t_sum > 1e-9 and abs(t_sum - pool_total) > 1e-6:
        scale = pool_total / t_sum
        row_targets = [x * scale for x in row_targets]

    n_rows = len(row_targets)
    mat = [[row_priors[r][t] * pool[t] for t in range(27)] for r in range(n_rows)]
    for _ in range(30):
        # 行缩放: 对齐牌墙与各在场对手未知张数
        max_err = 0.0
        for r in range(n_rows):
            rs = sum(mat[r])
            err = abs(rs - row_targets[r])
            if err > max_err:
                max_err = err
            if rs > 1e-12:
                fac = row_targets[r] / rs
                row = mat[r]
                for t in range(27):
                    row[t] *= fac
        # 列缩放: 对齐每种牌的 pool[t]
        for t in range(27):
            pt = pool[t]
            if pt <= 1e-12:
                for r in range(n_rows):
                    mat[r][t] = 0.0
                continue
            cs = sum(mat[r][t] for r in range(n_rows))
            if cs > 1e-12:
                fac = pt / cs
                for r in range(n_rows):
                    mat[r][t] *= fac
        if max_err < 1e-4:
            break

    wall_exp = mat[0]
    active_opp_exp = [locked_floor[t] + sum(mat[r][t] for r in range(1, n_rows)) for t in range(27)]
    return wall_exp, winner_exp, active_opp_exp


def analyze_danger(
    state: AssistantState,
    counts: TileCounts,
    winners: list[WinnerInference],
    floors: Floors | None = None,
    opponent_reads: dict[int, OpponentRead] | None = None,
) -> tuple[list[dict], list[Finding]]:
    floors = floors or {}
    reads = opponent_reads if opponent_reads is not None else read_all_opponents(state, floors)
    zero = (0,) * 27
    own_tiles = sorted({tidx(tile) for tile in state.own_hand})
    opponents = state.active_opponents()
    pool, pool_total = random_pool(counts, winners, floors)
    caps_by_seat = {view.seat: holding_caps(state, counts, winners, view, floors) for view in opponents}
    locked_any = any(any(inference.certain) for inference in winners) or len(floors) > 0
    holds_void = {}
    for view in opponents:
        floor = floors.get(view.seat, zero)
        if view.void_suit is not None:
            void_tiles = [t for t in range(27) if floor[t] and suit_of(t) == view.void_suit]
            if void_tiles:
                holds_void[view.seat] = void_tiles

    rows: list[dict] = []
    findings: list[Finding] = []
    for index in own_tiles:
        per_opponent = []
        safe_seats = []
        not_dealin_all = 1.0
        for view in opponents:
            who = seat_name(view.seat, state.perspective)
            opp_read = reads[view.seat]
            if view.void_suit is not None and suit_of(index) == view.void_suit:
                per_opponent.append({"seat": view.seat, "name": who, "safe": True, "reason": "void", "shapes": [], "dealin_prob": 0.0, "risk_tags": []})
                safe_seats.append(view.seat)
                continue
            if view.seat in holds_void:
                per_opponent.append(
                    {
                        "seat": view.seat,
                        "name": who,
                        "safe": True,
                        "reason": "holds_void",
                        "shapes": [],
                        "void_tiles": [label(t) for t in holds_void[view.seat]],
                        "dealin_prob": 0.0,
                        "risk_tags": [],
                    }
                )
                safe_seats.append(view.seat)
                continue
            caps = caps_by_seat[view.seat]
            floor = floors.get(view.seat, zero)
            free_count = view.hidden_count - sum(floor)  # 他暗手牌里随机来的部分
            shapes = []
            for name, needed in wait_shapes(index):
                need_counts: dict[int, int] = {}
                for t in needed:
                    need_counts[t] = need_counts.get(t, 0) + 1
                # 确定性剪枝: 若对手暗手牌总张数不足以放下该形状(例如 4 副露仅剩 1 张暗牌不能凑 2 张顺子搭), 必然不可能
                possible = (len(needed) <= view.hidden_count) and all(caps[t] >= n for t, n in need_counts.items())
                shape = {
                    "name": name,
                    "tiles": [label(t) for t in needed],
                    "possible": possible,
                    "basis": (
                        [f"对手仅剩 {view.hidden_count} 张暗手牌,凑不出 {len(needed)} 张搭子"]
                        if len(needed) > view.hidden_count
                        else [f"{label(t)}可持有上限 {caps[t]} 张" for t in sorted(set(needed))]
                    ),
                }
                # 一定有的牌已经满足的部分,不用再从随机部分里凑
                missing = [t for t, n in need_counts.items() if floor[t] < n]
                if possible and not missing:
                    shape["hold_prob"] = 1.0
                    shape["known"] = True
                    shape["basis"].append(f"他手里一定有{'、'.join(label(t) for t in sorted(set(needed)))}(你换给他、还没出手)")
                elif possible and free_count > 0 and pool_total > 0:
                    if len(missing) == 1:
                        prob = 1.0 - _p_none(pool_total, free_count, pool[missing[0]])
                    else:
                        a, b = missing
                        prob = (
                            1.0
                            - _p_none(pool_total, free_count, pool[a])
                            - _p_none(pool_total, free_count, pool[b])
                            + _p_none(pool_total, free_count, pool[a] + pool[b])
                        )
                    shape["hold_prob"] = max(0.0, min(1.0, prob))
                shapes.append(shape)
            possible_shapes = [shape for shape in shapes if shape["possible"]]
            entry = {"seat": view.seat, "name": who, "safe": not possible_shapes, "reason": "blocked" if not possible_shapes else None, "shapes": shapes}
            if possible_shapes:
                none_prob = 1.0
                for shape in possible_shapes:
                    none_prob *= 1.0 - shape.get("hold_prob", 0.0)
                entry["any_shape_prob"] = 1.0 - none_prob
                dealin_p, risk_tags = estimate_dealin_prob(index, opp_read, counts, possible_shapes)
                entry["dealin_prob"] = round(dealin_p, 4)
                entry["risk_tags"] = risk_tags
                not_dealin_all *= 1.0 - dealin_p
            else:
                entry["dealin_prob"] = 0.0
                entry["risk_tags"] = []
                safe_seats.append(view.seat)
            per_opponent.append(entry)

        safe_all = bool(opponents) and len(safe_seats) == len(opponents)
        total_dealin = 0.0 if (safe_all or not opponents) else round(1.0 - not_dealin_all, 4)
        rows.append(
            {
                "tile": label(index),
                "per_opponent": per_opponent,
                "safe_all": safe_all,
                "total_dealin_prob": total_dealin,
                "risk_label": describe_tile_risk(total_dealin),
            }
        )

        for entry in per_opponent:
            if not entry["safe"]:
                continue
            if entry["reason"] == "void":
                findings.append(
                    certain(
                        "danger",
                        f"{label(index)}对{entry['name']} 100% 安全",
                        [f"{entry['name']}定缺{suit_label(suit_of(index))},胡牌时手里不能有{suit_label(suit_of(index))},用不了这张"],
                        tile=label(index),
                        seat=entry["seat"],
                    )
                )
            elif entry["reason"] == "holds_void":
                findings.append(
                    certain(
                        "danger",
                        f"{label(index)}对{entry['name']} 100% 安全",
                        [
                            f"你换给{entry['name']}的{'、'.join(entry['void_tiles'])}是他的缺门,他还没打出,一定还在他手里",
                            "手里有缺门牌不能胡,他打出之前,任何牌对他都安全",
                        ],
                        tile=label(index),
                        seat=entry["seat"],
                    )
                )
            else:
                basis = [f"{shape['name']}({'、'.join(shape['tiles'])})不可能:{','.join(shape['basis'])}" for shape in entry["shapes"]]
                if locked_any:
                    basis.append("可持有上限已扣除别人一定有的牌(已胡者必然持有、你换给别家还没出手的牌)")
                findings.append(certain("danger", f"{label(index)}对{entry['name']} 100% 安全", basis, tile=label(index), seat=entry["seat"]))
        if opponents and not safe_all:
            for entry in per_opponent:
                if entry["safe"] or "any_shape_prob" not in entry:
                    continue
                possible = [s for s in entry["shapes"] if s["possible"]]
                findings.append(
                    counted(
                        "danger",
                        f"{label(index)}对{entry['name']}:还有 {len(possible)} 种等待形状可能",
                        [f"{s['name']}({'、'.join(s['tiles'])}):{','.join(s['basis'])}" for s in possible],
                        tile=label(index),
                        seat=entry["seat"],
                    )
                )
                findings.append(
                    probable(
                        "danger",
                        f"{label(index)}对{entry['name']}:他手里有能等这张的搭子的概率约 {entry['any_shape_prob']:.0%}",
                        [f"{s['name']}({'、'.join(s['tiles'])})约 {s.get('hold_prob', 0.0):.0%}" for s in possible]
                        + ["这是\"手里有对应搭子\"的概率,不等于他已经听牌,也不等于点炮概率", "多种形状合并时按相互独立近似"],
                        tile=label(index),
                        seat=entry["seat"],
                        prob=entry["any_shape_prob"],
                    )
                )
    return rows, findings
