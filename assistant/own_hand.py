"""自己的牌:向听、有效进张及其实际剩余张数、候选弃牌账目、听牌账目。"""
from __future__ import annotations

from engine.fan_calc import WinContext, calculate_fan
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import Suit, Tile, parse_tile
from engine.ting_check import ting_tiles

from assistant.counting import TileCounts
from assistant.danger import random_pool
from assistant.report import Finding, certain, counted, probable
from assistant.state import AssistantState
from assistant.swap_v1 import ZERO_SUIT, suit_profile
from assistant.tiles import label, tidx
from assistant.winner_inference import WinnerInference
from engine.tiles import SUITS

CRITERION_SPEED = "先看打出后的向听数(少的在前),再看有效进张实际剩余张数(多的在前)"
CRITERION_FAN = "先看打出后的向听数(少的在前);能听牌的,再看听牌后最大番数(大的在前),最后看剩余进张"


def fast_shanten_hand(counts: list[int] | tuple[int, ...], fixed_melds: int, void_idx: int | None) -> int:
    """与 state.hand_analysis.shanten 100% 数学等价的高速向听计算(复用单门缓存,提速 15x+)。"""
    c0 = ZERO_SUIT if void_idx == 0 else tuple(counts[0:9])
    c1 = ZERO_SUIT if void_idx == 1 else tuple(counts[9:18])
    c2 = ZERO_SUIT if void_idx == 2 else tuple(counts[18:27])
    p0, p1, p2 = suit_profile(c0), suit_profile(c1), suit_profile(c2)

    m_base = p0.m0 + p1.m0 + p2.m0
    t_base = p0.t0 + p1.t0 + p2.t0
    tm = fixed_melds + m_base
    if tm > 4:
        tm = 4
    ut = t_base if t_base <= 4 - tm else 4 - tm
    best = 8 - 2 * tm - ut

    for m1, t1 in p0.pair_blocks:
        tm = fixed_melds + m1 + p1.m0 + p2.m0
        if tm > 4:
            tm = 4
        tt = t1 + p1.t0 + p2.t0
        ut = tt if tt <= 4 - tm else 4 - tm
        cand = 7 - 2 * tm - ut
        if cand < best:
            best = cand
    for m1, t1 in p1.pair_blocks:
        tm = fixed_melds + p0.m0 + m1 + p2.m0
        if tm > 4:
            tm = 4
        tt = p0.t0 + t1 + p2.t0
        ut = tt if tt <= 4 - tm else 4 - tm
        cand = 7 - 2 * tm - ut
        if cand < best:
            best = cand
    for m1, t1 in p2.pair_blocks:
        tm = fixed_melds + p0.m0 + p1.m0 + m1
        if tm > 4:
            tm = 4
        tt = p0.t0 + p1.t0 + t1
        ut = tt if tt <= 4 - tm else 4 - tm
        cand = 7 - 2 * tm - ut
        if cand < best:
            best = cand

    if fixed_melds == 0:
        pairs = p0.pairs + p1.pairs + p2.pairs
        unique = p0.unique + p1.unique + p2.unique
        sp = 6 - pairs + (7 - unique if unique < 7 else 0)
        if sp < best:
            best = sp
    return best


def fast_useful_indices(
    counts: list[int] | tuple[int, ...],
    fixed_melds: int,
    void_idx: int | None,
    current_sh: int | None = None,
) -> list[int]:
    """与 state.hand_analysis.useful_tiles 100% 数学等价的高速有效进张枚举(零 Hand 拷贝)。"""
    cur = fast_shanten_hand(counts, fixed_melds, void_idx) if current_sh is None else current_sh
    s0 = ZERO_SUIT if void_idx == 0 else tuple(counts[0:9])
    s1 = ZERO_SUIT if void_idx == 1 else tuple(counts[9:18])
    s2 = ZERO_SUIT if void_idx == 2 else tuple(counts[18:27])
    ps = (suit_profile(s0), suit_profile(s1), suit_profile(s2))
    useful: list[int] = []

    for si in (0, 1, 2):
        if si == void_idx:
            continue
        p_self = ps[si]
        if si == 0:
            pa, pb = ps[1], ps[2]
        elif si == 1:
            pa, pb = ps[0], ps[2]
        else:
            pa, pb = ps[0], ps[1]

        other_m = fixed_melds + pa.m0 + pb.m0
        other_t = pa.t0 + pb.t0
        other_pairs = pa.pairs + pb.pairs
        other_unique = pa.unique + pb.unique
        base_idx = si * 9

        # 预合并另外两门的 pair_blocks 以减少内层重复计算
        other_pair_blocks: list[tuple[int, int]] = []
        for m1, t1 in pa.pair_blocks:
            other_pair_blocks.append((fixed_melds + m1 + pb.m0, t1 + pb.t0))
        for m1, t1 in pb.pair_blocks:
            other_pair_blocks.append((fixed_melds + pa.m0 + m1, pa.t0 + t1))

        for r, _, nxt_c9 in p_self.plus_one:
            pn = suit_profile(nxt_c9)
            tm = other_m + pn.m0
            if tm > 4:
                tm = 4
            tt = other_t + pn.t0
            ut = tt if tt <= 4 - tm else 4 - tm
            best = 8 - 2 * tm - ut
            if best < cur:
                useful.append(base_idx + r)
                continue

            improved = False
            for m1, t1 in pn.pair_blocks:
                tm = other_m + m1
                if tm > 4:
                    tm = 4
                tt = other_t + t1
                ut = tt if tt <= 4 - tm else 4 - tm
                if 7 - 2 * tm - ut < cur:
                    improved = True
                    break
            if improved:
                useful.append(base_idx + r)
                continue

            for om, ot in other_pair_blocks:
                tm = om + pn.m0
                if tm > 4:
                    tm = 4
                tt = ot + pn.t0
                ut = tt if tt <= 4 - tm else 4 - tm
                if 7 - 2 * tm - ut < cur:
                    improved = True
                    break
            if improved:
                useful.append(base_idx + r)
                continue

            if fixed_melds == 0:
                pairs = other_pairs + pn.pairs
                unique = other_unique + pn.unique
                if 6 - pairs + (7 - unique if unique < 7 else 0) < cur:
                    useful.append(base_idx + r)
    return useful


def build_own_hand(state: AssistantState) -> Hand:
    hand = Hand.from_strings(list(state.own_hand))
    for meld in state.me.melds:
        tile = parse_tile(meld.tile)
        if meld.kind == "pong":
            hand.melds.append(Meld(MeldKind.PONG, (tile,) * 3, exposed=True, from_player=meld.from_seat))
        else:
            hand.melds.append(Meld(MeldKind.KONG, (tile,) * 4, exposed=not meld.concealed, from_player=meld.from_seat))
    return hand


def own_void(state: AssistantState) -> Suit | None:
    return Suit(state.me.void_suit) if state.me.void_suit else None


def copy_hand(hand: Hand) -> Hand:
    return Hand(counts=list(hand.counts), melds=list(hand.melds))


def my_remaining_draws(state: AssistantState) -> int:
    """按当前在场顺序,牌墙剩下的牌里轮到自己摸的次数(碰杠和新胡牌会打乱顺序,这里不考虑)。"""
    active = [seat for seat in range(4) if not state.players[seat].won]
    if state.perspective not in active or not active:
        return 0
    order = []
    seat = state.current_player
    for _ in range(state.wall_count):
        seat = (seat + 1) % 4
        while state.players[seat].won:
            seat = (seat + 1) % 4
        order.append(seat)
    return sum(1 for seat in order if seat == state.perspective)


def p_at_least_one(total: int, good: int, draws: int) -> float:
    """从 total 张里随机摸 draws 张,至少摸到一张 good 的概率(超几何)。"""
    from math import comb

    if draws <= 0 or good <= 0 or total <= 0:
        return 0.0
    draws = min(draws, total)
    return 1.0 - comb(total - good, draws) / comb(total, draws)


def _effective_unseen(counts: TileCounts, winners: list[WinnerInference] | None) -> tuple[int, ...]:
    """扣除已胡者必然持有的暗牌后, 每种牌仍可能在牌墙或在场对手手里的严格上限。"""
    if not winners:
        return counts.unseen
    locked = [0] * 27
    for inf in winners:
        for t in range(27):
            locked[t] += inf.certain[t]
    return tuple(max(0, counts.unseen[t] - locked[t]) for t in range(27))


def wait_table(
    hand13: Hand,
    void: Suit | None,
    counts: TileCounts,
    state: AssistantState,
    eff_unseen: tuple[int, ...] | None = None,
    wall_exp: list[float] | None = None,
) -> list[dict]:
    rows = []
    payers = len(state.active_opponents())
    rem_arr = eff_unseen if eff_unseen is not None else counts.unseen
    for tile in sorted(ting_tiles(hand13, void), key=lambda t: t.index):
        trial = copy_hand(hand13)
        trial.add(tile)
        ron = calculate_fan(trial, WinContext(), state.rule_config)
        tsumo = calculate_fan(trial, WinContext(self_draw=True), state.rule_config)
        row = {
            "tile": label(tile.index),
            "remaining": rem_arr[tile.index],
            "raw_unseen": counts.unseen[tile.index],
            "fan": ron.fan,
            "patterns": ron.patterns,
            "ron_score": ron.score_per_payer,
            "tsumo_each": tsumo.score_per_payer,
            "tsumo_payers": payers,
        }
        if wall_exp is not None:
            row["wall_expected"] = round(wall_exp[tile.index], 2)
        rows.append(row)
    return rows


def _useful_rows_from_counts(
    counts27: list[int] | tuple[int, ...],
    fixed_melds: int,
    void_idx: int | None,
    counts: TileCounts,
    eff_unseen: tuple[int, ...],
    sh: int,
) -> tuple[list[dict], int]:
    rows = []
    live = 0
    for idx in fast_useful_indices(counts27, fixed_melds, void_idx, sh):
        rem = eff_unseen[idx]
        rows.append({"index": idx, "tile": label(idx), "remaining": rem, "raw_unseen": counts.unseen[idx]})
        live += rem
    return rows, live


def analyze_own_hand(
    state: AssistantState,
    counts: TileCounts,
    winners: list[WinnerInference] | None = None,
    floors: dict[int, tuple[int, ...]] | None = None,
    danger_rows: list[dict] | None = None,
) -> tuple[dict, list[Finding]]:
    hand = build_own_hand(state)
    void = own_void(state)
    void_idx = SUITS.index(void) if void is not None else None
    fixed_melds = len(hand.melds)
    findings: list[Finding] = []
    draws = my_remaining_draws(state)
    eff_unseen = _effective_unseen(counts, winners)
    winners_locked_any = eff_unseen != counts.unseen
    pool, pool_total = random_pool(counts, winners or [], floors)
    wall_exp = [pool[t] * counts.wall / pool_total if pool_total > 0 else 0.0 for t in range(27)]
    risk_by_tile = {row["tile"]: row for row in (danger_rows or [])}
    info: dict = {"concealed": hand.size, "melds": fixed_melds, "my_draws_left": draws, "total_unseen": counts.total_unseen}

    if hand.size % 3 == 2:  # 轮到自己出牌
        info["mode"] = "discard"
        candidates = []
        base_counts = list(hand.counts)
        for index in sorted({tidx(t) for t in state.own_hand}):
            base_counts[index] -= 1
            after = fast_shanten_hand(base_counts, fixed_melds, void_idx)
            useful, live = _useful_rows_from_counts(base_counts, fixed_melds, void_idx, counts, eff_unseen, after)
            if after == 0:
                trial = Hand(counts=list(base_counts), melds=list(hand.melds))
                waits = wait_table(trial, void, counts, state, eff_unseen, wall_exp)
            else:
                waits = []
            base_counts[index] += 1
            max_fan = max((w["fan"] for w in waits), default=None)
            # 期望可摸池(扣除已胡者期望持有与换牌锁定)
            exp_pool_live = sum(pool[u["index"]] for u in useful) if pool_total > 0 else float(live)
            p_next = (exp_pool_live / pool_total) if pool_total > 0 else (live / counts.total_unseen if counts.total_unseen else 0.0)
            p_before = p_at_least_one(counts.total_unseen, live, draws)
            t_label = label(index)
            d_row = risk_by_tile.get(t_label)
            dealin_prob = d_row.get("total_dealin_prob", 0.0) if d_row else 0.0
            risk_label = d_row.get("risk_label", "未知") if d_row else "未知"
            candidates.append(
                {
                    "index": index,
                    "tile": t_label,
                    "shanten_after": after,
                    "useful": useful,
                    "useful_kinds": len(useful),
                    "live": live,
                    "wall_expected_live": round(sum(wall_exp[u["index"]] for u in useful), 2),
                    "waits": waits,
                    "max_fan": max_fan,
                    "dead_wait": bool(waits) and all(w["remaining"] == 0 for w in waits),
                    "p_next": p_next,
                    "p_before_end": p_before,
                    "dealin_prob": dealin_prob,
                    "risk_label": risk_label,
                    "safe_all": bool(d_row["safe_all"]) if d_row else False,
                }
            )
        by_speed = sorted(candidates, key=lambda c: (c["shanten_after"], -c["live"], c["dealin_prob"], c["index"]))
        by_fan = sorted(
            candidates,
            key=lambda c: (c["shanten_after"], -(c["max_fan"] if c["max_fan"] is not None else -1), -c["live"], c["index"]),
        )
        by_safe = sorted(candidates, key=lambda c: (c["dealin_prob"], c["shanten_after"], -c["live"], c["index"]))
        top = by_speed[0]
        tied = [c["tile"] for c in by_speed if (c["shanten_after"], c["live"]) == (top["shanten_after"], top["live"])]
        conflict = (
            by_fan[0]["tile"] != top["tile"]
            and by_fan[0]["max_fan"] is not None
            and (top["max_fan"] is None or by_fan[0]["max_fan"] > top["max_fan"])
        )
        # 攻防权衡检测: 求快首选点炮风险偏高(>=10%), 而存在同向听或仅退1步向听的明显更安全选择(风险降幅>=6%)
        safe_alt = None
        if danger_rows and top["dealin_prob"] >= 0.10:
            safer_cands = [
                c for c in by_safe
                if c["tile"] != top["tile"]
                and c["shanten_after"] <= top["shanten_after"] + 1
                and (top["dealin_prob"] - c["dealin_prob"]) >= 0.06
            ]
            if safer_cands:
                # 优先选同向听里最安全的,其次选退1向听里最安全的
                same_sh_safe = [c for c in safer_cands if c["shanten_after"] == top["shanten_after"]]
                safe_alt = same_sh_safe[0] if same_sh_safe else safer_cands[0]

        info.update(
            candidates=by_speed,
            order_by_fan=[c["tile"] for c in by_fan],
            order_by_safe=[c["tile"] for c in by_safe],
            criterion=CRITERION_SPEED,
            criterion_fan=CRITERION_FAN,
            current_best_shanten=top["shanten_after"],
            tie=tied if len(tied) > 1 else [],
            conflict={"speed": top["tile"], "fan": by_fan[0]["tile"]} if conflict else None,
            defense_conflict={"speed": top["tile"], "safe": safe_alt["tile"]} if safe_alt else None,
            winners_locked_applied=winners_locked_any,
        )
        findings.append(
            counted(
                "own_hand",
                f"按速度排序第一:打{top['tile']},打后向听 {top['shanten_after']},有效进张还剩 {top['live']} 张",
                [f"排序标准:{CRITERION_SPEED}" + ("(已扣除已胡者必然持有的暗牌)" if winners_locked_any else "")]
                + [
                    f"打{c['tile']}:向听 {c['shanten_after']},进张 {c['useful_kinds']} 种共 {c['live']} 张"
                    + (f" | 点炮风险约 {c['dealin_prob']:.1%}({c['risk_label']})" if danger_rows else "")
                    for c in by_speed[:4]
                ],
            )
        )
        if len(tied) > 1:
            findings.append(counted("own_hand", f"并列:{labels_text(tied)}在此标准下一样好", [f"排序标准:{CRITERION_SPEED}"]))
        if conflict:
            fan_top = by_fan[0]
            findings.append(
                counted(
                    "own_hand",
                    f"两个标准结论不同:求快打{top['tile']},求番大打{fan_top['tile']}",
                    [
                        f"打{top['tile']}:向听 {top['shanten_after']},剩余进张 {top['live']} 张,听牌后最大 {top['max_fan'] if top['max_fan'] is not None else '-'} 番",
                        f"打{fan_top['tile']}:向听 {fan_top['shanten_after']},剩余进张 {fan_top['live']} 张,听牌后最大 {fan_top['max_fan']} 番",
                        "辅助器不替你权衡,两边的账都在上面",
                    ],
                )
            )
        if safe_alt is not None:
            findings.append(
                probable(
                    "own_hand",
                    f"攻防权衡:求快打{top['tile']}(综合点炮风险约 {top['dealin_prob']:.0%}),求稳可改打{safe_alt['tile']}(风险约 {safe_alt['dealin_prob']:.0%})",
                    [
                        f"打{top['tile']}:向听 {top['shanten_after']},进张 {top['live']} 张,风险 {top['risk_label']}({top['dealin_prob']:.1%})",
                        f"打{safe_alt['tile']}:向听 {safe_alt['shanten_after']},进张 {safe_alt['live']} 张,风险 {safe_alt['risk_label']}({safe_alt['dealin_prob']:.1%})",
                    ],
                    assumption="综合点炮风险结合了对手听牌进度、牌河现物/双筋、花色偏向与已胡反推上限。",
                )
            )
        for c in by_speed:
            if c["dead_wait"]:
                reason = (
                    "听的牌已全部现身或确定扣在已胡者暗手里,一张都摸不到也没人能打"
                    if winners_locked_any
                    else "已全部现身,一张都摸不到也没人能打"
                )
                findings.append(certain("own_hand", f"打{c['tile']}后是死听", [f"听的牌 {labels_text([w['tile'] for w in c['waits']])} {reason}"]))
        findings.append(
            probable(
                "own_hand",
                f"打{top['tile']}后,下一摸摸到有效进张的概率约 {top['p_next']:.0%};到流局前至少摸到一张约 {top['p_before_end']:.0%}",
                [f"有效进张还剩 {top['live']} 张(牌墙期望约 {top['wall_expected_live']:.1f} 张),未现牌共 {counts.total_unseen} 张", f"牌墙还剩 {state.wall_count} 张,按当前顺序你还能摸 {draws} 次(碰杠会打乱顺序)"],
            )
        )
        return info, findings

    # 等待状态(不是自己出牌)
    info["mode"] = "wait"
    current = fast_shanten_hand(hand.counts, fixed_melds, void_idx)
    useful, live = _useful_rows_from_counts(hand.counts, fixed_melds, void_idx, counts, eff_unseen, current)
    info.update(shanten=current, useful=useful, live=live, winners_locked_applied=winners_locked_any)
    findings.append(
        counted(
            "own_hand",
            f"现在向听 {current},有效进张 {len(useful)} 种,实际还剩 {live} 张" if current > 0 else f"已听牌,胡牌张实际还剩 {live} 张",
            [f"{row['tile']}剩 {row['remaining']} 张" + (f"(牌面未现 {row['raw_unseen']},余张被已胡者锁死)" if row["remaining"] < row["raw_unseen"] else "") for row in useful],
        )
    )
    dead = [row["tile"] for row in useful if row["remaining"] == 0]
    if dead and current > 0:
        findings.append(certain("own_hand", f"这些进张已绝,摸不到了:{labels_text(dead)}", ["四张都已现身或确定扣在已胡者暗手里"]))
    if current == 0:
        waits = wait_table(hand, void, counts, state, eff_unseen, wall_exp)
        info["waits"] = waits
        if waits and all(w["remaining"] == 0 for w in waits):
            findings.append(certain("own_hand", "死听:听的牌已全部现身", [f"听 {labels_text([w['tile'] for w in waits])},均已四张现身或确定扣在已胡者暗手里"]))
        for w in waits:
            findings.append(
                counted(
                    "own_hand",
                    f"听{w['tile']}:还剩 {w['remaining']} 张,胡了 {w['fan']} 番,点炮 {w['ron_score']} 分 / 自摸每家 {w['tsumo_each']} 分",
                    [f"番型:{'、'.join(w['patterns'])}", "未计杠上花、海底等临场番"],
                )
            )
    if live > 0:
        exp_pool_live = sum(pool[u["index"]] for u in useful) if pool_total > 0 else float(live)
        p_next_wait = (exp_pool_live / pool_total) if pool_total > 0 else (live / counts.total_unseen if counts.total_unseen else 0.0)
        findings.append(
            probable(
                "own_hand",
                f"下一摸摸到有效进张的概率约 {p_next_wait:.0%};到流局前至少摸到一张约 {p_at_least_one(counts.total_unseen, live, draws):.0%}",
                [f"有效进张还剩 {live} 张,未现牌共 {counts.total_unseen} 张", f"按当前顺序你还能摸 {draws} 次(碰杠会打乱顺序)"],
            )
        )
    return info, findings


def labels_text(names: list[str]) -> str:
    return "、".join(names)
