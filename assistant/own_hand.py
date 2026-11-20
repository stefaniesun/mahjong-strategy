"""自己的牌:向听、有效进张及其实际剩余张数、候选弃牌账目、听牌账目。"""
from __future__ import annotations

from engine.fan_calc import WinContext, calculate_fan
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import Suit, Tile, parse_tile
from engine.ting_check import ting_tiles

from assistant.counting import TileCounts
from assistant.report import Finding, certain, counted, probable
from assistant.state import AssistantState
from assistant.tiles import label, tidx
from state.hand_analysis import shanten, useful_tiles

CRITERION_SPEED = "先看打出后的向听数(少的在前),再看有效进张实际剩余张数(多的在前)"
CRITERION_FAN = "先看打出后的向听数(少的在前);能听牌的,再看听牌后最大番数(大的在前),最后看剩余进张"


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


def wait_table(hand13: Hand, void: Suit | None, counts: TileCounts, state: AssistantState) -> list[dict]:
    rows = []
    payers = len(state.active_opponents())
    for tile in sorted(ting_tiles(hand13, void), key=lambda t: t.index):
        trial = copy_hand(hand13)
        trial.add(tile)
        ron = calculate_fan(trial, WinContext(), state.rule_config)
        tsumo = calculate_fan(trial, WinContext(self_draw=True), state.rule_config)
        rows.append(
            {
                "tile": label(tile.index),
                "remaining": counts.unseen[tile.index],
                "fan": ron.fan,
                "patterns": ron.patterns,
                "ron_score": ron.score_per_payer,
                "tsumo_each": tsumo.score_per_payer,
                "tsumo_payers": payers,
            }
        )
    return rows


def _useful_rows(hand: Hand, void: Suit | None, counts: TileCounts) -> tuple[list[dict], int]:
    rows = [{"tile": label(t), "remaining": counts.unseen[tidx(t)]} for t in useful_tiles(hand, void)]
    return rows, sum(row["remaining"] for row in rows)


def analyze_own_hand(state: AssistantState, counts: TileCounts) -> tuple[dict, list[Finding]]:
    hand = build_own_hand(state)
    void = own_void(state)
    findings: list[Finding] = []
    draws = my_remaining_draws(state)
    info: dict = {"concealed": hand.size, "melds": len(hand.melds), "my_draws_left": draws, "total_unseen": counts.total_unseen}

    if hand.size % 3 == 2:  # 轮到自己出牌
        info["mode"] = "discard"
        candidates = []
        for index in sorted({tidx(t) for t in state.own_hand}):
            trial = copy_hand(hand)
            trial.remove(Tile.from_index(index))
            after = shanten(trial, void)
            useful, live = _useful_rows(trial, void, counts)
            waits = wait_table(trial, void, counts, state) if after == 0 else []
            max_fan = max((w["fan"] for w in waits), default=None)
            candidates.append(
                {
                    "index": index,
                    "tile": label(index),
                    "shanten_after": after,
                    "useful": useful,
                    "useful_kinds": len(useful),
                    "live": live,
                    "waits": waits,
                    "max_fan": max_fan,
                    "dead_wait": bool(waits) and all(w["remaining"] == 0 for w in waits),
                    "p_next": live / counts.total_unseen if counts.total_unseen else 0.0,
                    "p_before_end": p_at_least_one(counts.total_unseen, live, draws),
                }
            )
        by_speed = sorted(candidates, key=lambda c: (c["shanten_after"], -c["live"], c["index"]))
        by_fan = sorted(
            candidates,
            key=lambda c: (c["shanten_after"], -(c["max_fan"] if c["max_fan"] is not None else -1), -c["live"], c["index"]),
        )
        top = by_speed[0]
        tied = [c["tile"] for c in by_speed if (c["shanten_after"], c["live"]) == (top["shanten_after"], top["live"])]
        conflict = (
            by_fan[0]["tile"] != top["tile"]
            and by_fan[0]["max_fan"] is not None
            and (top["max_fan"] is None or by_fan[0]["max_fan"] > top["max_fan"])
        )
        info.update(
            candidates=by_speed,
            order_by_fan=[c["tile"] for c in by_fan],
            criterion=CRITERION_SPEED,
            criterion_fan=CRITERION_FAN,
            current_best_shanten=top["shanten_after"],
            tie=tied if len(tied) > 1 else [],
            conflict={"speed": top["tile"], "fan": by_fan[0]["tile"]} if conflict else None,
        )
        findings.append(
            counted(
                "own_hand",
                f"按速度排序第一:打{top['tile']},打后向听 {top['shanten_after']},有效进张还剩 {top['live']} 张",
                [f"排序标准:{CRITERION_SPEED}"]
                + [f"打{c['tile']}:向听 {c['shanten_after']},进张 {c['useful_kinds']} 种共 {c['live']} 张" for c in by_speed[:4]],
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
        for c in by_speed:
            if c["dead_wait"]:
                findings.append(certain("own_hand", f"打{c['tile']}后是死听", [f"听的牌 {labels_text([w['tile'] for w in c['waits']])} 已全部现身,一张都摸不到也没人能打"]))
        findings.append(
            probable(
                "own_hand",
                f"打{top['tile']}后,下一摸摸到有效进张的概率约 {top['p_next']:.0%};到流局前至少摸到一张约 {top['p_before_end']:.0%}",
                [f"有效进张还剩 {top['live']} 张,未现牌共 {counts.total_unseen} 张", f"牌墙还剩 {state.wall_count} 张,按当前顺序你还能摸 {draws} 次(碰杠会打乱顺序)"],
            )
        )
        return info, findings

    # 等待状态(不是自己出牌)
    info["mode"] = "wait"
    current = shanten(hand, void)
    useful, live = _useful_rows(hand, void, counts)
    info.update(shanten=current, useful=useful, live=live)
    findings.append(
        counted(
            "own_hand",
            f"现在向听 {current},有效进张 {len(useful)} 种,实际还剩 {live} 张" if current > 0 else f"已听牌,胡牌张实际还剩 {live} 张",
            [f"{row['tile']}剩 {row['remaining']} 张" for row in useful],
        )
    )
    dead = [row["tile"] for row in useful if row["remaining"] == 0]
    if dead and current > 0:
        findings.append(certain("own_hand", f"这些进张已绝,摸不到了:{labels_text(dead)}", ["四张都已现身"]))
    if current == 0:
        waits = wait_table(hand, void, counts, state)
        info["waits"] = waits
        if waits and all(w["remaining"] == 0 for w in waits):
            findings.append(certain("own_hand", "死听:听的牌已全部现身", [f"听 {labels_text([w['tile'] for w in waits])},都已四张现身"]))
        for w in waits:
            findings.append(
                counted(
                    "own_hand",
                    f"听{w['tile']}:还剩 {w['remaining']} 张,胡了 {w['fan']} 番,点炮 {w['ron_score']} 分 / 自摸每家 {w['tsumo_each']} 分",
                    [f"番型:{'、'.join(w['patterns'])}", "未计杠上花、海底等临场番"],
                )
            )
    if live > 0:
        findings.append(
            probable(
                "own_hand",
                f"下一摸摸到有效进张的概率约 {live / counts.total_unseen:.0%};到流局前至少摸到一张约 {p_at_least_one(counts.total_unseen, live, draws):.0%}",
                [f"有效进张还剩 {live} 张,未现牌共 {counts.total_unseen} 张", f"按当前顺序你还能摸 {draws} 次(碰杠会打乱顺序)"],
            )
        )
    return info, findings


def labels_text(names: list[str]) -> str:
    return "、".join(names)
