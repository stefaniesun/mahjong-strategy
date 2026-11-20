"""开局:换三张、定缺;对局中:换牌去向追踪。"""
from __future__ import annotations

from engine.actions import legal_swap_actions
from engine.hand import Hand
from engine.tiles import SUITS

from assistant.known_holdings import floor_text, swap_floors
from assistant.report import Finding, certain, counted
from assistant.state import AssistantState
from assistant.tiles import label, labels, seat_name, suit_label, suit_of, tidx
from state.hand_analysis import shanten

CRITERION_SWAP = "先看换出后留下的牌的向听数(少的在前),再看换出那门剩下的张数(少的在前,定缺后好打光)"
CRITERION_VOID = "先看缺掉这门后剩下的牌的向听数(少的在前),再看这门的张数(少的在前,打光得快)"


def analyze_opening(state: AssistantState) -> tuple[dict, list[Finding]]:
    hand = Hand.from_strings(list(state.own_hand))
    findings: list[Finding] = []
    suit_counts = {suit.value: sum(1 for t in hand.tiles() if t.suit is suit) for suit in SUITS}
    info: dict = {"suit_counts": {suit_label(k): v for k, v in suit_counts.items()}}

    if state.phase == "swap_three":
        rows = []
        for action in legal_swap_actions(hand):
            rest = Hand(counts=list(hand.counts))
            for tile in action.tiles:
                rest.remove(tile)
            suit = action.tiles[0].suit
            rows.append(
                {
                    "tiles": [label(t.index) for t in action.tiles],
                    "suit": suit_label(suit.value),
                    "shanten_rest": shanten(rest),
                    "suit_left": sum(1 for t in rest.tiles() if t.suit is suit),
                    "_order": tuple(t.index for t in action.tiles),
                }
            )
        rows.sort(key=lambda r: (r["shanten_rest"], r["suit_left"], r["_order"]))
        for row in rows:
            row.pop("_order")
        info.update(swap_options=rows, swap_criterion=CRITERION_SWAP)
        if rows:
            findings.append(
                counted(
                    "opening",
                    f"换三张:按标准排第一的是换出 {'、'.join(rows[0]['tiles'])}",
                    [f"排序标准:{CRITERION_SWAP}", f"共 {len(rows)} 种合法换法;收到的 3 张未知,向听按留下的 {hand.size - 3} 张计算"]
                    + [f"换出 {'、'.join(r['tiles'])}:留牌向听 {r['shanten_rest']},{r['suit']}还剩 {r['suit_left']} 张" for r in rows[:4]],
                )
            )

    if state.phase in ("swap_three", "declare_void"):
        rows = []
        for suit in SUITS:
            rows.append(
                {
                    "suit": suit_label(suit.value),
                    "count": suit_counts[suit.value],
                    "shanten_without": shanten(hand, void_suit=suit),
                }
            )
        rows.sort(key=lambda r: (r["shanten_without"], r["count"]))
        info.update(void_options=rows, void_criterion=CRITERION_VOID)
        findings.append(
            counted(
                "opening",
                f"定缺:按标准排第一的是缺{rows[0]['suit']}",
                [f"排序标准:{CRITERION_VOID}"] + [f"缺{r['suit']}:这门 {r['count']} 张,其余牌向听 {r['shanten_without']}" for r in rows],
            )
        )
    return info, findings


def swap_tracking(state: AssistantState) -> list[Finding]:
    swap = state.swap
    if swap is None or state.phase != "play":
        return []
    findings: list[Finding] = []
    receiver = state.players[swap.given_to]
    who = seat_name(swap.given_to, state.perspective)
    lines = []
    for tile in sorted(set(swap.given), key=tidx):
        lines.append(f"{label(tile)}:你换出 {swap.given.count(tile)} 张,{who}打出过 {receiver.discards.count(tile)} 张")
    findings.append(counted("opening", f"你换给{who}的是 {labels(swap.given)}", lines))
    floor = swap_floors(state).get(swap.given_to)
    if floor is not None:
        basis = [
            "换牌在开局,之后牌离开他手里只能是打出、碰杠拿出或被抢杠,都摆在明面上",
            "按张数扣除:他打出的同种牌就算是他原来的那张,也只会让他手里实际更多,不会更少",
        ]
        if receiver.won:
            basis.append("他已胡牌,这些牌在他扣着的暗手牌里,反推时已作为必有条件")
        findings.append(certain("opening", f"{who}手里一定还有:{floor_text(floor)}(你换给他的,还没出手)", basis))
        if receiver.void_suit is not None and not receiver.won:
            void_held = [t for t, c in enumerate(floor) if c and suit_of(t) == receiver.void_suit]
            if void_held:
                findings.append(
                    certain(
                        "opening",
                        f"{who}还拿着缺门牌 {labels(void_held)},打出之前不能胡",
                        [f"{who}定缺{suit_label(receiver.void_suit)},这些是你换给他的、还没打出", "所以现在打任何牌给他都 100% 安全"],
                    )
                )
    if swap.received and swap.received_from is not None:
        giver = seat_name(swap.received_from, state.perspective)
        findings.append(
            counted(
                "opening",
                f"你从{giver}收到 {labels(swap.received)}",
                [f"说明换牌时{giver}手里有这几张,并且他选择把它们送走"],
            )
        )
    return findings
