"""开局:换三张(基于 280 万局模拟校准的 swap_v2 盘面评估)、定缺;对局中:换牌去向追踪。"""
from __future__ import annotations

from engine.hand import Hand
from engine.tiles import SUITS

from assistant.known_holdings import floor_text, swap_floors
from assistant.report import Finding, certain, counted, probable
from assistant.state import AssistantState
from assistant.swap_v2 import choose_void_v2, score_swap_candidates_v2
from assistant.tiles import label, labels, seat_name, suit_label, suit_of, tidx
from assistant.void_v3 import score_void_candidates_v3
from state.hand_analysis import shanten

CRITERION_SWAP = "按换后四家定缺盘面综合代价从低到高排序(综合换后向听、缺门包袱与防卡壳退路、清一色/根潜力及送牌防守代价)"
CRITERION_VOID = "先看缺掉这门后剩下的牌的向听数(少的在前),再看这门的张数(少的在前,打光得快)"
CRITERION_VOID_V3 = "综合缺后向听、缺门清牌包袱、留牌有效进张宽度、暗刻/暗杠根基，并在平手胶着局面结合换牌流向防单行道"
ASSUMPTION_SWAP_V2 = (
    "按 280 万局正常玩家模拟标定的进牌花色倾向(我方某门牌越多,外池对手越易将其作短门换入)及拆牌模板估算换后定缺盘面。"
)
ASSUMPTION_VOID_V3 = (
    "按 4,000 局定缺瞬间供需与 1,750 局全盘配对大样本校准：优先保证向听与有效进张宽度，并在平手局面用换牌流向降低放单行道概率。"
)


def _swap_context(state: AssistantState, hand_size: int) -> tuple[bool, bool, str | None]:
    is_dealer = (state.dealer == state.perspective) if state.dealer is not None else (hand_size == 14)
    if state.swap_direction is None:
        return is_dealer, False, None
    recv_seat = (state.perspective - state.swap_direction) % 4
    recv_name = seat_name(recv_seat, state.perspective)
    recv_is_dealer = (
        (state.dealer == recv_seat)
        if state.dealer is not None
        else (state.players[recv_seat].hidden_count == 14)
    )
    return is_dealer, recv_is_dealer, recv_name


def analyze_opening(state: AssistantState) -> tuple[dict, list[Finding]]:
    hand = Hand.from_strings(list(state.own_hand))
    findings: list[Finding] = []
    suit_counts = {suit.value: sum(1 for t in hand.tiles() if t.suit is suit) for suit in SUITS}
    info: dict = {"suit_counts": {suit_label(k): v for k, v in suit_counts.items()}}
    is_dealer, recv_is_dealer, recv_name = _swap_context(state, hand.size)

    if state.phase == "swap_three":
        scored = score_swap_candidates_v2(
            hand.counts,
            is_dealer=is_dealer,
            receiver_is_dealer=recv_is_dealer,
            swap_direction=state.swap_direction,
        )
        rows = []
        for item in scored:
            rows.append(
                {
                    "tiles": [label(idx) for idx in item["tile_indices"]],
                    "suit": suit_label(item["suit_code"]),
                    "shanten_rest": item["shanten_rest"],
                    "suit_left": item["suit_left"],
                    "total_cost": round(item["total_cost"], 3),
                    "self_cost": round(item["self_cost"], 3),
                    "gift_cost": round(item["gift_cost"], 3),
                    "gift_shape": item["gift_shape"],
                    "exp_shanten": item["exp_shanten"],
                    "exp_void_cnt": item["exp_void_cnt"],
                    "exp_live": item["exp_live"],
                    "exp_jam_rate": item["exp_jam_rate"],
                    "exp_flush_rate": item["exp_flush_rate"],
                }
            )

        conflict_info = None
        conflict_text = None
        if rows:
            top = rows[0]
            # 1) 搏清一色备选路线检测
            flush_cands = [
                r for r in rows[1:]
                if r["total_cost"] <= top["total_cost"] + 0.35
                and r["exp_flush_rate"] - top["exp_flush_rate"] >= 0.18
            ]
            # 2) 纯防守不喂牌备选路线检测(相对当前首选可明显降低送牌风险且盘面差距很小)
            safe_cands = [
                r for r in rows[1:]
                if top["gift_cost"] - r["gift_cost"] >= 0.045
                and r["total_cost"] <= top["total_cost"] + 0.12
            ]
            if flush_cands:
                alt = max(flush_cands, key=lambda r: (r["exp_flush_rate"], -r["total_cost"]))
                conflict_info = {"type": "flush", "main": top["tiles"], "alt": alt["tiles"]}
                conflict_text = (
                    f"路线权衡:求稳换 {'、'.join(top['tiles'])}(预计向听{top['exp_shanten']}),"
                    f"搏清一色换 {'、'.join(alt['tiles'])}(清一色苗头约{alt['exp_flush_rate']:.0%})"
                )
            elif safe_cands:
                alt = min(safe_cands, key=lambda r: r["total_cost"])
                who_txt = f"{recv_name}{'庄家' if recv_is_dealer else ''}" if recv_name else "对手"
                conflict_info = {"type": "defense", "main": top["tiles"], "alt": alt["tiles"]}
                conflict_text = (
                    f"攻防权衡:综合换 {'、'.join(top['tiles'])}(送{top['gift_shape']}),"
                    f"重点防{who_txt}可换 {'、'.join(alt['tiles'])}(送{alt['gift_shape']})"
                )

        info.update(
            swap_options=rows,
            swap_criterion=CRITERION_SWAP,
            swap_target=recv_name,
            swap_target_is_dealer=recv_is_dealer,
            swap_conflict=conflict_info,
            swap_conflict_text=conflict_text,
        )

        if rows:
            top = rows[0]
            target_desc = (
                f"本局换给{recv_name}({'14张庄家,已加重防喂权重' if recv_is_dealer else '闲家'})"
                if recv_name
                else "换牌方向未指定(按闲家基准评估)"
            )
            findings.append(
                probable(
                    "opening",
                    f"换三张推荐:换出 {'、'.join(top['tiles'])} (送{top['gift_shape']},换后预计向听 {top['exp_shanten']},预计缺门 {top['exp_void_cnt']} 张)",
                    [
                        f"排序标准:{CRITERION_SWAP}",
                        target_desc,
                    ]
                    + [
                        f"换出 {'、'.join(r['tiles'])}:综合代价 {r['total_cost']:.2f} | 预计向听 {r['exp_shanten']},缺门 {r['exp_void_cnt']} 张(卡壳率 {r['exp_jam_rate']:.0%})"
                        + (f",清一色苗 {r['exp_flush_rate']:.0%}" if r["exp_flush_rate"] >= 0.1 else "")
                        + f" | 送{r['gift_shape']}"
                        for r in rows[:5]
                    ],
                    assumption=ASSUMPTION_SWAP_V2,
                )
            )
            findings.append(
                counted(
                    "opening",
                    f"合法换法共 {len(rows)} 种;首选换出后留牌 {hand.size - 3} 张向听 {top['shanten_rest']},{top['suit']}剩 {top['suit_left']} 张",
                    [f"各门初始张数:{'、'.join(f'{k}{v}张' for k, v in info['suit_counts'].items())}"],
                )
            )
            if conflict_text:
                findings.append(
                    probable(
                        "opening",
                        conflict_text,
                        ["两种方案综合盘面差距不大,可根据当前场况分差选择求稳快胡、搏大番或防守"],
                        assumption=ASSUMPTION_SWAP_V2,
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
        v2_suit = suit_label(choose_void_v2(hand, is_dealer=is_dealer).value)

        swap_out_code = suit_of(tidx(state.swap.given[0])) if (state.swap and state.swap.given) else None
        swap_in_code = suit_of(tidx(state.swap.received[0])) if (state.swap and state.swap.received) else None
        v3_rows = score_void_candidates_v3(
            hand.counts,
            is_dealer=is_dealer,
            swap_out_suit=swap_out_code,
            swap_in_suit=swap_in_code,
            swap_direction=state.swap_direction,
        )
        v3_suit = v3_rows[0]["suit"]

        info.update(
            void_options=rows,
            void_criterion=CRITERION_VOID,
            void_v2_pick=v2_suit,
            void_v3_options=v3_rows,
            void_v3_pick=v3_suit,
            void_v3_criterion=CRITERION_VOID_V3,
        )
        basis = [f"排序标准:{CRITERION_VOID}"] + [
            f"缺{r['suit']}:这门 {r['count']} 张,其余牌向听 {r['shanten_without']}" for r in rows
        ]
        if v3_suit != rows[0]["suit"]:
            basis.append(f"注:综合进张宽度、清牌耗时与换牌流向,void_v3 更推荐缺{v3_suit}")
        findings.append(
            counted(
                "opening",
                f"定缺:按标准排第一的是缺{rows[0]['suit']}",
                basis,
            )
        )
        if state.phase == "declare_void":
            top_v3 = v3_rows[0]
            v3_basis = [f"排序标准:{CRITERION_VOID_V3}"] + [
                f"缺{r['suit']}:综合代价 {r['total_cost']:.2f} | {r['void_cnt']} 张, 其余牌向听 {r['shanten']}, 有效进张 {r['live']} 张"
                + (f" | {r['ctx_note']}" if r["ctx_note"] else "")
                for r in v3_rows
            ]
            findings.append(
                probable(
                    "opening",
                    f"定缺推荐(void_v3): 缺{top_v3['suit']} ({top_v3['void_cnt']}张, 向听 {top_v3['shanten']}, 有效进张 {top_v3['live']} 张)",
                    v3_basis,
                    assumption=ASSUMPTION_VOID_V3,
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
