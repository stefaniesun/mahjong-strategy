"""规则提醒:过庄、缺门未打完、能不能胡、碰杠的账、海底、流局后果。"""
from __future__ import annotations

from engine.fan_calc import WinContext, calculate_fan
from engine.meld import Meld, MeldKind
from engine.tiles import Tile, parse_tile
from engine.win_check import can_win

from assistant.report import Finding, certain, counted
from assistant.state import AssistantState
from assistant.tiles import label, seat_name, suit_label, tidx
from assistant.own_hand import build_own_hand, copy_hand, own_void
from state.hand_analysis import shanten

LATE_WALL = 16  # 牌墙剩这么多张以内,开始提醒流局后果


def _best_shanten_after_discard(hand, void) -> int:
    best = 99
    for index in {t.index for t in hand.tiles()}:
        trial = copy_hand(hand)
        trial.remove(Tile.from_index(index))
        best = min(best, shanten(trial, void))
    return best


def _kong_income(state: AssistantState) -> int | None:
    """自己收到的杠钱合计;有杠不知道当时收了多少(暗杠时可能已有人胡牌离场)就返回 None,不报数。"""
    total = 0
    for meld in state.me.melds:
        if meld.kind != "kong" or meld.added:
            continue
        if meld.income is None:
            return None
        total += meld.income
    return total


def build_alerts(state: AssistantState, own_info: dict | None, danger_rows: list[dict]) -> list[Finding]:
    findings: list[Finding] = []
    if state.phase != "play" or state.me.won:
        return findings
    hand = build_own_hand(state)
    void = own_void(state)
    config = state.rule_config
    base = config.base_score
    me = state.perspective
    safe_all = {row["tile"]: row["safe_all"] for row in danger_rows}

    if void is not None:
        void_tiles = [t for t in state.own_hand if t.endswith(void.value)]
        if void_tiles:
            findings.append(
                certain(
                    "alert",
                    f"手里还有 {len(void_tiles)} 张缺门牌,打完之前不能胡",
                    [f"你定缺{suit_label(void.value)},还剩 {'、'.join(label(t) for t in void_tiles)}"],
                )
            )

    if state.passed_hu_lock:
        findings.append(
            certain(
                "alert",
                f"过庄中:下次摸牌前,只能胡大于 {state.passed_fan} 番的点炮",
                [f"你刚放弃过一个 {state.passed_fan} 番的胡", "自摸不受限制"],
            )
        )

    my_turn = (
        state.current_player == me
        and state.pending_discard is None
        and state.pending_rob_kong is None
        and hand.size % 3 == 2
    )
    if my_turn and can_win(hand, void):
        result = calculate_fan(hand, WinContext(self_draw=True, after_kong=state.draw_after_kong, haidi=state.draw_was_last), config)
        payers = len(state.active_opponents())
        findings.append(
            certain(
                "alert",
                f"可以自摸:{result.fan} 番,每家付 {result.score_per_payer} 分(在场 {payers} 家)",
                [f"番型:{'、'.join(result.patterns)}"],
            )
        )

    if state.pending_discard is not None and state.pending_discard[0] != me:
        discarder, tile_text = state.pending_discard
        tile = parse_tile(tile_text)
        who = seat_name(discarder, me)
        no_action = shanten(hand, void)
        if hand.count(tile) < 4:
            trial = copy_hand(hand)
            trial.add(tile)
            if can_win(trial, void):
                result = calculate_fan(trial, WinContext(after_kong=state.pending_after_kong, haidi=state.pending_last_wall), config)
                if state.passed_hu_lock and result.fan <= state.passed_fan:
                    findings.append(
                        certain(
                            "alert",
                            f"过庄中不能胡这张:只有 {result.fan} 番,不大于放弃时的 {state.passed_fan} 番",
                            ["过庄规则:下次摸牌前只能胡番数更大的点炮"],
                        )
                    )
                else:
                    findings.append(
                        certain(
                            "alert",
                            f"可以胡{label(tile_text)}:{result.fan} 番,{who}付 {result.score_per_payer} 分",
                            [f"番型:{'、'.join(result.patterns)}"],
                        )
                    )
        if hand.count(tile) >= 2:
            after = copy_hand(hand)
            after.remove(tile)
            after.remove(tile)
            after.melds.append(Meld(MeldKind.PONG, (tile,) * 3, exposed=True, from_player=discarder))
            findings.append(
                counted(
                    "alert",
                    f"碰{label(tile_text)}:碰后打出最优一张向听 {_best_shanten_after_discard(after, void)};不碰现在向听 {no_action}",
                    ["碰会亮出副露,也会跳过中间玩家的摸牌"],
                )
            )
        if hand.count(tile) >= 3 and state.wall_count > 1:
            after = copy_hand(hand)
            for _ in range(3):
                after.remove(tile)
            after.melds.append(Meld(MeldKind.KONG, (tile,) * 4, exposed=True, from_player=discarder))
            findings.append(
                counted(
                    "alert",
                    f"明杠{label(tile_text)}:{who}立即付 {2 * base} 分;杠后向听 {shanten(after, void)}(补牌前),不杠现在向听 {no_action}",
                    ["杠后紧接打出的牌若点炮,这笔杠钱转给胡牌者(杠转水)"],
                )
            )

    if state.pending_rob_kong is not None and state.pending_rob_kong[0] != me:
        kong_player, tile_text = state.pending_rob_kong
        tile = parse_tile(tile_text)
        if hand.count(tile) < 4:
            trial = copy_hand(hand)
            trial.add(tile)
            if can_win(trial, void):
                result = calculate_fan(trial, WinContext(robbing_kong=True), config)
                if not (state.passed_hu_lock and result.fan <= state.passed_fan):
                    findings.append(
                        certain(
                            "alert",
                            f"可以抢杠胡{label(tile_text)}:{result.fan} 番(含抢杠 1 番),{seat_name(kong_player, me)}付 {result.score_per_payer} 分",
                            [f"番型:{'、'.join(result.patterns)}"],
                        )
                    )

    if my_turn and state.wall_count > 1:
        no_kong = _best_shanten_after_discard(hand, void)
        for index in sorted({t.index for t in hand.tiles()}):
            tile = Tile.from_index(index)
            if hand.count(tile) == 4:
                after = copy_hand(hand)
                for _ in range(4):
                    after.remove(tile)
                after.melds.append(Meld(MeldKind.KONG, (tile,) * 4, exposed=False))
                findings.append(
                    counted(
                        "alert",
                        f"暗杠{label(index)}:仍在场的 {len(state.active_opponents())} 家各付 {2 * base} 分(已胡的人不付);杠后向听 {shanten(after, void)}(补牌前),不杠打出最优牌后向听 {no_kong}",
                        ["暗杠不会被抢杠胡", "杠出来的四张算一个根,+1 番"],
                    )
                )
        for meld_index, meld in enumerate(hand.melds):
            tile = meld.tiles[0]
            if meld.kind is MeldKind.PONG and hand.count(tile) >= 1:
                after = copy_hand(hand)
                after.remove(tile)
                after.melds[meld_index] = Meld(MeldKind.KONG, (tile,) * 4, exposed=True, from_player=meld.from_player)
                robbed = "这张对所有在场对手 100% 安全,不会被抢杠" if safe_all.get(label(tile.index)) else "这张不是对所有人都安全,有被抢杠胡的可能"
                findings.append(
                    counted(
                        "alert",
                        f"补杠{label(tile.index)}:不收杠钱;杠后向听 {shanten(after, void)}(补牌前),不杠打出最优牌后向听 {no_kong}",
                        [robbed, "补杠成的四张算一个根,+1 番"],
                    )
                )

    if state.wall_count == 1:
        findings.append(certain("alert", "牌墙只剩最后一张:这时不能杠", ["海底不能杠(杠要补牌,已无牌可补)"]))

    current = None
    if own_info is not None:
        current = own_info.get("current_best_shanten", own_info.get("shanten"))
    if current is not None and current > 0 and 0 < state.wall_count <= LATE_WALL:
        income = _kong_income(state)
        basis = ["流局时没听牌的人,要按查叫赔给每个听牌的人(按对方能胡的最大番型)"]
        if income:
            basis.append(f"你作为杠者收到的 {income} 分杠钱,流局时没听牌会原路退回")
        findings.append(certain("alert", f"牌墙还剩 {state.wall_count} 张,你还没听牌", basis))
    return findings
