"""总入口:对一个决策点生成完整辅助报告。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from assistant.alerts import build_alerts
from assistant.counting import TileCounts, breakdown_text, compute_counts, conservation_errors, seen_breakdown
from assistant.danger import analyze_danger, estimate_wall_expected, random_pool
from assistant.known_holdings import swap_floors
from assistant.opening import analyze_opening, swap_tracking
from assistant.opponent_read import OpponentRead, read_all_opponents
from assistant.own_hand import analyze_own_hand
from assistant.report import Finding, certain, counted, probable
from assistant.state import AssistantState
from assistant.tiles import label, seat_name
from assistant.winner_inference import (
    ENUM_LIMIT,
    STATUS_COMPLETE,
    STATUS_NO_CANDIDATE,
    STATUS_TOO_MANY,
    WinnerInference,
    infer_winners,
)


@dataclass
class Report:
    state: AssistantState
    errors: list[str]
    counts: TileCounts | None = None
    winners: list[WinnerInference] = field(default_factory=list)
    danger_rows: list[dict] = field(default_factory=list)
    opponent_reads: list[dict] = field(default_factory=list)
    own_info: dict | None = None
    opening_info: dict | None = None
    remaining_rows: list[dict] = field(default_factory=list)
    suit_inference: list[dict] = field(default_factory=list)
    sections: dict[str, list[Finding]] = field(default_factory=dict)
    winner_items: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def flags(self) -> dict[str, Any]:
        info = self.own_info or {}
        return {
            "tie": info.get("tie") or [],
            "conflict": info.get("conflict"),
            "defense_conflict": info.get("defense_conflict"),
        }

    def all_findings(self) -> list[Finding]:
        return [finding for findings in self.sections.values() for finding in findings]

    def to_dict(self) -> dict[str, Any]:
        state = self.state
        return {
            "meta": {
                "perspective": state.perspective,
                "phase": state.phase,
                "wall_count": state.wall_count,
                "current_player": state.current_player,
            },
            "errors": list(self.errors),
            "sections": {name: [f.to_dict() for f in findings] for name, findings in self.sections.items()},
            "own_info": self.own_info,
            "opening_info": self.opening_info,
            "danger_rows": self.danger_rows,
            "opponent_reads": self.opponent_reads,
            "winners": self.winner_items,
            "remaining": self.remaining_rows,
            "suit_inference": self.suit_inference,
            "flags": self.flags,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def _hand_text(vec) -> str:
    return " ".join(label(i) + (f"×{int(c)}" if c > 1 else "") for i, c in enumerate(vec) if c)


def _remaining_section(
    state: AssistantState,
    counts: TileCounts,
    winners: list[WinnerInference],
    floors: dict | None = None,
) -> tuple[list[dict], list[dict], list[Finding]]:
    from assistant.tiles import SUIT_ORDER, suit_label

    breakdown = seen_breakdown(state)
    pool, pool_total = random_pool(counts, winners, floors)
    wall_exp, winner_exp, active_opp_exp = estimate_wall_expected(state, counts, winners, floors)
    certain_locked = [sum(inf.certain[t] for inf in winners) for t in range(27)]
    rows = []
    for t in range(27):
        eff_unseen = max(0, counts.unseen[t] - certain_locked[t])
        rows.append(
            {
                "tile": label(t),
                "seen": counts.seen[t],
                "unseen": counts.unseen[t],
                "certain_locked": certain_locked[t],
                "effective_unseen": eff_unseen,
                "exhausted": eff_unseen == 0,
                "seen_from": breakdown_text(breakdown[t]),
                "wall_expected": round(wall_exp[t], 2),
                "winner_expected": round(winner_exp[t], 2),
                "active_opp_expected": round(active_opp_exp[t], 2),
            }
        )

    # 按花色汇总残局推理(定缺集中度 + 已胡者该花色高概率持牌 + 墙内占比 + 1~9去向拆分)
    suit_summary: list[dict] = []
    for s_idx, s_code in enumerate(SUIT_ORDER):
        base = s_idx * 9
        opp_void_seats = []
        for p in state.players:
            if p.seat == state.perspective:
                continue
            if p.void_suit == s_code:
                nm = seat_name(p.seat, state.perspective)
                opp_void_seats.append(f"{nm}(已胡)" if p.won else nm)
        s_unseen = sum(counts.unseen[base + r] for r in range(9))
        s_wall = sum(wall_exp[base + r] for r in range(9))
        s_win = sum(winner_exp[base + r] for r in range(9))
        s_opp = sum(active_opp_exp[base + r] for r in range(9))
        winner_clues: list[dict] = []
        for inf in winners:
            p = state.players[inf.seat]
            if p.void_suit == s_code:
                continue
            s_inf_exp = sum(inf.expected[base + r] for r in range(9))
            rec = state.win_of(inf.seat)
            how = "自摸" if rec.self_draw else ("抢杠胡" if rec.robbing_kong else "点炮胡")
            hot = []
            certain_in_suit = [
                {"tile": label(base + r), "count": int(inf.certain[base + r])}
                for r in range(9)
                if inf.certain[base + r] > 0
            ]
            if inf.hold_prob is not None:
                cands = sorted(
                    ((base + r, inf.hold_prob[base + r], inf.expected[base + r]) for r in range(9) if inf.hold_prob[base + r] >= 0.25 and inf.certain[base + r] == 0),
                    key=lambda x: -x[1],
                )[:4]
                hot = [{"tile": label(t), "prob": round(pr, 3), "expected": round(ex, 2)} for t, pr, ex in cands]
            if s_inf_exp >= 0.35 or hot or certain_in_suit:
                winner_clues.append(
                    {
                        "seat": inf.seat,
                        "name": seat_name(inf.seat, state.perspective),
                        "how": how,
                        "tile": label(rec.tile) if rec.tile else "未知",
                        "candidate_count": int(len(inf.candidates)) if inf.status == STATUS_COMPLETE else None,
                        "suit_expected": round(s_inf_exp, 2),
                        "certain_tiles": certain_in_suit,
                        "hot_tiles": hot,
                    }
                )
        tiles_detail = [
            {
                "rank": r + 1,
                "tile": label(base + r),
                "unseen": counts.unseen[base + r],
                "certain_locked": certain_locked[base + r],
                "effective_unseen": max(0, counts.unseen[base + r] - certain_locked[base + r]),
                "wall_expected": round(wall_exp[base + r], 2),
                "winner_expected": round(winner_exp[base + r], 2),
                "active_opp_expected": round(active_opp_exp[base + r], 2),
            }
            for r in range(9)
        ]
        suit_summary.append(
            {
                "suit": s_code,
                "label": suit_label(s_code),
                "is_my_void": state.me.void_suit == s_code,
                "opp_void_count": len(opp_void_seats),
                "void_seats": opp_void_seats,
                "unseen": s_unseen,
                "wall_expected": round(s_wall, 2),
                "wall_share": round(s_wall / counts.wall, 3) if counts.wall > 0 else 0.0,
                "winner_expected": round(s_win, 2),
                "active_opp_expected": round(s_opp, 2),
                "winner_clues": winner_clues,
                "tiles": tiles_detail,
            }
        )

    findings: list[Finding] = []
    exhausted = counts.exhausted
    if exhausted:
        findings.append(
            certain(
                "remaining",
                f"已绝(四张全部现身):{'、'.join(label(t) for t in exhausted)}",
                [f"{label(t)}:{breakdown_text(breakdown[t])}" for t in exhausted],
                tiles=[label(t) for t in exhausted],
            )
        )
    hidden = "、".join(f"{seat_name(seat, state.perspective)} {count} 张" for seat, count in counts.hidden.items() if count)
    findings.append(
        counted(
            "remaining",
            f"没出现的牌共 {counts.total_unseen} 张:牌墙 {counts.wall} 张,其余在别人手里",
            [f"看不见的暗手牌:{hidden or '无'}"],
        )
    )
    if pool_total > 0 and counts.wall > 0:
        findings.append(
            probable(
                "remaining",
                "每种牌在牌墙里的期望张数见余牌表",
                [
                    f"在牌墙和在场对手暗手牌之间分布的牌共约 {pool_total:.0f} 张,其中牌墙 {counts.wall} 张",
                    "已结合已胡者反推、换牌底牌与在场对手定缺清门进度校正牌墙占比",
                ],
            )
        )
    return rows, suit_summary, findings


def _winner_section(state: AssistantState, winners: list[WinnerInference]) -> tuple[list[dict], list[Finding]]:
    from assistant.tiles import SUIT_ORDER

    items: list[dict] = []
    findings: list[Finding] = []
    for inference in winners:
        record = state.win_of(inference.seat)
        who = seat_name(inference.seat, state.perspective)
        view = state.players[inference.seat]
        how = "自摸" if record.self_draw else ("抢杠胡" if record.robbing_kong else f"点炮(放炮:{seat_name(record.from_seat, state.perspective)})")
        tile_text = label(record.tile) if record.tile else "未知"
        sure = [(t, c) for t, c in enumerate(inference.certain) if c]
        item: dict[str, Any] = {
            "seat": inference.seat,
            "name": who,
            "tile": tile_text,
            "how": how,
            "hidden_count": inference.hidden_count,
            "status": inference.status,
            "joint": inference.joint,
            "certain_tiles": [
                {"tile": label(t), "code": f"{(t % 9) + 1}{SUIT_ORDER[t // 9]}", "count": int(c)} for t, c in sure
            ],
        }
        findings.append(
            certain(
                "winners",
                f"{who}{how}胡了{tile_text},手牌扣着 {inference.hidden_count} 张",
                [f"副露 {len(view.melds)} 组", _void_text(view.void_suit)],
            )
        )
        if sure:
            findings.append(
                certain(
                    "winners",
                    f"{who}手里一定有:{'、'.join(label(t) + (f'×{c}' if c > 1 else '') for t, c in sure)}",
                    ["所有合法胡牌组合均必然包含这些牌(含断张顺子必含推理)", "全局有效余牌与其他对手持有上限已同步扣除"],
                )
            )
        if inference.status == STATUS_COMPLETE:
            order = [int(i) for i in np.argsort(-inference.weights, kind="stable")]
            item["candidates"] = [
                {"hand": _hand_text(inference.candidates[i]), "weight": round(float(inference.weights[i]), 4)} for i in order[:30]
            ]
            item["candidate_count"] = len(inference.candidates)
            findings.append(
                counted(
                    "winners",
                    f"{who}的暗手牌只可能是这 {len(inference.candidates)} 种组合之一",
                    [f"{_hand_text(inference.candidates[i])} + {tile_text}" for i in order[:8]]
                    + (["已和另一名胡牌者做联合约束(两人合计用量不超过未现张数)"] if inference.joint else []),
                )
            )
            hold = sorted(
                ((t, p) for t, p in enumerate(inference.hold_prob or ()) if 0.05 <= p < 0.999),
                key=lambda x: -x[1],
            )[:8]
            if hold:
                item["hold_prob"] = [{"tile": label(t), "prob": round(p, 3)} for t, p in hold]
                findings.append(
                    probable(
                        "winners",
                        f"{who}手里各种牌的概率(按组合数加权)",
                        [f"有{label(t)}:约 {p:.0%}" for t, p in hold],
                    )
                )
        elif inference.status == STATUS_TOO_MANY:
            findings.append(
                counted(
                    "winners",
                    f"{who}可能的组合超过 {ENUM_LIMIT} 种,暂不展开",
                    ["余牌再少一些后会自动展开", "期望持有按未现张数在他的非缺门花色里均摊估算"],
                )
            )
        elif inference.status == STATUS_NO_CANDIDATE:
            findings.append(
                certain(
                    "winners",
                    f"按现有牌面找不到{who}任何能胡的组合,牌面信息可能有误",
                    ["请核对录入的牌河、副露和胡牌张"],
                )
            )
        else:
            findings.append(counted("winners", f"{who}的胡牌张不确定,无法反推", []))
        items.append(item)
    return items, findings


def _void_text(void: str | None) -> str:
    from assistant.tiles import suit_label

    return f"定缺{suit_label(void)}" if void else "定缺未知"


def analyze(state: AssistantState) -> Report:
    started = time.perf_counter()
    errors = conservation_errors(state)
    report = Report(state=state, errors=errors)
    if errors:
        # 牌面不守恒时,任何"确定"结论都可能是错的,只报警不出结论。
        report.sections = {"alerts": [certain("alert", "牌面信息不一致,已停止输出结论", errors)]}
        report.elapsed_ms = (time.perf_counter() - started) * 1000
        return report

    counts = compute_counts(state)
    report.counts = counts
    sections: dict[str, list[Finding]] = {}

    if state.phase in ("swap_three", "declare_void"):
        report.opening_info, sections["opening"] = analyze_opening(state)
    else:
        floors = swap_floors(state)
        winners = infer_winners(state, counts, floors=floors)
        opp_reads = read_all_opponents(state, floors)
        report.winners = winners
        report.opponent_reads = [r.to_dict() for r in opp_reads.values()]
        report.winner_items, sections["winners"] = _winner_section(state, winners)
        report.remaining_rows, report.suit_inference, sections["remaining"] = _remaining_section(state, counts, winners, floors)
        if not state.me.won and state.own_hand:
            report.danger_rows, danger_findings = analyze_danger(state, counts, winners, floors, opponent_reads=opp_reads)
            # 在安全度面板顶部置入对手听牌进度与大牌型预警摘要
            read_findings: list[Finding] = []
            for r in opp_reads.values():
                if r.holds_known_void:
                    continue
                if r.threats or r.tenpai_prob >= 0.45:
                    threat_txt = f" | 警戒:{'、'.join(r.threats)}" if r.threats else ""
                    read_findings.append(
                        probable(
                            "danger",
                            f"{r.name}听牌进度估计约 {r.tenpai_prob:.0%}{threat_txt}",
                            list(r.basis),
                            seat=r.seat,
                            prob=r.tenpai_prob,
                            assumption="基于副露数、非缺门弃牌节奏与牌河花色偏向在大样本对局上标定的经验概率。",
                        )
                    )
            sections["danger"] = read_findings + danger_findings
            report.own_info, sections["own_hand"] = analyze_own_hand(
                state, counts, winners=winners, floors=floors, danger_rows=report.danger_rows, opp_reads=opp_reads
            )
        sections["alerts"] = build_alerts(
            state,
            report.own_info,
            report.danger_rows,
            counts=counts,
            winners=winners,
            opp_reads=opp_reads,
        )
        tracking = swap_tracking(state)
        if tracking:
            sections["opening"] = tracking
    report.sections = sections
    report.elapsed_ms = (time.perf_counter() - started) * 1000
    return report
