"""总入口:对一个决策点生成完整辅助报告。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from assistant.alerts import build_alerts
from assistant.counting import TileCounts, breakdown_text, compute_counts, conservation_errors, seen_breakdown
from assistant.danger import analyze_danger, random_pool
from assistant.known_holdings import swap_floors
from assistant.opening import analyze_opening, swap_tracking
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
    own_info: dict | None = None
    opening_info: dict | None = None
    remaining_rows: list[dict] = field(default_factory=list)
    sections: dict[str, list[Finding]] = field(default_factory=dict)
    winner_items: list[dict] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def flags(self) -> dict[str, Any]:
        info = self.own_info or {}
        return {"tie": info.get("tie") or [], "conflict": info.get("conflict")}

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
            "winners": self.winner_items,
            "remaining": self.remaining_rows,
            "flags": self.flags,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def _hand_text(vec) -> str:
    return " ".join(label(i) + (f"×{int(c)}" if c > 1 else "") for i, c in enumerate(vec) if c)


def _remaining_section(
    state: AssistantState, counts: TileCounts, winners: list[WinnerInference], floors: dict | None = None
) -> tuple[list[dict], list[Finding]]:
    breakdown = seen_breakdown(state)
    pool, pool_total = random_pool(counts, winners, floors)
    rows = []
    for t in range(27):
        rows.append(
            {
                "tile": label(t),
                "seen": counts.seen[t],
                "unseen": counts.unseen[t],
                "exhausted": counts.unseen[t] == 0,
                "seen_from": breakdown_text(breakdown[t]),
                "wall_expected": round(pool[t] * counts.wall / pool_total, 2) if pool_total > 0 else 0.0,
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
                    f"在牌墙和在场对手暗手牌之间随机分布的牌共约 {pool_total:.0f} 张,其中牌墙 {counts.wall} 张",
                    "已胡者手里的部分按反推结果扣除;你换给别家、还没出手的牌一定在他手里,也已扣除",
                ],
            )
        )
    return rows, findings


def _winner_section(state: AssistantState, winners: list[WinnerInference]) -> tuple[list[dict], list[Finding]]:
    items: list[dict] = []
    findings: list[Finding] = []
    for inference in winners:
        record = state.win_of(inference.seat)
        who = seat_name(inference.seat, state.perspective)
        view = state.players[inference.seat]
        how = "自摸" if record.self_draw else ("抢杠胡" if record.robbing_kong else f"点炮(放炮:{seat_name(record.from_seat, state.perspective)})")
        tile_text = label(record.tile) if record.tile else "未知"
        item: dict[str, Any] = {
            "seat": inference.seat,
            "name": who,
            "tile": tile_text,
            "how": how,
            "hidden_count": inference.hidden_count,
            "status": inference.status,
            "joint": inference.joint,
        }
        findings.append(
            certain(
                "winners",
                f"{who}{how}胡了{tile_text},手牌扣着 {inference.hidden_count} 张",
                [f"副露 {len(view.melds)} 组", _void_text(view.void_suit)],
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
            sure = [(t, c) for t, c in enumerate(inference.certain) if c]
            if sure:
                findings.append(
                    certain(
                        "winners",
                        f"{who}手里一定有:{'、'.join(label(t) + (f'×{c}' if c > 1 else '') for t, c in sure)}",
                        ["所有可能组合里都有这些牌", "其他对手对这些牌的可持有上限已相应扣除"],
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
        report.winners = winners
        report.winner_items, sections["winners"] = _winner_section(state, winners)
        report.remaining_rows, sections["remaining"] = _remaining_section(state, counts, winners, floors)
        if not state.me.won and state.own_hand:
            report.danger_rows, sections["danger"] = analyze_danger(state, counts, winners, floors)
            report.own_info, sections["own_hand"] = analyze_own_hand(state, counts)
        sections["alerts"] = build_alerts(state, report.own_info, report.danger_rows)
        tracking = swap_tracking(state)
        if tracking:
            sections["opening"] = tracking
    report.sections = sections
    report.elapsed_ms = (time.perf_counter() - started) * 1000
    return report
