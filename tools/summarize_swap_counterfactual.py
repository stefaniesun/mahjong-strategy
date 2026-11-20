"""聚合换三张反事实 JSONL，按庄闲、牌数结构和换出花色生成统计报告。

用法:
    python tools/summarize_swap_counterfactual.py \
        outputs/swap_counterfactual.jsonl \
        --out outputs/swap_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _ci(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_half": 0.0}
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean": mean, "ci95_half": 0.0}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return {"mean": mean, "ci95_half": 1.96 * math.sqrt(variance) / math.sqrt(len(values))}


def _round_stats(values: list[float]) -> dict[str, float]:
    stats = _ci(values)
    return {key: round(value, 6) for key, value in stats.items()}


def _bucket(row: dict[str, Any]) -> tuple[str, str, str]:
    counts = "/".join(str(value) for value in sorted(row["suit_counts"].values()))
    role = "dealer" if row["target_is_dealer"] else "non_dealer"
    return role, counts, str(row["swap_direction"])


def _load(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是有效 JSONL") from exc
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    scenario_count: dict[tuple[str, str, str], set[int]] = defaultdict(set)

    for row in rows:
        key_base = _bucket(row)
        candidates = row["candidates"]
        if not candidates:
            continue
        mean_score = sum(float(item["score"]) for item in candidates) / len(candidates)
        mean_receiver = sum(float(item["receiver_score"]) for item in candidates) / len(candidates)
        best_score = max(float(item["score"]) for item in candidates)
        best_net_score = max(float(item.get("net_score", item["score"] - item["receiver_score"])) for item in candidates)
        for item in candidates:
            group_key = (*key_base, item["swap_suit"])
            values = groups[group_key]
            values["score"].append(float(item["score"]))
            values["receiver_score"].append(float(item["receiver_score"]))
            values["net_score"].append(float(item.get("net_score", item["score"] - item["receiver_score"])))
            values["score_delta_within_scenario"].append(float(item["score"]) - mean_score)
            values["receiver_delta_within_scenario"].append(float(item["receiver_score"]) - mean_receiver)
            values["post_shanten"].append(float(item["post_swap"]["shanten"]))
            values["useful_tiles"].append(float(item["post_swap"]["useful_tiles"]))
            values["receiver_shanten_delta"].append(float(item["receiver_shanten_delta"]))
            values["receiver_outgoing_count"].append(float(len(item.get("receiver_swap", []))))
            values["receiver_useful_tiles_delta"].append(float(item["receiver_useful_tiles_delta"]))
            values["receiver_helped"].append(float(item["receiver_helped"]))
            values["outgoing_pair_excess"].append(float(item["outgoing_pair_excess"]))
            values["outgoing_middle_count"].append(float(item["outgoing_middle_count"]))
            values["won"].append(float(bool(item["won"])))
            values["best_score"].append(float(item["score"]) == best_score)
            values["best_net_score"].append(float(item.get("net_score", item["score"] - item["receiver_score"])) == best_net_score)
            scenario_count[key_base].add(int(row["seed"]))

    report_groups = []
    for (role, counts, direction, swap_suit), values in sorted(groups.items()):
        sample_count = len(values["score"])
        report_groups.append({
            "role": role,
            "suit_counts": counts,
            "swap_direction": int(direction),
            "swap_suit": swap_suit,
            "samples": sample_count,
            "scenarios": len(scenario_count[(role, counts, direction)]),
            "score": _round_stats(values["score"]),
            "receiver_score": _round_stats(values["receiver_score"]),
            "net_score": _round_stats(values["net_score"]),
            "score_delta_within_scenario": _round_stats(values["score_delta_within_scenario"]),
            "receiver_delta_within_scenario": _round_stats(values["receiver_delta_within_scenario"]),
            "post_swap_shanten": _round_stats(values["post_shanten"]),
            "useful_tiles": _round_stats(values["useful_tiles"]),
            "receiver_shanten_delta": _round_stats(values["receiver_shanten_delta"]),
            "receiver_outgoing_count": _round_stats(values["receiver_outgoing_count"]),
            "receiver_useful_tiles_delta": _round_stats(values["receiver_useful_tiles_delta"]),
            "receiver_helped_rate": round(sum(values["receiver_helped"]) / sample_count, 6),
            "outgoing_pair_excess": _round_stats(values["outgoing_pair_excess"]),
            "outgoing_middle_count": _round_stats(values["outgoing_middle_count"]),
            "win_rate": round(sum(values["won"]) / sample_count, 6),
            "best_score_rate": round(sum(values["best_score"]) / sample_count, 6),
            "best_net_score_rate": round(sum(values["best_net_score"]) / sample_count, 6),
        })

    return {
        "source_rows": len(rows),
        "scenario_count": len({(row["seed"], row["dealer"], row["target"]) for row in rows}),
        "groups": report_groups,
        "notes": [
            "score_delta_within_scenario 是候选方案相对同一初始场景所有候选方案平均值的配对差值。",
            "当前数据使用固定 RulePolicy 对手，不能直接视为真人策略结论。",
            "ci95_half 是均值 95% 置信区间半宽；样本不足时不应过度解读。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="聚合换三张反事实结果")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = summarize(_load(args.input))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {args.out}，共 {len(report['groups'])} 个统计分组")


if __name__ == "__main__":
    main()
