"""辅助器大规模对账:引擎对局 + 上帝视角逐条核对,输出 docs/assistant_v1_audit.md。

用法:
    python tools/assistant_audit.py --games 500 --workers 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import ActionKind
from engine.hand import Hand
from engine.tiles import Tile, parse_tile, tile_to_str
from engine.win_check import can_win

from assistant.adapters.from_engine import build_state
from assistant.analyzer import analyze
from assistant.counting import compute_counts
from assistant.driver import run_game
from assistant.known_holdings import swap_floors
from assistant.tiles import from_label, label, tidx

BINS = 10


def _bucket() -> dict:
    return {"pred": [0.0] * BINS, "hit": [0.0] * BINS, "n": [0] * BINS}


def _add(bucket: dict, p: float, outcome: bool) -> None:
    i = min(BINS - 1, int(p * BINS))
    bucket["pred"][i] += p
    bucket["hit"][i] += 1.0 if outcome else 0.0
    bucket["n"][i] += 1


def _merge_bucket(a: dict, b: dict) -> None:
    for key in ("pred", "hit", "n"):
        a[key] = [x + y for x, y in zip(a[key], b[key])]


def _god_hidden_counts(game_state, recorder, perspective):
    counts = [0] * 27
    for tile in game_state.wall:
        counts[tile.index] += 1
    wins = {r.seat: r for r in recorder.wins}
    for seat in range(4):
        if seat == perspective:
            continue
        hand = list(game_state.hands[seat].counts)
        record = wins.get(seat)
        if record is not None and record.self_draw and record.tile is not None:
            hand[tidx(record.tile)] -= 1
        for i, c in enumerate(hand):
            counts[i] += c
    return counts


def _true_hidden(game_state, recorder, seat):
    hand = list(game_state.hands[seat].counts)
    record = next(r for r in recorder.wins if r.seat == seat)
    if record.self_draw and record.tile is not None:
        hand[tidx(record.tile)] -= 1
    return hand


def audit_game(seed: int) -> dict:
    st = {
        "games": 1,
        "decisions": 0,
        "rob_kong_games": 0,
        "conservation_errors": 0,
        "count_mismatch": 0,
        "count_checks": 0,
        "exhausted_claims": 0,
        "safe_claims": 0,
        "safe_holds_void": 0,
        "floor_checks": 0,
        "floor_violations": 0,
        "known_shapes": 0,
        "known_shape_violations": 0,
        "swap_decisions": 0,
        "safe_without_swap": 0,
        "safe_with_swap": 0,
        "safe_violations": 0,
        "safe_void": 0,
        "safe_blocked": 0,
        "impossible_shapes": 0,
        "impossible_violations": 0,
        "winner_checks": 0,
        "winner_recall_miss": 0,
        "winner_status": defaultdict(int),
        "certain_checks": 0,
        "certain_violations": 0,
        "discard_decisions": 0,
        "tie": 0,
        "conflict": 0,
        "ms": [],
        "wall_abs_err": 0.0,
        "wall_signed_err": 0.0,
        "wall_naive_abs_err": 0.0,
        "wall_samples": 0,
        "cal_winner": _bucket(),
        "cal_shape": _bucket(),
        "cal_next_draw": _bucket(),
    }
    pending_draw: dict[int, tuple] = {}

    def resolve_draws(recorder, game_state):
        for seat, (draws, melds, useful, p) in list(pending_draw.items()):
            now = recorder.draw_counts[seat]
            if now == draws:
                continue
            del pending_draw[seat]
            if now == draws + 1 and len(game_state.hands[seat].melds) == melds and recorder.last_drawn[seat] is not None:
                _add(st["cal_next_draw"], p, label(recorder.last_drawn[seat]) in useful)

    def hook(game, recorder, player, decision):
        gs = game.state
        resolve_draws(recorder, gs)
        state = build_state(gs, recorder, player)
        report = analyze(state)
        st["decisions"] += 1
        st["ms"].append(report.elapsed_ms)
        if report.errors:
            st["conservation_errors"] += 1
            return
        if gs.phase != "play":
            return

        counts = compute_counts(state)
        god = _god_hidden_counts(gs, recorder, player)
        st["count_checks"] += 1
        if list(counts.unseen) != god:
            st["count_mismatch"] += 1
        st["exhausted_claims"] += len(counts.exhausted)

        # 已胡者反推:召回 + 必然持有 + 持牌概率校准
        for inference in report.winners:
            st["winner_status"][inference.status] += 1
            if not inference.complete:
                continue
            truth = _true_hidden(gs, recorder, inference.seat)
            st["winner_checks"] += 1
            if not any(list(map(int, row)) == truth for row in inference.candidates):
                st["winner_recall_miss"] += 1
            st["certain_checks"] += 1
            if any(truth[t] < inference.certain[t] for t in range(27)):
                st["certain_violations"] += 1
            for t, p in enumerate(inference.hold_prob or ()):
                if 0.0 < p < 1.0:
                    _add(st["cal_winner"], p, truth[t] >= 1)

        # 安全度:100% 安全零误判;不可能形状零误判;形状持有概率校准
        for row in report.danger_rows:
            tile = parse_tile(from_label(row["tile"]))
            for entry in row["per_opponent"]:
                seat = entry["seat"]
                actual = gs.hands[seat]
                if entry["safe"]:
                    st["safe_claims"] += 1
                    st[{"void": "safe_void", "holds_void": "safe_holds_void"}.get(entry["reason"], "safe_blocked")] += 1
                    trial = Hand(counts=list(actual.counts), melds=list(actual.melds))
                    if trial.count(tile) < 4:
                        trial.add(tile)
                        if can_win(trial, gs.void_suits[seat]):
                            st["safe_violations"] += 1
                for shape in entry["shapes"]:
                    needed: dict[int, int] = {}
                    for name in shape["tiles"]:
                        t = tidx(from_label(name))
                        needed[t] = needed.get(t, 0) + 1
                    held = all(actual.counts[t] >= n for t, n in needed.items())
                    if not shape["possible"]:
                        st["impossible_shapes"] += 1
                        if held:
                            st["impossible_violations"] += 1
                    elif shape.get("known"):
                        st["known_shapes"] += 1
                        if not held:
                            st["known_shape_violations"] += 1
                    elif "hold_prob" in shape:
                        _add(st["cal_shape"], shape["hold_prob"], held)

        # 换三张:换给别家、还没出手的牌确实在他手里;并和"不用换牌信息"对照安全结论多了多少
        floors = swap_floors(state)
        for seat, floor in floors.items():
            truth = _true_hidden(gs, recorder, seat) if gs.won[seat] else list(gs.hands[seat].counts)
            st["floor_checks"] += 1
            if any(truth[t] < floor[t] for t in range(27)):
                st["floor_violations"] += 1
        if floors and report.danger_rows:
            base = analyze(replace(state, swap=None))
            st["swap_decisions"] += 1
            st["safe_with_swap"] += sum(e["safe"] for row in report.danger_rows for e in row["per_opponent"])
            st["safe_without_swap"] += sum(e["safe"] for row in base.danger_rows for e in row["per_opponent"])

        # 牌墙期望张数:与真实牌墙比较,并和"不扣已胡者"的朴素均摊对照
        wall_actual = [0] * 27
        for t in gs.wall:
            wall_actual[t.index] += 1
        if counts.total_unseen and gs.wall:
            for t, row in enumerate(report.remaining_rows):
                err = row["wall_expected"] - wall_actual[t]
                st["wall_abs_err"] += abs(err)
                st["wall_signed_err"] += err
                naive = counts.unseen[t] * len(gs.wall) / counts.total_unseen
                st["wall_naive_abs_err"] += abs(naive - wall_actual[t])
                st["wall_samples"] += 1

        # 候选弃牌:并列/冲突比例;记下下一摸进张的预测,等他下次摸牌时兑现
        info = report.own_info
        if info and info.get("mode") == "discard":
            st["discard_decisions"] += 1
            st["tie"] += bool(info.get("tie"))
            st["conflict"] += bool(info.get("conflict"))
            if decision.action.kind is ActionKind.DISCARD:
                chosen = label(tile_to_str(decision.action.tile))
                cand = next((c for c in info["candidates"] if c["tile"] == chosen), None)
                if cand is not None:
                    pending_draw[player] = (
                        recorder.draw_counts[player],
                        len(gs.hands[player].melds),
                        {u["tile"] for u in cand["useful"]},
                        cand["p_next"],
                    )

    game, recorder = run_game(seed, on_decision=hook)
    resolve_draws(recorder, game.state)
    st["rob_kong_games"] = int(any(record.robbing_kong for record in recorder.wins))
    st["winner_status"] = dict(st["winner_status"])
    return st


def merge(results: list[dict]) -> dict:
    total: dict = {}
    for res in results:
        for key, value in res.items():
            if key.startswith("cal_"):
                if key not in total:
                    total[key] = _bucket()
                _merge_bucket(total[key], value)
            elif key == "winner_status":
                bucket = total.setdefault(key, defaultdict(int))
                for k, v in value.items():
                    bucket[k] += v
            elif key == "ms":
                total.setdefault("ms", []).extend(value)
            else:
                total[key] = total.get(key, 0) + value
    total["winner_status"] = dict(total.get("winner_status", {}))
    return total


def _cal_table(bucket: dict) -> tuple[str, float, int]:
    lines = ["| 预测概率区间 | 样本数 | 平均预测 | 实际发生率 | 偏差 |", "|---|---:|---:|---:|---:|"]
    n_total = sum(bucket["n"])
    ece = 0.0
    for i in range(BINS):
        n = bucket["n"][i]
        if not n:
            continue
        pred = bucket["pred"][i] / n
        hit = bucket["hit"][i] / n
        ece += n / n_total * abs(pred - hit)
        lines.append(f"| {i / BINS:.1f}–{(i + 1) / BINS:.1f} | {n} | {pred:.1%} | {hit:.1%} | {hit - pred:+.1%} |")
    return "\n".join(lines), ece, n_total


def write_report(total: dict, path: Path, games: int, seed_start: int, elapsed: float, workers: int = 1) -> None:
    ms = sorted(total["ms"])
    pct = lambda q: ms[min(len(ms) - 1, int(len(ms) * q))] if ms else 0.0  # noqa: E731
    status = total["winner_status"]
    winners_total = sum(status.values())
    disc = max(1, total["discard_decisions"])
    wall_n = max(1, total["wall_samples"])
    cal_w, ece_w, n_w = _cal_table(total["cal_winner"])
    cal_s, ece_s, n_s = _cal_table(total["cal_shape"])
    cal_d, ece_d, n_d = _cal_table(total["cal_next_draw"])
    ok = lambda bad: "通过" if bad == 0 else "**未通过**"  # noqa: E731
    text = f"""# 辅助器 v1 大规模对账报告

- 对局:{games} 局(种子 {seed_start} 起),四家均为 S3 规则 AI;每个"有选择"的决策点为行动者生成报告
- 决策点:{total['decisions']} 个;耗时 {elapsed / 60:.1f} 分钟
- 其中出现抢杠胡的牌局 {total['rob_kong_games']} 局,全部决策点正常计入

## 一、确定档(必须零误判)

| 检查项 | 次数 | 误判 | 结论 |
|---|---:|---:|---|
| "100% 安全":把这张牌给该对手,他确实不能胡 | {total['safe_claims']}(缺门 {total['safe_void']},他还拿着缺门牌 {total['safe_holds_void']},形状全堵死 {total['safe_blocked']}) | {total['safe_violations']} | {ok(total['safe_violations'])} |
| "该等待形状不可能":对手手里确实没有对应搭子 | {total['impossible_shapes']} | {total['impossible_violations']} | {ok(total['impossible_violations'])} |
| 已胡者"手里一定有":真实手牌确实包含 | {total['certain_checks']} | {total['certain_violations']} | {ok(total['certain_violations'])} |
| 换给别家、还没出手的牌:他手里确实至少有这些 | {total['floor_checks']} | {total['floor_violations']} | {ok(total['floor_violations'])} |
| "他手里一定有这个搭子"(由换过去的牌确定) | {total['known_shapes']} | {total['known_shape_violations']} | {ok(total['known_shape_violations'])} |
| 守恒校验报警(应为 0) | {total['decisions']} | {total['conservation_errors']} | {ok(total['conservation_errors'])} |

绝张结论共 {total['exhausted_claims']} 条,由下面的计数一致性保证。

换三张信息的增益:在 {total['swap_decisions']} 个"换出去的牌还有没出手的"决策点上,"100% 安全"结论从 {total['safe_without_swap']} 条增加到 {total['safe_with_swap']} 条(+{total['safe_with_swap'] - total['safe_without_swap']},{(total['safe_with_swap'] - total['safe_without_swap']) / max(1, total['safe_without_swap']):.1%})。

## 二、计数档(必须 100%)

| 检查项 | 次数 | 不一致 | 结论 |
|---|---:|---:|---|
| 每种牌的未现张数 = 上帝视角下真实落在牌墙和别人暗手牌里的张数 | {total['count_checks']} | {total['count_mismatch']} | {ok(total['count_mismatch'])} |
| 已胡者真实暗手牌落在反推出的可能组合里(召回) | {total['winner_checks']} | {total['winner_recall_miss']} | {ok(total['winner_recall_miss'])} |

已胡者反推的状态分布(按决策点 × 胡牌者计):{', '.join(f'{k} {v}({v / max(1, winners_total):.0%})' for k, v in status.items())}。
`too_many` 表示可能组合超过 2 万种暂不展开,多出现在中盘、胡牌者没有副露时;余牌减少后会自动展开。

## 三、概率档校准(不设门槛,记录偏差)

引擎里的对手是 S3 规则 AI,留牌习惯和真人不同,以下校准结果只能作为参考。

### 3.1 已胡者持牌概率(样本 {n_w},ECE {ece_w:.3f})

{cal_w}

### 3.2 对手"手里有能等这张的搭子"的概率(样本 {n_s},ECE {ece_s:.3f})

{cal_s}

### 3.3 下一摸摸到有效进张的概率(样本 {n_d},ECE {ece_d:.3f})

{cal_d}

### 3.4 牌墙期望张数

- 平均绝对误差 {total['wall_abs_err'] / wall_n:.3f} 张/种,平均偏差 {total['wall_signed_err'] / wall_n:+.3f}(正数表示高估)
- 对照"不扣除已胡者持牌"的朴素均摊:平均绝对误差 {total['wall_naive_abs_err'] / wall_n:.3f}

## 四、需要权衡的局面

- 出牌决策点 {total['discard_decisions']} 个
- 并列(按速度标准第一名有多张):{total['tie']}({total['tie'] / disc:.1%})
- 标准冲突(求快和求番大推荐不同的牌):{total['conflict']}({total['conflict'] / disc:.1%})

## 五、耗时

单个决策点完整报告:均值 {statistics.mean(ms) if ms else 0:.1f} ms,中位 {pct(0.5):.1f} ms,95 分位 {pct(0.95):.1f} ms,99 分位 {pct(0.99):.1f} ms,最大 {pct(1.0):.1f} ms(目标 ≤200 ms)。

测量条件:{workers} 个进程并行跑对账,进程之间争抢 CPU,数字比单独使用时偏高。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="辅助器大规模对账")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed-start", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("docs/assistant_v1_audit.md"))
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    started = time.time()
    results = []
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(audit_game, seeds, chunksize=4), 1):
            results.append(res)
            if i % 25 == 0 or i == len(seeds):
                print(f"[{i}/{len(seeds)}] {time.time() - started:.0f}s", flush=True)
    total = merge(results)
    write_report(total, args.out, args.games, args.seed_start, time.time() - started, args.workers)
    raw = {k: v for k, v in total.items() if k != "ms"}
    args.out.with_suffix(".json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
