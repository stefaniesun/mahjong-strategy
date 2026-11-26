"""多核、可断点续跑的换三张后“定缺阶段”大样本配对评测脚本。

支持两种模式:
- `--mode moment` (推荐主力大样本): 只模拟到“四家全部定缺落地那一刻”, 完全隔离后期打牌AI策略噪声,
  以毫秒级速度统计真实物理供牌(牌墙+对手必打喂牌)、竞争人数、弃牌暴露度与单行道形成率;
- `--mode full` (双验证): 在记录“定缺后那一刻”全部物理指标的同时, 对三种定缺(缺万/缺条/缺筒)
  继续使用 RulePlusPolicy('all') 配对全打一整局, 统计最终得分、胡率、大赢率(>=+6)与重赔率(<=-6)。

重点采集四类换后纠结局面(跳过闭眼缺1/2张的明显局):
1. 4/4/5_crash    : 闲家换后 4/4/5 且换牌撞车 (换出花色 == 换入花色)
2. 4/4/5_noncrash : 闲家换后 4/4/5 且非撞车 (换出花色 != 换入花色)
3. 4/5/5_dealer   : 庄家换后 4/5/5 或 4/4/6 均衡牌型
4. 3_close        : 换后 3/4/6 或 3/5/5, 且缺某长门(>=4张)向听不输于缺3张短门

用法示例:
    # 5 秒环境冒烟自检
    python tools/run_void_eval.py --mode full --deals-per-bucket 10 --workers 4 --out outputs/void_check.json

    # 高算力主机主力大样本: 纯定缺瞬间物理模型 (每桶 1000 局, 约 3~8 分钟)
    python tools/run_void_eval.py --mode moment --deals-per-bucket 1000 --resume --out outputs/void_moment_large.json

    # 高算力主机完整配对全打大样本 (每桶 500 局 x 3配对, 约 25~45 分钟)
    python tools/run_void_eval.py --mode full --deals-per-bucket 500 --resume --out outputs/void_full_large.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.swap_v1 import fast_live_useful_two_suits, fast_shanten_two_suits, split_suits, suit_profile
from assistant.swap_v2 import choose_void_v2_idx, sample_swap_v2_indices
from engine.actions import Action, ActionKind
from engine.game import Game
from engine.tiles import SUITS, Tile, tile_to_str
from policies.decision_boundary import choose_policy_action
from policies.heuristics import choose_void_suit
from policies.rule_policy_plus import PlusConfig, RulePlusPolicy

_PLAY_POLICY = RulePlusPolicy(PlusConfig.parse("all"))


def _suit_bucket(counts27) -> str:
    s0 = sum(counts27[0:9])
    s1 = sum(counts27[9:18])
    s2 = sum(counts27[18:27])
    a, b, c = sorted((s0, s1, s2))
    return f"{a}/{b}/{c}"


def _classify_deal(counts27, is_dealer: bool, is_crash: bool) -> str | None:
    b = _suit_bucket(counts27)
    if not is_dealer and b == "4/4/5":
        return "4/4/5_crash" if is_crash else "4/4/5_noncrash"
    if is_dealer and b in ("4/5/5", "4/4/6"):
        return "4/5/5_dealer"
    if b in ("3/4/6", "3/5/5"):
        suits3 = split_suits(counts27)
        totals = [sum(s) for s in suits3]
        shs = [fast_shanten_two_suits(suits3[(i + 1) % 3], suits3[(i + 2) % 3]) for i in range(3)]
        min_idx = min(range(3), key=lambda i: totals[i])
        if any(shs[i] <= shs[min_idx] for i in range(3) if i != min_idx):
            return "3_close"
    return None


def _mean_ci(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def _play_from_void(base_state, void_decisions: tuple[int, int, int, int]) -> dict:
    game = Game(seed=0, dealer=base_state.dealer)
    state = copy.deepcopy(base_state)
    game.state = state

    for p in range(4):
        game.step(p, Action(ActionKind.DECLARE_VOID, suit=SUITS[void_decisions[p]]))

    steps = 0
    while not state.finished and steps < 1000:
        if state.pending_rob_kong is not None:
            pl = state.pending_rob_kong.winners[0]
            dec = choose_policy_action(state, pl, _PLAY_POLICY)
            game.step(pl, dec.action)
            steps += 1
            continue
        if state.pending_discard is not None:
            if state.pending_winners:
                pl = state.pending_winners[0]
                dec = choose_policy_action(state, pl, _PLAY_POLICY)
                game.step(pl, dec.action)
                steps += 1
                continue
            discarder = state.pending_discard.discarder
            for offset in range(1, 4):
                pl = (discarder + offset) % 4
                if state.won[pl] or pl in state.pending_passers:
                    continue
                dec = choose_policy_action(state, pl, _PLAY_POLICY)
                game.step(pl, dec.action)
                steps += 1
                if dec.action.kind is not ActionKind.PASS:
                    break
            continue
        if state.phase == "play":
            pl = state.current_player
            dec = choose_policy_action(state, pl, _PLAY_POLICY)
            game.step(pl, dec.action)
            steps += 1
            continue
        break

    return {
        "score0": float(state.scores[0]),
        "scores": [float(x) for x in state.scores],
        "won0": bool(state.won[0]),
        "kongs0": sum(1 for m in state.hands[0].melds if len(m.tiles) == 4),
        "opp_max": float(max(state.scores[1:])),
    }


def _worker_scan_chunk(args):
    start_seed, count, need_buckets, mode = args
    rng = random.Random(start_seed * 97 + 13)
    records = []

    for seed in range(start_seed, start_seed + count):
        dealer = seed % 4
        game = Game(seed=seed, dealer=dealer)
        state = game.reset()

        for p in range(4):
            idxs = sample_swap_v2_indices(
                state.hands[p].counts,
                rng=rng,
                is_dealer=(p == dealer),
                swap_direction=state.swap_direction,
                exact_live=False,
            )
            tiles = tuple(sorted(Tile.from_index(i) for i in idxs))
            game.step(p, Action(ActionKind.SWAP_THREE, tiles=tiles))

        p0_out_tiles = state.swap_choices[0]
        p0_out_suit = SUITS.index(p0_out_tiles[0].suit)
        giver_seat = (0 + state.swap_direction) % 4
        recv_seat = (0 - state.swap_direction) % 4
        p0_in_tiles = state.swap_choices[giver_seat]
        p0_in_suit = SUITS.index(p0_in_tiles[0].suit)
        is_crash = p0_out_suit == p0_in_suit

        bucket = _classify_deal(state.hands[0].counts, is_dealer=(dealer == 0), is_crash=is_crash)
        if bucket is None or bucket not in need_buckets:
            continue

        opp_v = {
            p: choose_void_v2_idx(split_suits(state.hands[p].counts), is_dealer=(p == dealer))
            for p in (1, 2, 3)
        }
        ov_list = [opp_v[1], opp_v[2], opp_v[3]]
        wall_s = [0, 0, 0]
        for t in state.wall:
            wall_s[SUITS.index(t.suit)] += 1
        opp_hand_s = {
            p: [sum(state.hands[p].counts[s * 9 : (s + 1) * 9]) for s in range(3)]
            for p in (1, 2, 3)
        }

        p0_s3 = split_suits(state.hands[0].counts)
        p0_cnts = [sum(p0_s3[s]) for s in range(3)]
        other_pairs = ((1, 2), (0, 2), (0, 1))
        v2_pick = choose_void_v2_idx(p0_s3, is_dealer=(dealer == 0))
        s3_pick = SUITS.index(choose_void_suit(state.hands[0]))

        opts = {}
        for v0 in (0, 1, 2):
            kept = other_pairs[v0]
            ca, cb, cv = p0_s3[kept[0]], p0_s3[kept[1]], p0_s3[v0]
            pa, pb, pv = suit_profile(ca), suit_profile(cb), suit_profile(cv)
            sh = fast_shanten_two_suits(ca, cb)
            live_pub = fast_live_useful_two_suits(ca, cb, sh)

            free_feed = sum(opp_hand_s[p][s] for p in (1, 2, 3) for s in kept if opp_v[p] == s)
            wall_avail = sum(wall_s[s] for s in kept)
            opp_locked = sum(opp_hand_s[p][s] for p in (1, 2, 3) for s in kept if opp_v[p] != s)
            opp_same_void = ov_list.count(v0)
            buyers_on_my_void = 3 - opp_same_void
            sole_opp_keeper = opp_same_void == 2
            sole_opp_feed = (
                p0_cnts[v0] + sum(opp_hand_s[p][v0] for p in (1, 2, 3) if opp_v[p] == v0)
                if sole_opp_keeper
                else 0
            )
            p0_sole_keeper = any(ov_list.count(s) == 3 for s in kept)

            entry = {
                "v0": v0,
                "suit": SUITS[v0].value,
                "void_cnt": pv.total,
                "void_pairs": pv.pairs,
                "void_conn": pv.conn,
                "shanten": sh,
                "live": live_pub,
                "kept_pairs": pa.pairs + pb.pairs,
                "kept_trips": pa.triplets + pb.triplets,
                "max_main": max(pa.total, pb.total),
                "min_main": min(pa.total, pb.total),
                "free_feed": free_feed,
                "wall_avail": wall_avail,
                "usable_supply": wall_avail + free_feed,
                "opp_locked": opp_locked,
                "opp_same_void": opp_same_void,
                "buyers_on_my_void": buyers_on_my_void,
                "dump_risk_units": pv.total * buyers_on_my_void,
                "sole_opp_keeper": sole_opp_keeper,
                "sole_opp_feed": sole_opp_feed,
                "p0_sole_keeper": p0_sole_keeper,
                "is_out_suit": v0 == p0_out_suit,
                "is_in_suit": v0 == p0_in_suit,
            }
            if mode == "full":
                entry["rollout"] = _play_from_void(state, (v0, opp_v[1], opp_v[2], opp_v[3]))
            opts[str(v0)] = entry

        records.append(
            {
                "seed": seed,
                "bucket": bucket,
                "is_dealer": dealer == 0,
                "swap_dir": state.swap_direction,
                "is_crash": is_crash,
                "p0_out_suit": p0_out_suit,
                "p0_in_suit": p0_in_suit,
                "p0_out_tiles": [tile_to_str(t) for t in p0_out_tiles],
                "p0_in_tiles": [tile_to_str(t) for t in p0_in_tiles],
                "giver_voids_in": opp_v[giver_seat] == p0_in_suit,
                "recv_voids_out": opp_v[recv_seat] == p0_out_suit,
                "v2_pick": v2_pick,
                "s3_pick": s3_pick,
                "opp_voids": ov_list,
                "opts": opts,
            }
        )
    return records


def _print_pair_row(title: str, pairs: list[tuple[dict, dict, bool]], show_rollout: bool) -> None:
    if not pairs:
        return
    n = len(pairs)
    da = lambda k: sum(a[k] - b[k] for a, b, _ in pairs) / n
    line = (
        f"  [{title}] n={n:4d} | 包袱差={da('void_cnt'):+.2f} | "
        f"可用总供牌差={da('usable_supply'):+.2f}(喂牌{da('free_feed'):+.2f}) | "
        f"弃牌危险差={da('dump_risk_units'):+.2f}"
    )
    if show_rollout and "rollout" in pairs[0][0]:
        diffs = [a["rollout"]["score0"] - b["rollout"]["score0"] for a, b, _ in pairs]
        m, ci = _mean_ci(diffs)
        line += f" | 实盘分差={m:+.3f}(±{ci:.3f})"
    print(line, flush=True)


def summarize_records(records: list[dict], mode: str) -> None:
    has_rollout = bool(records and "rollout" in records[0]["opts"]["0"])
    print(f"\n=== 总样本数: {len(records)} 局 (mode={mode}) ===", flush=True)

    non_cr = [r for r in records if not r["is_crash"]]
    cr = [r for r in records if r["is_crash"]]
    print("\n--- 1. 换牌流向信息在'定缺后那一刻'的对局占比 ---", flush=True)
    if non_cr:
        print(
            f"  非撞车 (n={len(non_cr):4d}): 给牌方缺换入花色={sum(r['giver_voids_in'] for r in non_cr)/len(non_cr):.1%}, "
            f"收牌方缺换出花色={sum(r['recv_voids_out'] for r in non_cr)/len(non_cr):.1%}"
        )
    if cr:
        for lbl, cond in [("对家互换(dir=2)", lambda r: r["swap_dir"] == 2), ("顺逆传(dir=±1)", lambda r: r["swap_dir"] != 2)]:
            sub = [r for r in cr if cond(r)]
            if sub:
                print(
                    f"  撞车-{lbl} (n={len(sub):4d}): 给牌方缺撞车门={sum(r['giver_voids_in'] for r in sub)/len(sub):.1%}, "
                    f"收牌方缺撞车门={sum(r['recv_voids_out'] for r in sub)/len(sub):.1%}"
                )

    print("\n--- 2. 同向听(sh_a == sh_b)条件下：'定缺后那一刻'物理供需与单行道统计 ---", flush=True)
    p_cnt, p_out_in_nc, p_in_3, p_out_3, p_c2, p_c0 = [], [], [], [], [], []
    p_out_in_dir2, p_out_in_dir1 = [], []
    for r in records:
        opts = [r["opts"][str(i)] for i in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                oi, oj = opts[i], opts[j]
                if oi["shanten"] != oj["shanten"]:
                    continue
                if oi["void_cnt"] < oj["void_cnt"]:
                    p_cnt.append((oi, oj, r["is_crash"]))
                elif oj["void_cnt"] < oi["void_cnt"]:
                    p_cnt.append((oj, oi, r["is_crash"]))
                if oi["void_cnt"] == oj["void_cnt"]:
                    if not r["is_crash"]:
                        if oi["is_out_suit"] and oj["is_in_suit"]:
                            p_out_in_nc.append((oi, oj, False))
                            (p_out_in_dir2 if r["swap_dir"] == 2 else p_out_in_dir1).append((oi, oj, False))
                        elif oj["is_out_suit"] and oi["is_in_suit"]:
                            p_out_in_nc.append((oj, oi, False))
                            (p_out_in_dir2 if r["swap_dir"] == 2 else p_out_in_dir1).append((oj, oi, False))
                    if oi["is_in_suit"] and not oj["is_in_suit"] and not oj["is_out_suit"]:
                        p_in_3.append((oi, oj, r["is_crash"]))
                    elif oj["is_in_suit"] and not oi["is_in_suit"] and not oi["is_out_suit"]:
                        p_in_3.append((oj, oi, r["is_crash"]))
                    if oi["is_out_suit"] and not oj["is_in_suit"] and not oj["is_out_suit"]:
                        p_out_3.append((oi, oj, r["is_crash"]))
                    elif oj["is_out_suit"] and not oi["is_in_suit"] and not oi["is_out_suit"]:
                        p_out_3.append((oj, oi, r["is_crash"]))
                if oi["opp_same_void"] == 2 and oj["opp_same_void"] in (0, 1):
                    p_c2.append((oi, oj, r["is_crash"]))
                elif oj["opp_same_void"] == 2 and oi["opp_same_void"] in (0, 1):
                    p_c2.append((oj, oi, r["is_crash"]))
                if oi["opp_same_void"] == 0 and oj["opp_same_void"] == 1:
                    p_c0.append((oi, oj, r["is_crash"]))
                elif oj["opp_same_void"] == 0 and oi["opp_same_void"] == 1:
                    p_c0.append((oj, oi, r["is_crash"]))

    _print_pair_row("缺自己张数少 - 缺张数多", p_cnt, has_rollout)
    _print_pair_row("非撞车同张: 缺换出门 - 缺换进门(全部)", p_out_in_nc, has_rollout)
    _print_pair_row("非撞车同张(对家dir=2): 缺换出门 - 缺换进门", p_out_in_dir2, has_rollout)
    _print_pair_row("非撞车同张(顺逆dir=±1): 缺换出门 - 缺换进门", p_out_in_dir1, has_rollout)
    _print_pair_row("同张: 缺换进门 - 缺第三门", p_in_3, has_rollout)
    _print_pair_row("同张: 缺换出门 - 缺第三门", p_out_3, has_rollout)
    _print_pair_row("缺'2对手已缺'(放单行道) - 缺'0/1对手已缺'", p_c2, has_rollout)
    _print_pair_row("缺'0对手已缺'(独缺) - 缺'1对手已缺'", p_c0, has_rollout)

    # 4/4/5_crash breakdown
    cr445 = [r for r in records if r["bucket"] == "4/4/5_crash"]
    if cr445:
        print(f"\n--- 3. 4/4/5 撞车专项 (n={len(cr445)})：4张撞车门并列最优向听时 缺撞车门 vs 缺另一4张门 ---", flush=True)
        for dlbl, dcond in [("对家互换(dir=2)", lambda r: r["swap_dir"] == 2), ("顺逆传(dir=±1)", lambda r: r["swap_dir"] != 2)]:
            pairs = []
            for r in cr445:
                if not dcond(r):
                    continue
                ocr = r["opts"][str(r["p0_out_suit"])]
                if ocr["void_cnt"] != 4:
                    continue
                other4 = [r["opts"][str(v)] for v in (0, 1, 2) if v != r["p0_out_suit"] and r["opts"][str(v)]["void_cnt"] == 4]
                if other4 and ocr["shanten"] == other4[0]["shanten"]:
                    pairs.append((ocr, other4[0], True))
            if pairs:
                sole_cr = sum(1 for a, _, _ in pairs if a["sole_opp_keeper"]) / len(pairs)
                sole_ot = sum(1 for _, b, _ in pairs if b["sole_opp_keeper"]) / len(pairs)
                print(f"  {dlbl} (n={len(pairs)}): 放对面单行道 {sole_cr:.1%} vs {sole_ot:.1%}", flush=True)
                _print_pair_row(f"  {dlbl} 缺撞车4张 - 缺另4张", pairs, has_rollout)

    if has_rollout:
        print("\n--- 4. 配对全打得分汇总 (含 95% 置信半宽 ±CI) ---", flush=True)
        by_b = defaultdict(list)
        for r in records:
            by_b[r["bucket"]].append(r)
        for b in sorted(by_b):
            rs = by_b[b]
            v2_vals = [r["opts"][str(r["v2_pick"])]["rollout"]["score0"] for r in rs]
            s3_vals = [r["opts"][str(r["s3_pick"])]["rollout"]["score0"] for r in rs]
            oracle_vals = [max(r["opts"][str(v)]["rollout"]["score0"] for v in (0, 1, 2)) for r in rs]
            m2, ci2 = _mean_ci(v2_vals)
            ms3, _ = _mean_ci(s3_vals)
            mo, _ = _mean_ci(oracle_vals)
            bl = sum(1 for x in v2_vals if x <= -6) / len(rs)
            win = sum(1 for r in rs if r["opts"][str(r["v2_pick"])]["rollout"]["won0"]) / len(rs)
            print(
                f"  [{b:<14}] n={len(rs):4d} | void_v2={m2:+.3f}(±{ci2:.3f}, 胡={win:.1%}, 重赔<=-6:{bl:.1%}) "
                f"| s3={ms3:+.3f} | 理论上限={mo:+.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="换三张后定缺大样本多核评测脚本")
    parser.add_argument("--mode", choices=("moment", "full"), default="moment", help="moment=仅定缺瞬间物理指标; full=含全盘配对打完")
    parser.add_argument("--deals-per-bucket", type=int, default=500, help="每个默认桶采样的局数")
    parser.add_argument("--noncrash-quota", type=int, default=None, help="4/4/5_noncrash 独立配额(默认等于 deals-per-bucket)")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 1))
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--chunk-seeds", type=int, default=250)
    parser.add_argument("--resume", action="store_true", help="从已有 checkpoint 断点续跑")
    parser.add_argument("--out", type=Path, default=Path("outputs/void_eval_large.json"))
    args = parser.parse_args()

    target_buckets = ("4/4/5_crash", "4/4/5_noncrash", "4/5/5_dealer", "3_close")
    quotas = {b: args.deals_per_bucket for b in target_buckets}
    if args.noncrash_quota is not None:
        quotas["4/4/5_noncrash"] = args.noncrash_quota

    ckpt_path = args.out.with_suffix(args.out.suffix + ".ckpt.json")
    collected_by_bucket: dict[str, list[dict]] = {b: [] for b in target_buckets}
    next_seed = args.seed_start

    if args.resume and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        for b in target_buckets:
            collected_by_bucket[b] = ckpt.get("buckets", {}).get(b, [])[: quotas[b]]
        next_seed = ckpt.get("next_seed", args.seed_start)
        print("已从断点恢复: " + ", ".join(f"{b}={len(collected_by_bucket[b])}/{quotas[b]}" for b in target_buckets), flush=True)

    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        while any(len(collected_by_bucket[b]) < quotas[b] for b in target_buckets):
            need = {b for b in target_buckets if len(collected_by_bucket[b]) < quotas[b]}
            tasks = [
                (next_seed + w * args.chunk_seeds, args.chunk_seeds, need, args.mode)
                for w in range(args.workers)
            ]
            next_seed += args.workers * args.chunk_seeds
            for chunk_recs in ex.map(_worker_scan_chunk, tasks):
                for rec in chunk_recs:
                    b = rec["bucket"]
                    if len(collected_by_bucket[b]) < quotas[b]:
                        collected_by_bucket[b].append(rec)

            ckpt_payload = {"next_seed": next_seed, "buckets": collected_by_bucket}
            ckpt_path.write_text(json.dumps(ckpt_payload, ensure_ascii=False), encoding="utf-8")
            counts_str = ", ".join(f"{b}={len(collected_by_bucket[b])}/{quotas[b]}" for b in target_buckets)
            print(f"  [进度 {time.time()-t0:.1f}s | seed->{next_seed}] {counts_str}", flush=True)
            if next_seed - args.seed_start > 500_000:
                break

    all_records = [r for b in target_buckets for r in collected_by_bucket[b]]
    payload = {
        "mode": args.mode,
        "elapsed_sec": round(time.time() - t0, 2),
        "workers": args.workers,
        "quotas": quotas,
        "records": all_records,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(all_records)} deals to {args.out} ({payload['elapsed_sec']}s)")
    summarize_records(all_records, args.mode)


if __name__ == "__main__":
    main()
