"""高速多核、可断点续跑的换三张四家盘面大样本模拟与盲区体检工具。

功能:
1. 四家陪练全部采用“正常人策略 v1”(choose_swap_v1 + choose_void_v1),消除呆板旧策略带来的进牌失真;
2. 按指定纠结牌型桶(如 4/4/5, 3/4/6, 3/3/7, 2/5/6, 1/5/7 等)随机发牌采样;
3. 同手牌同洗牌配对对比: legacy vs principled(现行S3) vs swap_v1;
4. 同时对每手牌的所有合法换法做四家盘面真值打分,统计 swap_v1 与盘面最优解的重合度,并保存分歧牌例供 v2 迭代;
5. 统计正常人 v1 实际换出的三张牌型分布与撞花色概率。

快速本地验证示例:
    python tools/run_swap_v1_eval.py --hands-per-bucket 100 --deals-per-hand 40 --workers 4 --out outputs/swap_v1_eval_local.json

高算力 CPU 机器大样本运行示例:
    python tools/run_swap_v1_eval.py --hands-per-bucket 3000 --deals-per-hand 80 --workers 16 --out outputs/swap_v1_eval_large.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from functools import lru_cache

from assistant.swap_v1 import (
    choose_swap_v1_indices,
    choose_void_v1_idx,
    choose_void_v1_with_sh,
    fast_live_useful_two_suits,
    fast_shanten_two_suits,
    gift_shape_info,
    legal_suit_combos,
    score_swap_candidates_v1,
    split_suits,
    suit_profile,
)
from engine.hand import Hand
from engine.tiles import Tile, tile_to_str
from policies.heuristics import choose_swap_tiles, choose_swap_tiles_legacy

DEFAULT_NON_DEALER_BUCKETS = ("4/4/5", "3/4/6", "3/5/5", "3/3/7", "2/5/6", "2/4/7", "1/5/7", "1/6/6")
DEFAULT_DEALER_BUCKETS = ("4/5/5", "4/4/6", "3/5/6", "3/4/7", "2/5/7", "2/6/6")

_FULL_DECK_IDX: tuple[int, ...] = tuple(i for i in range(27) for _ in range(4))
_DICE_DIRECTIONS: tuple[int, ...] = (
    # 按两颗骰子 36 种组合真实概率: 对家 24/36, 下家 6/36, 上家 6/36
    *([-1] * 6),  # 4,8,12 -> 给下家(seat 1)
    *([1] * 6),   # 2,6,10 -> 给上家(seat 3)
    *([2] * 24),  # 3,5,7,9,11 -> 给对家(seat 2)
)


def _bucket_of(c27: list[int]) -> str:
    s0 = sum(c27[0:9])
    s1 = sum(c27[9:18])
    s2 = sum(c27[18:27])
    a, b, c = sorted((s0, s1, s2))
    return f"{a}/{b}/{c}"


@lru_cache(maxsize=1 << 17)
def _cached_live(ca: tuple[int, ...], cb: tuple[int, ...], sh: int) -> int:
    return fast_live_useful_two_suits(ca, cb, sh)


def _board_eval_player(
    suits3: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    v_idx: int,
    sh: int,
    is_dealer: bool,
) -> dict[str, float]:
    other = ((1, 2), (0, 2), (0, 1))[v_idx]
    ca, cb = suits3[other[0]], suits3[other[1]]
    pa, pb, pv = suit_profile(ca), suit_profile(cb), suit_profile(suits3[v_idx])
    vcnt = pv.total
    live = _cached_live(ca, cb, sh)
    max_main = pa.total if pa.total >= pb.total else pb.total
    v_eff = max(0.0, vcnt - 0.6) if is_dealer else float(vcnt)
    jam = 1.0 if vcnt >= (4 if is_dealer else 3) else 0.0
    jam_pen = 0.32 * max(0, vcnt - (3 if is_dealer else 2))
    flush = 1.0 if max_main >= 9 and vcnt <= 2 else 0.0
    strong_flush = 1.0 if max_main >= 10 else 0.0
    roots = pa.quads + pb.quads
    triplets = pa.triplets + pb.triplets
    flush_bonus = -0.48 if strong_flush else (-0.22 if flush else 0.0)
    root_bonus = -0.25 * roots - 0.08 * triplets
    self_cost = 1.0 * sh + 0.36 * v_eff + jam_pen - 0.012 * live + flush_bonus + root_bonus
    return {
        "self_cost": self_cost,
        "shanten": float(sh),
        "void_cnt": float(vcnt),
        "jam": jam,
        "live": float(live),
        "flush": flush,
        "strong_flush": strong_flush,
        "roots": float(roots),
    }


def _legacy_or_principled_indices(c27: list[int], mode: str) -> tuple[int, int, int]:
    hand = Hand(counts=list(c27))
    tiles = choose_swap_tiles_legacy(hand) if mode == "legacy" else choose_swap_tiles(hand)
    i0, i1, i2 = sorted(t.index for t in tiles)
    return (i0, i1, i2)


def _evaluate_single_hand(task: tuple[int, str, bool, int]) -> dict:
    hand_seed, target_bucket, target_is_dealer, deals_per_hand = task
    rng = random.Random(hand_seed)
    hand_size = 14 if target_is_dealer else 13
    dealer = 0 if target_is_dealer else rng.choice((1, 2, 3))
    swap_dir = rng.choice(_DICE_DIRECTIONS)
    receiver = (0 - swap_dir) % 4
    giver_to_me = (0 + swap_dir) % 4
    recv_is_dealer = receiver == dealer

    # 抽样直到命中目标结构桶 (对常见桶只需几次,稀有桶也仅毫秒级)
    deck = list(_FULL_DECK_IDX)
    my_c27 = [0] * 27
    while True:
        rng.shuffle(deck)
        my_c27 = [0] * 27
        for idx in deck[:hand_size]:
            my_c27[idx] += 1
        if _bucket_of(my_c27) == target_bucket:
            break

    rem_pool = list(deck[hand_size:])
    my_suits3 = split_suits(my_c27)

    # 枚举我方所有合法换三张候选
    all_candidates: list[tuple[int, int, int]] = []
    for s_idx in (0, 1, 2):
        for r0, r1, r2 in legal_suit_combos(my_suits3[s_idx]):
            all_candidates.append((s_idx * 9 + r0, s_idx * 9 + r1, s_idx * 9 + r2))

    leg_act = _legacy_or_principled_indices(my_c27, "legacy")
    pri_act = _legacy_or_principled_indices(my_c27, "principled")
    v1_ranked = score_swap_candidates_v1(my_c27, is_dealer=target_is_dealer, receiver_is_dealer=recv_is_dealer)
    v1_act = v1_ranked[0]["tile_indices"]

    # 每个候选累计四家盘面指标
    cand_acc = {
        act: {
            "board_cost": 0.0,
            "self_cost": 0.0,
            "shanten": 0.0,
            "void_cnt": 0.0,
            "jam": 0.0,
            "live": 0.0,
            "flush": 0.0,
            "recv_helped": 0.0,
            "recv_wasted": 0.0,
            "recv_sh_delta": 0.0,
            "my_lane": 0.0,
            "opp_lane": 0.0,
        }
        for act in all_candidates
    }

    opp_swap_shapes: Counter[str] = Counter()
    collision_by_my_cnt: list[tuple[int, int]] = []

    for _ in range(deals_per_hand):
        rng.shuffle(rem_pool)
        opp_c27: dict[int, list[int]] = {}
        pos = 0
        for seat in (1, 2, 3):
            sz = 14 if seat == dealer else 13
            arr = [0] * 27
            for t_i in rem_pool[pos : pos + sz]:
                arr[t_i] += 1
            opp_c27[seat] = arr
            pos += sz

        # 三家陪练全部用正常人策略 v1 选出换三张
        opp_out: dict[int, tuple[int, int, int]] = {}
        for seat in (1, 2, 3):
            r_seat = (seat - swap_dir) % 4
            act = choose_swap_v1_indices(
                opp_c27[seat],
                is_dealer=(seat == dealer),
                receiver_is_dealer=(r_seat == dealer),
            )
            opp_out[seat] = act
            s_name, _ = gift_shape_info(act[0] % 9, act[1] % 9, act[2] % 9)
            opp_swap_shapes[s_name] += 1

        in_tiles = opp_out[giver_to_me]
        in_suit = in_tiles[0] // 9
        my_cnt_in_suit = sum(my_suits3[in_suit])
        collision_by_my_cnt.append((my_cnt_in_suit, in_suit))

        # 接牌家换牌前的参考向听
        recv_pre_s3 = split_suits(opp_c27[receiver])
        recv_pre_v, recv_pre_sh = choose_void_v1_with_sh(recv_pre_s3, is_dealer=recv_is_dealer)
        recv_pre_info = _board_eval_player(recv_pre_s3, recv_pre_v, recv_pre_sh, is_dealer=recv_is_dealer)

        # 预先算好除我与接牌家之外另外两家换完后的定缺(他们不受我换哪3张影响!)
        fixed_voids: dict[int, int] = {}
        for seat in (1, 2, 3):
            if seat == receiver:
                continue
            arr = list(opp_c27[seat])
            for ti in opp_out[seat]:
                arr[ti] -= 1
            src = (seat + swap_dir) % 4
            src_tiles = opp_out[src]  # src != 0 因为 seat != receiver
            for ti in src_tiles:
                arr[ti] += 1
            fixed_voids[seat] = choose_void_v1_idx(split_suits(arr), is_dealer=(seat == dealer))

        # 接牌家扣掉自己换出的3张后的基底
        recv_base = list(opp_c27[receiver])
        for ti in opp_out[receiver]:
            recv_base[ti] -= 1

        for act in all_candidates:
            # 我方换后手牌
            my_post = list(my_c27)
            for ti in act:
                my_post[ti] -= 1
            for ti in in_tiles:
                my_post[ti] += 1
            my_post_s3 = split_suits(my_post)
            my_v, my_sh = choose_void_v1_with_sh(my_post_s3, is_dealer=target_is_dealer)
            my_info = _board_eval_player(my_post_s3, my_v, my_sh, is_dealer=target_is_dealer)

            # 接牌家换后手牌
            recv_post = list(recv_base)
            for ti in act:
                recv_post[ti] += 1
            recv_post_s3 = split_suits(recv_post)
            recv_v, recv_sh = choose_void_v1_with_sh(recv_post_s3, is_dealer=recv_is_dealer)
            recv_info = _board_eval_player(recv_post_s3, recv_v, recv_sh, is_dealer=recv_is_dealer)

            # 四家定缺单行道检查
            vc = [0, 0, 0]
            vc[my_v] += 1
            vc[recv_v] += 1
            for fv in fixed_voids.values():
                vc[fv] += 1
            my_lane = 0.0
            opp_lane = 0.0
            for s_i in (0, 1, 2):
                if vc[s_i] == 3:
                    if my_v != s_i:
                        my_lane = 1.0
                    else:
                        opp_lane = 1.0

            recv_sh_delta = recv_info["shanten"] - recv_pre_info["shanten"]
            recv_helped = 1.0 if recv_sh_delta < 0 else 0.0
            recv_wasted = 1.0 if recv_v == (act[0] // 9) else 0.0

            # 四家盘面综合真实成本 (越小越好):
            # 自身盘面成本 - 0.25*(接牌家盘面变差量,即我喂肥他则加罚) + 单行道奖惩
            recv_gain = recv_pre_info["self_cost"] - recv_info["self_cost"]
            recv_weight = 0.32 if recv_is_dealer else 0.22
            lane_term = -0.25 * my_lane + 0.25 * opp_lane
            board_cost = my_info["self_cost"] + recv_weight * recv_gain + lane_term

            st = cand_acc[act]
            st["board_cost"] += board_cost
            st["self_cost"] += my_info["self_cost"]
            st["shanten"] += my_info["shanten"]
            st["void_cnt"] += my_info["void_cnt"]
            st["jam"] += my_info["jam"]
            st["live"] += my_info["live"]
            st["flush"] += my_info["flush"]
            st["recv_helped"] += recv_helped
            st["recv_wasted"] += recv_wasted
            st["recv_sh_delta"] += recv_sh_delta
            st["my_lane"] += my_lane
            st["opp_lane"] += opp_lane

    inv_n = 1.0 / deals_per_hand
    cand_means = {
        act: {k: v * inv_n for k, v in st.items()}
        for act, st in cand_acc.items()
    }
    oracle_act = min(all_candidates, key=lambda a: (cand_means[a]["board_cost"], a))
    oracle_cost = cand_means[oracle_act]["board_cost"]
    v1_regret = cand_means[v1_act]["board_cost"] - oracle_cost

    disagreement = None
    if v1_regret > 0.08:
        disagreement = {
            "hand_seed": hand_seed,
            "bucket": target_bucket,
            "is_dealer": target_is_dealer,
            "swap_dir": swap_dir,
            "recv_is_dealer": recv_is_dealer,
            "hand": [tile_to_str(Tile.from_index(i)) for i, c in enumerate(my_c27) for _ in range(c)],
            "v1_swap": [tile_to_str(Tile.from_index(i)) for i in v1_act],
            "oracle_swap": [tile_to_str(Tile.from_index(i)) for i in oracle_act],
            "v1_regret": round(v1_regret, 4),
            "v1_metrics": {k: round(v, 3) for k, v in cand_means[v1_act].items()},
            "oracle_metrics": {k: round(v, 3) for k, v in cand_means[oracle_act].items()},
        }

    return {
        "bucket": target_bucket,
        "is_dealer": target_is_dealer,
        "legacy": cand_means[leg_act],
        "principled": cand_means[pri_act],
        "v1": cand_means[v1_act],
        "oracle": cand_means[oracle_act],
        "v1_exact_match": int(v1_act == oracle_act),
        "v1_near_optimal": int(v1_regret <= 0.05),
        "v1_regret": v1_regret,
        "v1_same_as_principled": int(v1_act == pri_act),
        "opp_swap_shapes": dict(opp_swap_shapes),
        "incoming_suit_by_my_cnt": Counter(cnt for cnt, _ in collision_by_my_cnt),
        "disagreement": disagreement,
    }


def _mean_ci(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))
    return m, 1.96 * sd / math.sqrt(len(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description="运行换三张 v1 四家盘面多核大样本评估")
    parser.add_argument("--hands-per-bucket", type=int, default=100, help="每个结构桶随机抽取的不同初始手牌数")
    parser.add_argument("--deals-per-hand", type=int, default=40, help="每手牌重洗外面三家发牌的次数")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 1))
    parser.add_argument("--seed-start", type=int, default=20261125)
    parser.add_argument("--include-dealer", action="store_true", help="同时覆盖庄家14张核心纠结桶")
    parser.add_argument("--out", type=Path, default=Path("outputs/swap_v1_board_eval.json"))
    parser.add_argument("--resume", action="store_true", help="从已有 checkpoint 断点续跑")
    args = parser.parse_args()

    buckets: list[tuple[str, bool]] = [(b, False) for b in DEFAULT_NON_DEALER_BUCKETS]
    if args.include_dealer:
        buckets.extend((b, True) for b in DEFAULT_DEALER_BUCKETS)

    ckpt_path = args.out.with_suffix(args.out.suffix + ".ckpt.json")
    completed_buckets: dict[str, dict] = {}
    if args.resume and ckpt_path.exists():
        completed_buckets = json.loads(ckpt_path.read_text(encoding="utf-8"))
        print(f"已从断点恢复 {len(completed_buckets)} 个分桶结果")

    started = time.perf_counter()
    global_shapes: Counter[str] = Counter()
    all_disagreements: list[dict] = []

    with mp.Pool(processes=args.workers) as pool:
        for b_idx, (bucket, is_dealer) in enumerate(buckets):
            b_key = f"{'dealer' if is_dealer else 'non_dealer'}|{bucket}"
            if b_key in completed_buckets:
                continue
            t_b = time.perf_counter()
            base_seed = args.seed_start + b_idx * 1_000_000
            tasks = [
                (base_seed + i, bucket, is_dealer, args.deals_per_hand)
                for i in range(args.hands_per_bucket)
            ]
            rows = list(pool.imap_unordered(_evaluate_single_hand, tasks, chunksize=8))

            bucket_shapes: Counter[str] = Counter()
            bucket_in_by_cnt: Counter[int] = Counter()
            bucket_disagreements: list[dict] = []
            for r in rows:
                bucket_shapes.update(r["opp_swap_shapes"])
                bucket_in_by_cnt.update(r["incoming_suit_by_my_cnt"])
                if r["disagreement"] is not None:
                    bucket_disagreements.append(r["disagreement"])
            bucket_disagreements.sort(key=lambda d: -d["v1_regret"])

            n = float(len(rows))
            diff_vs_pri = [r["principled"]["board_cost"] - r["v1"]["board_cost"] for r in rows]
            diff_vs_leg = [r["legacy"]["board_cost"] - r["v1"]["board_cost"] for r in rows]
            gain_pri_m, gain_pri_ci = _mean_ci(diff_vs_pri)
            gain_leg_m, gain_leg_ci = _mean_ci(diff_vs_leg)

            def avg_metric(pol: str, key: str) -> float:
                return round(sum(r[pol][key] for r in rows) / n, 4)

            summary_b = {
                "hands": len(rows),
                "deals_per_hand": args.deals_per_hand,
                "total_board_sims": len(rows) * args.deals_per_hand,
                "elapsed_sec": round(time.perf_counter() - t_b, 2),
                "v1_same_as_principled_rate": round(sum(r["v1_same_as_principled"] for r in rows) / n, 4),
                "v1_exact_oracle_match_rate": round(sum(r["v1_exact_match"] for r in rows) / n, 4),
                "v1_near_optimal_rate": round(sum(r["v1_near_optimal"] for r in rows) / n, 4),
                "v1_avg_regret": round(sum(r["v1_regret"] for r in rows) / n, 4),
                "board_cost_gain_vs_principled": {"mean": round(gain_pri_m, 4), "ci95_half": round(gain_pri_ci, 4)},
                "board_cost_gain_vs_legacy": {"mean": round(gain_leg_m, 4), "ci95_half": round(gain_leg_ci, 4)},
                "metrics": {
                    pol: {
                        "board_cost": avg_metric(pol, "board_cost"),
                        "shanten": avg_metric(pol, "shanten"),
                        "void_cnt": avg_metric(pol, "void_cnt"),
                        "jam_rate": avg_metric(pol, "jam"),
                        "flush_rate": avg_metric(pol, "flush"),
                        "recv_helped_rate": avg_metric(pol, "recv_helped"),
                        "opp_lane_rate": avg_metric(pol, "opp_lane"),
                    }
                    for pol in ("legacy", "principled", "v1", "oracle")
                },
                "opp_swap_shapes": dict(bucket_shapes),
                "incoming_hits_by_my_suit_count": {str(k): v for k, v in sorted(bucket_in_by_cnt.items())},
                "top_disagreements": bucket_disagreements[:25],
            }
            completed_buckets[b_key] = summary_b
            args.out.parent.mkdir(parents=True, exist_ok=True)
            ckpt_path.write_text(json.dumps(completed_buckets, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{b_key:<18}] {len(rows)}手x{args.deals_per_hand}局 ({summary_b['elapsed_sec']:.1f}s) | "
                f"v1胜现行S3: {gain_pri_m:+.3f} (±{gain_pri_ci:.3f}) | "
                f"近似最优率: {summary_b['v1_near_optimal_rate']:.1%} (精确第一:{summary_b['v1_exact_oracle_match_rate']:.1%}) | "
                f"向听 {summary_b['metrics']['principled']['shanten']:.2f}->{summary_b['metrics']['v1']['shanten']:.2f} "
                f"卡壳 {summary_b['metrics']['principled']['jam_rate']:.1%}->{summary_b['metrics']['v1']['jam_rate']:.1%} "
                f"喂敌 {summary_b['metrics']['principled']['recv_helped_rate']:.1%}->{summary_b['metrics']['v1']['recv_helped_rate']:.1%}",
                flush=True,
            )

    for b_data in completed_buckets.values():
        global_shapes.update(b_data.get("opp_swap_shapes", {}))
        all_disagreements.extend(b_data.get("top_disagreements", []))
    total_shapes = sum(global_shapes.values()) or 1
    shape_dist = {k: round(v / total_shapes, 4) for k, v in global_shapes.most_common()}
    all_disagreements.sort(key=lambda d: -d["v1_regret"])

    payload = {
        "total_elapsed_sec": round(time.perf_counter() - started, 2),
        "hands_per_bucket": args.hands_per_bucket,
        "deals_per_hand": args.deals_per_hand,
        "workers": args.workers,
        "v1_opponent_swap_shape_distribution": shape_dist,
        "by_bucket": completed_buckets,
        "top_v1_disagreements": all_disagreements[:80],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全部完成,总耗时 {payload['total_elapsed_sec']}s,输出已保存至 {args.out}")


if __name__ == "__main__":
    main()
