"""多核、可断点续跑的多策略随机匹配麻将锦标赛评测脚本 (tools/run_arena_tournament.py)。

特点:
1. 支持同时让项目内全部主流策略随机组桌混战:
   - s3_rule        : S3 基础规则 AI (RulePolicy)
   - s3_plus        : S3+ 增强规则 AI (RulePlusPolicy('all'))
   - s4_v5          : S4 v5 监督学习神经网络模型 (若本地存在 checkpoint 则自动加入)
   - s5             : 可选, 通过 --s5-checkpoint 指定 S5 强化学习存档
   - assistant_det  : 辅助器全套 AI 确定性最优版 (swap_v2 + void_v3 + opponent_read + discard_v2 + tactics)
   - assistant_rand : 辅助器全套 AI 受控随机版 (用于实测受控随机性是否降棋力)
2. 采用“随机组桌 + 同牌四座轮转 (Duplicate Rotation)”方差缩减机制:
   - 每个 seed 随机抽取 4 个策略组成一桌(至少含 2 种不同策略);
   - 默认对该副牌做 4 次座位轮转([A,B,C,D] -> [D,A,B,C] -> [C,D,A,B] -> [B,C,D,A]),
     让每位参赛策略拿到完全相同的 4 手牌与庄闲位置, 大幅压缩发牌运气方差!
3. 自动统计: 总分榜(含 95% CI)、胡率、自摸率、场均点炮、胡牌均分、大赢率(>=+6)、重赔率(<=-6)、
   以及两两同桌净胜分矩阵 (Pairwise Head-to-Head)。

用法示例:
    # 10 秒本机冒烟自检
    python tools/run_arena_tournament.py --deals 8 --workers 2 --out outputs/arena_smoke.json

    # 高算力主机大样本正式赛 (例如 500 副牌 x 4 轮转 = 2,000 场对局, 8,000 人次)
    python tools/run_arena_tournament.py --deals 500 --workers 15 --resume --out outputs/arena_tournament_large.json

    # 如需排除 S4 神经网络只跑纯规则/辅助器极速大样本 (速度可快 3~5 倍)
    python tools/run_arena_tournament.py --deals 1000 --no-s4 --workers 15 --resume --out outputs/arena_fast_large.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.bot import AssistantPolicy
from assistant.recorder import PublicRecorder
from engine.actions import ActionKind
from engine.game import Game
from play.session import next_actor
from policies.decision_boundary import choose_policy_action
from policies.rule_policy import RulePolicy
from policies.rule_policy_plus import PlusConfig, RulePlusPolicy

DEFAULT_S4_POLICY = ROOT / "training_artifacts" / "S4" / "v5_20260718_encoder_v4" / "checkpoints" / "policy_s4.pt"

# 每个 worker 进程内的策略实例缓存(避免每局重复加载 PyTorch 模型)
_WORKER_POLICIES: dict[str, object] = {}


def _init_worker(policy_names: list[str], s4_path: str | None, s5_path: str | None) -> None:
    # 限制多进程下每个 worker 的 PyTorch 线程数为 1, 防止 CPU 核间争抢导致大幅减速
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if "s4_v5" in policy_names or "s5" in policy_names:
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass

    global _WORKER_POLICIES
    _WORKER_POLICIES = {}
    for name in policy_names:
        if name == "s3_rule":
            _WORKER_POLICIES[name] = RulePolicy()
        elif name == "s3_plus":
            _WORKER_POLICIES[name] = RulePlusPolicy(PlusConfig.parse("all"))
        elif name == "assistant_det":
            _WORKER_POLICIES[name] = AssistantPolicy(stochastic=False, exact_swap_live=False)
        elif name == "assistant_rand":
            _WORKER_POLICIES[name] = AssistantPolicy(stochastic=True, exact_swap_live=False)
        elif name == "s4_v5":
            from play.ai import load_ai

            pol, _ = load_ai("s4", Path(s4_path) if s4_path else DEFAULT_S4_POLICY)
            _WORKER_POLICIES[name] = pol
        elif name == "s5":
            from play.ai import load_ai

            assert s5_path is not None
            pol, _ = load_ai("s5", Path(s5_path))
            _WORKER_POLICIES[name] = pol
        else:
            from assistant.zoo import load_version, parse_zoo_spec

            parsed = parse_zoo_spec(name)
            if parsed is None:
                raise ValueError(f"Unknown policy name: {name}")
            ver, stoch = parsed
            _WORKER_POLICIES[name] = load_version(ver, stochastic=stoch, exact_swap_live=False)


def _play_one_game(seed: int, dealer: int, lineup: list[str], rand_seed: int) -> dict:
    """用给定 4 策略阵容 lineup[0..3] 打完一整局血战麻将。"""
    # 若阵容含受控随机策略(含历史版本快照), 为每个策略实例绑定独立可复现种子
    for idx, name in enumerate(sorted(set(lineup))):
        pol = _WORKER_POLICIES.get(name)
        if pol is not None and hasattr(pol, "rng"):
            pol.rng = random.Random(rand_seed + idx * 17)

    game = Game(seed=seed, dealer=dealer)
    game.reset()
    rec = PublicRecorder(game)

    steps = 0
    while not game.state.finished and steps < 1000:
        actor = next_actor(game)
        if actor is None:
            break
        pol = _WORKER_POLICIES[lineup[actor]]
        if hasattr(pol, "choose_from_game"):
            legal = game.legal_actions(actor)
            act = pol.choose_from_game(game.state, rec, actor, legal)
        else:
            act = choose_policy_action(game.state, actor, pol).action
        rec.step(actor, act)
        steps += 1

    st = game.state
    dealt_in = [0, 0, 0, 0]
    self_draw = [0, 0, 0, 0]
    for w in rec.wins:
        if w.self_draw:
            self_draw[w.seat] = 1
        elif w.from_seat is not None:
            dealt_in[w.from_seat] += 1

    kongs = [sum(1 for m in st.hands[p].melds if len(m.tiles) == 4) for p in range(4)]
    finished_clean = bool(st.finished)
    # 判断流局无叫(未胡且流局)
    exhausted_wall = len(st.wall) == 0 and sum(st.won) < 3
    seats_out = []
    for p in range(4):
        seats_out.append(
            {
                "seat": p,
                "policy": lineup[p],
                "score": float(st.scores[p]),
                "won": bool(st.won[p]),
                "self_draw": bool(self_draw[p]),
                "dealt_in": int(dealt_in[p]),
                "kongs": int(kongs[p]),
                "blowup": bool(st.scores[p] <= -6),
                "big_win": bool(st.scores[p] >= 6),
                "unready_end": bool(exhausted_wall and not st.won[p] and st.scores[p] < 0),
            }
        )
    return {
        "seed": seed,
        "dealer": dealer,
        "lineup": list(lineup),
        "finished": finished_clean,
        "steps": steps,
        "seats": seats_out,
    }


_CACHE_MODULES = ("assistant.winner_inference", "assistant.swap_v2", "assistant.swap_v1")


def _release_caches() -> None:
    """清空策略模块内的 lru_cache。

    这些缓存以 numpy 数组为值, 多进程长跑时单个 worker 可累积到 GB 级内存,
    导致 "Unable to allocate ... MiB" 的 OOM。缓存只是加速, 清空不改变任何决策结果。
    """
    for name in _CACHE_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        for value in list(vars(mod).values()):
            clear = getattr(value, "cache_clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    pass
    try:
        from assistant.zoo import clear_all_zoo_caches

        clear_all_zoo_caches()
    except Exception:
        pass


def _worker_run_deal_batch(args):
    deal_specs, rotate = args
    games_out = []
    for deal_idx, seed, dealer, base_lineup in deal_specs:
        rotations = 4 if rotate else 1
        for rot in range(rotations):
            # 顺时针轮转座位, 使 4 位策略拿到同一副牌的 4 个位置
            lineup = base_lineup[-rot:] + base_lineup[:-rot] if rot else list(base_lineup)
            res = _play_one_game(seed, dealer, lineup, rand_seed=seed * 31 + rot * 7 + deal_idx)
            res["deal_idx"] = deal_idx
            res["rot"] = rot
            games_out.append(res)
        _release_caches()
    return games_out


def _mean_ci(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def summarize_tournament(games: list[dict], policies: list[str]) -> dict:
    # 1) 按单副牌 (deal_idx) 聚合同牌轮转后的策略均分(用于准确算配对 CI), 同时统计全部人次指标
    by_pol_seats: dict[str, list[dict]] = defaultdict(list)
    by_deal_pol_scores: dict[tuple[int, str], list[float]] = defaultdict(list)
    # Pairwise same-table diffs (按 deal_idx 聚合以消除轮转内部相关性)
    by_deal_pair: dict[tuple[int, str, str], list[float]] = defaultdict(list)

    bad_games = 0
    for g in games:
        if not g.get("finished", True):
            bad_games += 1
        didx = g.get("deal_idx", g["seed"])
        seats = g["seats"]
        for s in seats:
            by_pol_seats[s["policy"]].append(s)
            by_deal_pol_scores[(didx, s["policy"])].append(s["score"])
        for i in range(4):
            for j in range(4):
                if i == j:
                    continue
                pi, pj = seats[i]["policy"], seats[j]["policy"]
                if pi != pj:
                    by_deal_pair[(didx, pi, pj)].append(seats[i]["score"] - seats[j]["score"])

    leaderboard = []
    for p in policies:
        ss = by_pol_seats[p]
        n = len(ss)
        if n == 0:
            continue
        # 以每副牌(4轮转合并)为独立样本算 95% CI, 比把4轮转当独立样本更严谨
        deal_means = [sum(v) / len(v) for (d, pol), v in by_deal_pol_scores.items() if pol == p]
        m_sc, ci_sc = _mean_ci(deal_means)
        win_r = sum(1 for x in ss if x["won"]) / n
        sd_r = sum(1 for x in ss if x["self_draw"]) / n
        di_r = sum(x["dealt_in"] for x in ss) / n
        kong_r = sum(x["kongs"] for x in ss) / n
        big_r = sum(1 for x in ss if x["big_win"]) / n
        bl_r = sum(1 for x in ss if x["blowup"]) / n
        win_scores = [x["score"] for x in ss if x["won"]]
        avg_win_sc = sum(win_scores) / len(win_scores) if win_scores else 0.0
        leaderboard.append(
            {
                "policy": p,
                "seat_games": n,
                "deals": len(deal_means),
                "mean_score": round(m_sc, 4),
                "ci95": round(ci_sc, 4),
                "win_rate": round(win_r, 4),
                "self_draw_rate": round(sd_r, 4),
                "deal_in_per_game": round(di_r, 4),
                "kongs_per_game": round(kong_r, 4),
                "avg_win_score": round(avg_win_sc, 3),
                "big_win_rate": round(big_r, 4),
                "blowup_rate": round(bl_r, 4),
            }
        )
    leaderboard.sort(key=lambda r: -r["mean_score"])

    # Pairwise matrix
    pair_deals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (didx, pi, pj), vals in by_deal_pair.items():
        pair_deals[(pi, pj)].append(sum(vals) / len(vals))

    pairwise: dict[str, dict[str, dict]] = defaultdict(dict)
    for pi in policies:
        for pj in policies:
            if pi == pj:
                continue
            vals = pair_deals[(pi, pj)]
            if vals:
                dm, dci = _mean_ci(vals)
                pairwise[pi][pj] = {"deals": len(vals), "diff": round(dm, 4), "ci95": round(dci, 4), "sig": abs(dm) > dci}

    print(f"\n================ 多策略随机匹配锦标赛总榜 (共 {len(games)} 场对局, 异常局={bad_games}) ================")
    print(
        f"{'策略':<16} | {'场均净分 (±95%CI)':<19} | {'胡率':>6} | {'自摸率':>6} | {'场均点炮':>7} | "
        f"{'大赢>=6':>7} | {'重赔<=-6':>8} | {'参战人次':>7}"
    )
    print("-" * 102)
    for r in leaderboard:
        sc_txt = f"{r['mean_score']:+.3f} (±{r['ci95']:.3f})"
        print(
            f"{r['policy']:<16} | {sc_txt:<19} | {r['win_rate']:6.1%} | {r['self_draw_rate']:6.1%} | "
            f"{r['deal_in_per_game']:7.3f} | {r['big_win_rate']:7.1%} | {r['blowup_rate']:8.1%} | {r['seat_games']:7d}"
        )

    print("\n================ 两两同桌净胜分矩阵 (行策略 - 列策略, * 表示 95% 显著) ================")
    ordered_pols = [r["policy"] for r in leaderboard]
    header = f"{'行 - 列':<16}" + "".join(f" | {p:<15}" for p in ordered_pols)
    print(header)
    print("-" * len(header))
    for pi in ordered_pols:
        cells = []
        for pj in ordered_pols:
            if pi == pj:
                cells.append(f"{'—':<15}")
            elif pj in pairwise[pi]:
                d = pairwise[pi][pj]
                star = "*" if d["sig"] else " "
                cells.append(f"{star}{d['diff']:+.3f}(±{d['ci95']:.2f})".ljust(15))
            else:
                cells.append(f"{'N/A':<15}")
        print(f"{pi:<16}" + "".join(f" | {c}" for c in cells))

    return {"total_games": len(games), "bad_games": bad_games, "leaderboard": leaderboard, "pairwise": dict(pairwise)}


def build_deal_schedule(deals: int, seed_start: int, policies: list[str]) -> list[tuple[int, int, int, list[str]]]:
    """为每副牌生成可复现的随机 4 策略阵容(保证同桌至少有 2 种不同策略, 且各策略出场频率严格均衡)。"""
    rng = random.Random(seed_start * 17 + 2026)
    schedule = []
    for i in range(deals):
        seed = seed_start + i
        dealer = i % 4
        if len(policies) >= 4:
            # 优先不放回抽取 4 个不同策略(70% 概率), 或放回随机混搭(30% 概率, 覆盖同策略2v2/2v1v1 生态)
            if rng.random() < 0.70:
                lineup = rng.sample(policies, 4)
            else:
                while True:
                    lineup = [rng.choice(policies) for _ in range(4)]
                    if len(set(lineup)) >= 2:
                        break
        else:
            while True:
                lineup = [rng.choice(policies) for _ in range(4)]
                if len(set(lineup)) >= 2:
                    break
        schedule.append((i, seed, dealer, lineup))
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="多策略随机匹配锦标赛大样本评测")
    parser.add_argument("--deals", type=int, default=2500, help="独立牌副数(默认开启4座轮转 -> 总局数 = deals * 4)")
    parser.add_argument("--no-rotate", action="store_true", help="关闭4座同牌轮转(仅打1次)")
    parser.add_argument("--seed-start", type=int, default=120_000)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 1))
    parser.add_argument("--batch-deals", type=int, default=12, help="每次 checkpoint 批处理的牌副数")
    parser.add_argument("--no-s4", action="store_true", help="不加载 S4 v5 神经网络模型")
    parser.add_argument("--s4-checkpoint", type=Path, default=DEFAULT_S4_POLICY)
    parser.add_argument("--s5-checkpoint", type=Path, default=None, help="可选: 加入 S5 强化学习存档路径")
    parser.add_argument("--policies", type=str, default=None, help="手动指定参赛策略列表, 逗号分隔")
    parser.add_argument("--resume", action="store_true", help="从已有 checkpoint 断点续跑")
    parser.add_argument("--out", type=Path, default=Path("outputs/arena_tournament_large.json"))
    args = parser.parse_args()

    if args.policies:
        policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    else:
        policies = ["s3_rule", "s3_plus", "assistant_det", "assistant_rand"]
        if not args.no_s4 and args.s4_checkpoint.is_file():
            policies.append("s4_v5")
        if args.s5_checkpoint is not None:
            if not args.s5_checkpoint.is_file():
                raise FileNotFoundError(f"S5 checkpoint not found: {args.s5_checkpoint}")
            policies.append("s5")

    rotate = not args.no_rotate
    schedule = build_deal_schedule(args.deals, args.seed_start, policies)
    ckpt_path = args.out.with_suffix(args.out.suffix + ".ckpt.json")
    completed_deals: set[int] = set()
    all_games: list[dict] = []

    if args.resume and ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        if ckpt.get("policies") == policies and ckpt.get("rotate") == rotate:
            all_games = ckpt.get("games", [])
            completed_deals = {g["deal_idx"] for g in all_games}
            print(f"已从断点恢复: 完成 {len(completed_deals)}/{args.deals} 副牌 ({len(all_games)} 场对局)", flush=True)

    pending = [item for item in schedule if item[0] not in completed_deals]
    print(
        f"参赛策略({len(policies)}): {', '.join(policies)} | 计划 {args.deals} 副牌 x {'4轮转' if rotate else '1次'} "
        f"= {args.deals * (4 if rotate else 1)} 场 | 待跑 {len(pending)} 副 | workers={args.workers}",
        flush=True,
    )

    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    s4_arg = str(args.s4_checkpoint) if "s4_v5" in policies else None
    s5_arg = str(args.s5_checkpoint) if ("s5" in policies and args.s5_checkpoint) else None

    if pending:
        # 切分成小任务块, 每跑完一批 workers*chunk 即写 checkpoint
        chunk = max(1, args.batch_deals // max(1, args.workers))
        batches = [pending[i : i + chunk] for i in range(0, len(pending), chunk)]
        wave_size = max(1, args.workers * 2)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(policies, s4_arg, s5_arg),
        ) as ex:
            for start in range(0, len(batches), wave_size):
                wave = batches[start : start + wave_size]
                for games_chunk in ex.map(_worker_run_deal_batch, [(b, rotate) for b in wave]):
                    all_games.extend(games_chunk)
                    for g in games_chunk:
                        completed_deals.add(g["deal_idx"])
                ckpt_path.write_text(
                    json.dumps({"policies": policies, "rotate": rotate, "games": all_games}, ensure_ascii=False),
                    encoding="utf-8",
                )
                elapsed = time.time() - t0
                done_d = len(completed_deals)
                rate = done_d / max(0.1, elapsed)
                eta = (args.deals - done_d) / max(1e-6, rate)
                print(
                    f"  [进度 {elapsed:6.1f}s | ETA ~{eta:5.0f}s] {done_d}/{args.deals} 副牌 ({len(all_games)} 场对局)",
                    flush=True,
                )

    summary = summarize_tournament(all_games, policies)
    payload = {
        "policies": policies,
        "deals": args.deals,
        "rotate": rotate,
        "seed_start": args.seed_start,
        "elapsed_sec": round(time.time() - t0, 2),
        "workers": args.workers,
        "summary": summary,
        "games": all_games,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(all_games)} games to {args.out} ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
