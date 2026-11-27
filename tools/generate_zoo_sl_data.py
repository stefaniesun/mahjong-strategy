"""多核、可断点续跑的多代版本库高质量监督学习(SL)数据生成器 (s2.v4 兼容格式)。

核心设计:
1. 每局随机分配 2 个「顶级老师座位」+ 2 个「多样对手座位」(四方位均衡轮换):
   - 老师池默认: v4_waitq_tactics_det (大牌/好口型, 60%) + v3_goodwait_read_det (快攻型, 40%)
   - 对手池默认: v4_waitq_tactics_rand, v3_goodwait_read_rand, v2_ev_base_det, s3_plus, s3_rule
   - 只记录「老师座位」的决策作为训练样本(保证策略标签纯度为 78~80 分顶级水平, 绝不学弱策略的坏习惯);
   - 对手混编多代多风格策略, 保证老师访问到的牌面分布、对手牌河与攻防格局高度多样化;
   - 同步写入真实上帝视角 Belief 标签 (generate_belief_labels), 让新版 BeliefNet 专门适配高手牌风。
2. 100% 兼容 S4 训练管线 (`tools/cloud_train_s4_50k_cached.py`):
   - 输出完全同构的 `shards/part_{start}_{end}.jsonl.gz` + `manifest.json` (`schema: "s2.v4"`);
   - 每条记录均通过合法掩码校验与牌局零和校验, 杜绝脏样本;
   - 支持分片级 `--resume` 断点续跑, 并且每 5 局自动清空 lru_cache 杜绝多核 OOM。

用法示例:
    # 10 秒冒烟自检 (8 局)
    python tools/generate_zoo_sl_data.py --games 8 --shard-games 4 --workers 2 --out-dir outputs/s4_v6_smoke

    # 高算力多核主机正式采数 (30,000 局 -> 约 150 万条顶级老师决策样本)
    python tools/generate_zoo_sl_data.py --games 30000 --shard-games 500 --workers 10 --resume --out-dir outputs/s4_v6_zoo_data
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.recorder import PublicRecorder
from assistant.zoo import clear_all_zoo_caches, load_version, parse_zoo_spec
from engine.game import Game
from engine.settlement import assert_zero_sum
from learning.datasets.dataset_builder import _clean_action, _mask_from_actions
from play.session import next_actor
from policies.decision_boundary import choose_policy_action
from policies.rule_policy import RulePolicy
from policies.rule_policy_plus import PlusConfig, RulePlusPolicy
from selfplay.data_recorder import _action_to_dict
from state.action_space import action_to_index, legal_mask
from state.adapters.from_engine import from_engine
from state.tile_belief import generate_belief_labels

DEFAULT_TEACHERS = (
    "v4_waitq_tactics_det",
    "v4_waitq_tactics_det",
    "v4_waitq_tactics_det",
    "v3_goodwait_read_det",
    "v3_goodwait_read_det",
)  # 60% v4 + 40% v3
DEFAULT_OPPONENTS = (
    "v4_waitq_tactics_rand",
    "v3_goodwait_read_rand",
    "v2_ev_base_det",
    "s3_plus",
    "s3_rule",
)

_WORKER_POLICIES: dict[str, Any] = {}
_WORKER_TEACHERS: tuple[str, ...] = ()
_WORKER_OPPONENTS: tuple[str, ...] = ()
_WORKER_RECORD_ALL_TEACHERS: bool = True


def _build_policy(name: str) -> Any:
    if name == "s3_rule":
        return RulePolicy()
    if name == "s3_plus":
        return RulePlusPolicy(PlusConfig.parse("all"))
    parsed = parse_zoo_spec(name)
    if parsed is None:
        raise ValueError(f"未知策略名: {name}")
    ver, stoch = parsed
    return load_version(ver, stochastic=stoch, exact_swap_live=False)


def _worker_init(teachers: list[str], opponents: list[str]) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    global _WORKER_POLICIES, _WORKER_TEACHERS, _WORKER_OPPONENTS
    _WORKER_TEACHERS = tuple(teachers)
    _WORKER_OPPONENTS = tuple(opponents)
    _WORKER_POLICIES = {}
    for name in sorted(set(teachers) | set(opponents)):
        _WORKER_POLICIES[name] = _build_policy(name)


def _release_caches() -> None:
    clear_all_zoo_caches()
    for mod_name in ("assistant.winner_inference", "assistant.swap_v2", "assistant.swap_v1"):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for val in list(vars(mod).values()):
            clear = getattr(val, "cache_clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    pass


def _play_one_sl_game(seed: int, max_steps: int = 1000) -> tuple[list[dict[str, Any]], bool, bool]:
    rng = random.Random(seed * 131 + 17)
    dealer = seed % 4

    # 选 2 个老师座位 + 2 个多样对手座位(位置完全随机, 保证四方位庄闲均衡)
    seats = [0, 1, 2, 3]
    rng.shuffle(seats)
    teacher_seats = {seats[0], seats[1]}
    lineup = [""] * 4
    for s in range(4):
        if s in teacher_seats:
            lineup[s] = rng.choice(_WORKER_TEACHERS)
        else:
            lineup[s] = rng.choice(_WORKER_OPPONENTS)

    # 给阵容中带随机性的策略绑定独立可复现种子
    for idx, name in enumerate(sorted(set(lineup))):
        pol = _WORKER_POLICIES[name]
        if hasattr(pol, "rng"):
            pol.rng = random.Random(seed * 31 + idx * 17 + 5)

    game = Game(seed=seed, dealer=dealer)
    state = game.reset()
    rec = PublicRecorder(game)

    pending: list[dict[str, Any]] = []
    steps = 0
    while not state.finished and steps < max_steps:
        actor = next_actor(game)
        if actor is None:
            break
        legal = game.legal_actions(actor)
        if not legal:
            break
        steps += 1

        pol_name = lineup[actor]
        pol = _WORKER_POLICIES[pol_name]

        if actor in teacher_seats:
            protocol_state = from_engine(state, player_id=actor)
            protocol_actions = [
                {k: v for k, v in item.items() if k not in {"conditionally_legal", "depends_on"}}
                for item in (protocol_state.legal_actions.value or [])
            ]
            mask = _mask_from_actions(protocol_actions)
            if hasattr(pol, "choose_from_game"):
                act = pol.choose_from_game(state, rec, actor, legal)
            else:
                act = choose_policy_action(state, actor, pol).action

            act_dict = _action_to_dict(act)
            # 严格快速失败校验: 老师动作必须落在训练侧合法掩码内
            clean_act = _clean_action(act_dict)
            if not mask[action_to_index(clean_act)]:
                raise ValueError(
                    f"Teacher {pol_name} produced action outside protocol legal mask at seed={seed} step={steps}: {act_dict}"
                )
            pending.append(
                {
                    "game_id": f"zoo_sl-{seed}",
                    "step": steps,
                    "player": actor,
                    "phase": state.phase,
                    "state": protocol_state.to_dict(),
                    "legal_actions": protocol_actions,
                    "action": act_dict,
                    "labels": generate_belief_labels(state, protocol_state),
                }
            )
        else:
            if hasattr(pol, "choose_from_game"):
                act = pol.choose_from_game(state, rec, actor, legal)
            else:
                act = choose_policy_action(state, actor, pol).action

        rec.step(actor, act)

    assert_zero_sum(state.scores)
    scores = list(state.scores)
    for record in pending:
        record["final_scores"] = scores
    finished = bool(state.finished)
    drawn = finished and not state.win_order
    return pending, finished, drawn


def _generate_shard(args: tuple[int, int, str, str]) -> dict[str, Any]:
    seed_start, seed_end, out_file_str, meta_file_str = args
    out_file = Path(out_file_str)
    meta_file = Path(meta_file_str)
    records: list[dict[str, Any]] = []
    finished_games = 0
    drawn_games = 0
    started = time.perf_counter()

    for idx, seed in enumerate(range(seed_start, seed_end), 1):
        game_records, finished, drawn = _play_one_sl_game(seed)
        records.extend(game_records)
        finished_games += int(finished)
        drawn_games += int(drawn)
        if idx % 5 == 0:
            _release_caches()
    _release_caches()

    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    data = gzip.compress(payload.encode("utf-8"), 6)
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    tmp_file.write_bytes(data)
    tmp_file.replace(out_file)

    info = {
        "data_file": out_file.name,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "games": seed_end - seed_start,
        "finished_games": finished_games,
        "drawn_games": drawn_games,
        "decision_records": len(records),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "schema": "s2.v4",
        "format_version": 1,
    }
    meta_tmp = meta_file.with_suffix(meta_file.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    meta_tmp.replace(meta_file)
    return info


def generate_dataset(
    *,
    games: int,
    seed_start: int,
    shard_games: int,
    out_dir: Path,
    workers: int,
    teachers: list[str],
    opponents: list[str],
    resume: bool = True,
) -> dict[str, Any]:
    shards_dir = out_dir / "shards"
    meta_dir = out_dir / "shard_meta"
    shards_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    all_specs: list[tuple[int, int, Path, Path]] = []
    for start in range(seed_start, seed_start + games, shard_games):
        end = min(start + shard_games, seed_start + games)
        data_path = shards_dir / f"part_{start}_{end}.jsonl.gz"
        meta_path = meta_dir / f"part_{start}_{end}.meta.json"
        all_specs.append((start, end, data_path, meta_path))

    completed: dict[str, dict[str, Any]] = {}
    pending_tasks: list[tuple[int, int, str, str]] = []
    for start, end, data_path, meta_path in all_specs:
        if resume and data_path.is_file() and meta_path.is_file():
            try:
                info = json.loads(meta_path.read_text(encoding="utf-8"))
                if data_path.stat().st_size == int(info["bytes"]):
                    completed[data_path.name] = info
                    continue
            except Exception:
                pass
        pending_tasks.append((start, end, str(data_path), str(meta_path)))

    print(
        f"Zoo SL 采数启动 | 总计 {games} 局 ({len(all_specs)} 分片) | "
        f"已完成 {len(completed)} 分片 | 待跑 {len(pending_tasks)} 分片 | workers={workers}",
        flush=True,
    )
    print(f"  老师池(记录标签): {', '.join(teachers)}", flush=True)
    print(f"  对手池(丰富牌风): {', '.join(opponents)}", flush=True)

    started = time.perf_counter()
    if pending_tasks:
        if workers <= 1:
            _worker_init(teachers, opponents)
            for done_idx, task in enumerate(pending_tasks, 1):
                info = _generate_shard(task)
                completed[info["data_file"]] = info
                elapsed = time.perf_counter() - started
                eta = elapsed / done_idx * (len(pending_tasks) - done_idx)
                print(
                    f"  [shard {len(completed)}/{len(all_specs)} | {elapsed:6.1f}s | ETA ~{eta/60:4.1f}m] "
                    f"{info['data_file']} games={info['games']} records={info['decision_records']} ({info['bytes']/1e6:.1f}MB)",
                    flush=True,
                )
        else:
            with mp.Pool(
                processes=workers,
                initializer=_worker_init,
                initargs=(teachers, opponents),
            ) as pool:
                for done_idx, info in enumerate(pool.imap_unordered(_generate_shard, pending_tasks, chunksize=1), 1):
                    completed[info["data_file"]] = info
                    elapsed = time.perf_counter() - started
                    eta = elapsed / done_idx * (len(pending_tasks) - done_idx)
                    print(
                        f"  [shard {len(completed)}/{len(all_specs)} | {elapsed:6.1f}s | ETA ~{eta/60:4.1f}m] "
                        f"{info['data_file']} games={info['games']} records={info['decision_records']} ({info['bytes']/1e6:.1f}MB)",
                        flush=True,
                    )

    # 按 seed_start 严格排序, 保证 manifest 确定性
    shard_infos = [completed[data_path.name] for _, _, data_path, _ in all_specs]
    manifest = {
        "format_version": 1,
        "schema": "s2.v4",
        "kind": "zoo_teacher_sl",
        "teachers": list(teachers),
        "opponents": list(opponents),
        "games": sum(s["games"] for s in shard_infos),
        "finished_games": sum(s["finished_games"] for s in shard_infos),
        "drawn_games": sum(s["drawn_games"] for s in shard_infos),
        "decision_records": sum(s["decision_records"] for s in shard_infos),
        "compressed_bytes": sum(s["bytes"] for s in shard_infos),
        "seed_range": {"start": seed_start, "end_exclusive": seed_start + games},
        "dataset_fingerprint": hashlib.sha256("".join(s["sha256"] for s in shard_infos).encode()).hexdigest(),
        "shards": shard_infos,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    total = time.perf_counter() - started
    print(
        f"DONE: {manifest['games']} games, {manifest['decision_records']} records, "
        f"{manifest['compressed_bytes'] / 1e6:.1f}MB compressed, {total / 60:.1f} min -> {out_dir / 'manifest.json'}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="用多代版本库(v3+v4 老师池)采集 S4 v6 监督学习数据 (s2.v4 格式)")
    parser.add_argument("--games", type=int, default=30000, help="总对局数(每局采 2 个顶级老师座位, 约 50 条决策/局)")
    parser.add_argument("--seed-start", type=int, default=600_000, help="起始随机种子")
    parser.add_argument("--shard-games", type=int, default=500, help="每个 .jsonl.gz 分片包含的对局数")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="并发进程数")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/s4_v6_zoo_data"), help="输出目录")
    parser.add_argument(
        "--teachers",
        type=str,
        default=",".join(DEFAULT_TEACHERS),
        help="逗号分隔老师策略列表(允许重复表示加权比例)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default=",".join(DEFAULT_OPPONENTS),
        help="逗号分隔对手策略列表",
    )
    parser.add_argument("--resume", action="store_true", help="跳过已完成的有效分片, 断点续跑")
    args = parser.parse_args()

    teachers = [x.strip() for x in args.teachers.split(",") if x.strip()]
    opponents = [x.strip() for x in args.opponents.split(",") if x.strip()]
    generate_dataset(
        games=args.games,
        seed_start=args.seed_start,
        shard_games=args.shard_games,
        out_dir=args.out_dir,
        workers=args.workers,
        teachers=teachers,
        opponents=opponents,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
