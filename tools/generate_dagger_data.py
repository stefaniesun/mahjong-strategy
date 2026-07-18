"""DAgger 数据生成器:学生策略打牌,老师(S3 规则)在学生访问的每个决策点给出标签。

- 座位:每局 1 个学生 + 3 个 S3,学生座位 = seed % 4(四方位轮换)。
- 轨迹由学生的真实选择推进(这是 DAgger 的关键);记录的 action 是老师在同一局面的选择。
- 只记录学生座位的决策点(其余座位的局面分布已被原始 S3 自对局数据覆盖)。
- 输出:与 S3 自对局数据完全同构的 .jsonl.gz 分片 + manifest(schema s2.v4),
  训练侧无需任何格式适配,与原数据混合重训即可。
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.game import Game
from engine.settlement import assert_zero_sum
from learning.datasets.dataset_builder import _clean_action, _mask_from_actions
from policies.decision_boundary import choose_policy_action
from policies.learned_policy import LearnedPolicy
from policies.rule_policy import RulePolicy
from selfplay.data_recorder import _action_to_dict
from state.action_space import action_to_index
from state.tile_belief import generate_belief_labels


_STUDENT: LearnedPolicy | None = None


def _worker_init(policy_ckpt: str, belief_ckpt: str) -> None:
    torch.set_num_threads(1)
    global _STUDENT
    _STUDENT = LearnedPolicy(policy_ckpt, belief_model_path=belief_ckpt)


def _select_actor(game: Game):
    state = game.state
    if state.phase in ("swap_three", "declare_void"):
        for player in range(4):
            if state.phase == "swap_three" and state.swap_choices[player] is None:
                return player
            if state.phase == "declare_void" and state.void_suits[player] is None:
                return player
        return None
    if state.pending_rob_kong is not None:
        return state.pending_rob_kong.winners[0]
    if state.pending_discard is not None:
        if state.pending_winners:
            return state.pending_winners[0]
        for offset in range(1, 4):
            player = (state.pending_discard.discarder + offset) % 4
            if game.legal_actions(player):
                return player
        return state.pending_discard.discarder
    if state.phase == "play":
        return state.current_player
    return None


def _play_dagger_game(seed: int, max_steps: int = 1000) -> tuple[list[dict[str, Any]], bool, bool]:
    assert _STUDENT is not None
    game = Game(seed=seed)
    state = game.reset()
    student_seat = seed % 4
    teacher = RulePolicy()
    rules = [RulePolicy() for _ in range(4)]
    pending: list[dict[str, Any]] = []
    steps = 0
    while not state.finished and steps < max_steps:
        steps += 1
        actor = _select_actor(game)
        if actor is None or not game.legal_actions(actor):
            break

        if actor == student_seat:
            student_decision = choose_policy_action(state, actor, _STUDENT)
            teacher_decision = choose_policy_action(state, actor, teacher)
            teacher_action_dict = _action_to_dict(teacher_decision.action)
            protocol_actions = [
                {k: v for k, v in item.items() if k not in {"conditionally_legal", "depends_on"}}
                for item in student_decision.protocol_state.legal_actions.value or []
            ]
            # 快速失败:与训练侧完全同路径的校验——老师标签必须落在合法掩码内
            mask = _mask_from_actions(protocol_actions)
            if not mask[action_to_index(_clean_action(teacher_action_dict))]:
                raise ValueError(f"teacher action outside legal mask at seed={seed} step={steps}")
            pending.append(
                {
                    "game_id": f"dagger1-{seed}",
                    "step": steps,
                    "player": actor,
                    "phase": state.phase,
                    "state": student_decision.protocol_state.to_dict(),
                    "legal_actions": protocol_actions,
                    "action": teacher_action_dict,
                    "labels": generate_belief_labels(state, student_decision.protocol_state),
                }
            )
            game.step(actor, student_decision.action)
        else:
            decision = choose_policy_action(state, actor, rules[actor])
            game.step(actor, decision.action)

    assert_zero_sum(state.scores)
    scores = list(state.scores)
    for record in pending:
        record["final_scores"] = scores
    finished = state.finished
    drawn = finished and not state.win_order
    return pending, finished, drawn


def _generate_shard(args: tuple[int, int, str]) -> dict[str, Any]:
    seed_start, seed_end, out_file = args
    records: list[dict[str, Any]] = []
    finished_games = 0
    drawn_games = 0
    started = time.perf_counter()
    for seed in range(seed_start, seed_end):
        game_records, finished, drawn = _play_dagger_game(seed)
        records.extend(game_records)
        finished_games += int(finished)
        drawn_games += int(drawn)
    payload = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    data = gzip.compress(payload.encode("utf-8"), 6)
    Path(out_file).write_bytes(data)
    return {
        "data_file": Path(out_file).name,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DAgger data: student trajectories, teacher labels.")
    parser.add_argument("--games", type=int, default=20000)
    parser.add_argument("--seed-start", type=int, default=302607170000)
    parser.add_argument("--shard-games", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("data_dagger1"))
    parser.add_argument("--policy-checkpoint", type=str, required=True)
    parser.add_argument("--belief-checkpoint", type=str, required=True)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 2)
    shards_dir = args.out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for index, start in enumerate(range(args.seed_start, args.seed_start + args.games, args.shard_games)):
        end = min(start + args.shard_games, args.seed_start + args.games)
        tasks.append((start, end, str(shards_dir / f"part_{start}_{end}.jsonl.gz")))

    started = time.perf_counter()
    with mp.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(args.policy_checkpoint, args.belief_checkpoint),
    ) as pool:
        shard_infos = []
        for done, info in enumerate(pool.imap(_generate_shard, tasks, chunksize=1), 1):
            shard_infos.append(info)
            print(
                f"[shard {done}/{len(tasks)}] games={info['games']} records={info['decision_records']} "
                f"{info['elapsed_seconds']:.0f}s",
                flush=True,
            )

    manifest = {
        "format_version": 1,
        "schema": "s2.v4",
        "kind": "dagger",
        "dagger_iteration": 1,
        "student_policy": str(args.policy_checkpoint),
        "games": sum(s["games"] for s in shard_infos),
        "finished_games": sum(s["finished_games"] for s in shard_infos),
        "drawn_games": sum(s["drawn_games"] for s in shard_infos),
        "decision_records": sum(s["decision_records"] for s in shard_infos),
        "compressed_bytes": sum(s["bytes"] for s in shard_infos),
        "seed_range": {"start": args.seed_start, "end_exclusive": args.seed_start + args.games},
        "dataset_fingerprint": hashlib.sha256("".join(s["sha256"] for s in shard_infos).encode()).hexdigest(),
        "shards": shard_infos,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    total = time.perf_counter() - started
    print(
        f"DONE: {manifest['games']} games, {manifest['decision_records']} records, "
        f"{manifest['compressed_bytes'] / 1e6:.0f}MB compressed, {total / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
