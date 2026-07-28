from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning.eval.arena import ArenaConfig, run_arena
from policies.rule_policy import RulePolicy
from rl.checkpoints import load_checkpoint
from rl.opponent_resolver import ModelPolicy
from rl.train_rl import S5TrainingConfig, _default_model
from rl.rollout import FrozenBeliefProvider


@dataclass(frozen=True, slots=True)
class CandidateResult:
    checkpoint: str
    global_step: int
    games: int
    wins: int
    win_rate: float
    win_rate_confidence95: float
    average_score: float
    score_confidence95: float
    unfinished: int
    illegal_actions: int
    zero_sum_violations: int
    seat_games: list[int]
    seat_win_rates: list[float]
    seat_average_scores: list[float]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Uniformly evaluate S5 snapshots against the frozen S3 policy")
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--include-s4-baseline",
        action="store_true",
        help="Evaluate the unmodified frozen S4 policy with the same seeds and seats",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=202607230001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-game-steps", type=int, default=1000)
    parser.add_argument(
        "--belief-checkpoint",
        type=Path,
        default=Path("training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/belief_s4.pt"),
    )
    parser.add_argument(
        "--s4-policy-checkpoint",
        type=Path,
        default=Path("training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/policy_s4.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("training_artifacts/S5/local_console/evaluations/snapshot_selection.json"))
    parser.add_argument("--best-output", type=Path, default=Path("training_artifacts/S5/local_console/checkpoints/best_s5.pt"))
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.games <= 0:
        raise ValueError("games must be positive")
    if args.max_game_steps <= 0:
        raise ValueError("max_game_steps must be positive")
    if len(set(args.checkpoints)) != len(args.checkpoints):
        raise ValueError("checkpoints must be unique")
    for path in (*args.checkpoints, args.belief_checkpoint, args.s4_policy_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)


def _base_model(*, device: str, s4_policy_checkpoint: Path) -> Any:
    return _default_model(
        S5TrainingConfig(
            output_dir=s4_policy_checkpoint.parent,
            frozen_s4_belief_path=s4_policy_checkpoint,
            frozen_s4_policy_path=s4_policy_checkpoint,
            frozen_s4_provenance={"evaluation": True},
        )
    ).to(torch.device(device))


def _learner_policy(
    checkpoint: Path,
    *,
    device: str,
    belief_provider: Any,
    s4_policy_checkpoint: Path,
) -> tuple[ModelPolicy, int]:
    model = _base_model(device=device, s4_policy_checkpoint=s4_policy_checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    restored = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        restore_rng=False,
        map_location=device,
    )
    model.eval()
    return ModelPolicy(model, belief_provider=belief_provider, seed=0), restored.global_step


def _s4_policy(*, device: str, belief_provider: Any, s4_policy_checkpoint: Path) -> tuple[ModelPolicy, int]:
    model = _base_model(device=device, s4_policy_checkpoint=s4_policy_checkpoint)
    model.eval()
    return ModelPolicy(model, belief_provider=belief_provider, seed=0), 0


def _s3_policies() -> list[RulePolicy]:
    return [RulePolicy() for _ in range(3)]


def _confidence95(values: Sequence[float]) -> float:
    return 0.0 if len(values) <= 1 else 1.96 * stdev(values) / math.sqrt(len(values))


def _wilson95(successes: int, count: int) -> float:
    if count <= 0:
        return 0.0
    z = 1.96
    proportion = successes / count
    denominator = 1.0 + z * z / count
    return z * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)) / denominator


def evaluate_candidate(
    args: argparse.Namespace,
    checkpoint: Path,
    *,
    belief_provider: Any,
    s4_baseline: bool = False,
) -> CandidateResult:
    learner, global_step = (
        _s4_policy(
            device=args.device,
            belief_provider=belief_provider,
            s4_policy_checkpoint=args.s4_policy_checkpoint,
        )
        if s4_baseline
        else _learner_policy(
            checkpoint,
            device=args.device,
            belief_provider=belief_provider,
            s4_policy_checkpoint=args.s4_policy_checkpoint,
        )
    )
    opponents = _s3_policies()
    policies = [learner, *opponents]
    results = []
    learner_seats = []
    for game_index in range(args.games):
        learner_seat = game_index % 4
        seated = list(policies[1:])
        seated.insert(learner_seat, policies[0])
        result = run_arena(
            seated,
            ArenaConfig(games=1, seed=args.seed + game_index, max_steps=args.max_game_steps),
        ).results[0]
        results.append(result)
        learner_seats.append(learner_seat)

    learner_scores = [float(result.scores[seat]) for result, seat in zip(results, learner_seats, strict=True)]
    learner_wins = [seat in result.win_order for result, seat in zip(results, learner_seats, strict=True)]
    seat_games: list[int] = []
    seat_win_rates: list[float] = []
    seat_average_scores: list[float] = []
    for seat in range(4):
        indices = [index for index, learner_seat in enumerate(learner_seats) if learner_seat == seat]
        seat_games.append(len(indices))
        seat_win_rates.append(mean(float(learner_wins[index]) for index in indices) if indices else 0.0)
        seat_average_scores.append(mean(learner_scores[index] for index in indices) if indices else 0.0)

    wins = sum(learner_wins)
    return CandidateResult(
        checkpoint="s4-baseline" if s4_baseline else str(checkpoint.resolve()),
        global_step=global_step,
        games=args.games,
        wins=wins,
        win_rate=wins / args.games,
        win_rate_confidence95=_wilson95(wins, args.games),
        average_score=mean(learner_scores),
        score_confidence95=_confidence95(learner_scores),
        unfinished=sum(not result.finished for result in results),
        illegal_actions=sum(result.illegal_action for result in results),
        zero_sum_violations=sum(sum(result.scores) != 0 for result in results),
        seat_games=seat_games,
        seat_win_rates=seat_win_rates,
        seat_average_scores=seat_average_scores,
    )


def _selection_key(result: CandidateResult) -> tuple[float, float, float, int]:
    healthy = result.unfinished == 0 and result.illegal_actions == 0 and result.zero_sum_violations == 0
    return (float(healthy), result.average_score - result.score_confidence95, result.win_rate, result.global_step)


def main() -> int:
    args = _parser().parse_args()
    _validate_args(args)
    belief_provider = FrozenBeliefProvider.from_checkpoint(str(args.belief_checkpoint))
    results = [evaluate_candidate(args, checkpoint, belief_provider=belief_provider) for checkpoint in args.checkpoints]
    if args.include_s4_baseline:
        results.append(
            evaluate_candidate(
                args,
                args.s4_policy_checkpoint,
                belief_provider=belief_provider,
                s4_baseline=True,
            )
        )
    s5_results = [result for result in results if result.checkpoint != "s4-baseline"]
    best = max(s5_results, key=_selection_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "opponent": "s3",
        "games_per_candidate": args.games,
        "seed_start": args.seed,
        "seat_rotation": "round_robin",
        "selection_rule": "healthy candidate with highest lower 95% score bound, then win rate, then global step",
        "best_checkpoint": best.checkpoint,
        "best_global_step": best.global_step,
        "candidates": [asdict(result) for result in sorted(results, key=lambda item: item.global_step)],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.best_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best.checkpoint, args.best_output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"best_s5={args.best_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
