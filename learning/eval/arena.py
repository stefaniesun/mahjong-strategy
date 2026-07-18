from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import json
from math import sqrt
from pathlib import Path
import sys
from typing import Sequence


from engine.actions import ActionKind
from engine.game import Game
from engine.settlement import assert_zero_sum
from policies.base_policy import BasePolicy
from policies.decision_boundary import choose_policy_action
from policies.learned_policy import LearnedPolicy
from policies.rule_policy import RulePolicy



@dataclass(frozen=True)
class ArenaConfig:
    games: int = 100
    seed: int = 1
    max_steps: int = 1000


@dataclass(frozen=True)
class ArenaGameResult:
    seed: int
    steps: int
    scores: list[int]
    finished: bool
    win_order: list[int]
    drawn: bool
    gang_count: int
    illegal_action: bool = False
    error: str | None = None


@dataclass(frozen=True)
class ArenaReport:
    games: int
    unfinished: int
    illegal_actions: int
    zero_sum_violations: int
    draw_rate: float
    win_rate_by_seat: list[float]
    score_totals: list[int]
    average_scores: list[float]
    score_confidence95: list[float]
    winner_count_distribution: dict[int, int]
    gang_count_distribution: dict[int, int]
    results: list[ArenaGameResult] = field(repr=False)


def run_arena(policies: Sequence[BasePolicy], config: ArenaConfig | None = None) -> ArenaReport:
    cfg = config or ArenaConfig()
    if cfg.games < 0:
        raise ValueError("games must be non-negative")
    if len(policies) != 4:
        raise ValueError("arena requires exactly four policies")
    results = [run_arena_game(deepcopy(policies), seed=cfg.seed + offset, max_steps=cfg.max_steps) for offset in range(cfg.games)]

    return summarize_arena(results)


def run_arena_game(policies: Sequence[BasePolicy], *, seed: int, max_steps: int = 1000) -> ArenaGameResult:
    if len(policies) != 4:
        raise ValueError("arena requires exactly four policies")
    game = Game(seed=seed)
    state = game.reset()
    steps = 0
    illegal_action = False
    error: str | None = None

    try:
        while not state.finished and steps < max_steps:
            if state.phase in {"swap_three", "declare_void"}:
                for player in range(4):
                    if state.finished or state.phase not in {"swap_three", "declare_void"}:
                        break
                    if state.phase == "swap_three" and state.swap_choices[player] is not None:
                        continue
                    if state.phase == "declare_void" and state.void_suits[player] is not None:
                        continue
                    _step_policy_action(game, policies[player], player)
                    steps += 1
                    if steps >= max_steps:
                        break
                continue

            if state.pending_rob_kong is not None:
                player = state.pending_rob_kong.winners[0]
                _step_policy_action(game, policies[player], player)
                steps += 1
                continue

            if state.pending_discard is not None:
                resolved, decisions = _resolve_pending_discard(
                    game,
                    policies,
                    max_decisions=max_steps - steps,
                )
                steps += decisions
                if steps >= max_steps:
                    continue
                if not resolved and state.pending_discard is not None:
                    raise RuntimeError("pending discard responses made no progress")
                continue



            if state.phase == "play":
                _step_policy_action(game, policies[state.current_player], state.current_player)
                steps += 1
                continue

            break
        assert_zero_sum(state.scores)
    except Exception as exc:
        illegal_action = isinstance(exc, IllegalPolicyAction)
        error = str(exc)

    return ArenaGameResult(
        seed=seed,
        steps=steps,
        scores=list(state.scores),
        finished=state.finished and not illegal_action,
        win_order=list(state.win_order),
        drawn=state.finished and not state.win_order,
        gang_count=len(state.gang_records),
        illegal_action=illegal_action,
        error=error,
    )


def summarize_arena(results: Sequence[ArenaGameResult]) -> ArenaReport:
    games = len(results)
    score_totals = [sum(result.scores[player] for result in results) for player in range(4)]
    average_scores = [total / games if games else 0.0 for total in score_totals]
    return ArenaReport(
        games=games,
        unfinished=sum(not result.finished for result in results),
        illegal_actions=sum(result.illegal_action for result in results),
        zero_sum_violations=sum(sum(result.scores) != 0 for result in results),
        draw_rate=sum(result.drawn for result in results) / games if games else 0.0,
        win_rate_by_seat=[sum(player in result.win_order for result in results) / games if games else 0.0 for player in range(4)],
        score_totals=score_totals,
        average_scores=average_scores,
        score_confidence95=[_confidence95([result.scores[player] for result in results]) for player in range(4)],
        winner_count_distribution=dict(sorted(Counter(len(result.win_order) for result in results).items())),
        gang_count_distribution=dict(sorted(Counter(result.gang_count for result in results).items())),
        results=list(results),
    )


class IllegalPolicyAction(RuntimeError):
    pass


def _step_policy_action(game: Game, policy: BasePolicy, player: int) -> None:
    if game.state is None:
        raise RuntimeError("cannot choose an action before game reset")
    try:
        decision = choose_policy_action(game.state, player, policy)
    except ValueError as exc:
        raise IllegalPolicyAction(str(exc)) from exc
    game.step(player, decision.action)



def _resolve_pending_discard(
    game: Game,
    policies: Sequence[BasePolicy],
    max_decisions: int | None = None,
) -> tuple[bool, int]:

    state = game.state
    if state is None or state.pending_discard is None:
        return False, 0
    if max_decisions is not None and max_decisions <= 0:
        return False, 0
    if state.pending_winners:
        player = state.pending_winners[0]
        _step_policy_action(game, policies[player], player)
        return True, 1
    decisions = 0
    discarder = state.pending_discard.discarder
    for offset in range(1, 4):
        if max_decisions is not None and decisions >= max_decisions:
            break
        player = (discarder + offset) % 4
        if state.won[player] or player in state.pending_passers:
            continue
        try:
            decision = choose_policy_action(state, player, policies[player])
        except ValueError as exc:
            raise IllegalPolicyAction(str(exc)) from exc
        game.step(player, decision.action)
        decisions += 1
        if decision.action.kind is ActionKind.PASS:
            continue

        return True, decisions

    return False, decisions



def _confidence95(values: Sequence[int]) -> float:
    count = len(values)
    if count <= 1:
        return 0.0
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return 1.96 * sqrt(variance / count)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a deterministic S4 policy arena against three rule-policy opponents."
    )
    parser.add_argument("--seed", type=int, required=True, help="first game seed")
    parser.add_argument("--games", type=int, required=True, help="number of arena games (non-negative)")
    parser.add_argument("--model-seat", type=int, required=True, choices=range(4), help="seat occupied by the S4 policy")
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        required=True,
        help="S4 policy checkpoint; learned-belief checkpoints use sibling belief_s4.pt",
    )
    parser.add_argument(
        "--opponent",
        action="append",
        required=True,
        choices=("rule",),
        help="one opponent policy per non-model seat; only 'rule' is supported",
    )
    return parser


def _cli_report(report: ArenaReport, *, seed: int, model_seat: int) -> dict[str, object]:
    """Return the stable, compact JSON contract emitted by the arena command."""
    return {
        "seed": seed,
        "games": report.games,
        "model_seat": model_seat,
        "unfinished": report.unfinished,
        "illegal_actions": report.illegal_actions,
        "zero_sum_violations": report.zero_sum_violations,
        "draw_rate": report.draw_rate,
        "win_rate_by_seat": report.win_rate_by_seat,
        "average_scores": report.average_scores,
        "score_confidence95": report.score_confidence95,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the documented S4 gate command and emit one JSON report to stdout."""
    args = _cli_parser().parse_args(argv)
    if len(args.opponent) != 3:
        _cli_parser().error("exactly three --opponent rule values are required")
    if args.games < 0:
        _cli_parser().error("--games must be non-negative")

    try:
        checkpoint = args.policy_checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"policy checkpoint not found: {checkpoint}")
        # Accepted S4 v5 policies use learned belief and record the paired
        # artifact as a sibling.  Older prior-belief checkpoints remain valid
        # because LearnedPolicy only requires this argument when applicable.
        sibling_belief = checkpoint.with_name("belief_s4.pt")
        model_policy = LearnedPolicy(
            checkpoint,
            belief_model_path=sibling_belief if sibling_belief.is_file() else None,
        )
        policies: list[BasePolicy] = [RulePolicy(), RulePolicy(), RulePolicy(), RulePolicy()]
        policies[args.model_seat] = model_policy
        report = run_arena(policies, ArenaConfig(games=args.games, seed=args.seed))
    except Exception as exc:
        print(json.dumps({"error": str(exc), "error_type": type(exc).__name__}, sort_keys=True), file=sys.stderr)
        return 2

    print(json.dumps(_cli_report(report, seed=args.seed, model_seat=args.model_seat), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
