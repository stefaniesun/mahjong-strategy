from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.game import Game
from engine.settlement import assert_zero_sum
from policies.base_policy import BasePolicy
from policies.decision_boundary import choose_policy_action
from policies.rule_policy import RulePolicy



@dataclass(frozen=True)
class SelfplayResult:
    seed: int
    steps: int
    scores: list[int]
    finished: bool
    win_order: list[int]
    winner_count: int
    drawn: bool
    gang_count: int


@dataclass(frozen=True)
class SelfplayStats:
    games: int
    unfinished: int
    draw_rate: float
    average_winners: float
    score_totals: list[int]
    average_scores: list[float]
    winner_count_distribution: dict[int, int]
    gang_count_distribution: dict[int, int]


def run_selfplay_game(
    seed: int,
    max_steps: int = 1000,
    policies: Sequence[BasePolicy] | None = None,
) -> SelfplayResult:
    game = Game(seed=seed)
    state = game.reset()
    players = list(policies) if policies is not None else [RulePolicy() for _ in range(4)]
    if len(players) != 4:
        raise ValueError("selfplay requires exactly four policies")

    steps = 0
    while not state.finished and steps < max_steps:
        if state.phase in {"swap_three", "declare_void"}:
            for player in range(4):
                if state.finished or state.phase not in {"swap_three", "declare_void"}:
                    break
                if state.phase == "swap_three" and state.swap_choices[player] is not None:
                    continue
                if state.phase == "declare_void" and state.void_suits[player] is not None:
                    continue
                _step_policy_action(game, players[player], player)
                steps += 1
                if steps >= max_steps:
                    break
            continue

        if state.pending_rob_kong is not None:
            player = state.pending_rob_kong.winners[0]
            _step_policy_action(game, players[player], player)
            steps += 1
            continue

        if state.pending_discard is not None:
            resolved, decisions = _resolve_pending_discard(
                game,
                players,
                max_decisions=max_steps - steps,
            )
            steps += decisions
            if steps >= max_steps:
                continue
            if not resolved and state.pending_discard is not None:
                raise RuntimeError("pending discard responses made no progress")
            continue



        if state.phase == "play":
            _step_policy_action(game, players[state.current_player], state.current_player)
            steps += 1
            continue

        break

    assert_zero_sum(state.scores)
    return SelfplayResult(
        seed=seed,
        steps=steps,
        scores=list(state.scores),
        finished=state.finished,
        win_order=list(state.win_order),
        winner_count=len(state.win_order),
        drawn=state.finished and not state.win_order,
        gang_count=len(state.gang_records),
    )


def run_many(games: int, seed: int = 1, max_steps: int = 1000) -> list[SelfplayResult]:
    if games < 0:
        raise ValueError("games must be non-negative")
    return [run_selfplay_game(seed + offset, max_steps=max_steps) for offset in range(games)]


def summarize(results: Sequence[SelfplayResult]) -> SelfplayStats:
    games = len(results)
    score_totals = [sum(result.scores[player] for result in results) for player in range(4)]
    winner_counts = Counter(result.winner_count for result in results)
    gang_counts = Counter(result.gang_count for result in results)
    unfinished = sum(not result.finished for result in results)
    draws = sum(result.drawn for result in results)
    total_winners = sum(result.winner_count for result in results)
    return SelfplayStats(
        games=games,
        unfinished=unfinished,
        draw_rate=draws / games if games else 0.0,
        average_winners=total_winners / games if games else 0.0,
        score_totals=score_totals,
        average_scores=[total / games if games else 0.0 for total in score_totals],
        winner_count_distribution=dict(sorted(winner_counts.items())),
        gang_count_distribution=dict(sorted(gang_counts.items())),
    )


def _step_policy_action(game: Game, policy: BasePolicy, player: int) -> None:
    if game.state is None:
        raise RuntimeError("cannot choose an action before game reset")
    decision = choose_policy_action(game.state, player, policy)
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
        decision = choose_policy_action(state, player, policies[player])
        game.step(player, decision.action)
        decisions += 1
        if decision.action.kind.value == "pass":
            continue
        return True, decisions

    return False, decisions



def main() -> None:
    parser = argparse.ArgumentParser(description="Run S3 rule-policy Sichuan Mahjong selfplay.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--json", action="store_true", help="Print summary as JSON.")
    args = parser.parse_args()

    results = run_many(games=args.games, seed=args.seed, max_steps=args.max_steps)
    stats = summarize(results)
    for result in results:
        assert_zero_sum(result.scores)

    if args.json:
        print(json.dumps(asdict(stats), ensure_ascii=False, sort_keys=True))
    else:
        print(
            " ".join(
                [
                    f"games={stats.games}",
                    f"unfinished={stats.unfinished}",
                    f"draw_rate={stats.draw_rate:.3f}",
                    f"average_winners={stats.average_winners:.3f}",
                    "zero_sum=ok",
                ]
            )
        )

    if stats.unfinished:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
