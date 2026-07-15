from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import Action, ActionKind
from engine.game import Game
from engine.settlement import assert_zero_sum


@dataclass(frozen=True)
class PlayoutResult:
    seed: int
    steps: int
    scores: list[int]
    finished: bool
    win_order: list[int]


def _choose_action(game: Game, player: int, rng: random.Random) -> Action:
    actions = game.legal_actions(player)
    if not actions:
        raise RuntimeError(f"player {player} has no legal actions in phase {game.state.phase if game.state else None}")
    win_actions = [action for action in actions if action.kind in {ActionKind.WIN, ActionKind.SELF_WIN}]
    if win_actions:
        return win_actions[0]
    discard_actions = [action for action in actions if action.kind is ActionKind.DISCARD]
    if discard_actions:
        return rng.choice(discard_actions)
    return rng.choice(actions)


def run_random_game(seed: int, max_steps: int = 1000) -> PlayoutResult:
    rng = random.Random(seed)
    game = Game(seed=seed)
    state = game.reset()
    steps = 0

    while not state.finished and steps < max_steps:
        if state.phase in {"swap_three", "declare_void"}:
            for player in range(4):
                if state.phase == "swap_three" and state.swap_choices[player] is not None:
                    continue
                if state.phase == "declare_void" and state.void_suits[player] is not None:
                    continue
                game.step(player, _choose_action(game, player, rng))
                steps += 1
        elif state.pending_discard is not None:
            responders = list(state.pending_winners)
            if not responders:
                discarder = state.pending_discard.discarder
                responders = [
                    player
                    for offset in range(1, 4)
                    if game.legal_actions(player := (discarder + offset) % 4)
                ]
            game.step(responders[0], _choose_action(game, responders[0], rng))
            steps += 1

        elif state.phase == "play":
            game.step(state.current_player, _choose_action(game, state.current_player, rng))
            steps += 1
        else:
            break

    assert_zero_sum(state.scores)
    return PlayoutResult(
        seed=seed,
        steps=steps,
        scores=list(state.scores),
        finished=state.finished,
        win_order=list(state.win_order),
    )


def run_many(games: int, seed: int = 1, max_steps: int = 1000) -> list[PlayoutResult]:
    return [run_random_game(seed + offset, max_steps=max_steps) for offset in range(games)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random Sichuan Mahjong playouts.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    results = run_many(games=args.games, seed=args.seed, max_steps=args.max_steps)
    unfinished = [result for result in results if not result.finished]
    for result in results:
        assert_zero_sum(result.scores)
    print(f"games={len(results)} unfinished={len(unfinished)} zero_sum=ok")
    if unfinished:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
