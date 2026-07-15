from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import Action, ActionKind
from engine.game import Game
from engine.settlement import assert_zero_sum
from engine.state import GameState
from engine.tiles import Tile, tile_to_str
from policies.base_policy import BasePolicy
from policies.decision_boundary import PolicyDecision, choose_policy_action
from policies.rule_policy import RulePolicy

from selfplay.run_selfplay import SelfplayResult


def export_s3_replay(
    seed: int,
    max_steps: int = 1000,
    game_id: str | None = None,
    policies: Sequence[BasePolicy] | None = None,
) -> dict[str, Any]:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int):
        raise TypeError("max_steps must be a non-negative integer")
    if max_steps < 0:
        raise ValueError("max_steps must be a non-negative integer")

    game = Game(seed=seed)

    state = game.reset()
    players = list(policies) if policies is not None else [RulePolicy() for _ in range(4)]
    if len(players) != 4:
        raise ValueError("replay export requires exactly four policies")

    replay: dict[str, Any] = {
        "schema": "s3.replay.v1",
        "meta": {
            "game_id": game_id or f"s3-seed-{seed}",
            "seed": seed,
            "max_steps": max_steps,
            "engine": "sichuan-mahjong-engine",
            "policy": "RulePolicy",
        },
        "players": [{"id": player, "name": f"P{player}", "policy": type(players[player]).__name__} for player in range(4)],
        "initial_state": _state_to_replay_dict(state),
        "steps": [],
    }

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
                _append_policy_step(replay, game, players[player], player, steps)
                steps += 1
                if steps >= max_steps:
                    break
            continue

        if state.pending_rob_kong is not None:
            player = state.pending_rob_kong.winners[0]
            _append_policy_step(replay, game, players[player], player, steps)
            steps += 1
            continue

        if state.pending_discard is not None:
            resolved, decisions = _resolve_pending_discard(
                replay,
                game,
                players,
                steps,
                max_decisions=max_steps - steps,
            )
            steps += decisions
            if steps >= max_steps:
                continue
            if not resolved and state.pending_discard is not None:
                raise RuntimeError("pending discard responses made no progress")
            continue



        if state.phase == "play":
            _append_policy_step(replay, game, players[state.current_player], state.current_player, steps)
            steps += 1
            continue

        break

    assert_zero_sum(state.scores)
    result = SelfplayResult(
        seed=seed,
        steps=steps,
        scores=list(state.scores),
        finished=state.finished,
        win_order=list(state.win_order),
        winner_count=len(state.win_order),
        drawn=state.finished and not state.win_order,
        gang_count=len(state.gang_records),
    )
    replay["final_state"] = _state_to_replay_dict(state)
    replay["result"] = {
        "seed": result.seed,
        "steps": result.steps,
        "scores": result.scores,
        "finished": result.finished,
        "win_order": result.win_order,
        "winner_count": result.winner_count,
        "drawn": result.drawn,
        "gang_count": result.gang_count,
    }
    return replay


def write_replay_json(replay: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _append_policy_step(
    replay: dict[str, Any],
    game: Game,
    policy: BasePolicy,
    player: int,
    step: int,
) -> None:
    if game.state is None:
        raise RuntimeError("cannot export replay before game reset")
    decision = choose_policy_action(game.state, player, policy)
    _append_decision_step(replay, game, player, step, decision)


def _append_decision_step(
    replay: dict[str, Any],
    game: Game,
    player: int,
    step: int,
    decision: PolicyDecision,
) -> None:
    if game.state is None:
        raise RuntimeError("cannot export replay before game reset")
    before = _state_to_replay_dict(game.state)
    game.step(player, decision.action)
    after = _state_to_replay_dict(game.state)
    replay["steps"].append(
        {
            "step": step,
            "player": player,
            "phase": before["phase"],
            "protocol_state": decision.protocol_state.to_dict(),
            "legal_mask": list(decision.legal_mask),
            "action": _action_to_dict(decision.action),

            "legal_actions": [
                _action_to_dict(item)
                for item in decision.legal_actions
            ],
            "before": before,
            "after": after,
        }
    )



def _resolve_pending_discard(
    replay: dict[str, Any],
    game: Game,
    policies: Sequence[BasePolicy],
    step: int,
    max_decisions: int | None = None,
) -> tuple[bool, int]:
    state = game.state
    if state is None or state.pending_discard is None:
        return False, 0
    if max_decisions is not None and max_decisions <= 0:
        return False, 0

    if state.pending_winners:
        player = state.pending_winners[0]
        _append_policy_step(replay, game, policies[player], player, step)
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
        _append_decision_step(
            replay,
            game,
            player,
            step + decisions,
            decision,
        )
        decisions += 1
        if decision.action.kind is ActionKind.PASS:
            continue
        return True, decisions

    return False, decisions



def _state_to_replay_dict(state: GameState) -> dict[str, Any]:
    base = state.to_dict()
    base.pop("pending_winners", None)
    pending_rob_kong = base.get("pending_rob_kong")
    if pending_rob_kong is not None:
        pending_rob_kong.pop("winners", None)
    base["players"] = [

        {
            "id": player,
            "hand": [_tile_to_text(tile) for tile in state.hands[player].tiles()],
            "melds": [_meld_to_dict(meld) for meld in state.hands[player].melds],
            "river": [_tile_to_text(tile) for tile in state.rivers[player]],
            "void_suit": None if state.void_suits[player] is None else state.void_suits[player].value,
            "score": state.scores[player],
            "won": state.won[player],
            "swap_choice": None
            if state.swap_choices[player] is None
            else [_tile_to_text(tile) for tile in state.swap_choices[player]],
        }
        for player in range(4)
    ]
    return base


def _meld_to_dict(meld: Any) -> dict[str, Any]:
    return {
        "kind": meld.kind.value,
        "tiles": [_tile_to_text(tile) for tile in meld.tiles],
        "exposed": meld.exposed,
        "from_player": meld.from_player,
    }


def _action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "tiles": [_tile_to_text(tile) for tile in action.tiles],
        "tile": None if action.tile is None else _tile_to_text(action.tile),
        "suit": None if action.suit is None else action.suit.value,
        "kong_kind": None if action.kong_kind is None else action.kong_kind.value,
    }


def _tile_to_text(tile: Tile) -> str:
    return tile_to_str(tile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a complete S3 selfplay replay JSON for the HTML viewer.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--output", default="docs/replays/s3_replay_seed_1.json")
    args = parser.parse_args()

    replay = export_s3_replay(seed=args.seed, max_steps=args.max_steps, game_id=args.game_id)
    write_replay_json(replay, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
