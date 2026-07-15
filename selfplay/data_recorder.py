from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.actions import Action, ActionKind
from engine.game import Game
from policies.base_policy import BasePolicy
from policies.decision_boundary import PolicyDecision, choose_policy_action
from policies.rule_policy import RulePolicy
from selfplay.run_selfplay import SelfplayResult
from state.tile_belief import generate_belief_labels



@dataclass(frozen=True)
class DecisionRecord:
    game_id: str
    step: int
    player: int
    phase: str
    state: dict[str, Any]
    legal_actions: list[dict[str, Any]]
    action: dict[str, Any]
    final_scores: list[int]
    labels: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "step": self.step,
            "player": self.player,
            "phase": self.phase,
            "state": self.state,
            "legal_actions": self.legal_actions,
            "action": self.action,
            "final_scores": list(self.final_scores),
            "labels": self.labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        return cls(
            game_id=str(data["game_id"]),
            step=int(data["step"]),
            player=int(data["player"]),
            phase=str(data["phase"]),
            state=dict(data["state"]),
            legal_actions=list(data["legal_actions"]),
            action=dict(data["action"]),
            final_scores=list(data["final_scores"]),
            labels=dict(data["labels"]),
        )


class SelfplayDataRecorder:
    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self._pending: list[dict[str, Any]] = []
        self.records: list[DecisionRecord] = []

    def record_decision(self, *, step: int, game: Game, player: int, decision: PolicyDecision) -> None:
        if game.state is None:
            raise RuntimeError("cannot record decision before game reset")
        self._pending.append(
            {
                "game_id": self.game_id,
                "step": step,
                "player": player,
                "phase": game.state.phase,
                "state": decision.protocol_state.to_dict(),
                "legal_actions": [_action_to_dict(item) for item in decision.legal_actions],
                "action": _action_to_dict(decision.action),
                "labels": generate_belief_labels(game.state, decision.protocol_state),

            }
        )

    def finalize_game(self, final_scores: Sequence[int]) -> list[DecisionRecord]:
        scores = list(final_scores)
        self.records = [DecisionRecord(final_scores=scores, **item) for item in self._pending]
        return list(self.records)


def run_recorded_selfplay_game(
    game_id: str,
    seed: int,
    max_steps: int = 1000,
    policies: Sequence[BasePolicy] | None = None,
) -> tuple[SelfplayResult, list[DecisionRecord]]:
    game = Game(seed=seed)
    state = game.reset()
    players = list(policies) if policies is not None else [RulePolicy() for _ in range(4)]
    if len(players) != 4:
        raise ValueError("recorded selfplay requires exactly four policies")

    recorder = SelfplayDataRecorder(game_id=game_id)
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
                _step_recorded_policy_action(game, players[player], player, recorder, steps)
                steps += 1
                if steps >= max_steps:
                    break
            continue

        if state.pending_rob_kong is not None:
            player = state.pending_rob_kong.winners[0]
            _step_recorded_policy_action(game, players[player], player, recorder, steps)
            steps += 1
            continue

        if state.pending_discard is not None:
            resolved, decisions = _resolve_recorded_pending_discard(
                game,
                players,
                recorder,
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
            _step_recorded_policy_action(game, players[state.current_player], state.current_player, recorder, steps)
            steps += 1
            continue

        break

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
    return result, recorder.finalize_game(state.scores)


def write_jsonl(records: Sequence[DecisionRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            file.write("\n")


def load_jsonl(path: str | Path) -> list[DecisionRecord]:
    with Path(path).open("r", encoding="utf-8") as file:
        return [DecisionRecord.from_dict(json.loads(line)) for line in file if line.strip()]


def _step_recorded_policy_action(
    game: Game,
    policy: BasePolicy,
    player: int,
    recorder: SelfplayDataRecorder,
    step: int,
) -> None:
    if game.state is None:
        raise RuntimeError("cannot choose an action before game reset")
    decision = choose_policy_action(game.state, player, policy)
    recorder.record_decision(step=step, game=game, player=player, decision=decision)
    game.step(player, decision.action)



def _resolve_recorded_pending_discard(
    game: Game,
    policies: Sequence[BasePolicy],
    recorder: SelfplayDataRecorder,
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
        _step_recorded_policy_action(game, policies[player], player, recorder, step)
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
        recorder.record_decision(
            step=step + decisions,
            game=game,
            player=player,
            decision=decision,
        )
        game.step(player, decision.action)
        decisions += 1
        if decision.action.kind is ActionKind.PASS:
            continue

        return True, decisions

    return False, decisions



def _action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "tiles": [str(tile) for tile in action.tiles],
        "tile": None if action.tile is None else str(action.tile),
        "suit": None if action.suit is None else action.suit.value,
        "kong_kind": None if action.kong_kind is None else action.kong_kind.value,
    }
