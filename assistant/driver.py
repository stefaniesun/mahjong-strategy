"""用引擎跑一局,在每个"有选择"的决策点回调,供回放查看器和对账工具使用。

推进逻辑与 learning/eval/arena.py 一致;只有一个合法动作(比如只能"过")的点不算决策点。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from engine.actions import ActionKind
from engine.game import Game
from policies.base_policy import BasePolicy
from policies.decision_boundary import PolicyDecision, choose_policy_action
from policies.rule_policy import RulePolicy

from assistant.recorder import PublicRecorder

DecisionHook = Callable[[Game, PublicRecorder, int, PolicyDecision], None]


def run_game(
    seed: int,
    policies: Sequence[BasePolicy] | None = None,
    on_decision: DecisionHook | None = None,
    max_steps: int = 1000,
    dealer: int = 0,
) -> tuple[Game, PublicRecorder]:
    game = Game(seed=seed, dealer=dealer)
    state = game.reset()
    recorder = PublicRecorder(game)
    players = list(policies) if policies is not None else [RulePolicy() for _ in range(4)]
    steps = 0

    def act(player: int) -> PolicyDecision:
        nonlocal steps
        decision = choose_policy_action(game.state, player, players[player])
        if on_decision is not None and len(decision.legal_actions) > 1:
            on_decision(game, recorder, player, decision)
        recorder.step(player, decision.action)
        steps += 1
        return decision

    while not state.finished and steps < max_steps:
        if state.phase in ("swap_three", "declare_void"):
            for player in range(4):
                if state.finished or state.phase not in ("swap_three", "declare_void"):
                    break
                if state.phase == "swap_three" and state.swap_choices[player] is not None:
                    continue
                if state.phase == "declare_void" and state.void_suits[player] is not None:
                    continue
                act(player)
            continue
        if state.pending_rob_kong is not None:
            act(state.pending_rob_kong.winners[0])
            continue
        if state.pending_discard is not None:
            if state.pending_winners:
                act(state.pending_winners[0])
                continue
            discarder = state.pending_discard.discarder
            resolved = False
            for offset in range(1, 4):
                player = (discarder + offset) % 4
                if state.won[player] or player in state.pending_passers:
                    continue
                if act(player).action.kind is not ActionKind.PASS:
                    resolved = True
                    break
            if not resolved and state.pending_discard is not None:
                raise RuntimeError("pending discard responses made no progress")
            continue
        if state.phase == "play":
            act(state.current_player)
            continue
        break
    return game, recorder
