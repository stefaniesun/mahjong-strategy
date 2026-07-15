from __future__ import annotations

from dataclasses import dataclass

from engine.actions import Action
from engine.state import GameState
from policies.base_policy import BasePolicy
from policies.protocol_actions import actions_from_mask, validate_policy_action
from state.action_space import legal_mask
from state.adapters.from_engine import from_engine
from state.protocol import S2ProtocolState


@dataclass(frozen=True)
class PolicyDecision:
    protocol_state: S2ProtocolState
    legal_mask: list[bool]
    legal_actions: list[Action]
    action: Action


def choose_policy_action(engine_state: GameState, player: int, policy: BasePolicy) -> PolicyDecision:
    protocol_state = from_engine(engine_state, player_id=player)
    mask = legal_mask(protocol_state)
    actions = actions_from_mask(mask)
    action = validate_policy_action(policy.choose_action(protocol_state, mask), mask)
    return PolicyDecision(
        protocol_state=protocol_state,
        legal_mask=mask,
        legal_actions=actions,
        action=action,
    )
