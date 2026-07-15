from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from engine.actions import Action, ActionKind
from engine.hand import Hand
from engine.tiles import Suit
from policies.base_policy import BasePolicy
from policies.heuristics import choose_discard, choose_void_suit, visible_hand_and_void_suit
from policies.protocol_actions import actions_from_mask, validate_policy_action
from policies.rule_policy import RulePolicy
from state.protocol import S2ProtocolState



@dataclass(frozen=True)
class OpponentSpec:
    key: str
    strength: str
    label: str
    policy_cls: type[BasePolicy]
    weight: float


class RandomPolicy(BasePolicy):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose_action(self, protocol_state: S2ProtocolState, legal_mask: Sequence[bool]) -> Action:
        action = self._rng.choice(actions_from_mask(legal_mask))
        return validate_policy_action(action, legal_mask)


class GreedyPolicy(BasePolicy):
    def choose_action(self, protocol_state: S2ProtocolState, legal_mask: Sequence[bool]) -> Action:
        legal_actions = actions_from_mask(legal_mask)
        hand, void_suit = visible_hand_and_void_suit(protocol_state)

        for kind in (ActionKind.ROB_KONG_WIN, ActionKind.WIN, ActionKind.SELF_WIN):
            action = _first_kind(legal_actions, kind)
            if action is not None:
                return validate_policy_action(action, legal_mask)

        action = _choose_void_suit(hand, legal_actions)
        if action is None:
            action = _choose_discard(hand, void_suit, legal_actions)
        if action is None:
            action = _first_kind(legal_actions, ActionKind.KONG)
        if action is None:
            action = legal_actions[0]
        return validate_policy_action(action, legal_mask)



def create_standard_opponents() -> list[OpponentSpec]:
    return [
        OpponentSpec("random", "weak", "Random weak opponent", RandomPolicy, 0.20),
        OpponentSpec("greedy", "medium", "Greedy medium opponent", GreedyPolicy, 0.30),
        OpponentSpec("s3_rule", "baseline", "S3 rule baseline opponent", RulePolicy, 0.50),
    ]


def make_policy(key: str, seed: int | None = None) -> BasePolicy:
    specs = {opponent.key: opponent for opponent in create_standard_opponents()}
    if key not in specs:
        raise KeyError(f"unknown opponent key: {key}")
    policy_cls = specs[key].policy_cls
    if policy_cls is RandomPolicy:
        return RandomPolicy(seed=seed)
    return policy_cls()


def sample_opponents(size: int, seed: int | None = None) -> list[BasePolicy]:
    if size < 0:
        raise ValueError("size must be non-negative")
    rng = random.Random(seed)
    specs = create_standard_opponents()
    keys = [opponent.key for opponent in specs]
    weights = [opponent.weight for opponent in specs]
    chosen = rng.choices(keys, weights=weights, k=size)
    return [make_policy(key, seed=rng.randrange(2**32)) for key in chosen]


def standard_baseline() -> RulePolicy:
    return RulePolicy()


def _choose_void_suit(hand: Hand, legal_actions: Sequence[Action]) -> Action | None:
    declares = [action for action in legal_actions if action.kind is ActionKind.DECLARE_VOID]
    if not declares:
        return None
    preferred = choose_void_suit(hand)
    return next((action for action in declares if action.suit is preferred), declares[0])


def _choose_discard(hand: Hand, void_suit: Suit | None, legal_actions: Sequence[Action]) -> Action | None:
    discards = [action for action in legal_actions if action.kind is ActionKind.DISCARD and action.tile is not None]
    if not discards:
        return None
    preferred = choose_discard(hand, void_suit=void_suit)

    return next((action for action in discards if action.tile == preferred), min(discards, key=lambda action: action.tile.index if action.tile else 99))


def _first_kind(actions: Sequence[Action], kind: ActionKind) -> Action | None:
    return next((action for action in actions if action.kind is kind), None)
