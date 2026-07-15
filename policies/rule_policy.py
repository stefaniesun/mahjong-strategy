from __future__ import annotations

from collections.abc import Sequence

from engine.actions import Action, ActionKind
from engine.hand import Hand
from engine.tiles import Suit
from policies.base_policy import BasePolicy
from policies.heuristics import choose_discard, choose_swap_tiles, choose_void_suit, should_pong, visible_hand_and_void_suit
from policies.protocol_actions import actions_from_mask, validate_policy_action
from state.protocol import S2ProtocolState



class RulePolicy(BasePolicy):
    def choose_action(self, protocol_state: S2ProtocolState, legal_mask: Sequence[bool]) -> Action:
        legal_actions = actions_from_mask(legal_mask)
        hand, void_suit = visible_hand_and_void_suit(protocol_state)

        for kind in (ActionKind.ROB_KONG_WIN, ActionKind.WIN, ActionKind.SELF_WIN):
            action = _first_kind(legal_actions, kind)
            if action is not None:
                return validate_policy_action(action, legal_mask)

        action = self._choose_swap(hand, legal_actions)
        if action is None:
            action = self._choose_declare_void(hand, legal_actions)
        if action is None:
            action = _first_kind(legal_actions, ActionKind.KONG)
        if action is None:
            action = self._choose_discard(hand, void_suit, legal_actions)
        if action is None:
            action = self._choose_pong(hand, void_suit, legal_actions)
        if action is None:
            action = legal_actions[0]
        return validate_policy_action(action, legal_mask)

    def _choose_swap(self, hand: Hand, legal_actions: Sequence[Action]) -> Action | None:

        swaps = [action for action in legal_actions if action.kind is ActionKind.SWAP_THREE]
        if not swaps:
            return None
        preferred = set(choose_swap_tiles(hand))

        for action in swaps:
            if set(action.tiles) == preferred:
                return action
        return min(swaps, key=lambda action: tuple(tile.index for tile in action.tiles))

    def _choose_declare_void(self, hand: Hand, legal_actions: Sequence[Action]) -> Action | None:
        declares = [action for action in legal_actions if action.kind is ActionKind.DECLARE_VOID]
        if not declares:
            return None
        preferred = choose_void_suit(hand)

        for action in declares:
            if action.suit is preferred:
                return action
        return declares[0]

    def _choose_discard(self, hand: Hand, void_suit: Suit | None, legal_actions: Sequence[Action]) -> Action | None:
        discards = [action for action in legal_actions if action.kind is ActionKind.DISCARD and action.tile is not None]
        if not discards:
            return None
        preferred = choose_discard(hand, void_suit=void_suit)

        for action in discards:
            if action.tile == preferred:
                return action
        return min(discards, key=lambda action: action.tile.index if action.tile is not None else 99)

    def _choose_pong(self, hand: Hand, void_suit: Suit | None, legal_actions: Sequence[Action]) -> Action | None:
        pong = _first_kind(legal_actions, ActionKind.PONG)
        if pong is None or pong.tile is None:
            return None
        if should_pong(hand, pong.tile, void_suit=void_suit):

            return pong
        return _first_kind(legal_actions, ActionKind.PASS)


def _first_kind(actions: Sequence[Action], kind: ActionKind) -> Action | None:
    return next((action for action in actions if action.kind is kind), None)
