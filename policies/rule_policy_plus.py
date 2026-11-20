"""S3+:在 S3 规则 AI 上逐项加入打牌经验,每一项都能单独开关,用配对对局测出它到底值多少分。

开关全关时打法必须和 S3 完全一样(有测试保证),所以测出来的差异只来自打开的那一项。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

from assistant.danger import wait_shapes
from engine.actions import Action, ActionKind
from engine.gang import GangKind
from engine.hand import Hand
from engine.meld import Meld, MeldKind
from engine.tiles import Suit, Tile, parse_tile
from policies.heuristics import (
    _discard_value_after_removal,
    _duplicate_count,
    choose_discard,
    visible_hand_and_void_suit,
)
from policies.protocol_actions import actions_from_mask, validate_policy_action
from policies.rule_policy import RulePolicy, _first_kind
from policies.shanten import best_discards, shanten, useful_tiles
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState


@dataclass(frozen=True)
class PlusConfig:
    smart_kong: bool = False  # 离听牌近时,杠了会变差就不杠(S3 能杠就杠)
    pong_no_regress: bool = False  # 碰后向听不退步就碰(S3 要求向听变小才碰)
    void_order: bool = False  # 打缺门:先打没现过的,再打现过的;同样没现过时先幺九、再偏张、后中张
    live_tiebreak: bool = False  # 向听相同时比实际剩余进张张数(S3 比进张种类)
    safe_tiebreak: bool = False  # 进张也相同时,优先打对更多在场对手 100% 安全、牌面上现得多的牌

    @classmethod
    def parse(cls, text: str) -> "PlusConfig":
        """"smart_kong,void_order" -> 打开这两项;"all" 全开;"none" 或空全关。"""
        names = {field.name for field in fields(cls)}
        if text == "all":
            return cls(**{name: True for name in names})
        wanted = {item.strip() for item in text.split(",") if item.strip() and item.strip() != "none"}
        unknown = wanted - names
        if unknown:
            raise ValueError(f"unknown S3+ switches: {sorted(unknown)}")
        return cls(**{name: True for name in wanted})


def _copy(hand: Hand) -> Hand:
    return Hand(counts=list(hand.counts), melds=list(hand.melds))


def _observed_list(value: ObservedValue) -> list:
    return list(value.value or []) if value.status is not ObservationStatus.UNKNOWN else []


def _remaining(state: S2ProtocolState) -> list[float]:
    """每种牌还没出现的张数(已现包括自己手里的)。"""
    seen = _observed_list(state.facts.seen_counts)
    return [max(0.0, 4.0 - float(seen[i])) if i < len(seen) else 4.0 for i in range(27)]


def _live_count(hand: Hand, void_suit: Suit | None, remaining: list[float]) -> float:
    return sum(remaining[parse_tile(text).index] for text in useful_tiles(hand, void_suit=void_suit))


def _table_seen(hand: Hand, tile: Tile, remaining: list[float]) -> float:
    """这张牌在牌面上(别人牌河、副露)现过几张,不算自己暗手牌。"""
    return max(0.0, 4.0 - remaining[tile.index] - hand.count(tile))


def _centrality(tile: Tile) -> int:
    return {1: 0, 9: 0, 2: 1, 8: 1}.get(tile.rank, 2)


def _safe_count(state: S2ProtocolState, tile: Tile, remaining: list[float]) -> int:
    """这张牌对几家在场对手 100% 安全:他缺这门,或者能等这张的形状全都凑不出来。"""
    count = 0
    for player in _observed_list(state.facts.players):
        if player.get("relative_position") == 0:
            continue
        won = player.get("won")
        if isinstance(won, ObservedValue) and won.value:
            continue
        void = player.get("void_suit")
        if isinstance(void, ObservedValue) and void.value == tile.suit.value:
            count += 1
            continue
        blocked = True
        for _, needed in wait_shapes(tile.index):
            need: dict[int, int] = {}
            for t in needed:
                need[t] = need.get(t, 0) + 1
            if all(remaining[t] >= n for t, n in need.items()):
                blocked = False
                break
        count += blocked
    return count


class RulePlusPolicy(RulePolicy):
    def __init__(self, config: PlusConfig | None = None) -> None:
        self.config = config or PlusConfig()

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
            action = self._choose_kong(protocol_state, hand, void_suit, legal_actions)
        if action is None:
            action = self._choose_discard_plus(protocol_state, hand, void_suit, legal_actions)
        if action is None:
            action = self._choose_pong_plus(hand, void_suit, legal_actions)
        if action is None:
            action = legal_actions[0]
        return validate_policy_action(action, legal_mask)

    # ---- 杠 ----

    def _choose_kong(self, state, hand: Hand, void_suit: Suit | None, legal_actions: Sequence[Action]) -> Action | None:
        kongs = [action for action in legal_actions if action.kind is ActionKind.KONG]
        if not kongs:
            return None
        if not self.config.smart_kong:
            return kongs[0]
        remaining = _remaining(state)
        return next((action for action in kongs if self._kong_worth_it(hand, void_suit, action, remaining)), None)

    def _kong_worth_it(self, hand: Hand, void_suit: Suit | None, action: Action, remaining: list[float]) -> bool:
        tile = action.tile
        after = _copy(hand)
        if action.kong_kind is GangKind.CONCEALED:
            for _ in range(4):
                after.remove(tile)
            after.add_meld(Meld(MeldKind.KONG, (tile,) * 4, exposed=False))
            before_shanten, before_live = self._best_after_discard(hand, void_suit, remaining)
        elif action.kong_kind is GangKind.ADDED:
            after.remove(tile)
            before_shanten, before_live = self._best_after_discard(hand, void_suit, remaining)
        else:  # 直明杠:别人打的,手里 3 张
            for _ in range(3):
                after.remove(tile)
            after.add_meld(Meld(MeldKind.KONG, (tile,) * 4, exposed=True))
            before_shanten, before_live = shanten(hand, void_suit=void_suit), _live_count(hand, void_suit, remaining)
        if before_shanten >= 2:
            return True  # 离听牌远,杠的钱和多摸的一张更值
        after_shanten = shanten(after, void_suit=void_suit)
        if after_shanten != before_shanten:
            return after_shanten < before_shanten
        return before_shanten > 0 or _live_count(after, void_suit, remaining) >= before_live

    def _best_after_discard(self, hand: Hand, void_suit: Suit | None, remaining: list[float]) -> tuple[int, float]:
        best: tuple[int, float] | None = None
        for tile in set(hand.tiles()):
            trial = _copy(hand)
            trial.remove(tile)
            s = shanten(trial, void_suit=void_suit)
            if best is None or s < best[0]:
                best = (s, _live_count(trial, void_suit, remaining))
            elif s == best[0]:
                best = (s, max(best[1], _live_count(trial, void_suit, remaining)))
        return best if best is not None else (8, 0.0)

    # ---- 出牌 ----

    def _choose_discard_plus(self, state, hand: Hand, void_suit: Suit | None, legal_actions: Sequence[Action]) -> Action | None:
        discards = [action for action in legal_actions if action.kind is ActionKind.DISCARD and action.tile is not None]
        if not discards:
            return None
        preferred = self._preferred_discard(state, hand, void_suit)
        for action in discards:
            if action.tile == preferred:
                return action
        return min(discards, key=lambda action: action.tile.index)

    def _preferred_discard(self, state, hand: Hand, void_suit: Suit | None) -> Tile:
        cfg = self.config
        void_tiles = [tile for tile in set(hand.tiles()) if void_suit is not None and tile.suit is void_suit]
        if void_tiles and cfg.void_order:
            remaining = _remaining(state)
            return min(
                void_tiles,
                key=lambda t: (_table_seen(hand, t, remaining), _duplicate_count(hand, t), _centrality(t), t.index),
            )
        if void_tiles or not (cfg.live_tiebreak or cfg.safe_tiebreak):
            return choose_discard(hand, void_suit=void_suit)

        remaining = _remaining(state)
        texts = set(best_discards(hand, void_suit=void_suit))
        candidates = [tile for tile in set(hand.tiles()) if f"{tile.rank}{tile.suit.value}" in texts]

        def key(tile: Tile) -> tuple:
            trial = _copy(hand)
            trial.remove(tile)
            useful, *rest = _discard_value_after_removal(hand, tile, void_suit)
            primary = _live_count(trial, void_suit, remaining) if cfg.live_tiebreak else useful
            safety = (_safe_count(state, tile, remaining), _table_seen(hand, tile, remaining)) if cfg.safe_tiebreak else (0, 0)
            return (primary, *safety, *rest)

        return max(candidates, key=key)

    # ---- 碰 ----

    def _choose_pong_plus(self, hand: Hand, void_suit: Suit | None, legal_actions: Sequence[Action]) -> Action | None:
        if not self.config.pong_no_regress:
            return self._choose_pong(hand, void_suit, legal_actions)
        pong = _first_kind(legal_actions, ActionKind.PONG)
        if pong is None or pong.tile is None:
            return None
        trial = _copy(hand)
        trial.remove(pong.tile)
        trial.remove(pong.tile)
        trial.add_meld(Meld(MeldKind.PONG, (pong.tile,) * 3, exposed=True))
        best = min(shanten(self._without(trial, t), void_suit=void_suit) for t in set(trial.tiles()))
        if best <= shanten(hand, void_suit=void_suit):
            return pong
        return _first_kind(legal_actions, ActionKind.PASS)

    @staticmethod
    def _without(hand: Hand, tile: Tile) -> Hand:
        trial = _copy(hand)
        trial.remove(tile)
        return trial
