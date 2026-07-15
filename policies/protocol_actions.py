from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from engine.actions import Action, ActionKind
from engine.gang import GangKind
from engine.tiles import Suit, parse_tile, tile_to_str
from state.action_space import action_space_size, action_to_index, index_to_action


def action_to_protocol(action: Action) -> dict[str, Any]:
    protocol_action: dict[str, Any] = {"kind": action.kind.value}
    if action.kind in {ActionKind.DISCARD, ActionKind.PONG, ActionKind.KONG}:
        if action.tile is None:
            raise ValueError(f"{action.kind.value} action requires a tile")
        protocol_action["tile"] = tile_to_str(action.tile)
    if action.kind is ActionKind.KONG:
        if action.kong_kind is None:
            raise ValueError("kong action requires a kong kind")
        protocol_action["kong_kind"] = action.kong_kind.value
    if action.kind is ActionKind.DECLARE_VOID:
        if action.suit is None:
            raise ValueError("declare_void action requires a suit")
        protocol_action["suit"] = action.suit.value
    if action.kind is ActionKind.SWAP_THREE:
        if len(action.tiles) != 3:
            raise ValueError("swap_three action requires exactly three tiles")
        protocol_action["tiles"] = [tile_to_str(tile) for tile in sorted(action.tiles)]
    try:
        canonical = index_to_action(action_to_index(protocol_action))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"action is not in the fixed action space: {action!r}") from exc
    canonical_action = action_from_protocol(canonical)
    if canonical_action != action:
        raise ValueError(f"action is not canonical for the fixed action space: {action!r}")
    return canonical



def action_from_protocol(protocol_action: dict[str, Any]) -> Action:
    try:
        canonical = index_to_action(action_to_index(protocol_action))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"protocol action is not in the fixed action space: {protocol_action!r}") from exc
    if canonical != protocol_action:
        raise ValueError(f"protocol action is not canonical: {protocol_action!r}")
    try:
        kind = ActionKind(canonical["kind"])
        tile = parse_tile(canonical["tile"]) if "tile" in canonical else None
        tiles = tuple(parse_tile(text) for text in canonical.get("tiles", ()))
        suit = Suit(canonical["suit"]) if "suit" in canonical else None
        kong_kind = GangKind(canonical["kong_kind"]) if "kong_kind" in canonical else None
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"protocol action is not in the fixed action space: {protocol_action!r}") from exc
    return Action(kind=kind, tiles=tiles, tile=tile, suit=suit, kong_kind=kong_kind)



# 最近一次通过校验的掩码对象(持有强引用,防止 id 复用误判)。
# 同一决策点内掩码会被校验两次(actions_from_mask + validate_policy_action),
# 内部流程从不改动已生成的掩码,重复整表扫描是纯开销。
_LAST_VALIDATED_MASK: Sequence[bool] | None = None


def validate_legal_mask(legal_mask: Sequence[bool]) -> None:
    global _LAST_VALIDATED_MASK
    if legal_mask is _LAST_VALIDATED_MASK:
        return
    expected = action_space_size()
    if len(legal_mask) != expected:
        raise ValueError(f"legal mask length must be {expected}, got {len(legal_mask)}")
    any_true = False
    for index, allowed in enumerate(legal_mask):
        # True/False 是单例,身份比较与 isinstance(bool) 等价(bool 不可子类化)且快数倍;
        # 该函数每决策点被调用多次、掩码约 640 项,曾是 profile 热点。
        if allowed is True:
            any_true = True
        elif allowed is not False:
            raise TypeError(f"legal mask entry {index} must be boolean, got {type(allowed).__name__}")
    if not any_true:
        raise ValueError("legal mask must contain at least one legal action")
    _LAST_VALIDATED_MASK = legal_mask


_ACTION_TABLE: list[Action] | None = None


def _action_table() -> list[Action]:
    # 动作空间固定(约 640 项),Action 是 frozen dataclass,可安全共享同一批实例。
    global _ACTION_TABLE
    if _ACTION_TABLE is None:
        _ACTION_TABLE = [action_from_protocol(index_to_action(index)) for index in range(action_space_size())]
    return _ACTION_TABLE


def actions_from_mask(legal_mask: Sequence[bool]) -> list[Action]:
    validate_legal_mask(legal_mask)
    table = _action_table()
    return [table[index] for index, allowed in enumerate(legal_mask) if allowed]


def validate_policy_action(action: Action, legal_mask: Sequence[bool]) -> Action:
    validate_legal_mask(legal_mask)
    index = action_to_index(action_to_protocol(action))
    if not legal_mask[index]:

        raise ValueError(f"policy action is outside the legal mask: {action!r}")
    return action
