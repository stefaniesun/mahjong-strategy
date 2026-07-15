from __future__ import annotations

from engine.gang import GangKind
from engine.tiles import SUITS, Tile, tile_to_str
from state.legality import legal_actions
from state.protocol import S2ProtocolState


_ACTIONS: list[dict] = []


def _key(action: dict) -> tuple:
    kind = action["kind"]
    if kind in {"discard", "pong"}:
        return (kind, action["tile"])
    if kind == "kong":
        return (kind, action["tile"], action["kong_kind"])
    if kind == "declare_void":
        return (kind, action["suit"])
    if kind == "swap_three":
        return (kind, tuple(action["tiles"]))
    return (kind,)


for index in range(27):

    _ACTIONS.append({"kind": "discard", "tile": tile_to_str(Tile.from_index(index))})
for kind in ("pong", "win", "self_win", "rob_kong_win", "pass"):
    if kind == "pong":
        for index in range(27):
            _ACTIONS.append({"kind": "pong", "tile": tile_to_str(Tile.from_index(index))})
    else:
        _ACTIONS.append({"kind": kind})
for kong_kind in (GangKind.EXPOSED.value, GangKind.CONCEALED.value, GangKind.ADDED.value):
    for index in range(27):
        _ACTIONS.append({"kind": "kong", "tile": tile_to_str(Tile.from_index(index)), "kong_kind": kong_kind})
for suit in SUITS:
    _ACTIONS.append({"kind": "declare_void", "suit": suit.value})
for first in range(27):
    for second in range(first, 27):
        for third in range(second, 27):
            tiles = [Tile.from_index(first), Tile.from_index(second), Tile.from_index(third)]
            if len({tile.suit for tile in tiles}) == 1:
                _ACTIONS.append({"kind": "swap_three", "tiles": [tile_to_str(tile) for tile in tiles]})

_INDEX = {_key(action): index for index, action in enumerate(_ACTIONS)}


def action_space_size() -> int:
    return len(_ACTIONS)


def index_to_action(index: int) -> dict:
    return dict(_ACTIONS[index])


def action_to_index(action: dict) -> int:
    return _INDEX[_key(action)]


def legal_mask(state: S2ProtocolState) -> list[bool]:
    mask = [False] * action_space_size()
    for action in legal_actions(state):
        clean_action = {key: value for key, value in action.items() if key not in {"conditionally_legal", "depends_on"}}
        mask[action_to_index(clean_action)] = True
    return mask

