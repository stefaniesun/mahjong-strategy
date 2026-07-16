from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.tiles import Tile, parse_tile, tile_to_str
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState

ENCODER_VERSION = "s2.v4.encoder.v3"

_TILE_COUNT = 27
_LOCATION_ORDER = ("wall", "1", "2", "3")
_PHASE_ORDER = ("exchange", "dingque", "play", "settlement")
# Per-player discard sequence boundaries: 1-6 early, 7-12 middle, 13+ late.
_DISCARD_EARLY_END = 6
_DISCARD_MIDDLE_END = 12

_SECTION_SPECS: tuple[tuple[str, int, str], ...] = (
    ("phase", 6, "phase one-hot(4), unknown flag, confidence"),
    ("current_player_relative", 6, "relative current player one-hot(4), unknown flag, confidence"),
    ("wall_count", 3, "wall count / 108, unknown flag, confidence"),
    ("own_hand_counts", 29, "27 own concealed tile counts / 4, unknown flag, confidence"),
    ("seen_counts", 29, "27 seen counts / 4, unknown flag, confidence"),
    ("remaining_tile_counts", 29, "27 remaining counts / 4, unknown flag, confidence"),
    ("unknown_pool_breakdown", 6, "wall and opponent unknown pool sizes / 108, unknown flag, confidence"),
    ("player_void_suits", 20, "four relative players x void suit one-hot(3), unknown flag, confidence"),
    ("tile_location_beliefs", 110, "27 tiles x wall/next/across/previous probabilities, unknown flag, confidence"),

    ("opponent_tenpai_beliefs", 5, "next/across/previous tenpai probabilities, unknown flag, confidence"),
    ("discard_danger_summary", 5, "max discard danger by opponent, any danger, unknown flag, confidence"),
    ("observation_summary", 5, "observation start / 108, observed ratio, estimated ratio, degraded flag, confidence"),
    ("rule_config_summary", 4, "rule config numeric summary, unknown flag, confidence"),
    ("legal_action_summary", 6, "discard/pong/kong/hu/pass/action counts normalized, unknown flag"),
    ("opponent_discard_counts", 87, "three opponents x 27 discard counts / 4, unknown flag, confidence"),
    ("opponent_discard_phases", 249, "three opponents x early/middle/late discard counts / 4, unknown flag, confidence"),
    ("opponent_meld_tiles", 87, "three opponents x 27 public meld tile counts / 4, unknown flag, confidence"),
    ("last_discards", 120, "four relative players x last discard one-hot(27), none flag, unknown flag, confidence"),
)



@dataclass(frozen=True)
class EncodingSection:
    name: str
    offset: int
    end: int
    size: int
    description: str


@dataclass(frozen=True)
class EncodedState:
    values: tuple[float, ...]
    sections: tuple[EncodingSection, ...]
    version: str = ENCODER_VERSION

    @property
    def size(self) -> int:
        return len(self.values)

    def section(self, name: str) -> EncodingSection:
        for section in self.sections:
            if section.name == name:
                return section
        raise KeyError(name)


def encode_state(state: S2ProtocolState) -> EncodedState:
    values: list[float] = []
    values.extend(_categorical_observed(state.phase, _PHASE_ORDER))
    values.extend(_categorical_observed(state.current_player_relative, (0, 1, 2, 3)))
    values.extend(_scalar_observed(state.facts.wall_count, scale=108.0))
    values.extend(_own_hand_counts(state))
    values.extend(_count_vector(state.facts.seen_counts))
    values.extend(_count_vector(state.statistics.remaining_tile_counts))
    values.extend(_unknown_pool_breakdown(state.statistics.unknown_pool_breakdown))
    values.extend(_player_void_suits(state.facts.players))
    values.extend(_tile_location_beliefs(state.beliefs.tile_location_beliefs))

    values.extend(_opponent_tenpai_beliefs(state.beliefs.opponent_tenpai_beliefs))
    values.extend(_discard_danger_summary(state.beliefs.discard_danger))
    values.extend(_observation_summary(state))
    values.extend(_rule_config_summary(state.rule_config))
    values.extend(_legal_action_summary(state.legal_actions))
    values.extend(_opponent_discard_counts(state.facts.players))
    values.extend(_opponent_discard_phases(state.facts.players))
    values.extend(_opponent_meld_tiles(state.facts.players))
    values.extend(_last_discards(state.facts.players))
    return EncodedState(values=tuple(float(value) for value in values), sections=_sections())



def encoding_table() -> list[dict[str, Any]]:
    return [section.__dict__.copy() for section in _sections()]


def _sections() -> tuple[EncodingSection, ...]:
    offset = 0
    sections: list[EncodingSection] = []
    for name, size, description in _SECTION_SPECS:
        sections.append(EncodingSection(name=name, offset=offset, end=offset + size, size=size, description=description))
        offset += size
    return tuple(sections)


def _categorical_observed(value: ObservedValue[Any], categories: tuple[Any, ...]) -> list[float]:
    encoded = [0.0] * len(categories)
    if value.status is not ObservationStatus.UNKNOWN and value.value in categories:
        encoded[categories.index(value.value)] = 1.0
    return encoded + [_unknown_flag(value), value.confidence]


def _scalar_observed(value: ObservedValue[Any], scale: float) -> list[float]:
    normalized = 0.0 if value.status is ObservationStatus.UNKNOWN or value.value is None else _clamp(float(value.value) / scale)
    return [normalized, _unknown_flag(value), value.confidence]


def _count_vector(value: ObservedValue[list[float]]) -> list[float]:
    counts = [0.0] * _TILE_COUNT if value.status is ObservationStatus.UNKNOWN or value.value is None else list(value.value[:_TILE_COUNT])
    counts = (counts + [0.0] * _TILE_COUNT)[:_TILE_COUNT]
    return [_clamp(float(count) / 4.0) for count in counts] + [_unknown_flag(value), value.confidence]


def _own_hand_counts(state: S2ProtocolState) -> list[float]:
    players = state.facts.players
    if players.status is ObservationStatus.UNKNOWN or players.value is None:
        return [0.0] * _TILE_COUNT + [1.0, 0.0]
    for player in players.value:
        if player.get("relative_position") != 0:
            continue
        concealed = player.get("concealed_hand")
        if not isinstance(concealed, ObservedValue) or concealed.status is ObservationStatus.UNKNOWN:
            confidence = 0.0 if not isinstance(concealed, ObservedValue) else concealed.confidence
            return [0.0] * _TILE_COUNT + [1.0, confidence]
        counts = [0.0] * _TILE_COUNT
        for tile_text in concealed.value or []:
            counts[parse_tile(tile_text).index] += 1.0
        return [_clamp(count / 4.0) for count in counts] + [0.0, concealed.confidence]
    return [0.0] * _TILE_COUNT + [1.0, 0.0]


def _unknown_pool_breakdown(value: ObservedValue[dict[str, Any]]) -> list[float]:
    if value.status is ObservationStatus.UNKNOWN or value.value is None:
        return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    opponents = value.value.get("opponents", {})
    return [
        _optional_count(value.value.get("wall"), 108.0),
        _optional_count(opponents.get("1"), 108.0),
        _optional_count(opponents.get("2"), 108.0),
        _optional_count(opponents.get("3"), 108.0),
        0.0,
        value.confidence,
    ]


def _player_void_suits(players: ObservedValue[list[dict[str, Any]]]) -> list[float]:
    by_relative = {}
    if players.status is not ObservationStatus.UNKNOWN and players.value is not None:
        by_relative = {player.get("relative_position"): player for player in players.value}

    encoded: list[float] = []
    for relative in range(4):
        void_suit = by_relative.get(relative, {}).get("void_suit")
        if not isinstance(void_suit, ObservedValue):
            void_suit = ObservedValue.unknown()
        encoded.extend(_categorical_observed(void_suit, ("W", "T", "B")))
    return encoded


def _players_by_relative(players: ObservedValue[list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    if players.status is ObservationStatus.UNKNOWN or players.value is None:
        return {}
    return {
        int(player["relative_position"]): player
        for player in players.value
        if player.get("relative_position") in (0, 1, 2, 3)
    }


def _player_field(
    players: ObservedValue[list[dict[str, Any]]], relative: int, field: str
) -> ObservedValue[Any]:
    value = _players_by_relative(players).get(relative, {}).get(field)
    return value if isinstance(value, ObservedValue) else ObservedValue.unknown()


def _tile_counts(value: ObservedValue[Any], tiles: list[str] | None = None) -> list[float]:
    if value.status is ObservationStatus.UNKNOWN or value.value is None:
        return [0.0] * _TILE_COUNT + [1.0, 0.0]
    counts = [0.0] * _TILE_COUNT
    for tile_text in tiles if tiles is not None else value.value:
        counts[parse_tile(tile_text).index] += 1.0
    return [_clamp(count / 4.0) for count in counts] + [0.0, value.confidence]


def _opponent_discard_counts(players: ObservedValue[list[dict[str, Any]]]) -> list[float]:
    encoded: list[float] = []
    for relative in (1, 2, 3):
        encoded.extend(_tile_counts(_player_field(players, relative, "rivers")))
    return encoded


def _opponent_discard_phases(players: ObservedValue[list[dict[str, Any]]]) -> list[float]:
    encoded: list[float] = []
    for relative in (1, 2, 3):
        rivers = _player_field(players, relative, "rivers")
        if rivers.status is ObservationStatus.UNKNOWN or rivers.value is None:
            encoded.extend([0.0] * (_TILE_COUNT * 3) + [1.0, 0.0])
            continue
        tiles = list(rivers.value)
        counts: list[float] = []
        for phase_tiles in (
            tiles[:_DISCARD_EARLY_END],
            tiles[_DISCARD_EARLY_END:_DISCARD_MIDDLE_END],
            tiles[_DISCARD_MIDDLE_END:],
        ):
            counts.extend(_tile_counts(rivers, phase_tiles)[:_TILE_COUNT])
        encoded.extend(counts + [0.0, rivers.confidence])
    return encoded


def _opponent_meld_tiles(players: ObservedValue[list[dict[str, Any]]]) -> list[float]:
    encoded: list[float] = []
    for relative in (1, 2, 3):
        melds = _player_field(players, relative, "melds")
        if melds.status is ObservationStatus.UNKNOWN or melds.value is None:
            encoded.extend([0.0] * _TILE_COUNT + [1.0, 0.0])
            continue
        tiles = [tile for meld in melds.value for tile in meld.get("tiles", [])]
        encoded.extend(_tile_counts(melds, tiles))
    return encoded


def _last_discards(players: ObservedValue[list[dict[str, Any]]]) -> list[float]:
    encoded: list[float] = []
    for relative in range(4):
        rivers = _player_field(players, relative, "rivers")
        one_hot = [0.0] * _TILE_COUNT
        if rivers.status is ObservationStatus.UNKNOWN or rivers.value is None:
            encoded.extend(one_hot + [0.0, 1.0, 0.0])
            continue
        if rivers.value:
            one_hot[parse_tile(rivers.value[-1]).index] = 1.0
        encoded.extend(one_hot + [float(not rivers.value), 0.0, rivers.confidence])
    return encoded


def _tile_location_beliefs(value: ObservedValue[dict[str, Any]]) -> list[float]:


    if value.status is ObservationStatus.UNKNOWN or value.value is None:
        return [0.25] * (_TILE_COUNT * len(_LOCATION_ORDER)) + [1.0, 0.0]
    encoded: list[float] = []
    for index in range(_TILE_COUNT):
        tile = tile_to_str(Tile.from_index(index))
        locations = value.value.get(tile, {})
        row = [float(locations.get(location, 0.0)) for location in _LOCATION_ORDER]
        total = sum(row)
        encoded.extend([item / total for item in row] if total > 0.0 else [0.25, 0.25, 0.25, 0.25])
    return encoded + [_unknown_flag(value), value.confidence]


def _opponent_tenpai_beliefs(value: ObservedValue[dict[str, Any]]) -> list[float]:
    if value.status is ObservationStatus.UNKNOWN or value.value is None:
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    return [_clamp(float(value.value.get(relative, 0.0))) for relative in ("1", "2", "3")] + [0.0, value.confidence]


def _discard_danger_summary(value: ObservedValue[dict[str, Any]]) -> list[float]:
    if value.status is ObservationStatus.UNKNOWN or value.value is None:
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    maxima = {relative: 0.0 for relative in ("1", "2", "3")}
    for per_opponent in value.value.values():
        for relative in maxima:
            maxima[relative] = max(maxima[relative], float(per_opponent.get(relative, 0.0)))
    return [maxima["1"], maxima["2"], maxima["3"], 0.0, value.confidence]


def _observation_summary(state: S2ProtocolState) -> list[float]:
    observed = estimated = unknown = total = 0
    for value in _observed_values(state):
        total += 1
        if value.status is ObservationStatus.OBSERVED:
            observed += 1
        elif value.status is ObservationStatus.ESTIMATED:
            estimated += 1
        else:
            unknown += 1
    start = 0.0 if state.observation_start.status is ObservationStatus.UNKNOWN else float(state.observation_start.value or 0.0)
    degraded = 1.0 if start > 0.0 or estimated > 0 or unknown > 0 else 0.0
    confidence = (observed + 0.5 * estimated) / total if total else 0.0
    return [_clamp(start / 108.0), observed / total, estimated / total, degraded, confidence]


def _rule_config_summary(value: ObservedValue[dict[str, Any]]) -> list[float]:
    if value.status is ObservationStatus.UNKNOWN or value.value is None:
        return [0.0, 0.0, 1.0, 0.0]
    numeric = [float(item) for item in value.value.values() if isinstance(item, (int, float, bool))]
    avg = sum(numeric) / len(numeric) if numeric else 0.0
    return [_clamp(avg / 16.0), _clamp(len(value.value) / 32.0), 0.0, value.confidence]


def _legal_action_summary(value: ObservedValue[list[dict[str, Any]]]) -> list[float]:
    if value.status is ObservationStatus.UNKNOWN or value.value is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    actions = value.value or []
    kinds = {"discard": 0, "pong": 0, "kong": 0, "hu": 0, "pass": 0}
    for action in actions:
        kind = action.get("kind")
        if kind in kinds:
            kinds[kind] += 1
    scale = max(1.0, float(len(actions)))
    return [kinds[kind] / scale for kind in ("discard", "pong", "kong", "hu", "pass")] + [_unknown_flag(value)]


def _observed_values(state: S2ProtocolState) -> list[ObservedValue[Any]]:
    values: list[ObservedValue[Any]] = [
        state.phase,
        state.current_player,
        state.current_player_relative,
        state.legal_actions,
        state.observation_start,
        state.rule_config,
    ]
    values.extend(getattr(state.facts, name) for name in state.facts.__dataclass_fields__)
    values.extend(getattr(state.statistics, name) for name in state.statistics.__dataclass_fields__)
    values.extend(getattr(state.beliefs, name) for name in state.beliefs.__dataclass_fields__)
    return values


def _unknown_flag(value: ObservedValue[Any]) -> float:
    return 1.0 if value.status is ObservationStatus.UNKNOWN else 0.0


def _optional_count(value: Any, scale: float) -> float:
    return 0.0 if value is None else _clamp(float(value) / scale)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))
