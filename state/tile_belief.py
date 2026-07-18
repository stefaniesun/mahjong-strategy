from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from engine.hand import Hand
from engine.state import GameState
from engine.tiles import Tile, parse_tile, tile_to_str
from engine.ting_check import ting_tiles
from engine.win_check import can_win
from state.protocol import Beliefs, ObservationStatus, ObservedValue, S2ProtocolState

from state.tile_counting import hypergeometric_location_prior



class TileBelief(ABC):
    @abstractmethod
    def infer(self, state: S2ProtocolState) -> Beliefs:
        raise NotImplementedError


@dataclass(frozen=True)
class PriorBelief(TileBelief):
    source: str = "prior"

    def infer(self, state: S2ProtocolState) -> Beliefs:
        tile_locations = hypergeometric_location_prior(
            state.statistics.remaining_tile_counts.value or [4.0] * 27,
            state.statistics.unknown_pool_breakdown,
        )
        tile_locations = {tile: _normalized_locations(locations) for tile, locations in tile_locations.items()}
        return Beliefs(
            source=ObservedValue.observed(self.source),
            tile_location_beliefs=ObservedValue.observed(tile_locations),
            opponent_tenpai_beliefs=ObservedValue.observed(_default_opponent_tenpai(state)),
            discard_danger=ObservedValue.observed(_default_discard_danger(state)),
        )


@dataclass(frozen=True)
class LearnedBelief(TileBelief):
    model_path: str | None = None
    model: Any | None = None
    source: str = "learned"
    confidence: float = 0.8

    def __post_init__(self) -> None:
        if self.model is None and self.model_path is not None:
            object.__setattr__(self, "model", self._load_model(self.model_path))
        if self.model is None:
            raise ValueError("LearnedBelief requires a model or model_path")
        self.model.eval()

    def infer(self, state: S2ProtocolState) -> Beliefs:
        return self.infer_batch([state])[0]

    def infer_batch(self, states: list[S2ProtocolState]) -> list[Beliefs]:
        if not states:
            return []
        import torch
        from state.encoder import encode_state

        inputs = [_without_beliefs(state) for state in states]
        device = next(self.model.parameters()).device
        features = torch.tensor([encode_state(state).values for state in inputs], dtype=torch.float32, device=device)
        with torch.no_grad():
            output = self.model(features)

            tile_probs = output.tile_location_probs.detach().cpu().tolist()
            tenpai_probs = output.opponent_tenpai_probs.detach().cpu().tolist()
            danger_probs = output.discard_danger_probs.detach().cpu().tolist()
        return [self._belief_from_prediction(state, tile_probs[row], tenpai_probs[row], danger_probs[row]) for row, state in enumerate(states)]

    def _belief_from_prediction(
        self,
        state: S2ProtocolState,
        tile_probs: list[list[float]],
        tenpai_probs: list[float],
        danger_probs: list[list[float]],
    ) -> Beliefs:
        tile_locations = {
            tile_to_str(Tile.from_index(index)): _normalized_locations(
                {location: float(tile_probs[index][offset]) for offset, location in enumerate(("wall", "1", "2", "3"))}
            )
            for index in range(27)
        }
        tenpai = {relative: _clamp_probability(tenpai_probs[index]) for index, relative in enumerate(("1", "2", "3"))}
        own_tiles = _own_visible_tiles(state)
        danger = {
            tile: {
                relative: _clamp_probability(danger_probs[_tile_index(tile)][index])
                for index, relative in enumerate(("1", "2", "3"))
            }
            for tile in own_tiles
        }
        return Beliefs(
            source=ObservedValue.observed(self.source),
            tile_location_beliefs=ObservedValue.estimated(tile_locations, self.confidence),
            opponent_tenpai_beliefs=ObservedValue.estimated(tenpai, self.confidence),
            discard_danger=ObservedValue.estimated(danger, self.confidence),
        )

    @staticmethod
    def _load_model(model_path: str):
        import torch
        from learning.models.belief_net import BeliefNet, BeliefNetConfig

        from state.encoder import ENCODER_VERSION, encoding_size

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("belief checkpoint must be a mapping")
        checkpoint_version = checkpoint.get("encoder_version")

        if checkpoint_version != ENCODER_VERSION:
            raise ValueError(
                f"checkpoint encoder version {checkpoint_version!r} is incompatible with {ENCODER_VERSION!r}"
            )
        config_data = checkpoint.get("model_config") or checkpoint.get("config")

        if config_data is None:
            raise ValueError("checkpoint must include model_config")
        if not isinstance(config_data, dict):
            raise ValueError("belief checkpoint model_config must be a mapping")
        checkpoint_input_size = config_data.get("input_size")
        current_input_size = encoding_size()
        if checkpoint_input_size != current_input_size:
            raise ValueError(
                f"belief checkpoint input_size={checkpoint_input_size!r} does not "
                f"match current encoder input_size={current_input_size}"
            )
        model = BeliefNet(BeliefNetConfig(**config_data))
        model.load_state_dict(checkpoint.get("state_dict") or checkpoint.get("model_state_dict"))
        return model



def generate_belief_labels(engine_state: GameState, protocol_state: S2ProtocolState) -> dict[str, Any]:
    return {
        "tile_locations": _oracle_tile_locations(engine_state, protocol_state),
        "opponent_tenpai": _oracle_opponent_tenpai(engine_state, protocol_state),
        "discard_danger": _oracle_discard_danger(engine_state, protocol_state),
    }


def with_prior_beliefs(state: S2ProtocolState, belief: TileBelief | None = None) -> S2ProtocolState:
    beliefs = (belief or PriorBelief()).infer(state)
    return S2ProtocolState(
        perspective_player=state.perspective_player,
        phase=state.phase,
        current_player=state.current_player,
        current_player_relative=state.current_player_relative,
        facts=state.facts,
        statistics=state.statistics,
        beliefs=beliefs,
        legal_actions=state.legal_actions,
        observation_start=state.observation_start,
        rule_config=state.rule_config,
    )


def _without_beliefs(state: S2ProtocolState) -> S2ProtocolState:
    return S2ProtocolState(
        perspective_player=state.perspective_player,
        phase=state.phase,
        current_player=state.current_player,
        current_player_relative=state.current_player_relative,
        facts=state.facts,
        statistics=state.statistics,
        beliefs=Beliefs(
            source=ObservedValue.unknown(),
            tile_location_beliefs=ObservedValue.unknown(),
            opponent_tenpai_beliefs=ObservedValue.unknown(),
            discard_danger=ObservedValue.unknown(),
        ),
        legal_actions=state.legal_actions,
        observation_start=state.observation_start,
        rule_config=state.rule_config,
    )


_NORMALIZED_CACHE: dict[tuple, dict[str, float]] = {}


def _normalized_locations(locations: dict[str, float]) -> dict[str, float]:
    # 同一决策点 27 种牌的分布完全相同,逐张重复归一化是热点;按内容缓存,返回副本防别名。
    key = tuple(locations.items())
    cached = _NORMALIZED_CACHE.get(key)
    if cached is None:
        total = sum(float(value) for value in locations.values())
        if total <= 0.0:
            cached = {location: 1.0 / len(locations) for location in locations}
        else:
            cached = {location: float(value) / total for location, value in locations.items()}
        if len(_NORMALIZED_CACHE) < 100000:
            _NORMALIZED_CACHE[key] = cached
    return dict(cached)


def _clamp_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))



def _opponent_relatives(state: S2ProtocolState) -> list[str]:
    if state.facts.players.status is not ObservationStatus.OBSERVED:
        return ["1", "2", "3"]
    relatives: list[str] = []
    for player in state.facts.players.value or []:
        relative = player.get("relative_position")
        if relative in (1, 2, 3):
            relatives.append(str(relative))
    return relatives or ["1", "2", "3"]


def _default_opponent_tenpai(state: S2ProtocolState) -> dict[str, float]:
    return {relative: 0.0 for relative in _opponent_relatives(state)}


def _default_discard_danger(state: S2ProtocolState) -> dict[str, dict[str, float]]:
    own_tiles = _own_visible_tiles(state)
    return {tile: {relative: 0.0 for relative in _opponent_relatives(state)} for tile in own_tiles}


def _own_visible_tiles(state: S2ProtocolState) -> list[str]:
    if state.facts.players.status is not ObservationStatus.OBSERVED:
        return []
    for player in state.facts.players.value or []:
        if player.get("relative_position") != 0:
            continue
        concealed = player.get("concealed_hand")
        if isinstance(concealed, ObservedValue) and concealed.status is ObservationStatus.OBSERVED:
            return sorted(set(concealed.value or []), key=lambda text: _tile_index(text))
    return []


def _oracle_tile_locations(engine_state: GameState, protocol_state: S2ProtocolState) -> dict[str, Any]:
    counts = [[0, 0, 0, 0] for _ in range(27)]
    for tile in engine_state.wall:
        counts[tile.index][0] += 1
    for player in protocol_state.facts.players.value or []:
        relative = player.get("relative_position")
        if relative not in (1, 2, 3):
            continue
        player_id = player.get("player_id")
        if engine_state.won[player_id]:
            continue
        for tile in engine_state.hands[player_id].tiles():
            counts[tile.index][relative] += 1
        if engine_state.phase == "swap_three":
            for tile in engine_state.swap_choices[player_id] or ():
                counts[tile.index][relative] += 1

    from state.adapters.from_engine import from_engine

    oracle_state = from_engine(
        engine_state,
        player_id=protocol_state.perspective_player,
    )

    seen = oracle_state.facts.seen_counts.value or [0] * 27
    distribution: list[list[float]] = []

    mask: list[bool] = []
    for index, location_counts in enumerate(counts):
        hidden = sum(location_counts)
        visible = int(seen[index])
        tile_text = tile_to_str(Tile.from_index(index))
        if visible + hidden != 4:
            raise ValueError(
                f"tile copy count must equal four for {tile_text}: "
                f"visible={visible}, hidden={hidden}"
            )

        mask.append(hidden > 0)
        distribution.append([count / hidden for count in location_counts] if hidden else [0.0, 0.0, 0.0, 0.0])
    return {"counts": counts, "distribution": distribution, "mask": mask}



def _oracle_opponent_tenpai(engine_state: GameState, protocol_state: S2ProtocolState) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    for player in protocol_state.facts.players.value or []:
        relative = player.get("relative_position")
        if relative == 0:
            continue
        player_id = player.get("player_id")
        if engine_state.won[player_id]:
            labels[str(relative)] = False
            continue
        labels[str(relative)] = bool(
            ting_tiles(engine_state.hands[player_id], engine_state.void_suits[player_id])
        )

    return labels


def _oracle_discard_danger(engine_state: GameState, protocol_state: S2ProtocolState) -> dict[str, dict[str, bool]]:
    danger: dict[str, dict[str, bool]] = {}
    for tile_text in _own_visible_tiles(protocol_state):
        tile = _tile_from_text(tile_text)
        per_opponent: dict[str, bool] = {}
        for player in protocol_state.facts.players.value or []:
            relative = player.get("relative_position")
            if relative == 0:
                continue
            player_id = player.get("player_id")
            trial = Hand(counts=list(engine_state.hands[player_id].counts), melds=list(engine_state.hands[player_id].melds))
            if trial.count(tile) >= 4:
                per_opponent[str(relative)] = False
                continue
            trial.add(tile)
            per_opponent[str(relative)] = can_win(trial, engine_state.void_suits[player_id])
        danger[tile_text] = per_opponent
    return danger


def _tile_from_text(tile_text: str) -> Tile:
    return parse_tile(tile_text)


def _tile_index(tile_text: str) -> int:
    return _tile_from_text(tile_text).index
