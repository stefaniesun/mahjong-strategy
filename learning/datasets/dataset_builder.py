from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any


from state.action_space import action_space_size, action_to_index
from state.encoder import EncodedState, encode_state
from state.observation_degradation import DegradationPipeline, MaskExchange, MaskField, MidGameSnapshot, VisionNoise
from state.protocol import Beliefs, ObservedValue, S2ProtocolState
from state.tile_belief import TileBelief, with_prior_beliefs


from selfplay.data_recorder import DecisionRecord, load_jsonl
from learning.datasets.splits import SplitRecords, split_records_by_game


@dataclass(frozen=True)
class DatasetBuildConfig:
    seed: int = 0
    degradation_profile: str = "perfect"


@dataclass(frozen=True)
class BeliefSample:
    game_id: str
    step: int
    player: int
    phase: str
    encoded: EncodedState
    labels: dict[str, Any]
    degradation_profile: str


@dataclass(frozen=True)
class PolicySample:
    game_id: str
    step: int
    player: int
    phase: str
    encoded: EncodedState
    action_index: int
    legal_mask: tuple[bool, ...]
    degradation_profile: str
    action_kind: str
    legal_action_count: int
    is_pong_pass_decision: bool



def load_decision_records(
    path: str | Path,
    *,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
) -> SplitRecords:
    return split_records_by_game(load_jsonl(path), seed=seed, ratios=ratios)


def build_belief_sample(record: DecisionRecord, config: DatasetBuildConfig | None = None) -> BeliefSample:
    cfg = config or DatasetBuildConfig()
    state = _record_state(record)
    degraded = _degrade(state, cfg, record)
    belief_input = _without_beliefs(degraded)
    return BeliefSample(
        game_id=record.game_id,
        step=record.step,
        player=record.player,
        phase=record.phase,
        encoded=encode_state(belief_input),
        labels=dict(record.labels),
        degradation_profile=cfg.degradation_profile,
    )


def build_policy_sample(
    record: DecisionRecord,
    config: DatasetBuildConfig | None = None,
    *,
    belief: TileBelief | None = None,
) -> PolicySample:
    cfg = config or DatasetBuildConfig()
    state = _record_state(record)
    degraded = _degrade(state, cfg, record)
    with_beliefs = with_prior_beliefs(degraded, belief)
    action = _clean_action(record.action)
    legal_actions = [_clean_action(item) for item in record.legal_actions]
    action_index = action_to_index(action)
    mask = _mask_from_actions(record.legal_actions)
    if not mask[action_index]:
        raise ValueError("record action is not included in legal actions")
    legal_kinds = {item["kind"] for item in legal_actions}
    return PolicySample(
        game_id=record.game_id,
        step=record.step,
        player=record.player,
        phase=record.phase,
        encoded=encode_state(with_beliefs),
        action_index=action_index,
        legal_mask=mask,
        degradation_profile=cfg.degradation_profile,
        action_kind=str(action["kind"]),
        legal_action_count=sum(mask),
        is_pong_pass_decision={"pong", "pass"}.issubset(legal_kinds),
    )



def _record_state(record: DecisionRecord) -> S2ProtocolState:
    state = S2ProtocolState.from_dict(record.state)
    return _with_record_legal_actions(state, record)


def _degrade(state: S2ProtocolState, config: DatasetBuildConfig, record: DecisionRecord) -> S2ProtocolState:
    profile = config.degradation_profile
    if profile == "perfect":
        return state

    pipeline = _pipeline_for_profile(profile, config.seed + record.step * 997 + record.player * 131)
    return _with_record_legal_actions(pipeline.apply(state), record)


def _pipeline_for_profile(profile: str, seed: int) -> DegradationPipeline:
    rng = random.Random(seed)
    if profile == "light_noise":
        return DegradationPipeline([VisionNoise(miss_rate=0.03, seed=rng.randrange(1_000_000))])
    if profile == "midgame":
        return DegradationPipeline([MidGameSnapshot(k=rng.randrange(0, 18))])
    if profile == "heavy":
        return DegradationPipeline(
            [
                MidGameSnapshot(k=rng.randrange(6, 36)),
                VisionNoise(miss_rate=0.10, seed=rng.randrange(1_000_000)),
                MaskExchange(p=0.5, seed=rng.randrange(1_000_000)),
                MaskField("wall_count", p=0.25, seed=rng.randrange(1_000_000)),
            ]
        )
    raise ValueError(f"unsupported degradation profile: {profile}")


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


def _with_record_legal_actions(state: S2ProtocolState, record: DecisionRecord) -> S2ProtocolState:
    return S2ProtocolState(
        perspective_player=state.perspective_player,
        phase=state.phase,
        current_player=state.current_player,
        current_player_relative=state.current_player_relative,
        facts=state.facts,
        statistics=state.statistics,
        beliefs=state.beliefs,
        legal_actions=ObservedValue.observed([_clean_action(action) for action in record.legal_actions]),
        observation_start=state.observation_start,
        rule_config=state.rule_config,
    )


def _mask_from_actions(actions: list[dict[str, Any]]) -> tuple[bool, ...]:
    mask = [False] * action_space_size()
    for action in actions:
        mask[action_to_index(_clean_action(action))] = True
    return tuple(mask)


def _clean_action(action: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        key: value
        for key, value in action.items()
        if value is not None and value != [] and key not in {"conditionally_legal", "depends_on"}
    }
    if "tile" in cleaned:
        cleaned["tile"] = _normalize_tile_text(str(cleaned["tile"]))
    if "tiles" in cleaned:
        cleaned["tiles"] = [_normalize_tile_text(str(tile)) for tile in cleaned["tiles"]]
    return cleaned


_TILE_REPR_PATTERN = re.compile(r"Tile\(suit=<Suit\.[A-Z]+: '([WTB])'>, rank=([1-9])\)")


def _normalize_tile_text(text: str) -> str:
    match = _TILE_REPR_PATTERN.fullmatch(text)
    if match is not None:
        suit, rank = match.groups()
        return f"{rank}{suit}"
    return text

