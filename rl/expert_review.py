"""Derived, privacy-safe JSON material for the eight S5 expert behaviours.

The exporter accepts only retained learner-visible state and canonical action
indices.  It intentionally returns plain JSON-safe dictionaries: there is no
public review-record class, factory token, or caller-controlled record ID to
bypass the export boundary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from engine.tiles import Tile, parse_tile, tile_to_str
from rl.rollout import LearnerView
from rl.types import TrajectoryStep
from state.action_space import action_space_size, index_to_action
from state.protocol import ObservationStatus, ObservedValue


class ExpertBehaviorCategory(str, Enum):
    """The fixed eight human-reviewed behaviours from the S5 checklist."""

    DEFENSIVE_EXCHANGE_DIRECTION = "defensive_exchange_direction"
    DISADVANTAGE_OPENING_ESCAPE = "disadvantage_opening_escape"
    POST_KONG_DISCARD_SAFETY = "post_kong_discard_safety"
    SEVEN_PAIRS_VALUE = "seven_pairs_value"
    ENDGAME_DEFENSIVE_FOLD = "endgame_defensive_fold"
    EARLY_TENPAI_SELF_DRAW = "early_tenpai_self_draw"
    DEAD_TILE_INFERENCE = "dead_tile_inference"
    FAVORABLE_POSITION_BIG_HAND = "favorable_position_big_hand"


class ExpertReviewPrivacyError(ValueError):
    """A requested review export cannot be derived from safe learner data."""


@dataclass(frozen=True, slots=True)
class ExpertReviewSource:
    """A category-tagged learner decision with canonical S3/S4 action IDs only."""

    category: ExpertBehaviorCategory
    learner_view: LearnerView
    trajectory_step: TrajectoryStep
    s3_action_index: int
    s4_action_index: int

    def __post_init__(self) -> None:
        _validate_review_source(self)


def select_review_sources(
    sources: Sequence[ExpertReviewSource], category: ExpertBehaviorCategory, *, limit: int | None = None
) -> tuple[ExpertReviewSource, ...]:
    """Select explicit checklist categories without inspecting game state."""
    if not isinstance(category, ExpertBehaviorCategory):
        raise TypeError("category must be an ExpertBehaviorCategory")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise TypeError("sources must be a sequence of ExpertReviewSource values")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be a positive integer when provided")
    for source in sources:
        if not isinstance(source, ExpertReviewSource):
            raise TypeError("expert review sources must be ExpertReviewSource values")
        _validate_review_source(source)
    selected = tuple(source for source in sources if source.category is category)
    return selected if limit is None else selected[:limit]


def generate_expert_review_records(sources: Sequence[ExpertReviewSource]) -> tuple[dict[str, Any], ...]:
    """Return fresh, plain JSON-safe reviewer payloads derived from safe sources.

    IDs are assigned only by source order, so an upstream collector cannot use
    an arbitrary external identifier to smuggle review text or metadata.
    """
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise TypeError("sources must be a sequence of ExpertReviewSource values")
    records: list[dict[str, Any]] = []
    for ordinal, source in enumerate(sources, start=1):
        if not isinstance(source, ExpertReviewSource):
            raise TypeError("expert review sources must be ExpertReviewSource values")
        _validate_review_source(source)
        payload = {
            "record_id": f"review-{ordinal:06d}",
            "category": source.category.value,
            "public_observation": _derive_public_observation(source.learner_view),
            "policy_action": _canonical_action(source.trajectory_step.action),
            "s3_action": _canonical_action(source.s3_action_index),
            "s4_action": _canonical_action(source.s4_action_index),
            "belief_summary": _derive_belief_summary(source.learner_view),
        }
        records.append(_json_safe_copy(payload))
    return tuple(records)


def _validate_review_source(source: ExpertReviewSource) -> None:
    if not isinstance(source.category, ExpertBehaviorCategory):
        raise TypeError("category must be an ExpertBehaviorCategory")
    if not isinstance(source.learner_view, LearnerView):
        raise TypeError("expert review requires a learner-visible LearnerView")
    if not isinstance(source.trajectory_step, TrajectoryStep):
        raise TypeError("expert review requires a retained TrajectoryStep")
    if tuple(source.learner_view.legal_mask) != tuple(source.trajectory_step.legal_mask):
        raise ValueError("retained trajectory legal mask does not match the learner view")
    _validate_action_index(source.trajectory_step.action, "retained trajectory action")
    if not source.learner_view.legal_mask[source.trajectory_step.action]:
        raise ValueError("retained trajectory action is outside the learner legal mask")
    _validate_action_index(source.s3_action_index, "s3_action_index")
    _validate_action_index(source.s4_action_index, "s4_action_index")
    _validate_derived_inputs(source.learner_view)


def _validate_action_index(index: object, field_name: str) -> None:
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"{field_name} must be an integer action-space index")
    if not 0 <= index < action_space_size():
        raise ValueError(f"{field_name} is outside the canonical action space")


def _validate_derived_inputs(view: LearnerView) -> None:
    phase = view.observation.phase
    if not _is_observed(phase) or phase.value not in {"swap_three", "declare_void", "play", "finished"}:
        raise ExpertReviewPrivacyError("expert export requires an observed safe phase")
    players = view.observation.facts.players
    if not _is_observed(players) or not isinstance(players.value, list):
        raise ExpertReviewPrivacyError("expert export requires observed public players")
    for player in players.value:
        if not isinstance(player, dict):
            raise ExpertReviewPrivacyError("public player entries must be dictionaries")
        rivers = player.get("rivers")
        if not _is_observed(rivers) or not isinstance(rivers.value, list):
            raise ExpertReviewPrivacyError("expert export requires observed public rivers")
        for tile in rivers.value:
            _canonical_tile(tile)

    beliefs = view.observation.beliefs
    if not _is_observed(beliefs.source) or beliefs.source.value != "learned":
        raise ExpertReviewPrivacyError("expert export requires frozen learned Belief values")
    _learned_tile_locations(beliefs.tile_location_beliefs)
    _learned_tenpai(beliefs.opponent_tenpai_beliefs)
    _learned_danger(beliefs.discard_danger)


def _is_observed(value: object) -> bool:
    return isinstance(value, ObservedValue) and value.status is ObservationStatus.OBSERVED and value.confidence == 1.0


def _is_learned_estimate(value: object, field_name: str) -> bool:
    if not isinstance(value, ObservedValue) or value.status is not ObservationStatus.ESTIMATED:
        raise ExpertReviewPrivacyError(f"{field_name} must be an estimated frozen learned output")
    if not isinstance(value.confidence, (int, float)) or isinstance(value.confidence, bool) or not math.isfinite(value.confidence):
        raise ExpertReviewPrivacyError(f"{field_name} confidence must be finite")
    return True


def _derive_public_observation(view: LearnerView) -> dict[str, Any]:
    visible_discards: list[str] = []
    for player in view.observation.facts.players.value:
        visible_discards.extend(_canonical_tile(tile) for tile in player["rivers"].value)
    return {"phase": view.observation.phase.value, "visible_discards": visible_discards}


def _derive_belief_summary(view: LearnerView) -> dict[str, Any]:
    tenpai = _learned_tenpai(view.observation.beliefs.opponent_tenpai_beliefs)
    danger = _learned_danger(view.observation.beliefs.discard_danger)
    top_tile, top_danger = max(danger.items(), key=lambda item: (item[1], item[0]))
    return {
        "max_tenpai_probability": max(tenpai.values()),
        "highest_danger_tile": top_tile,
        "highest_danger": top_danger,
    }


def _learned_tile_locations(observed: object) -> dict[str, dict[str, float]]:
    _is_learned_estimate(observed, "tile_location_beliefs")
    value = observed.value
    expected_tiles = {tile_to_str(Tile.from_index(index)) for index in range(27)}
    if not isinstance(value, dict) or set(value) != expected_tiles:
        raise ExpertReviewPrivacyError("tile_location_beliefs must have the canonical learned tile schema")
    result: dict[str, dict[str, float]] = {}
    for tile, locations in value.items():
        _canonical_tile(tile)
        if not isinstance(locations, dict) or set(locations) != {"wall", "1", "2", "3"}:
            raise ExpertReviewPrivacyError("tile location values must be learned wall/player probabilities")
        canonical = {location: _probability(probability, "tile_location_beliefs") for location, probability in locations.items()}
        if not math.isclose(sum(canonical.values()), 1.0, abs_tol=1e-5):
            raise ExpertReviewPrivacyError("tile location probabilities must sum to one")
        result[tile] = canonical
    return result


def _learned_tenpai(observed: object) -> dict[str, float]:
    _is_learned_estimate(observed, "opponent_tenpai_beliefs")
    value = observed.value
    if not isinstance(value, dict) or set(value) != {"1", "2", "3"}:
        raise ExpertReviewPrivacyError("opponent_tenpai_beliefs must have three relative opponents")
    return {relative: _probability(probability, "opponent_tenpai_beliefs") for relative, probability in value.items()}


def _learned_danger(observed: object) -> dict[str, float]:
    _is_learned_estimate(observed, "discard_danger")
    value = observed.value
    if not isinstance(value, dict) or not value:
        raise ExpertReviewPrivacyError("discard_danger must contain learned tile probabilities")
    result: dict[str, float] = {}
    for tile, per_player in value.items():
        canonical_tile = _canonical_tile(tile)
        if not isinstance(per_player, dict) or set(per_player) != {"1", "2", "3"}:
            raise ExpertReviewPrivacyError("discard danger entries must cover the three relative opponents")
        result[canonical_tile] = max(_probability(probability, "discard_danger") for probability in per_player.values())
    return result


def _probability(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ExpertReviewPrivacyError(f"{field_name} must contain finite probabilities in [0, 1]")
    return float(value)


def _canonical_tile(value: object) -> str:
    if not isinstance(value, str):
        raise ExpertReviewPrivacyError("public tiles must be canonical tile strings")
    try:
        canonical = tile_to_str(parse_tile(value))
    except ValueError as exc:
        raise ExpertReviewPrivacyError("public tiles must be canonical tile strings") from exc
    if value != canonical:
        raise ExpertReviewPrivacyError("public tiles must be canonical tile strings")
    return canonical


def _canonical_action(index: int) -> dict[str, Any]:
    _validate_action_index(index, "action index")
    return {"index": index, "action": index_to_action(index)}


def _json_safe_copy(payload: dict[str, Any]) -> dict[str, Any]:
    """Detach export data and prove it contains only JSON values."""
    try:
        copied = json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExpertReviewPrivacyError("expert review payload is not JSON-safe") from exc
    if not isinstance(copied, dict):  # Defensive guard for the return annotation.
        raise ExpertReviewPrivacyError("expert review payload is invalid")
    return copied
