from __future__ import annotations

from dataclasses import replace
import json

import pytest

from engine.game import Game
from engine.tiles import Tile, tile_to_str
from rl.types import TrajectoryStep
from state.action_space import action_space_size, index_to_action
from state.adapters.from_engine import from_engine
from state.protocol import Beliefs, ObservedValue


def _view_and_step():
    from rl.rollout import LearnerView

    observation = from_engine(Game(seed=7).reset(), 0)
    # Expert export may only use frozen learned Belief values.  The fixture uses
    # a small, explicit learned output rather than caller-provided review text.
    beliefs = Beliefs(
        source=ObservedValue.observed("learned"),
        tile_location_beliefs=ObservedValue.estimated(
            {
                tile_to_str(Tile.from_index(index)): {"wall": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}
                for index in range(27)
            },
            confidence=0.8,
        ),
        opponent_tenpai_beliefs=ObservedValue.estimated({"1": 0.25, "2": 0.5, "3": 0.75}, confidence=0.8),
        discard_danger=ObservedValue.estimated({"1W": {"1": 0.2, "2": 0.8, "3": 0.4}}, confidence=0.8),
    )
    observation = replace(observation, beliefs=beliefs)
    mask = (True,) + (False,) * (action_space_size() - 1)
    view = LearnerView(observation=observation, feature_values=(0.25, 0.75), legal_mask=mask, decision_seed=123456789)
    step = TrajectoryStep(
        feature_values=view.feature_values,
        legal_mask=view.legal_mask,
        action=0,
        old_log_prob=-0.2,
        value=0.1,
        reward=0.0,
        done=False,
        policy_version="s5-test",
    )
    return view, step


def _valid_source(category):
    from rl.expert_review import ExpertReviewSource

    view, step = _view_and_step()
    return ExpertReviewSource(
        category=category,
        learner_view=view,
        trajectory_step=step,
        s3_action_index=1,
        s4_action_index=2,
    )


def test_generates_derived_json_safe_material_for_all_eight_expert_categories() -> None:
    from rl.expert_review import ExpertBehaviorCategory, generate_expert_review_records

    records = generate_expert_review_records([_valid_source(category) for category in ExpertBehaviorCategory])

    assert {record["category"] for record in records} == {category.value for category in ExpertBehaviorCategory}
    assert len(records) == 8
    for index, payload in enumerate(records, start=1):
        assert set(payload) == {"record_id", "category", "public_observation", "policy_action", "s3_action", "s4_action", "belief_summary"}
        assert payload["record_id"] == f"review-{index:06d}"
        assert payload["policy_action"] == {"index": 0, "action": index_to_action(0)}
        assert payload["s3_action"] == {"index": 1, "action": index_to_action(1)}
        assert payload["s4_action"] == {"index": 2, "action": index_to_action(2)}
        assert payload["belief_summary"] == {"max_tenpai_probability": 0.75, "highest_danger_tile": "1W", "highest_danger": 0.8}
        serialized = json.dumps(payload, sort_keys=True)
        for forbidden in ("concealed_hand", "hidden", "game_state", "true_label", "decision_seed", "123456789"):
            assert forbidden not in serialized.lower()


@pytest.mark.parametrize("private_text", ("hidden hand", "Opponent__Hand", "Player_1_Hand", "ground_truth", "TRUE__STATE"))
def test_source_has_no_field_that_can_carry_disguised_private_text(private_text: str) -> None:
    from rl.expert_review import ExpertBehaviorCategory, ExpertReviewSource

    view, step = _view_and_step()
    with pytest.raises(TypeError):
        ExpertReviewSource(
            record_id=private_text,
            category=ExpertBehaviorCategory.DEAD_TILE_INFERENCE,
            learner_view=view,
            trajectory_step=step,
            s3_action_index=1,
            s4_action_index=2,
        )


def test_exporter_exposes_no_public_record_or_factory_token() -> None:
    import rl.expert_review as expert_review

    assert not hasattr(expert_review, "ExpertReviewRecord")
    assert not hasattr(expert_review, "_EXPORT_TOKEN")


def test_observation_and_actions_are_derived_from_view_and_canonical_indices() -> None:
    from rl.expert_review import ExpertBehaviorCategory, generate_expert_review_records

    payload = generate_expert_review_records([_valid_source(ExpertBehaviorCategory.DEAD_TILE_INFERENCE)])[0]
    assert payload["public_observation"] == {"phase": "swap_three", "visible_discards": []}
    assert payload["policy_action"]["action"] == index_to_action(0)
    assert payload["s3_action"]["action"] == index_to_action(1)
    assert payload["s4_action"]["action"] == index_to_action(2)


@pytest.mark.parametrize("field,value", (("s3_action_index", -1), ("s4_action_index", action_space_size())))
def test_source_rejects_out_of_range_action_indices(field: str, value: int) -> None:
    from rl.expert_review import ExpertBehaviorCategory, ExpertReviewSource

    view, step = _view_and_step()
    kwargs = dict(category=ExpertBehaviorCategory.DEAD_TILE_INFERENCE, learner_view=view, trajectory_step=step, s3_action_index=1, s4_action_index=2)
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError)):
        ExpertReviewSource(**kwargs)


def test_select_review_sources_preserves_category_selection() -> None:
    from rl.expert_review import ExpertBehaviorCategory, select_review_sources

    sources = [_valid_source(category) for category in ExpertBehaviorCategory]
    selected = select_review_sources(sources, ExpertBehaviorCategory.DEAD_TILE_INFERENCE)
    assert selected == (_valid_source(ExpertBehaviorCategory.DEAD_TILE_INFERENCE),)


def test_source_has_no_external_record_id_parameter() -> None:
    from rl.expert_review import ExpertBehaviorCategory, ExpertReviewSource

    view, step = _view_and_step()
    with pytest.raises(TypeError, match="record_id"):
        ExpertReviewSource(
            record_id="caller-controlled",
            category=ExpertBehaviorCategory.DEAD_TILE_INFERENCE,
            learner_view=view,
            trajectory_step=step,
            s3_action_index=1,
            s4_action_index=2,
        )


@pytest.mark.parametrize("mutation", ("unlearned", "unknown", "oracle_shape"))
def test_export_rejects_noncanonical_or_unlearned_belief_outputs(mutation: str) -> None:
    from rl.expert_review import ExpertBehaviorCategory, ExpertReviewPrivacyError, ExpertReviewSource, generate_expert_review_records

    view, step = _view_and_step()
    beliefs = view.observation.beliefs
    if mutation == "unlearned":
        beliefs = replace(beliefs, source=ObservedValue.observed("prior"))
    elif mutation == "unknown":
        beliefs = replace(beliefs, opponent_tenpai_beliefs=ObservedValue.unknown())
    else:
        beliefs = replace(
            beliefs,
            tile_location_beliefs=ObservedValue.estimated({"1W": {"opponent_hand": 1.0}}, confidence=0.8),
        )
    view = replace(view, observation=replace(view.observation, beliefs=beliefs))
    with pytest.raises(ExpertReviewPrivacyError):
        generate_expert_review_records(
            [
                ExpertReviewSource(
                    category=ExpertBehaviorCategory.DEAD_TILE_INFERENCE,
                    learner_view=view,
                    trajectory_step=step,
                    s3_action_index=1,
                    s4_action_index=2,
                )
            ]
        )
