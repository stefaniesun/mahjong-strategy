import pytest

from engine.hand import Hand
from engine.state import GameState
from engine.tiles import Suit, full_wall, parse_tile


from state.adapters.from_engine import from_engine
from state.protocol import ObservationStatus, ObservedValue, S2ProtocolState
from state.tile_belief import LearnedBelief, PriorBelief, TileBelief, generate_belief_labels


def _tiles(texts):
    return [parse_tile(text) for text in texts]


def _base_engine_state() -> GameState:
    state = GameState(
        hands=[
            Hand.from_strings(["1W", "2W", "3W"]),
            Hand.from_strings(["4W", "4W"]),
            Hand.from_strings(["5W", "5W", "5W"]),
            Hand.from_strings(["6W", "7W"]),
        ],
        wall=_tiles(["9W", "9T", "9B", "8W"]),
        dealer=0,
        current_player=0,
        phase="play",
    )
    state.rivers = [[parse_tile("1T")], [], [], []]
    state.void_suits[1] = parse_tile("1W").suit
    return state


def _complete_with_visible_tiles(state: GameState) -> None:
    for tile in full_wall():
        physical_count = sum(hand.count(tile) for hand in state.hands)
        physical_count += sum(river.count(tile) for river in state.rivers)
        physical_count += sum(wall_tile == tile for wall_tile in state.wall)
        physical_count += sum(
            meld.tiles.count(tile)
            for hand in state.hands
            for meld in hand.melds
        )
        if physical_count < 4:
            state.rivers[0].extend([tile] * (4 - physical_count))



def test_prior_belief_fills_protocol_beliefs_with_normalized_location_probabilities():
    protocol_state = from_engine(_base_engine_state(), player_id=0)

    belief = PriorBelief().infer(protocol_state)

    assert isinstance(PriorBelief(), TileBelief)
    assert belief.source.value == "prior"
    assert belief.tile_location_beliefs.status is ObservationStatus.OBSERVED
    locations = belief.tile_location_beliefs.value["1W"]
    assert sum(locations.values()) == pytest.approx(1.0)
    assert locations["wall"] == pytest.approx(4 / 11)
    assert locations["1"] == pytest.approx(2 / 11)
    assert locations["2"] == pytest.approx(3 / 11)
    assert locations["3"] == pytest.approx(2 / 11)
    assert belief.opponent_tenpai_beliefs.value == {"1": 0.0, "2": 0.0, "3": 0.0}
    assert set(belief.discard_danger.value) == {"1W", "2W", "3W"}


def test_prior_belief_degrades_uniformly_when_pool_breakdown_unknown():
    protocol_state = from_engine(_base_engine_state(), player_id=0)
    degraded = S2ProtocolState(
        perspective_player=protocol_state.perspective_player,
        phase=protocol_state.phase,
        current_player=protocol_state.current_player,
        current_player_relative=protocol_state.current_player_relative,
        facts=protocol_state.facts,
        statistics=type(protocol_state.statistics)(
            **{**protocol_state.statistics.__dict__, "unknown_pool_breakdown": ObservedValue.unknown()}
        ),
        beliefs=protocol_state.beliefs,
        legal_actions=protocol_state.legal_actions,
        observation_start=protocol_state.observation_start,
        rule_config=protocol_state.rule_config,
    )

    belief = PriorBelief().infer(degraded)

    assert belief.tile_location_beliefs.value["1W"] == {
        "wall": pytest.approx(0.25),
        "1": pytest.approx(0.25),
        "2": pytest.approx(0.25),
        "3": pytest.approx(0.25),
    }


def test_learned_belief_infers_normalized_protocol_beliefs_from_model():
    torch = pytest.importorskip("torch")
    from learning.models.belief_net import BeliefNet, BeliefNetConfig, set_torch_seed
    from state.encoder import encode_state

    protocol_state = from_engine(_base_engine_state(), player_id=0)
    input_size = encode_state(protocol_state).size
    set_torch_seed(101)
    model = BeliefNet(BeliefNetConfig(input_size=input_size, hidden_size=32, residual_blocks=1))

    learned = LearnedBelief(model=model)
    first = learned.infer(protocol_state)
    second = learned.infer(protocol_state)

    assert first == second
    assert first.source.value == "learned"
    assert first.tile_location_beliefs.status is ObservationStatus.ESTIMATED
    assert set(first.tile_location_beliefs.value) == {f"{rank}{suit}" for suit in ("W", "T", "B") for rank in range(1, 10)}
    assert sum(first.tile_location_beliefs.value["1W"].values()) == pytest.approx(1.0)
    assert set(first.tile_location_beliefs.value["1W"]) == {"wall", "1", "2", "3"}
    assert set(first.opponent_tenpai_beliefs.value) == {"1", "2", "3"}
    assert all(0.0 <= value <= 1.0 for value in first.opponent_tenpai_beliefs.value.values())
    assert set(first.discard_danger.value) == {"1W", "2W", "3W"}
    assert torch.is_grad_enabled()


def test_learned_belief_loads_checkpoint_and_supports_batch_infer(tmp_path):
    torch = pytest.importorskip("torch")
    from learning.models.belief_net import BeliefNet, BeliefNetConfig, set_torch_seed
    from state.encoder import ENCODER_VERSION, encode_state


    first_state = from_engine(_base_engine_state(), player_id=0)
    second_state = from_engine(_base_engine_state(), player_id=1)
    config = BeliefNetConfig(input_size=encode_state(first_state).size, hidden_size=32, residual_blocks=1)
    set_torch_seed(102)
    model = BeliefNet(config)
    checkpoint = tmp_path / "belief.pt"
    torch.save(
        {
            "model_config": config.__dict__,
            "encoder_version": ENCODER_VERSION,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )

    learned = LearnedBelief(model_path=str(checkpoint))

    reports = learned.infer_batch([first_state, second_state])

    assert len(reports) == 2
    assert [report.source.value for report in reports] == ["learned", "learned"]
    assert reports[0] == learned.infer(first_state)


def test_learned_belief_rejects_checkpoint_without_current_encoder_version(tmp_path):
    torch = pytest.importorskip("torch")
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from state.encoder import encode_state

    state = from_engine(_base_engine_state(), player_id=0)
    config = BeliefNetConfig(input_size=encode_state(state).size, hidden_size=32, residual_blocks=1)
    model = BeliefNet(config)
    checkpoint = tmp_path / "legacy-belief.pt"
    torch.save({"model_config": config.__dict__, "state_dict": model.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="encoder version"):
        LearnedBelief(model_path=str(checkpoint))


def test_generate_belief_labels_uses_oracle_only_for_labels():

    engine_state = _base_engine_state()
    _complete_with_visible_tiles(engine_state)
    protocol_state = from_engine(engine_state, player_id=0)

    labels = generate_belief_labels(engine_state, protocol_state)

    tile_locations = labels["tile_locations"]
    assert tile_locations["counts"][parse_tile("9W").index] == [1, 0, 0, 0]
    assert tile_locations["counts"][parse_tile("4W").index] == [0, 2, 0, 0]
    assert tile_locations["counts"][parse_tile("5W").index] == [0, 0, 3, 0]
    assert tile_locations["counts"][parse_tile("6W").index] == [0, 0, 0, 1]
    assert tile_locations["distribution"][parse_tile("5W").index] == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert tile_locations["mask"][parse_tile("5W").index] is True
    assert tile_locations["mask"][parse_tile("1W").index] is False

    assert labels["opponent_tenpai"] == {"1": False, "2": False, "3": False}
    assert set(labels["discard_danger"]) == {"1W", "2W", "3W"}
    assert protocol_state.facts.players.value[1]["concealed_hand"].status is ObservationStatus.UNKNOWN


def test_opponent_tenpai_labels_require_clearing_the_void_suit():
    state = _base_engine_state()
    state.hands[1] = Hand.from_strings(
        ["1W", "1W", "1W", "2T", "3T", "4T", "2B", "3B", "4B", "5B", "6B", "7B", "9T"]
    )
    state.void_suits[1] = Suit.WAN
    _complete_with_visible_tiles(state)
    protocol_state = from_engine(state, player_id=0)

    blocked = generate_belief_labels(state, protocol_state)["opponent_tenpai"]
    cleared_state = _base_engine_state()
    cleared_state.hands[1] = Hand.from_strings(
        ["1T", "1T", "1T", "2T", "3T", "4T", "2B", "3B", "4B", "5B", "6B", "7B", "9T"]
    )
    _complete_with_visible_tiles(cleared_state)
    cleared = generate_belief_labels(
        cleared_state,
        from_engine(cleared_state, player_id=0),
    )["opponent_tenpai"]

    assert blocked["1"] is False
    assert cleared["1"] is True


def test_tile_location_labels_preserve_multiple_copies_and_conserve_four_tiles():

    state = _base_engine_state()
    state.wall = _tiles(["9W", "9W"])
    state.hands[1] = Hand.from_strings(["9W"])
    state.hands[2] = Hand.from_strings(["9W"])
    state.hands[3] = Hand()
    _complete_with_visible_tiles(state)
    protocol_state = from_engine(state, player_id=0)

    locations = generate_belief_labels(state, protocol_state)["tile_locations"]
    index = parse_tile("9W").index

    assert locations["counts"][index] == [2, 1, 1, 0]
    assert locations["distribution"][index] == pytest.approx([0.5, 0.25, 0.25, 0.0])
    assert locations["mask"][index] is True
    assert sum(locations["counts"][index]) == 4


def test_tile_location_labels_reject_more_than_four_physical_copies():
    state = _base_engine_state()
    state.wall = _tiles(["9W", "9W", "9W"])
    state.hands[1] = Hand.from_strings(["9W", "9W"])
    _complete_with_visible_tiles(state)
    protocol_state = from_engine(state, player_id=0)

    with pytest.raises(ValueError, match="9W"):
        generate_belief_labels(state, protocol_state)


def test_tile_location_labels_reject_fewer_than_four_physical_copies():
    wall = full_wall()
    wall.remove(parse_tile("9W"))
    state = GameState(
        hands=[Hand(), Hand(), Hand(), Hand()],
        wall=wall,
        dealer=0,
        current_player=0,
        phase="play",
    )
    protocol_state = from_engine(state, player_id=0)

    with pytest.raises(ValueError, match="9W"):
        generate_belief_labels(state, protocol_state)


