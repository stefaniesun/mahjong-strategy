import math
from dataclasses import replace

import pytest

from learning.datasets.dataset_builder import DatasetBuildConfig, build_belief_sample
from selfplay.data_recorder import run_recorded_selfplay_game


def torch():
    return pytest.importorskip("torch")


def test_belief_training_step_updates_model_and_returns_loss_components():
    torch_module = torch()
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from learning.training.train_belief import belief_batch_from_samples, train_belief_step

    _, records = run_recorded_selfplay_game(game_id="train-belief", seed=21, max_steps=200)
    samples = [build_belief_sample(record, DatasetBuildConfig(seed=2, degradation_profile="perfect")) for record in records[:3]]
    batch = belief_batch_from_samples(samples)
    model = BeliefNet(BeliefNetConfig(input_size=len(samples[0].encoded.values), hidden_size=32, residual_blocks=1))
    optimizer = torch_module.optim.Adam(model.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_belief_step(model, batch, optimizer)

    assert metrics["loss"] > 0.0
    assert metrics["tile_location_loss"] > 0.0
    assert metrics["opponent_tenpai_loss"] >= 0.0
    assert any(not torch_module.allclose(old, new) for old, new in zip(before, model.parameters()))


def test_twenty_record_dataset_uses_upgraded_encoder_and_trains_one_epoch():
    torch()
    from learning.models.belief_net import BeliefNetConfig
    from learning.training.train_belief import TrainBeliefConfig, train_belief_epoch
    from state.encoder import ENCODER_VERSION

    _, records = run_recorded_selfplay_game(game_id="encoder-v3-smoke", seed=120, max_steps=200)
    assert len(records) >= 20
    samples = [
        build_belief_sample(record, DatasetBuildConfig(seed=5, degradation_profile="perfect"))
        for record in records[:20]
    ]

    model, metrics = train_belief_epoch(
        samples,
        TrainBeliefConfig(
            model=BeliefNetConfig(input_size=samples[0].encoded.size, hidden_size=16, residual_blocks=1),
            batch_size=10,
            device="cpu",
        ),
    )

    assert {sample.encoded.version for sample in samples} == {ENCODER_VERSION}
    assert {sample.encoded.size for sample in samples} == {893}
    assert model.config.input_size == 893
    assert metrics["samples"] == 20
    assert math.isfinite(metrics["loss"])


def test_train_belief_epoch_returns_average_metrics_for_batches():

    torch()
    from learning.models.belief_net import BeliefNetConfig
    from learning.training.train_belief import TrainBeliefConfig, train_belief_epoch

    _, records = run_recorded_selfplay_game(game_id="train-belief-epoch", seed=22, max_steps=200)
    samples = [build_belief_sample(record, DatasetBuildConfig(seed=3, degradation_profile="perfect")) for record in records[:4]]

    model, metrics = train_belief_epoch(
        samples,
        TrainBeliefConfig(
            model=BeliefNetConfig(input_size=len(samples[0].encoded.values), hidden_size=32, residual_blocks=1),
            batch_size=2,
            learning_rate=0.01,
            seed=9,
            device="cpu",
        ),
    )

    assert model.config.input_size == len(samples[0].encoded.values)
    assert next(model.parameters()).device.type == "cpu"
    assert metrics["batches"] == 2
    assert metrics["samples"] == 4
    assert metrics["device"] == "cpu"
    assert metrics["loss"] > 0.0


def test_belief_batch_uses_soft_tile_location_distributions():
    torch_module = torch()
    from learning.training.train_belief import belief_batch_from_samples

    samples = _belief_samples(2)
    batch = belief_batch_from_samples(samples)

    assert batch.tile_location_targets.shape == (2, 27, 4)
    assert batch.tile_location_targets.dtype == torch_module.float32
    assert batch.tile_location_mask.shape == (2, 27)
    expected = torch_module.tensor(samples[0].labels["tile_locations"]["distribution"])
    assert torch_module.allclose(batch.tile_location_targets[0], expected)


def test_masked_tile_location_loss_matches_soft_cross_entropy_and_backpropagates_empty_mask():
    torch_module = torch()
    from learning.training.train_belief import _masked_tile_location_loss

    logits = torch_module.tensor([[[math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)]]], requires_grad=True)
    targets = torch_module.tensor([[[0.25, 0.75, 0.0, 0.0]]])
    loss = _masked_tile_location_loss(logits, targets, torch_module.tensor([[True]]))
    assert loss.item() == pytest.approx(-(0.25 * math.log(0.4) + 0.75 * math.log(0.3)))

    empty_loss = _masked_tile_location_loss(logits, targets, torch_module.tensor([[False]]))
    assert empty_loss.item() == 0.0
    assert torch_module.isfinite(empty_loss)
    empty_loss.backward()
    assert logits.grad is not None


def test_belief_batch_rejects_legacy_tile_location_labels():
    from learning.training.train_belief import belief_batch_from_samples

    sample = _belief_samples(1)[0]
    legacy = replace(sample, labels={**sample.labels, "tile_locations": {"1W": "wall"}})

    with pytest.raises(ValueError, match="legacy|旧|incompatible"):
        belief_batch_from_samples([legacy])


def _belief_samples(count):
    _, records = run_recorded_selfplay_game(game_id="belief-soft-targets", seed=23, max_steps=200)
    return [
        build_belief_sample(record, DatasetBuildConfig(seed=4, degradation_profile="perfect"))
        for record in records[:count]
    ]


@pytest.mark.skipif(not torch().cuda.is_available(), reason="CUDA unavailable")
def test_train_belief_epoch_moves_model_to_cuda():
    from learning.models.belief_net import BeliefNetConfig
    from learning.training.train_belief import TrainBeliefConfig, train_belief_epoch

    samples = _belief_samples(4)
    model, metrics = train_belief_epoch(
        samples,
        TrainBeliefConfig(
            model=BeliefNetConfig(input_size=len(samples[0].encoded.values), hidden_size=16, residual_blocks=1),
            batch_size=2,
            device="cuda",
        ),
    )

    assert next(model.parameters()).device.type == "cuda"
    assert metrics["device"] == "cuda"



