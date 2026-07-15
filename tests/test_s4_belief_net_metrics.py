import math

import pytest

from learning.training.metrics import (
    binary_brier_score,
    binary_ece,
    multiclass_brier_score,
    multiclass_log_loss,
    soft_multiclass_brier_score,
    soft_multiclass_log_loss,
)


def torch():
    return pytest.importorskip("torch")


def test_belief_net_forward_shapes_and_normalizes_tile_locations():
    torch_module = torch()
    from learning.models.belief_net import BeliefNet, BeliefNetConfig

    model = BeliefNet(BeliefNetConfig(input_size=243, hidden_size=32, residual_blocks=1))
    output = model(torch_module.zeros(2, 243))

    assert output.tile_location_probs.shape == (2, 27, 4)
    assert output.opponent_tenpai_probs.shape == (2, 3)
    assert output.discard_danger_probs.shape == (2, 27, 3)
    assert torch_module.allclose(output.tile_location_probs.sum(dim=-1), torch_module.ones(2, 27), atol=1e-6)
    assert torch_module.all((output.opponent_tenpai_probs >= 0.0) & (output.opponent_tenpai_probs <= 1.0))
    assert torch_module.all((output.discard_danger_probs >= 0.0) & (output.discard_danger_probs <= 1.0))


def test_belief_net_is_deterministic_with_same_seed():
    torch_module = torch()
    from learning.models.belief_net import BeliefNet, BeliefNetConfig, set_torch_seed

    set_torch_seed(13)
    first = BeliefNet(BeliefNetConfig(input_size=10, hidden_size=16, residual_blocks=1))
    set_torch_seed(13)
    second = BeliefNet(BeliefNetConfig(input_size=10, hidden_size=16, residual_blocks=1))
    features = torch_module.arange(20, dtype=torch_module.float32).reshape(2, 10) / 20.0

    assert torch_module.allclose(first(features).tile_location_probs, second(features).tile_location_probs)


def test_multiclass_metrics_match_manual_values():
    predictions = [
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
    ]
    targets = [0, 2]

    assert multiclass_log_loss(predictions, targets) == pytest.approx(-(math.log(0.7) + math.log(0.1)) / 2)
    expected_brier = ((0.7 - 1) ** 2 + 0.2**2 + 0.1**2 + 0.1**2 + 0.8**2 + (0.1 - 1) ** 2) / 2
    assert multiclass_brier_score(predictions, targets) == pytest.approx(expected_brier)


def test_soft_multiclass_metrics_match_distribution_targets():
    predictions = [[0.4, 0.3, 0.2, 0.1]]
    targets = [[0.25, 0.75, 0.0, 0.0]]

    assert soft_multiclass_log_loss(predictions, targets) == pytest.approx(
        -(0.25 * math.log(0.4) + 0.75 * math.log(0.3))
    )
    assert soft_multiclass_brier_score(predictions, targets) == pytest.approx(
        sum((prediction - target) ** 2 for prediction, target in zip(predictions[0], targets[0]))
    )


def test_binary_metrics_compute_brier_and_ece_bins():
    probabilities = [0.1, 0.4, 0.8, 0.9]
    targets = [0, 0, 1, 1]

    assert binary_brier_score(probabilities, targets) == pytest.approx((0.1**2 + 0.4**2 + 0.2**2 + 0.1**2) / 4)
    assert binary_ece(probabilities, targets, bins=2) == pytest.approx(0.2)
