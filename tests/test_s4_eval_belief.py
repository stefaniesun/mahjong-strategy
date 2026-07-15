import math
from dataclasses import replace

import pytest

from learning.datasets.dataset_builder import DatasetBuildConfig, build_belief_sample
from selfplay.data_recorder import run_recorded_selfplay_game


def torch():
    return pytest.importorskip("torch")


def _samples():
    _, records = run_recorded_selfplay_game(game_id="eval-belief", seed=31, max_steps=240)
    return [build_belief_sample(record, DatasetBuildConfig(seed=4, degradation_profile="perfect")) for record in records[:4]]


def test_evaluate_belief_model_returns_tile_and_binary_metrics():
    torch()
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from learning.eval.eval_belief import evaluate_belief_model

    samples = _samples()
    model = BeliefNet(BeliefNetConfig(input_size=len(samples[0].encoded.values), hidden_size=32, residual_blocks=1))

    report = evaluate_belief_model(model, samples)

    assert report.samples == len(samples)
    assert report.tile_count > 0
    assert report.tile_log_loss > 0.0
    assert report.tile_brier >= 0.0
    assert 0.0 <= report.opponent_tenpai_brier <= 1.0
    assert 0.0 <= report.discard_danger_ece <= 1.0


def test_evaluate_belief_model_is_deterministic_and_groups_by_phase():
    torch()
    from learning.models.belief_net import BeliefNet, BeliefNetConfig, set_torch_seed
    from learning.eval.eval_belief import evaluate_belief_model_by_phase

    samples = _samples()
    set_torch_seed(17)
    model = BeliefNet(BeliefNetConfig(input_size=len(samples[0].encoded.values), hidden_size=32, residual_blocks=1))

    first = evaluate_belief_model_by_phase(model, samples)
    second = evaluate_belief_model_by_phase(model, samples)

    assert first == second
    assert set(first) <= {"opening", "middle", "late"}
    assert sum(report.samples for report in first.values()) == len(samples)


@pytest.mark.skipif(not torch().cuda.is_available(), reason="CUDA unavailable")
def test_evaluate_belief_model_accepts_cuda_model():
    from learning.models.belief_net import BeliefNet, BeliefNetConfig
    from learning.eval.eval_belief import evaluate_belief_model

    samples = _samples()
    model = BeliefNet(BeliefNetConfig(input_size=len(samples[0].encoded.values), hidden_size=32, residual_blocks=1)).cuda()

    report = evaluate_belief_model(model, samples)

    assert report.samples == len(samples)


def test_evaluate_prior_belief_records_provides_baseline_and_profile_buckets():
    from learning.eval.eval_belief import evaluate_prior_belief_records, evaluate_prior_belief_records_by_profile

    _, records = run_recorded_selfplay_game(game_id="eval-prior", seed=32, max_steps=240)
    records = records[:4]

    report = evaluate_prior_belief_records(records, DatasetBuildConfig(seed=5, degradation_profile="perfect"))
    by_profile = evaluate_prior_belief_records_by_profile(records, [
        DatasetBuildConfig(seed=5, degradation_profile="perfect"),
        DatasetBuildConfig(seed=5, degradation_profile="light_noise"),
    ])

    assert report.samples == len(records)
    assert report.tile_count > 0
    assert report.tile_log_loss > 0.0
    assert set(by_profile) == {"perfect", "light_noise"}
    assert all(bucket.samples == len(records) for bucket in by_profile.values())


def test_report_uses_soft_tile_distribution_metrics_and_mask():
    from learning.eval.eval_belief import _report_from_predictions

    sample = _samples()[0]
    distribution = [[0.0, 0.0, 0.0, 0.0] for _ in range(27)]
    mask = [False] * 27
    distribution[0] = [0.25, 0.75, 0.0, 0.0]
    mask[0] = True
    sample = replace(sample, labels={**sample.labels, "tile_locations": {"counts": [[0] * 4 for _ in range(27)], "distribution": distribution, "mask": mask}})
    tile_probs = [[[0.4, 0.3, 0.2, 0.1] for _ in range(27)]]
    tenpai_probs = [[0.5, 0.5, 0.5]]
    danger_probs = [[[0.5, 0.5, 0.5] for _ in range(27)]]

    report = _report_from_predictions([sample], tile_probs, tenpai_probs, danger_probs)

    assert report.tile_count == 1
    assert report.tile_log_loss == pytest.approx(-(0.25 * math.log(0.4) + 0.75 * math.log(0.3)))
    assert report.tile_brier == pytest.approx(sum((p - q) ** 2 for p, q in zip(tile_probs[0][0], distribution[0])))


def test_evaluation_rejects_legacy_tile_location_labels():
    from learning.eval.eval_belief import _report_from_predictions

    sample = replace(_samples()[0], labels={"tile_locations": {"1W": "wall"}})
    with pytest.raises(ValueError, match="legacy|旧|incompatible"):
        _report_from_predictions(
            [sample],
            [[[0.25] * 4 for _ in range(27)]],
            [[0.5] * 3],
            [[[0.5] * 3 for _ in range(27)]],
        )



