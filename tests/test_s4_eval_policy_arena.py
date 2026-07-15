import pytest

from policies.opponent_pool import RandomPolicy
from policies.rule_policy import RulePolicy


def torch():
    return pytest.importorskip("torch")


def test_arena_runs_policy_mix_with_reproducible_zero_sum_stats():
    from learning.eval.arena import ArenaConfig, run_arena

    policies = [RulePolicy(), RandomPolicy(seed=7), RulePolicy(), RandomPolicy(seed=11)]
    first = run_arena(policies, ArenaConfig(games=2, seed=41, max_steps=300))
    second = run_arena(policies, ArenaConfig(games=2, seed=41, max_steps=300))

    assert first.games == 2
    assert first.illegal_actions == 0
    assert first.zero_sum_violations == 0
    assert first.average_scores == second.average_scores
    assert first.score_totals == second.score_totals
    assert sum(first.score_totals) == 0
    assert 0.0 <= first.draw_rate <= 1.0
    assert len(first.score_confidence95) == 4


def test_arena_counts_illegal_policy_action_without_crashing_batch():
    from engine.actions import Action, ActionKind
    from learning.eval.arena import ArenaConfig, run_arena
    from policies.base_policy import BasePolicy

    class IllegalPolicy(BasePolicy):
        def choose_action(self, protocol_state, legal_mask):
            return Action(ActionKind.DRAW)


    report = run_arena([IllegalPolicy(), RulePolicy(), RulePolicy(), RulePolicy()], ArenaConfig(games=1, seed=3, max_steps=20))

    assert report.games == 1
    assert report.illegal_actions >= 1
    assert report.results[0].illegal_action is True
    assert report.unfinished == 1


def test_evaluate_policy_samples_reports_accuracy_and_illegal_probability():
    torch_module = torch()
    from learning.datasets.dataset_builder import DatasetBuildConfig, build_policy_sample
    from learning.eval.eval_policy import evaluate_policy_samples
    from learning.models.policy_net import PolicyNet, PolicyNetConfig
    from selfplay.data_recorder import run_recorded_selfplay_game
    from state.action_space import action_space_size

    _, records = run_recorded_selfplay_game(game_id="eval-policy", seed=51, max_steps=200)
    samples = [build_policy_sample(record, DatasetBuildConfig(seed=2, degradation_profile="perfect")) for record in records[:3]]
    model = PolicyNet(PolicyNetConfig(input_size=len(samples[0].encoded.values), action_size=action_space_size(), hidden_size=16, residual_blocks=1))

    report = evaluate_policy_samples(model, samples)

    assert report.samples == 3
    assert 0.0 <= report.top1_accuracy <= 1.0
    assert report.illegal_argmax_count == 0
    assert report.illegal_probability_mass == pytest.approx(0.0)
    assert report.forced_samples + report.non_forced_samples == report.samples
    assert report.forced_rate == pytest.approx(report.forced_samples / report.samples)
    assert report.non_forced_accuracy is None or 0.0 <= report.non_forced_accuracy <= 1.0
    assert sum(item.samples for item in report.by_action_kind.values()) == report.samples
    assert sum(item.samples for item in report.by_phase.values()) == report.samples
    assert report.pong_pass_response.samples >= 0


@pytest.mark.skipif(not torch().cuda.is_available(), reason="CUDA unavailable")
def test_evaluate_policy_samples_accepts_cuda_model():
    from learning.datasets.dataset_builder import DatasetBuildConfig, build_policy_sample
    from learning.eval.eval_policy import evaluate_policy_samples
    from learning.models.policy_net import PolicyNet, PolicyNetConfig
    from selfplay.data_recorder import run_recorded_selfplay_game
    from state.action_space import action_space_size

    _, records = run_recorded_selfplay_game(game_id="eval-policy-cuda", seed=52, max_steps=200)
    samples = [build_policy_sample(record, DatasetBuildConfig(seed=2, degradation_profile="perfect")) for record in records[:3]]
    model = PolicyNet(
        PolicyNetConfig(input_size=len(samples[0].encoded.values), action_size=action_space_size(), hidden_size=16, residual_blocks=1)
    ).cuda()

    report = evaluate_policy_samples(model, samples)

    assert report.samples == len(samples)
