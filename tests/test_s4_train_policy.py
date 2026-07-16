from dataclasses import replace

import pytest

from learning.datasets.dataset_builder import DatasetBuildConfig, build_policy_sample

from selfplay.data_recorder import run_recorded_selfplay_game
from state.action_space import action_space_size


def torch():
    return pytest.importorskip("torch")


def test_policy_net_masks_illegal_actions_and_normalizes_legal_probs():
    torch_module = torch()
    from learning.models.policy_net import PolicyNet, PolicyNetConfig

    model = PolicyNet(PolicyNetConfig(input_size=12, action_size=action_space_size(), hidden_size=32, residual_blocks=1))
    features = torch_module.zeros(2, 12)
    legal_mask = torch_module.zeros(2, action_space_size(), dtype=torch_module.bool)
    legal_mask[0, [0, 3]] = True
    legal_mask[1, [2]] = True

    output = model(features, legal_mask=legal_mask)

    assert output.logits.shape == (2, action_space_size())
    assert output.probs.shape == (2, action_space_size())
    assert torch_module.allclose(output.probs.sum(dim=-1), torch_module.ones(2), atol=1e-6)
    assert float(output.probs[0, 1].detach()) == pytest.approx(0.0)
    assert output.probs[0, 0] > 0.0
    assert output.probs[0, 3] > 0.0
    assert float(output.probs[1, 2].detach()) == pytest.approx(1.0)



def test_policy_batch_applies_forced_and_key_decision_weights():
    from learning.models.policy_net import PolicyNetConfig
    from learning.training.train_policy import TrainPolicyConfig, policy_batch_from_samples

    _, records = run_recorded_selfplay_game(game_id="policy-weights", seed=30, max_steps=200)
    base = build_policy_sample(records[0], DatasetBuildConfig(seed=4, degradation_profile="perfect"))
    config = TrainPolicyConfig(
        model=PolicyNetConfig(input_size=len(base.encoded.values), action_size=action_space_size(), hidden_size=16, residual_blocks=1)
    )
    samples = [
        replace(base, action_kind="pass", legal_action_count=1, is_pong_pass_decision=False),
        replace(base, action_kind="discard", legal_action_count=2, is_pong_pass_decision=False),
        replace(base, action_kind="swap_three", legal_action_count=2, is_pong_pass_decision=False),
        replace(base, action_kind="declare_void", legal_action_count=2, is_pong_pass_decision=False),
        replace(base, action_kind="pong", legal_action_count=2, is_pong_pass_decision=True),
    ]

    batch = policy_batch_from_samples(samples, config)

    assert batch.sample_weights.tolist() == pytest.approx([0.1, 1.5, 2.0, 1.5, 2.0])


def test_weighted_policy_loss_matches_manual_cross_entropy():
    torch_module = torch()
    from learning.models.policy_net import PolicyNet, PolicyNetConfig
    from learning.training.train_policy import PolicyBatch, train_policy_step

    model = PolicyNet(PolicyNetConfig(input_size=3, action_size=action_space_size(), hidden_size=8, residual_blocks=1))
    optimizer = torch_module.optim.SGD(model.parameters(), lr=0.0)
    legal_mask = torch_module.zeros(2, action_space_size(), dtype=torch_module.bool)
    legal_mask[:, :2] = True
    batch = PolicyBatch(
        features=torch_module.zeros(2, 3),
        action_targets=torch_module.tensor([0, 1]),
        legal_mask=legal_mask,
        sample_weights=torch_module.tensor([1.0, 3.0]),
    )
    with torch_module.no_grad():
        logits = model(batch.features, legal_mask=batch.legal_mask).logits
        per_sample = torch_module.nn.functional.cross_entropy(logits, batch.action_targets, reduction="none")
        expected = float((per_sample * batch.sample_weights).sum() / batch.sample_weights.sum())

    metrics = train_policy_step(model, batch, optimizer)

    assert metrics["loss"] == pytest.approx(expected)


def test_policy_training_step_updates_model_and_reports_accuracy():

    torch_module = torch()
    from learning.models.policy_net import PolicyNet, PolicyNetConfig
    from learning.training.train_policy import policy_batch_from_samples, train_policy_step

    _, records = run_recorded_selfplay_game(game_id="train-policy", seed=31, max_steps=200)
    samples = [build_policy_sample(record, DatasetBuildConfig(seed=4, degradation_profile="perfect")) for record in records[:4]]
    batch = policy_batch_from_samples(samples)
    model = PolicyNet(PolicyNetConfig(input_size=len(samples[0].encoded.values), action_size=action_space_size(), hidden_size=32, residual_blocks=1))
    optimizer = torch_module.optim.Adam(model.parameters(), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    metrics = train_policy_step(model, batch, optimizer)

    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert any(not torch_module.allclose(old, new) for old, new in zip(before, model.parameters()))


def test_train_policy_epoch_returns_metrics_and_checkpoint_payload(tmp_path):
    torch_module = torch()
    from learning.models.policy_net import PolicyNetConfig
    from learning.training.train_policy import TrainPolicyConfig, save_policy_checkpoint, train_policy_epoch

    _, records = run_recorded_selfplay_game(game_id="train-policy-epoch", seed=32, max_steps=200)
    samples = [build_policy_sample(record, DatasetBuildConfig(seed=5, degradation_profile="perfect")) for record in records[:4]]
    config = TrainPolicyConfig(
        model=PolicyNetConfig(input_size=len(samples[0].encoded.values), action_size=action_space_size(), hidden_size=32, residual_blocks=1),
        batch_size=2,
        learning_rate=0.01,
        seed=12,
        device="cpu",
    )

    model, metrics = train_policy_epoch(samples, config)
    checkpoint = tmp_path / "policy.pt"
    save_policy_checkpoint(checkpoint, model, config, metrics, data_fingerprint="tiny")
    payload = torch_module.load(checkpoint, map_location="cpu")

    assert model.config.input_size == len(samples[0].encoded.values)
    assert next(model.parameters()).device.type == "cpu"
    assert metrics["batches"] == 2
    assert metrics["samples"] == 4
    assert metrics["device"] == "cpu"
    assert metrics["loss"] > 0.0
    assert payload["model_config"]["action_size"] == action_space_size()
    assert payload["training_config"]["seed"] == 12
    assert payload["metrics"] == metrics
    from state.encoder import ENCODER_VERSION

    assert payload["data_fingerprint"] == "tiny"
    assert payload["encoder_version"] == ENCODER_VERSION
    assert payload["execution_device"] == "cpu"


def test_train_policy_runs_multiple_epochs_and_restores_best_state():
    from learning.models.policy_net import PolicyNetConfig
    from learning.training.train_policy import TrainPolicyConfig, train_policy

    _, train_records = run_recorded_selfplay_game(game_id="multi-epoch-train", seed=34, max_steps=200)
    _, val_records = run_recorded_selfplay_game(game_id="multi-epoch-val", seed=35, max_steps=200)
    train_samples = [build_policy_sample(record) for record in train_records[:8]]
    val_samples = [build_policy_sample(record) for record in val_records[:4]]
    config = TrainPolicyConfig(
        model=PolicyNetConfig(input_size=len(train_samples[0].encoded.values), action_size=action_space_size(), hidden_size=16, residual_blocks=1),
        batch_size=2,
        learning_rate=0.01,
        max_epochs=3,
        patience=2,
        min_delta=0.0,
        device="cpu",
    )

    model, metrics = train_policy(train_samples, val_samples, config)

    assert next(model.parameters()).device.type == "cpu"
    assert 1 <= metrics["best_epoch"] <= metrics["epochs_trained"] <= 3
    assert len(metrics["history"]) == metrics["epochs_trained"]
    assert metrics["best_validation_metrics"]["samples"] == len(val_samples)


def test_train_policy_rejects_empty_validation_samples():
    from learning.models.policy_net import PolicyNetConfig
    from learning.training.train_policy import TrainPolicyConfig, train_policy

    _, records = run_recorded_selfplay_game(game_id="empty-val", seed=36, max_steps=100)
    sample = build_policy_sample(records[0])
    config = TrainPolicyConfig(
        model=PolicyNetConfig(input_size=len(sample.encoded.values), action_size=action_space_size(), hidden_size=16, residual_blocks=1)
    )

    with pytest.raises(ValueError, match="validation_samples"):
        train_policy([sample], [], config)


def test_learned_policy_loads_checkpoint_on_cpu_and_selects_legal_action(tmp_path):

    torch_module = torch()
    from engine.game import Game
    from learning.models.policy_net import PolicyNet, PolicyNetConfig
    from policies.learned_policy import LearnedPolicy
    from policies.protocol_actions import action_to_protocol
    from state.action_space import action_to_index, legal_mask
    from state.adapters.from_engine import from_engine
    from state.encoder import ENCODER_VERSION, encode_state

    state = Game(seed=71).reset()
    protocol_state = from_engine(state, player_id=0)
    mask = legal_mask(protocol_state)
    target_index = next(index for index, allowed in enumerate(mask) if allowed)
    model = PolicyNet(
        PolicyNetConfig(
            input_size=encode_state(protocol_state).size,
            action_size=action_space_size(),
            hidden_size=16,
            residual_blocks=1,
        )
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.action_head.bias.data[target_index] = 1.0
    checkpoint = tmp_path / "policy.pt"
    torch_module.save(
        {
            "model_config": model.config.__dict__,
            "encoder_version": ENCODER_VERSION,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )

    policy = LearnedPolicy(checkpoint)
    action = policy.choose_action(protocol_state, mask)

    assert next(policy.model.parameters()).device.type == "cpu"
    assert action_to_index(action_to_protocol(action)) == target_index


def test_learned_policy_rejects_previous_encoder_checkpoint(tmp_path):
    torch_module = torch()
    from learning.models.policy_net import PolicyNet, PolicyNetConfig
    from policies.learned_policy import LearnedPolicy

    model = PolicyNet(
        PolicyNetConfig(input_size=263, action_size=action_space_size(), hidden_size=16, residual_blocks=1)
    )
    checkpoint = tmp_path / "legacy-policy.pt"
    torch_module.save(
        {
            "model_config": model.config.__dict__,
            "encoder_version": "s2.v4.encoder.v2",
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="encoder version"):
        LearnedPolicy(checkpoint)


def test_learned_policy_requires_belief_checkpoint_for_learned_input(tmp_path):

    torch_module = torch()
    from learning.models.policy_net import PolicyNet, PolicyNetConfig
    from policies.learned_policy import LearnedPolicy
    from state.encoder import ENCODER_VERSION

    model = PolicyNet(PolicyNetConfig(input_size=263, action_size=action_space_size(), hidden_size=16, residual_blocks=1))
    checkpoint = tmp_path / "learned-policy.pt"
    torch_module.save(
        {
            "model_config": model.config.__dict__,
            "encoder_version": ENCODER_VERSION,
            "state_dict": model.state_dict(),
            "belief_metadata": {"source": "learned"},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="belief_model_path"):
        LearnedPolicy(checkpoint)


@pytest.mark.skipif(not torch().cuda.is_available(), reason="CUDA unavailable")

def test_train_policy_epoch_moves_model_to_cuda():
    from learning.models.policy_net import PolicyNetConfig
    from learning.training.train_policy import TrainPolicyConfig, train_policy_epoch

    _, records = run_recorded_selfplay_game(game_id="train-policy-cuda", seed=33, max_steps=200)
    samples = [build_policy_sample(record, DatasetBuildConfig(seed=5, degradation_profile="perfect")) for record in records[:4]]
    model, metrics = train_policy_epoch(
        samples,
        TrainPolicyConfig(
            model=PolicyNetConfig(input_size=len(samples[0].encoded.values), action_size=action_space_size(), hidden_size=16, residual_blocks=1),
            batch_size=2,
            device="cuda",
        ),
    )

    assert next(model.parameters()).device.type == "cuda"
    assert metrics["device"] == "cuda"

