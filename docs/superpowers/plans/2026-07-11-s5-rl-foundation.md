# S5 RL Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete, testable S5 PPO self-play training foundation specified in `strategy_S5_rl_spec.md`, ready for a later GPU training run.

**Architecture:** Add an isolated `rl` package around the existing S1 game loop, S2 observation pipeline, frozen S4 Belief and S4 Policy model. Rollouts collect only learner-visible state and terminal S1 reward; PPO optimizes a policy/value network with legal masks and a decaying frozen-S4 KL reference. League, curriculum, checkpoints, arena reports and cloud packaging are orchestration layers around that core.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing S1 engine/S2 state/S3 policies/S4 models and arena.

---

## File structure

- Create `rl/types.py`: immutable trajectory, rollout, health and training-report data structures.
- Create `rl/reward.py`: terminal-score normalization and reward assignment.
- Create `rl/models/value_net.py`: S4-compatible policy trunk, action head and scalar value head.
- Create `rl/ppo_trainer.py`: masked distributions, GAE and PPO update.
- Create `rl/rollout.py`: learner-seat game runner and S2/S4 feature boundary.
- Create `rl/league.py`: typed opponent/snapshot registry, sampling and admission gate.
- Create `rl/curriculum.py`: learner-only degradation stages and advancement state.
- Create `rl/checkpoints.py`: atomic checkpoint serialization and restore.
- Create `rl/train_rl.py`: configuration, training loop, reports and CLI.
- Create `tools/cloud_train_s5.py`: cloud smoke/full-job wrapper and package report.
- Create `docs/concepts.md`: user-facing S5 concepts and operations guide.
- Create `tests/test_s5_*.py`: focused tests named by module.
- Modify `learning/eval/arena.py` only to add a narrow S5 policy adapter interface if the existing `BasePolicy` boundary cannot represent a frozen-Belief network policy.

### Task 1: Typed trajectory and terminal reward

**Files:**
- Create: `rl/types.py`
- Create: `rl/reward.py`
- Create: `tests/test_s5_reward.py`

- [ ] **Step 1: Write the failing reward tests.**

```python
def test_terminal_reward_is_only_on_final_transition() -> None:
    transitions = [transition(), transition(), transition()]
    rewarded = assign_terminal_reward(transitions, final_score=24, score_scale=48.0)
    assert [item.reward for item in rewarded] == [0.0, 0.0, 0.5]
    assert [item.done for item in rewarded] == [False, False, True]

def test_terminal_reward_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="score_scale"):
        normalize_terminal_score(12, 0.0)
```

- [ ] **Step 2: Run `python -m pytest tests/test_s5_reward.py -q`; expect import failure.**
- [ ] **Step 3: Implement immutable `TrajectoryStep` with CPU tensors/features, action, legal mask, log probability, value, reward and done; implement `normalize_terminal_score` and `assign_terminal_reward`.**
- [ ] **Step 4: Re-run the targeted tests; expect all pass.**

### Task 2: Policy/value network and legal masked action distribution

**Files:**
- Create: `rl/models/__init__.py`
- Create: `rl/models/value_net.py`
- Create: `tests/test_s5_value_net.py`

- [ ] **Step 1: Write failing tests that load the S4-compatible policy trunk, assert value output shape `(batch,)`, and assert illegal actions have zero probability.**

```python
output = model(features, legal_mask)
assert output.values.shape == (2,)
assert torch.all(output.action_logits[~legal_mask] == torch.finfo(torch.float32).min)
```

- [ ] **Step 2: Run `python -m pytest tests/test_s5_value_net.py -q`; expect import failure.**
- [ ] **Step 3: Implement `PolicyValueNetConfig`, `PolicyValueNetOutput` and `PolicyValueNet`. Reuse the PolicyNet trunk layout exactly; expose `load_s4_policy_state_dict` that copies trunk/action-head weights and leaves the value head initialized.**
- [ ] **Step 4: Re-run the targeted tests; expect all pass.**

### Task 3: PPO mathematics and update health metrics

**Files:**
- Create: `rl/ppo_trainer.py`
- Create: `tests/test_s5_ppo.py`

- [ ] **Step 1: Write failing deterministic tests for `compute_gae`, clipped-ratio selection, masked log-probability, KL reference loss, one optimizer update and NaN rejection.**

```python
advantages, returns = compute_gae(rewards, values, dones, gamma=1.0, gae_lambda=1.0)
assert advantages.tolist() == pytest.approx([3.0, 2.0])
assert torch.isfinite(metrics.total_loss)
```

- [ ] **Step 2: Run `python -m pytest tests/test_s5_ppo.py -q`; expect import failure.**
- [ ] **Step 3: Implement `PPOConfig`, `PPOBatch`, `PPOHealth`, `compute_gae` and `ppo_update`. Require every row to have a legal action, normalize advantages, clip gradients, compute entropy/KL only from masked logits, and return clip fraction, entropy, KL, policy loss, value loss and grad norm.**
- [ ] **Step 4: Re-run the targeted tests; expect all pass.**

### Task 4: Learner-visible rollout and invariants

**Files:**
- Create: `rl/rollout.py`
- Create: `tests/test_s5_rollout.py`
- Modify: `rl/types.py`

- [ ] **Step 1: Write failing integration tests that run a short learner-vs-S3 game, assert only the learner receives trajectory entries, all selected actions are legal, rewards are terminal-only, scores sum to zero and seeded rollouts reproduce actions.**
- [ ] **Step 2: Run `python -m pytest tests/test_s5_rollout.py -q`; expect import failure.**
- [ ] **Step 3: Implement `RolloutConfig`, `FrozenBeliefProvider`, `PolicyValueAgent` and `run_rollout_game`. Mirror the S1 pending-discard/swap/void/rob-kong control flow in `arena.py`; route learner decisions through S2 degraded observation plus frozen Belief, route opponents through their own perfect-observation policies, and raise `RolloutInvariantError` on illegal, unfinished or non-zero-sum games.**
- [ ] **Step 4: Re-run targeted tests; expect all pass.**

### Task 5: League and curriculum state machines

**Files:**
- Create: `rl/league.py`
- Create: `rl/curriculum.py`
- Create: `tests/test_s5_league.py`
- Create: `tests/test_s5_curriculum.py`

- [ ] **Step 1: Write failing tests for exact weighted sampling, milestone retention, capacity eviction, failed anti-forgetting admission, JSON round trip, learner-only degradation and deterministic stage advancement.**
- [ ] **Step 2: Run `python -m pytest tests/test_s5_league.py tests/test_s5_curriculum.py -q`; expect import failure.**
- [ ] **Step 3: Implement `LeagueConfig`/`OpponentLeague` and `CurriculumConfig`/`ObservationCurriculum`. Use default source weights latest `.35`, history `.35`, S3 `.20`, greedy/random `.10`; accept a candidate snapshot only when its supplied dual-arena metrics satisfy the configured minimums. Ensure `opponent_profile()` always returns perfect observation.**
- [ ] **Step 4: Re-run targeted tests; expect all pass.**

### Task 6: Checkpoint, orchestrator and dual-track evaluation

**Files:**
- Create: `rl/checkpoints.py`
- Create: `rl/train_rl.py`
- Create: `tests/test_s5_checkpoint.py`
- Create: `tests/test_s5_train_smoke.py`

- [ ] **Step 1: Write failing tests that save/restore model, optimizer, RNG, league and curriculum state; assert resumed training emits the same next rollout seed; run a two-update CPU smoke job and assert both arena tracks and health metrics are written.**
- [ ] **Step 2: Run `python -m pytest tests/test_s5_checkpoint.py tests/test_s5_train_smoke.py -q`; expect import failure.**
- [ ] **Step 3: Implement atomic `save_checkpoint`/`load_checkpoint` using a sibling temporary file and replacement. Implement `S5TrainingConfig` and `run_s5_training`: load frozen local S4 artifacts, generate rollout batches, call PPO, schedule KL, snapshot candidates, evaluate perfect and degraded tracks, write JSON/Markdown report and preserve a diagnostic checkpoint for health alerts.**
- [ ] **Step 4: Re-run targeted tests; expect all pass.**

### Task 7: Expert-review materials and operator documentation

**Files:**
- Create: `rl/expert_review.py`
- Create: `tests/test_s5_expert_review.py`
- Create: `docs/concepts.md`

- [ ] **Step 1: Write a failing test that turns a retained rollout into a JSON review item containing public observation summary, policy action, S3 action, S4 action, Belief summary and one checklist category without hidden-hand fields.**
- [ ] **Step 2: Run `python -m pytest tests/test_s5_expert_review.py -q`; expect import failure.**
- [ ] **Step 3: Implement checklist-category selection and redacted review serialization. Write `docs/concepts.md` covering self-play, PPO/GAE, terminal reward, opponent league, observation curriculum, incomplete-information safety, training health and the two arena tracks.**
- [ ] **Step 4: Re-run targeted tests; expect all pass.**

### Task 8: Cloud package and end-to-end verification

**Files:**
- Create: `tools/cloud_train_s5.py`
- Create: `S5_CLOUD_TRAINING_README.md`
- Create: `tests/test_s5_cloud_package.py`
- Modify: packaging configuration only if `tools/cloud_train_s4.py` has a reusable package helper.

- [ ] **Step 1: Write a failing package test requiring the ZIP to contain `rl/`, the frozen local S4 checkpoints, S5 docs, test-free runtime sources and `--device cuda` smoke command.**
- [ ] **Step 2: Run `python -m pytest tests/test_s5_cloud_package.py -q`; expect failure.**
- [ ] **Step 3: Implement a deterministic ZIP builder and runner. The smoke command must use a tiny rollout/update budget, require CUDA when selected, and write an artifact manifest containing S4 checkpoint hashes, config, device and two-track arena result.**
- [ ] **Step 4: Run all S5 tests then the full suite: `python -m pytest tests/test_s5_*.py -q` and `python -m pytest -q`; expect zero failures. Build the ZIP, extract to a temporary directory and execute its CPU smoke command.**

## Coverage self-review

- Tasks 1 and 4 implement terminal-only reward, S1 episode boundaries, trajectory schema, zero-sum and hidden-information boundaries.
- Tasks 2 and 3 implement S4 initialization, frozen KL reference, value head, PPO/GAE/action masks and health metrics.
- Task 5 implements the required opponent-pool diversity, anti-forgetting and learner-only observation curriculum.
- Task 6 implements resumability, dual-track arena and artifact reports.
- Task 7 implements the eight-category human review material and concepts documentation.
- Task 8 implements the required cloud-ready package and reproducible smoke verification.

The plan has no deferred requirements: long-duration training is explicitly deferred only because it needs a newly provisioned cloud instance and an operator-selected compute budget; all code, configuration, package and smoke validation precede it.
