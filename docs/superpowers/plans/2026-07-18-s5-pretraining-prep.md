# S5 Pretraining Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect the accepted S4 v2–v5 assets in Git, complete the S4-v5 belief bucket evaluation, and prove the S5 runtime can use the 893-feature v5 policy/belief artifacts without starting long S5 training.

**Architecture:** Preserve the current dirty main worktree first with narrowly scoped commits and a push to `origin/main`. Add a deterministic offline belief evaluator that shares the S4 data split and degradation pipeline. Then adapt S5 only at configuration/loading boundaries so dimensions come from the v5 checkpoint/encoder, and use a bounded local smoke harness for rollout, PPO, resume, league, curriculum isolation, and throughput evidence.

**Tech Stack:** Python 3, PyTorch, pytest, existing S1 engine/S2 protocol/S3 policies/S4 artifacts, Git.

---

### Task 1: Preserve the current S4 assets in scoped Git commits

**Files:**
- Create: `.gitignore` if absent
- Create: `docs/s4_gate_history.md`
- Modify: `state/hand_analysis.py`
- Add: `tools/cloud_train_s4_50k_cached.py`, `tools/generate_dagger_data.py`
- Add: `training_artifacts/S4/v2_20260716_50k_encoder_v3/**`, `v3_20260717_dagger1/**`, `v4_20260717_dagger2/**`, `v5_20260718_encoder_v4/**`
- Add: `strategy_S2_encoder_upgrade_spec.md`, `strategy_S2_encoder_v4_candidate_features_spec.md`, `strategy_local_training_infra_spec.md`, `strategy_S5_prep_spec.md`

- [ ] **Step 1: Inspect the exact dirty set and ignore rules before staging.**

Run: `git status --short; git check-ignore -v __pycache__ .pytest_cache data_dagger_probe cloud_outputs/run/data`

Expected: only the specified cache fix, tools, v2–v5 artifacts, and specification documents are candidates; data shards are ignored.

- [ ] **Step 2: Add an ignore file if needed and prove it excludes generated bulk data.**

```gitignore
__pycache__/
*.out
.pytest_cache/
data_dagger*/
cloud_outputs/*/data/
```

Run: `git check-ignore -v __pycache__/x.py .pytest_cache/x data_dagger_probe/a.jsonl cloud_outputs/run/data/a.jsonl`

Expected: all four generated paths are ignored while `training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/policy_s4.pt` is not ignored.

- [ ] **Step 3: Write the gate-history record before committing documentation.**

`docs/s4_gate_history.md` must contain the v1–v5 table from the supplied spec, the causal diagnosis sequence `data volume -> distribution shift -> representation capability`, the accepted v5 result `-0.066 +/- 0.42`, and this reproduction command:

```powershell
python -m learning.eval.arena --seed 90000 --games 500 --model-seat 0 --policy-checkpoint training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/policy_s4.pt --opponent rule --opponent rule --opponent rule
```

- [ ] **Step 4: Create the code-fix commit from only the cache-fix file.**

Run: `git add state/hand_analysis.py; git diff --cached -- state/hand_analysis.py; git commit -m "fix: bound hand block cache after production incident"`

Expected: the commit contains no tools, artifacts, reports, or data shards.

- [ ] **Step 5: Create the tool commit from only the two named tools.**

Run: `git add tools/cloud_train_s4_50k_cached.py tools/generate_dagger_data.py; git diff --cached --stat; git commit -m "feat: add cached S4 training and DAgger data tools"`

Expected: the index contains exactly the two tools.

- [ ] **Step 6: Archive only the listed S4 version directories.**

Run: `git add training_artifacts/S4/v2_20260716_50k_encoder_v3 training_artifacts/S4/v3_20260717_dagger1 training_artifacts/S4/v4_20260717_dagger2 training_artifacts/S4/v5_20260718_encoder_v4; git diff --cached --name-only; git commit -m "docs: archive accepted S4 iteration artifacts"`

Expected: checkpoints and reports are included; no raw decision JSONL or `cloud_outputs/*/data` path is staged.

- [ ] **Step 7: Commit specifications and gate history, then push the approved main branch.**

Run: `git add .gitignore docs/s4_gate_history.md strategy_S2_encoder_upgrade_spec.md strategy_S2_encoder_v4_candidate_features_spec.md strategy_local_training_infra_spec.md strategy_S5_prep_spec.md docs/superpowers/plans/2026-07-18-s5-pretraining-prep.md; git commit -m "docs: record S4 gate history and S5 pretraining criteria"; git status --short; git push origin main`

Expected: status is clean except intentional ignored files, `origin/main` accepts all topic commits.

### Task 2: Implement the deterministic S4-v5 belief bucket examination

**Files:**
- Create: `tools/eval_belief_buckets.py`
- Create: `tests/test_eval_belief_buckets.py`
- Create: `training_artifacts/S4/v5_20260718_encoder_v4/reports/belief_bucket_report.md`
- Read: `learning/datasets/dataset_builder.py`, `learning/eval/eval_belief.py`, `tools/cloud_train_s4_50k_cached.py`

- [ ] **Step 1: Write failing tests for the split, phase buckets, profile buckets, and acceptance calculations.**

```python
def test_bucket_evaluation_uses_training_hash_validation_split_and_all_profiles() -> None:
    report = evaluate_bucket_records(records, seed=20260718, checkpoint=checkpoint)
    assert set(report.profiles) == {"perfect", "light_noise", "midgame", "heavy"}
    assert set(report.phase_buckets) == {"opening", "midgame", "endgame"}

def test_acceptance_requires_endgame_gain_to_exceed_opening_and_every_profile_to_beat_prior() -> None:
    assert evaluate_acceptance(passing_metrics).passed
    assert not evaluate_acceptance(failing_metrics).passed
```

- [ ] **Step 2: Run the new tests to establish RED.**

Run: `python -m pytest tests/test_eval_belief_buckets.py -q`

Expected: import failure because `tools.eval_belief_buckets` and its public functions do not exist.

- [ ] **Step 3: Implement deterministic record loading and sampling.**

Implement `load_validation_records(...)` to discover local extracted shards or regenerate a fixed seed range through the existing generator when absent. Reuse `_split_name(game_id, seed)` from `tools/cloud_train_s4_50k_cached.py`, retain only `val` records, deterministically sample at least 20,000 records when available, and write the selected seed range/count into report metadata.

- [ ] **Step 4: Implement metrics through existing S4 dataset/degradation components.**

Implement `evaluate_bucket_records(...)` using `DatasetBuildConfig` and `build_belief_sample` for all profiles. Bucket by wall count as `opening > 40`, `midgame 20..40`, `endgame < 20`; calculate model and prior tile log-loss plus `prior - model`; reject NaN/empty buckets.

- [ ] **Step 5: Implement report serialization and acceptance gate.**

Write a deterministic Markdown table to `training_artifacts/S4/v5_20260718_encoder_v4/reports/belief_bucket_report.md`. Include checkpoint SHA256, encoder version, seed, source/regeneration metadata, every phase/profile metric, and an explicit PASS/EXCEPTION result. The gate passes only when endgame gain exceeds opening gain and every profile model loss beats prior loss.

- [ ] **Step 6: Verify GREEN and run the v5 evaluation locally.**

Run: `python -m pytest tests/test_eval_belief_buckets.py -q; python tools/eval_belief_buckets.py --checkpoint training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/belief_s4.pt --seed 20260718 --min-samples 20000`

Expected: tests pass; the command writes the report or exits nonzero with an explicit gate-exception report rather than silently accepting a failing bucket.

- [ ] **Step 7: Commit evaluator, tests, and generated report.**

Run: `git add tools/eval_belief_buckets.py tests/test_eval_belief_buckets.py training_artifacts/S4/v5_20260718_encoder_v4/reports/belief_bucket_report.md; git commit -m "test: add S4 belief bucket gate evaluation"`

### Task 3: Adapt S5 configuration/loading to v5’s dynamic encoder contract

**Files:**
- Modify: `rl/train_rl.py`, `rl/models/value_net.py`, `rl/ppo_trainer.py`, `rl/rollout.py`
- Modify: `tools/cloud_train_s5.py`
- Create: `tests/test_s5_v5_adapter.py`
- Read: `state/encoder.py`, `learning/models/policy_net.py`, `state/tile_belief.py`

- [ ] **Step 1: Write failing adapter tests using the accepted v5 checkpoints.**

```python
def test_v5_s4_assets_define_the_s5_policy_value_architecture() -> None:
    model, reference = load_v5_s5_models(V5_POLICY, V5_BELIEF)
    assert model.config.input_size == encode_state(from_engine(Game(seed=1).reset(), 0)).size == 893
    assert reference.encoder_version == "s2.v4.encoder.v4"

def test_v5_policy_is_the_frozen_kl_reference_and_belief_is_frozen() -> None:
    trainer = build_v5_trainer()
    assert all(not parameter.requires_grad for parameter in trainer.reference_policy.parameters())
    assert trainer.kl_coefficient_at(0) > trainer.kl_coefficient_at(1)
```

- [ ] **Step 2: Run RED and locate non-test legacy dimensions.**

Run: `python -m pytest tests/test_s5_v5_adapter.py -q; rg -n "\b(806|263)\b" rl learning -g '*.py' -g '!tests/**'`

Expected: new tests fail before adaptation; production search returns no encoder-size literal used as a default/input contract.

- [ ] **Step 3: Implement dynamic encoder-size and v5 checkpoint validation.**

Load policy architecture from the v5 policy checkpoint only after verifying `encoder_version == ENCODER_VERSION` and that its feature size equals `encode_state(...)` size. Make `PolicyValueNet` and all S5 model factories use that resolved architecture. Load `belief_s4.pt` solely through `LearnedBelief`, set evaluation mode and disable gradients. Preserve the frozen v5 policy copy as PPO’s KL reference and retain the configured decay schedule.

- [ ] **Step 4: Verify GREEN and run all S5 tests.**

Run: `python -m pytest tests/test_s5_v5_adapter.py -q; python -m pytest tests/test_s5_*.py -q`

Expected: adapter tests and the existing S5 suite pass with v5/893-dimensional evidence.

### Task 4: Run bounded local S5 smoke, resume, isolation, and throughput evidence

**Files:**
- Modify: `tools/cloud_train_s5.py` only if it needs a bounded local-prep mode
- Create: `tests/test_s5_v5_smoke.py`
- Create: `docs/s5_local_rollout_benchmark.md`
- Create: `training_artifacts/S5/prep_20260718/**`

- [ ] **Step 1: Write failing end-to-end smoke assertions.**

```python
def test_v5_local_smoke_has_fifty_completed_zero_illegal_rollouts_and_ten_finite_updates() -> None:
    result = run_local_v5_prep_smoke(games=50, updates=10, seed=20260718)
    assert result.completed_games == 50
    assert result.illegal_actions == 0
    assert result.zero_sum_failures == 0
    assert result.ppo_updates == 10
    assert all(math.isfinite(value) for value in result.losses)

def test_resume_league_and_curriculum_isolation_are_preserved() -> None:
    result = run_local_v5_prep_smoke(games=50, updates=10, seed=20260718)
    assert result.resume_curve_matches
    assert result.v5_snapshot_in_league
    assert result.opponents_used_perfect_observation
    assert result.learner_used_curriculum_degradation
```

- [ ] **Step 2: Run RED.**

Run: `python -m pytest tests/test_s5_v5_smoke.py -q`

Expected: failure because the bounded v5 preparation smoke API does not yet exist.

- [ ] **Step 3: Implement the bounded smoke harness without long training.**

Use the v5 policy as learner and frozen reference, v5 learned belief, and three opponent-pool members. Run 50 complete `run_rollout_game` episodes with fixed seeds, validate trajectory schema/zero illegal/zero-sum, run exactly 10 PPO updates, checkpoint then reload and continue five updates, and make the harness assert an unbroken resumed curve. Add v5 to the league and assert sampling weights. Apply curriculum degradation only to the learner’s `LearnerView`; retain perfect observations for opponents.

- [ ] **Step 4: Measure local rollout throughput and write the ledger.**

Run the same 50-game loop with `os.cpu_count()` and the active PyTorch thread count recorded. Write `docs/s5_local_rollout_benchmark.md` with machine/OS/CPU/thread values, games, elapsed seconds, games/minute, v5 artifact hashes, and the command used. Do not start a long S5 run.

- [ ] **Step 5: Verify the smoke and full tests.**

Run: `python -m pytest tests/test_s5_v5_smoke.py -q; python -m pytest tests/test_s5_*.py -q; python -m pytest -q`

Expected: all selected and full suites pass; smoke artifacts live under `training_artifacts/S5/prep_20260718` and contain no raw training shards.

- [ ] **Step 6: Commit smoke evidence and push.**

Run: `git add rl tools/cloud_train_s5.py tests/test_s5_v5_adapter.py tests/test_s5_v5_smoke.py docs/s5_local_rollout_benchmark.md training_artifacts/S5/prep_20260718; git commit -m "feat: prepare S5 runtime for v5 encoder smoke validation"; git status --short; git push origin main`

### Task 5: Deliver the non-executed formal S5 launch command

**Files:**
- Modify: `docs/s5_local_rollout_benchmark.md`

- [ ] **Step 1: Record the ready-but-not-executed command and time estimate.**

Append a section named `Human approval required` with:

```powershell
python -S tools/cloud_train_s5.py --mode train --device cuda --updates 1000 --episodes-per-update 64 --arena-games 100 --seed 20260718 --output-dir cloud_outputs/s5_v5_formal
```

Include the measured games/minute, the formula `estimated minutes = (updates * episodes_per_update + arena workload) / measured games_per_minute`, and state explicitly that this command was not executed during preparation.

- [ ] **Step 2: Verify the final state and push the documentation update.**

Run: `python -m pytest -q; git status --short; git log --oneline -6; git push origin main`

Expected: tests pass, status is clean except intentional ignores, history has separate code/tool/artifact/documentation/S5-prep commits, and origin contains them.

## Plan self-review

- Task 1 covers `.gitignore`, scoped commits, v2–v5 archive, gate-history documentation, and `origin/main` push.
- Task 2 covers split-compatible validation sampling, phase/profile buckets, two stated gates, deterministic report, and an exception path.
- Tasks 3–4 cover dynamic 893-dimensional resolution, all v5 assets, frozen Belief/KL anchor, 50-game/10-update/resume/league/curriculum smoke, and throughput evidence.
- Task 5 records but never executes the formal CUDA launch command and estimate.
- No encoder changes, S4 retraining, cloud activity, or long S5 training is included.
