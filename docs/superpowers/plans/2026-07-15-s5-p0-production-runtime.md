# S5 P0 Production Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 S5 升级为 8 个 CPU worker 每轮采集 1024 局、主进程 GPU PPO，并以全 League 双赛道各 1000 局严格控制晋级的生产运行时。

**Architecture:** 把云入口中的真实环境闭包拆为模块级、可 spawn 的生产适配器和对手 resolver；主进程发布不可变 generation 快照并原子收集整轮结果。Arena 返回带完成性与完整性计数的双赛道指标，League 用单一原子 promotion 操作同时更新 current/history。

**Tech Stack:** Python 3.10+、PyTorch、`multiprocessing` spawn、pytest、现有 S1/S2/S3/S4 模块。

---

## 文件边界

- Create: `rl/opponent_resolver.py`：校验并加载 League entry 对应的真实 CPU 策略。
- Create: `rl/production_adapter.py`：模块级真实单局 rollout、Arena 和 worker runtime 依赖。
- Create: `rl/parallel_rollout.py`：任务 schema、seed 计划、spawn worker 生命周期与整轮原子聚合。
- Modify: `rl/league.py`：完整 Arena 指标和原子 promotion。
- Modify: `rl/train_rl.py`：运行时状态传递、完整轮次校验、严格晋级与 committed checkpoint 顺序。
- Modify: `tools/cloud_train_s5.py`：生产默认配置、恢复 CLI，并改用新模块。
- Modify: `S5_CLOUD_TRAINING_README.md`：正式参数和恢复命令。
- Test: `tests/test_s5_opponent_resolver.py`、`tests/test_s5_parallel_rollout.py`、`tests/test_s5_production_adapter.py`，并扩展现有 League、训练与云入口测试。

### Task 1: 严格双赛道指标与原子晋级

**Files:**
- Modify: `rl/league.py:54-72,183-209,270-285`
- Modify: `rl/train_rl.py:321-329,594-603`
- Modify: `tests/test_s5_league.py`
- Modify: `tests/test_s5_train_smoke.py:234-244`

- [ ] **Step 1: 写失败测试**

新增 `test_dual_arena_metrics_require_exact_completed_game_counts_and_clean_invariants`，断言局数不足 1000、非法动作非零、零和失败非零均拒绝；新增 `test_rejected_candidate_leaves_current_and_history_unchanged`，在完美或退化赛道低于门槛时比较 promotion 前后的 `league.to_json()` 完全相同；新增 exact-key 测试拒绝缺失和额外 opponent key。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_s5_league.py tests/test_s5_train_smoke.py -q`
Expected: FAIL，现有指标无局数/异常字段且训练仍无条件更新 current。

- [ ] **Step 3: 实现最小严格门禁**

扩展 `DualArenaMetrics` 为 `perfect_win_rate`、`degraded_win_rate`、`perfect_games`、`degraded_games`、`illegal_actions`、`zero_sum_failures`；在 `LeagueConfig` 增加 `arena_games_per_track=1000`。新增 `OpponentLeague.promote_candidate(candidate, metrics, milestone=False) -> bool`，先验证 exact keys、两赛道局数和 invariant，再在通过时一次性提交 history/current；删除训练层分离的 `admit_snapshot()+set_current_policy()`。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_s5_league.py tests/test_s5_train_smoke.py tests/test_s5_checkpoint.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

Run: `git add rl/league.py rl/train_rl.py tests/test_s5_league.py tests/test_s5_train_smoke.py && git commit -m "fix: make S5 promotion dual-track atomic"`

### Task 2: League 真实对手 Resolver

**Files:**
- Create: `rl/opponent_resolver.py`
- Create: `tests/test_s5_opponent_resolver.py`
- Modify: `tools/cloud_train_s5.py:319-470`

- [ ] **Step 1: 写失败测试**

覆盖五种 `OpponentKind`；对 CURRENT/SNAPSHOT 断言从 `entry.snapshot.checkpoint_path` 加载而非 candidate 内存模型；校验 SHA-256 不匹配、缺文件和无 state dict 均失败；断言模型在 CPU、eval 且全部参数 `requires_grad=False`；对 Random 断言 seed 可复现。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_s5_opponent_resolver.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现 resolver**

实现模块级 `ModelPolicy` 和 `OpponentResolver.resolve(entry, *, seed)`。resolver 接收模型工厂、冻结 belief provider 和 checkpoint loader；CURRENT 与 SNAPSHOT 统一走 snapshot 元数据及 checksum 校验，固定基线直接构造。移除 `_formal_dependencies()` 内部的 `ModelPolicy/model_opponent`。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_s5_opponent_resolver.py tests/test_s5_league.py tests/test_s5_cloud_package.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

Run: `git add rl/opponent_resolver.py tools/cloud_train_s5.py tests/test_s5_opponent_resolver.py && git commit -m "feat: resolve real S5 league opponents"`

### Task 3: 模块级真实生产 Adapter

**Files:**
- Create: `rl/production_adapter.py`
- Create: `tests/test_s5_production_adapter.py`
- Modify: `rl/train_rl.py:520-575`
- Modify: `tools/cloud_train_s5.py:319-470`

- [ ] **Step 1: 写失败测试**

新增真实短局测试，断言每局一个学习者加三个由 League 抽样的真实对手、学习者座位轮转、对手观测不退化、终局才写非零奖励、policy generation 贯穿全部 step；恢复测试断言 adapter 使用传入的 restored League/Curriculum，而非构造时冷启动副本。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_s5_production_adapter.py -q`
Expected: FAIL，模块不存在或闭包仍捕获旧状态。

- [ ] **Step 3: 实现生产 adapter**

定义 `RolloutRuntimeState(league, curriculum, policy_generation, policy_checksum)` 和模块级 `run_production_game(task, runtime)`、`run_production_arena(candidate, entry, runtime, games_per_track)`。把 `S5TrainingDependencies.rollout_factory`/`arena_evaluator` 调整为显式接收当前 runtime state；Arena 将真实完成局数与 invariant 计数写入 `DualArenaMetrics`。

- [ ] **Step 4: 运行测试**

Run: `python -m pytest tests/test_s5_production_adapter.py tests/test_s5_rollout.py tests/test_s5_train_smoke.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

Run: `git add rl/production_adapter.py rl/train_rl.py tools/cloud_train_s5.py tests/test_s5_production_adapter.py && git commit -m "feat: add S5 production runtime adapter"`

### Task 4: Windows Spawn 并行完整轮次

**Files:**
- Create: `rl/parallel_rollout.py`
- Create: `tests/test_s5_parallel_rollout.py`
- Modify: `rl/train_rl.py:569-578`

- [ ] **Step 1: 写失败测试**

覆盖默认 8 worker/1024 局、模块级 worker 可 pickle、workers=1/8 seed 计划一致、乱序结果按 game index 合并、超时同任务重试、重试耗尽整轮失败、重复/缺失/stale generation/checksum 被拒绝、worker CPU/eval/frozen 且不触碰 CUDA。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_s5_parallel_rollout.py -q`
Expected: FAIL，模块不存在。

- [ ] **Step 3: 实现并行 collector**

定义不可变 `RolloutTask`、`RolloutGameResult`、`RolloutRoundConfig` 与 `RolloutRoundError`。实现版本化 seed 派生；用 `multiprocessing.get_context("spawn")` 和模块级 `worker_main`；worker 初始化 CPU 模型/belief；主进程执行超时终止、同任务重试、exact-set 与 generation/checksum 校验，并仅在完整后返回排序轨迹。

- [ ] **Step 4: 接入训练原子边界**

`run_s5_training()` 在 rollout 完整成功前不得调用 `_batch_from_steps` 或 `ppo_update`；失败时 `global_step`、下一 seed、League 与 Curriculum 保持 committed 状态。下一轮从新的 generation 快照启动，不允许一轮混合参数。

- [ ] **Step 5: 运行测试**

Run: `python -m pytest tests/test_s5_parallel_rollout.py tests/test_s5_train_smoke.py tests/test_s5_checkpoint.py -q`
Expected: PASS。

- [ ] **Step 6: 提交**

Run: `git add rl/parallel_rollout.py rl/train_rl.py tests/test_s5_parallel_rollout.py tests/test_s5_train_smoke.py && git commit -m "feat: add atomic multiprocess S5 rollout"`

### Task 5: 正式配置、恢复与云入口

**Files:**
- Modify: `tools/cloud_train_s5.py:33-60,509-535,564-586`
- Modify: `S5_CLOUD_TRAINING_README.md`
- Modify: `tests/test_s5_cloud_package.py`
- Modify: `tests/test_s5_checkpoint.py`

- [ ] **Step 1: 写失败测试**

断言 train 模式默认 workers=8、episodes=1024、arena games=1000、spawn；smoke 仍为 workers=1 和 4/4；CLI 支持 worker、超时、重试和 `--resume-checkpoint`；恢复后下一轮 seed/generation/League/Curriculum 与不中断运行一致。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_s5_cloud_package.py tests/test_s5_checkpoint.py -q`
Expected: FAIL，正式默认值和恢复参数尚不存在。

- [ ] **Step 3: 实现配置与恢复接线**

为 `S5CloudRunConfig` 增加 `workers`、`worker_start_method`、`task_timeout_seconds`、`max_task_retries`、`resume_checkpoint`；按 mode 解析默认值，向 `S5TrainingConfig` 和生产 adapter 传递。checkpoint 持久化 seed 派生版本、round id、policy generation/checksum；README 给出正式启动和恢复命令。

- [ ] **Step 4: 全量验证**

Run: `python -m pytest -q`
Expected: PASS。

Run: `python tools/cloud_train_s5.py --mode smoke --device cpu --output-dir cloud_outputs/s5_p0_smoke`
Expected: exit 0，生成可校验 report、latest checkpoint 与 manifest。

- [ ] **Step 5: 提交并推送**

Run: `git add tools/cloud_train_s5.py S5_CLOUD_TRAINING_README.md tests/test_s5_cloud_package.py tests/test_s5_checkpoint.py docs/superpowers/specs/2026-07-15-s5-p0-production-runtime-design.md docs/superpowers/plans/2026-07-15-s5-p0-production-runtime.md && git commit -m "feat: configure S5 production rollout runtime"`

Run: `git push`
Expected: 当前分支推送成功。

## 自审结论

- 四个 P0 均有独立实现边界和验收测试。
- 严格晋级覆盖全池、双赛道、完成局数和 invariant，拒绝时 current/history 同时不变。
- Windows spawn、CPU/GPU 隔离、seed 确定性、generation/checksum、超时重试和整轮原子失败均有明确测试。
- 恢复语义以 committed latest checkpoint 为准，避免半轮或半次晋级状态。
- 未引入新 RL 框架，也未修改 S1/S2/S3 规则。
