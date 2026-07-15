# S4 Split Policy Training Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 split policy 的 belief 输入、样本加权、训练循环、模型选择和关键决策评估，使训练指标能够反映真实决策能力。

**Architecture:** 数据构建层接受 `TileBelief` 注入并保留 prior 默认路径；训练层集中计算样本权重并维持单一模型/优化器跨 epoch 训练；评估层生成结构化分项报告；云训练编排层按 `game_id` 划分数据，训练 belief 后冻结注入 policy，并保存输入来源元数据。

**Tech Stack:** Python 3、PyTorch、pytest、现有 S2 状态编码与 S4 训练模块。

---

## 文件结构

- 修改 `learning/datasets/dataset_builder.py`：belief 注入和 policy 样本决策元数据。
- 修改 `state/tile_belief.py`：确保 learned-belief 推理跟随模型设备。
- 修改 `learning/training/train_policy.py`：样本权重、加权损失、多 epoch、early stopping、best state 和 checkpoint 元数据。
- 修改 `learning/eval/eval_policy.py`：总体、非强制、动作类别、响应场景和阶段指标。
- 修改 `policies/learned_policy.py`：显式 learned-belief 推理和 checkpoint 输入约束。
- 修改 `tools/cloud_train_s4.py`：按局划分、learned-belief 特征、训练配置、报告和 CLI。
- 修改 `tests/test_s4_dataset_builder.py`：belief 注入测试。
- 修改 `tests/test_s4_train_policy.py`：权重、加权损失、多 epoch、元数据和在线 belief 测试。
- 修改 `tests/test_s4_eval_policy_arena.py`：分项评估测试。
- 修改 `tests/test_s4_cloud_training_package.py`：完整 smoke 链路测试。

### Task 1: 可注入的 Policy Belief 输入

- [ ] 在 `tests/test_s4_dataset_builder.py` 添加 fake `TileBelief`，断言默认样本 belief source 为 prior，注入后 source 与 belief 编码区段变化。
- [ ] 运行 `python -m pytest tests/test_s4_dataset_builder.py -q`，确认新增测试失败。
- [ ] 修改 `build_policy_sample(record, config=None, *, belief=None)`，将 provider 传给 `with_prior_beliefs(degraded, belief)`；给 `PolicySample` 增加 `action_kind`、`legal_action_count`、`is_pong_pass_decision`，这些字段由清洗后的动作和合法动作集合派生。
- [ ] 修改 `LearnedBelief.infer_batch()`，从模型首个参数确定设备，并把 features 移至该设备。
- [ ] 运行数据构建测试，期望全部通过。

### Task 2: 样本权重和加权训练步

- [ ] 在 `tests/test_s4_train_policy.py` 添加权重测试：强制动作 `0.1`、弃牌 `1.5`、换三张 `2.0`、定缺 `1.5`、pong/pass 响应 `2.0`；添加加权交叉熵与手工张量结果一致测试。
- [ ] 运行目标测试，确认因缺少权重接口而失败。
- [ ] 扩展 `TrainPolicyConfig`，加入 `forced_action_weight=0.1`、`discard_weight=1.5`、`swap_three_weight=2.0`、`declare_missing_suit_weight=1.5`、`pong_pass_weight=2.0` 及正数校验。
- [ ] 给 `PolicyBatch` 增加 `sample_weights` 并在 `.to()` 中迁移；由 `policy_batch_from_samples(samples, config=None)` 计算权重。
- [ ] 将 `train_policy_step()` 改为逐样本 cross entropy 后按 `sum(w*loss)/sum(w)` 聚合，权重和非正时报错。
- [ ] 运行训练目标测试，期望全部通过。

### Task 3: 关键决策评估

- [ ] 在 `tests/test_s4_eval_policy_arena.py` 构造包含强制和非强制决策的样本，断言报告包含 `forced_samples`、`non_forced_samples`、`non_forced_accuracy`、`by_action_kind`、`by_phase` 和 `pong_pass_response`。
- [ ] 运行目标测试，确认报告字段缺失。
- [ ] 在 `eval_policy.py` 增加 `PolicySliceMetrics`，扩展 `PolicyEvalReport`；按预测结果与 `PolicySample` 元数据聚合各切片，空切片准确率使用 `None`。
- [ ] 使用批次或可控 chunk 评估，保留 illegal argmax 和 illegal probability mass。
- [ ] 运行评估测试，期望全部通过。

### Task 4: 多 Epoch 与 Best Checkpoint

- [ ] 在 `tests/test_s4_train_policy.py` 添加多 epoch 测试，断言 `epochs_trained >= 2`、存在 `best_epoch`、`history`，以及 best state 被恢复；添加非法空验证集与非法 patience 测试。
- [ ] 运行目标测试，确认缺少训练接口。
- [ ] 扩展 `TrainPolicyConfig`：`max_epochs=10`、`patience=3`、`min_delta=1e-4`。
- [ ] 保留 `train_policy_epoch()` 作为兼容的一轮入口；新增 `train_policy(train_samples, validation_samples, config)`，模型和优化器只创建一次，每 epoch 确定性重排。
- [ ] 每轮调用评估，以非强制准确率为首要指标、验证加权损失为平局判据；深拷贝 CPU best state，达到 patience 后停止并恢复。
- [ ] 扩展 `save_policy_checkpoint()` 的可选参数：`belief_metadata`、`split_summary`；保存 best epoch、历史和输入来源。
- [ ] 运行训练测试，期望全部通过。

### Task 5: 在线 Learned Belief 一致性

- [ ] 在 `tests/test_s4_train_policy.py` 添加测试：带 `belief_metadata.source=learned` 的 policy checkpoint 未提供 belief checkpoint 时初始化失败；提供 fake/真实 tiny belief checkpoint 后，`choose_action()` 在编码前填充 learned belief。
- [ ] 运行目标测试，确认当前 `LearnedPolicy` 忽略 belief 元数据。
- [ ] 修改 `LearnedPolicy.__init__(model_path, belief_model_path=None)`：历史 checkpoint 默认保持原输入；声明 learned 来源时强制提供配套 checkpoint并构造 `LearnedBelief`。
- [ ] 在 `choose_action()` 中，如配置 provider，则通过 `with_prior_beliefs(protocol_state, provider)` 填充 belief 后编码；未配置时不改变传入状态。
- [ ] 运行训练与 arena 测试，期望全部通过。

### Task 6: 云训练编排

- [ ] 修改 `tests/test_s4_cloud_training_package.py`，使用至少 3 局且每个 split 非空；断言 split game id 无交集、policy checkpoint belief 来源为 learned、训练 epoch/history 和新报告字段存在。
- [ ] 运行 smoke 测试，确认当前编排失败。
- [ ] 修改 `run_cloud_training()`：在 sample limit 后按 `game_id` 划分；各 split 分别构建 belief samples；训练 belief 后构造冻结 `LearnedBelief(model=belief_model)`；各 split 构建 policy samples。
- [ ] 若数据局数不足以形成非空 train/val/test，给出明确错误；`sample_limit` 不得破坏按局隔离，需按记录截断后再次校验。
- [ ] policy 调用 `train_policy(train, val, config)`，test 仅做最终评估；checkpoint 保存 learned belief 元数据与 split 摘要。
- [ ] 扩展 `CloudTrainingConfig` 和 CLI 的 epochs、patience、min-delta 与各样本权重参数。
- [ ] 扩展 JSON/Markdown 报告，展示 split、非强制准确率、关键动作指标、best epoch 和 early stopping。
- [ ] 运行 smoke 测试，期望全部通过。

### Task 7: 回归与 Smoke 验证

- [ ] 运行 `python -m pytest tests/test_s4_dataset_builder.py tests/test_s4_train_policy.py tests/test_s4_eval_policy_arena.py tests/test_s4_cloud_training_package.py -q`，修复所有失败。
- [ ] 运行 `python -m pytest -q`，确认完整回归通过。
- [ ] 运行小规模命令 `python tools/cloud_train_s4.py --output-dir cloud_outputs/s4_repair_smoke --games 3 --max-steps 120 --sample-limit 300 --batch-size 32 --hidden-size 32 --residual-blocks 1 --max-epochs 2 --patience 2 --device cpu`；确认生成两个 checkpoint 和报告，且 policy 元数据来源为 learned。
- [ ] 检查 `read_lints`，修复本次修改引入的诊断。
- [ ] 当前目录若仍无 `.git`，记录无法提交/推送；若恢复为 Git 仓库，则按用户偏好提交并推送当前分支。
