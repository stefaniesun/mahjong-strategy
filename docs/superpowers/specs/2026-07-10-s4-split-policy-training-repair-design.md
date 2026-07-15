# S4 Split Policy 训练链路修复设计

## 背景

当前 split policy 的 held-out top-1 约为 83%，但约 59% 的样本只有一个合法动作。去除这类强制动作后，非强制决策准确率约为 58%，其中弃牌、换三张和碰牌决策明显偏弱。

现有训练链路还存在以下确定性缺口：

1. `build_policy_sample()` 固定使用 `PriorBelief`，没有消费已训练的 `LearnedBelief`。
2. policy 仅完整遍历训练集一次，没有多 epoch、验证集 early stopping 或 best checkpoint。
3. 损失由大量强制动作和 `pass` 样本主导。
4. 评估只强调总体 top-1，不能反映关键决策质量。
5. policy 训练与在线推理缺少显式、统一的 belief 注入机制。

## 目标

本次修复应实现：

- policy 训练和在线推理均可使用同一冻结的 `LearnedBelief`。
- 训练、验证和测试按 `game_id` 隔离。
- policy 支持逐样本加权、多 epoch、early stopping 和 best checkpoint。
- 报告总体与关键决策指标，避免强制动作虚高准确率。
- 保持旧调用和旧 checkpoint 的兼容性。

本次不拆分多输出头，也不启动正式大规模云训练；完成后只执行单元测试和小规模 smoke training。

## 数据管线

### Belief 注入

`build_policy_sample()` 增加可选 belief provider 参数，其接口遵循现有 `TileBelief`：

- 未传入时继续使用 `PriorBelief`，保持旧行为。
- 传入冻结的 `LearnedBelief` 时，用其推理结果填充状态 belief 区段后再编码。
- 复用现有状态填充函数，不复制 belief 到状态的逻辑。

云训练顺序调整为：

1. 按 `game_id` 划分原始决策记录。
2. 使用训练集训练 belief，并通过验证集选择最佳 belief checkpoint。
3. 冻结最佳 belief 模型。
4. 分别为 policy 的 train、validation、test 记录生成 learned-belief 特征。
5. 使用 train 训练 policy，validation 选择 best checkpoint，test 仅用于最终报告。

### 输入一致性

`LearnedPolicy` 支持显式加载 belief checkpoint，arena 和训练报告调用方必须使用与 policy 训练配套的 belief checkpoint。未配置时保留 prior-belief 回退，以兼容历史模型。

policy checkpoint 元数据记录 belief 来源和配套 checkpoint 标识，使评估能检测明显的输入错配。

## 样本权重与损失

### 基础权重

- 唯一合法动作：`0.1`
- 其他非强制动作：`1.0`

### 关键动作乘数

- 弃牌：`1.5`
- 换三张：`2.0`
- 定缺：`1.5`
- `pong/pass` 响应决策：`2.0`

最终样本权重为基础权重与适用动作乘数的乘积。所有权重均通过训练配置开放，并具有上述默认值。

`pong/pass` 响应场景通过合法动作集合识别：当同一决策点同时允许 `pong` 与 `pass` 时，对该样本应用响应决策乘数，而不是对所有普通 `pass` 样本加权。

训练损失采用逐样本 masked cross-entropy：

\[
L = \frac{\sum_i w_i \ell_i}{\sum_i w_i}
\]

若批次权重和为零，应显式报错；默认配置不会产生该情况。

## 多轮训练与模型选择

新增 policy 训练配置：

- `max_epochs`
- `patience`
- `learning_rate`
- `batch_size`
- `min_delta`
- 各类样本权重

每个 epoch 后执行验证。模型选择规则为：

1. 首要指标：非强制动作准确率最大。
2. 若首要指标改善小于 `min_delta`，以验证加权损失更低者为优。
3. 连续 `patience` 个 epoch 无改善时停止。
4. 最终恢复并保存最佳 epoch 的模型，而不是最后一轮模型。

checkpoint 保存：

- 模型参数与模型配置
- 编码器版本
- 训练配置
- best epoch
- best validation metrics
- belief 来源及配套 checkpoint 标识
- 数据 split 摘要

## 评估与报告

评估结果至少包含：

- 总体准确率与样本数
- 强制动作占比
- 非强制动作准确率与样本数
- 加权验证损失
- 按阶段统计
- 按动作类别统计：弃牌、换三张、定缺、`pong`、`pass`、胡牌等
- `pong/pass` 响应场景准确率
- 各类别训练样本数和总权重
- 主要动作类别混淆
- best epoch、实际训练 epoch 和 early-stopping 原因

总体准确率保留用于兼容，但不能再作为唯一验收指标。

## 错误处理与兼容性

- 未提供 belief provider 时继续构建 prior-belief policy 样本。
- 旧 policy checkpoint 仍可加载；缺少 belief 元数据时按 prior-belief 历史模型处理。
- belief checkpoint 的编码器版本或输入维度不匹配时立即报错。
- 配套 learned-belief checkpoint 缺失时，不允许静默改用 prior 进行正式 learned-belief 评估。
- 空训练集、空验证集、非法权重和权重和为零均给出明确错误。

## 测试策略

### 单元测试

- 默认 policy sample 仍使用 prior belief。
- 注入 learned/fake belief 后，编码中的 belief 区段发生预期变化。
- train/validation/test 不共享 `game_id`。
- 唯一合法动作和各关键动作权重计算正确。
- `pong/pass` 只在对应合法动作集合中获得加权。
- 加权交叉熵与手工计算结果一致。
- 多 epoch 训练保持同一模型和优化器状态。
- early stopping 能恢复 best epoch。
- 分项评估正确排除强制动作。
- 旧 checkpoint 兼容加载，新 checkpoint 元数据完整。
- 在线 policy 能使用显式 learned belief。

### 集成验证

使用小规模固定种子数据执行：

1. belief 训练与最佳 checkpoint 选择。
2. learned-belief policy 样本构建。
3. 多 epoch policy smoke training。
4. 报告字段和 checkpoint 元数据校验。
5. 小规模四座轮换 arena，确认无非法动作及 belief 输入错配。

正式 10,000 局重训不属于本次代码修改范围。

## 验收标准

- 所有现有测试与新增测试通过。
- smoke training 至少运行两个 epoch，并能选择或恢复 best checkpoint。
- 报告明确展示强制动作占比、非强制准确率和关键动作准确率。
- policy checkpoint 明确记录 belief 来源。
- learned-belief policy 在训练和 arena 中使用相同 belief 推理路径。
- 默认 prior-belief 兼容路径不被破坏。
