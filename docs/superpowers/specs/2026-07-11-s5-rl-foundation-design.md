# S5 自博弈强化学习基础设计

## 目标与边界

落实 `strategy_S5_rl_spec.md` 的任务 1--7，交付可在云端启动正式训练的 S5 基础闭环。S5 从 S4 v1 的 Policy 初始化；S4 v1 Belief 在本阶段冻结、只读。此轮只完成实现、测试、本地小规模冒烟和云端打包，不启动长时间正式 RL。

训练目标是最大化每局由 S1 结算给学习者的最终净得分（按底分和封顶归一化）。中间奖励恒为零；胡牌不是 episode 终点，结算完成才是终点。

学习者输入始终是 S2 的退化观测编码和冻结 Belief 输出，绝不包含隐藏真值、其他玩家手牌或 Belief 的训练标签。对手使用完美观测。

## 架构

新增 `rl/` 包：

- `rollout.py`：运行一个学习者座位和三个对手座位的完整 S1 对局，收集 `(encoded_state, legal_mask, action, old_log_prob, value, reward, done, policy_version)`。采样前对非法动作 logit 置为负无穷；每局做零和与零非法动作校验。
- `reward.py`：从最终 S1 结算中抽取学习者净得分、执行归一化，并为整条轨迹写入仅末步非零的 reward。
- `models/value_net.py`：复用 S4 Policy 主干，增加标量价值头。S4 Policy 权重加载到共享主干；价值头随机初始化。
- `ppo_trainer.py`：GAE、归一化 advantage、PPO clipped objective、价值损失、熵、梯度裁剪和可衰减的 S4-reference KL。所有损失只在合法动作掩码内计算。
- `league.py`：维护 S3、贪心、随机、当前策略和历史快照；支持加权采样、快照序列化、容量淘汰、里程碑保留。候选快照须通过反遗忘 arena 门槛才可入池。
- `curriculum.py`：根据显式阶段配置选择学习者观测退化分布；阶段提升由最近 arena 指标触发。对手侧不使用退化配置。
- `train_rl.py`：配置化入口，协调 rollout、PPO、league、curriculum、周期性评测、结构化日志和完整 checkpoint 续训。

复用 `learning/eval/arena.py` 的对局与统计能力，新增 S5 适配器而非复制引擎规则。训练入口应可从本地归档 `training_artifacts/S4/v1_20260711_repaired_cuda/checkpoints/` 读取 S4 初始权重。

## 数据流与运行阶段

1. 加载冻结的 S4 Belief、S4 Policy 和训练配置；将 S3/贪心/随机注册进 league。
2. curriculum 选择学习者的退化档位；league 为三个对手座位采样对手。
3. rollout 使用 S1 完整跑局，学习者仅得到退化的 S2 编码及冻结 Belief 预测；终局后奖励模块写入归一化净得分。
4. PPO 对积累轨迹更新策略/价值网络；KL 系数按训练进度衰减，且记录 clip fraction、熵、KL、值损失和梯度范数。
5. 定期保存原子 checkpoint；checkpoint 包含模型、优化器、全局步数、随机状态、league、curriculum、KL 调度及指标历史。
6. 定期运行完美观测和残缺观测双赛道 arena，并将达到门槛的策略快照加入 league。

## 配置和默认策略

配置文件或 CLI 显式声明随机种子、设备、rollout 局数、并发数、PPO 批大小、clip/GAE/gamma、KL 起止系数、checkpoint 周期、league 配额、课程阈值和 arena 局数。默认训练对手权重遵循规格：最新策略 35%、历史快照 35%、S3 20%、贪心+随机 10%。

小规模本地冒烟使用少量局数和 CPU；云端配置面向单张 24 GB GPU，使用 `--device cuda`，并在打包前显式验证 CUDA 可用。正式长训的局数与计算预算只在云端启动时确定。

## 失败处理与健康监控

任一非法动作、零和失败、隐藏信息字段进入学习者特征、NaN/Inf、无法加载 S4 权重、或 checkpoint 不完整均立即失败并保留诊断 checkpoint。训练健康告警包括熵塌缩、KL 超限、价值损失发散以及相对 S3 的 arena 成绩回退；告警不悄悄吞掉，会写入日志和 checkpoint 元数据。

## 验证

单元测试覆盖：终局奖励与 episode 边界、合法动作掩码、GAE/PPO 数值、冻结 Belief 无梯度、轨迹无隐藏信息、零和、league 入池门槛及序列化、课程隔离和断点续训连续性。

集成冒烟覆盖：至少一个完整 rollout、一次 PPO 更新、一次 checkpoint 恢复、一次双赛道 arena；训练轨迹无非法动作。上云前还需构建独立 S5 包，验证包含冻结 S4 v1 检查点、入口、配置、依赖说明和 GPU 冒烟命令。

## 验收产物

`cloud_outputs/s5_*` 产生策略/价值 checkpoint、league 快照、课程状态、结构化训练指标、双赛道 arena 报告及专家行为审阅候选集。`docs/concepts.md` 说明 PPO、自博弈、终局奖励、league、课程和不完全信息约束；`S5_expert_behavior_checklist.md` 的八类场景生成可供人工审阅的对局材料。

## 非目标

本阶段不解冻或联合微调 Belief，不修改 S1/S2/S3 规则，不引入 MCTS/CFR/外部 RL 框架，也不以短冒烟成绩宣称 S5 已超过 S3。正式验收仍要求大样本双赛道 arena、反遗忘门槛和人工高手行为审阅。
