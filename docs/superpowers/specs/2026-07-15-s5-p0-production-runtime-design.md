# S5 P0 生产训练运行时设计

## 目标

把现有 S5 基础闭环升级为可用于正式训练的运行时，完成四项 P0：真实 rollout adapter、League 对手真实接线、生产双赛道 Arena、CPU 多进程并行 rollout。

## 已确认的生产约束

- 正式 rollout 默认使用 8 个 Windows `spawn` CPU worker，每轮恰好完成 1024 局真实 S1 对局。
- worker 独立加载只读 CPU 策略与冻结 S4 belief，`eval()`、`requires_grad=False`，不得初始化 CUDA。
- 主进程独占训练模型和 optimizer，并在 GPU 可用时执行 PPO。
- Arena 对每个现存 League entry 分别运行完美与退化赛道，每赛道恰好 1000 局。
- 任一对手、任一赛道未完成或未达门槛，候选均不得晋级；`current` 与 `history` 同时保持不变。
- 单进程和小预算参数继续可配，用于测试与诊断；正式模式采用上述默认值。

## P0-1：生产真实 Rollout Adapter

新增模块级生产适配器，负责从任务描述构建完整 S1 对局，而不是在云入口内保留闭包。

每局任务固定携带 `round_id`、`game_index`、`policy_generation`、模型 checksum、环境 seed、学习者采样 seed、对手抽样 seed、学习者座位、League 状态和课程画像。seed 仅由轮次 seed 与 `game_index` 推导，不受 worker 编号、完成顺序和重试次数影响。

学习者路径为退化 S2 观测、冻结 belief 填充、编码、合法动作掩码、策略采样；三名对手使用完美观测。轨迹只包含学习者可见特征、合法掩码、动作、旧 log-prob、value、终局奖励、done 与 policy generation。奖励只在 S1 完整结算后的最后一步非零。

适配器必须拒绝空轨迹、非法动作、非终局 episode、非零和结算、NaN/Inf、隐藏字段泄漏及 generation/checksum 不一致。

## P0-2：League 对手真实接线

`OpponentLeague` 继续只持有可序列化元数据；新增独立 resolver 将 `OpponentEntry` 转为真实策略：

- `S3` -> `RulePolicy`
- `GREEDY` -> `GreedyPolicy`
- `RANDOM` -> 带任务 seed 的 `RandomPolicy`
- `CURRENT` / `SNAPSHOT` -> 从 `entry.snapshot.checkpoint_path` 加载冻结 CPU 模型

`CURRENT` 不得引用正在训练或待评估 candidate 的内存模型。模型 checkpoint 加载前校验文件存在与 SHA-256；加载后进入 `eval()` 并冻结梯度。每局三名对手分别按 League 权重真实抽样和实例化，同桌允许混搭。

rollout 与 Arena 均从训练编排器传入当前恢复后的 League/Curriculum 状态，禁止闭包捕获冷启动副本，确保 checkpoint 恢复后的抽样和退化课程连续。

## P0-3：生产双赛道 Arena 与严格晋级

候选对 `league.entries` 的精确集合逐项评估；不允许缺项、额外项或混入另一 generation。每个 `DualArenaMetrics` 除胜率外记录两赛道完成局数、非法动作数与零和失败数。

每个对手的完美和退化赛道各完成 1000 局后才形成有效指标。任一赛道局数不足、存在非法动作、存在零和失败、指标非有限或胜率低于门槛，候选均拒绝。

晋级是一个逻辑事务：先完整评估，再检查容量与门禁，最后同时将旧 current 纳入 history 并将 candidate 设为新 current。拒绝或异常时二者均不变化。不可变 candidate 文件可保留用于审计，但不能进入可采样 League。

## P0-4：CPU 多进程并行 Rollout

主进程为每轮生成 1024 个不可变任务，使用 `multiprocessing.get_context("spawn")` 启动默认 8 个 worker。worker 初始化一次冻结 S4 belief，并按 generation/checksum 加载轮次策略快照；每次返回完整单局结果。

主进程按 `game_index` 聚合结果，拒绝重复、缺失、越界、旧 generation 或错误 checksum。超时/崩溃时终止并重建 worker，以完全相同的任务和 seed 重试。超过重试上限时抛出整轮失败，不向 PPO 暴露部分轨迹。只有验证恰好 1024 个完整局后才按 `game_index` 排序并构建 PPO batch。

## 原子性与恢复

`latest.pt` 是唯一 committed 训练状态。rollout 期间不推进 `global_step` 或 seed；PPO、Arena、League/Curriculum 变更全部成功后才原子替换 latest checkpoint。中途失败时诊断 checkpoint 只用于调查，不作为自动恢复点。重启从旧 latest 重跑同一轮，因此任务 seed、League 抽样和课程画像一致。

checkpoint 配置持久化 rollout seed 推导版本、下一轮 ID、policy generation、worker 参数、League 与 Curriculum。正式云入口暴露 `resume_checkpoint`，并把恢复后的状态传给生产适配器。

## 配置默认值

- `rollout.workers = 8`
- `rollout.games_per_round = 1024`
- `rollout.inference_device = "cpu"`
- `rollout.start_method = "spawn"`
- `rollout.torch_threads_per_worker = 1`
- `rollout.max_game_steps = 1000`
- `rollout.task_timeout_seconds = 120`
- `rollout.max_task_retries = 2`
- `arena.games_per_track = 1000`
- `arena.require_finished_games = true`
- `arena.require_zero_illegal_actions = true`
- `arena.require_zero_sum = true`

smoke 模式保持单进程、4 局 rollout、每赛道 4 局，不作为牌力或吞吐证据。

## 验收

1. Windows spawn 下 8 worker 完成 1024 局，无非法动作、零和失败或隐藏信息泄漏。
2. workers=1 与 workers=8 在同一 seed 下按 `game_index` 排序后的轨迹摘要一致。
3. worker 超时以同任务重试；重试耗尽时整轮失败且 PPO 未执行。
4. 三名对手全部来自恢复后的 League 抽样，并加载其对应真实策略/checkpoint。
5. candidate 对每个 League entry 完成完美/退化各 1000 局；任一失败时 current/history 字节级状态不变。
6. 中断恢复后下一轮 seed、generation、League、Curriculum 与未中断运行一致。
7. GPU 环境中只有主进程训练模型使用 CUDA，worker 模型和 belief 均留在 CPU。
