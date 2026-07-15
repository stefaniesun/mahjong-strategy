# S4 本地 CPU 冒烟验收报告（2026-07-10）

## 产物

- `cloud_outputs/s4_split_retrain_20260710/checkpoints/belief_s4_split.pt`
- `cloud_outputs/s4_split_retrain_20260710/checkpoints/policy_s4_split.pt`
- 用途：900/50/50 按局无泄漏泛化验证模型，不是全量部署模型。

## 兼容性

- 本地环境：Python 3.14.3、PyTorch 2.12.1+cpu
- checkpoint 原训练设备：CUDA
- CPU 反序列化：通过
- 编码器版本：`s2.v4.encoder.v2`，与当前代码一致
- 输入维度：263，与当前编码器一致
- 动作空间：637，与当前固定动作空间一致
- `LearnedBelief` 单次 CPU 推理：通过
- `LearnedPolicy` 单次 CPU 推理：通过

## 回归测试

- 命令：`python -m pytest tests/test_s4_train_policy.py tests/test_s4_eval_policy_arena.py tests/test_s2_tile_belief.py -q`
- 结果：17 passed，2 skipped（CUDA 测试在 CPU 环境跳过）

## Arena 冒烟

### 固定座位

- 配置：split policy 位于座位 0，另外三个座位为 `RulePolicy`，100 局
- 未完成：0
- 非法动作：0
- 零和异常：0
- split policy 平均分：-4.00
- 95% CI 半宽：0.7614

### 四座轮换

每个座位 50 局，共 200 局：

| 座位 | 平均分 | 95% CI 半宽 |
|---:|---:|---:|
| 0 | -3.74 | 1.0489 |
| 1 | -3.92 | 1.2805 |
| 2 | -2.78 | 1.1870 |
| 3 | -3.84 | 1.2158 |

汇总：

- split policy 平均分：-3.57
- 95% CI 半宽：0.5920
- 非法动作：0
- 未完成：0
- 零和异常：0

### S3 自对照

四个 `RulePolicy` 运行 200 局，各座平均分：

- 座位 0：0.715
- 座位 1：-0.285
- 座位 2：-0.545
- 座位 3：0.115

## CPU 性能

- policy 单点推理：约 4.23 ms/次（1000 次重复测量）

## 结论

checkpoint 与本地代码完全兼容，推理和对局链路稳定，未出现非法动作、未完成牌局或零和异常。

但 split policy 在 200 局四座轮换中平均分为 -3.57，明显弱于 S3 规则策略。离线 top-1 一致率约 83% 没有转化为接近 S3 的在线牌力。当前模型不应直接作为 S5 冷启动的正式基线，也不建议立即扩大到 10000 局验收。

下一步应先对约 17% 的模仿错误按阶段和动作类型分桶，并检查训练 policy 输入所使用的 belief 来源与 arena 推理输入是否一致；修复或重训后再运行大规模 arena。
