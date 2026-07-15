# S2 回归矩阵

S2 的目标不是单个模块可运行，而是三条输入链路都能落到同一个 `s2.v4` 玩家视角协议，并能继续进入统计、belief、合法动作和编码层。

## 链路矩阵

| 链路 | 入口 | 覆盖测试 | 验收点 |
| --- | --- | --- | --- |
| 完美观测链路 | `state.adapters.from_engine.from_engine` | `tests/test_s2_from_engine.py`, `tests/test_s2_end_to_end.py` | 对手暗手牌隐藏；自己手牌、牌河、副露、已胡亮牌进入 `seen_counts`；协议可 round-trip；编码 shape 固定 |
| 退化观测链路 | `state.observation_degradation.DegradationPipeline` | `tests/test_s2_observation_degradation.py`, `tests/test_s2_end_to_end.py` | 中途接入、字段缺失、视觉噪声能保持协议合法；软计数和 unknown 标志不丢；合法动作可标记 `conditionally_legal` |
| 模拟视觉链路 | `state.adapters.from_vision.from_vision_events` | `tests/test_s2_from_vision.py`, `tests/test_s2_end_to_end.py` | 事件/快照可转换为 `s2.v4`；置信度进入 estimated 字段；牌数矛盾触发调和报告；后续统计、belief、编码可运行 |
| 合法动作链路 | `state.legality.legal_actions`, `state.action_space.legal_mask` | `tests/test_s2_legality_action_space.py`, `tests/test_s2_end_to_end.py` | 训练/部署同源；完美观测下与 S1 引擎手工交叉验证；unknown 条件下使用 conditional action |
| belief 标签链路 | `state.tile_belief.generate_belief_labels` | `tests/test_s2_tile_belief.py`, `tests/test_s2_end_to_end.py` | 标签可使用 oracle；协议输入不得泄露对手暗手牌；prior belief 可在退化状态上回填 |
| 编码链路 | `state.encoder.encode_state` | `tests/test_s2_encoder.py`, `tests/test_s2_end_to_end.py` | 定长、确定性、unknown-aware；相同可见信息必须相同编码 |

## 当前自动化命令

```powershell
python -m pytest tests/test_s2_protocol.py tests/test_s2_from_engine.py tests/test_s2_legality_action_space.py tests/test_s2_tile_counting.py tests/test_s2_observation_degradation.py tests/test_s2_unknown_aware_features.py tests/test_s2_tile_belief.py tests/test_s2_encoder.py tests/test_s2_from_vision.py tests/test_s2_end_to_end.py -q
python -m pytest -q
```

## 红线

1. 玩家视角不得包含未胡对手暗手牌。
2. `unknown` 不得当作 `0`、`false` 或空列表处理。
3. 统计层只能放可验证计算；无 oracle 的行为判断必须留给 belief 或后续策略层。
4. 标签生成器可以用上帝视角，但其输出不能作为玩家输入字段。
5. 编码器必须定长、确定性，并显式编码 unknown/estimated/confidence。

## 未自动化的大规模验收

当前测试是单元与小规模端到端回归。规格中提到的 ≥100 局退化链路、≥1000 局 legal mask 交叉验证、≥10000 决策点训练样本文件属于后续批量验收，应在生成稳定 replay/采样工具后补充为慢测试或离线 CI 任务。
