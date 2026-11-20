# 换三张策略训练与配对评估任务

## 目标

在完全相同的初始牌局上，对比旧版换三张策略与新版 `PrincipledSwapPolicy`，判断新版是否能够改善自身长期收益，并检查它是否会增加接收方收益。

本任务是离线配对模拟，不直接修改默认换牌策略。

## 对比策略

### `legacy`

调用：

```python
choose_swap_tiles_legacy()
```

### `principled`

调用：

```python
choose_swap_tiles()
```

新版主要考虑：

- 定缺花色倾向；
- 是否拆对子；
- 换出牌的连接价值；
- 换牌后向听数；
- 换牌后有效牌数量。

## 实验控制变量

每个 `seed` 使用同一副初始牌、同一庄家座位、同一目标玩家和同一套对手策略。两次模拟只有目标玩家的换三张动作不同：

```text
旧版换法 → 完整牌局
新版换法 → 完整牌局
```

其他三家使用 `RulePolicy`，因此结果是可复现的配对比较，而不是两批独立随机样本。

## 评估维度

每个 seed 记录：

- 两种策略是否选择相同换法；
- 目标玩家是否胡牌；
- 目标玩家最终得分；
- 接收目标玩家换出牌的玩家最终得分；
- 目标玩家相对接收方的净得分；
- 胡牌顺序；
- 换牌前按三门牌数排序后的结构，例如 `4/4/5`、`3/4/6`；
- 是否为庄家目标。

核心差值定义：

```text
score_delta = principled 自身得分 - legacy 自身得分
receiver_score_delta = principled 接收方得分 - legacy 接收方得分
net_score_delta = principled 净得分 - legacy 净得分
```

## 推荐运行方式

### 快速验证

```powershell
python tools/compare_swap_policies.py `
  --games 100 `
  --seed-start 987500 `
  --dealer 0 `
  --target 0 `
  --out outputs/swap_policy_comparison_smoke.json
```

### 分片运行

分片入口：

```text
tools/run_swap_shard.py
```

示例：

```powershell
python tools/run_swap_shard.py `
  --seed-start 100000 `
  --games 10000 `
  --dealer 0 `
  --target 1 `
  --out outputs/shards/swap_100000_d0_t1.json
```

参数说明：

```text
--seed-start：分片起始 seed
--games：分片内 seed 数量
--dealer：庄家座位，0～3
--target：目标玩家座位，0～3
--out：独立输出文件
```

每个分片会额外生成：

```text
<输出文件>.checkpoint
```

任务中断后重新执行相同命令，会跳过已经完成或已记录失败的 seed。不同分片必须使用不同输出路径。

### 庄家与闲家覆盖

建议至少运行四类目标：

```powershell
python tools/run_swap_shard.py --seed-start 100000 --games 10000 --dealer 0 --target 0 --out outputs/shards/dealer_t0.json
python tools/run_swap_shard.py --seed-start 110000 --games 10000 --dealer 0 --target 1 --out outputs/shards/nondealer_t1.json
python tools/run_swap_shard.py --seed-start 120000 --games 10000 --dealer 0 --target 2 --out outputs/shards/nondealer_t2.json
python tools/run_swap_shard.py --seed-start 130000 --games 10000 --dealer 0 --target 3 --out outputs/shards/nondealer_t3.json
```

如果要比较不同庄家座位，应为每组使用新的 seed 区间和独立输出文件。

## 输出与汇总

`compare_swap_policies.py` 输出完整 JSON，包含：

```text
same_action_rate
legacy_win_rate
principled_win_rate
average_score_delta
average_receiver_score_delta
average_net_score_delta
by_suit_count_bucket
rows
```

分片输出包含：

```text
schema_version
seed_start
seed_end
requested_games
completed_games
failed_games
dealer
target
policies
elapsed_seconds
rows
failures
```

汇总时应按以下维度分层：

```text
庄家 / 闲家
换牌前牌数结构
相同换法 / 不同换法
```

重点结构：

```text
4/4/5
4/4/6
4/5/5
3/4/6
3/5/5
```

## 结果判定

不能只看总平均值。每个结构至少报告：

- 样本数；
- 平均自身得分差；
- 平均接收方得分差；
- 平均净得分差；
- 新版得分更高比例；
- 配对差值的标准差和 95% 置信区间。

只有在样本量足够、关键分层的置信区间支持结论时，才考虑调整默认策略。

如果新版只在特定结构或庄闲角色中表现更好，应采用分层策略，而不是全局替换。

## 已知阶段性结论

此前小规模配对结果显示：

- 静态牌型指标中，新版减少拆对子、有效牌数量略好；
- 完整牌局收益没有稳定优于旧版；
- 庄家与闲家、不同牌数结构之间可能存在差异；
- 因此暂不应仅凭静态牌效把新版设为默认策略。

这些结论需要通过更大样本重新确认，不能把小样本趋势当成最终结论。

## 运行环境与性能基准

当前模拟是 Python 规则逻辑，适合 CPU 多进程，不适合直接搬到 GPU。单进程早期基准约为：

```text
0.57 秒 / seed
约 6,280 seed / 小时
```

建议执行顺序：

```text
本地 100～1,000 seed 冒烟
→ 本地多进程分片
→ 10,000～100,000 seed 稳定性验证
→ 必要时再使用云端 CPU 分片
```

## 注意事项

- 不要让多个 worker 写同一个输出文件；
- 不要混合不同 `dealer`、`target` 或策略版本的分片；
- 汇总前检查 seed 是否重复或缺失；
- 保留失败 seed 和异常信息；
- 任何策略修改都应先添加配对测试，再重新运行基准；
- 在结论中明确对手策略为 `RulePolicy`，不能直接等同真人策略。
