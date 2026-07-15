# S2 Protocol Spec (`s2.v4`)

S2 的统一状态协议用于把 S1 引擎、视觉识别、人工录入都转换为同一种玩家视角输入。协议核心目标是同时表达完整观测、残缺观测和带噪声观测，并严格区分“没有”和“不知道”。

## 可观测性包装

每个事实字段使用 `ObservedValue`：

```json
{
  "value": [],
  "status": "observed",
  "confidence": 1.0
}
```

`status` 取值：

- `observed`：确定事实，`confidence` 必须为 `1.0`。
- `estimated`：估计值或视觉识别结果，`confidence` 在 `[0, 1]`。
- `unknown`：未知，`value` 必须为 `null`，`confidence` 必须为 `0.0`。

空列表、`0`、`false` 都可以是 `observed` 值，不能用来表示未知。

## 顶层字段

- `version`：固定为 `s2.v4`。
- `perspective_player`：当前视角玩家的绝对座位号。
- `phase`：牌局阶段，如 `swap_three`、`declare_void`、`play`。
- `current_player`：当前行动玩家的绝对座位号。
- `current_player_relative`：当前行动玩家相对视角玩家的位置，`0=自己, 1=下家, 2=对家, 3=上家`。
- `facts`：事实层，只记录可见事实与可观测性状态。
- `statistics`：统计层，保存可验证的硬编码统计。
- `beliefs`：信念层，保存先验或学习模型输出。
- `legal_actions`：合法动作列表；后续由 `legality.py` 填充。
- `observation_start`：观测起点，`0` 表示全程观测。
- `rule_config`：规则配置，如 `base_score`、`max_fan`、`self_draw_mode`。

## `facts`

当前地基批次包含：

- `players`：按相对位置排序的 4 家信息。
  - `player_id`：绝对座位号。
  - `relative_position`：相对位置。
  - `concealed_hand`：自己暗手牌为 `observed`；未胡对手暗手牌为 `unknown`；已胡玩家亮牌为 `observed`。
  - `hand_count`：暗手牌张数。
  - `melds`：公开/自家副露，包含 `kind`、`tiles`、`exposed`、`from_player`。
  - `rivers`：弃牌序列。
  - `void_suit`：定缺花色，可为 `null`。
  - `won`：是否已胡。
  - `passed_hu_lock` / `passed_fan`：当前批次仅对自己暴露真实值，对他人填 `observed null`，后续可按部署需要调整为 `unknown`。
- `dealer` / `dealer_relative_position` / `is_dealer`。
- `wall_count` / `is_last_tile`。
- `pending_discard` / `pending_rob_kong`。
- `exchange_tracking`：换三张追踪，当前 from_engine 输出自己的换出与方向。
- `event_history`：事件历史，当前为空列表骨架。
- `revealed_win_hands`：已胡玩家亮出的完整手牌。
- `seen_counts`：27 维已见牌计数，包含自己手牌、弃牌、所有副露、已胡玩家亮牌。

## `statistics`

当前地基批次包含：

- `remaining_tile_counts`：`4 - seen_counts` 的 27 维计数，下限截断到 `0`。
- `unknown_pool_breakdown`：未知区拆分，包含 `wall` 与未胡对手暗手牌张数。
- `own_hand_analysis`、`candidate_action_features`、`dingque_constraints`：后续任务填充。

## `beliefs`

当前地基批次只固定结构：

- `source`：`prior` 或后续的 `learned`。
- `tile_location_beliefs`。
- `opponent_tenpai_beliefs`。
- `discard_danger`。

## 示例：完整观测 from_engine

```json
{
  "version": "s2.v4",
  "perspective_player": 2,
  "phase": {"value": "play", "status": "observed", "confidence": 1.0},
  "current_player_relative": {"value": 0, "status": "observed", "confidence": 1.0},
  "observation_start": {"value": 0, "status": "observed", "confidence": 1.0},
  "facts": {
    "players": {
      "status": "observed",
      "confidence": 1.0,
      "value": [
        {"player_id": 2, "relative_position": 0, "concealed_hand": {"value": ["7W", "8W", "9W"], "status": "observed", "confidence": 1.0}},
        {"player_id": 3, "relative_position": 1, "concealed_hand": {"value": null, "status": "unknown", "confidence": 0.0}}
      ]
    }
  }
}
```

## 示例：中途接入

```json
{
  "observation_start": {"value": 37, "status": "observed", "confidence": 1.0},
  "facts": {
    "exchange_tracking": {"value": null, "status": "unknown", "confidence": 0.0},
    "event_history": {"value": null, "status": "unknown", "confidence": 0.0}
  }
}
```

含义：不是“没有换三张/没有历史”，而是接入前信息未知。

## 示例：视觉噪声

```json
{
  "facts": {
    "players": {
      "status": "estimated",
      "confidence": 0.82,
      "value": [
        {"player_id": 1, "relative_position": 1, "rivers": {"value": ["3T", "4T"], "status": "estimated", "confidence": 0.82}}
      ]
    }
  }
}
```

含义：视觉识别认为弃牌为 `3T, 4T`，但不作为确定事实。

## 可见性红线

玩家视角产物不得包含未胡对手暗手牌。已胡玩家亮牌属于公开事实，应写入 `revealed_win_hands` 并计入 `seen_counts`。

## 端到端链路

S2 当前支持三类入口落到同一协议：

- `from_engine`：S1 完美观测转玩家视角，用于训练、回归和 oracle 对账。
- `DegradationPipeline`：把完美观测退化为中途接入、字段缺失或视觉噪声状态，用于 sim2real 训练。
- `from_vision_events`：从视觉事件/快照直接构造 `s2.v4`，并通过调和报告记录牌数矛盾。

三条链路都应继续通过 `compute_tile_statistics`、`PriorBelief`、`legal_actions`、`encode_state`。回归矩阵见 `docs/s2_regression_matrix.md`。

