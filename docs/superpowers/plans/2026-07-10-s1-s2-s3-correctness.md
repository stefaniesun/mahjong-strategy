# S1–S3 Correctness Remediation Implementation Plan

> **执行要求：** 使用 `executing-plans` 或 `subagent-driven-development` 按任务执行；每个行为修改都遵循红—绿—重构。三个阶段分别验证。只有当前工作区具备有效 Git 仓库与远端时，才按阶段提交并推送。

**目标：** 闭合 S1 胡牌优先级与海底/杠后规则，修复 S2 玩家视角泄漏和 belief 标签语义，并将 S3/S4 策略调用统一迁移到 `S2ProtocolState + legal_mask`。

**架构：** `GameState` 只属于裁判和 oracle 标签生成层。每个策略决策点由裁判生成观察者专属 `S2ProtocolState`，共享合法性生成固定动作空间 mask，策略返回动作空间索引对应的引擎 `Action`，最后由 `Game.step()` 再次执行强制合法性校验。S1、S2、S3 依次落地，避免在错误规则或泄漏协议上迁移上层接口。

**技术栈：** Python 3.10+、pytest、PyTorch、现有 `engine`/`state`/`policies`/`selfplay`/`learning` 模块。

---

## 阶段一：S1 规则闭环

### Task 1：锁定胡牌响应优先级和 `step()` 原子门禁

**文件：**
- 修改：`tests/test_game_loop_full.py`
- 修改：`engine/game.py`

- [ ] **Step 1：增加失败回归测试**
  - 构造 `pending_discard`、`pending_winners == [1]`，并让玩家 2 手牌可碰/直杠。
  - 断言 `game.legal_actions(2) == []`，玩家 1 仍只得到 `WIN/PASS`。
  - 分别直接调用玩家 2 的 `PONG`、`KONG(EXPOSED)`，断言抛出 `ValueError`。
  - 调用前后深拷贝并比较 `pending_discard`、`pending_winners`、手牌、副露、河牌、牌墙、分数、杠流水、当前玩家和阶段，确保失败无状态修改。
  - 增加错误玩家、过期动作及换三张/定缺阶段非法动作测试，证明门禁覆盖所有阶段。

- [ ] **Step 2：运行测试并确认按预期失败**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_game_loop_full.py -q
```

预期：其他响应者仍能获得/执行碰杠，或非法阶段动作未被统一拒绝。

- [ ] **Step 3：在状态变更前实施统一校验**
  - 在 `Game.step()` 开头调用 `Game.legal_actions(player)`，提交动作不在集合时立即抛出 `ValueError`。
  - 保持 `finished` 与未知阶段错误清晰；任何 `_step_*` 执行前不得修改状态。
  - 调整 `Game.legal_actions()`：`pending_winners` 非空时，非候选胡牌玩家不得获得碰、杠或弃牌响应；候选玩家只得到胡/过。
  - 不移除各 `_step_*` 的参数校验，让内部校验继续作为防御层。

- [ ] **Step 4：运行专项测试至通过**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_game_loop_full.py tests/test_game_flow.py -q
```

### Task 2：所有杠统一执行“牌墙至少两张”约束

**文件：**
- 修改：`tests/test_game_loop_full.py`
- 修改：`tests/test_s2_legality_crosscheck.py`
- 修改：`engine/game.py`
- 修改：`state/legality.py`

- [ ] **Step 1：增加三类杠生成与强制执行测试**
  - 对直杠、暗杠、补杠分别构造 `len(wall) == 1`。
  - 断言引擎动作列表不含对应杠。
  - 直接调用 `Game.step()` 时断言拒绝且状态不变。
  - 保留 `len(wall) == 2` 的正向边界用例。
  - 在 S2 legality 交叉验证中断言相同边界，避免协议层再次生成末张杠。

- [ ] **Step 2：运行测试并确认失败**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_game_loop_full.py tests/test_s2_legality_crosscheck.py -q
```

- [ ] **Step 3：统一修改杠条件**
  - 将 `Game.legal_actions()` 中直杠、暗杠、补杠条件统一为 `len(state.wall) > 1`。
  - 将 `_step_exposed_kong()`、`_step_self_kong()` 和补杠完成路径的防御条件统一为相同规则。
  - 修改 `state.legality._pending_discard_actions()`、`_turn_actions()` 的对应判断。

- [ ] **Step 4：运行专项测试至通过**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_game_loop_full.py tests/test_s2_legality_action_space.py tests/test_s2_legality_crosscheck.py -q
```

### Task 3：显式维护摸牌与弃牌胡牌上下文

**文件：**
- 修改：`engine/state.py`
- 修改：`engine/game.py`
- 修改：`state/adapters/from_engine.py`
- 修改：`state/legality.py`
- 修改：`tests/test_game_loop_full.py`
- 修改：`tests/test_settlement.py`
- 修改：`tests/test_s2_from_engine.py`

- [ ] **Step 1：增加真实链路结算测试**
  - 七对在杠后补牌自摸，断言分数为 `[27, -9, -9, -9]`（按赢家座位旋转）。
  - 普通摸走牌墙最后一张后自摸，断言触发海底捞月。
  - 摸走最后一张后弃牌，另一玩家胡，断言触发海底炮。
  - 杠后补牌弃牌后被胡，断言触发杠上炮。
  - 后续普通回合、抢杠胡均不继承过期 `after_kong`/`haidi`。
  - `GameState.to_dict()` 覆盖新增上下文字段。

- [ ] **Step 2：运行测试并确认上下文缺失导致失败**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_game_loop_full.py tests/test_settlement.py tests/test_s2_from_engine.py -q
```

- [ ] **Step 3：实现显式上下文状态机**
  - 在 `GameState` 增加当前摸牌上下文和待弃牌响应上下文，至少表达 `after_kong_draw`、`last_wall_draw` 及其弃牌版本；同步 `to_dict()`。
  - `_draw_replacement_after_kong()` 在摸牌前判断是否最后一张，并设置杠后标记。
  - `_advance_after_resolved_discard()` 普通摸牌时清除杠后标记，并记录是否最后一张。
  - `_step_play()` 弃牌时把当前摸牌上下文转移到 `pending_discard` 对应上下文，再清除当前自摸上下文。
  - `_settle_self_win()` 统一构造包含 `self_draw=True`、当前 `after_kong` 和当前 `haidi` 值的 `WinContext`。
  - `_step_response()`、`_discard_win_fan()` 和过胡番数比较使用同一个弃牌 `WinContext`。
  - `_step_rob_kong_response()` 只传 `robbing_kong=True`。
  - `_clear_pending_discard()` 清理弃牌上下文，防止泄漏到后续回合。
  - `from_engine()` 与共享 legality 只消费公开的杠后弃牌/海底弃牌事实，不暴露内部候选赢家。

- [ ] **Step 4：运行阶段一全部测试**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_game_flow.py tests/test_game_loop_full.py tests/test_fan_calc.py tests/test_settlement.py tests/test_random_playout.py tests/test_s2_from_engine.py tests/test_s2_legality_action_space.py tests/test_s2_legality_crosscheck.py -q
```

- [ ] **Step 5：运行全量测试**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest -q
```

- [ ] **Step 6：阶段一提交（仅 Git 可用时）**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; git status --short; git add engine state tests; git commit -m "fix: enforce mahjong response and win context rules"; git push
```

---

## 阶段二：S2 可见性与 belief 标签

### Task 4：移除协议中的候选赢家泄漏

**文件：**
- 修改：`state/adapters/from_engine.py`
- 修改：`state/legality.py`
- 修改：`tests/test_s2_from_engine.py`
- 修改：`tests/test_s2_legality_action_space.py`
- 修改：`tests/test_s2_legality_crosscheck.py`
- 修改：`tests/test_s2_protocol.py`

- [ ] **Step 1：增加协议可见性失败测试**
  - 对 `pending_discard` 和 `pending_rob_kong` 分别遍历四个观察者，递归断言协议中不存在 `winners`、`winner_relatives`。
  - 构造公开事实相同但其他玩家暗手牌不同的状态，断言非当事观察者的 `facts` 和编码完全相同。
  - 断言观察者本人可胡时，`state.legality.legal_actions()` 仍能仅依赖本人手牌产生胡动作。

- [ ] **Step 2：运行测试并确认泄漏存在**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s2_from_engine.py tests/test_s2_legality_action_space.py tests/test_s2_legality_crosscheck.py tests/test_s2_protocol.py -q
```

- [ ] **Step 3：删除泄漏字段并改造合法性**
  - `_pending_discard()` 只保留弃牌者、相对位置、牌面及公开上下文。
  - `_pending_rob_kong()` 只保留补杠者、相对位置和牌面。
  - 删除 `state.legality._pending_discard_actions()`、`_rob_kong_actions()` 对 `winner_relatives` 的依赖，直接用观察者自己的可见手牌、定缺和过胡锁计算。
  - 旧 JSON 中多余字段继续由协议解析自然忽略，不将其编码或重新输出。

- [ ] **Step 4：运行专项测试至通过**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s2_from_engine.py tests/test_s2_protocol.py tests/test_s2_legality_action_space.py tests/test_s2_legality_crosscheck.py tests/test_s2_end_to_end.py -q
```

### Task 5：将 tile-location oracle 改为副本计数分布

**文件：**
- 修改：`state/tile_belief.py`
- 修改：`tests/test_s2_tile_belief.py`
- 修改：`tests/test_s4_dataset_builder.py`

- [ ] **Step 1：增加多副本、多位置与守恒测试**
  - 同种牌分别位于牌墙和多个对手暗手，断言 `counts` 保留所有副本。
  - 断言每个有未知副本的 `distribution` 行和为 1，`mask` 为真；无未知副本时 mask 为假。
  - 断言每行计数总和等于该牌种未知副本数，且不超过四张。
  - 人工构造超过四张或可见数与隐藏数不守恒状态，断言明确抛出 `ValueError`。

- [ ] **Step 2：运行测试并确认旧 `setdefault()` 语义失败**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s2_tile_belief.py tests/test_s4_dataset_builder.py -q
```

- [ ] **Step 3：实现新标签结构**
  - `_oracle_tile_locations()` 逐个物理副本计数，位置顺序固定为 `wall/1/2/3`。
  - `generate_belief_labels()` 输出 tile-location 的 `counts`、`distribution`、`mask`，不再输出单一位置字符串。
  - 用协议可见牌计数校验 `visible + hidden == 4`；已揭示胡牌手不重复进入未知位置。
  - `DecisionRecord` 维持通用字典序列化，但新生成数据只写新语义。

- [ ] **Step 4：运行专项测试至通过**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s2_tile_belief.py tests/test_s3_data_recorder.py tests/test_s4_dataset_builder.py -q
```

### Task 6：训练与评估改用 masked soft-target loss

**文件：**
- 修改：`learning/training/train_belief.py`
- 修改：`learning/eval/eval_belief.py`
- 修改：`tests/test_s4_train_belief.py`
- 修改：`tests/test_s4_eval_belief.py`
- 修改：`tests/test_s4_belief_net_metrics.py`

- [ ] **Step 1：增加 batch 形状和损失测试**
  - 断言 `tile_location_targets.shape == (batch, 27, 4)` 且为浮点分布。
  - 断言 `tile_location_mask.shape == (batch, 27)`。
  - 构造同一牌种 1:3 分布，手算软目标交叉熵并与实现一致。
  - 全 mask 关闭时损失为有限零值并可反向传播。
  - 评估统计按副本分布计算交叉熵/准确性，不再读取字符串硬类别。

- [ ] **Step 2：运行测试并确认旧硬标签失败**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s4_train_belief.py tests/test_s4_eval_belief.py tests/test_s4_belief_net_metrics.py -q
```

- [ ] **Step 3：实现软目标数据流**
  - `belief_batch_from_samples()` 读取新 `distribution/mask`，构造 `(N,27,4)` target。
  - `_masked_tile_location_loss()` 使用 `log_softmax` 后按目标分布加权，并只对 mask 行求平均。
  - `evaluate_belief_model()` 与 prior 评估同步新标签语义。
  - 对旧单字符串标签明确报出不兼容错误，不静默伪造分布。

- [ ] **Step 4：运行训练评估专项测试至通过**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s4_train_belief.py tests/test_s4_eval_belief.py tests/test_s4_belief_net_metrics.py -q
```

### Task 7：听牌 oracle 遵守定缺并编码各玩家公开定缺

**文件：**
- 修改：`state/tile_belief.py`
- 修改：`state/encoder.py`
- 修改：`tests/test_s2_tile_belief.py`
- 修改：`tests/test_s2_encoder.py`
- 修改：`tests/test_s4_cloud_training_package.py`

- [ ] **Step 1：增加定缺标签与编码测试**
  - 构造牌型本身听牌但仍持有定缺门牌的对手，断言 oracle 为 `False`。
  - 去掉定缺门牌后断言 oracle 恢复正确结果。
  - 修改任一玩家公开 `void_suit`，断言 `player_void_suits` 编码区段变化；隐藏手牌不变时其他区段不受意外影响。

- [ ] **Step 2：运行测试并确认失败**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s2_tile_belief.py tests/test_s2_encoder.py tests/test_s4_cloud_training_package.py -q
```

- [ ] **Step 3：实现定缺语义**
  - `_oracle_opponent_tenpai()` 调用 `ting_tiles(hand, engine_state.void_suits[player_id])`。
  - 在 `state.encoder._SECTION_SPECS` 增加四家相对位置顺序的定缺 one-hot/unknown/confidence 区段，并提升 `ENCODER_VERSION`。
  - 编码从 `facts.players` 的公开 `void_suit` 读取，不读取引擎状态。
  - 同步依赖固定输入维度的测试和云训练包元数据；旧 checkpoint 因输入版本变化明确不兼容。

- [ ] **Step 4：运行阶段二全部测试和全量测试**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s2_protocol.py tests/test_s2_from_engine.py tests/test_s2_encoder.py tests/test_s2_legality_action_space.py tests/test_s2_legality_crosscheck.py tests/test_s2_tile_belief.py tests/test_s2_end_to_end.py tests/test_s3_data_recorder.py tests/test_s4_dataset_builder.py tests/test_s4_train_belief.py tests/test_s4_eval_belief.py tests/test_s4_belief_net_metrics.py tests/test_s4_cloud_training_package.py -q; python -m pytest -q
```

- [ ] **Step 5：阶段二提交（仅 Git 可用时）**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; git status --short; git add state learning selfplay tests; git commit -m "fix: remove hidden information and correct belief labels"; git push
```

---

## 阶段三：S3/S4 策略接口迁移

### Task 8：建立协议动作转换和严格策略契约

**文件：**
- 创建：`policies/protocol_actions.py`
- 修改：`policies/base_policy.py`
- 修改：`tests/test_s3_rule_policy.py`
- 修改：`tests/test_s3_opponent_pool.py`

- [ ] **Step 1：先把策略契约测试改为新接口**
  - `BasePolicy.choose_action(protocol_state, legal_mask)` 不再接收 `GameState`、player 或可变动作列表。
  - mask 长度不等于 `state.action_space.action_space_size()` 时抛出 `ValueError`。
  - mask 全假时抛出 `ValueError`。
  - 返回值必须对应 mask 内动作；测试覆盖弃牌、碰、三类杠、胡、过、定缺、换三张。

- [ ] **Step 2：运行测试并确认旧签名失败**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s3_rule_policy.py tests/test_s3_opponent_pool.py -q
```

- [ ] **Step 3：实现共享转换边界**
  - `policies/protocol_actions.py` 提供动作字典与 `engine.actions.Action` 的双向转换，以及从 mask 枚举候选动作。
  - 所有转换复用 `state.action_space.action_to_index/index_to_action`，不另建动作顺序。
  - `BasePolicy` 类型签名改为 `S2ProtocolState + Sequence[bool] -> Action`。
  - 增加共享 mask 校验函数，策略和裁判调用边界共同使用。

- [ ] **Step 4：运行契约测试至通过**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s3_rule_policy.py tests/test_s3_opponent_pool.py tests/test_s2_legality_action_space.py -q
```

### Task 9：迁移 Rule/Greedy/Random 策略到玩家视角

**文件：**
- 修改：`policies/rule_policy.py`
- 修改：`policies/opponent_pool.py`
- 修改：`policies/heuristics.py`（仅在需要接受可见 `Hand`/协议事实时做最小调整）
- 修改：`tests/test_s3_rule_policy.py`
- 修改：`tests/test_s3_opponent_pool.py`

- [ ] **Step 1：增加隐藏状态独立性测试**
  - 从两个公开信息一致、对手暗手牌和牌墙顺序不同的引擎状态生成同一观察者协议。
  - 断言协议编码、mask 和确定性 `RulePolicy`/`GreedyPolicy` 决策一致。
  - 使用会在访问 `GameState` 时失败的策略桩，证明策略入口只收到 `S2ProtocolState`。

- [ ] **Step 2：实现内置策略迁移**
  - 从协议中提取本玩家 `concealed_hand`、`void_suit`、公开待响应牌和 beliefs。
  - `RulePolicy` 继续执行胡 > 杠 > 牌效弃牌 > 有益碰 > 过，但候选集仅来自 mask。
  - `GreedyPolicy` 使用同一可见手牌提取函数；`RandomPolicy` 只在 mask 真值索引中采样。
  - 删除 `GameState` import 和任何对 `hands[player]`、`wall`、内部 pending winners 的访问。

- [ ] **Step 3：运行策略专项测试至通过**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s3_rule_policy.py tests/test_s3_opponent_pool.py -q
```

### Task 10：迁移自对局、记录器和竞技场裁判边界

**文件：**
- 修改：`selfplay/run_selfplay.py`
- 修改：`selfplay/data_recorder.py`
- 修改：`learning/eval/arena.py`
- 修改：`tests/test_s3_selfplay.py`
- 修改：`tests/test_s3_data_recorder.py`
- 修改：`tests/test_s4_eval_policy_arena.py`

- [ ] **Step 1：增加调用边界和非法策略测试**
  - 捕获策略参数，断言每次均为当前玩家 `S2ProtocolState` 和固定长度 mask。
  - 策略返回 mask 外动作时，自对局和 arena 都明确失败/记录 `IllegalPolicyAction`，不得随机替换。
  - 记录器中的 `state` 不含候选赢家；oracle 只存在 `labels`。

- [ ] **Step 2：实现统一裁判决策函数**
  - 每次决策执行 `from_engine(game.state, player)`，再调用 `state.action_space.legal_mask(protocol_state)`。
  - 策略返回引擎 `Action` 后，用 `action_to_index` 校验 mask，再交给 `Game.step()`。
  - 响应遍历仍可由裁判读取 `pending_winners` 决定轮到谁，但绝不把该列表传给策略。
  - 移除人为给策略候选列表追加 `PASS` 的做法；合法过牌语义由 S1/S2 合法性统一生成。
  - `SelfplayDataRecorder.record_decision()` 复用本次决策已生成的协议和 mask 对应动作，避免再次构造不一致状态。

- [ ] **Step 3：运行调用方专项测试至通过**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s3_selfplay.py tests/test_s3_data_recorder.py tests/test_s4_eval_policy_arena.py -q
```

### Task 11：迁移回放工具并清除旧策略调用点

**文件：**
- 修改：`tools/export_s3_replay.py`
- 修改：`tests/test_s3_replay_export.py`
- 检查：`tools/random_playout.py`
- 检查：全仓库 Python 文件

- [ ] **Step 1：迁移回放导出**
  - 回放策略决策使用相同 `from_engine + legal_mask` 边界。
  - 输出公开协议和合法动作，不输出 `GameState.pending_winners` 或抢杠候选人。
  - 保持纯随机引擎压力工具 `tools/random_playout.py` 直接消费 `Game.legal_actions()`；它不是策略接口，不需要伪装成 S3 policy。

- [ ] **Step 2：运行回放测试**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s3_replay_export.py -q
```

- [ ] **Step 3：静态搜索确认旧入口归零**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; Select-String -Path policies\*.py,selfplay\*.py,learning\eval\*.py,tools\export_s3_replay.py -Pattern 'choose_action\(.*state.*,.*player.*,.*legal_actions|choose_action\(game\.state|from engine\.state import GameState'
```

预期：策略与策略调用方中无旧签名、无 `GameState` 策略依赖；仅裁判模块自身可持有 `Game`/`GameState`。

### Task 12：S3/S4 端到端验收

**文件：**
- 修改：必要的 S3/S4 测试夹具
- 验证：全仓库

- [ ] **Step 1：运行 S3/S4 专项回归**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest tests/test_s3_shanten.py tests/test_s3_rule_policy.py tests/test_s3_opponent_pool.py tests/test_s3_selfplay.py tests/test_s3_data_recorder.py tests/test_s3_replay_export.py tests/test_s4_dataset_builder.py tests/test_s4_train_policy.py tests/test_s4_eval_policy_arena.py -q
```

- [ ] **Step 2：运行单局与多局冒烟**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python selfplay/run_selfplay.py --games 20 --seed 1 --max-steps 1000 --json
```

预期：`unfinished=0`、零和断言通过、无非法策略动作。

- [ ] **Step 3：运行最终全量测试**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; python -m pytest -q
```

- [ ] **Step 4：检查工作区与诊断**
  - 检查所有修改文件无 lint/类型诊断。
  - 确认没有临时脚本、测试缓存或生成数据进入改动。
  - 确认旧 belief checkpoint/数据兼容性已在文档或错误信息中明确表达。

- [ ] **Step 5：阶段三提交（仅 Git 可用时）**

```powershell
Set-Location -LiteralPath 'D:\sichuan-mahjong-engine'; git status --short; git add policies selfplay learning tools tests; git commit -m "refactor: isolate policies behind S2 protocol interface"; git push
```

---

## 最终验收清单

- [ ] 有胡牌候选时，其他玩家既不能生成也不能强制执行碰/杠，失败保持状态原子性。
- [ ] 牌墙剩一张时，直杠、暗杠、补杠均不可生成和执行。
- [ ] 杠上花、杠上炮、海底捞月、海底炮由真实对局链路触发且上下文不过期泄漏。
- [ ] 玩家协议不含 `winners`、`winner_relatives` 或等价暗手牌推导信息。
- [ ] tile-location 标签完整保存每种牌在四位置的副本计数、分布与 mask。
- [ ] belief 训练使用 masked soft-target loss，定缺听牌标签正确。
- [ ] 编码包含四家公开定缺信息并提升版本。
- [ ] 所有内置策略、自对局、arena、记录器和回放只通过 `S2ProtocolState + legal_mask` 决策。
- [ ] S1–S4 专项测试、20 局冒烟和完整测试套件通过。
- [ ] 若 `.git` 与远端可用，三个阶段分别提交并推送；否则只报告实际环境阻塞。
