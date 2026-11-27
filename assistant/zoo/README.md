# 策略版本库 (`assistant/zoo/`)

每次迭代升级的策略都会单独冻结保存在本目录下的子文件夹中（每个版本自包含 `manifest.json` + 相对导入隔离的 `pkg/`）。

## 为什么要按版本单独保存？

1. **纵向精准对照**：直接把新版本和老版本放进同一张牌桌（同一批随机种子 + 四座轮转），一眼看出每次升级净涨多少分，避免“感觉变强了”。
2. **训练数据与对手池多样性**：后续做监督学习（SL）轨迹采集或强化学习（RL）对手池训练（League / Self-Play）时，可以让不同代际、不同随机度的历史版本混在一起当对手，防止神经网络过拟合到某一个固定习惯。
3. **100% 版本隔离**：主线 `assistant/` 继续往 `v4 / v5 / ...` 迭代时，老版本文件夹里的冻结代码完全不受影响。

---

## 已归档版本档案

| 版本文件夹 | Git Commit | 核心升级点 | 10,000 局大样本记录 |
| :--- | :--- | :--- | :--- |
| `v2_ev_base` | `8cb0cfc` | 初代完整全套 `AssistantPolicy`（`swap_v2 + void_v3 + opponent_read + discard_v2 + tactics`） | `det: +0.204 (±0.086)` / `rand: +0.079 (±0.086)` (`outputs/arena_tournament_large.json`) |
| `v3_goodwait_read` | `3f0bcd9` | 摸切/手牌未变放过牌读牌 + 1向听好型前瞻 + 三家碰提速 + 杠钱追回 + 收紧随机抽样 + 缓存限额 | 待跑 `v2 vs v3` 10,000 局对照赛 |

---

## 常用命令

### 1) 以后每次迭代完一键归档新版本

```bash
python -m assistant.zoo.snapshot_tool --name v4_xxx --note "一句话写清本次升级内容"
```

### 2) 锦标赛直接拉任意历史版本同桌混战

支持写法：`<版本名>_det`（确定性）、`<版本名>_rand`（受控随机）、`zoo:<版本名>`：

```bash
# 直接让 v2 和 v3 同桌打 10,000 局 (2500 副牌 x 4 轮转)
python tools/run_arena_tournament.py \
  --policies s3_rule,s3_plus,v2_ev_base_det,v3_goodwait_read_det,v3_goodwait_read_rand \
  --deals 2500 --workers 10 --resume --out outputs/arena_v2_vs_v3.json
```

### 3) 网页人机对战指定任意历史版本当对手

```bash
python -m play.server --ai zoo:v2_ev_base --port 8770
python -m play.server --ai zoo:v3_goodwait_read --port 8770
```

### 4) 后续 SL/RL 代码里一键构建多代对手池

```python
from assistant.zoo import list_versions, load_version, build_opponent_pool

print(list_versions())  # ['v2_ev_base', 'v3_goodwait_read']
pool = build_opponent_pool(include_deterministic=True, include_stochastic=True)
```
