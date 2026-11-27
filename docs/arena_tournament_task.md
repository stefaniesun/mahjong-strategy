# 多策略随机匹配锦标赛大样本评测说明 (`tools/run_arena_tournament.py`)

本任务用于让项目内全部主流策略在同一牌桌随机混战（配合 **同牌四座轮转** 压缩发牌运气方差），摸清我们当前整套辅助器 AI 在全库里的真实水平与受控随机性损耗。

## 参赛策略池（自动识别）
1. `s3_rule`：旧版 S3 基础规则 AI（`RulePolicy`）
2. `s3_plus`：S3+ 增强规则 AI（`RulePlusPolicy('all')`）
3. `s4_v5`：S4 v5 监督学习神经网络模型（自动读取 `training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/policy_s4.pt`）
4. `assistant_det`：当前辅助器全套 AI **确定性最优版**（`swap_v2 + void_v3 + opponent_read + discard_v2 + tactics`）
5. `assistant_rand`：当前辅助器全套 AI **受控随机版**（用于直接检验受控随机抽样是否损失棋力）
6. `s5`（可选）：如果高算力机上有 `s5-step-900.pt`，可通过 `--s5-checkpoint` 一并拉入同台 PK

---

## 推荐运行命令（高算力多核主机）

### 1. 10 秒环境冒烟自检
```powershell
python tools/run_arena_tournament.py --deals 4 --workers 2 --out outputs/arena_smoke.json
```

### 2. 主力大样本正式锦标赛（推荐：1,000 副牌 × 4 座轮转 = 4,000 场对局，16,000 人次）
含 `s4_v5` 神经网络同台混战（支持随时 Ctrl+C 中断、`--resume` 断点续跑）：
```powershell
python tools/run_arena_tournament.py --deals 1000 --workers 15 --resume --out outputs/arena_tournament_large.json
```
*(实测 2 核 16 场含 S4 仅需 24 秒；15 核跑满 4,000 场预计约 10~15 分钟；若想先快跑 2,000 场把 `--deals 1000` 改为 `--deals 500` 即可，约 5~8 分钟。若高算力机上也有 `s5-step-900.pt`，可追加 `--s5-checkpoint <你的路径>`)*

### 3. 极速纯规则+辅助器大样本（可选，跳过 PyTorch，速度约快 4 倍）
```powershell
python tools/run_arena_tournament.py --deals 1000 --no-s4 --workers 15 --resume --out outputs/arena_fast_large.json
```

---

## 跑完后提交回仓库
```powershell
git add outputs/arena_tournament_large.json
git commit -m "data: add 2000-game multi-policy random arena tournament results"
git push origin main
```
推上来之后告诉我，我拉取下来帮你做完整的分层战绩分析、牌力评分和弱点拆解！
