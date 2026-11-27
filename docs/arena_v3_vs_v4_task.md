# v3 vs v4 迭代升级 10,000 局同台验收任务 (`tools/run_arena_tournament.py`)

本任务把 **三代冻结版本**（`v2_ev_base`、`v3_goodwait_read`、`v4_waitq_tactics`）与强规则锚点 `s3_plus` 放在同一牌桌随机匹配 + 四座轮转（共 **10,000 场对局、40,000 人次**），直接检验 `v4` 的 4 项新升级是否稳定超越 `v3` 并冲向 80+ 分老师水平。

## 参赛策略池（6 个）
1. `s3_plus`：S3+ 增强规则 AI（强规则锚点）
2. `v2_ev_base_det`：冻结旧版 `v2`（`git 8cb0cfc`）
3. `v3_goodwait_read_det`：冻结 `v3` 确定版（`git 3f0bcd9`）
4. `v3_goodwait_read_rand`：冻结 `v3` 随机版（上轮 10,000 局冠军 `+0.1301`）
5. `v4_waitq_tactics_det`：新版 `v4` 确定版（`git 66e601a`，听对手缺门/边幺/换牌必打加成 + 番数动态防守 + 中盘暗杠追回 + 收紧缩手碰牌）
6. `v4_waitq_tactics_rand`：新版 `v4` 受控随机版

---

## 运行步骤（高算力多核主机）

### 0. 拉取最新代码
```powershell
git pull origin main
```

### 1. 15 秒冒烟自检（可选）
```powershell
python tools/run_arena_tournament.py --policies s3_plus,v2_ev_base_det,v3_goodwait_read_det,v3_goodwait_read_rand,v4_waitq_tactics_det,v4_waitq_tactics_rand --deals 8 --workers 4 --out outputs/arena_v3_vs_v4_smoke.json
```

### 2. 主力 10,000 局正式对照赛（2,500 副牌 × 4 座轮转 = 10,000 场）
支持随时 `Ctrl+C` 中断并用 `--resume` 续跑（10 核下预计约 **10~12 分钟**）：
```powershell
python tools/run_arena_tournament.py --policies s3_plus,v2_ev_base_det,v3_goodwait_read_det,v3_goodwait_read_rand,v4_waitq_tactics_det,v4_waitq_tactics_rand --deals 2500 --workers 10 --resume --out outputs/arena_v3_vs_v4.json
```

---

## 跑完后提交回仓库
```powershell
git add outputs/arena_v3_vs_v4.json
git commit -m "data: add 10000-game v3 vs v4 arena tournament results"
git push origin main
```

推上来喊我，我拉取下来出 **v2 → v3 → v4 三代演进总表**，达标就正式用整套版本库对手池启动下一代 SL/RL 数据采集与训练！
