# v4 vs v5 结构大升级 10,000 局同台验收任务 (`tools/run_arena_tournament.py`)

本任务检验辅助器 **v5 三大结构性升级**（碰/不碰统一 `discard_v2` 同口径 EV、1 向听真实番数+听口双期望、`Kabe` 绝张壁断搭与早巡切中张读牌）在 10,000 场实战对局中相比 `v4 / v3 / s3_plus` 的真实涨幅。

## 参赛策略池（6 个）
1. `s3_plus`：强规则基线锚点
2. `v3_goodwait_read_det`：冻结 `v3` 快攻版（`git 3f0bcd9`）
3. `v4_waitq_tactics_det`：冻结 `v4` 确定版（`git 66e601a`）
4. `v4_waitq_tactics_rand`：冻结 `v4` 随机版
5. `v5_unified_ev_det`：新版 `v5` 确定版（`git 0eaef06`）
6. `v5_unified_ev_rand`：新版 `v5` 随机版

---

## 运行步骤（高算力多核主机）

### 0. 拉取最新代码
```powershell
git pull origin main
```

### 1. 15 秒冒烟自检（可选）
```powershell
python tools/run_arena_tournament.py --policies s3_plus,v3_goodwait_read_det,v4_waitq_tactics_det,v4_waitq_tactics_rand,v5_unified_ev_det,v5_unified_ev_rand --deals 8 --workers 4 --out outputs/arena_v4_vs_v5_smoke.json
```

### 2. 主力 10,000 局正式对照赛（2,500 副牌 × 4 座轮转 = 10,000 场）
支持随时 `Ctrl+C` 中断并用 `--resume` 续跑（10 核下约 **12~14 分钟**）：
```powershell
python tools/run_arena_tournament.py --policies s3_plus,v3_goodwait_read_det,v4_waitq_tactics_det,v4_waitq_tactics_rand,v5_unified_ev_det,v5_unified_ev_rand --deals 2500 --workers 10 --resume --out outputs/arena_v4_vs_v5.json
```

---

## 跑完后提交回仓库
```powershell
git add outputs/arena_v4_vs_v5.json
git commit -m "data: add 10000-game v4 vs v5 arena tournament results"
git push origin main
```

推上来喊我，如果 `v5` 确认再上新台阶，我们顺手把 `tools/generate_zoo_sl_data.py` 的默认老师升级成 `v5 + v4 + v3` 最强组合，再送 Colab 训练！
