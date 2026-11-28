# v5 vs v6 实战精调大升级 10,000 局同台验收任务 (`tools/run_arena_tournament.py`)

本任务检验辅助器 **v6 人类实战精调升级**（早巡清缺生张中张顺序、普通 4~5 对子果断碰牌、听牌换听改良/摸杠潜力、四川血战龙七对精确向听、早中盘 10+ 张清一色动态潜力与成势碰、断张胡牌确定档反推、定缺感知牌墙 IPF 校正）在 10,000 场实战对局中相比 `v5 / v4 / s3_plus` 的真实涨幅。

> 本地 320 局配对预热：`v6_human_tactics_det` 领先 `v5_unified_ev_det +0.1531` 分/局，胡牌率 `71.6% vs 70.9%`，清一色胡率翻倍（`6.25% vs 2.97%`），点炮率持平（`45.62% vs 45.47%`）。

## 参赛策略池（6 个）
1. `s3_plus`：强规则基线锚点
2. `v4_waitq_tactics_det`：冻结 `v4` 确定版（`git 66e601a`）
3. `v5_unified_ev_det`：冻结 `v5` 冠军确定版（`git 0eaef06`）
4. `v5_unified_ev_rand`：冻结 `v5` 随机版
5. `v6_human_tactics_det`：新版 `v6` 确定版（`git 481df17`）
6. `v6_human_tactics_rand`：新版 `v6` 随机版

---

## 运行步骤（高算力多核主机）

### 0. 拉取最新代码
```powershell
git pull origin main
```

### 1. 15 秒冒烟自检（可选）
```powershell
python tools/run_arena_tournament.py --policies s3_plus,v4_waitq_tactics_det,v5_unified_ev_det,v5_unified_ev_rand,v6_human_tactics_det,v6_human_tactics_rand --deals 8 --workers 4 --out outputs/arena_v5_vs_v6_smoke.json
```

### 2. 主力 10,000 局正式对照赛（2,500 副牌 × 4 座轮转 = 10,000 场）
支持随时 `Ctrl+C` 中断并用 `--resume` 续跑（10 核下约 **12~15 分钟**）：
```powershell
python tools/run_arena_tournament.py --policies s3_plus,v4_waitq_tactics_det,v5_unified_ev_det,v5_unified_ev_rand,v6_human_tactics_det,v6_human_tactics_rand --deals 2500 --workers 10 --resume --out outputs/arena_v5_vs_v6.json
```

---

## 跑完后提交回仓库
```powershell
git add outputs/arena_v5_vs_v6.json
git commit -m "data: add 10000-game v5 vs v6 arena tournament results"
git push origin main
```

推上来喊我，验收 `v6` 实测数据后即可直接用已升级好的 `v6 + v5 + v4` 顶级老师池（`tools/generate_zoo_sl_data.py`）采数送 Colab 训练 `S4 v6`！
