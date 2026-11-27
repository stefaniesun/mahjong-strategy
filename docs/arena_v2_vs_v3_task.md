# v2 vs v3 迭代升级 10,000 局同台验收任务 (`tools/run_arena_tournament.py`)

本任务用于直接把 **冻结旧版 `v2_ev_base`** 和 **全新升级版 `v3_goodwait_read`** 拉到同一张牌桌，在完全相同的 2,500 副牌上做四座轮转（共 **10,000 场对局、40,000 人次**），精确测量本次 5 项升级的真实净涨幅，并检验收紧随机抽样后 `v3_rand` 是否已追平 `v3_det`。

## 参赛策略池（5 个）
1. `s3_rule`：S3 基础规则 AI（基线锚点）
2. `s3_plus`：S3+ 增强规则 AI（强规则锚点）
3. `v2_ev_base_det`：冻结旧版 `v2`（`git 8cb0cfc`，上轮 10,000 局榜首 `+0.204`）
4. `v3_goodwait_read_det`：新版 `v3` 确定性版（摸切/放过牌读牌 + 1向听好型 + 碰提速 + 杠钱追回）
5. `v3_goodwait_read_rand`：新版 `v3` 受控随机版（出牌抽样收紧至 `top_k=2, max_gap=0.10, T=0.03`）

> 本次跳过 `s4_v5`（上轮 10,000 局已确认 `-0.326` 垫底），省掉 PyTorch 推理后速度极快；同时内置每副牌清空 `lru_cache` + 缓存上限，避免多核爆内存。

---

## 运行步骤（高算力多核主机）

### 0. 拉取最新代码
```powershell
git pull origin main
```

### 1. 15 秒冒烟自检（可选）
```powershell
python tools/run_arena_tournament.py --policies s3_rule,s3_plus,v2_ev_base_det,v3_goodwait_read_det,v3_goodwait_read_rand --deals 8 --workers 4 --out outputs/arena_v2_vs_v3_smoke.json
```

### 2. 主力 10,000 局正式对照赛（2,500 副牌 × 4 座轮转 = 10,000 场）
支持随时 `Ctrl+C` 中断并用 `--resume` 续跑（10 核下预计仅约 **3~5 分钟**）：
```powershell
python tools/run_arena_tournament.py --policies s3_rule,s3_plus,v2_ev_base_det,v3_goodwait_read_det,v3_goodwait_read_rand --deals 2500 --workers 10 --resume --out outputs/arena_v2_vs_v3.json
```

- **可选超高精度档（16,000 场 / 64,000 人次）**：如果想把两两同桌误差压得更细，可直接把 `--deals 2500` 改成 `--deals 4000`（约 6~8 分钟）。

---

## 跑完后提交回仓库
```powershell
git add outputs/arena_v2_vs_v3.json
git commit -m "data: add 10000-game v2 vs v3 arena tournament results"
git push origin main
```

推上来告诉我，我拉取下来直接出 **v2 vs v3 逐项指标对比表**（场均净分、两两同桌净胜分、胡率、自摸率、点炮率、场均杠数、大赢/重赔率、以及 `v3_rand` vs `v3_det` 的随机损耗）！
