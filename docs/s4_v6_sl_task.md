# S4 v6 监督学习全流程任务文档（多核 PC 采数 → Colab GPU 训练 → 10,000 局验收）

本任务用我们刚打磨好的 **78~80 分双风格老师池（`v4_waitq_tactics` + `v3_goodwait_read`）**，在多代版本库混战桌里采集高质量 `s2.v4` 监督学习数据集，送到 **Google Colab GPU** 训练新一代 **S4 v6（BeliefNet + PolicyNet 512×5）**，把神经网络起点从老 S4 v5（`-0.326`，约 48 分）直接拉升到 **75~78 分**，为下一步 S5 PPO 强化学习突破 85~95 分打好底座。

---

## 第一步：高算力多核 CPU 主机采集高质量数据（`tools/generate_zoo_sl_data.py`）

### 0. 拉取最新代码
```powershell
git pull origin main
```

### 1. 15 秒冒烟自检（可选）
```powershell
python tools/generate_zoo_sl_data.py --games 8 --shard-games 4 --workers 2 --out-dir outputs/s4_v6_smoke
```

### 2. 正式采数（推荐 30,000 局 ≈ 150 万条顶级老师样本，约 350~450 MB）
支持随时 `Ctrl+C` 中断并用 `--resume` 按分片续跑（10 核下约 **30~40 分钟**；内存安全，每 5 局自动清缓存）：
```powershell
python tools/generate_zoo_sl_data.py --games 30000 --shard-games 500 --workers 10 --resume --out-dir outputs/s4_v6_zoo_data
```

- **可选加量档（50,000 局 ≈ 250 万条样本）**：把 `--games 30000` 改成 `--games 50000` 即可在 30,000 局基础上直接 `--resume` 追加。
- **上传到 Google Drive**：采完后把 `outputs/s4_v6_zoo_data/` 打包成 `s4_v6_zoo_data.zip`（或者直接整文件夹上传到 Google Drive 的 `MyDrive/mahjong_s4_v6/s4_v6_zoo_data/`），不要把大数据集推到 GitHub。

```powershell
# PowerShell 一键打包上传用 zip（只打包 manifest.json 和 shards/）
Compress-Archive -Path outputs\s4_v6_zoo_data\manifest.json, outputs\s4_v6_zoo_data\shards -DestinationPath outputs\s4_v6_zoo_data.zip -Force
```

---

## 第二步：Google Colab GPU 训练 S4 v6（推荐选 **L4** 或 **A100** 运行时）

在 Colab 新建 Notebook，按顺序粘贴运行下面 3 个 Cell：

### Cell 1：挂载 Google Drive & 克隆最新代码
```python
from google.colab import drive
drive.mount('/content/drive')

!rm -rf /content/mahjong-strategy
!git clone --depth 1 https://github.com/stefaniesun/mahjong-strategy.git /content/mahjong-strategy
%cd /content/mahjong-strategy
!nvidia-smi
```

### Cell 2：解压数据到 Colab 本地 SSD（IO 极快，避免直接读 Drive 卡顿）
```python
!mkdir -p /content/s4_v6_data
# 按你的 Drive 实际路径修改下面这行：
!unzip -q /content/drive/MyDrive/mahjong_s4_v6/s4_v6_zoo_data.zip -d /content/s4_v6_data
!ls -lh /content/s4_v6_data/manifest.json && ls /content/s4_v6_data/shards | wc -l
```

### Cell 3：启动 S4 v6 GPU 训练（`512×5` 加宽加深网络，L4/A100 约 15~30 分钟）
```python
!python tools/cloud_train_s4_50k_cached.py \
  --data-dir /content/s4_v6_data/shards \
  --manifest /content/s4_v6_data/manifest.json \
  --output-dir /content/s4_v6_out \
  --hidden-size 512 \
  --residual-blocks 5 \
  --dropout 0.05 \
  --batch-size 1024 \
  --inference-batch-size 2048 \
  --learning-rate 3e-4 \
  --belief-epochs 4 \
  --policy-epochs 8 \
  --eval-samples 30000 \
  --device cuda \
  --pipeline cached \
  --workers 6

# 训练完成后立刻把模型与报告同步回 Google Drive，防止 Colab 断开丢失
!mkdir -p /content/drive/MyDrive/mahjong_s4_v6/v6_zoo_512x5
!cp -r /content/s4_v6_out/checkpoints /content/s4_v6_out/reports /content/drive/MyDrive/mahjong_s4_v6/v6_zoo_512x5/
!cat /content/s4_v6_out/reports/s4_training_report.md
```

---

## 第三步：下载 S4 v6 存档，回高算力 PC 跑 10,000 局实战验收

把 Drive 里的 `v6_zoo_512x5/` 放进本仓库的：
`training_artifacts/S4/v6_zoo_512x5/`（内含 `checkpoints/belief_s4.pt`、`checkpoints/policy_s4.pt` 和 `reports/`）

然后直接让 **S4 v6** 和 `s3_rule / s3_plus / v2 / v3 / v4` 同桌混战 10,000 局验分：
```powershell
python tools/run_arena_tournament.py `
  --policies s3_rule,s3_plus,v2_ev_base_det,v3_goodwait_read_det,v4_waitq_tactics_det,s4_v5 `
  --s4-checkpoint training_artifacts/S4/v6_zoo_512x5/checkpoints/policy_s4.pt `
  --deals 2500 --workers 10 --resume --out outputs/arena_s4_v6_eval.json
```

> 注：上面 `--s4-checkpoint` 会让锦标赛里的 `s4_v5` 槽位加载我们新训好的 **S4 v6**（并自动加载同目录下的新版 `belief_s4.pt`）。

### 验收成功标准（通过后即可直接进入 S5 v2 强化学习冲 85~95 分）
1. **Belief 校准**：`opponent tenpai ECE <= 0.015`，`discard danger ECE <= 0.002`；
2. **Policy 准确率**：`non-forced accuracy >= 80%`，`illegal argmax count == 0`；
3. **10,000 局实战分**：从老 S4 v5 的 `-0.326` 一举跨越 `s3_rule` 与 `s3_plus`，紧咬甚至逼近 `v3 / v4`（且单步推理速度比 `v4` 快 **20 倍**）！
