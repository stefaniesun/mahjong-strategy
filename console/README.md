# S5 双机本地训练控制台

本目录实现 Windows 训练机 A 与 Ubuntu 评估机 B 的可信局域网控制台。它**没有认证**，严禁公网暴露、端口映射或公网反向代理。

## 1. 安装

两台机器均在仓库根目录创建 Python 3.12 虚拟环境并执行：

```powershell
python -m pip install -r requirements.txt
```

复制 `configs/s5_console.example.yaml` 为 `configs/s5_console.local.yaml`，按机器修改绝对 `project_root`、A/B 固定内网地址和 S4 评估资产路径。真实配置已被 Git 忽略。

## 2. 训练机 A（Windows）

1. 安装并启动 LibreHardwareMonitor，启用 WMI。控制台只读取 `root/LibreHardwareMonitor`，不使用不可靠的 ACPI thermal zone。
2. 确认 `nvidia-smi` 可从 PowerShell 执行。
3. 将 `agent.host` 设为 A 的固定内网 IP；Windows 防火墙只允许 B 的固定 IP 访问 `agent.port`。
4. 手工启动：

```powershell
powershell -NoProfile -File console/deploy/windows/start-agent.ps1 -Config configs/s5_console.local.yaml
```

安装开机服务：

```powershell
powershell -NoProfile -File console/deploy/windows/install-scheduled-task.ps1
```

计划任务只启动 Agent，不会自动开始训练。

## 3. 评估机 B（Ubuntu）

安装传感器工具并检查 CPU 温度：

```bash
sudo apt install lm-sensors
sensors -j
```

创建低权限用户，将仓库放到 `/opt/sichuan-mahjong-engine`，复制配置到 `/etc/s5-console.yaml`。如目录不同，先修改 unit 中的三个绝对路径，然后安装 `console/deploy/linux/s5-console.service`：

```bash
sudo cp console/deploy/linux/s5-console.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now s5-console
```

仅允许家庭局域网访问 `server.port`。浏览器打开 `http://<B内网IP>:8766/`。

## 4. 操作语义

- 页面只能覆盖 `updates`、`episodes_per_update`、`arena_games`、`seed`、`device`、`max_game_steps`。
- “优雅暂停”创建 stop flag；训练完成当前 update、发布完整 `latest.pt` 和指标后正常退出。
- “恢复”从 `latest.pt` 连续恢复 optimizer、RNG、league、curriculum、global step 和累计局数。
- A CPU >85°C 或训练盘 <10 GiB 自动优雅暂停。
- B CPU >90°C 暂停评估，≤80°C 自动恢复。
- 温度未知会显示异常，但不会伪造安全温度。

## 5. 联调验收

### 正常流程

1. A/B 服务均启动，页面显示在线，训练仍为 `IDLE/PAUSED`。
2. 页面启动 1 update smoke，确认 `metrics.jsonl`、`latest.pt`、日志产生。
3. B 下载 checkpoint，SHA-256 与 A 一致；完成 100 局快评。
4. 点击暂停，状态依次为 `RUNNING → PAUSING → PAUSED`，进程退出码为 0。
5. 点击恢复，global step 和累计局数严格递增，无重复或回退。
6. 最新档完成每日 500 局、seed 90000 正式评估。

### 异常流程

1. 停止 A 网络：B 显示 `OFFLINE`，历史曲线仍可查看。
2. 重启 Agent：若训练 PID 仍属于受管命令，禁止第二次启动；若进程已死，自愈为 `PAUSED/IDLE`，绝不自动开训。
3. 模拟 A 磁盘低于阈值或 CPU 过热，确认创建 stop flag 而非强杀进程。
4. 模拟 B 温度超过 90°C，确认不领取评估；降至 80°C 后恢复。
5. 截断 JSONL 最后一行，服务忽略坏尾并继续展示此前完整历史。
6. 下载中断或 hash 不符，临时文件被清理，损坏 checkpoint 不进入归档。

## 6. 备份与恢复

定期备份 A 的训练输出目录和 B 的 `.console/server-b`。不要只复制 `latest.pt` 而遗漏 S4 policy/belief 资产。恢复前先停止服务，核验 SHA-256，再复制完整目录。`diagnostic.pt` 仅用于故障分析，禁止恢复训练。
