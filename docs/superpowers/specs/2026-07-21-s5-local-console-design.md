# S5 本地训练控制台设计

**日期：** 2026-07-21  
**状态：** 已确认

## 目标

在不改变 S5 训练数值行为的前提下，为 Windows 训练机 A 和 Ubuntu 常驻机 B 提供可恢复、可观测、可安全暂停的双机本地长训系统。仓库内先完成代码、测试和部署模板，再在真实机器上远程协助部署联调。

## 已确认约束

- 可信家庭局域网内免登录；不实现 token、账号、HTTPS 或公网访问。
- 服务仅用于内网，禁止端口映射和公网反向代理。
- 启动训练采用固定 YAML 配置和少量白名单参数；不接受任意命令、路径或 shell 文本。
- 无数据库、Docker、消息队列和前端构建工具；持久化使用 JSON/JSONL。
- UI 为单个离线 HTML/CSS/JS 页面，图表使用手写 SVG。

## 架构

### 机器 A：训练代理

`console/agent_a.py` 托管 FastAPI API，管理 `tools/cloud_train_s5.py` 子进程，读取训练指标和 checkpoint，采集温度、内存及磁盘。状态机为 `IDLE/RUNNING/PAUSING/PAUSED/COMPLETED/ERROR`。暂停只创建 stop flag，由训练循环在 update 完成、完整 checkpoint 发布后正常退出。

### 机器 B：聚合与评估

`console/server_b.py` 托管单页控制台，轮询 A、保存历史指标、拉取并校验 checkpoint、维护“最新优先”的评估队列。新档运行 100 局快评，每日最新档运行 500 局、seed 90000 正评。B 温度高于 90°C 暂停评估，降至 80°C 恢复。

### 训练基础层

`rl/train_rl.py` 增加可选 stop flag 和 metrics JSONL 输出。每个 update 完成且 `latest.pt` 原子发布后追加一行指标，再检查 stop flag。checkpoint 格式、PPO、rollout、arena、league 和 curriculum 的数值顺序不变。

`tools/cloud_train_s5.py` 暴露 `--resume`、`--stop-file`、`--metrics-file`，并将其透传给训练配置。恢复使用现有完整 checkpoint 机制。

## 配置与安全

`console/config.py` 以严格 dataclass 加载 YAML。训练命令通过参数数组启动，永不使用 shell。可从页面覆盖的字段仅包括：`updates`、`episodes_per_update`、`arena_games`、`seed`、`device`、`max_game_steps`。所有 checkpoint、日志和指标路径必须解析后位于配置根目录内。

## 状态恢复

Agent 将状态原子写入 JSON。重启后检查 PID 是否仍属于受管训练命令；不自动恢复训练。若存在有效 checkpoint，则进入 `PAUSED`，否则 `IDLE`。训练异常退出进入 `ERROR` 并保留日志尾部；优雅 stop 退出进入 `PAUSED`；自然完成进入 `COMPLETED`。

## 指标格式

每行至少包含：UTC 时间、update、global step、总局数、耗时、局/分钟、policy loss、value loss、entropy、KL、课程阶段、league 大小、退化画像比例、非法动作数、零和失败数和 checkpoint。JSONL 写入使用单行 flush + fsync，坏尾行读取时忽略并报告。

## checkpoint 同步与评估

A 只列出受管 checkpoint 目录内的 `.pt` 文件，下载接口拒绝路径穿越。B 先下载到临时文件，核对大小和 SHA-256 后原子发布。稳定任务 ID 由 checkpoint SHA-256、评估类型、局数和 seed 构成，`evals.jsonl` 用于去重恢复。队列积压时保留正在执行项并优先最新未评档。

独立评估使用 S5 `PolicyValueNet` checkpoint 适配到现有 production arena；对手固定为 3×S3。正式评估固定 500 局、seed 90000，输出场均分差、CI95、胡牌率、对 S3 胜率以及非法动作/零和统计。

## 设备保护

- Windows CPU：优先 LibreHardwareMonitor WMI；不可用时为 unknown，禁止使用 `MSAcpi_ThermalZoneTemperature`。
- NVIDIA GPU：`nvidia-smi`。
- Ubuntu CPU：`sensors -j`，回退 `/sys/class/hwmon`。
- A CPU >85°C 或训练盘剩余 <10 GiB 时创建 stop flag；避免重复触发。
- B CPU >90°C 停止领取评估，≤80°C 自动恢复。
- 温度每 10 秒采样并保留最近 24 小时 JSONL。

## API 与界面

A：`POST /start`、`POST /pause`、`GET /status`、`GET /checkpoints`、`GET /checkpoints/{name}`、`GET /log_tail`。

B：根页面、聚合状态、历史指标、评估结果、转发 start/pause。页面展示控制区、双机健康、训练曲线、强度曲线和事件日志；响应式适配手机。

## 部署

提供 Windows PowerShell 启动脚本和计划任务安装模板，以及 Ubuntu systemd unit 和环境文件模板。服务启动不等于自动开训。配置样例纳入 Git，真实配置和运行状态忽略。

## 验收

自动化测试覆盖：暂停/恢复连续性、指标格式、状态机、重启自愈、白名单参数、路径保护、温度/磁盘守护、checkpoint 校验去重、最新优先评估、API、静态页面和模拟 A/B 全链路。最后运行 S5 专项及全量 pytest 回归。真实 LHM、systemd、计划任务和局域网联调在代码完成后现场验收。
