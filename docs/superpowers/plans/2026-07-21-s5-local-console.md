# S5 Local Console Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete repository-side S5 two-machine local training console with graceful pause/resume, metrics, health guards, checkpoint evaluation, offline UI, and deployment templates.

**Architecture:** Keep training changes narrow and optional, place orchestration in a new `console` package, and persist all runtime data as atomic JSON/JSONL files. Agent A owns the training process; Server B aggregates history and evaluates newest checkpoints.

**Tech Stack:** Python 3, PyTorch, FastAPI, uvicorn, PyYAML, pytest, native HTML/CSS/JavaScript/SVG.

---

### Task 1: Graceful training lifecycle and metrics

**Files:**
- Modify: `rl/train_rl.py`
- Modify: `tools/cloud_train_s5.py`
- Test: `tests/test_s5_local_lifecycle.py`

- [ ] Write failing tests proving stop is checked only after an atomic latest checkpoint, metrics contain all required fields, and resume continues update/global-step/episode counters.
- [ ] Run `python -m pytest tests/test_s5_local_lifecycle.py -q` and confirm RED.
- [ ] Add optional `stop_file`, `metrics_file`, and episode-count metadata to `S5TrainingConfig`; append one durable JSON line after every completed update without changing training operation order.
- [ ] Add `resume_checkpoint`, stop, and metrics fields to `S5CloudRunConfig` and CLI flags `--resume`, `--stop-file`, `--metrics-file`.
- [ ] Run the lifecycle test and existing checkpoint/train smoke tests; expect PASS.

### Task 2: Shared console configuration and persistence

**Files:**
- Create: `console/__init__.py`
- Create: `console/config.py`
- Create: `console/storage.py`
- Create: `configs/s5_console.example.yaml`
- Modify: `.gitignore`
- Test: `tests/test_s5_console_config.py`

- [ ] Write failing tests for strict YAML parsing, defaults, invalid thresholds, allowed override fields, safe descendant paths, atomic JSON, and tolerant JSONL tailing.
- [ ] Run `python -m pytest tests/test_s5_console_config.py -q`; confirm RED.
- [ ] Implement frozen dataclass configuration for A/B/training/health/evaluation and reject unknown keys.
- [ ] Implement atomic JSON writes, durable JSONL append/tail, path containment, and SHA-256 helpers.
- [ ] Add a complete example configuration and ignore only real runtime/config-secret files.
- [ ] Run the configuration tests; expect PASS.

### Task 3: Hardware telemetry and guards

**Files:**
- Create: `console/telemetry.py`
- Test: `tests/test_s5_console_telemetry.py`

- [ ] Write failing tests for LibreHardwareMonitor parsing, `nvidia-smi`, `sensors -j`, hwmon fallback, memory/disk reporting, 24-hour retention, and A/B hysteresis decisions.
- [ ] Run the telemetry tests; confirm RED.
- [ ] Implement injectable command/file readers and platform-specific collectors without using Windows ACPI thermal zones.
- [ ] Implement A >85°C / <10 GiB pause decision and B >90°C pause / ≤80°C resume hysteresis.
- [ ] Run telemetry tests; expect PASS.

### Task 4: Machine A process manager and API

**Files:**
- Create: `console/agent_a.py`
- Create: `console/run_agent_a.py`
- Test: `tests/test_s5_agent_a.py`

- [ ] Write failing tests for all state transitions, whitelist-only command construction, no shell invocation, graceful pause, natural completion, abnormal exit, restart recovery, guards, checkpoint listing/download traversal rejection, status, and log tail.
- [ ] Run agent tests; confirm RED.
- [ ] Implement a lock-protected process manager with atomic state, process monitoring, stop flag creation, and startup recovery that never auto-starts training.
- [ ] Implement FastAPI routes `POST /start`, `POST /pause`, `GET /status`, `GET /checkpoints`, `GET /checkpoints/{name}`, and `GET /log_tail`.
- [ ] Implement periodic telemetry sampling and automatic graceful pause reasons.
- [ ] Run agent tests; expect PASS.

### Task 5: S5 independent evaluation adapter

**Files:**
- Create: `console/evaluator.py`
- Test: `tests/test_s5_console_evaluator.py`

- [ ] Write failing tests for loading a complete S5 checkpoint, adapting policy inference to the production arena, fixed 3×S3 opponents, quick/formal budgets, seed 90000 formal semantics, and serializable statistics.
- [ ] Run evaluator tests; confirm RED.
- [ ] Implement the thin S5 checkpoint policy adapter using existing encoder/action mapping and `run_production_arena` APIs.
- [ ] Report checkpoint SHA-256, games, seed, average score difference, CI95, win rate, illegal actions, zero-sum violations, and global step/episode metadata.
- [ ] Run evaluator and existing arena adapter tests; expect PASS.

### Task 6: Machine B synchronization and evaluation scheduler

**Files:**
- Create: `console/scheduler.py`
- Test: `tests/test_s5_console_scheduler.py`

- [ ] Write failing tests for polling A, temporary download, size/SHA verification, atomic archive, retention, disk protection, stable task IDs, restart deduplication, newest-first backlog collapse, daily formal selection, and thermal pause/resume.
- [ ] Run scheduler tests; confirm RED.
- [ ] Implement an injectable HTTP client and evaluator runner, local historical metric cache, checkpoint repository, and single-worker scheduler.
- [ ] Persist events and evaluations as JSONL; make completed task reconstruction deterministic.
- [ ] Run scheduler tests; expect PASS.

### Task 7: Machine B API and offline responsive UI

**Files:**
- Create: `console/server_b.py`
- Create: `console/run_server_b.py`
- Create: `console/static/index.html`
- Test: `tests/test_s5_server_b.py`

- [ ] Write failing API tests for online/offline aggregation, history, evaluations, event log, start/pause forwarding, and static asset delivery with no CDN references.
- [ ] Run server tests; confirm RED.
- [ ] Implement FastAPI lifecycle tasks for polling, telemetry, synchronization, and evaluation.
- [ ] Implement a single responsive page with control/health panels, hand-written SVG training and strength charts, 24-hour/all-time switch, and event log.
- [ ] Run server tests; expect PASS.

### Task 8: Deployment templates and operations guide

**Files:**
- Create: `console/README.md`
- Create: `console/deploy/windows/start-agent.ps1`
- Create: `console/deploy/windows/install-scheduled-task.ps1`
- Create: `console/deploy/linux/s5-console.service`
- Create: `console/deploy/linux/s5-console.env.example`
- Modify: `requirements.txt`
- Test: `tests/test_s5_console_deployment.py`

- [ ] Write failing tests checking templates reference existing entry points, disable auto-training, avoid public bind defaults, and contain no CDN/Docker requirements.
- [ ] Run deployment tests; confirm RED.
- [ ] Add bounded FastAPI/uvicorn dependencies and reproducible Windows/Ubuntu templates.
- [ ] Document installation, LHM/lm-sensors, firewall, fixed LAN addresses, startup, recovery, backups, and complete normal/abnormal drill checklists.
- [ ] Run deployment tests; expect PASS.

### Task 9: End-to-end simulation and regression

**Files:**
- Create: `tests/test_s5_console_e2e.py`
- Modify only as required by verified failures in prior files.

- [ ] Add a bounded fake-process/fake-agent end-to-end test: start, metrics, checkpoint sync, quick evaluation, graceful pause, resume, offline history, and guard events.
- [ ] Run all console tests and `tests/test_s5_*.py`; expect PASS.
- [ ] Run `python -m pytest -q`; expect the full suite to pass.
- [ ] Inspect lints and `git diff --check`; resolve all introduced issues.
- [ ] Commit and push the verified implementation to the current branch according to the project preference.
