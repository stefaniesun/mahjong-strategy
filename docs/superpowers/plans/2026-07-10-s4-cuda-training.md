# S4 CUDA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run S4 training and model evaluation on CUDA when available while preserving explicit CPU execution and automatic CPU fallback.

**Architecture:** A small shared device resolver validates `auto`, `cpu`, and `cuda`. Training batches expose a type-safe `.to(device)` method, training epochs move their model and each mini-batch to the resolved device, and evaluators infer the model device before creating temporary tensors. The cloud entry point reports the resolved device and passes it into both epoch trainers.

**Tech Stack:** Python 3.10+, PyTorch, pytest.

---

### Task 1: Add device resolution with a failing test

**Files:**
- Create: `learning/device.py`
- Create: `tests/test_s4_device.py`

- [ ] **Step 1: Write the failing resolver tests**

```python
import pytest

from learning.device import resolve_device


def test_resolve_device_supports_cpu_and_auto():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_resolve_device_rejects_unknown_requests():
    with pytest.raises(ValueError, match="auto, cpu, or cuda"):
        resolve_device("metal")
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `python -m pytest tests/test_s4_device.py -q`

Expected: import failure because `learning.device` does not exist.

- [ ] **Step 3: Implement the minimal resolver**

```python
from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return torch.device("cuda")
    raise ValueError("device must be one of: auto, cpu, or cuda")
```

- [ ] **Step 4: Run the resolver tests and verify GREEN**

Run: `python -m pytest tests/test_s4_device.py -q`

Expected: `2 passed` on CPU-only CI; `auto` may resolve to CUDA on the cloud host.

- [ ] **Step 5: Commit**

Not applicable: this workspace has no `.git` directory.

### Task 2: Move S4 training batches and models to the selected device

**Files:**
- Modify: `learning/training/train_belief.py`
- Modify: `learning/training/train_policy.py`
- Modify: `tests/test_s4_train_belief.py`
- Modify: `tests/test_s4_train_policy.py`

- [ ] **Step 1: Write failing CPU-device epoch tests**

```python
model, metrics = train_belief_epoch(samples, TrainBeliefConfig(
    model=model_config, batch_size=2, device="cpu",
))
assert next(model.parameters()).device.type == "cpu"
assert metrics["device"] == "cpu"
```

Use the corresponding `train_policy_epoch` call and assertions in the policy test.

- [ ] **Step 2: Run targeted tests and verify RED**

Run: `python -m pytest tests/test_s4_train_belief.py tests/test_s4_train_policy.py -q`

Expected: `TrainBeliefConfig` and `TrainPolicyConfig` reject the `device` keyword.

- [ ] **Step 3: Implement batch/device movement**

Add `device: str = "auto"` to both train config dataclasses. Add `.to(device)` methods to `BeliefBatch` and `PolicyBatch` that return a new batch with every tensor moved through `tensor.to(device)`. In each epoch, resolve the config device, call `model.to(device)`, move each mini-batch before its training step, and append `"device": device.type` to returned metrics. Keep direct step functions device-agnostic so existing callers can use matching model and batch devices.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run: `python -m pytest tests/test_s4_train_belief.py tests/test_s4_train_policy.py tests/test_s4_device.py -q`

Expected: all tests pass on CPU.

- [ ] **Step 5: Add a CUDA-only integration test**

```python
@pytest.mark.skipif(not torch().cuda.is_available(), reason="CUDA unavailable")
def test_belief_epoch_moves_model_to_cuda():
    model, metrics = train_belief_epoch(samples, TrainBeliefConfig(model=model_config, device="cuda"))
    assert next(model.parameters()).device.type == "cuda"
    assert metrics["device"] == "cuda"
```

Add the matching policy epoch assertion.

- [ ] **Step 6: Commit**

Not applicable: this workspace has no `.git` directory.

### Task 3: Make model evaluation device-safe

**Files:**
- Modify: `learning/eval/eval_belief.py`
- Modify: `learning/eval/eval_policy.py`
- Modify: `tests/test_s4_eval_belief.py`
- Modify: `tests/test_s4_eval_policy_arena.py`

- [ ] **Step 1: Write failing CUDA evaluation tests**

```python
@pytest.mark.skipif(not torch().cuda.is_available(), reason="CUDA unavailable")
def test_evaluate_belief_model_accepts_cuda_model():
    report = evaluate_belief_model(model.cuda(), samples)
    assert report.samples == len(samples)
```

Add the same `model.cuda()` shape for policy evaluation.

- [ ] **Step 2: Run targeted tests and verify RED on a CUDA host**

Run: `python -m pytest tests/test_s4_eval_belief.py tests/test_s4_eval_policy_arena.py -q`

Expected on CUDA: device mismatch between CPU features and CUDA model.

- [ ] **Step 3: Implement model-device inference**

Use `next(model.parameters()).device` in each evaluator. Move temporary belief feature tensors to that device and call `policy_batch_from_samples(samples).to(device)` before policy inference. Keep every metric conversion through `.detach().cpu()`.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run: `python -m pytest tests/test_s4_eval_belief.py tests/test_s4_eval_policy_arena.py tests/test_s4_device.py -q`

Expected: CPU tests pass locally; CUDA tests pass on the cloud host and skip locally without CUDA.

- [ ] **Step 5: Commit**

Not applicable: this workspace has no `.git` directory.

### Task 4: Wire device selection into the cloud entry point

**Files:**
- Modify: `tools/cloud_train_s4.py`
- Modify: `tests/test_s4_cloud_training_package.py`
- Modify: `CLOUD_TRAINING_README.md`

- [ ] **Step 1: Write a failing cloud CPU configuration test**

```python
result = run_cloud_training(CloudTrainingConfig(..., device="cpu"))
report = json.loads(result.json_report.read_text(encoding="utf-8"))
assert report["execution"]["device"] == "cpu"
```

- [ ] **Step 2: Run the package test and verify RED**

Run: `python -m pytest tests/test_s4_cloud_training_package.py -q`

Expected: `CloudTrainingConfig` rejects the `device` keyword.

- [ ] **Step 3: Implement CLI/report wiring**

Add `device: str = "auto"` to `CloudTrainingConfig` and `--device {auto,cpu,cuda}` to argument parsing. Resolve it once in `run_cloud_training`, pass `device.type` into both training configs, write `{"device": device.type}` under a new `execution` report section, and include that section in the Markdown report. Document `--device auto` as the default and `--device cuda` as a strict cloud-only mode.

- [ ] **Step 4: Run the cloud package test and verify GREEN**

Run: `python -m pytest tests/test_s4_cloud_training_package.py -q`

Expected: `2 passed` and report execution device is `cpu` in its explicit CPU smoke test.

- [ ] **Step 5: Commit**

Not applicable: this workspace has no `.git` directory.

### Task 5: Full verification, packaging, and cloud restart

**Files:**
- Modify: generated `s4_cloud_training_package_20260710_cuda.zip`

- [ ] **Step 1: Run local regression**

Run: `python -m pytest -q`

Expected: all test modules pass; CUDA-only tests skip on a host without CUDA.

- [ ] **Step 2: Create and inspect the package**

Create the ZIP from source, configuration, documentation, and tests; exclude caches and prior archives. Read every ZIP entry and verify `tools/cloud_train_s4.py`, the device module, and the CUDA tests are present.

- [ ] **Step 3: Run cloud smoke training on CUDA**

Upload the package, run `python tools/cloud_train_s4.py --device cuda` with one game and a four-sample limit, then assert the report records `cuda` and that `nvidia-smi` observes a Python process during model work.

- [ ] **Step 4: Stop the previous CPU job only after CUDA smoke succeeds**

Use its recorded PID, wait for process termination, and preserve its log and partially generated data directory for audit.

- [ ] **Step 5: Start the CUDA full training job**

Run the existing full configuration with `--device cuda`, write a distinct PID/log/output path, and verify it remains alive after startup.
