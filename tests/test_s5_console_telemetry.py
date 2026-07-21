from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from console.config import HealthConfig
from console.storage import append_jsonl, read_jsonl
from console.telemetry import TelemetrySample, agent_pause_reason, evaluator_paused, parse_lhm_json, parse_nvidia_smi, parse_sensors_json, read_hwmon, record_telemetry


def sample(cpu: float | None, disk: float = 20) -> TelemetrySample:
    return TelemetrySample(datetime.now(timezone.utc).isoformat(), cpu, 60, 1, 2, disk)


def test_temperature_parsers_and_hwmon(tmp_path: Path) -> None:
    assert parse_nvidia_smi("67\n") == 67
    assert parse_sensors_json(json.dumps({"coretemp": {"temp1": {"temp1_input": 71.5}}})) == 71.5
    assert parse_lhm_json(json.dumps({"Children": [{"Name": "CPU Package", "SensorType": "Temperature", "Value": 69}]})) == 69
    root = tmp_path / "hwmon"
    target = root / "hwmon0"
    target.mkdir(parents=True)
    (target / "name").write_text("coretemp")
    (target / "temp1_input").write_text("72500")
    nvme = root / "hwmon1"
    nvme.mkdir()
    (nvme / "name").write_text("nvme")
    (nvme / "temp1_input").write_text("99000")
    assert read_hwmon(root) == 72.5


def test_agent_guards_and_evaluator_hysteresis() -> None:
    config = HealthConfig()
    assert agent_pause_reason(sample(86), config) == "cpu_overheat"
    assert agent_pause_reason(sample(70, 9), config) == "low_disk"
    assert agent_pause_reason(sample(None), config) is None
    assert evaluator_paused(sample(91), config, False) is True
    assert evaluator_paused(sample(85), config, True) is True
    assert evaluator_paused(sample(80), config, True) is False
    assert evaluator_paused(sample(None), config, True) is True


def test_telemetry_retention(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.jsonl"
    old = sample(60)
    object.__setattr__(old, "timestamp", (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat())
    append_jsonl(path, {"timestamp": old.timestamp, "cpu_temp_c": 60})
    record_telemetry(path, sample(70), 24)
    rows = read_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["cpu_temp_c"] == 70
