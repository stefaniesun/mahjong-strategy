from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from console.config import HealthConfig
from console.storage import append_jsonl, read_jsonl, rewrite_jsonl


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    timestamp: str
    cpu_temp_c: float | None
    gpu_temp_c: float | None
    memory_used_gib: float | None
    memory_total_gib: float | None
    disk_free_gib: float


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout


def parse_nvidia_smi(text: str) -> float | None:
    for line in text.splitlines():
        try:
            return float(line.strip().split(",", 1)[0])
        except ValueError:
            continue
    return None


def _valid_temperature(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and -20 <= value <= 150


def parse_sensors_json(text: str) -> float | None:
    data = json.loads(text)
    values: list[float] = []
    if not isinstance(data, dict):
        return None
    for chip, sensors in data.items():
        if not any(marker in chip.lower() for marker in ("coretemp", "k10temp", "zenpower", "cpu")) or not isinstance(sensors, dict):
            continue
        for sensor in sensors.values():
            if not isinstance(sensor, dict):
                continue
            for key, value in sensor.items():
                if key.endswith("_input") and _valid_temperature(value):
                    values.append(float(value))
    return max(values) if values else None


def parse_lhm_json(text: str) -> float | None:
    data = json.loads(text)
    values: list[float] = []
    def walk(value: object) -> None:
        if isinstance(value, dict):
            sensor_type = str(value.get("SensorType", value.get("Type", ""))).lower()
            name = str(value.get("Name", "")).lower()
            reading = value.get("Value")
            if sensor_type == "temperature" and "cpu" in name and _valid_temperature(reading):
                values.append(float(reading))
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(data)
    return max(values) if values else None


def read_hwmon(root: Path = Path("/sys/class/hwmon")) -> float | None:
    values: list[float] = []
    if not root.exists():
        return None
    for device in root.glob("hwmon*"):
        try:
            name = (device / "name").read_text().strip().lower()
        except OSError:
            name = "coretemp" if root != Path("/sys/class/hwmon") else ""
        if not any(marker in name for marker in ("coretemp", "k10temp", "zenpower", "cpu")):
            continue
        for path in device.glob("temp*_input"):
            try:
                value = float(path.read_text().strip()) / 1000
                if _valid_temperature(value):
                    values.append(value)
            except (OSError, ValueError):
                continue
    return max(values) if values else None


def memory_usage() -> tuple[float | None, float | None]:
    if platform.system() == "Windows":
        try:
            output = _run(["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory,TotalVisibleMemorySize | ConvertTo-Json -Compress"])
            value = json.loads(output)
            total = float(value["TotalVisibleMemorySize"]) / 1024 / 1024
            free = float(value["FreePhysicalMemory"]) / 1024 / 1024
            return total - free, total
        except Exception:
            return None, None
    try:
        values: dict[str, float] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            values[key] = float(raw.strip().split()[0]) / 1024 / 1024
        return values["MemTotal"] - values.get("MemAvailable", values.get("MemFree", 0)), values["MemTotal"]
    except Exception:
        return None, None


def _read_lhm_wmi() -> str:
    command = (
        "Get-CimInstance -Namespace root/LibreHardwareMonitor -ClassName Sensor "
        "-Filter \"SensorType='Temperature'\" | Select-Object Name,SensorType,Value | ConvertTo-Json -Compress"
    )
    return _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command])


def collect_telemetry(disk_path: Path, *, runner: Callable[[list[str]], str] = _run, lhm_reader: Callable[[], str] | None = None) -> TelemetrySample:
    cpu: float | None = None
    system = platform.system()
    if system == "Windows":
        try:
            cpu = parse_lhm_json((lhm_reader or _read_lhm_wmi)())
        except Exception:
            cpu = None
    elif system != "Windows":
        try:
            cpu = parse_sensors_json(runner(["sensors", "-j"]))
        except Exception:
            cpu = read_hwmon()
    try:
        gpu = parse_nvidia_smi(runner(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]))
        if gpu is not None and not _valid_temperature(gpu):
            gpu = None
    except Exception:
        gpu = None
    used, total = memory_usage()
    free = shutil.disk_usage(disk_path).free / 1024 ** 3
    return TelemetrySample(datetime.now(timezone.utc).isoformat(), cpu, gpu, used, total, free)


def agent_pause_reason(sample: TelemetrySample, config: HealthConfig) -> str | None:
    if sample.disk_free_gib < config.agent_disk_pause_gib:
        return "low_disk"
    if sample.cpu_temp_c is not None and sample.cpu_temp_c > config.agent_cpu_pause_c:
        return "cpu_overheat"
    return None


def evaluator_paused(sample: TelemetrySample, config: HealthConfig, was_paused: bool) -> bool:
    if sample.cpu_temp_c is None:
        return was_paused
    if was_paused:
        return sample.cpu_temp_c > config.evaluator_cpu_resume_c
    return sample.cpu_temp_c > config.evaluator_cpu_pause_c


def record_telemetry(path: Path, sample: TelemetrySample, retention_hours: int = 24) -> None:
    append_jsonl(path, asdict(sample))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
    rows = [row for row in read_jsonl(path) if datetime.fromisoformat(str(row["timestamp"])) >= cutoff]
    rewrite_jsonl(path, rows)
