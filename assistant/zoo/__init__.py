"""策略版本库 (assistant.zoo): 每次迭代升级的独立冻结快照与对手池管理。

用途:
1. 纵向对比: 直接让 v2 vs v3 vs v4 在同一批牌上同桌对打, 精确测量每次升级净涨幅;
2. 训练对手池多样性: 后续做监督学习(SL)数据生成或强化学习(RL)对手池(League)时,
   可同时混用历代快照(v2/v3/v4...)与不同随机度/风格, 防止神经网络对单一策略过拟合;
3. 100% 隔离冻结: 每个版本文件夹内的 pkg/ 使用相对导入, 后续继续迭代 assistant/ 主线
   绝不会破坏或污染历史版本的行为。

快速上手:
    from assistant.zoo import list_versions, load_version, build_opponent_pool

    print(list_versions())
    v2 = load_version("v2_ev_base", stochastic=False)
    v3 = load_version("v3_goodwait_read", stochastic=True, seed=42)
    pool = build_opponent_pool(stochastic=True)
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

ZOO_DIR = Path(__file__).resolve().parent


def _version_dirs() -> list[Path]:
    if not ZOO_DIR.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(ZOO_DIR.iterdir()):
        if p.is_dir() and (p / "manifest.json").is_file() and (p / "__init__.py").is_file():
            out.append(p)
    return out


def list_versions() -> list[str]:
    """返回版本库中已归档的所有策略版本名(按字典序排列)。"""
    return [p.name for p in _version_dirs()]


def get_manifest(version: str) -> dict[str, Any]:
    """读取指定版本的元信息 (manifest.json)。"""
    mpath = ZOO_DIR / version / "manifest.json"
    if not mpath.is_file():
        avail = ", ".join(list_versions()) or "(空)"
        raise KeyError(f"未找到策略版本 '{version}', 当前可用版本: {avail}")
    return json.loads(mpath.read_text(encoding="utf-8"))


def list_manifests() -> list[dict[str, Any]]:
    """返回所有版本的元信息列表。"""
    return [get_manifest(v) for v in list_versions()]


def load_version(
    version: str,
    *,
    stochastic: bool = False,
    seed: int | None = None,
    exact_swap_live: bool = False,
):
    """加载指定冻结版本的独立策略实例 (具 choose_from_game 接口)。"""
    # 校验版本存在
    get_manifest(version)
    mod = importlib.import_module(f"assistant.zoo.{version}")
    return mod.build_policy(
        stochastic=stochastic,
        seed=seed,
        exact_swap_live=exact_swap_live,
    )


def parse_zoo_spec(name: str) -> tuple[str, bool] | None:
    """解析策略标识符, 返回 (version_name, stochastic) 或 None。

    支持写法:
    - "zoo:v2_ev_base"            -> ("v2_ev_base", True)
    - "zoo:v2_ev_base:det"        -> ("v2_ev_base", False)
    - "zoo:v2_ev_base:rand"       -> ("v2_ev_base", True)
    - "v2_ev_base_det"            -> ("v2_ev_base", False)
    - "v2_ev_base_rand"           -> ("v2_ev_base", True)
    - "v2_ev_base"                -> ("v2_ev_base", False)
    """
    versions = set(list_versions())
    if not versions:
        return None

    if name.startswith("zoo:"):
        rest = name[4:]
        if rest.endswith(":det"):
            ver = rest[:-4]
            if ver in versions:
                return ver, False
        elif rest.endswith(":rand"):
            ver = rest[:-5]
            if ver in versions:
                return ver, True
        elif rest in versions:
            return rest, True
        return None

    if name in versions:
        return name, False
    if name.endswith("_det") and name[:-4] in versions:
        return name[:-4], False
    if name.endswith("_rand") and name[:-5] in versions:
        return name[:-5], True
    return None


def build_opponent_pool(
    versions: list[str] | None = None,
    *,
    include_deterministic: bool = True,
    include_stochastic: bool = True,
    seed: int = 0,
    exact_swap_live: bool = False,
) -> dict[str, Any]:
    """一键构建多代策略混合对手池 (供后续 SL/RL 训练数据生成或 League 对战直接采样)。"""
    target_versions = versions if versions is not None else list_versions()
    pool: dict[str, Any] = {}
    for idx, ver in enumerate(target_versions):
        if include_deterministic:
            pool[f"{ver}_det"] = load_version(
                ver, stochastic=False, seed=seed + idx * 101, exact_swap_live=exact_swap_live
            )
        if include_stochastic:
            pool[f"{ver}_rand"] = load_version(
                ver, stochastic=True, seed=seed + idx * 101 + 17, exact_swap_live=exact_swap_live
            )
    return pool


def clear_all_zoo_caches() -> None:
    """清空所有已加载历史版本快照中的 lru_cache (多进程大样本长跑防 OOM)。"""
    suffixes = (".winner_inference", ".swap_v2", ".swap_v1")
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not mod_name.startswith("assistant.zoo."):
            continue
        if not mod_name.endswith(suffixes):
            continue
        for value in list(vars(mod).values()):
            clear = getattr(value, "cache_clear", None)
            if callable(clear):
                try:
                    clear()
                except Exception:
                    pass
