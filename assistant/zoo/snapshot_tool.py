"""一键冻结当前工作区(或指定历史 git commit)的辅助器策略到 assistant/zoo/<version>/。

示例:
    # 1) 归档当前最新策略为 v4_xxx
    python -m assistant.zoo.snapshot_tool --name v4_my_upgrade --note "升级说明"

    # 2) 从历史 git commit 补档归档 (例如把旧版 8cb0cfc 存为 v2_ev_base)
    python -m assistant.zoo.snapshot_tool --name v2_ev_base --git-ref 8cb0cfc --note "初代完整 AssistantPolicy"
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZOO_DIR = Path(__file__).resolve().parent

# AssistantPolicy 运行时所需的全部 assistant/ 内部模块(完全自包含)
RUNTIME_FILES = (
    "alerts.py",
    "bot.py",
    "counting.py",
    "danger.py",
    "discard_v2.py",
    "known_holdings.py",
    "opponent_read.py",
    "own_hand.py",
    "report.py",
    "state.py",
    "swap_v1.py",
    "swap_v2.py",
    "tactics.py",
    "tiles.py",
    "void_v3.py",
    "winner_inference.py",
    "adapters/from_engine.py",
)

# 保持指向共享的 PublicRecorder, 使不同版本快照能接入同一套对局记录器
KEEP_SHARED = {"recorder"}


def _git_output(args: list[str]) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return res.stdout


def _read_source(rel_path: str, git_ref: str | None) -> str:
    if git_ref is None:
        return (ROOT / "assistant" / rel_path).read_text(encoding="utf-8")
    return _git_output(["show", f"{git_ref}:assistant/{rel_path}"])


def _rewrite_imports(src: str, is_adapter: bool) -> str:
    """将 `from assistant.xxx import ...` 重写为包内相对导入, 实现 100% 版本隔离。"""
    prefix = ".." if is_adapter else "."

    def _repl(match: re.Match[str]) -> str:
        indent = match.group(1)
        mod = match.group(2)
        top = mod.split(".", 1)[0]
        if top in KEEP_SHARED:
            return match.group(0)
        return f"{indent}from {prefix}{mod} import"

    return re.sub(r"^(\s*)from assistant\.([a-zA-Z0-9_.]+) import", _repl, src, flags=re.MULTILINE)


def create_snapshot(
    name: str,
    *,
    note: str = "",
    git_ref: str | None = None,
    benchmark: dict | None = None,
    overwrite: bool = False,
) -> Path:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_]*", name):
        raise ValueError(f"版本名必须是合法 Python 模块名(字母/数字/下划线): {name!r}")

    target = ZOO_DIR / name
    if target.exists() and not overwrite:
        raise FileExistsError(f"版本目录已存在: {target} (如需覆盖请加 --overwrite)")

    commit_hash = _git_output(["rev-parse", "--short", git_ref or "HEAD"]).strip()
    commit_subject = _git_output(["log", "-1", "--format=%s", git_ref or "HEAD"]).strip()

    pkg_dir = target / "pkg"
    (pkg_dir / "adapters").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text('"""Frozen assistant runtime snapshot."""\n', encoding="utf-8")
    (pkg_dir / "adapters" / "__init__.py").write_text("", encoding="utf-8")

    for rel in RUNTIME_FILES:
        raw = _read_source(rel, git_ref)
        rewritten = _rewrite_imports(raw, is_adapter=rel.startswith("adapters/"))
        out_path = pkg_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rewritten, encoding="utf-8", newline="\n")

    init_code = f'''"""冻结策略版本: {name} (git {commit_hash})。"""
from __future__ import annotations

from .pkg.bot import AssistantPolicy


def build_policy(
    *,
    stochastic: bool = False,
    seed: int | None = None,
    exact_swap_live: bool = False,
) -> AssistantPolicy:
    return AssistantPolicy(
        stochastic=stochastic,
        seed=seed,
        exact_swap_live=exact_swap_live,
    )


__all__ = ["AssistantPolicy", "build_policy"]
'''
    (target / "__init__.py").write_text(init_code, encoding="utf-8", newline="\n")

    manifest = {
        "version": name,
        "git_commit": commit_hash,
        "git_subject": commit_subject,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "note": note or commit_subject,
        "benchmark": benchmark or {},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结当前(或指定 commit)的辅助器策略到 assistant/zoo/")
    parser.add_argument("--name", required=True, help="版本文件夹名, 如 v4_endgame_plus")
    parser.add_argument("--note", default="", help="版本升级说明")
    parser.add_argument("--git-ref", default=None, help="可选: 从历史 git commit 打包(默认打包当前工作区)")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有同名版本")
    args = parser.parse_args()

    out = create_snapshot(args.name, note=args.note, git_ref=args.git_ref, overwrite=args.overwrite)
    print(f"已归档冻结策略快照: {out}")


if __name__ == "__main__":
    main()
