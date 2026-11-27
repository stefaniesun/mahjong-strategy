"""人机对局网页服务。

用法:
    python -m play.server                       # 对手为 S4 v5
    python -m play.server --ai rule             # 对手为 S3 规则 AI(启动快,调试用)
    python -m play.server --ai s5 --checkpoint path/to/s5-step-900.pt
浏览器打开 http://127.0.0.1:8770
"""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from play.ai import load_ai
from play.session import GameSession

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
TILES = ROOT / "tools" / "mahjong_tiles"


def create_app(session: GameSession) -> FastAPI:
    app = FastAPI(title="四川麻将 · 血战到底")
    lock = threading.Lock()

    def respond(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {"events": events or [], "view": session.view()}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/tiles/{name}")
    def tile(name: str) -> FileResponse:
        path = TILES / Path(name).name
        if path.suffix != ".png" or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        with lock:
            return respond()

    @app.post("/api/advance")
    def advance() -> dict[str, Any]:
        with lock:
            return respond(session.advance())

    @app.post("/api/act")
    def act(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        with lock:
            try:
                events = session.act(payload)
            except (ValueError, KeyError) as exc:
                raise HTTPException(400, str(exc)) from exc
            return respond(events)

    @app.post("/api/next_hand")
    def next_hand() -> dict[str, Any]:
        with lock:
            if not session.state.finished:
                raise HTTPException(400, "这一局还没结束")
            session.new_hand()
            return respond()

    @app.post("/api/mark")
    def mark(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        with lock:
            return session.mark(str(payload.get("note", "")).strip())

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="四川麻将人机对局 + 打牌辅助器")
    parser.add_argument(
        "--ai",
        default="s4",
        help="对手类型: s4 / s5 / rule / rule_plus / assistant / zoo:<版本名> (如 zoo:v2_ev_base)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--seed", type=int, default=None, help="固定随机种子,便于复现")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    print("正在加载 AI 模型…", flush=True)
    ai, ai_name = load_ai(args.ai, args.checkpoint)
    log_dir = args.log_dir or ROOT / "outputs" / "play" / time.strftime("%Y%m%d_%H%M%S")
    session = GameSession(ai, ai_name=ai_name, log_dir=log_dir, rng_seed=args.seed)
    url = f"http://127.0.0.1:{args.port}"
    print(f"对手:{ai_name}\n对局记录与标记保存在:{log_dir}\n打开 {url}", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(session), host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
