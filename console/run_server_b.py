from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from console.config import load_config
from console.server_b import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="S5 评估机 B 控制台")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    uvicorn.run(create_app(config), host=config.server.host, port=config.server.port)


if __name__ == "__main__":
    main()
