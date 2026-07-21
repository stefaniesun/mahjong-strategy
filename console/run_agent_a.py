from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from console.agent_a import create_app
from console.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="S5 训练机 A 代理")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    uvicorn.run(create_app(config), host=config.agent.host, port=config.agent.port)


if __name__ == "__main__":
    main()
