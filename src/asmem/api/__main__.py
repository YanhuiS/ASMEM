"""``python -m asmem.api [--host H] [--port P] [--recover]``."""

from __future__ import annotations

import argparse
import asyncio

from .server import create_app, load_config
from .store import recover_all


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m asmem.api")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default="")
    parser.add_argument("--recover", action="store_true",
                        help="settle interrupted adds, then exit")
    args = parser.parse_args(argv)
    config = load_config(args.config or None)
    if args.recover:
        from pathlib import Path

        from ..io import PROJECT_ROOT

        root = Path(str(config.get("store_dir") or "data/api_store"))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        print(f"recovered {asyncio.run(recover_all(root))} user(s)")
        return

    import uvicorn

    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
