"""Command-line entry point; downloads and offline analysis are separate commands."""

import argparse
import json
from pathlib import Path

from .data import acquire


def main():
    parser = argparse.ArgumentParser(description="Math 628 sector ETF research")
    parser.add_argument("command", choices=["download", "stage1"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--factors", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    cfg = json.loads((args.config or root / "config/stage1.json").read_text())
    if args.command == "download":
        manifest = acquire(root, cfg, args.factors, args.refresh)
        print(f"Acquired/verified {len(manifest['files'])} files in {root / 'data/raw'}")
    else:
        from .pipeline import run_stage1
        run_stage1(root, cfg)


if __name__ == "__main__":
    main()
