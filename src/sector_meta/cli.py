"""Command-line entry point; downloads and offline analysis are separate commands."""

import argparse
import json
from pathlib import Path

from .data import acquire


def main():
    parser = argparse.ArgumentParser(description="Math 628 sector ETF research")
    parser.add_argument("command", choices=["download", "stage1", "cnn-visualize", "cnn-develop", "cnn-test"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--cnn-config", type=Path, default=None)
    parser.add_argument("--factors", type=Path, default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    cfg = json.loads((args.config or root / "config/stage1.json").read_text())
    if args.command == "download":
        manifest = acquire(root, cfg, args.factors, args.refresh)
        print(f"Acquired/verified {len(manifest['files'])} files in {root / 'data/raw'}")
    elif args.command == "stage1":
        from .pipeline import run_stage1
        run_stage1(root, cfg)
    else:
        from .data import audit_and_load

        spec = json.loads((args.cnn_config or root / "config/cnn.json").read_text())
        prices, calendar, vix, _, manifest = audit_and_load(root, cfg)
        if args.command == "cnn-visualize":
            from .cnn_viz import create_input_figure
            create_input_figure(root, prices, calendar, vix, cfg, spec)
        else:
            from .cnn_experiment import develop, evaluate
            command = develop if args.command == "cnn-develop" else evaluate
            command(root, prices, calendar, vix, cfg, spec, manifest)


if __name__ == "__main__":
    main()
