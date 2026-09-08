from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from .config import load_config
from .runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark robust tabular learning models")
    parser.add_argument("config", help="Path to a YAML benchmark configuration")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore an existing checkpoint and start the configured run again",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.restart:
        config = replace(config, resume=False)
    results = run_benchmark(config)
    counts = results["status"].value_counts().to_dict() if not results.empty else {}
    print(f"Completed {len(results)} runs: {counts}. Results: {config.output_dir}")
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
