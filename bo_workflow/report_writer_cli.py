"""CLI for generating narrative report text from a BO run."""

from __future__ import annotations

import argparse
import json
import sys

from .report_writer import write_report_sections


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate written report sections from a BO run"
    )
    parser.add_argument("run_id", help="Run ID, e.g. jolly-badger-9410")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing run folders",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the generated JSON to stdout",
    )
    args = parser.parse_args()

    try:
        payload = write_report_sections(args.run_id, runs_root=args.runs_root)
        print(f"Written report saved to {args.runs_root}/{args.run_id}/written_report.json")
        if args.stdout:
            print(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()