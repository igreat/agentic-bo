#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bo_workflow.config import load_experiment_file
from src.bo_workflow.experiment import run_experiment
from src.bo_workflow.problems.her import build_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HER BO experiment")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/her.toml"),
        help="Path to TOML experiment config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, problem_kwargs = load_experiment_file(args.config)
    problem = build_problem(**problem_kwargs)
    run_experiment(config=config, problem=problem)


if __name__ == "__main__":
    main()
