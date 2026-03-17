#!/usr/bin/env python3
"""Materialize a stripped public benchmark workspace from the current repo."""

import argparse
import shutil
from pathlib import Path

PUBLIC_ROOT_FILES = (
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
)

PUBLIC_ROOT_DIRS = (
    "bo_workflow",
    ".agents",
    ".claude",
)

PUBLIC_BENCHMARK_FILES = (
    "README.md",
    "run_task_evaluator.py",
)

COPYTREE_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    ".DS_Store",
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
)

PUBLIC_TASK_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    ".DS_Store",
    "assessment.md",
    "answer_key*",
    "*.private.*",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, ignore) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def materialize_workspace(
    *,
    output_dir: Path,
    task_ids: list[str],
    overwrite: bool = False,
) -> Path:
    root = repo_root()
    benchmarks_root = root / "benchmarks"
    template_root = benchmarks_root / "workspace_template"
    task_template_root = template_root / "benchmark_tasks"

    if output_dir.exists():
        if not overwrite and any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        if overwrite:
            shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for rel_path in PUBLIC_ROOT_FILES:
        copy_file(root / rel_path, output_dir / rel_path)

    for rel_path in PUBLIC_ROOT_DIRS:
        copy_tree(root / rel_path, output_dir / rel_path, COPYTREE_IGNORE)

    for rel_path in PUBLIC_BENCHMARK_FILES:
        copy_file(benchmarks_root / rel_path, output_dir / "benchmarks" / rel_path)

    task_manifest_dir = output_dir / "benchmark_tasks"
    task_manifest_dir.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        src = task_template_root / task_id
        if not src.exists():
            raise FileNotFoundError(f"Unknown benchmark task bundle: {task_id}")
        copy_tree(src, task_manifest_dir / task_id, PUBLIC_TASK_IGNORE)

    (output_dir / "bo_runs").mkdir(parents=True, exist_ok=True)
    (output_dir / "research_runs").mkdir(parents=True, exist_ok=True)

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize a stripped public benchmark workspace."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Benchmark task bundle ids to expose in the public workspace.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = materialize_workspace(
        output_dir=args.output_dir,
        task_ids=list(args.tasks),
        overwrite=bool(args.overwrite),
    )
    print(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
