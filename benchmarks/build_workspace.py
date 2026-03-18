"""Build a stripped public benchmark workspace from the current repo."""

import argparse
import json
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_claude_settings() -> dict:
    return {
        "defaultMode": "acceptEdits",
        "permissions": {
            "allow": ["Bash"],
            "deny": ["WebSearch", "WebFetch"],
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_workspace(
    *,
    output_dir: Path,
    task_ids: list[str],
    overwrite: bool = False,
) -> Path:
    root = repo_root()
    benchmarks_root = root / "benchmarks"
    tasks_root = benchmarks_root / "tasks"
    source_backends_root = root / "evaluation_backends"

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

    write_json(
        output_dir / ".claude" / "settings.local.json",
        benchmark_claude_settings(),
    )

    public_tasks_root = output_dir / "tasks"
    public_tasks_root.mkdir(parents=True, exist_ok=True)
    public_backends_root = output_dir / "evaluation_backends"

    for task_id in task_ids:
        src = tasks_root / task_id
        if not src.exists():
            raise FileNotFoundError(f"Unknown benchmark task bundle: {task_id}")
        dst = public_tasks_root / task_id
        copy_tree(src, dst, PUBLIC_TASK_IGNORE)

        manifest = load_json(dst / "task_manifest.json")
        backend_id = manifest.get("evaluation", {}).get("backend_id")
        if backend_id:
            source_backend = source_backends_root / str(backend_id)
            if source_backend.exists():
                copy_tree(
                    source_backend,
                    public_backends_root / str(backend_id),
                    COPYTREE_IGNORE,
                )

    (output_dir / "bo_runs").mkdir(parents=True, exist_ok=True)
    (output_dir / "research_runs").mkdir(parents=True, exist_ok=True)

    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a stripped public benchmark workspace."
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
    workspace = build_workspace(
        output_dir=args.output_dir,
        task_ids=list(args.tasks),
        overwrite=bool(args.overwrite),
    )
    print(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
