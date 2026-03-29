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

RESEARCH_LAYER_SKILL_DIRS = (
    ".agents/skills/research-agent",
    ".agents/skills/literature-review",
    ".agents/skills/scientific-writing",
    ".agents/skills/evaluator-design",
    ".claude/skills/research-agent",
    ".claude/skills/literature-review",
    ".claude/skills/scientific-writing",
    ".claude/skills/evaluator-design",
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


def bo_only_template_root() -> Path:
    return repo_root() / "benchmarks" / "templates" / "open_world_reruns" / "bo_only"


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


def apply_skill_profile(output_dir: Path, *, skill_profile: str) -> None:
    if skill_profile == "full":
        return
    if skill_profile != "bo_only":
        raise ValueError(f"Unknown skill profile: {skill_profile}")

    for rel_path in RESEARCH_LAYER_SKILL_DIRS:
        target = output_dir / rel_path
        if target.exists():
            shutil.rmtree(target)

    templates = bo_only_template_root()
    copy_file(templates / "AGENTS.md", output_dir / "AGENTS.md")
    copy_file(templates / "CLAUDE.md", output_dir / "CLAUDE.md")


def validate_output_dir(output_dir: Path, *, root: Path, overwrite: bool) -> Path:
    resolved_output = output_dir.resolve()
    resolved_root = root.resolve()

    # Refuse destructive overwrite targets that would delete the repo itself
    # or any ancestor directory containing it.
    if overwrite and resolved_root.is_relative_to(resolved_output):
        raise ValueError(
            f"Refusing to overwrite unsafe output directory: {resolved_output}"
        )

    # Always refuse output targets nested inside the repo. Even without
    # --overwrite, building into a descendant of the source tree can recurse
    # into copied content or leave benchmark artifacts mixed into the checkout.
    if resolved_output.is_relative_to(resolved_root) and resolved_output != resolved_root:
        raise ValueError(f"Refusing to build output directory inside repo: {resolved_output}")

    # Also reject obvious filesystem roots like '/'.
    if overwrite and resolved_output == resolved_output.parent:
        raise ValueError(
            f"Refusing to overwrite filesystem root: {resolved_output}"
        )

    return resolved_output


def build_workspace(
    *,
    output_dir: Path,
    task_ids: list[str],
    skill_profile: str = "full",
    overwrite: bool = False,
) -> Path:
    root = repo_root()
    output_dir = validate_output_dir(output_dir, root=root, overwrite=overwrite)
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

    apply_skill_profile(output_dir, skill_profile=skill_profile)

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
            source_backend = source_backends_root / backend_id
            if source_backend.exists():
                copy_tree(
                    source_backend,
                    public_backends_root / backend_id,
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
    parser.add_argument(
        "--skill-profile",
        choices=["full", "bo_only"],
        default="full",
        help=(
            "Skill surface to copy into the public workspace. "
            "'full' keeps the full research workflow; 'bo_only' strips the "
            "top-level research-layer skills while keeping BO-level skills."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = build_workspace(
        output_dir=args.output_dir,
        task_ids=list(args.tasks),
        skill_profile=str(args.skill_profile),
        overwrite=bool(args.overwrite),
    )
    print(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
