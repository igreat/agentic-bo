"""Helpers for open-world rerun workspace setup and supplementary packaging."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT_COPY_FILES = (
    "AGENTS.md",
    "CLAUDE.md",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
)

ROOT_COPY_DIRS = (
    "bo_workflow",
    ".agents",
    ".claude",
    "data",
)

PROMPT_FILES_BY_TASK = {
    "her": (
        "benchmarks/prompts/her_live_structural_naive.md",
        "benchmarks/prompts/her_live_structural_strong.md",
    ),
    "hea": (
        "benchmarks/prompts/hea_live_structural_naive.md",
        "benchmarks/prompts/hea_live_structural_base_recovery.md",
        "benchmarks/prompts/hea_live_structural_strong.md",
    ),
}

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
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "results",
    "bo_runs",
    "research_runs",
    "evaluation_backends",
)

REPETITION_TO_RUN_DIR = {
    "rerun_a": "run_02",
    "rerun_b": "run_03",
}

RERUN_LAYOUT = (
    ("her", "rerun_a", "run_02", "naive"),
    ("her", "rerun_a", "run_02", "orchestrated"),
    ("her", "rerun_b", "run_03", "naive"),
    ("her", "rerun_b", "run_03", "orchestrated"),
    ("hea", "run_01", "run_01", "naive"),
    ("hea", "run_01", "run_01", "orchestrated"),
    ("hea", "rerun_a", "run_02", "naive"),
    ("hea", "rerun_a", "run_02", "orchestrated"),
    ("hea", "rerun_b", "run_03", "naive"),
    ("hea", "rerun_b", "run_03", "orchestrated"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def bundle_root() -> Path:
    return repo_root() / "results" / "open_world_reruns"


def template_root() -> Path:
    return repo_root() / "benchmarks" / "templates" / "open_world_reruns"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=COPYTREE_IGNORE)


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def resolve_commit(workspace: Path) -> str | None:
    return _run_git(["rev-parse", "HEAD"], workspace) or _run_git(
        ["rev-parse", "HEAD"], repo_root()
    )


def default_workspace_root() -> Path:
    return Path(tempfile.gettempdir()) / "agentic-bo-workspaces"


def _validate_workspace_output_root(
    output_root: Path, *, root: Path, overwrite: bool
) -> Path:
    resolved_output = output_root.resolve()
    resolved_root = root.resolve()

    if overwrite and resolved_root.is_relative_to(resolved_output):
        raise ValueError(
            f"Refusing to overwrite unsafe output directory: {resolved_output}"
        )

    if resolved_output.is_relative_to(resolved_root) and resolved_output != resolved_root:
        raise ValueError(
            f"Refusing to build workspace output directory inside repo: {resolved_output}"
        )

    if overwrite and resolved_output == resolved_output.parent:
        raise ValueError(f"Refusing to overwrite filesystem root: {resolved_output}")

    return resolved_output


def _resolve_descendant_path(base_dir: Path, rel_path: str, *, label: str) -> Path:
    resolved_base = base_dir.resolve()
    candidate = Path(rel_path)
    resolved_candidate = (
        candidate.resolve() if candidate.is_absolute() else (resolved_base / candidate).resolve()
    )
    if not resolved_candidate.is_relative_to(resolved_base):
        raise ValueError(
            f"{label} must stay inside {resolved_base}: {rel_path}"
        )
    return resolved_candidate


def setup_workspaces(output_root: Path, *, overwrite: bool) -> Path:
    root = repo_root()
    output_root = _validate_workspace_output_root(
        output_root, root=root, overwrite=overwrite
    )
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    for task, repetition, run_dir, baseline in RERUN_LAYOUT:
        _create_workspace(
            output_root=output_root,
            task=task,
            repetition=repetition,
            run_dir=run_dir,
            baseline=baseline,
        )

    return output_root


def _create_workspace(
    *,
    output_root: Path,
    task: str,
    repetition: str,
    run_dir: str,
    baseline: str,
    overwrite: bool = False,
) -> Path:
    root = repo_root()
    output_root = _validate_workspace_output_root(
        output_root, root=root, overwrite=overwrite
    )
    task_root = output_root / task
    run_root = _resolve_descendant_path(
        task_root, run_dir, label="Workspace run_dir"
    )
    workspace = _resolve_descendant_path(
        run_root, baseline, label="Workspace baseline"
    )
    if workspace.exists():
        if not overwrite:
            raise FileExistsError(
                f"Workspace already exists: {workspace}. Pass --overwrite to replace it."
            )
        shutil.rmtree(workspace)

    workspace.mkdir(parents=True, exist_ok=True)

    for rel_path in ROOT_COPY_FILES:
        copy_file(root / rel_path, workspace / rel_path)

    for rel_path in ROOT_COPY_DIRS:
        copy_tree(root / rel_path, workspace / rel_path)

    _copy_prompt_files(task=task, workspace=workspace)
    _apply_skill_profile(workspace=workspace, baseline=baseline)

    (workspace / "bo_runs").mkdir(parents=True, exist_ok=True)
    (workspace / "research_runs").mkdir(parents=True, exist_ok=True)

    write_json(
        workspace / "rerun_workspace.json",
        {
            "task": task,
            "repetition": repetition,
            "run_dir": run_dir,
            "baseline": baseline,
            "source_branch": _run_git(["branch", "--show-current"], root),
            "source_commit": resolve_commit(root),
            "workspace_path": str(workspace),
        },
    )
    return workspace


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        copy_file(src, dst)


def _copy_prompt_files(*, task: str, workspace: Path) -> None:
    root = repo_root()
    for rel_path in PROMPT_FILES_BY_TASK[task]:
        copy_file(root / rel_path, workspace / rel_path)


def _apply_skill_profile(*, workspace: Path, baseline: str) -> None:
    if baseline != "naive":
        return

    for rel_path in RESEARCH_LAYER_SKILL_DIRS:
        target = workspace / rel_path
        if target.exists():
            shutil.rmtree(target)

    bo_only_templates = template_root() / "bo_only"
    write_text(
        workspace / "AGENTS.md",
        (bo_only_templates / "AGENTS.md").read_text(encoding="utf-8"),
    )
    write_text(
        workspace / "CLAUDE.md",
        (bo_only_templates / "CLAUDE.md").read_text(encoding="utf-8"),
    )


def _load_jsonish(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{"):
        return json.loads(stripped)
    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.S)
    if match:
        return json.loads(match.group(1))
    return None


def _workspace_metadata(workspace: Path) -> dict:
    metadata_path = workspace / "rerun_workspace.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _summarize_judging_dir(path: Path, *, task: str, repetition: str) -> None:
    judge_files: list[dict] = []
    for candidate in sorted(path.glob("pairwise_judge_*.json")):
        if candidate.name.endswith("output_schema.json"):
            continue
        parsed = _load_jsonish(candidate)
        if not parsed:
            continue
        comparison = parsed.get("pairwise_comparison", {})
        judge_files.append(
            {
                "file": candidate.name,
                "judge_model": parsed.get("judge_model"),
                "winner": comparison.get("winner"),
                "why": comparison.get("why"),
            }
        )

    winner_counts: dict[str, int] = {}
    for item in judge_files:
        winner = item.get("winner")
        if winner:
            winner_counts[winner] = winner_counts.get(winner, 0) + 1

    overall_winner = None
    overall_preference = "inconclusive"
    if winner_counts:
        candidate_winner = max(winner_counts, key=winner_counts.get)
        max_votes = winner_counts[candidate_winner]
        if max_votes == len(judge_files):
            overall_winner = candidate_winner
            overall_preference = "unanimous"
        elif max_votes > len(judge_files) / 2:
            overall_winner = candidate_winner
            overall_preference = "majority"

    payload = {
        "task": task,
        "repetition_id": repetition,
        "judge_files": judge_files,
        "winner_counts": winner_counts,
        "overall_winner": overall_winner,
        "overall_preference": overall_preference,
    }
    write_json(path / "judge_pair_summary.json", payload)


def stage_run(
    *,
    task: str,
    repetition: str,
    baseline: str,
    workspace: Path,
    bo_run_id: str,
    research_id: str,
    prompt_file: str,
    model_runtime: str,
    effort_level: str,
    completion_status: str,
    stop_reason: str,
    overwrite: bool,
    start_timestamp: str | None,
    end_timestamp: str | None,
    extra_paths: list[str],
) -> Path:
    workspace = workspace.resolve()
    dest = bundle_root() / task / repetition / baseline
    if dest.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dest}. Pass overwrite=True to replace it."
            )
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=False)

    bo_dir = workspace / "bo_runs" / bo_run_id
    research_dir = workspace / "research_runs" / research_id
    if not bo_dir.exists():
        raise FileNotFoundError(f"BO run not found: {bo_dir}")
    if not research_dir.exists():
        raise FileNotFoundError(f"Research run not found: {research_dir}")

    copy_tree(bo_dir, dest / "bo_run")

    for item in sorted(research_dir.iterdir()):
        target = dest / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            copy_file(item, target)

    extras_root = dest / "extras"
    for rel_path in extra_paths:
        src = _resolve_descendant_path(
            workspace, rel_path, label="Extra path"
        )
        relative_src = src.relative_to(workspace)
        if src.is_dir():
            copy_tree(src, extras_root / relative_src)
        elif src.exists():
            copy_file(src, extras_root / relative_src)
        else:
            raise FileNotFoundError(f"Extra path not found: {src}")

    state_path = bo_dir / "state.json"
    state_payload = {}
    if state_path.exists():
        state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    workspace_metadata = _workspace_metadata(workspace)

    metadata = {
        "task": task,
        "repetition_id": repetition,
        "baseline": baseline,
        "prompt_file": prompt_file,
        "commit_hash": workspace_metadata.get("source_commit") or resolve_commit(workspace),
        "model_runtime": model_runtime,
        "effort_level": effort_level,
        "workspace_path": str(workspace),
        "bo_run_id": bo_run_id,
        "research_id": research_id,
        "start_timestamp": start_timestamp or state_payload.get("created_at"),
        "end_timestamp": end_timestamp or state_payload.get("updated_at"),
        "completion_status": completion_status,
        "stop_reason": stop_reason,
    }
    write_json(dest / "run_metadata.json", metadata)
    return dest


def stage_judging(
    *,
    task: str,
    repetition: str,
    source_dir: Path,
    include_files: list[str],
    overwrite: bool,
) -> Path:
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Judge artifact directory not found: {source_dir}")
    dest = bundle_root() / "judging" / task / repetition
    if dest.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dest}. Pass overwrite=True to replace it."
            )
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=False)

    if include_files:
        candidates = [
            _resolve_descendant_path(
                source_dir, rel_path, label="Judge include file"
            )
            for rel_path in include_files
        ]
    else:
        candidates = []
        for path in sorted(source_dir.iterdir()):
            if not path.is_file():
                continue
            if path.name.endswith(".pid"):
                continue
            if path.name.endswith("_started_at.txt"):
                continue
            if path.name.endswith("_model_used.txt"):
                continue
            if re.search(r" \d+\.[A-Za-z0-9]+$", path.name):
                continue
            candidates.append(path)

    for src in candidates:
        if not src.exists():
            raise FileNotFoundError(f"Judge artifact not found: {src}")
        copy_file(src, dest / src.name)

    _summarize_judging_dir(dest, task=task, repetition=repetition)
    return dest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up and stage open-world rerun workspaces and artifacts."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    setup_cmd = sub.add_parser(
        "setup-workspaces",
        help="Create clean external HER/HEA rerun workspaces from the current repo.",
    )
    setup_cmd.add_argument(
        "--output-root",
        type=Path,
        default=default_workspace_root(),
    )
    setup_cmd.add_argument("--overwrite", action="store_true")

    setup_single_cmd = sub.add_parser(
        "setup-single-workspace",
        help="Create one clean external workspace without replacing the whole tree.",
    )
    setup_single_cmd.add_argument("--output-root", type=Path, default=default_workspace_root())
    setup_single_cmd.add_argument("--task", choices=["her", "hea"], required=True)
    setup_single_cmd.add_argument("--repetition", required=True)
    setup_single_cmd.add_argument("--run-dir", required=True)
    setup_single_cmd.add_argument(
        "--baseline", choices=["naive", "orchestrated"], required=True
    )
    setup_single_cmd.add_argument("--overwrite", action="store_true")

    stage_run_cmd = sub.add_parser(
        "stage-run",
        help="Copy one curated run into results/open_world_reruns/.",
    )
    stage_run_cmd.add_argument("--task", choices=["her", "hea"], required=True)
    stage_run_cmd.add_argument(
        "--repetition",
        choices=["original", "run_01", "run_02", "run_03", "rerun_a", "rerun_b"],
        required=True,
    )
    stage_run_cmd.add_argument(
        "--baseline", choices=["naive", "orchestrated"], required=True
    )
    stage_run_cmd.add_argument("--workspace", type=Path, required=True)
    stage_run_cmd.add_argument("--bo-run-id", type=str, required=True)
    stage_run_cmd.add_argument("--research-id", type=str, required=True)
    stage_run_cmd.add_argument("--prompt-file", type=str, required=True)
    stage_run_cmd.add_argument("--model-runtime", type=str, required=True)
    stage_run_cmd.add_argument("--effort-level", type=str, required=True)
    stage_run_cmd.add_argument(
        "--completion-status",
        choices=["completed", "incomplete", "failed"],
        required=True,
    )
    stage_run_cmd.add_argument("--stop-reason", type=str, default="")
    stage_run_cmd.add_argument("--start-timestamp", type=str, default=None)
    stage_run_cmd.add_argument("--end-timestamp", type=str, default=None)
    stage_run_cmd.add_argument(
        "--extra-path",
        action="append",
        default=[],
        help="Relative path inside the source workspace to copy under extras/.",
    )
    stage_run_cmd.add_argument("--overwrite", action="store_true")

    stage_judging_cmd = sub.add_parser(
        "stage-judging",
        help="Copy judge JSON/output files into the supplementary bundle.",
    )
    stage_judging_cmd.add_argument("--task", choices=["her", "hea"], required=True)
    stage_judging_cmd.add_argument(
        "--repetition",
        choices=["original", "run_01", "run_02", "run_03", "rerun_a", "rerun_b"],
        required=True,
    )
    stage_judging_cmd.add_argument("--source-dir", type=Path, required=True)
    stage_judging_cmd.add_argument(
        "--include-file",
        action="append",
        default=[],
        help="Relative file inside the source judge directory to copy explicitly.",
    )
    stage_judging_cmd.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "setup-workspaces":
        workspace_root = setup_workspaces(
            output_root=args.output_root,
            overwrite=bool(args.overwrite),
        )
        print(workspace_root)
        return 0
    if args.command == "setup-single-workspace":
        workspace = _create_workspace(
            output_root=args.output_root.resolve(),
            task=args.task,
            repetition=args.repetition,
            run_dir=args.run_dir,
            baseline=args.baseline,
            overwrite=bool(args.overwrite),
        )
        print(workspace)
        return 0
    if args.command == "stage-run":
        dest = stage_run(
            task=args.task,
            repetition=args.repetition,
            baseline=args.baseline,
            workspace=args.workspace,
            bo_run_id=args.bo_run_id,
            research_id=args.research_id,
            prompt_file=args.prompt_file,
            model_runtime=args.model_runtime,
            effort_level=args.effort_level,
            completion_status=args.completion_status,
            stop_reason=args.stop_reason,
            overwrite=bool(args.overwrite),
            start_timestamp=args.start_timestamp,
            end_timestamp=args.end_timestamp,
            extra_paths=list(args.extra_path),
        )
        print(dest)
        return 0
    if args.command == "stage-judging":
        dest = stage_judging(
            task=args.task,
            repetition=args.repetition,
            source_dir=args.source_dir,
            include_files=list(args.include_file),
            overwrite=bool(args.overwrite),
        )
        print(dest)
        return 0
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
