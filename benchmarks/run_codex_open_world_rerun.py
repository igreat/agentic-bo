"""Launch one Codex open-world rerun and auto-stage artifacts on completion."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from open_world_reruns import stage_run


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def find_single_child_dir(path: Path) -> str:
    dirs = sorted(p for p in path.iterdir() if p.is_dir())
    if len(dirs) != 1:
        raise RuntimeError(f"Expected exactly one run directory under {path}, found {len(dirs)}")
    return dirs[0].name


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def workspace_env_path(workspace: Path) -> Path:
    return workspace / ".venv"


def bootstrap_workspace_env(workspace: Path) -> None:
    """Create a clean workspace-local environment with only base rerun deps."""
    venv_path = workspace_env_path(workspace)
    if venv_path.exists():
        subprocess.run(["rm", "-rf", str(venv_path)], check=True)

    clean_env = os.environ.copy()
    clean_env.pop("VIRTUAL_ENV", None)
    clean_env.pop("PYTHONPATH", None)

    subprocess.run(["uv", "sync"], cwd=workspace, check=True, env=clean_env)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--no-deps",
            "hebo @ git+https://github.com/huawei-noah/HEBO.git#subdirectory=HEBO",
            "--python",
            str(venv_path / "bin" / "python"),
        ],
        cwd=workspace,
        check=True,
        env=clean_env,
    )


def build_clean_launch_env(workspace: Path) -> dict[str, str]:
    """Force Codex to prefer the workspace-local Python environment."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)

    workspace = workspace.resolve()
    workspace_venv_bin = str(workspace_env_path(workspace) / "bin")

    path_entries = []
    for entry in env.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        normalized = str(Path(entry).resolve()) if Path(entry).exists() else entry
        if normalized.endswith("/.venv/bin"):
            continue
        path_entries.append(entry)

    env["VIRTUAL_ENV"] = str(workspace_env_path(workspace))
    env["PATH"] = os.pathsep.join([workspace_venv_bin, *path_entries])
    return env


def build_prompt(prompt_path: Path, *, invoke_research_agent: bool) -> str:
    prompt = prompt_path.read_text(encoding="utf-8")
    if invoke_research_agent:
        return "/research-agent\n\n" + prompt
    return prompt


def launch_run(
    *,
    workspace: Path,
    task: str,
    repetition: str,
    baseline: str,
    prompt_file: str,
    invoke_research_agent: bool,
    model: str,
    model_runtime: str,
    effort_level: str,
    timeout_seconds: int,
) -> int:
    workspace = workspace.resolve()
    prompt_path = workspace / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found in workspace: {prompt_path}")

    bootstrap_workspace_env(workspace)
    launch_env = build_clean_launch_env(workspace)

    started_at = utc_now_iso()
    prompt_text = build_prompt(prompt_path, invoke_research_agent=invoke_research_agent)

    prompt_sent_path = workspace / "codex_prompt_sent.md"
    exec_log_path = workspace / "codex_exec.jsonl"
    last_message_path = workspace / "codex_last_message.txt"
    started_path = workspace / "codex_started_at.txt"
    finished_path = workspace / "codex_finished_at.txt"
    exit_code_path = workspace / "codex_exit_code.txt"

    write_text(prompt_sent_path, prompt_text)
    write_text(started_path, started_at + "\n")

    cmd = [
        "codex",
        "-c",
        'model_reasoning_effort="high"',
        "--search",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--json",
        "-m",
        model,
        "-C",
        str(workspace),
        "-o",
        str(last_message_path),
        "-",
    ]

    completion_status = "completed"
    stop_reason = ""
    return_code = None

    with exec_log_path.open("w", encoding="utf-8") as log_fp:
        try:
            proc = subprocess.run(
                cmd,
                cwd=workspace,
                env=launch_env,
                input=prompt_text,
                text=True,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
            )
            return_code = proc.returncode
            if proc.returncode != 0:
                completion_status = "failed"
                stop_reason = f"codex exec exited with code {proc.returncode}"
        except subprocess.TimeoutExpired as exc:
            completion_status = "incomplete"
            stop_reason = f"timeout after {timeout_seconds} seconds"
            return_code = 124
            if exc.stdout:
                log_fp.write(exc.stdout)
            if exc.stderr:
                log_fp.write(exc.stderr)

    finished_at = utc_now_iso()
    write_text(finished_path, finished_at + "\n")
    write_text(exit_code_path, f"{return_code}\n")

    bo_runs_root = workspace / "bo_runs"
    research_runs_root = workspace / "research_runs"

    try:
        bo_run_id = find_single_child_dir(bo_runs_root)
        research_id = find_single_child_dir(research_runs_root)
    except RuntimeError as exc:
        print(f"autostage skipped: {exc}", file=sys.stderr)
        return 0 if completion_status == "completed" else return_code or 1

    extra_paths = [
        "codex_exec.jsonl",
        "codex_last_message.txt",
        "codex_prompt_sent.md",
        "codex_started_at.txt",
        "codex_finished_at.txt",
        "codex_exit_code.txt",
    ]

    stage_run(
        task=task,
        repetition=repetition,
        baseline=baseline,
        workspace=workspace,
        bo_run_id=bo_run_id,
        research_id=research_id,
        prompt_file=prompt_file,
        model_runtime=model_runtime,
        effort_level=effort_level,
        completion_status=completion_status,
        stop_reason=stop_reason,
        overwrite=True,
        start_timestamp=started_at,
        end_timestamp=finished_at,
        extra_paths=extra_paths,
    )

    return 0 if completion_status == "completed" else return_code or 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch one Codex open-world rerun and auto-stage its artifacts."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task", choices=["her", "hea"], required=True)
    parser.add_argument(
        "--repetition",
        choices=["run_01", "run_02", "run_03", "rerun_a", "rerun_b"],
        required=True,
    )
    parser.add_argument(
        "--baseline", choices=["naive", "orchestrated"], required=True
    )
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--invoke-research-agent", action="store_true")
    parser.add_argument("--model", default="gpt-5-codex")
    parser.add_argument("--model-runtime", default="Codex")
    parser.add_argument("--effort-level", default="default")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return launch_run(
        workspace=args.workspace,
        task=args.task,
        repetition=args.repetition,
        baseline=args.baseline,
        prompt_file=args.prompt_file,
        invoke_research_agent=bool(args.invoke_research_agent),
        model=str(args.model),
        model_runtime=str(args.model_runtime),
        effort_level=str(args.effort_level),
        timeout_seconds=int(args.timeout_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
