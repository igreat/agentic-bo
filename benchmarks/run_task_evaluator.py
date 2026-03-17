#!/usr/bin/env python3
"""Resolve an opaque benchmark handle and run hidden evaluation."""

import argparse
import json
import os
from pathlib import Path

from bo_workflow.engine import BOEngine
from bo_workflow.evaluation.cli import run_hidden_oracle_evaluator


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_optional_path(
    *,
    cli_value: Path | None,
    env_name: str,
) -> Path | None:
    if cli_value is not None:
        return cli_value
    raw = os.environ.get(env_name)
    if not raw:
        return None
    return Path(raw)


def resolve_required_path(
    *,
    cli_value: Path | None,
    env_name: str,
    flag_name: str,
    purpose: str,
) -> Path:
    resolved = resolve_optional_path(cli_value=cli_value, env_name=env_name)
    if resolved is None:
        raise FileNotFoundError(
            f"Missing {purpose}. Provide {flag_name} or set {env_name}."
        )
    return resolved


def resolve_backend_dir(
    *,
    manifest: dict,
    handle_map: dict,
    backends_root: Path | None,
) -> Path:
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("mode") != "external_hidden":
        raise ValueError("Task manifest is not configured for external hidden evaluation.")
    if evaluation.get("runner") != "benchmark-evaluator":
        raise ValueError("Task manifest does not use the benchmark evaluator runner.")

    handle = evaluation.get("handle")
    if not handle:
        raise KeyError("Task manifest is missing evaluation.handle.")
    if handle not in handle_map:
        raise KeyError(f"Evaluator handle not found in private handle map: {handle}")

    entry = handle_map[handle]
    if "backend_dir" in entry:
        return Path(entry["backend_dir"])
    if "backend_id" in entry:
        if backends_root is None:
            raise FileNotFoundError(
                "Handle map entry uses backend_id but no backends root was provided. "
                "Set BENCHMARK_BACKENDS_ROOT or pass --backends-root."
            )
        return backends_root / str(entry["backend_id"])
    raise KeyError(
        f"Handle map entry for {handle} must include backend_dir or backend_id."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run hidden benchmark evaluation from an opaque task handle."
    )
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--handle-map", type=Path, default=None)
    parser.add_argument("--backends-root", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_json(args.task_manifest)

    runs_root = resolve_optional_path(
        cli_value=args.runs_root,
        env_name="BENCHMARK_RUNS_ROOT",
    )
    if runs_root is None:
        runs_root = Path("bo_runs")

    handle_map_path = resolve_required_path(
        cli_value=args.handle_map,
        env_name="BENCHMARK_HANDLE_MAP",
        flag_name="--handle-map",
        purpose="benchmark handle map",
    )
    backends_root = resolve_optional_path(
        cli_value=args.backends_root,
        env_name="BENCHMARK_BACKENDS_ROOT",
    )
    handle_map = load_json(handle_map_path)

    iterations = args.iterations
    if iterations is None:
        iterations = int(manifest.get("budget", {}).get("iterations", 0))
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = int(manifest.get("budget", {}).get("batch_size", 1))

    backend_dir = resolve_backend_dir(
        manifest=manifest,
        handle_map=handle_map,
        backends_root=backends_root,
    )

    payload = run_hidden_oracle_evaluator(
        BOEngine(runs_root=runs_root),
        run_id=args.run_id,
        backend_dir=backend_dir,
        num_iterations=iterations,
        batch_size=batch_size,
        verbose=args.verbose,
    )
    public_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"backend_id", "backend_dir"}
    }
    public_payload["evaluation_handle"] = str(manifest["evaluation"]["handle"])
    print(json.dumps(public_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
