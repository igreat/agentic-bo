"""Helpers for scaffolding and validating open-world research-agent runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


REQUIRED_OPEN_WORLD_FILES = (
    "research_state.json",
    "research_plan.md",
    "paper.md",
    "initial_prompt.md",
    "discovered_search_space.json",
    "evaluator.py",
    "operationalization_log.jsonl",
)

REQUIRED_OPEN_WORLD_DIRS = ("verification_artifacts",)

REQUIRED_OPEN_WORLD_STATE_FIELDS = (
    "nudge_tier",
    "prompt_path",
    "source_urls",
    "discovered_search_space_path",
    "evaluator_module_path",
    "helper_script_paths",
    "verification_artifacts",
    "dependency_installs",
    "approach_revisions",
    "final_setup_frozen",
)

OPEN_WORLD_EVENT_TYPES = {
    "source_selected",
    "search_space_resolved",
    "evaluator_written",
    "helper_script_written",
    "dependency_installed",
    "verification_generated",
    "approach_revised",
    "setup_frozen",
    "bo_started",
    "bo_completed",
}

REQUIRED_OPERATIONALIZATION_EVENT_FIELDS = (
    "timestamp",
    "event_type",
    "summary",
    "artifact_paths",
)

REQUIRED_OPERATOR_SPEC_FIELDS = (
    "task_id",
    "title",
    "agent_prompt_family",
    "canonical_solution",
    "acceptable_alternatives",
    "evaluation_window",
    "success_checks",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def _resolve_artifact_path(research_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    search_roots = (
        Path.cwd(),
        research_dir,
        research_dir.parent,
        research_dir.parent.parent,
    )
    for root in search_roots:
        resolved = root / candidate
        if resolved.exists():
            return resolved

    return search_roots[0] / candidate


def _default_research_state(research_dir: Path) -> dict[str, Any]:
    research_id = research_dir.name
    rel_prefix = Path("research_runs") / research_id
    return {
        "research_id": research_id,
        "research_question": None,
        "system": None,
        "objective_property": None,
        "objective_direction": None,
        "dataset_path": None,
        "prior_observations_path": None,
        "bo_run_id": None,
        "literature_findings": {
            "baselines": [],
            "key_variables": [],
            "known_constraints": [],
            "source_urls": [],
            "summary": "",
        },
        "open_world": {
            "nudge_tier": None,
            "prompt_path": str(rel_prefix / "initial_prompt.md"),
            "source_urls": [],
            "discovered_search_space_path": str(
                rel_prefix / "discovered_search_space.json"
            ),
            "evaluator_module_path": str(rel_prefix / "evaluator.py"),
            "helper_script_paths": [],
            "verification_artifacts": [],
            "dependency_installs": [],
            "approach_revisions": [],
            "final_setup_frozen": False,
        },
        "experiment_spec": {
            "target_column": None,
            "design_parameters": [],
            "fixed_features": {},
            "constraints": [],
            "seed_observations_count": 0,
        },
        "bo_results": {
            "best_value": None,
            "best_x": None,
            "best_iteration": None,
            "num_observations": None,
            "oracle_model": None,
            "oracle_rmse": None,
            "report_path": None,
            "convergence_plot_path": None,
        },
        "paper_path": str(rel_prefix / "paper.md"),
        "phases": {
            "problem_framing": "pending",
            "literature_search": "pending",
            "experiment_setup": "pending",
            "bo_execution": "pending",
            "interpretation": "pending",
            "paper_writing": "pending",
        },
    }


def _ensure_file(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def _research_plan_template() -> str:
    return (
        "# Research Plan\n\n"
        "## Research Question\n\n"
        "TODO\n\n"
        "## Problem Framing\n\n"
        "TODO\n\n"
        "## Literature Context\n\n"
        "TODO\n\n"
        "## Experiment Design\n\n"
        "TODO\n\n"
        "## BO Results\n\n"
        "TODO\n\n"
        "## Interpretation\n\n"
        "TODO\n\n"
        "## Paper Draft Link\n\n"
        "TODO\n"
    )


def _paper_template() -> str:
    return "# Paper Draft\n\nTODO\n"


def _prompt_template() -> str:
    return (
        "# Initial Prompt\n\n"
        "**Nudge Tier:** TODO\n\n"
        "## Prompt\n\n"
        "TODO: record the exact prompt shown to the agent.\n"
    )


def _search_space_template() -> str:
    payload = {
        "design_parameters": [],
        "fixed_features": {},
        "constraints": [],
    }
    return json.dumps(payload, indent=2) + "\n"


def _evaluator_template() -> str:
    return (
        '"""Run-local open-world evaluator stub."""\n\n'
        "def evaluate(x):\n"
        '    raise NotImplementedError("Replace this stub with the discovered evaluator.")\n'
    )


def scaffold_open_world_research_dir(research_dir: Path) -> dict[str, Path]:
    """Create the standard directory/file scaffold for an open-world run."""
    research_dir = research_dir.resolve()
    research_dir.mkdir(parents=True, exist_ok=True)
    verification_dir = research_dir / "verification_artifacts"
    verification_dir.mkdir(parents=True, exist_ok=True)
    log_path = research_dir / "operationalization_log.jsonl"
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")

    _ensure_file(
        research_dir / "research_state.json",
        json.dumps(_default_research_state(research_dir), indent=2) + "\n",
    )
    _ensure_file(research_dir / "research_plan.md", _research_plan_template())
    _ensure_file(research_dir / "paper.md", _paper_template())
    _ensure_file(research_dir / "initial_prompt.md", _prompt_template())
    _ensure_file(
        research_dir / "discovered_search_space.json",
        _search_space_template(),
    )
    _ensure_file(research_dir / "evaluator.py", _evaluator_template())

    return {
        "research_dir": research_dir,
        "verification_dir": verification_dir,
        "operationalization_log_path": log_path,
        "research_state_path": research_dir / "research_state.json",
        "research_plan_path": research_dir / "research_plan.md",
        "paper_path": research_dir / "paper.md",
        "initial_prompt_path": research_dir / "initial_prompt.md",
        "search_space_path": research_dir / "discovered_search_space.json",
        "evaluator_path": research_dir / "evaluator.py",
    }


def write_initial_prompt(
    research_dir: Path,
    *,
    prompt_text: str,
    nudge_tier: str,
    nudge_text: str | None = None,
) -> Path:
    """Write the exact prompt/nudge text shown to the agent."""
    paths = scaffold_open_world_research_dir(research_dir)
    prompt_path = paths["initial_prompt_path"]
    blocks = [
        f"# Initial Prompt",
        "",
        f"**Nudge Tier:** {nudge_tier}",
        "",
        "## Prompt",
        "",
        prompt_text.rstrip(),
    ]
    if nudge_text:
        blocks.extend(["", "## Nudge", "", nudge_text.rstrip()])
    prompt_path.write_text("\n".join(blocks) + "\n", encoding="utf-8")
    return prompt_path


def build_operationalization_event(
    *,
    event_type: str,
    summary: str,
    artifact_paths: list[str],
    source_urls: list[str] | None = None,
    reason: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Construct and validate one operationalization event object."""
    event = {
        "timestamp": timestamp or utc_now_iso(),
        "event_type": event_type,
        "summary": summary,
        "artifact_paths": artifact_paths,
    }
    if source_urls is not None:
        event["source_urls"] = source_urls
    if reason is not None:
        event["reason"] = reason

    errors = validate_operationalization_events([event])
    if errors:
        raise ValueError("; ".join(errors))
    return event


def append_operationalization_event(
    research_dir: Path,
    *,
    event_type: str,
    summary: str,
    artifact_paths: list[str],
    source_urls: list[str] | None = None,
    reason: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Append one validated event to operationalization_log.jsonl."""
    paths = scaffold_open_world_research_dir(research_dir)
    event = build_operationalization_event(
        event_type=event_type,
        summary=summary,
        artifact_paths=artifact_paths,
        source_urls=source_urls,
        reason=reason,
        timestamp=timestamp,
    )
    with paths["operationalization_log_path"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
    return event


def validate_operationalization_events(
    events: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    for index, event in enumerate(events, start=1):
        for field in REQUIRED_OPERATIONALIZATION_EVENT_FIELDS:
            if field not in event:
                errors.append(f"event {index}: missing required field '{field}'")

        event_type = event.get("event_type")
        if event_type not in OPEN_WORLD_EVENT_TYPES:
            errors.append(
                f"event {index}: unsupported event_type {event_type!r}"
            )

        artifact_paths = event.get("artifact_paths")
        if not isinstance(artifact_paths, list):
            errors.append(f"event {index}: artifact_paths must be a list")

        source_urls = event.get("source_urls")
        if source_urls is not None and not isinstance(source_urls, list):
            errors.append(f"event {index}: source_urls must be a list when present")

    return errors


def validate_operationalization_log(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing operationalization log: {path}"]
    return validate_operationalization_events(_load_jsonl(path))


def validate_open_world_operator_spec(
    spec: dict[str, Any],
    *,
    root_dir: Path | None = None,
) -> list[str]:
    """Validate the hidden operator-side answer-key spec."""
    errors: list[str] = []

    for field in REQUIRED_OPERATOR_SPEC_FIELDS:
        if field not in spec:
            errors.append(f"operator spec: missing required field '{field}'")

    prompt_family = spec.get("agent_prompt_family", {})
    if not isinstance(prompt_family, dict):
        errors.append("operator spec: agent_prompt_family must be an object")
    else:
        if "primary_prompt_path" not in prompt_family:
            errors.append("operator spec: missing agent_prompt_family.primary_prompt_path")
        if "nudge_tiers" not in prompt_family:
            errors.append("operator spec: missing agent_prompt_family.nudge_tiers")

        if root_dir and prompt_family.get("primary_prompt_path"):
            prompt_path = root_dir / str(prompt_family["primary_prompt_path"])
            if not prompt_path.exists():
                errors.append(
                    "operator spec: primary_prompt_path does not exist: "
                    f"{prompt_family['primary_prompt_path']}"
                )

    canonical_solution = spec.get("canonical_solution", {})
    if not isinstance(canonical_solution, dict):
        errors.append("operator spec: canonical_solution must be an object")
    else:
        for field in (
            "evaluator_family",
            "design_parameter_family",
            "constraints",
            "verification_artifact",
        ):
            if field not in canonical_solution:
                errors.append(f"operator spec: missing canonical_solution.{field}")

    alternatives = spec.get("acceptable_alternatives")
    if not isinstance(alternatives, list):
        errors.append("operator spec: acceptable_alternatives must be a list")

    evaluation_window = spec.get("evaluation_window", {})
    if not isinstance(evaluation_window, dict):
        errors.append("operator spec: evaluation_window must be an object")
    elif "time_budget_minutes" not in evaluation_window:
        errors.append("operator spec: missing evaluation_window.time_budget_minutes")

    success_checks = spec.get("success_checks")
    if not isinstance(success_checks, list) or not success_checks:
        errors.append("operator spec: success_checks must be a non-empty list")

    return errors


def validate_open_world_operator_spec_file(path: Path) -> list[str]:
    resolved = path.resolve()
    root_dir = resolved.parents[3] if len(resolved.parents) >= 4 else None
    return validate_open_world_operator_spec(_load_json(path), root_dir=root_dir)


def validate_open_world_research_run(
    research_dir: Path,
    *,
    require_frozen: bool = True,
) -> list[str]:
    errors: list[str] = []
    research_dir = research_dir.resolve()

    for rel_path in REQUIRED_OPEN_WORLD_FILES:
        if not (research_dir / rel_path).exists():
            errors.append(f"missing required open-world file: {rel_path}")

    for rel_dir in REQUIRED_OPEN_WORLD_DIRS:
        if not (research_dir / rel_dir).is_dir():
            errors.append(f"missing required open-world directory: {rel_dir}")

    state_path = research_dir / "research_state.json"
    if not state_path.exists():
        return errors

    state = _load_json(state_path)
    open_world = state.get("open_world")
    if not isinstance(open_world, dict):
        errors.append("research_state.json: missing open_world object")
        return errors

    for field in REQUIRED_OPEN_WORLD_STATE_FIELDS:
        if field not in open_world:
            errors.append(f"research_state.json: missing open_world.{field}")

    if not isinstance(open_world.get("source_urls", []), list):
        errors.append("research_state.json: open_world.source_urls must be a list")
    if not isinstance(open_world.get("helper_script_paths", []), list):
        errors.append(
            "research_state.json: open_world.helper_script_paths must be a list"
        )
    if not isinstance(open_world.get("verification_artifacts", []), list):
        errors.append(
            "research_state.json: open_world.verification_artifacts must be a list"
        )
    if not isinstance(open_world.get("dependency_installs", []), list):
        errors.append(
            "research_state.json: open_world.dependency_installs must be a list"
        )
    if not isinstance(open_world.get("approach_revisions", []), list):
        errors.append(
            "research_state.json: open_world.approach_revisions must be a list"
        )
    if not isinstance(open_world.get("final_setup_frozen"), bool):
        errors.append(
            "research_state.json: open_world.final_setup_frozen must be a boolean"
        )

    if require_frozen and open_world.get("final_setup_frozen") is not True:
        errors.append("research_state.json: open_world.final_setup_frozen must be true")

    prompt_path = _resolve_artifact_path(research_dir, open_world.get("prompt_path"))
    if prompt_path is None or not prompt_path.exists():
        errors.append("research_state.json: open_world.prompt_path does not exist")

    search_space_path = _resolve_artifact_path(
        research_dir,
        open_world.get("discovered_search_space_path"),
    )
    if search_space_path is None or not search_space_path.exists():
        errors.append(
            "research_state.json: open_world.discovered_search_space_path does not exist"
        )

    evaluator_path = _resolve_artifact_path(
        research_dir,
        open_world.get("evaluator_module_path"),
    )
    if evaluator_path is None or not evaluator_path.exists():
        errors.append(
            "research_state.json: open_world.evaluator_module_path does not exist"
        )

    verification_artifacts = open_world.get("verification_artifacts", [])
    if not verification_artifacts:
        errors.append(
            "research_state.json: open_world.verification_artifacts must not be empty"
        )
    else:
        for artifact in verification_artifacts:
            resolved = _resolve_artifact_path(research_dir, artifact)
            if resolved is None or not resolved.exists():
                errors.append(
                    "research_state.json: verification artifact does not exist: "
                    f"{artifact}"
                )

    for helper_path in open_world.get("helper_script_paths", []):
        resolved = _resolve_artifact_path(research_dir, helper_path)
        if resolved is None or not resolved.exists():
            errors.append(
                f"research_state.json: helper_script_path does not exist: {helper_path}"
            )

    for index, install in enumerate(open_world.get("dependency_installs", []), start=1):
        if not isinstance(install, dict):
            errors.append(
                f"research_state.json: dependency_installs[{index}] must be an object"
            )
            continue
        for field in ("packages", "command", "reason"):
            if field not in install:
                errors.append(
                    f"research_state.json: dependency_installs[{index}] missing '{field}'"
                )

    for index, revision in enumerate(open_world.get("approach_revisions", []), start=1):
        if not isinstance(revision, dict):
            errors.append(
                f"research_state.json: approach_revisions[{index}] must be an object"
            )
            continue
        for field in ("timestamp", "reason", "changed"):
            if field not in revision:
                errors.append(
                    f"research_state.json: approach_revisions[{index}] missing '{field}'"
                )

    log_path = research_dir / "operationalization_log.jsonl"
    log_errors = validate_operationalization_log(log_path)
    errors.extend(log_errors)
    if not log_errors:
        events = _load_jsonl(log_path)
        setup_frozen_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event_type") == "setup_frozen"
        ]
        bo_started_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event_type") == "bo_started"
        ]
        bo_completed_indices = [
            index
            for index, event in enumerate(events)
            if event.get("event_type") == "bo_completed"
        ]

        if open_world.get("final_setup_frozen") is True and not setup_frozen_indices:
            errors.append(
                "operationalization_log.jsonl: missing setup_frozen event for a frozen final setup"
            )
        if bo_started_indices and not any(
            setup_index < bo_started_indices[0]
            for setup_index in setup_frozen_indices
        ):
            errors.append(
                "operationalization_log.jsonl: setup_frozen must occur before bo_started"
            )
        if bo_completed_indices and not any(
            start_index < bo_completed_indices[0]
            for start_index in bo_started_indices
        ):
            errors.append(
                "operationalization_log.jsonl: bo_started must occur before bo_completed"
            )
    return errors


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _read_optional_text(text: str | None, path: Path | None) -> str | None:
    if text is not None:
        return text
    if path is not None:
        return path.read_text(encoding="utf-8")
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bo_workflow.open_world",
        description="Scaffold and validate open-world research-agent artifacts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scaffold_cmd = sub.add_parser(
        "scaffold",
        help="Create the standard open-world research-run scaffold.",
    )
    scaffold_cmd.add_argument("--research-dir", type=Path, required=True)

    write_prompt_cmd = sub.add_parser(
        "write-prompt",
        help="Write the exact initial prompt shown to the agent.",
    )
    write_prompt_cmd.add_argument("--research-dir", type=Path, required=True)
    write_prompt_cmd.add_argument("--nudge-tier", required=True)
    prompt_group = write_prompt_cmd.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt-text")
    prompt_group.add_argument("--prompt-file", type=Path)
    nudge_group = write_prompt_cmd.add_mutually_exclusive_group()
    nudge_group.add_argument("--nudge-text")
    nudge_group.add_argument("--nudge-file", type=Path)

    log_event_cmd = sub.add_parser(
        "log-event",
        help="Append one operationalization event to the run log.",
    )
    log_event_cmd.add_argument("--research-dir", type=Path, required=True)
    log_event_cmd.add_argument("--event-type", required=True)
    log_event_cmd.add_argument("--summary", required=True)
    log_event_cmd.add_argument(
        "--artifact-path",
        action="append",
        default=[],
        help="Artifact path to associate with the event. May be repeated.",
    )
    log_event_cmd.add_argument(
        "--source-url",
        action="append",
        default=[],
        help="Source URL used for this event. May be repeated.",
    )
    log_event_cmd.add_argument("--reason")
    log_event_cmd.add_argument("--timestamp")

    validate_run_cmd = sub.add_parser(
        "validate-run",
        help="Validate an open-world research run evidence package.",
    )
    validate_run_cmd.add_argument("--research-dir", type=Path, required=True)
    validate_run_cmd.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="Do not require open_world.final_setup_frozen to be true.",
    )

    validate_spec_cmd = sub.add_parser(
        "validate-spec",
        help="Validate a hidden operator-side open-world spec file.",
    )
    validate_spec_cmd.add_argument("--path", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "scaffold":
        payload = {
            "created": {
                key: str(value)
                for key, value in scaffold_open_world_research_dir(args.research_dir).items()
            }
        }
        _json_print(payload)
        return 0

    if args.command == "write-prompt":
        prompt_text = _read_optional_text(args.prompt_text, args.prompt_file)
        nudge_text = _read_optional_text(args.nudge_text, args.nudge_file)
        prompt_path = write_initial_prompt(
            args.research_dir,
            prompt_text=prompt_text or "",
            nudge_tier=args.nudge_tier,
            nudge_text=nudge_text,
        )
        _json_print({"prompt_path": str(prompt_path)})
        return 0

    if args.command == "log-event":
        event = append_operationalization_event(
            args.research_dir,
            event_type=args.event_type,
            summary=args.summary,
            artifact_paths=list(args.artifact_path),
            source_urls=list(args.source_url) or None,
            reason=args.reason,
            timestamp=args.timestamp,
        )
        _json_print(event)
        return 0

    if args.command == "validate-run":
        errors = validate_open_world_research_run(
            args.research_dir,
            require_frozen=not bool(args.allow_unfrozen),
        )
        payload = {
            "research_dir": str(args.research_dir),
            "valid": len(errors) == 0,
            "errors": errors,
        }
        _json_print(payload)
        return 0 if not errors else 1

    if args.command == "validate-spec":
        errors = validate_open_world_operator_spec_file(args.path)
        payload = {
            "path": str(args.path),
            "valid": len(errors) == 0,
            "errors": errors,
        }
        _json_print(payload)
        return 0 if not errors else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
