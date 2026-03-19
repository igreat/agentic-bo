"""Helpers for scaffolding and validating open-world research-agent runs."""

from __future__ import annotations

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


def scaffold_open_world_research_dir(research_dir: Path) -> dict[str, Path]:
    """Create the standard directory/file scaffold for an open-world run."""
    research_dir = research_dir.resolve()
    research_dir.mkdir(parents=True, exist_ok=True)
    verification_dir = research_dir / "verification_artifacts"
    verification_dir.mkdir(parents=True, exist_ok=True)
    log_path = research_dir / "operationalization_log.jsonl"
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    return {
        "research_dir": research_dir,
        "verification_dir": verification_dir,
        "operationalization_log_path": log_path,
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
    errors.extend(validate_operationalization_log(log_path))
    return errors
