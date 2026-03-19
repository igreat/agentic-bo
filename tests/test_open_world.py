import json
from pathlib import Path

from bo_workflow.open_world import validate_open_world_research_run
from bo_workflow.open_world import validate_operationalization_events


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_operationalization_events_accepts_expected_shape() -> None:
    events = [
        {
            "timestamp": "2026-03-19T20:00:00Z",
            "event_type": "source_selected",
            "summary": "Selected the tutorial HER example as the evaluator family.",
            "artifact_paths": ["research_runs/her_demo/initial_prompt.md"],
            "source_urls": [
                "https://github.com/zwyu-ai/BO-Tutorial-for-Sci/blob/main/examples/HER"
            ],
        },
        {
            "timestamp": "2026-03-19T20:05:00Z",
            "event_type": "setup_frozen",
            "summary": "Froze evaluator and search space before the reported BO run.",
            "artifact_paths": [
                "research_runs/her_demo/discovered_search_space.json",
                "research_runs/her_demo/evaluator.py",
            ],
        },
    ]

    assert validate_operationalization_events(events) == []


def test_validate_operationalization_events_rejects_bad_shape() -> None:
    events = [
        {
            "timestamp": "2026-03-19T20:00:00Z",
            "event_type": "made_up_event",
            "summary": "bad event type",
            "artifact_paths": "not-a-list",
        }
    ]

    errors = validate_operationalization_events(events)

    assert any("unsupported event_type" in error for error in errors)
    assert any("artifact_paths must be a list" in error for error in errors)


def test_validate_open_world_research_run_contract(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    research_dir = repo_root / "research_runs" / "her_demo"
    verification_path = research_dir / "verification_artifacts" / "volcano.png"
    helper_path = research_dir / "converter.py"

    _write(research_dir / "research_plan.md", "# plan\n")
    _write(research_dir / "paper.md", "# paper\n")
    _write(research_dir / "initial_prompt.md", "Discover a useful HER evaluator.\n")
    _write(research_dir / "discovered_search_space.json", "{}\n")
    _write(research_dir / "evaluator.py", "def evaluate(x):\n    return 1.0\n")
    _write(helper_path, "def convert(x):\n    return x\n")
    _write(verification_path, "fake-image\n")

    operationalization_events = [
        {
            "timestamp": "2026-03-19T20:00:00Z",
            "event_type": "source_selected",
            "summary": "Selected HER tutorial source.",
            "artifact_paths": ["research_runs/her_demo/initial_prompt.md"],
            "source_urls": [
                "https://github.com/zwyu-ai/BO-Tutorial-for-Sci/blob/main/examples/HER"
            ],
        },
        {
            "timestamp": "2026-03-19T20:10:00Z",
            "event_type": "dependency_installed",
            "summary": "Installed one small package for plotting.",
            "artifact_paths": ["research_runs/her_demo/verification_artifacts/volcano.png"],
            "reason": "Needed to render the verification artifact cleanly.",
        },
        {
            "timestamp": "2026-03-19T20:15:00Z",
            "event_type": "setup_frozen",
            "summary": "Froze final setup before BO.",
            "artifact_paths": [
                "research_runs/her_demo/discovered_search_space.json",
                "research_runs/her_demo/evaluator.py",
            ],
        },
    ]
    _write(
        research_dir / "operationalization_log.jsonl",
        "\n".join(json.dumps(event) for event in operationalization_events) + "\n",
    )

    state = {
        "research_id": "her_demo",
        "research_question": "Discover a useful HER catalyst setup.",
        "open_world": {
            "nudge_tier": "N0",
            "prompt_path": "research_runs/her_demo/initial_prompt.md",
            "source_urls": [
                "https://github.com/zwyu-ai/BO-Tutorial-for-Sci/blob/main/examples/HER"
            ],
            "discovered_search_space_path": "research_runs/her_demo/discovered_search_space.json",
            "evaluator_module_path": "research_runs/her_demo/evaluator.py",
            "helper_script_paths": ["research_runs/her_demo/converter.py"],
            "verification_artifacts": [
                "research_runs/her_demo/verification_artifacts/volcano.png"
            ],
            "dependency_installs": [
                {
                    "packages": ["matplotlib"],
                    "command": "uv pip install matplotlib",
                    "reason": "Needed to render the verification artifact cleanly.",
                }
            ],
            "approach_revisions": [
                {
                    "timestamp": "2026-03-19T20:08:00Z",
                    "reason": "Switched from a descriptor-only framing to a composition-level setup.",
                    "changed": ["search_space", "verification"],
                }
            ],
            "final_setup_frozen": True,
        },
    }
    _write(
        research_dir / "research_state.json",
        json.dumps(state, indent=2) + "\n",
    )

    assert validate_open_world_research_run(research_dir) == []
