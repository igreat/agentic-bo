import json
from pathlib import Path

from bo_workflow.open_world import append_operationalization_event
from bo_workflow.open_world import main
from bo_workflow.open_world import scaffold_open_world_research_dir
from bo_workflow.open_world import validate_open_world_operator_spec
from bo_workflow.open_world import validate_open_world_operator_spec_file
from bo_workflow.open_world import validate_open_world_research_run
from bo_workflow.open_world import validate_operationalization_events
from bo_workflow.open_world import write_initial_prompt


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


def test_scaffold_and_prompt_helpers_create_expected_open_world_artifacts(
    tmp_path: Path,
) -> None:
    research_dir = tmp_path / "research_runs" / "her_demo"

    paths = scaffold_open_world_research_dir(research_dir)
    prompt_path = write_initial_prompt(
        research_dir,
        prompt_text="Find a useful HER setup and optimize it.",
        nudge_tier="N0",
    )

    assert paths["verification_dir"].is_dir()
    assert paths["operationalization_log_path"].exists()
    assert paths["research_state_path"].exists()
    assert paths["research_plan_path"].exists()
    assert paths["paper_path"].exists()
    assert paths["search_space_path"].exists()
    assert paths["evaluator_path"].exists()
    assert prompt_path.exists()
    state = json.loads(paths["research_state_path"].read_text(encoding="utf-8"))
    assert state["research_id"] == "her_demo"
    assert state["open_world"]["prompt_path"] == "research_runs/her_demo/initial_prompt.md"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "**Nudge Tier:** N0" in prompt_text
    assert "Find a useful HER setup and optimize it." in prompt_text


def test_append_operationalization_event_writes_jsonl(
    tmp_path: Path,
) -> None:
    research_dir = tmp_path / "research_runs" / "her_demo"

    event = append_operationalization_event(
        research_dir,
        event_type="source_selected",
        summary="Selected the HER tutorial example.",
        artifact_paths=["research_runs/her_demo/initial_prompt.md"],
        source_urls=[
            "https://github.com/zwyu-ai/BO-Tutorial-for-Sci/blob/main/examples/HER"
        ],
    )

    log_path = research_dir / "operationalization_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == event


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
        {
            "timestamp": "2026-03-19T20:16:00Z",
            "event_type": "bo_started",
            "summary": "Started the reported BO run.",
            "artifact_paths": ["bo_runs/her_demo_bo/state.json"],
        },
        {
            "timestamp": "2026-03-19T20:17:00Z",
            "event_type": "bo_completed",
            "summary": "Completed the reported BO run.",
            "artifact_paths": ["bo_runs/her_demo_bo/report.json"],
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


def test_validate_open_world_research_run_requires_setup_frozen_event_for_frozen_state(
    tmp_path: Path,
) -> None:
    research_dir = tmp_path / "research_runs" / "her_demo"
    verification_path = research_dir / "verification_artifacts" / "volcano.png"

    _write(research_dir / "research_plan.md", "# plan\n")
    _write(research_dir / "paper.md", "# paper\n")
    _write(research_dir / "initial_prompt.md", "Discover a useful HER evaluator.\n")
    _write(research_dir / "discovered_search_space.json", "{}\n")
    _write(research_dir / "evaluator.py", "def evaluate(x):\n    return 1.0\n")
    _write(verification_path, "fake-image\n")
    _write(
        research_dir / "operationalization_log.jsonl",
        json.dumps(
            {
                "timestamp": "2026-03-19T20:00:00Z",
                "event_type": "source_selected",
                "summary": "Selected HER tutorial source.",
                "artifact_paths": ["research_runs/her_demo/initial_prompt.md"],
            }
        )
        + "\n",
    )
    _write(
        research_dir / "research_state.json",
        json.dumps(
            {
                "research_id": "her_demo",
                "research_question": "Discover a useful HER setup.",
                "open_world": {
                    "nudge_tier": "N0",
                    "prompt_path": "research_runs/her_demo/initial_prompt.md",
                    "source_urls": [
                        "https://github.com/zwyu-ai/BO-Tutorial-for-Sci/blob/main/examples/HER"
                    ],
                    "discovered_search_space_path": "research_runs/her_demo/discovered_search_space.json",
                    "evaluator_module_path": "research_runs/her_demo/evaluator.py",
                    "helper_script_paths": [],
                    "verification_artifacts": [
                        "research_runs/her_demo/verification_artifacts/volcano.png"
                    ],
                    "dependency_installs": [],
                    "approach_revisions": [],
                    "final_setup_frozen": True,
                },
            },
            indent=2,
        )
        + "\n",
    )

    errors = validate_open_world_research_run(research_dir)

    assert any("missing setup_frozen event" in error for error in errors)


def test_validate_open_world_research_run_requires_frozen_setup_before_bo_started(
    tmp_path: Path,
) -> None:
    research_dir = tmp_path / "research_runs" / "her_demo"
    verification_path = research_dir / "verification_artifacts" / "volcano.png"

    _write(research_dir / "research_plan.md", "# plan\n")
    _write(research_dir / "paper.md", "# paper\n")
    _write(research_dir / "initial_prompt.md", "Discover a useful HER evaluator.\n")
    _write(research_dir / "discovered_search_space.json", "{}\n")
    _write(research_dir / "evaluator.py", "def evaluate(x):\n    return 1.0\n")
    _write(verification_path, "fake-image\n")
    _write(
        research_dir / "operationalization_log.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-03-19T20:00:00Z",
                        "event_type": "bo_started",
                        "summary": "Started BO too early.",
                        "artifact_paths": ["bo_runs/her_demo_bo/state.json"],
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-03-19T20:01:00Z",
                        "event_type": "setup_frozen",
                        "summary": "Froze the setup only after BO had started.",
                        "artifact_paths": [
                            "research_runs/her_demo/discovered_search_space.json",
                            "research_runs/her_demo/evaluator.py",
                        ],
                    }
                ),
            ]
        )
        + "\n",
    )
    _write(
        research_dir / "research_state.json",
        json.dumps(
            {
                "research_id": "her_demo",
                "research_question": "Discover a useful HER setup.",
                "open_world": {
                    "nudge_tier": "N0",
                    "prompt_path": "research_runs/her_demo/initial_prompt.md",
                    "source_urls": [
                        "https://github.com/zwyu-ai/BO-Tutorial-for-Sci/blob/main/examples/HER"
                    ],
                    "discovered_search_space_path": "research_runs/her_demo/discovered_search_space.json",
                    "evaluator_module_path": "research_runs/her_demo/evaluator.py",
                    "helper_script_paths": [],
                    "verification_artifacts": [
                        "research_runs/her_demo/verification_artifacts/volcano.png"
                    ],
                    "dependency_installs": [],
                    "approach_revisions": [],
                    "final_setup_frozen": True,
                },
            },
            indent=2,
        )
        + "\n",
    )

    errors = validate_open_world_research_run(research_dir)

    assert any("setup_frozen must occur before bo_started" in error for error in errors)


def test_validate_open_world_operator_spec_accepts_expected_shape(
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "benchmarks" / "open_world_cases" / "her" / "agent_prompt.md"
    _write(prompt_path, "prompt\n")

    spec = {
        "task_id": "her_open_world",
        "title": "Hydrogen evolution reaction catalyst design",
        "agent_prompt_family": {
            "primary_prompt_path": "benchmarks/open_world_cases/her/agent_prompt.md",
            "nudge_tiers": {
                "N0": "base",
                "N1": "light",
                "N2": "strong",
            },
        },
        "canonical_solution": {
            "evaluator_family": "HER tutorial example",
            "design_parameter_family": "10 bounded continuous variables",
            "constraints": [],
            "verification_artifact": "sanity-check plot",
        },
        "acceptable_alternatives": [{"description": "similar public HER example"}],
        "evaluation_window": {"time_budget_minutes": 180},
        "success_checks": ["Prompt saved", "BO completed"],
    }

    assert validate_open_world_operator_spec(spec, root_dir=tmp_path) == []


def test_repository_her_operator_spec_is_valid() -> None:
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "open_world_cases"
        / "her"
        / "operator_spec.json"
    )

    assert validate_open_world_operator_spec_file(spec_path) == []


def test_open_world_cli_scaffold_creates_expected_structure(
    tmp_path: Path,
) -> None:
    research_dir = tmp_path / "research_runs" / "her_demo"

    exit_code = main(["scaffold", "--research-dir", str(research_dir)])

    assert exit_code == 0
    assert (research_dir / "verification_artifacts").is_dir()
    assert (research_dir / "operationalization_log.jsonl").exists()
    assert (research_dir / "research_state.json").exists()
    assert (research_dir / "research_plan.md").exists()
    assert (research_dir / "paper.md").exists()
    assert (research_dir / "discovered_search_space.json").exists()
    assert (research_dir / "evaluator.py").exists()


def test_open_world_cli_write_prompt_creates_initial_prompt(
    tmp_path: Path,
) -> None:
    research_dir = tmp_path / "research_runs" / "her_demo"

    exit_code = main(
        [
            "write-prompt",
            "--research-dir",
            str(research_dir),
            "--nudge-tier",
            "N1",
            "--prompt-text",
            "Find a computable HER evaluator and optimize it.",
            "--nudge-text",
            "Look for a literature-grounded public tutorial or repo.",
        ]
    )

    prompt_path = research_dir / "initial_prompt.md"
    assert exit_code == 0
    assert prompt_path.exists()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert "**Nudge Tier:** N1" in prompt_text
    assert "Find a computable HER evaluator and optimize it." in prompt_text
    assert "Look for a literature-grounded public tutorial or repo." in prompt_text


def test_open_world_cli_log_event_appends_jsonl(
    tmp_path: Path,
) -> None:
    research_dir = tmp_path / "research_runs" / "her_demo"

    exit_code = main(
        [
            "log-event",
            "--research-dir",
            str(research_dir),
            "--event-type",
            "source_selected",
            "--summary",
            "Selected a public HER tutorial source.",
            "--artifact-path",
            "research_runs/her_demo/initial_prompt.md",
            "--source-url",
            "https://github.com/zwyu-ai/BO-Tutorial-for-Sci/blob/main/examples/HER",
        ]
    )

    log_path = research_dir / "operationalization_log.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert exit_code == 0
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "source_selected"
    assert event["artifact_paths"] == ["research_runs/her_demo/initial_prompt.md"]


def test_open_world_cli_validate_run_handles_unfrozen_runs(
    tmp_path: Path,
) -> None:
    research_dir = tmp_path / "research_runs" / "her_demo"
    verification_path = research_dir / "verification_artifacts" / "volcano.png"

    _write(research_dir / "research_plan.md", "# plan\n")
    _write(research_dir / "paper.md", "# paper\n")
    _write(research_dir / "initial_prompt.md", "Find a useful HER evaluator.\n")
    _write(research_dir / "discovered_search_space.json", "{}\n")
    _write(research_dir / "evaluator.py", "def evaluate(x):\n    return 1.0\n")
    _write(verification_path, "fake-image\n")
    _write(
        research_dir / "operationalization_log.jsonl",
        json.dumps(
            {
                "timestamp": "2026-03-19T20:00:00Z",
                "event_type": "source_selected",
                "summary": "Selected HER tutorial source.",
                "artifact_paths": ["research_runs/her_demo/initial_prompt.md"],
            }
        )
        + "\n",
    )
    _write(
        research_dir / "research_state.json",
        json.dumps(
            {
                "research_id": "her_demo",
                "research_question": "Discover a useful HER setup.",
                "open_world": {
                    "nudge_tier": "N0",
                    "prompt_path": "research_runs/her_demo/initial_prompt.md",
                    "source_urls": [
                        "https://github.com/zwyu-ai/BO-Tutorial-for-Sci/blob/main/examples/HER"
                    ],
                    "discovered_search_space_path": "research_runs/her_demo/discovered_search_space.json",
                    "evaluator_module_path": "research_runs/her_demo/evaluator.py",
                    "helper_script_paths": [],
                    "verification_artifacts": [
                        "research_runs/her_demo/verification_artifacts/volcano.png"
                    ],
                    "dependency_installs": [],
                    "approach_revisions": [],
                    "final_setup_frozen": False,
                },
            },
            indent=2,
        )
        + "\n",
    )

    strict_exit_code = main(["validate-run", "--research-dir", str(research_dir)])
    relaxed_exit_code = main(
        [
            "validate-run",
            "--research-dir",
            str(research_dir),
            "--allow-unfrozen",
        ]
    )

    assert strict_exit_code == 1
    assert relaxed_exit_code == 0


def test_open_world_cli_validate_spec_accepts_repository_her_spec() -> None:
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "open_world_cases"
        / "her"
        / "operator_spec.json"
    )

    exit_code = main(["validate-spec", "--path", str(spec_path)])

    assert exit_code == 0
