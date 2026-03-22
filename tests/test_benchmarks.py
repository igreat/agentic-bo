"""Tests for benchmark packaging helpers."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchmarks.build_workspace as build_workspace_module
from benchmarks.build_workspace import build_workspace
from bo_workflow.engine import BOEngine
from bo_workflow.evaluation.cli import run_hidden_oracle_evaluator
from bo_workflow.evaluation.cli import run_python_module_evaluator
from bo_workflow.evaluation.oracle import build_proxy_oracle
from bo_workflow.evaluation.python_module import _normalize_python_evaluator_result
from bo_workflow.utils import RunPaths, read_jsonl


def test_build_workspace_copies_public_oer_bundle(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "benchmark_workspace"

    build_workspace(
        output_dir=output_dir,
        task_ids=["oer"],
        overwrite=False,
    )

    assert (output_dir / "AGENTS.md").exists()
    assert (output_dir / "bo_workflow").is_dir()
    assert (output_dir / ".agents").is_dir()
    assert (output_dir / ".claude").is_dir()
    claude_settings = json.loads(
        (output_dir / ".claude" / "settings.local.json").read_text()
    )
    assert claude_settings["defaultMode"] == "acceptEdits"
    assert claude_settings["permissions"]["allow"] == ["Bash"]
    assert claude_settings["permissions"]["deny"] == ["WebSearch", "WebFetch"]
    assert not (output_dir / "benchmarks").exists()
    assert (output_dir / "tasks" / "oer" / "brief.md").exists()
    assert (output_dir / "tasks" / "oer" / "literature" / "background.md").exists()
    assert not (output_dir / "tasks" / "oer" / "assessment.md").exists()
    manifest = json.loads((output_dir / "tasks" / "oer" / "task_manifest.json").read_text())
    assert manifest["workflow"]["entrypoint"] == "research-agent"
    assert (output_dir / "bo_runs").is_dir()
    assert (output_dir / "research_runs").is_dir()
    backend_id = manifest["evaluation"]["backend_id"]
    source_backend = build_workspace_module.repo_root() / "evaluation_backends" / backend_id
    copied_backend = output_dir / "evaluation_backends" / backend_id
    if source_backend.exists():
        assert copied_backend.exists()
    else:
        assert not (output_dir / "evaluation_backends").exists()


def test_run_evaluator_with_prebuilt_backend_records_observations(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "toy.csv"
    pd.DataFrame(
        [
            {"x": 0.0, "target": 0.1},
            {"x": 0.1, "target": 0.2},
            {"x": 0.2, "target": 0.3},
            {"x": 0.3, "target": 0.35},
            {"x": 0.4, "target": 0.4},
            {"x": 0.5, "target": 0.45},
            {"x": 0.6, "target": 0.5},
            {"x": 0.7, "target": 0.55},
        ]
    ).to_csv(dataset_path, index=False)

    runs_root = tmp_path / "bo_runs"
    backends_root = tmp_path / "evaluation_backends"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="target",
        objective="max",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 0.7}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )
    run_id = state["run_id"]

    build_proxy_oracle(
        dataset_path=dataset_path,
        target_column="target",
        objective="max",
        backend_dir=backends_root / "toy_backend",
        seed=42,
    )

    payload = run_hidden_oracle_evaluator(
        engine,
        run_id=run_id,
        backend_dir=backends_root / "toy_backend",
        num_iterations=2,
        batch_size=1,
    )
    assert payload["backend_id"] == "toy_backend"
    paths = RunPaths(run_dir=runs_root / run_id)
    assert paths.report.exists()
    assert paths.convergence_plot.exists()
    report = json.loads(paths.report.read_text())
    assert report["best_observation_number"] == report["best_iteration"] + 1
    assert report["oracle"]["source"] == "evaluation_backend_metadata"
    assert "cv_rmse" in report["oracle"]
    assert report["trajectory"]["random_phase"]["num_observations"] == 2
    assert report["trajectory"]["random_phase"]["start_observation"] == 1
    assert report["trajectory"]["random_phase"]["end_observation"] == 2
    assert "model_guided_phase" not in report["trajectory"]
    observations = read_jsonl(paths.observations)
    assert len(observations) == 2
    assert {row["source"] for row in observations} == {"benchmark-evaluator"}


def test_run_python_evaluator_records_observations_and_resolves_pending(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "blackbox_her.py"
    module_path.write_text(
        "\n".join(
            [
                "def evaluate(composition):",
                "    x = float(composition['x'])",
                "    y = 1.0 - abs(x - 0.3)",
                "    return {",
                "        'y': y,",
                "        'score_raw': y - 0.1,",
                "        'score_calibrated': y,",
                "        'failure_reason': None,",
                "        'x': {'hacked': True},",
                "        'engine': 'not-the-engine',",
                "        'suggestion_id': 'fake-id',",
                "    }",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="exchange_current_density",
        objective="max",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )
    run_id = state["run_id"]

    engine.suggest(run_id, batch_size=1)

    payload = run_python_module_evaluator(
        engine,
        run_id=run_id,
        module_path=module_path,
        num_iterations=2,
        batch_size=1,
    )

    paths = RunPaths(run_dir=runs_root / run_id)
    report = json.loads(paths.report.read_text())
    observations = read_jsonl(paths.observations)

    assert payload["resolved_pending"] == 1
    assert payload["recorded"] == 3
    assert report["num_observations"] == 3
    assert report["oracle"]["source"] == "python_evaluator_module"
    assert report["oracle"]["selected_model"] == "python-callback"
    assert {row["source"] for row in observations} == {"python-evaluator"}
    assert {row["engine"] for row in observations} == {"hebo"}
    assert all("hacked" not in row["x"] for row in observations)
    assert all(row["suggestion_id"] != "fake-id" for row in observations)
    assert all("score_raw" in row for row in observations)
    assert all("score_calibrated" in row for row in observations)
    assert all("failure_reason" in row for row in observations)


def test_normalize_python_evaluator_result_accepts_scalar_output() -> None:
    suggestion = {
        "x": {"x": 0.3},
        "engine": "hebo",
        "suggestion_id": "abc123",
    }

    payload = _normalize_python_evaluator_result(
        result=0.42,
        suggestion=suggestion,
        target_column="exchange_current_density",
        default_engine="hebo",
    )

    assert payload == {
        "x": {"x": 0.3},
        "y": 0.42,
        "engine": "hebo",
        "suggestion_id": "abc123",
    }


def test_normalize_python_evaluator_result_accepts_target_column_key() -> None:
    suggestion = {
        "x": {"x": 0.3},
        "engine": "botorch",
        "suggestion_id": "abc123",
    }

    payload = _normalize_python_evaluator_result(
        result={
            "exchange_current_density": 0.42,
            "score_raw": 0.39,
            "engine": "clobber-me",
        },
        suggestion=suggestion,
        target_column="exchange_current_density",
        default_engine="hebo",
    )

    assert payload == {
        "x": {"x": 0.3},
        "y": 0.42,
        "engine": "botorch",
        "suggestion_id": "abc123",
        "score_raw": 0.39,
    }


def test_observe_preserves_engine_metadata_when_extras_collide(tmp_path: Path) -> None:
    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="target",
        objective="min",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )
    run_id = state["run_id"]

    engine.observe(
        run_id,
        [
            {
                "x": {"x": 0.2},
                "y": 0.5,
                "event_time": "spoofed",
                "iteration": 999,
                "source": "spoofed",
                "y_internal": 123.0,
                "score_raw": 0.45,
            }
        ],
        source="benchmark-evaluator",
    )

    observation = read_jsonl(RunPaths(run_dir=runs_root / run_id).observations)[0]
    assert observation["source"] == "benchmark-evaluator"
    assert observation["iteration"] == 0
    assert observation["y_internal"] == 0.5
    assert observation["event_time"] != "spoofed"
    assert observation["score_raw"] == 0.45


def test_observe_recursively_normalizes_nested_extras(tmp_path: Path) -> None:
    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="target",
        objective="min",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )
    run_id = state["run_id"]

    engine.observe(
        run_id,
        [
            {
                "x": {"x": 0.2},
                "y": 0.5,
                "diagnostics": {
                    "score_raw": np.float64(0.45),
                    "components": np.array([1.0, 2.0]),
                    "failures": [np.int64(0), np.int64(1)],
                },
            }
        ],
        source="benchmark-evaluator",
    )

    observation = read_jsonl(RunPaths(run_dir=runs_root / run_id).observations)[0]
    assert observation["diagnostics"] == {
        "score_raw": 0.45,
        "components": [1.0, 2.0],
        "failures": [0, 1],
    }


def test_run_python_evaluator_zero_iterations_only_resolves_pending(
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "blackbox_her.py"
    module_path.write_text(
        "\n".join(
            [
                "def evaluate(composition):",
                "    x = float(composition['x'])",
                "    return {'y': 1.0 - abs(x - 0.3)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="exchange_current_density",
        objective="max",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )
    run_id = state["run_id"]

    suggestion = engine.suggest(run_id, batch_size=1)["suggestions"][0]

    payload = run_python_module_evaluator(
        engine,
        run_id=run_id,
        module_path=module_path,
        num_iterations=0,
        batch_size=1,
    )

    paths = RunPaths(run_dir=runs_root / run_id)
    report = json.loads(paths.report.read_text())
    final_state = json.loads(paths.state.read_text())
    observations = read_jsonl(paths.observations)

    assert payload["resolved_pending"] == 1
    assert payload["recorded"] == 1
    assert final_state["status"] == "running"
    assert report["num_observations"] == 1
    assert observations[0]["suggestion_id"] == suggestion["suggestion_id"]


def test_run_python_evaluator_supports_sibling_helper_imports(
    tmp_path: Path,
) -> None:
    helper_path = tmp_path / "helper.py"
    helper_path.write_text(
        "\n".join(
            [
                "def objective(x):",
                "    return 1.0 - abs(x - 0.3)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    module_path = tmp_path / "blackbox_her.py"
    module_path.write_text(
        "\n".join(
            [
                "from helper import objective",
                "",
                "def evaluate(composition):",
                "    x = float(composition['x'])",
                "    return {'y': objective(x)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="exchange_current_density",
        objective="max",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )

    payload = run_python_module_evaluator(
        engine,
        run_id=state["run_id"],
        module_path=module_path,
        num_iterations=1,
        batch_size=1,
    )

    assert payload["recorded"] == 1


def test_run_python_evaluator_supports_deferred_sibling_helper_imports(
    tmp_path: Path,
) -> None:
    helper_path = tmp_path / "helper.py"
    helper_path.write_text(
        "\n".join(
            [
                "def objective(x):",
                "    return 1.0 - abs(x - 0.3)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    module_path = tmp_path / "blackbox_her.py"
    module_path.write_text(
        "\n".join(
            [
                "def evaluate(composition):",
                "    from helper import objective",
                "",
                "    x = float(composition['x'])",
                "    return {'y': objective(x)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="exchange_current_density",
        objective="max",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )

    payload = run_python_module_evaluator(
        engine,
        run_id=state["run_id"],
        module_path=module_path,
        num_iterations=1,
        batch_size=1,
    )

    assert payload["recorded"] == 1


def test_report_trajectory_summary_matches_observations(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="target",
        objective="min",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )
    run_id = state["run_id"]

    suggestions = engine.suggest(run_id, batch_size=4)["suggestions"]
    engine.observe(
        run_id,
        [
            {"x": {"x": 0.1}, "y": 0.42, "suggestion_id": suggestions[0]["suggestion_id"]},
            {"x": {"x": 0.2}, "y": 0.36, "suggestion_id": suggestions[1]["suggestion_id"]},
            {"x": {"x": 0.3}, "y": 0.37, "suggestion_id": suggestions[2]["suggestion_id"]},
            {"x": {"x": 0.4}, "y": 0.355, "suggestion_id": suggestions[3]["suggestion_id"]},
        ],
        source="benchmark-evaluator",
    )

    report = engine.report(run_id)
    trajectory = report["trajectory"]

    assert report["best_value"] == 0.355
    assert report["best_iteration"] == 3
    assert report["best_observation_number"] == 4
    assert trajectory["best_observation_number"] == 4
    assert trajectory["last_improvement_observation"] == 4
    assert trajectory["observed_range"] == {
        "min_value": 0.355,
        "max_value": 0.42,
    }
    assert trajectory["random_phase"] == {
        "num_observations": 2,
        "start_observation": 1,
        "end_observation": 2,
        "min_value": 0.36,
        "max_value": 0.42,
        "best_value": 0.36,
        "best_observation_number": 2,
    }
    assert trajectory["model_guided_phase"]["num_observations"] == 2
    assert trajectory["model_guided_phase"]["start_observation"] == 3
    assert trajectory["model_guided_phase"]["end_observation"] == 4
    assert trajectory["model_guided_phase"]["min_value"] == pytest.approx(0.355)
    assert trajectory["model_guided_phase"]["max_value"] == pytest.approx(0.37)
    assert trajectory["model_guided_phase"]["best_value"] == pytest.approx(0.355)
    assert trajectory["model_guided_phase"]["best_observation_number"] == 4
    assert trajectory["model_guided_phase"]["improvement_over_random_best"] == pytest.approx(
        0.005
    )


def test_report_trajectory_summary_treats_random_engine_as_fully_random(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="target",
        objective="min",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        default_engine="random",
        seed=42,
        num_initial_random_samples=2,
    )
    run_id = state["run_id"]

    suggestions = engine.suggest(run_id, batch_size=4)["suggestions"]
    engine.observe(
        run_id,
        [
            {"x": {"x": 0.1}, "y": 0.42, "suggestion_id": suggestions[0]["suggestion_id"]},
            {"x": {"x": 0.2}, "y": 0.36, "suggestion_id": suggestions[1]["suggestion_id"]},
            {"x": {"x": 0.3}, "y": 0.37, "suggestion_id": suggestions[2]["suggestion_id"]},
            {"x": {"x": 0.4}, "y": 0.355, "suggestion_id": suggestions[3]["suggestion_id"]},
        ],
        source="benchmark-evaluator",
    )

    report = engine.report(run_id)
    trajectory = report["trajectory"]

    assert trajectory["random_phase"] == {
        "num_observations": 4,
        "start_observation": 1,
        "end_observation": 4,
        "min_value": 0.355,
        "max_value": 0.42,
        "best_value": 0.355,
        "best_observation_number": 4,
    }
    assert "model_guided_phase" not in trajectory


def test_report_trajectory_summary_treats_pre_suggest_observations_as_seed_phase(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "bo_runs"
    engine = BOEngine(runs_root=runs_root)
    state = engine.init_run(
        target_column="target",
        objective="min",
        search_space_spec={
            "design_parameters": [
                {"name": "x", "type": "num", "lb": 0.0, "ub": 1.0}
            ],
            "fixed_features": {},
        },
        seed=42,
        num_initial_random_samples=2,
    )
    run_id = state["run_id"]

    # Pre-suggest observations are warm-start seed data and should be summarized
    # separately from the BO-controlled phase split.
    engine.observe(
        run_id,
        [
            {"x": {"x": 0.01}, "y": 0.50},
            {"x": {"x": 0.02}, "y": 0.49},
        ],
        source="pool-seed",
    )

    # Later BO observations may be recorded without suggestion_id in manual
    # suggest/observe flows; they should still be treated as BO-phase rows once
    # suggestions exist.
    engine.suggest(run_id, batch_size=4)
    engine.observe(
        run_id,
        [
            {"x": {"x": 0.10}, "y": 0.42, "engine": "hebo"},
            {"x": {"x": 0.20}, "y": 0.36, "engine": "hebo"},
            {"x": {"x": 0.30}, "y": 0.37, "engine": "hebo"},
            {"x": {"x": 0.40}, "y": 0.355, "engine": "hebo"},
        ],
        source="benchmark-evaluator",
    )

    report = engine.report(run_id)
    trajectory = report["trajectory"]

    assert trajectory["seed_phase"] == {
        "num_observations": 2,
        "start_observation": 1,
        "end_observation": 2,
        "min_value": 0.49,
        "max_value": 0.5,
        "best_value": 0.49,
        "best_observation_number": 2,
    }
    assert "random_phase" not in trajectory
    assert trajectory["model_guided_phase"]["num_observations"] == 4
    assert trajectory["model_guided_phase"]["start_observation"] == 3
    assert trajectory["model_guided_phase"]["end_observation"] == 6
    assert trajectory["model_guided_phase"]["best_observation_number"] == 6
    assert trajectory["model_guided_phase"]["improvement_over_seed_best"] == pytest.approx(
        0.135
    )


def test_build_workspace_rejects_overwriting_repo_or_ancestor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_root = tmp_path / "repo"
    fake_root.mkdir(parents=True)
    monkeypatch.setattr(build_workspace_module, "repo_root", lambda: fake_root)

    with pytest.raises(ValueError, match="unsafe output directory"):
        build_workspace(output_dir=fake_root, task_ids=["oer"], overwrite=True)

    with pytest.raises(ValueError, match="unsafe output directory"):
        build_workspace(output_dir=fake_root.parent, task_ids=["oer"], overwrite=True)

    with pytest.raises(ValueError, match="inside repo"):
        build_workspace(
            output_dir=fake_root / "nested-output",
            task_ids=["oer"],
            overwrite=True,
        )

    with pytest.raises(ValueError, match="inside repo"):
        build_workspace(
            output_dir=fake_root / "nested-output-no-overwrite",
            task_ids=["oer"],
            overwrite=False,
        )


def test_build_workspace_rejects_unknown_task_id(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Unknown benchmark task bundle"):
        build_workspace(
            output_dir=tmp_path / "workspace",
            task_ids=["definitely-not-a-task"],
            overwrite=False,
        )


def test_build_workspace_copies_prebuilt_backend_when_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_root = tmp_path / "repo"
    fake_root.mkdir(parents=True)
    (fake_root / "AGENTS.md").write_text("agents\n", encoding="utf-8")
    (fake_root / "README.md").write_text("readme\n", encoding="utf-8")
    (fake_root / "pyproject.toml").write_text("[project]\nname='fake'\nversion='0.1.0'\n", encoding="utf-8")
    (fake_root / "uv.lock").write_text("", encoding="utf-8")
    (fake_root / ".python-version").write_text("3.14\n", encoding="utf-8")
    (fake_root / ".gitignore").write_text("", encoding="utf-8")
    (fake_root / "bo_workflow").mkdir(parents=True)
    (fake_root / "bo_workflow" / "__init__.py").write_text("", encoding="utf-8")
    (fake_root / ".agents" / "skills").mkdir(parents=True)
    (fake_root / ".claude" / "skills").mkdir(parents=True)
    (fake_root / ".claude" / "README.md").write_text("claude\n", encoding="utf-8")
    task_dir = fake_root / "benchmarks" / "tasks" / "oer"
    task_dir.mkdir(parents=True)
    (task_dir / "brief.md").write_text("brief\n", encoding="utf-8")
    (task_dir / "task_manifest.json").write_text(
        json.dumps(
            {
                "task_id": "oer",
                "evaluation": {
                    "mode": "prebuilt_backend",
                    "backend_id": "oer_hidden",
                },
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "search_space.json").write_text("{}", encoding="utf-8")
    backend_dir = fake_root / "evaluation_backends" / "oer_hidden"
    backend_dir.mkdir(parents=True)
    (backend_dir / "oracle.pkl").write_bytes(b"pickle")
    (backend_dir / "oracle_meta.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(build_workspace_module, "repo_root", lambda: fake_root)

    output_dir = tmp_path / "public_workspace"
    build_workspace(
        output_dir=output_dir,
        task_ids=["oer"],
        overwrite=False,
    )

    assert (output_dir / "evaluation_backends" / "oer_hidden" / "oracle.pkl").exists()
    assert (
        output_dir / "evaluation_backends" / "oer_hidden" / "oracle_meta.json"
    ).exists()
    claude_settings = json.loads(
        (output_dir / ".claude" / "settings.local.json").read_text()
    )
    assert claude_settings["defaultMode"] == "acceptEdits"
    assert claude_settings["permissions"]["allow"] == ["Bash"]
    assert claude_settings["permissions"]["deny"] == ["WebSearch", "WebFetch"]
