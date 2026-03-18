"""Tests for benchmark packaging helpers."""

import json
from pathlib import Path

import pandas as pd

from benchmarks.materialize_workspace import materialize_workspace
from benchmarks.run_task_evaluator import main as run_task_evaluator_main
from bo_workflow.engine import BOEngine
from bo_workflow.evaluation.oracle import build_proxy_oracle
from bo_workflow.utils import RunPaths, read_jsonl


def test_materialize_workspace_copies_public_oer_bundle(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "benchmark_workspace"

    materialize_workspace(
        output_dir=output_dir,
        task_ids=["oer"],
        overwrite=False,
    )

    assert (output_dir / "AGENTS.md").exists()
    assert (output_dir / "bo_workflow").is_dir()
    assert (output_dir / ".agents").is_dir()
    assert (output_dir / ".claude").is_dir()
    assert (output_dir / "benchmarks" / "run_task_evaluator.py").exists()
    assert not (output_dir / "benchmarks" / "scoring.md").exists()
    assert (output_dir / "benchmark_tasks" / "oer" / "brief.md").exists()
    assert (
        output_dir
        / "benchmark_tasks"
        / "oer"
        / "literature"
        / "background.md"
    ).exists()
    assert not (output_dir / "benchmark_tasks" / "oer" / "assessment.md").exists()
    assert (output_dir / "bo_runs").is_dir()
    assert (output_dir / "research_runs").is_dir()
    assert not (output_dir / "evaluation_backends").exists()


def test_run_task_evaluator_resolves_handle_map_and_records_observations(
    tmp_path: Path,
    monkeypatch,
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
    )
    run_id = state["run_id"]

    build_proxy_oracle(
        dataset_path=dataset_path,
        target_column="target",
        objective="max",
        backend_dir=backends_root / "toy_backend",
        seed=42,
    )

    task_manifest = tmp_path / "task_manifest.json"
    task_manifest.write_text(
        json.dumps(
            {
                "task_id": "toy",
                "budget": {"iterations": 2, "batch_size": 1},
                "evaluation": {
                    "mode": "external_hidden",
                    "runner": "benchmark-evaluator",
                    "handle": "toy_v1",
                },
            }
        ),
        encoding="utf-8",
    )

    handle_map = tmp_path / "handle_map.json"
    handle_map.write_text(
        json.dumps({"toy_v1": {"backend_id": "toy_backend"}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("BENCHMARK_HANDLE_MAP", str(handle_map))
    monkeypatch.setenv("BENCHMARK_BACKENDS_ROOT", str(backends_root))
    monkeypatch.setenv("BENCHMARK_RUNS_ROOT", str(runs_root))

    exit_code = run_task_evaluator_main(
        ["--task-manifest", str(task_manifest), "--run-id", run_id]
    )

    assert exit_code == 0
    paths = RunPaths(run_dir=runs_root / run_id)
    assert paths.report.exists()
    assert paths.convergence_plot.exists()
    observations = read_jsonl(paths.observations)
    assert len(observations) == 2
    assert {row["source"] for row in observations} == {"benchmark-evaluator"}


def test_run_task_evaluator_output_hides_backend_identity(
    tmp_path: Path,
    monkeypatch,
    capsys,
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
    )
    run_id = state["run_id"]

    build_proxy_oracle(
        dataset_path=dataset_path,
        target_column="target",
        objective="max",
        backend_dir=backends_root / "toy_backend",
        seed=42,
    )

    task_manifest = tmp_path / "task_manifest.json"
    task_manifest.write_text(
        json.dumps(
            {
                "task_id": "toy",
                "budget": {"iterations": 1, "batch_size": 1},
                "evaluation": {
                    "mode": "external_hidden",
                    "runner": "benchmark-evaluator",
                    "handle": "toy_v1",
                },
            }
        ),
        encoding="utf-8",
    )

    handle_map = tmp_path / "handle_map.json"
    handle_map.write_text(
        json.dumps({"toy_v1": {"backend_id": "toy_backend"}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("BENCHMARK_HANDLE_MAP", str(handle_map))
    monkeypatch.setenv("BENCHMARK_BACKENDS_ROOT", str(backends_root))
    monkeypatch.setenv("BENCHMARK_RUNS_ROOT", str(runs_root))

    exit_code = run_task_evaluator_main(
        ["--task-manifest", str(task_manifest), "--run-id", run_id]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["evaluation_handle"] == "toy_v1"
    assert "backend_id" not in payload
    assert "backend_dir" not in payload
