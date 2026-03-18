"""Tests for benchmark packaging helpers."""

import json
from pathlib import Path

import pandas as pd

import benchmarks.build_workspace as build_workspace_module
from benchmarks.build_workspace import build_workspace
from bo_workflow.engine import BOEngine
from bo_workflow.evaluation.cli import run_hidden_oracle_evaluator
from bo_workflow.evaluation.oracle import build_proxy_oracle
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
    assert not (output_dir / "benchmarks").exists()
    assert (output_dir / "tasks" / "oer" / "brief.md").exists()
    assert (output_dir / "tasks" / "oer" / "literature" / "background.md").exists()
    assert not (output_dir / "tasks" / "oer" / "assessment.md").exists()
    assert (output_dir / "bo_runs").is_dir()
    assert (output_dir / "research_runs").is_dir()
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
    observations = read_jsonl(paths.observations)
    assert len(observations) == 2
    assert {row["source"] for row in observations} == {"benchmark-evaluator"}


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
