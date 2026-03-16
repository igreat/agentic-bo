"""Integration tests for the BO workflow.

Each test exercises the Python API end-to-end (no subprocess CLI calls),
using tmp_path for full isolation between tests.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from bo_workflow.engine import BOEngine
from bo_workflow.evaluation.cli import handle as handle_evaluation
from bo_workflow.evaluation.cli import run_hidden_oracle_evaluator
from bo_workflow.evaluation.oracle import build_proxy_oracle
from bo_workflow.evaluation.proxy import ProxyObserver
from bo_workflow.utils import EvaluationBackendPaths, RunPaths, read_json, read_jsonl

ITERATIONS = 5


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _run_full_proxy_loop(
    engine: BOEngine,
    dataset_path: Path,
    target: str,
    objective: str,
    *,
    iterations: int = ITERATIONS,
    max_features: int | None = None,
) -> tuple[str, RunPaths, EvaluationBackendPaths]:
    """Init → build-oracle → run-proxy → report. Returns run and backend paths."""
    state = engine.init_run(
        dataset_path=dataset_path,
        target_column=target,
        objective=objective,
        seed=42,
    )
    run_id = state["run_id"]
    run_dir = engine.get_run_dir(run_id)
    backend_paths = EvaluationBackendPaths(
        engine.runs_root.parent / "evaluation_backends" / run_id
    )

    build_proxy_oracle(
        dataset_path=dataset_path,
        target_column=target,
        objective=objective,
        backend_dir=backend_paths.backend_dir,
        seed=42,
        max_features=max_features,
    )

    observer = ProxyObserver(backend_paths.backend_dir)
    engine.run_optimization(
        run_id,
        observer=observer,
        num_iterations=iterations,
    )

    paths = RunPaths(run_dir=run_dir)
    return run_id, paths, backend_paths


def _assert_standard_artifacts(
    paths: RunPaths,
    backend_paths: EvaluationBackendPaths,
    iterations: int = ITERATIONS,
) -> None:
    """Assert standard run artifacts exist and have expected content."""
    assert paths.state.exists()
    assert paths.input_spec.exists()
    assert paths.suggestions.exists()
    assert paths.observations.exists()
    assert paths.convergence_plot.exists()
    assert paths.report.exists()
    assert backend_paths.oracle_model.exists()
    assert backend_paths.oracle_meta.exists()

    state = read_json(paths.state)
    assert state["status"] == "completed"

    oracle_meta = read_json(backend_paths.oracle_meta)
    rmse = oracle_meta["selected_rmse"]
    assert math.isfinite(rmse) and rmse > 0

    report = read_json(paths.report)
    assert math.isfinite(report["best_value"])

    observations = read_jsonl(paths.observations)
    assert len(observations) == iterations

    suggestions = read_jsonl(paths.suggestions)
    assert len(suggestions) == iterations


# ------------------------------------------------------------------
# Happy-path full proxy loop tests
# ------------------------------------------------------------------


def test_her_full_proxy_loop(engine: BOEngine, her_csv: Path) -> None:
    """HER dataset, max objective, full proxy loop."""
    _, paths, backend_paths = _run_full_proxy_loop(engine, her_csv, "Target", "max")
    _assert_standard_artifacts(paths, backend_paths)


def test_her_full_proxy_loop_with_hebo_rf(engine: BOEngine, her_csv: Path) -> None:
    """HER dataset, max objective, full proxy loop using HEBO's RF surrogate."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        default_engine="hebo",
        hebo_model="rf",
        seed=42,
    )
    run_id = state["run_id"]
    run_dir = engine.get_run_dir(run_id)
    backend_paths = EvaluationBackendPaths(
        engine.runs_root.parent / "evaluation_backends" / run_id
    )

    build_proxy_oracle(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        backend_dir=backend_paths.backend_dir,
        seed=42,
        default_engine="hebo",
    )
    observer = ProxyObserver(backend_paths.backend_dir)
    engine.run_optimization(run_id, observer=observer, num_iterations=ITERATIONS)

    paths = RunPaths(run_dir=run_dir)
    _assert_standard_artifacts(paths, backend_paths)

    final_state = read_json(paths.state)
    assert final_state["hebo_model"] == "rf"

    report = read_json(paths.report)
    assert report["default_engine"] == "hebo"
    assert report["hebo_model"] == "rf"


def test_hea_full_proxy_loop(engine: BOEngine, hea_csv: Path) -> None:
    """HEA dataset, max objective, full proxy loop."""
    _, paths, backend_paths = _run_full_proxy_loop(engine, hea_csv, "target", "max")
    _assert_standard_artifacts(paths, backend_paths)


def test_oer_mixed_variables(engine: BOEngine, oer_csv: Path) -> None:
    """OER dataset, min objective, verifies categorical detection."""
    _, paths, backend_paths = _run_full_proxy_loop(
        engine, oer_csv, "Overpotential mV @10 mA cm-2", "min",
    )
    _assert_standard_artifacts(paths, backend_paths)

    state = read_json(paths.state)
    cat_params = [p for p in state["design_parameters"] if p["type"] == "cat"]
    assert len(cat_params) >= 1, "OER dataset should have at least one categorical parameter"


def test_botorch_supports_mixed_categorical_suggestions(
    engine: BOEngine, tmp_path: Path
) -> None:
    """BoTorch should support mixed categorical + numeric suggestions."""
    dataset = tmp_path / "mixed_botorch.csv"
    pd.DataFrame(
        [
            {"cat": "A", "num": 0.0, "target": 1.0},
            {"cat": "A", "num": 0.5, "target": 0.8},
            {"cat": "B", "num": 0.2, "target": 0.9},
            {"cat": "B", "num": 0.7, "target": 0.7},
            {"cat": "C", "num": 0.4, "target": 0.6},
            {"cat": "C", "num": 0.9, "target": 0.5},
        ]
    ).to_csv(dataset, index=False)

    state = engine.init_run(
        dataset_path=dataset,
        target_column="target",
        objective="min",
        default_engine="botorch",
        num_initial_random_samples=3,
        default_batch_size=1,
        seed=42,
    )
    run_id = state["run_id"]

    for y in [1.0, 0.9, 0.8]:
        suggestion = engine.suggest(run_id)["suggestions"][0]
        engine.observe(run_id, [{"x": suggestion["x"], "y": y}])

    result = engine.suggest(run_id, batch_size=2)
    assert result["engine"] == "botorch"
    assert len(result["suggestions"]) == 2
    for suggestion in result["suggestions"]:
        assert suggestion["x"]["cat"] in {"A", "B", "C"}
        assert 0.0 <= float(suggestion["x"]["num"]) <= 0.9


def test_oer_simplex_constraint_projects_suggestions(
    engine: BOEngine, oer_csv: Path
) -> None:
    """Simplex-constrained OER suggestions should sum to the declared total."""
    state = engine.init_run(
        dataset_path=oer_csv,
        target_column="Overpotential mV @10 mA cm-2",
        objective="min",
        seed=42,
        constraints=[
            {
                "type": "simplex",
                "cols": [
                    "Metal_1_Proportion",
                    "Metal_2_Proportion",
                    "Metal_3_Proportion",
                ],
                "total": 100.0,
            }
        ],
    )
    run_id = state["run_id"]

    result = engine.suggest(run_id, batch_size=4)

    assert state["constraints"] == [
        {
            "type": "simplex",
            "cols": [
                "Metal_1_Proportion",
                "Metal_2_Proportion",
                "Metal_3_Proportion",
            ],
            "total": 100.0,
        }
    ]
    for suggestion in result["suggestions"]:
        total = (
            float(suggestion["x"]["Metal_1_Proportion"])
            + float(suggestion["x"]["Metal_2_Proportion"])
            + float(suggestion["x"]["Metal_3_Proportion"])
        )
        assert total == pytest.approx(100.0)

    status = engine.status(run_id)
    assert status["constraints"] == state["constraints"]


@pytest.mark.slow
def test_bh_feature_selection(engine: BOEngine, bh_csv: Path) -> None:
    """BH dataset, max objective, feature selection with max_features=20."""
    _, paths, backend_paths = _run_full_proxy_loop(
        engine, bh_csv, "yield", "max", max_features=20
    )
    _assert_standard_artifacts(paths, backend_paths)

    backend_meta = read_json(backend_paths.oracle_meta)
    assert len(backend_meta["active_features"]) == 20
    assert len(backend_meta["ignored_features"]) > 0


@pytest.mark.slow
def test_build_oracle_feature_selection_does_not_mutate_run_state(
    engine: BOEngine, oer_csv: Path
) -> None:
    """Direct backend building should not rewrite run-state feature metadata."""
    simplex_cols = ["Metal_1_Proportion", "Metal_2_Proportion", "Metal_3_Proportion"]
    state = engine.init_run(
        dataset_path=oer_csv,
        target_column="Overpotential mV @10 mA cm-2",
        objective="min",
        seed=42,
        constraints=[{"type": "simplex", "cols": simplex_cols, "total": 100.0}],
    )
    run_id = state["run_id"]
    paths = RunPaths(run_dir=engine.get_run_dir(run_id))
    backend_paths = EvaluationBackendPaths(
        engine.runs_root.parent / "evaluation_backends" / run_id
    )
    original_state = read_json(paths.state)

    build_proxy_oracle(
        dataset_path=oer_csv,
        target_column="Overpotential mV @10 mA cm-2",
        objective="min",
        backend_dir=backend_paths.backend_dir,
        max_features=3,
        seed=42,
    )

    updated_state = read_json(paths.state)
    assert updated_state["active_features"] == original_state["active_features"]
    assert updated_state["ignored_features"] == original_state["ignored_features"]

    backend_meta = read_json(backend_paths.oracle_meta)
    assert len(backend_meta["active_features"]) == 3

    result = engine.suggest(run_id, batch_size=2)
    for suggestion in result["suggestions"]:
        total = sum(float(suggestion["x"][c]) for c in simplex_cols)
        assert total == pytest.approx(100.0)


# ------------------------------------------------------------------
# Human-in-the-loop test
# ------------------------------------------------------------------


def test_human_loop_suggest_observe(engine: BOEngine, her_csv: Path) -> None:
    """Suggest/observe cycle without oracle (human-in-the-loop pattern)."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        seed=42,
    )
    run_id = state["run_id"]

    for _ in range(2):
        result = engine.suggest(run_id)
        suggestion = result["suggestions"][0]
        assert "x" in suggestion
        assert set(state["active_features"]).issubset(suggestion["x"].keys())

        engine.observe(run_id, [{"x": suggestion["x"], "y": 1.23}])

    paths = RunPaths(run_dir=engine.get_run_dir(run_id))
    observations = read_jsonl(paths.observations)
    assert len(observations) == 2

    final_state = read_json(paths.state)
    assert final_state["status"] == "running"


def test_search_space_init_suggest_observe_report_min(engine: BOEngine) -> None:
    """Runs initialized from explicit search-space JSON should work without a dataset."""
    state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B", "C"]},
            ],
            "fixed_features": {"pressure_bar": 1.0},
        },
        target_column="yield_pct",
        objective="min",
        seed=42,
    )
    run_id = state["run_id"]
    paths = RunPaths(run_dir=engine.get_run_dir(run_id))

    result = engine.suggest(run_id, batch_size=2)
    assert len(result["suggestions"]) == 2
    for suggestion in result["suggestions"]:
        assert 20.0 <= float(suggestion["x"]["temperature_c"]) <= 100.0
        assert suggestion["x"]["solvent"] in {"A", "B", "C"}
        assert suggestion["x"]["pressure_bar"] == 1.0

    engine.observe(
        run_id,
        [
            {"x": result["suggestions"][0]["x"], "y": 5.0},
            {"x": result["suggestions"][1]["x"], "y": 3.0},
        ],
    )
    report = engine.report(run_id)

    assert report["best_value"] == pytest.approx(3.0)
    input_spec = read_json(paths.input_spec)
    assert input_spec["input_source"] == "search_space_json"
    assert input_spec["dataset_path"] is None


def test_search_space_init_supports_max_objective(engine: BOEngine) -> None:
    """Max-objective runs initialized from search-space JSON should optimize correctly."""
    state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B"]},
            ]
        },
        target_column="yield_pct",
        objective="max",
        seed=42,
        default_engine="botorch",
        num_initial_random_samples=1,
    )
    run_id = state["run_id"]

    first = engine.suggest(run_id)["suggestions"][0]
    engine.observe(run_id, [{"x": first["x"], "y": 1.0}])

    second_batch = engine.suggest(run_id, batch_size=2)
    engine.observe(
        run_id,
        [
            {"x": second_batch["suggestions"][0]["x"], "y": 5.0},
            {"x": second_batch["suggestions"][1]["x"], "y": 3.0},
        ],
    )

    status = engine.status(run_id)
    report = engine.report(run_id)
    assert status["best_value"] == pytest.approx(5.0)
    assert report["best_value"] == pytest.approx(5.0)


def test_build_proxy_oracle_requires_existing_dataset(tmp_path: Path) -> None:
    """Direct backend build should fail clearly for a missing dataset path."""
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        build_proxy_oracle(
            dataset_path=tmp_path / "missing.csv",
            target_column="yield_pct",
            objective="max",
            backend_dir=tmp_path / "evaluation_backends" / "missing",
        )


def test_build_proxy_oracle_requires_target_column(tmp_path: Path) -> None:
    """Direct backend build should fail clearly when the target column is missing."""
    dataset = tmp_path / "missing_target.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A"},
            {"temperature_c": 40.0, "solvent": "B"},
            {"temperature_c": 60.0, "solvent": "A"},
            {"temperature_c": 80.0, "solvent": "B"},
            {"temperature_c": 100.0, "solvent": "A"},
        ]
    ).to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="Target column 'yield_pct' is not in dataset columns"):
        build_proxy_oracle(
            dataset_path=dataset,
            target_column="yield_pct",
            objective="max",
            backend_dir=tmp_path / "evaluation_backends" / "missing-target",
        )


def test_build_proxy_oracle_requires_enough_labeled_rows(tmp_path: Path) -> None:
    """Direct backend build needs at least five labeled rows after dropping NaNs."""
    dataset = tmp_path / "too_few_labels.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "yield_pct": None},
            {"temperature_c": 60.0, "solvent": "B", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": None},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    with pytest.raises(ValueError, match="Need at least 5 non-null target rows"):
        build_proxy_oracle(
            dataset_path=dataset,
            target_column="yield_pct",
            objective="max",
            backend_dir=tmp_path / "evaluation_backends" / "too-few",
        )


def test_build_proxy_oracle_drop_cols_excludes_requested_columns(tmp_path: Path) -> None:
    """Dropped columns should not appear in the trained backend feature set."""
    dataset = tmp_path / "drop_cols_dataset.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "batch_id": "b1", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "batch_id": "b2", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "B", "batch_id": "b3", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "batch_id": "b4", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "batch_id": "b5", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)
    backend_dir = tmp_path / "evaluation_backends" / "drop-cols"

    payload = build_proxy_oracle(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        backend_dir=backend_dir,
        drop_cols=["batch_id"],
    )

    assert "batch_id" not in payload["active_features"]
    assert "batch_id" in payload["ignored_features"]


def test_build_proxy_oracle_does_not_create_or_mutate_runs(
    engine: BOEngine, tmp_path: Path
) -> None:
    """Building a backend directly should not create or touch BO run state."""
    dataset = tmp_path / "standalone_backend_dataset.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "B", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    build_proxy_oracle(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        backend_dir=engine.runs_root.parent / "evaluation_backends" / "standalone",
    )

    assert list(engine.runs_root.glob("*/state.json")) == []


def test_hidden_oracle_evaluator_runs_search_space_loop(engine: BOEngine, tmp_path: Path) -> None:
    """Hidden evaluator should drive a search-space-only run from an external oracle dir."""
    dataset = tmp_path / "evaluator_dataset.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "B", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    backend_paths = EvaluationBackendPaths(
        engine.runs_root.parent / "evaluation_backends" / "evaluator-hidden"
    )
    build_proxy_oracle(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        backend_dir=backend_paths.backend_dir,
        seed=42,
    )

    search_state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B"]},
            ]
        },
        target_column="yield_pct",
        objective="max",
        seed=7,
    )
    run_id = search_state["run_id"]
    paths = RunPaths(run_dir=engine.get_run_dir(run_id))

    payload = run_hidden_oracle_evaluator(
        engine,
        run_id=run_id,
        backend_dir=backend_paths.backend_dir,
        num_iterations=2,
        batch_size=2,
    )

    assert payload["recorded"] == 4
    assert paths.observations.exists()
    observations = read_jsonl(paths.observations)
    assert len(observations) == 4
    assert {row["source"] for row in observations} == {"benchmark-evaluator"}

    report = read_json(paths.report)
    assert report["observation_sources"] == ["benchmark-evaluator"]
    assert report["best_value"] is not None


def test_hidden_oracle_evaluator_resolves_pending_suggestions(
    engine: BOEngine, tmp_path: Path
) -> None:
    """Evaluator resume should observe already-pending suggestions before new rounds."""
    dataset = tmp_path / "resume_dataset.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "B", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    backend_paths = EvaluationBackendPaths(
        engine.runs_root.parent / "evaluation_backends" / "resume-hidden"
    )
    build_proxy_oracle(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        backend_dir=backend_paths.backend_dir,
        seed=42,
    )

    search_state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B"]},
            ]
        },
        target_column="yield_pct",
        objective="max",
        seed=7,
    )
    run_id = search_state["run_id"]
    paths = RunPaths(run_dir=engine.get_run_dir(run_id))

    engine.suggest(run_id, batch_size=1)

    payload = run_hidden_oracle_evaluator(
        engine,
        run_id=run_id,
        backend_dir=backend_paths.backend_dir,
        num_iterations=0,
        batch_size=1,
    )

    assert payload["resolved_pending"] == 1
    assert payload["recorded"] == 1

    suggestions = read_jsonl(paths.suggestions)
    observations = read_jsonl(paths.observations)
    assert len(suggestions) == 1
    assert len(observations) == 1
    assert observations[0]["suggestion_id"] == suggestions[0]["suggestion_id"]


def test_hidden_oracle_evaluator_accepts_backend_features_satisfied_by_fixed_features(
    engine: BOEngine, tmp_path: Path
) -> None:
    """Backend validation should treat fixed_features as satisfying oracle-required inputs."""
    dataset = tmp_path / "fixed_feature_backend.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "B", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "A", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    backend_paths = EvaluationBackendPaths(
        engine.runs_root.parent / "evaluation_backends" / "fixed-feature-hidden"
    )
    build_proxy_oracle(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        backend_dir=backend_paths.backend_dir,
        seed=42,
    )

    search_state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
            ],
            "fixed_features": {"solvent": "A"},
        },
        target_column="yield_pct",
        objective="max",
        seed=7,
    )

    payload = run_hidden_oracle_evaluator(
        engine,
        run_id=search_state["run_id"],
        backend_dir=backend_paths.backend_dir,
        num_iterations=1,
        batch_size=1,
    )

    assert payload["recorded"] == 1


def test_build_oracle_supports_custom_backend_id(
    engine: BOEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI build-oracle should write to the requested backend id."""
    dataset = tmp_path / "custom_backend_dataset.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "B", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    backend_id = "shared-yield-backend"
    backends_root = engine.runs_root.parent / "evaluation_backends"

    exit_code = handle_evaluation(
        argparse.Namespace(
            command="build-oracle",
            dataset=dataset,
            target="yield_pct",
            objective="max",
            backend_id=backend_id,
            drop_cols=None,
            seed=7,
            engine="hebo",
            backends_root=backends_root,
            cv_folds=3,
            max_features=None,
            verbose=False,
        ),
        engine,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    backend_paths = EvaluationBackendPaths(backends_root / backend_id)
    backend_meta = read_json(backend_paths.oracle_meta)
    assert payload["backend_id"] == backend_id
    assert payload["backend_dir"] == str(backend_paths.backend_dir)
    assert backend_paths.oracle_model.exists()
    assert backend_paths.oracle_meta.exists()
    assert "dataset_path" not in backend_meta
    assert list(engine.runs_root.glob("*/state.json")) == []


def test_run_proxy_reuses_backend_across_runs(
    engine: BOEngine, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI run-proxy should accept a backend id built independently of the target run."""
    dataset = tmp_path / "shared_backend_dataset.csv"
    pd.DataFrame(
        [
            {"temperature_c": 20.0, "solvent": "A", "yield_pct": 1.0},
            {"temperature_c": 40.0, "solvent": "A", "yield_pct": 2.5},
            {"temperature_c": 60.0, "solvent": "B", "yield_pct": 4.0},
            {"temperature_c": 80.0, "solvent": "B", "yield_pct": 5.0},
            {"temperature_c": 100.0, "solvent": "A", "yield_pct": 3.5},
        ]
    ).to_csv(dataset, index=False)

    backend_id = "shared-yield-backend"
    backends_root = engine.runs_root.parent / "evaluation_backends"
    source_backend_paths = EvaluationBackendPaths(backends_root / backend_id)
    build_proxy_oracle(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        backend_dir=source_backend_paths.backend_dir,
        seed=42,
    )

    target_state = engine.init_run(
        search_space_spec={
            "design_parameters": [
                {"name": "temperature_c", "type": "num", "lb": 20.0, "ub": 100.0},
                {"name": "solvent", "type": "cat", "categories": ["A", "B"]},
            ]
        },
        target_column="yield_pct",
        objective="max",
        seed=7,
    )
    target_run_id = target_state["run_id"]
    target_paths = RunPaths(run_dir=engine.get_run_dir(target_run_id))

    exit_code = handle_evaluation(
        argparse.Namespace(
            command="run-proxy",
            run_id=target_run_id,
            backend_id=backend_id,
            backends_root=backends_root,
            iterations=2,
            batch_size=1,
            seed_pool=None,
            verbose=False,
        ),
        engine,
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == target_run_id
    assert source_backend_paths.oracle_model.exists()
    assert source_backend_paths.oracle_meta.exists()

    observations = read_jsonl(target_paths.observations)
    assert len(observations) == 2
    assert {row["source"] for row in observations} == {"proxy-oracle"}

    target_state_after = read_json(target_paths.state)
    assert target_state_after["status"] == "completed"
    assert target_state_after["oracle"]["selected_model"] is not None


def test_run_proxy_rejects_backend_missing_simplex_columns(
    engine: BOEngine, tmp_path: Path
) -> None:
    """Reduced backends must still include every simplex-constrained column."""
    dataset = tmp_path / "simplex_backend_dataset.csv"
    pd.DataFrame(
        [
            {"a": 0.7, "b": 0.2, "c": 0.1, "temperature_c": 20.0, "yield_pct": 1.0},
            {"a": 0.5, "b": 0.3, "c": 0.2, "temperature_c": 40.0, "yield_pct": 2.0},
            {"a": 0.4, "b": 0.4, "c": 0.2, "temperature_c": 60.0, "yield_pct": 3.0},
            {"a": 0.3, "b": 0.5, "c": 0.2, "temperature_c": 80.0, "yield_pct": 4.0},
            {"a": 0.2, "b": 0.5, "c": 0.3, "temperature_c": 100.0, "yield_pct": 5.0},
        ]
    ).to_csv(dataset, index=False)

    backend_id = "reduced-simplex-backend"
    backends_root = engine.runs_root.parent / "evaluation_backends"
    backend_paths = EvaluationBackendPaths(backends_root / backend_id)
    build_proxy_oracle(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        backend_dir=backend_paths.backend_dir,
        max_features=1,
        seed=42,
    )

    target_state = engine.init_run(
        dataset_path=dataset,
        target_column="yield_pct",
        objective="max",
        seed=7,
        constraints=[
            {"type": "simplex", "cols": ["a", "b", "c"], "total": 1.0},
        ],
    )

    with pytest.raises(ValueError, match="simplex constraints"):
        handle_evaluation(
            argparse.Namespace(
                command="run-proxy",
                run_id=target_state["run_id"],
                backend_id=backend_id,
                backends_root=backends_root,
                iterations=1,
                batch_size=1,
                seed_pool=None,
                verbose=False,
            ),
            engine,
        )


def test_init_hebo_rf_persists_in_status(engine: BOEngine, her_csv: Path) -> None:
    """HEBO surrogate selection should persist in state and status."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        default_engine="hebo",
        hebo_model="rf",
        seed=42,
    )

    assert state["hebo_model"] == "rf"

    status = engine.status(state["run_id"])
    assert status["default_engine"] == "hebo"
    assert status["hebo_model"] == "rf"


# ------------------------------------------------------------------
# Negative / error-path tests
# ------------------------------------------------------------------


def test_proxy_observer_missing_oracle(engine: BOEngine, her_csv: Path) -> None:
    """ProxyObserver raises FileNotFoundError when oracle hasn't been built."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
    )
    backend_dir = engine.runs_root.parent / "evaluation_backends" / state["run_id"]

    with pytest.raises(FileNotFoundError, match="build-oracle"):
        ProxyObserver(backend_dir)


def test_observe_missing_y_raises(engine: BOEngine, her_csv: Path) -> None:
    """Observing without a 'y' value raises ValueError."""
    state = engine.init_run(
        dataset_path=her_csv,
        target_column="Target",
        objective="max",
        seed=42,
    )
    run_id = state["run_id"]
    result = engine.suggest(run_id)
    suggestion = result["suggestions"][0]

    with pytest.raises(ValueError, match="[Mm]issing objective value"):
        engine.observe(run_id, [{"x": suggestion["x"]}])


def test_init_invalid_target_column(engine: BOEngine, her_csv: Path) -> None:
    """init_run with a nonexistent target column raises ValueError."""
    with pytest.raises(ValueError, match="not in dataset columns"):
        engine.init_run(
            dataset_path=her_csv,
            target_column="nonexistent_column",
            objective="max",
        )


def test_init_non_hebo_engine_rejects_hebo_model(
    engine: BOEngine, her_csv: Path
) -> None:
    """Non-HEBO engines should reject HEBO surrogate configuration."""
    with pytest.raises(ValueError, match="only supported when --engine hebo"):
        engine.init_run(
            dataset_path=her_csv,
            target_column="Target",
            objective="max",
            default_engine="random",
            hebo_model="rf",
        )


def test_init_simplex_constraint_unknown_feature_raises(
    engine: BOEngine, her_csv: Path
) -> None:
    """Simplex constraints must reference active features."""
    with pytest.raises(ValueError, match="not in active features"):
        engine.init_run(
            dataset_path=her_csv,
            target_column="Target",
            objective="max",
            constraints=[
                {
                    "type": "simplex",
                    "cols": ["unknown_a", "unknown_b"],
                    "total": 1.0,
                }
            ],
        )
