"""Core BO engine.

This module keeps optimization state on disk (`runs/<run_id>/`) and rebuilds
optimizers from logged observations when needed. That replay-first design keeps
the workflow resumable and robust for human-in-the-loop usage.
"""

from pathlib import Path
import secrets
import sys
from typing import Any

from .observers.base import Observer

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.bo import BO
from hebo.optimizers.hebo import HEBO
import numpy as np
import pandas as pd
from tqdm import tqdm

from .plotting import plot_optimization_convergence
from .utils import (
    Objective,
    OptimizerName,
    RunPaths,
    append_jsonl,
    generate_run_id,
    read_json,
    read_jsonl,
    row_to_python_dict,
    to_python_scalar,
    utc_now_iso,
    write_json,
)


def _detect_smiles_column(df: pd.DataFrame, hint: str | None = None) -> str:
    """Auto-detect which column in *df* contains SMILES strings.

    Strategy:
    1. If *hint* is given and exists in *df*, validate it.
    2. Otherwise, check columns whose names look like SMILES
       (``smiles``, ``smi``, ``molecule``, ``mol``, ``structure``).
    3. Fallback: try all string/object columns.

    Validation: sample up to 10 non-null values and parse with RDKit.
    A column is accepted if ≥70 % parse successfully.

    Raises ``ValueError`` if no SMILES column can be identified.
    """
    from rdkit import Chem

    def _smiles_hit_rate(series: pd.Series) -> float:
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return 0.0
        hits = sum(1 for v in sample if Chem.MolFromSmiles(str(v)) is not None)
        return hits / len(sample)

    # Explicit hint
    if hint is not None:
        if hint not in df.columns:
            raise ValueError(
                f"Specified SMILES column '{hint}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        rate = _smiles_hit_rate(df[hint])
        if rate >= 0.5:
            return hint
        raise ValueError(
            f"Column '{hint}' does not appear to contain SMILES "
            f"(only {rate:.0%} parsed successfully)."
        )

    # Name-based heuristic (priority order)
    _SMILES_NAMES = {"smiles", "smi", "molecule", "mol", "structure", "canonical_smiles"}
    candidates: list[tuple[str, float]] = []

    for col in df.columns:
        if col.lower().strip() in _SMILES_NAMES:
            rate = _smiles_hit_rate(df[col])
            if rate >= 0.7:
                return col  # early return on strong name match
            candidates.append((col, rate))

    # Fallback: check all object/string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col.lower().strip() in _SMILES_NAMES:
            continue  # already checked
        rate = _smiles_hit_rate(df[col])
        if rate >= 0.7:
            candidates.append((col, rate))

    if candidates:
        best = max(candidates, key=lambda x: x[1])
        if best[1] >= 0.5:
            return best[0]

    raise ValueError(
        "Cannot auto-detect a SMILES column. Please specify --smiles-column. "
        f"Columns checked: {list(df.columns)}"
    )


def _detect_smiles_column(df: pd.DataFrame, hint: str | None = None) -> str:
    """Auto-detect which column in *df* contains SMILES strings.

    Strategy:
    1. If *hint* is given and exists in *df*, validate it.
    2. Otherwise, check columns whose names look like SMILES
       (``smiles``, ``smi``, ``molecule``, ``mol``, ``structure``).
    3. Fallback: try all string/object columns.

    Validation: sample up to 10 non-null values and parse with RDKit.
    A column is accepted if ≥70 % parse successfully.

    Raises ``ValueError`` if no SMILES column can be identified.
    """
    from rdkit import Chem

    def _smiles_hit_rate(series: pd.Series) -> float:
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return 0.0
        hits = sum(1 for v in sample if Chem.MolFromSmiles(str(v)) is not None)
        return hits / len(sample)

    # Explicit hint
    if hint is not None:
        if hint not in df.columns:
            raise ValueError(
                f"Specified SMILES column '{hint}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        rate = _smiles_hit_rate(df[hint])
        if rate >= 0.5:
            return hint
        raise ValueError(
            f"Column '{hint}' does not appear to contain SMILES "
            f"(only {rate:.0%} parsed successfully)."
        )

    # Name-based heuristic (priority order)
    _SMILES_NAMES = {"smiles", "smi", "molecule", "mol", "structure", "canonical_smiles"}
    candidates: list[tuple[str, float]] = []

    for col in df.columns:
        if col.lower().strip() in _SMILES_NAMES:
            rate = _smiles_hit_rate(df[col])
            if rate >= 0.7:
                return col  # early return on strong name match
            candidates.append((col, rate))

    # Fallback: check all object/string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col.lower().strip() in _SMILES_NAMES:
            continue  # already checked
        rate = _smiles_hit_rate(df[col])
        if rate >= 0.7:
            candidates.append((col, rate))

    if candidates:
        best = max(candidates, key=lambda x: x[1])
        if best[1] >= 0.5:
            return best[0]

    raise ValueError(
        "Cannot auto-detect a SMILES column. Please specify --smiles-column. "
        f"Columns checked: {list(df.columns)}"
    )


def _infer_design_parameters(
    frame: pd.DataFrame,
    *,
    max_categories: int = 64,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    """Infer HEBO design parameters from a feature frame.

    Returns a tuple of:
    - optimizable parameters
    - fixed features (constant columns)
    - dropped features (empty/unusable columns)
    """
    params: list[dict[str, Any]] = []
    fixed_features: dict[str, Any] = {}
    dropped_features: list[str] = []

    for col in frame.columns:
        series = frame[col]

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if numeric.empty:
                dropped_features.append(col)
                continue
            lb = float(numeric.min())
            ub = float(numeric.max())
            if np.isclose(lb, ub):
                fixed_features[col] = lb
                continue
            params.append({"name": col, "type": "num", "lb": lb, "ub": ub})
            continue

        categories = sorted({str(v) for v in series.dropna().tolist()})
        if not categories:
            dropped_features.append(col)
            continue
        if len(categories) == 1:
            fixed_features[col] = categories[0]
            continue
        if len(categories) > max_categories:
            raise ValueError(
                f"Feature '{col}' has {len(categories)} categories; max supported is {max_categories}."
            )
        params.append({"name": col, "type": "cat", "categories": categories})

    if not params:
        raise ValueError("No optimizable features were inferred from the dataset.")

    return params, fixed_features, dropped_features


class BOEngine:
    """Dataset-driven Bayesian optimization engine with persisted run state."""

    def __init__(self, runs_root: str | Path = "runs") -> None:
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self, run_id: str) -> Path:
        """Return the run directory for *run_id*."""
        return self.runs_root / run_id

    def _paths(self, run_id: str) -> RunPaths:
        return RunPaths(run_dir=self.runs_root / run_id)

    def _load_state(self, run_id: str) -> dict[str, Any]:
        paths = self._paths(run_id)
        if not paths.state.exists():
            raise FileNotFoundError(f"Run '{run_id}' not found at {paths.run_dir}")
        return read_json(paths.state)

    def _save_state(self, run_id: str, state: dict[str, Any]) -> None:
        write_json(self._paths(run_id).state, state)

    def _log(self, verbose: bool, message: str) -> None:
        if verbose:
            print(message, file=sys.stderr)

    def init_run(
        self,
        *,
        dataset_path: str | Path,
        target_column: str,
        objective: Objective,
        default_engine: OptimizerName = "hebo",
        run_id: str | None = None,
        num_initial_random_samples: int = 10,
        default_batch_size: int = 1,
        seed: int = 7,
        max_categories: int = 64,
        intent: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Initialize a run from a dataset and inferred design space."""
        if objective not in {"min", "max"}:
            raise ValueError("objective must be either 'min' or 'max'")
        if default_engine not in {"hebo", "bo_lcb", "random"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random")

        dataset_path = Path(dataset_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path)
        self._log(verbose, f"[init] dataset={dataset_path} rows={len(data)}")
        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' is not in dataset columns: {list(data.columns)}"
            )

        feature_frame = data.drop(columns=[target_column])
        design_params, fixed_features, dropped_features = _infer_design_parameters(
            feature_frame,
            max_categories=max_categories,
        )

        if run_id is None:
            for _ in range(20):
                candidate = generate_run_id()
                if not self._paths(candidate).state.exists():
                    run_id = candidate
                    break
            if run_id is None:
                raise RuntimeError("Failed to generate a unique run_id after retries.")
        elif self._paths(run_id).state.exists():
            raise ValueError(
                f"Run '{run_id}' already exists. Provide a different --run-id or omit it."
            )

        numeric_target = pd.to_numeric(data[target_column], errors="coerce")
        if objective == "max":
            if numeric_target.notna().sum() == 0:
                raise ValueError(
                    "Target column has no numeric values required for max objective transform."
                )
            target_max_for_restore = float(numeric_target.max())
        else:
            target_max_for_restore = float("nan")

        state = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "initialized",
            "dataset_path": str(dataset_path),
            "target_column": target_column,
            "objective": objective,
            "default_engine": default_engine,
            "seed": int(seed),
            "num_initial_random_samples": int(num_initial_random_samples),
            "default_batch_size": int(default_batch_size),
            "design_parameters": design_params,
            "active_features": [p["name"] for p in design_params],
            "fixed_features": fixed_features,
            "dropped_features": dropped_features,
            "ignored_features": [],
            "objective_transform": {
                "internal_objective": "min",
                "target_max_for_restore": target_max_for_restore,
            },
        }
        self._save_state(run_id, state)
        self._log(
            verbose,
            f"[init] run_id={run_id} engine={default_engine} features={len(state['active_features'])}",
        )
        if intent is not None:
            intent_payload = {
                "run_id": run_id,
                "created_at": utc_now_iso(),
                "intent": intent,
                "resolved": {
                    "dataset_path": str(dataset_path),
                    "target_column": target_column,
                    "objective": objective,
                    "seed": int(seed),
                    "num_initial_random_samples": int(num_initial_random_samples),
                    "default_batch_size": int(default_batch_size),
                    "max_categories": int(max_categories),
                },
            }
            write_json(self._paths(run_id).intent, intent_payload)
        return state

    def init_molecular_run(
        self,
        *,
        scaffold_spec_path: str | Path,
        target_column: str,
        objective: Objective,
        dataset_path: str | Path | None = None,
        default_engine: OptimizerName = "hebo",
        run_id: str | None = None,
        num_initial_random_samples: int = 10,
        default_batch_size: int = 1,
        seed: int = 7,
        intent: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Initialize a molecular optimization run from a scaffold spec.

        The scaffold spec defines a molecular scaffold with variable
        positions, each having a library of allowed substituents.  HEBO
        optimizes over the substituent choices (categorical) plus any
        reaction-condition parameters (numeric/categorical).

        The oracle sees an expanded feature set: the HEBO design parameters
        plus computed molecular descriptors (fingerprints, electronic/steric
        properties) derived from the assembled molecule.
        """
        from .molecular.scaffold import load_scaffold_spec, spec_to_design_parameters
        from .molecular.features import get_descriptor_feature_names

        if objective not in {"min", "max"}:
            raise ValueError("objective must be either 'min' or 'max'")
        if default_engine not in {"hebo", "bo_lcb", "random"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random")

        scaffold_spec_path = Path(scaffold_spec_path).resolve()
        if not scaffold_spec_path.exists():
            raise FileNotFoundError(f"Scaffold spec not found: {scaffold_spec_path}")

        spec = load_scaffold_spec(scaffold_spec_path)
        design_params = spec_to_design_parameters(spec)
        descriptor_feature_names = get_descriptor_feature_names(spec)

        self._log(
            verbose,
            f"[init-molecular] scaffold={spec.scaffold.smiles} "
            f"positions={spec.scaffold.variable_positions} "
            f"design_params={len(design_params)} "
            f"descriptor_features={len(descriptor_feature_names)}",
        )

        # Generate run_id
        if run_id is None:
            for _ in range(20):
                candidate = generate_run_id()
                if not self._paths(candidate).state.exists():
                    run_id = candidate
                    break
            if run_id is None:
                raise RuntimeError("Failed to generate a unique run_id after retries.")
        elif self._paths(run_id).state.exists():
            raise ValueError(
                f"Run '{run_id}' already exists. Provide a different --run-id or omit it."
            )

        # If dataset provided, read target max for objective transform
        target_max_for_restore = float("nan")
        dataset_path_resolved: str | None = None
        if dataset_path is not None:
            dataset_path_obj = Path(dataset_path).resolve()
            if not dataset_path_obj.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path_obj}")
            dataset_path_resolved = str(dataset_path_obj)
            data = pd.read_csv(dataset_path_obj)
            if target_column not in data.columns:
                raise ValueError(
                    f"Target column '{target_column}' not in dataset: {list(data.columns)}"
                )
            numeric_target = pd.to_numeric(data[target_column], errors="coerce")
            if objective == "max" and numeric_target.notna().sum() > 0:
                target_max_for_restore = float(numeric_target.max())
        elif objective == "max":
            # No dataset: use a default high value; will be updated on first observe
            target_max_for_restore = 100.0

        # Active features for HEBO are the design params only
        # (descriptors are used by the oracle but not by the optimizer)
        active_features = [p["name"] for p in design_params]

        # Serialize descriptor config and feasibility config for state
        dc = spec.descriptor_config
        fc = spec.feasibility

        state: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "initialized",
            "mode": "molecular",
            "dataset_path": dataset_path_resolved,
            "target_column": target_column,
            "objective": objective,
            "default_engine": default_engine,
            "seed": int(seed),
            "num_initial_random_samples": int(num_initial_random_samples),
            "default_batch_size": int(default_batch_size),
            "design_parameters": design_params,
            "active_features": active_features,
            "fixed_features": {},
            "dropped_features": [],
            "ignored_features": [],
            "oracle": None,
            "objective_transform": {
                "internal_objective": "min",
                "target_max_for_restore": target_max_for_restore,
            },
            "molecular": {
                "scaffold_spec_path": str(scaffold_spec_path),
                "scaffold_smiles": spec.scaffold.smiles,
                "variable_positions": list(spec.scaffold.variable_positions),
                "descriptor_feature_names": descriptor_feature_names,
                "descriptor_config": {
                    "basic": dc.basic,
                    "fingerprint": {
                        "enabled": dc.fingerprint_enabled,
                        "n_bits": dc.fingerprint_n_bits,
                        "radius": dc.fingerprint_radius,
                    },
                    "electronic": dc.electronic,
                    "steric": dc.steric,
                    "dft": {
                        "enabled": dc.dft_enabled,
                        "data_path": dc.dft_data_path,
                    },
                    "energy_backend": dc.energy_backend,
                },
                "feasibility": {
                    "mode": fc.mode,
                    "sa_threshold": fc.sa_threshold,
                    "incompatible_pairs": fc.incompatible_pairs,
                },
            },
        }

        # Save state and scaffold spec copy
        paths = self._paths(run_id)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_state(run_id, state)

        # Copy scaffold spec into run directory for reproducibility
        import shutil
        shutil.copy2(scaffold_spec_path, paths.scaffold_spec)

        self._log(
            verbose,
            f"[init-molecular] run_id={run_id} engine={default_engine} "
            f"features={len(active_features)} descriptors={len(descriptor_feature_names)}",
        )

        if intent is not None:
            intent_payload = {
                "run_id": run_id,
                "created_at": utc_now_iso(),
                "intent": intent,
                "resolved": {
                    "scaffold_spec_path": str(scaffold_spec_path),
                    "dataset_path": dataset_path_resolved,
                    "target_column": target_column,
                    "objective": objective,
                    "seed": int(seed),
                },
            }
            write_json(paths.intent, intent_payload)

        return state

    # ------------------------------------------------------------------
    # SMILES-direct (virtual screening) mode
    # ------------------------------------------------------------------

    def init_smiles_run(
        self,
        *,
        dataset_path: str | Path,
        target_column: str,
        objective: Objective,
        smiles_column: str | None = None,
        default_engine: OptimizerName = "hebo",
        run_id: str | None = None,
        num_initial_random_samples: int = 10,
        default_batch_size: int = 1,
        seed: int = 7,
        fingerprint_bits: int = 128,
        energy_backend: str = "none",
        discovery: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Initialize a virtual-screening run from a CSV with SMILES + target.

        This is the simplest entry point: the user provides a CSV where each
        row is a molecule (SMILES) plus an experimental measurement.  The
        system auto-detects the SMILES column, computes molecular descriptors
        for each unique molecule, and sets up HEBO to search over the
        categorical SMILES choices.

        When *discovery=True*, HEBO optimizes in continuous descriptor space
        instead of categorical SMILES.  Each suggestion is mapped back to
        the nearest molecule in the dataset via descriptor-space distance.
        This enables the optimizer to learn structure-activity patterns and
        discover high-performing regions even when no high-activity examples
        exist in the training data.

        No scaffold spec or substituent library is needed.
        """
        if objective not in {"min", "max"}:
            raise ValueError("objective must be either 'min' or 'max'")
        if default_engine not in {"hebo", "bo_lcb", "random"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random")

        dataset_path = Path(dataset_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path)
        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' not in dataset: {list(data.columns)}"
            )

        # Detect SMILES column
        smi_col = _detect_smiles_column(data, hint=smiles_column)
        self._log(verbose, f"[screen] SMILES column detected: '{smi_col}'")

        # Infer design parameters from all non-target columns
        feature_frame = data.drop(columns=[target_column])
        design_params, fixed_features, dropped_features = _infer_design_parameters(
            feature_frame, max_categories=2048,  # higher limit for SMILES
        )

        # Generate run_id
        if run_id is None:
            for _ in range(20):
                candidate = generate_run_id()
                if not self._paths(candidate).state.exists():
                    run_id = candidate
                    break
            if run_id is None:
                raise RuntimeError("Failed to generate a unique run_id after retries.")
        elif self._paths(run_id).state.exists():
            raise ValueError(f"Run '{run_id}' already exists.")

        # Compute descriptor feature names from one sample SMILES
        from .molecular.features import compute_descriptors
        from .molecular.types import DescriptorConfig

        desc_config = DescriptorConfig(
            basic=True,
            fingerprint_enabled=True,
            fingerprint_n_bits=fingerprint_bits,
            fingerprint_radius=2,
            electronic=True,
            steric=False,  # No substituent context available
            dft_enabled=energy_backend != "none",
            energy_backend=energy_backend,
        )

        sample_smiles = data[smi_col].dropna().iloc[0]
        sample_desc = compute_descriptors(str(sample_smiles), config=desc_config)
        descriptor_feature_names = sorted(sample_desc.keys())

        # Add energy feature names if backend is active
        if energy_backend != "none":
            energy_names = [
                "dft_total_energy_hartree",
                "dft_homo_ev",
                "dft_lumo_ev",
                "dft_gap_ev",
                "dft_dipole_debye",
                "dft_strain_energy_kcal",
                "dft_delta_energy",
            ]
            for n in energy_names:
                if n not in descriptor_feature_names:
                    descriptor_feature_names.append(n)
            descriptor_feature_names = sorted(descriptor_feature_names)

        # ----- Discovery mode: compute descriptors for ALL molecules ----------
        # In discovery mode HEBO searches a continuous descriptor space and
        # each suggestion is mapped back to the nearest known molecule.
        descriptor_lookup: dict[str, dict[str, float]] = {}
        if discovery:
            unique_smiles = data[smi_col].dropna().unique()
            for smi in unique_smiles:
                try:
                    desc = compute_descriptors(str(smi), config=desc_config)
                    descriptor_lookup[str(smi)] = desc
                except Exception:
                    pass  # skip unparseable
            self._log(
                verbose,
                f"[screen] Discovery mode: computed descriptors for "
                f"{len(descriptor_lookup)}/{len(unique_smiles)} molecules",
            )
            if len(descriptor_lookup) < 2:
                raise ValueError(
                    "Discovery mode requires at least 2 parseable molecules."
                )

            # Build descriptor matrix and select top features by importance
            desc_df = pd.DataFrame.from_dict(descriptor_lookup, orient="index")

            # Align each data row with its descriptor vector to run
            # feature importance against the target.
            y_for_importance = pd.to_numeric(
                data[target_column], errors="coerce"
            )
            smi_series = data[smi_col].astype(str)
            valid_rows = []
            y_vals = []
            for idx, (smi, yv) in enumerate(zip(smi_series, y_for_importance)):
                if pd.notna(yv) and smi in descriptor_lookup:
                    valid_rows.append(descriptor_lookup[smi])
                    y_vals.append(float(yv))
            if len(valid_rows) < 5:
                raise ValueError("Discovery mode needs ≥5 rows with valid SMILES+target.")

            x_imp = pd.DataFrame(valid_rows)
            y_imp = np.array(y_vals)

            # Keep only variable descriptors
            variable_cols = []
            for col in x_imp.columns:
                vals = x_imp[col].dropna()
                if len(vals) > 0 and not np.isclose(vals.min(), vals.max()):
                    variable_cols.append(col)
            x_imp = x_imp[variable_cols].fillna(0.0)

            # Feature importance via RandomForest
            # 15 dims is the HEBO GP stability ceiling — 20+ causes jitter
            # collapse with only ~50 training points.  Instead of increasing
            # dims, we use 256-bit fingerprints to give the RF feature selector
            # a richer pool to choose from, producing *better* 15 features.
            max_discovery_dims = min(15, len(variable_cols))
            if len(variable_cols) > max_discovery_dims:
                rf = RandomForestRegressor(
                    n_estimators=200, random_state=seed, n_jobs=1
                )
                rf.fit(x_imp, y_imp)
                importances = rf.feature_importances_
                top_idx = np.argsort(importances)[::-1][:max_discovery_dims]
                selected_cols = [variable_cols[i] for i in top_idx]
                self._log(
                    verbose,
                    f"[screen] Discovery: selected top-{max_discovery_dims} descriptors "
                    f"from {len(variable_cols)} by importance",
                )
            else:
                selected_cols = variable_cols

            # Build continuous design parameters from selected descriptors
            discovery_design_params: list[dict[str, Any]] = []
            for col in selected_cols:
                vals = desc_df[col].dropna() if col in desc_df.columns else pd.Series(dtype=float)
                if vals.empty:
                    continue
                lb = float(vals.min())
                ub = float(vals.max())
                if np.isclose(lb, ub):
                    continue
                discovery_design_params.append(
                    {"name": col, "type": "num", "lb": lb, "ub": ub}
                )
            if not discovery_design_params:
                raise ValueError(
                    "No variable descriptors found. Discovery mode requires "
                    "at least some descriptor variation across molecules."
                )
            # Override design_params for HEBO
            design_params = discovery_design_params
            fixed_features = {}
            dropped_features = []
            self._log(
                verbose,
                f"[screen] Discovery: {len(design_params)} continuous descriptor dimensions",
            )

        self._log(
            verbose,
            f"[screen] {len(data)} rows, {data[smi_col].nunique()} unique SMILES, "
            f"{len(design_params)} design params, {len(descriptor_feature_names)} descriptors",
        )

        # Objective transform
        numeric_target = pd.to_numeric(data[target_column], errors="coerce")
        target_max_for_restore = float("nan")
        if objective == "max" and numeric_target.notna().sum() > 0:
            target_max_for_restore = float(numeric_target.max())

        state: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "initialized",
            "mode": "smiles_direct",
            "dataset_path": str(dataset_path),
            "target_column": target_column,
            "objective": objective,
            "default_engine": default_engine,
            "seed": int(seed),
            "num_initial_random_samples": int(num_initial_random_samples),
            "default_batch_size": int(default_batch_size),
            "design_parameters": design_params,
            "active_features": [p["name"] for p in design_params],
            "fixed_features": fixed_features,
            "dropped_features": dropped_features,
            "ignored_features": [],
            "oracle": None,
            "objective_transform": {
                "internal_objective": "min",
                "target_max_for_restore": target_max_for_restore,
            },
            "smiles_direct": {
                "smiles_column": smi_col,
                "discovery": discovery,
                "descriptor_feature_names": descriptor_feature_names,
                "descriptor_config": {
                    "basic": desc_config.basic,
                    "fingerprint": {
                        "enabled": desc_config.fingerprint_enabled,
                        "n_bits": desc_config.fingerprint_n_bits,
                        "radius": desc_config.fingerprint_radius,
                    },
                    "electronic": desc_config.electronic,
                    "steric": desc_config.steric,
                    "energy_backend": desc_config.energy_backend,
                },
            },
        }

        # Save discovery lookup table alongside state
        if discovery:
            state["smiles_direct"]["descriptor_lookup_file"] = "descriptor_lookup.json"

        paths = self._paths(run_id)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_state(run_id, state)

        if discovery:
            lookup_path = paths.run_dir / "descriptor_lookup.json"
            write_json(lookup_path, descriptor_lookup)

        self._log(
            verbose,
            f"[screen] run_id={run_id} engine={default_engine}"
            f"{' discovery=True' if discovery else ''} ready",
        )
        return state

    def _expand_smiles_direct_features(
        self,
        x_df: pd.DataFrame,
        state: dict[str, Any],
    ) -> pd.DataFrame:
        """Expand design rows with molecular descriptors for smiles_direct mode.

        For each row, extracts the SMILES value and calls
        ``compute_descriptors()`` to produce basic + fingerprint + electronic
        features.  No scaffold spec or substituent context needed.
        """
        from .molecular.features import compute_descriptors
        from .molecular.types import DescriptorConfig

        sd = state["smiles_direct"]
        smi_col = sd["smiles_column"]
        descriptor_names = sd["descriptor_feature_names"]

        # Rebuild DescriptorConfig from stored dict
        dc = sd["descriptor_config"]
        fp = dc.get("fingerprint", {})
        config = DescriptorConfig(
            basic=dc.get("basic", True),
            fingerprint_enabled=fp.get("enabled", True),
            fingerprint_n_bits=fp.get("n_bits", 128),
            fingerprint_radius=fp.get("radius", 2),
            electronic=dc.get("electronic", True),
            steric=dc.get("steric", False),
        )

        # Optionally compute energy features
        energy_backend = dc.get("energy_backend", "none")
        energy_fn = None
        if energy_backend != "none":
            from .molecular.energy import get_energy_features
            energy_fn = lambda smi: get_energy_features(smi, cache=None, backend=energy_backend)

        expanded_rows = []
        for _, row in x_df.iterrows():
            design_row = row.to_dict()
            smiles = str(design_row.get(smi_col, ""))
            try:
                desc = compute_descriptors(smiles, config=config)
            except (ValueError, Exception):
                desc = {}

            # Energy features
            if energy_fn is not None:
                try:
                    energy = energy_fn(smiles)
                    for k, v in energy.items():
                        desc[f"dft_{k}"] = v
                except Exception:
                    pass

            merged = {**design_row, **desc}
            expanded_rows.append(merged)

        expanded_df = pd.DataFrame(expanded_rows)
        all_features = list(state["active_features"]) + descriptor_names
        for col in all_features:
            if col not in expanded_df.columns:
                expanded_df[col] = 0.0
        return expanded_df[all_features]

    def _load_descriptor_lookup(
        self, run_id: str, state: dict[str, Any]
    ) -> dict[str, dict[str, float]]:
        """Load the precomputed descriptor lookup table for discovery mode."""
        sd = state.get("smiles_direct", {})
        fname = sd.get("descriptor_lookup_file")
        if not fname:
            raise ValueError("No descriptor lookup file found in state.")
        path = self._paths(run_id).run_dir / fname
        return read_json(path)

    def _nearest_molecule(
        self,
        descriptor_row: dict[str, float],
        lookup: dict[str, dict[str, float]],
        feature_names: list[str],
        exclude: set[str] | None = None,
    ) -> tuple[str, float]:
        """Find the nearest molecule in the lookup table by normalised distance.

        Uses min-max normalised Euclidean distance so that descriptors with
        different scales (e.g. MolWt ~200 vs fp_bit 0/1) contribute equally.

        Args:
            descriptor_row: Descriptor values from HEBO suggestion.
            lookup: SMILES → descriptor dict mapping.
            feature_names: Ordered list of feature names to compare.
            exclude: SMILES to skip (already observed).  When all molecules
                are excluded, the nearest overall is returned regardless.

        Returns:
            (best_smiles, normalised_distance) tuple.
        """
        # Build matrix of all candidate vectors
        smiles_list = list(lookup.keys())
        mat = np.array(
            [[float(lookup[smi].get(f, 0.0)) for f in feature_names] for smi in smiles_list]
        )
        query = np.array([float(descriptor_row.get(f, 0.0)) for f in feature_names])

        # Min-max normalisation per feature (avoid div-by-zero for constant cols)
        col_min = mat.min(axis=0)
        col_max = mat.max(axis=0)
        col_range = col_max - col_min
        col_range[col_range < 1e-12] = 1.0  # constant columns → no effect

        mat_norm = (mat - col_min) / col_range
        query_norm = (query - col_min) / col_range

        dists = np.linalg.norm(mat_norm - query_norm, axis=1)

        # Rank by distance and pick the first non-excluded molecule
        ranked = np.argsort(dists)
        if exclude:
            for idx in ranked:
                if smiles_list[idx] not in exclude:
                    return smiles_list[idx], float(dists[idx])
        # All excluded (or no exclusion set) → return closest
        best_idx = int(ranked[0])
        return smiles_list[best_idx], float(dists[best_idx])

    def _load_molecular_spec(self, state: dict[str, Any]) -> Any:
        """Load the MolecularDesignSpec for a molecular run."""
        from .molecular.scaffold import load_scaffold_spec
        mol_state = state.get("molecular", {})
        spec_path = mol_state.get("scaffold_spec_path")
        if spec_path is None:
            raise ValueError("Molecular run missing scaffold_spec_path in state.")
        return load_scaffold_spec(spec_path)

    def _load_energy_cache(self, run_id: str, state: dict[str, Any]) -> Any:
        """Load or create the EnergyCache for a molecular run.

        Merges entries from the user-provided ``dft_data_path`` if present.
        """
        from .molecular.energy import EnergyCache

        mol_state = state.get("molecular", {})
        desc_cfg = mol_state.get("descriptor_config", {})
        energy_backend = desc_cfg.get("energy_backend", "none")

        # Map backend label to method string for cache metadata
        method_map = {
            "xtb": "GFN2-xTB",
            "dft": "B3LYP/6-31G*",
            "ml": "ANI-2x",
            "auto": "auto",
            "none": "none",
        }
        method = method_map.get(energy_backend, energy_backend)

        paths = self._paths(run_id)
        cache = EnergyCache(path=paths.energy_cache, method=method)

        # Merge user-provided precomputed data if specified
        dft_cfg = desc_cfg.get("dft", {})
        dft_data_path = dft_cfg.get("data_path") if isinstance(dft_cfg, dict) else None
        if dft_data_path is not None:
            ext_path = Path(dft_data_path)
            if ext_path.exists():
                import json as _json
                with ext_path.open("r", encoding="utf-8") as fh:
                    external = _json.load(fh)
                added = cache.merge(external)
                if added > 0:
                    cache.save()

        return cache

    def _expand_molecular_features(
        self,
        x_df: pd.DataFrame,
        state: dict[str, Any],
        spec: Any,
        energy_cache: Any | None = None,
    ) -> pd.DataFrame:
        """Expand a DataFrame of HEBO design rows with molecular descriptors.

        Takes the raw HEBO suggestion DataFrame (substituent names + conditions)
        and appends computed descriptor columns.  Returns the augmented DataFrame.

        Parameters
        ----------
        energy_cache : EnergyCache | None
            When provided and ``dft_enabled`` is True, energy features from
            the cache are merged into each row's descriptor dict.
        """
        from .molecular.features import expand_features_for_oracle

        mol_state = state["molecular"]
        descriptor_names = mol_state["descriptor_feature_names"]

        # Convert energy cache to plain dict for features.py compatibility
        dft_cache_dict: dict | None = None
        if energy_cache is not None:
            dft_cache_dict = energy_cache.as_dict()

        expanded_rows = []
        for _, row in x_df.iterrows():
            design_row = row.to_dict()
            descriptors = expand_features_for_oracle(
                design_row, spec, dft_cache=dft_cache_dict
            )
            merged = {**design_row, **descriptors}
            expanded_rows.append(merged)

        expanded_df = pd.DataFrame(expanded_rows)
        all_features = list(state["active_features"]) + descriptor_names
        # Ensure all expected columns exist (fill missing with 0)
        for col in all_features:
            if col not in expanded_df.columns:
                expanded_df[col] = 0.0
        return expanded_df[all_features]

    def build_oracle(
        self,
        run_id: str,
        *,
        model_candidates: tuple[str, ...] = ("random_forest", "extra_trees"),
        cv_folds: int = 5,
        max_features: int | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Train/select a proxy oracle and persist model + metadata."""
        state = self._load_state(run_id)
        dataset = pd.read_csv(state["dataset_path"])
        self._log(
            verbose,
            f"[oracle] run_id={run_id} rows={len(dataset)} cv_folds={cv_folds}",
        )

        target_column = state["target_column"]
        y_raw = pd.to_numeric(dataset[target_column], errors="coerce")
        valid_mask = y_raw.notna()
        if valid_mask.sum() < 5:
            raise ValueError("Need at least 5 non-null target rows to train an oracle.")

        active_features = list(state["active_features"])
        y_full = y_raw.loc[valid_mask].to_numpy(dtype=float)

        # Discovery mode: features are descriptors, not in the raw CSV.
        # Build the training matrix from the descriptor lookup table early.
        sd = state.get("smiles_direct", {})
        is_discovery = sd.get("discovery", False)
        if is_discovery:
            x_full = pd.DataFrame()  # placeholder, built below
        else:
            x_full = dataset.loc[valid_mask, active_features].copy()

        # Molecular mode: expand features with computed descriptors
        if state.get("mode") == "molecular":
            spec = self._load_molecular_spec(state)
            mol_state = state["molecular"]
            descriptor_names = mol_state["descriptor_feature_names"]

            # Load energy cache and precompute energy features if needed
            energy_cache = None
            desc_cfg = mol_state.get("descriptor_config", {})
            energy_backend = desc_cfg.get("energy_backend", "none")
            if energy_backend != "none":
                from .molecular.energy import get_energy_features
                from .molecular.scaffold import decode_suggestion

                energy_cache = self._load_energy_cache(run_id, state)
                self._log(
                    verbose,
                    f"[oracle] Precomputing energy features (backend={energy_backend}, "
                    f"cache_size={len(energy_cache)})...",
                )
                # Precompute energy for each design row in the dataset
                for _, row in x_full.iterrows():
                    design_row = row.to_dict()
                    try:
                        full_smiles, _ = decode_suggestion(design_row, spec)
                        get_energy_features(
                            full_smiles, energy_cache, backend=energy_backend
                        )
                    except Exception as exc:
                        self._log(
                            verbose,
                            f"[oracle] Energy computation failed for row: {exc}",
                        )
                energy_cache.save()
                self._log(
                    verbose,
                    f"[oracle] Energy cache updated: {len(energy_cache)} entries.",
                )

            x_full = self._expand_molecular_features(
                x_full, state, spec, energy_cache=energy_cache
            )
            active_features = list(x_full.columns)
            # Update state so oracle knows the full feature set
            state["oracle_active_features"] = active_features

        # SMILES-direct / scaffold_edit mode: expand features with computed descriptors
        elif state.get("mode") in ("smiles_direct", "scaffold_edit"):
            sd = state.get("smiles_direct", {})
            is_discovery = sd.get("discovery", False)

            if is_discovery:
                # In discovery mode, build training matrix from descriptor
                # lookup: each row's SMILES → precomputed descriptor vector.
                self._log(verbose, "[oracle] Discovery mode: building descriptor training matrix...")
                lookup = self._load_descriptor_lookup(run_id, state)
                smi_col = sd["smiles_column"]
                desc_names = sd["descriptor_feature_names"]

                # Re-read dataset to get SMILES for each training row
                raw_dataset = pd.read_csv(state["dataset_path"])
                raw_valid = raw_dataset.loc[valid_mask]

                desc_rows = []
                valid_indices = []
                for idx, (_, raw_row) in enumerate(raw_valid.iterrows()):
                    smi = str(raw_row.get(smi_col, ""))
                    if smi in lookup:
                        row_desc = lookup[smi]
                        desc_rows.append(row_desc)
                        valid_indices.append(idx)

                if len(desc_rows) < 5:
                    raise ValueError(
                        "Discovery mode: too few molecules with computed descriptors "
                        f"({len(desc_rows)} < 5)."
                    )

                x_full = pd.DataFrame(desc_rows)
                # Align columns with active_features (descriptor dimensions)
                for col in active_features:
                    if col not in x_full.columns:
                        x_full[col] = 0.0
                x_full = x_full[active_features]
                y_full = y_full[valid_indices]
                state["oracle_active_features"] = active_features
                self._log(
                    verbose,
                    f"[oracle] Discovery training matrix: {len(x_full)} rows × "
                    f"{len(active_features)} descriptor features",
                )
            else:
                # Standard mode: expand SMILES to descriptors
                self._log(verbose, "[oracle] Expanding SMILES descriptors...")
                x_full = self._expand_smiles_direct_features(x_full, state)
                active_features = list(x_full.columns)
                state["oracle_active_features"] = active_features
                self._log(
                    verbose,
                    f"[oracle] Descriptor expansion done: {len(active_features)} features",
                )

        y_internal, target_max = _normalize_objective_values(y_full, state["objective"])

        if (
            max_features is not None
            and max_features > 0
            and len(active_features) > max_features
        ):
            x_for_importance = x_full.copy()
            for col in x_for_importance.columns:
                if pd.api.types.is_numeric_dtype(x_for_importance[col]):
                    x_for_importance[col] = pd.to_numeric(
                        x_for_importance[col], errors="coerce"
                    ).fillna(x_for_importance[col].median())
                else:
                    codes, _ = pd.factorize(
                        x_for_importance[col].astype(str), sort=True
                    )
                    x_for_importance[col] = codes

            selector = RandomForestRegressor(
                n_estimators=200, random_state=state["seed"], n_jobs=1
            )
            selector.fit(x_for_importance, y_internal)
            ranked = np.argsort(selector.feature_importances_)[::-1]
            keep_idx = ranked[:max_features]
            keep_features = [x_for_importance.columns[i] for i in keep_idx]
            ignored = [name for name in active_features if name not in keep_features]

            state["active_features"] = keep_features
            state["ignored_features"] = ignored
            if "original_design_parameters" not in state:
                state["original_design_parameters"] = list(state["design_parameters"])
            state["design_parameters"] = [
                p for p in state["design_parameters"] if p["name"] in set(keep_features)
            ]
            active_features = keep_features
            x_full = x_full[active_features]

        numeric_cols = [
            c for c in active_features if pd.api.types.is_numeric_dtype(x_full[c])
        ]
        categorical_cols = [c for c in active_features if c not in numeric_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                    numeric_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                            ),
                        ]
                    ),
                    categorical_cols,
                ),
            ],
            remainder="drop",
            sparse_threshold=0,
        )

        model_pool: dict[str, Any] = {}
        if "random_forest" in model_candidates:
            model_pool["random_forest"] = RandomForestRegressor(
                n_estimators=200,
                random_state=state["seed"],
                n_jobs=1,
            )
        if "extra_trees" in model_candidates:
            model_pool["extra_trees"] = ExtraTreesRegressor(
                n_estimators=240,
                random_state=state["seed"],
                n_jobs=1,
            )
        if not model_pool:
            raise ValueError("No supported model candidates were provided.")

        n_rows = len(x_full)
        n_splits = min(max(2, cv_folds), n_rows)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=state["seed"])

        scores: dict[str, float] = {}
        trained_pipelines: dict[str, Pipeline] = {}
        for model_name, regressor in model_pool.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", regressor),
                ]
            )
            cv_scores = cross_val_score(
                pipeline,
                x_full,
                y_internal,
                scoring="neg_root_mean_squared_error",
                cv=cv,
                n_jobs=1,
            )
            rmse = float(-np.mean(cv_scores))
            scores[model_name] = rmse
            trained_pipelines[model_name] = pipeline
            self._log(verbose, f"[oracle] {model_name}: cv_rmse={rmse:.4f}")

        best_model_name = min(scores, key=lambda k: scores[k])
        best_pipeline = trained_pipelines[best_model_name]
        best_pipeline.fit(x_full, y_internal)

        paths = self._paths(run_id)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        with paths.oracle_model.open("wb") as handle:
            pickle.dump(best_pipeline, handle)

        oracle_meta = {
            "built_at": utc_now_iso(),
            "model_candidates": list(model_pool.keys()),
            "cv_rmse": scores,
            "selected_model": best_model_name,
            "selected_rmse": scores[best_model_name],
            "rows_used": int(n_rows),
            "active_features": list(active_features),
            "objective_internal": "min",
            "target_max_for_restore": target_max,
        }
        write_json(paths.oracle_meta, oracle_meta)

        state["objective_transform"] = {
            "internal_objective": "min",
            "target_max_for_restore": target_max,
        }

        state["oracle"] = oracle_meta
        state["status"] = "oracle_ready"
        state["updated_at"] = utc_now_iso()
        self._save_state(run_id, state)
        self._log(
            verbose,
            f"[oracle] selected={best_model_name} rmse={scores[best_model_name]:.4f}",
        )

        return {
            "run_id": run_id,
            "status": state["status"],
            "active_features": state["active_features"],
            "ignored_features": state["ignored_features"],
            "selected_model": best_model_name,
            "selected_rmse": scores[best_model_name],
            "cv_rmse": scores,
        }

    def _build_optimizer(
        self,
        state: dict[str, Any],
        observations: list[dict[str, Any]],
        engine_name: OptimizerName,
    ) -> HEBO | BO:
        """Build optimizer from replayed observation history.

        We reconstruct from history (instead of keeping in-memory optimizer
        state) so commands remain resumable and deterministic from run files.
        """
        np.random.seed(int(state["seed"]) + len(observations))
        design_space = DesignSpace().parse(state["design_parameters"])
        if engine_name == "hebo":
            optimizer: HEBO | BO = HEBO(
                design_space,
                model_name="gp",
                rand_sample=int(state["num_initial_random_samples"]),
                scramble_seed=int(state["seed"]),
            )
            if observations:
                # Replay Sobol sequence position to match previous suggestions.
                optimizer.sobol.fast_forward(len(observations))
        elif engine_name == "bo_lcb":
            optimizer = BO(
                design_space,
                model_name="gp",
                rand_sample=int(state["num_initial_random_samples"]),
            )
        else:
            raise ValueError(f"Unsupported optimizer engine '{engine_name}'")

        if observations:
            x_rows = [
                {k: row["x"][k] for k in state["active_features"]}
                for row in observations
            ]
            x_obs = pd.DataFrame(x_rows)
            y_obs = np.array(
                [float(row["y_internal"]) for row in observations], dtype=float
            ).reshape(-1, 1)
            optimizer.observe(x_obs, y_obs)
        return optimizer

    def suggest(
        self,
        run_id: str,
        *,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        state = self._load_state(run_id)
        if state["status"] not in {"initialized", "oracle_ready", "running"}:
            raise ValueError(
                f"Run '{run_id}' is not ready for suggestions. Current status: {state['status']}"
            )

        engine = str(state.get("default_engine", "hebo"))
        if engine not in {"hebo", "bo_lcb", "random"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random")
        engine_typed: OptimizerName = engine  # type: ignore[assignment]  # validated above

        size = int(batch_size or state["default_batch_size"])
        observations = read_jsonl(self._paths(run_id).observations)
        if engine_typed == "random":
            np.random.seed(int(state["seed"]) + len(observations))
            proposals = DesignSpace().parse(state["design_parameters"]).sample(size)
        else:
            if engine_typed == "bo_lcb" and size != 1:
                raise ValueError("bo_lcb currently supports batch-size=1 only.")
            optimizer = self._build_optimizer(state, observations, engine_typed)
            proposals = optimizer.suggest(n_suggestions=size)

        # Molecular mode: feasibility filtering
        feasibility_results: list[Any] = []
        if state.get("mode") == "molecular":
            from .molecular.feasibility import assess_feasibility_batch
            from .molecular.scaffold import decode_suggestion

            spec = self._load_molecular_spec(state)
            mol_state = state["molecular"]
            feas_mode = mol_state["feasibility"]["mode"]
            sa_threshold = mol_state["feasibility"]["sa_threshold"]

            # Load energy cache for strain-based feasibility
            energy_cache = None
            desc_cfg = mol_state.get("descriptor_config", {})
            energy_backend = desc_cfg.get("energy_backend", "none")
            if energy_backend != "none":
                energy_cache = self._load_energy_cache(run_id, state)

            proposal_dicts = [
                row_to_python_dict(row) for _, row in proposals.iterrows()
            ]
            feasibility_results = assess_feasibility_batch(
                proposal_dicts, spec,
                sa_threshold=sa_threshold,
                energy_cache=energy_cache,
            )

            if feas_mode == "hard":
                # Filter out infeasible proposals
                filtered_rows = []
                filtered_results = []
                for p_dict, fr in zip(proposal_dicts, feasibility_results):
                    if fr.is_feasible:
                        filtered_rows.append(p_dict)
                        filtered_results.append(fr)
                if filtered_rows:
                    proposals = pd.DataFrame(filtered_rows)
                    feasibility_results = filtered_results
                else:
                    self._log(
                        verbose,
                        f"[suggest] All {len(proposal_dicts)} proposals failed "
                        f"feasibility (hard mode). Keeping original proposals.",
                    )
                    # Fall through with unfiltered proposals

            # Log feasibility results
            paths = self._paths(run_id)
            for p_dict, fr in zip(
                [row_to_python_dict(r) for _, r in proposals.iterrows()],
                feasibility_results,
            ):
                feas_log = {
                    "event_time": utc_now_iso(),
                    "x": p_dict,
                    "is_feasible": fr.is_feasible,
                    "sa_score": fr.sa_score,
                    "penalty": fr.penalty,
                    "reasons": fr.reasons,
                    "strain_energy_kcal": fr.strain_energy_kcal,
                }
                append_jsonl(paths.feasibility_log, feas_log)

        # Discovery mode: map descriptor-space proposals to nearest molecules
        sd = state.get("smiles_direct", {})
        is_discovery = sd.get("discovery", False)
        discovery_lookup: dict[str, dict[str, float]] | None = None
        already_observed_smiles: set[str] = set()
        if is_discovery:
            discovery_lookup = self._load_descriptor_lookup(run_id, state)
            # Collect SMILES that have already been observed so we can prefer
            # novel molecules (avoids GP jitter from duplicate observations).
            smi_col = sd.get("smiles_column", "smiles")
            for obs in observations:
                ms = obs.get("matched_smiles")
                if ms:
                    already_observed_smiles.add(ms)

        rows = []
        for idx_row, (_, row) in enumerate(proposals.iterrows()):
            x = row_to_python_dict(row)
            x.update(state["fixed_features"])

            payload: dict[str, Any] = {
                "event_time": utc_now_iso(),
                "suggestion_id": secrets.token_hex(16),
                "iteration": len(observations),
                "engine": engine_typed,
                "x": x,
            }

            # Discovery mode: resolve descriptor vector to nearest real molecule
            if is_discovery and discovery_lookup is not None:
                # Use active_features (selected top-N) for distance, not all descriptors
                feature_names = list(state["active_features"])
                best_smi, dist = self._nearest_molecule(
                    x, discovery_lookup, feature_names,
                    exclude=already_observed_smiles,
                )
                payload["x_descriptor"] = x  # keep the raw descriptor suggestion
                payload["x"] = {sd["smiles_column"]: best_smi}  # map to SMILES
                payload["discovery_distance"] = dist
                already_observed_smiles.add(best_smi)  # track for batch dedup
                self._log(
                    verbose,
                    f"[suggest] Discovery: mapped to {best_smi} (dist={dist:.3f})",
                )

            # Attach feasibility metadata if available
            if feasibility_results and idx_row < len(feasibility_results):
                fr = feasibility_results[idx_row]
                payload["feasibility"] = {
                    "sa_score": fr.sa_score,
                    "is_feasible": fr.is_feasible,
                    "penalty": fr.penalty,
                    "strain_energy_kcal": fr.strain_energy_kcal,
                }

            append_jsonl(self._paths(run_id).suggestions, payload)
            rows.append(payload)

        state["status"] = "running"
        state["updated_at"] = utc_now_iso()
        self._save_state(run_id, state)
        self._log(
            verbose,
            f"[suggest] run_id={run_id} engine={engine_typed} n={len(rows)}",
        )

        return {
            "run_id": run_id,
            "engine": engine_typed,
            "num_suggestions": len(rows),
            "suggestions": rows,
        }

    def observe(
        self,
        run_id: str,
        observations: list[dict[str, Any]],
        *,
        source: str = "user",
        verbose: bool = False,
    ) -> dict[str, Any]:
        state = self._load_state(run_id)
        if not observations:
            raise ValueError("No observations provided.")

        target_col = state["target_column"]
        existing = read_jsonl(self._paths(run_id).observations)
        next_iteration = len(existing)
        rows = []

        for idx, obs in enumerate(observations):
            x = dict(obs.get("x", {}))
            engine = str(obs.get("engine", state.get("default_engine", "hebo")))
            for feature in state["active_features"]:
                if feature not in x:
                    if feature in state["fixed_features"]:
                        x[feature] = state["fixed_features"][feature]
                    else:
                        raise ValueError(
                            f"Observation missing required feature '{feature}'."
                        )

            y_value = obs.get("y", obs.get(target_col))
            if y_value is None:
                raise ValueError(
                    f"Observation missing objective value. Provide 'y' or '{target_col}'."
                )
            y_float = float(y_value)

            if state["objective"] == "min":
                y_internal = y_float
            else:
                transform = state.get("objective_transform") or {}
                target_max_for_restore = transform.get("target_max_for_restore")
                if target_max_for_restore is None:
                    raise ValueError(
                        "Missing target_max_for_restore for max objective. Reinitialize run."
                    )
                y_internal = float(target_max_for_restore) - y_float

            payload: dict[str, Any] = {
                "event_time": utc_now_iso(),
                "iteration": next_iteration + idx,
                "source": source,
                "engine": engine,
                "suggestion_id": obs.get("suggestion_id"),
                "x": {k: to_python_scalar(v) for k, v in x.items()},
                "y": y_float,
                "y_internal": y_internal,
            }
            # Discovery mode: preserve matched molecule info
            if obs.get("matched_smiles") is not None:
                payload["matched_smiles"] = obs["matched_smiles"]
            if obs.get("discovery_distance") is not None:
                payload["discovery_distance"] = obs["discovery_distance"]
            append_jsonl(self._paths(run_id).observations, payload)
            rows.append(payload)

        state["updated_at"] = utc_now_iso()
        self._save_state(run_id, state)
        self._log(
            verbose,
            f"[observe] run_id={run_id} source={source} recorded={len(rows)}",
        )
        return {"run_id": run_id, "recorded": len(rows), "observations": rows}

    def evaluate_last_suggestions(
        self, run_id: str, *, max_new: int | None = None, verbose: bool = False
    ) -> dict[str, Any]:
        """Evaluate pending suggestions with proxy oracle (simulation mode)."""
        state = self._load_state(run_id)
        if state.get("oracle") is None:
            raise ValueError(
                f"Oracle not built for run '{run_id}'. Run build-oracle first."
            )
        suggestions = read_jsonl(self._paths(run_id).suggestions)
        observations = read_jsonl(self._paths(run_id).observations)

        observed_suggestion_ids = {
            str(row["suggestion_id"])
            for row in observations
            if row.get("suggestion_id") is not None
        }
        already_seen_x = {json.dumps(row["x"], sort_keys=True) for row in observations}
        pending: list[dict[str, Any]] = []
        for suggestion in suggestions:
            suggestion_id = suggestion.get("suggestion_id")
            if suggestion_id is not None:
                if str(suggestion_id) in observed_suggestion_ids:
                    continue
                pending.append(suggestion)
                continue

            # Backward-compatible fallback for historical runs without IDs.
            if json.dumps(suggestion["x"], sort_keys=True) not in already_seen_x:
                pending.append(suggestion)
        if max_new is not None:
            pending = pending[:max_new]

        if not pending:
            self._log(verbose, f"[eval] run_id={run_id} no pending suggestions")
            return {"run_id": run_id, "evaluated": 0, "observations": []}

        # Discovery mode: x contains SMILES (not descriptors), handle separately
        sd = state.get("smiles_direct", {})
        is_discovery_eval = sd.get("discovery", False)
        if is_discovery_eval:
            x_df = pd.DataFrame()  # placeholder, built below
        else:
            x_df = pd.DataFrame([row["x"] for row in pending])[state["active_features"]]

        # Molecular mode: expand features for oracle prediction
        if state.get("mode") == "molecular":
            spec = self._load_molecular_spec(state)
            # Load energy cache if energy backend is active
            energy_cache = None
            mol_state = state.get("molecular", {})
            desc_cfg = mol_state.get("descriptor_config", {})
            energy_backend = desc_cfg.get("energy_backend", "none")
            if energy_backend != "none":
                from .molecular.energy import get_energy_features
                from .molecular.scaffold import decode_suggestion

                energy_cache = self._load_energy_cache(run_id, state)
                # Precompute energy for new molecules
                for row_dict in [row["x"] for row in pending]:
                    try:
                        full_smiles, _ = decode_suggestion(row_dict, spec)
                        get_energy_features(
                            full_smiles, energy_cache, backend=energy_backend
                        )
                    except Exception:
                        pass
                energy_cache.save()

            x_df = self._expand_molecular_features(
                x_df, state, spec, energy_cache=energy_cache
            )

        # SMILES-direct / scaffold_edit mode: expand features for oracle prediction
        elif state.get("mode") in ("smiles_direct", "scaffold_edit"):
            sd = state.get("smiles_direct", {})
            is_discovery = sd.get("discovery", False)

            if is_discovery:
                # In discovery mode, the oracle was trained on descriptor features.
                # Use the *matched real molecule's* descriptors for prediction
                # (not the HEBO-proposed virtual descriptor point).
                desc_rows = []
                lookup = self._load_descriptor_lookup(run_id, state)
                for p in pending:
                    # Always use matched molecule's real descriptors
                    smi = p["x"].get(sd["smiles_column"], "")
                    if smi in lookup:
                        desc_rows.append(lookup[smi])
                    elif "x_descriptor" in p:
                        # Fallback: use HEBO's descriptor point
                        desc_rows.append(p["x_descriptor"])
                    else:
                        desc_rows.append({})
                x_df = pd.DataFrame(desc_rows)
                for col in state["active_features"]:
                    if col not in x_df.columns:
                        x_df[col] = 0.0
                x_df = x_df[state["active_features"]]
            else:
                x_df = self._expand_smiles_direct_features(x_df, state)

        y_pred = self._oracle_predict_original_scale(run_id, state, x_df)

        # Molecular mode (soft feasibility): apply penalty to predictions
        if state.get("mode") == "molecular":
            mol_state = state["molecular"]
            feas_mode = mol_state["feasibility"]["mode"]
            if feas_mode == "soft":
                for i, row in enumerate(pending):
                    feas_info = row.get("feasibility")
                    if feas_info and feas_info.get("penalty", 0) > 0:
                        penalty = float(feas_info["penalty"])
                        if state["objective"] == "max":
                            y_pred[i] -= penalty  # lower is worse for max
                        else:
                            y_pred[i] += penalty  # higher is worse for min

        # Build observation payloads
        sd = state.get("smiles_direct", {})
        is_discovery = sd.get("discovery", False)

        # For discovery mode, load lookup for real descriptor values
        if is_discovery:
            _disc_lookup = self._load_descriptor_lookup(run_id, state)
        else:
            _disc_lookup = {}

        payloads = []
        for row, y_val in zip(pending, y_pred, strict=True):
            if is_discovery and "x_descriptor" in row:
                # In discovery mode, store the *real molecule's* descriptor
                # values as x (for HEBO replay), so the GP learns from real
                # descriptor-target pairs.  Falls back to HEBO's virtual point.
                matched_smi = row["x"].get(sd["smiles_column"], "")
                if matched_smi in _disc_lookup:
                    real_desc = {
                        f: float(_disc_lookup[matched_smi].get(f, 0.0))
                        for f in state["active_features"]
                    }
                else:
                    real_desc = row["x_descriptor"]  # fallback
                obs_payload = {
                    "x": real_desc,
                    "y": float(y_val),
                    "engine": row.get("engine", state.get("default_engine", "hebo")),
                    "suggestion_id": row.get("suggestion_id"),
                    "matched_smiles": matched_smi,
                    "discovery_distance": row.get("discovery_distance"),
                }
            else:
                obs_payload = {
                    "x": row["x"],
                    "y": float(y_val),
                    "engine": row.get("engine", state.get("default_engine", "hebo")),
                    "suggestion_id": row.get("suggestion_id"),
                }
            payloads.append(obs_payload)

        observed = self.observe(
            run_id, payloads, source="proxy-oracle", verbose=verbose
        )
        self._log(verbose, f"[eval] run_id={run_id} evaluated={observed['recorded']}")
        return {
            "run_id": run_id,
            "evaluated": observed["recorded"],
            "observations": observed["observations"],
        }

    def run_proxy_optimization(
        self,
        run_id: str,
        *,
        observer: Observer,
        num_iterations: int,
        batch_size: int = 1,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run a BO loop with a pluggable observer for evaluation."""
        self._log(
            verbose,
            f"[run] run_id={run_id} iterations={num_iterations} batch_size={batch_size} observer={observer.source}",
        )
        progress = tqdm(
            range(num_iterations),
            desc=f"run {run_id}",
            unit="iter",
            disable=not verbose,
            file=sys.stderr,
        )
        for _ in progress:
            result = self.suggest(run_id, batch_size=batch_size, verbose=verbose)
            observations = observer.evaluate(result["suggestions"])
            if observations:
                self.observe(
                    run_id, observations, source=observer.source, verbose=verbose
                )
            if verbose:
                status = self.status(run_id)
                best = status.get("best_value")
                if best is not None:
                    progress.set_postfix(best=f"{float(best):.4f}")
        state = self._load_state(run_id)
        state["status"] = "completed"
        state["updated_at"] = utc_now_iso()
        self._save_state(run_id, state)
        self._log(verbose, f"[run] run_id={run_id} completed")
        return self.report(run_id, verbose=verbose)

    def status(self, run_id: str) -> dict[str, Any]:
        state = self._load_state(run_id)
        observations = read_jsonl(self._paths(run_id).observations)

        payload: dict[str, Any] = {
            "run_id": run_id,
            "status": state["status"],
            "objective": state["objective"],
            "default_engine": state.get("default_engine", "hebo"),
            "target_column": state["target_column"],
            "active_features": state["active_features"],
            "ignored_features": state["ignored_features"],
            "num_observations": len(observations),
        }
        if observations:
            y_values = np.asarray(
                [float(row["y"]) for row in observations], dtype=float
            )
            if state["objective"] == "min":
                best_idx = int(np.argmin(y_values))
                best_value = float(np.min(y_values))
            else:
                best_idx = int(np.argmax(y_values))
                best_value = float(np.max(y_values))
            payload["best_value"] = best_value
            payload["best_iteration"] = best_idx
            best_obs = observations[best_idx]
            payload["best_x"] = best_obs["x"]
            # Discovery mode: also report the matched real molecule
            if best_obs.get("matched_smiles"):
                payload["best_matched_smiles"] = best_obs["matched_smiles"]
                payload["best_discovery_distance"] = best_obs.get("discovery_distance")

        if state["oracle"] is not None:
            payload["oracle"] = {
                "selected_model": state["oracle"]["selected_model"],
                "selected_rmse": state["oracle"]["selected_rmse"],
            }
        return payload

    def report(self, run_id: str, *, verbose: bool = False) -> dict[str, Any]:
        """Generate report JSON and convergence plot for a run."""
        state = self._load_state(run_id)
        observations = read_jsonl(self._paths(run_id).observations)
        if not observations:
            report = {
                "run_id": run_id,
                "message": "No observations recorded yet.",
                "generated_at": utc_now_iso(),
            }
            write_json(self._paths(run_id).report, report)
            return report

        grouped: dict[str, list[float]] = {}
        for row in observations:
            engine = str(row.get("engine", state.get("default_engine", "hebo")))
            grouped.setdefault(engine, []).append(float(row["y"]))

        methods_data: dict[str, np.ndarray] = {}
        for engine, values in grouped.items():
            label = {
                "hebo": "HEBO",
                "bo_lcb": "BO (LCB)",
                "random": "Random Search",
            }.get(engine, engine)
            methods_data[label] = np.asarray(values, dtype=float)

        plot_optimization_convergence(
            methods_data,
            title=f"Run {run_id}",
            ylabel=state["target_column"],
            objective=state["objective"],
            fig_path=str(self._paths(run_id).convergence_plot),
            show=False,
        )

        status = self.status(run_id)
        report: dict[str, Any] = {
            "run_id": run_id,
            "generated_at": utc_now_iso(),
            "num_observations": len(observations),
            "objective": state["objective"],
            "target_column": state["target_column"],
            "best_value": status.get("best_value"),
            "best_iteration": status.get("best_iteration"),
            "best_x": status.get("best_x"),
            "oracle": status.get("oracle"),
            "artifacts": {
                "plot": str(self._paths(run_id).convergence_plot),
                "state": str(self._paths(run_id).state),
                "observations": str(self._paths(run_id).observations),
            },
        }
        # Discovery mode: include matched molecule in report
        if status.get("best_matched_smiles"):
            report["best_matched_smiles"] = status["best_matched_smiles"]
            report["best_discovery_distance"] = status.get("best_discovery_distance")
        write_json(self._paths(run_id).report, report)
        self._log(
            verbose,
            f"[report] run_id={run_id} observations={len(observations)} best={report.get('best_value')}",
        )
        return report

    # ------------------------------------------------------------------
    # Scaffold editing (CReM + HEBO discovery)
    # ------------------------------------------------------------------

    def init_scaffold_edit_run(
        self,
        *,
        dataset_path: str | Path,
        target_column: str,
        objective: Objective,
        smiles_column: str | None = None,
        seed_smiles: list[str] | None = None,
        top_k_seeds: int = 10,
        crem_db_path: str | Path | None = None,
        mutation_constraint: dict[str, Any] | None = None,
        max_size: int = 3,
        radius: int = 3,
        max_replacements_per_seed: int = 100,
        operations: list[str] | None = None,
        sa_threshold: float = 6.0,
        default_engine: OptimizerName = "hebo",
        run_id: str | None = None,
        num_initial_random_samples: int = 10,
        default_batch_size: int = 1,
        seed: int = 7,
        fingerprint_bits: int = 128,
        energy_backend: str = "none",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Initialize a scaffold-editing run using CReM mutations + HEBO.

        Generates a pool of CReM mutations from seed molecules, filters by
        feasibility, computes descriptors, and sets up HEBO to search the
        continuous descriptor space (discovery mode) over the CReM pool.

        The *energy_backend* parameter here controls only the initial
        descriptor computation.  Call :meth:`augment_with_energy` after
        this method to add xTB/DFT features (with user confirmation).
        """
        if objective not in {"min", "max"}:
            raise ValueError("objective must be 'min' or 'max'")
        if default_engine not in {"hebo", "bo_lcb", "random"}:
            raise ValueError("default_engine must be one of: hebo, bo_lcb, random")

        dataset_path = Path(dataset_path).resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path)
        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' not in dataset: {list(data.columns)}"
            )

        smi_col = _detect_smiles_column(data, hint=smiles_column)
        self._log(verbose, f"[scaffold-edit] SMILES column: '{smi_col}'")

        # --- Seed selection ---
        from .molecular.crem_ops import (
            constraint_from_dict,
            generate_candidate_pool,
            locate_crem_db,
            select_seed_molecules,
        )

        if seed_smiles is None or len(seed_smiles) == 0:
            seed_smiles = select_seed_molecules(
                dataset_path, target_column, objective, smi_col,
                top_k=top_k_seeds,
            )
        self._log(verbose, f"[scaffold-edit] {len(seed_smiles)} seed molecules selected")

        # --- CReM database ---
        db_path = locate_crem_db(crem_db_path)
        if db_path is None:
            raise FileNotFoundError(
                "CReM fragment database not found. Download with:\n"
                "  wget http://www.qsar4u.com/files/cremdb/replacements02_sc2.db.gz\n"
                "  gunzip replacements02_sc2.db.gz\n"
                "  mv replacements02_sc2.db data/"
            )

        # --- Generate CReM candidate pool ---
        constraint = constraint_from_dict(mutation_constraint)
        known_smiles = set(data[smi_col].dropna().astype(str).unique())

        candidates = generate_candidate_pool(
            seed_smiles,
            db_path=db_path,
            constraint=constraint,
            max_size=max_size,
            radius=radius,
            max_replacements_per_seed=max_replacements_per_seed,
            operations=operations,
            exclude_smiles=known_smiles,
            verbose=verbose,
        )
        total_generated = len(candidates)
        self._log(verbose, f"[scaffold-edit] CReM generated {total_generated} novel candidates")

        if total_generated == 0:
            raise ValueError(
                "CReM produced no novel candidates.  Try increasing --max-size "
                "or --max-replacements, or providing different seed molecules."
            )

        # --- Feasibility pre-screening ---
        from .molecular.energy import _estimate_strain_energy
        from .molecular.feasibility import assess_feasibility

        feasible_candidates = []
        feasibility_results: dict[str, dict[str, Any]] = {}
        for cand in candidates:
            try:
                strain = _estimate_strain_energy(cand.smiles)
                feas = assess_feasibility(
                    cand.smiles, sa_threshold=sa_threshold,
                    strain_energy_kcal=strain,
                )
            except Exception:
                continue
            feasibility_results[cand.smiles] = {
                "is_feasible": feas.is_feasible,
                "sa_score": feas.sa_score,
                "strain_energy_kcal": feas.strain_energy_kcal,
                "penalty": feas.penalty,
                "reasons": feas.reasons,
            }
            if feas.is_feasible:
                feasible_candidates.append(cand)

        total_feasible = len(feasible_candidates)
        total_filtered = total_generated - total_feasible
        self._log(
            verbose,
            f"[scaffold-edit] Feasibility: {total_feasible} passed, {total_filtered} filtered",
        )

        if total_feasible < 2:
            raise ValueError(
                f"Only {total_feasible} candidate(s) passed feasibility. "
                "Try relaxing --sa-threshold or adjusting CReM parameters."
            )

        # --- Compute descriptors for CReM candidates + dataset molecules ---
        from .molecular.features import compute_descriptors
        from .molecular.types import DescriptorConfig

        desc_config = DescriptorConfig(
            basic=True,
            fingerprint_enabled=True,
            fingerprint_n_bits=fingerprint_bits,
            fingerprint_radius=2,
            electronic=True,
            steric=False,
            dft_enabled=False,
            energy_backend="none",
        )

        descriptor_lookup: dict[str, dict[str, float]] = {}

        # Dataset molecules
        for smi in data[smi_col].dropna().unique():
            try:
                desc = compute_descriptors(str(smi), config=desc_config)
                descriptor_lookup[str(smi)] = desc
            except Exception:
                pass

        # CReM candidates
        for cand in feasible_candidates:
            if cand.smiles not in descriptor_lookup:
                try:
                    desc = compute_descriptors(cand.smiles, config=desc_config)
                    descriptor_lookup[cand.smiles] = desc
                except Exception:
                    pass

        self._log(
            verbose,
            f"[scaffold-edit] Descriptors computed for {len(descriptor_lookup)} molecules",
        )

        # Determine descriptor feature names from first entry
        sample_desc = next(iter(descriptor_lookup.values()))
        descriptor_feature_names = sorted(sample_desc.keys())

        # --- Feature importance + discovery design params ---
        y_for_importance = pd.to_numeric(data[target_column], errors="coerce")
        smi_series = data[smi_col].astype(str)
        valid_rows = []
        y_vals = []
        for smi, yv in zip(smi_series, y_for_importance):
            if pd.notna(yv) and smi in descriptor_lookup:
                valid_rows.append(descriptor_lookup[smi])
                y_vals.append(float(yv))

        if len(valid_rows) < 5:
            raise ValueError(
                "Need at least 5 dataset rows with valid SMILES+target for "
                "feature selection."
            )

        desc_df = pd.DataFrame.from_dict(descriptor_lookup, orient="index")
        x_imp = pd.DataFrame(valid_rows)
        y_imp = np.array(y_vals)

        variable_cols = [
            col for col in x_imp.columns
            if len(x_imp[col].dropna()) > 0 and not np.isclose(
                x_imp[col].dropna().min(), x_imp[col].dropna().max()
            )
        ]
        x_imp = x_imp[variable_cols].fillna(0.0)

        max_dims = min(15, len(variable_cols))
        if len(variable_cols) > max_dims:
            rf = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
            rf.fit(x_imp, y_imp)
            importances = rf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:max_dims]
            selected_cols = [variable_cols[i] for i in top_idx]
            self._log(
                verbose,
                f"[scaffold-edit] Selected top-{max_dims} descriptors by importance",
            )
        else:
            selected_cols = variable_cols

        design_params: list[dict[str, Any]] = []
        for col in selected_cols:
            vals = desc_df[col].dropna() if col in desc_df.columns else pd.Series(dtype=float)
            if vals.empty:
                continue
            lb, ub = float(vals.min()), float(vals.max())
            if np.isclose(lb, ub):
                continue
            design_params.append({"name": col, "type": "num", "lb": lb, "ub": ub})

        if not design_params:
            raise ValueError("No variable descriptors found across CReM candidates.")

        self._log(
            verbose,
            f"[scaffold-edit] {len(design_params)} continuous descriptor dimensions for HEBO",
        )

        # --- Generate run_id ---
        if run_id is None:
            for _ in range(20):
                candidate_id = generate_run_id()
                if not self._paths(candidate_id).state.exists():
                    run_id = candidate_id
                    break
            if run_id is None:
                raise RuntimeError("Failed to generate a unique run_id.")
        elif self._paths(run_id).state.exists():
            raise ValueError(f"Run '{run_id}' already exists.")

        # --- Objective transform ---
        numeric_target = pd.to_numeric(data[target_column], errors="coerce")
        target_max = float("nan")
        if objective == "max" and numeric_target.notna().sum() > 0:
            target_max = float(numeric_target.max())

        # --- Save state ---
        state: dict[str, Any] = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "status": "initialized",
            "mode": "scaffold_edit",
            "dataset_path": str(dataset_path),
            "target_column": target_column,
            "objective": objective,
            "default_engine": default_engine,
            "seed": int(seed),
            "num_initial_random_samples": int(num_initial_random_samples),
            "default_batch_size": int(default_batch_size),
            "design_parameters": design_params,
            "active_features": [p["name"] for p in design_params],
            "fixed_features": {},
            "dropped_features": [],
            "ignored_features": [],
            "oracle": None,
            "objective_transform": {
                "internal_objective": "min",
                "target_max_for_restore": target_max,
            },
            # Reuse smiles_direct + discovery infrastructure
            "smiles_direct": {
                "smiles_column": smi_col,
                "discovery": True,
                "descriptor_feature_names": descriptor_feature_names,
                "descriptor_config": {
                    "basic": desc_config.basic,
                    "fingerprint": {
                        "enabled": desc_config.fingerprint_enabled,
                        "n_bits": desc_config.fingerprint_n_bits,
                        "radius": desc_config.fingerprint_radius,
                    },
                    "electronic": desc_config.electronic,
                    "steric": desc_config.steric,
                    "energy_backend": energy_backend,
                },
                "descriptor_lookup_file": "descriptor_lookup.json",
            },
            "scaffold_edit": {
                "seed_smiles": seed_smiles,
                "crem_db_path": str(db_path),
                "mutation_constraint": mutation_constraint,
                "crem_params": {
                    "max_size": max_size,
                    "radius": radius,
                    "max_replacements_per_seed": max_replacements_per_seed,
                    "operations": operations or ["mutate"],
                },
                "sa_threshold": sa_threshold,
                "total_generated": total_generated,
                "total_feasible": total_feasible,
                "total_filtered": total_filtered,
                "energy_backend": energy_backend,
                "xtb_confirmed": False,
                "validation_method": None,
            },
        }

        paths = self._paths(run_id)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_state(run_id, state)

        # Save descriptor lookup (shared with discovery mode)
        write_json(paths.run_dir / "descriptor_lookup.json", descriptor_lookup)

        # Save CReM candidate metadata
        crem_meta = {c.smiles: c.to_dict() for c in feasible_candidates}
        write_json(paths.crem_candidates, crem_meta)

        # Save feasibility results
        write_json(paths.crem_feasibility, feasibility_results)

        self._log(
            verbose,
            f"[scaffold-edit] run_id={run_id} ready "
            f"(seeds={len(seed_smiles)} candidates={total_feasible} dims={len(design_params)})",
        )
        return state

    def augment_with_energy(
        self,
        run_id: str,
        *,
        energy_backend: str = "xtb",
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Augment descriptor lookup with xTB/DFT energy features.

        Call this after :meth:`init_scaffold_edit_run` and before
        :meth:`build_oracle` to enrich the oracle feature space with
        electronic structure features (HOMO/LUMO/gap/dipole/strain).
        """
        state = self._load_state(run_id)
        if state.get("mode") != "scaffold_edit":
            raise ValueError("augment_with_energy requires a scaffold_edit run.")

        from .molecular.energy import EnergyCache, get_energy_features

        paths = self._paths(run_id)
        cache = EnergyCache(path=paths.energy_cache, method=energy_backend.upper())

        lookup = self._load_descriptor_lookup(run_id, state)
        smiles_list = list(lookup.keys())
        computed = 0
        errors = 0

        for smi in tqdm(smiles_list, desc="xTB energy", disable=not verbose):
            if smi in cache:
                energy = cache.get(smi)
            else:
                try:
                    energy = get_energy_features(
                        smi, cache, backend=energy_backend
                    )
                    computed += 1
                except Exception as exc:
                    self._log(verbose, f"[augment] Energy failed for {smi}: {exc}")
                    errors += 1
                    continue

            # Merge energy features into descriptor dict
            if energy:
                for k, v in energy.items():
                    lookup[smi][f"dft_{k}"] = float(v) if v is not None and not np.isnan(v) else 0.0

        cache.save()
        self._log(
            verbose,
            f"[augment] Computed {computed} new energies ({errors} errors, "
            f"{len(smiles_list) - computed - errors} cached)",
        )

        # Update descriptor feature names
        energy_names = [
            "dft_total_energy_hartree", "dft_homo_ev", "dft_lumo_ev",
            "dft_gap_ev", "dft_dipole_debye", "dft_strain_energy_kcal",
        ]
        sd = state["smiles_direct"]
        existing_names = set(sd["descriptor_feature_names"])
        for n in energy_names:
            if n not in existing_names:
                sd["descriptor_feature_names"].append(n)
        sd["descriptor_feature_names"] = sorted(sd["descriptor_feature_names"])
        sd["descriptor_config"]["energy_backend"] = energy_backend

        # Re-run feature selection with augmented descriptors
        desc_df = pd.DataFrame.from_dict(lookup, orient="index")
        data = pd.read_csv(state["dataset_path"])
        smi_col = sd["smiles_column"]
        y_for_importance = pd.to_numeric(data[state["target_column"]], errors="coerce")
        smi_series = data[smi_col].astype(str)

        valid_rows = []
        y_vals = []
        for smi, yv in zip(smi_series, y_for_importance):
            if pd.notna(yv) and smi in lookup:
                valid_rows.append(lookup[smi])
                y_vals.append(float(yv))

        if len(valid_rows) >= 5:
            x_imp = pd.DataFrame(valid_rows).fillna(0.0)
            variable_cols = [
                col for col in x_imp.columns
                if len(x_imp[col].dropna()) > 0
                and not np.isclose(x_imp[col].dropna().min(), x_imp[col].dropna().max())
            ]
            x_imp_var = x_imp[variable_cols]

            max_dims = min(15, len(variable_cols))
            if len(variable_cols) > max_dims:
                rf = RandomForestRegressor(
                    n_estimators=200, random_state=state["seed"], n_jobs=1
                )
                rf.fit(x_imp_var, np.array(y_vals))
                top_idx = np.argsort(rf.feature_importances_)[::-1][:max_dims]
                selected_cols = [variable_cols[i] for i in top_idx]
            else:
                selected_cols = variable_cols

            design_params: list[dict[str, Any]] = []
            for col in selected_cols:
                vals = desc_df[col].dropna() if col in desc_df.columns else pd.Series(dtype=float)
                if vals.empty:
                    continue
                lb, ub = float(vals.min()), float(vals.max())
                if np.isclose(lb, ub):
                    continue
                design_params.append({"name": col, "type": "num", "lb": lb, "ub": ub})

            if design_params:
                state["design_parameters"] = design_params
                state["active_features"] = [p["name"] for p in design_params]
                self._log(
                    verbose,
                    f"[augment] Re-selected {len(design_params)} features (with energy)",
                )

        # Update scaffold_edit metadata
        state["scaffold_edit"]["energy_backend"] = energy_backend
        state["scaffold_edit"]["xtb_confirmed"] = True
        state["updated_at"] = utc_now_iso()

        # Write updated lookup + state
        write_json(paths.run_dir / "descriptor_lookup.json", lookup)
        self._save_state(run_id, state)

        return {
            "run_id": run_id,
            "energy_backend": energy_backend,
            "molecules_computed": computed,
            "errors": errors,
            "total_molecules": len(smiles_list),
            "active_features": state["active_features"],
        }

    def get_top_candidates(
        self,
        run_id: str,
        *,
        top_k: int = 10,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Extract top-K candidates from a completed optimisation run.

        Reads observations, deduplicates by matched SMILES, and returns
        the best candidates with CReM metadata attached.
        """
        state = self._load_state(run_id)
        observations = read_jsonl(self._paths(run_id).observations)

        if not observations:
            return {"run_id": run_id, "candidates": [], "message": "No observations yet."}

        is_max = state["objective"] == "max"
        sorted_obs = sorted(
            observations,
            key=lambda o: o.get("y", float("-inf") if is_max else float("inf")),
            reverse=is_max,
        )

        # Deduplicate by matched_smiles
        seen: set[str] = set()
        top_candidates: list[dict[str, Any]] = []
        for obs in sorted_obs:
            smi = obs.get("matched_smiles") or obs.get("x", {}).get(
                state.get("smiles_direct", {}).get("smiles_column", ""), ""
            )
            if not smi or smi in seen:
                continue
            seen.add(smi)
            top_candidates.append(obs)
            if len(top_candidates) >= top_k:
                break

        # Load CReM metadata if available
        crem_meta: dict[str, dict[str, Any]] = {}
        crem_path = self._paths(run_id).crem_candidates
        if crem_path.exists():
            crem_meta = read_json(crem_path)

        results = []
        for rank, obs in enumerate(top_candidates, 1):
            smi = obs.get("matched_smiles", "")
            entry: dict[str, Any] = {
                "rank": rank,
                "smiles": smi,
                "predicted_target": obs.get("y"),
                "descriptor_distance": obs.get("discovery_distance"),
            }
            cm = crem_meta.get(smi, {})
            entry["mutation_type"] = cm.get("mutation_type", "unknown")
            entry["parent_smiles"] = cm.get("parent_smiles", "")
            entry["ring_delta"] = cm.get("ring_delta", 0)
            results.append(entry)

        return {
            "run_id": run_id,
            "top_k": top_k,
            "candidates": results,
        }

    def validate_candidates(
        self,
        run_id: str,
        *,
        method: str,
        top_k: int = 10,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Validate top-K candidates using the chosen method.

        Args:
            method: ``"feasibility_only"`` or ``"xtb_plus_feasibility"``.
            top_k: Number of top candidates to validate.

        Returns validated and re-ranked candidates with full data.
        """
        if method not in {"feasibility_only", "xtb_plus_feasibility"}:
            raise ValueError("method must be 'feasibility_only' or 'xtb_plus_feasibility'")

        state = self._load_state(run_id)
        top_data = self.get_top_candidates(run_id, top_k=top_k, verbose=verbose)
        candidates = top_data.get("candidates", [])

        if not candidates:
            return {"run_id": run_id, "method": method, "candidates": []}

        from .molecular.energy import _estimate_strain_energy
        from .molecular.feasibility import assess_feasibility

        sa_threshold = state.get("scaffold_edit", {}).get("sa_threshold", 6.0)

        if method == "xtb_plus_feasibility":
            from .molecular.energy import EnergyCache, get_energy_features

            paths = self._paths(run_id)
            cache = EnergyCache(path=paths.energy_cache, method="XTB")

        results = []
        for cand in candidates:
            smi = cand["smiles"]
            entry = dict(cand)

            if method == "xtb_plus_feasibility":
                try:
                    energy = get_energy_features(smi, cache, backend="xtb")
                except Exception:
                    energy = {}
                entry["homo_ev"] = energy.get("homo_ev")
                entry["lumo_ev"] = energy.get("lumo_ev")
                entry["gap_ev"] = energy.get("gap_ev")
                entry["dipole_debye"] = energy.get("dipole_debye")
                strain = energy.get("strain_energy_kcal")
            else:
                try:
                    strain = _estimate_strain_energy(smi)
                except Exception:
                    strain = None

            try:
                feas = assess_feasibility(
                    smi, sa_threshold=sa_threshold, strain_energy_kcal=strain,
                )
            except Exception:
                feas = None

            if feas is not None:
                entry["sa_score"] = feas.sa_score
                entry["strain_energy_kcal"] = feas.strain_energy_kcal
                entry["is_feasible"] = feas.is_feasible
                entry["penalty"] = feas.penalty
                entry["feasibility_reasons"] = feas.reasons
                pred = cand.get("predicted_target", 0.0) or 0.0
                if state["objective"] == "max":
                    entry["adjusted_target"] = pred - feas.penalty
                else:
                    entry["adjusted_target"] = pred + feas.penalty
            else:
                entry["adjusted_target"] = cand.get("predicted_target")

            results.append(entry)

        if method == "xtb_plus_feasibility":
            cache.save()

        # Re-rank by adjusted target
        is_max = state["objective"] == "max"
        results.sort(
            key=lambda r: r.get("adjusted_target", float("-inf") if is_max else float("inf")),
            reverse=is_max,
        )
        for i, r in enumerate(results, 1):
            r["rank"] = i

        output = {
            "run_id": run_id,
            "validation_method": method,
            "top_k": top_k,
            "candidates": results,
        }
        write_json(self._paths(run_id).validation_results, output)

        # Update state
        state["scaffold_edit"]["validation_method"] = method
        state["updated_at"] = utc_now_iso()
        self._save_state(run_id, state)

        self._log(
            verbose,
            f"[validate] {len(results)} candidates validated via {method}",
        )
        return output
