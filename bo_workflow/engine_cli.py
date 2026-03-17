"""CLI subcommands for the BO engine: init, suggest, observe, status, report."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .engine import BOEngine


def _json_print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parse_json_object(value: str) -> dict[str, Any]:
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return dict(payload)


def _parse_search_space_spec(value: str) -> dict[str, Any]:
    payload = _parse_json_object(value)
    return payload


def _parse_observation_records(value: str) -> list[dict[str, Any]]:
    path = Path(value)
    if path.exists():
        if path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError("JSON observation payload must be a list of objects.")
            return [dict(x) for x in payload]

        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            if "y" not in frame.columns:
                raise ValueError("CSV observations must include a 'y' column.")
            rows: list[dict[str, Any]] = []
            for _, row in frame.iterrows():
                x = row.drop(labels=["y"]).to_dict()
                rows.append({"x": x, "y": float(row["y"])})
            return rows

    payload = json.loads(value)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Inline observation payload must be JSON object or list.")
    return [dict(x) for x in payload]


def _parse_simplex_groups(raw: list[str] | None) -> list[dict[str, Any]]:
    """Parse --simplex-groups 'col1,col2,col3:100' into constraint dicts."""
    if not raw:
        return []
    result = []
    for entry in raw:
        if ":" not in entry:
            raise ValueError(
                f"--simplex-groups entry '{entry}' must be 'col1,col2,...:total' "
                "(columns separated by commas, then colon, then numeric total)."
            )
        cols_part, total_part = entry.rsplit(":", 1)
        cols = [c.strip() for c in cols_part.split(",") if c.strip()]
        if len(cols) < 2:
            raise ValueError(
                f"--simplex-groups entry '{entry}' must list at least 2 columns."
            )
        try:
            total = float(total_part.strip())
        except ValueError as exc:
            raise ValueError(
                f"--simplex-groups total '{total_part}' is not a valid number."
            ) from exc
        result.append({"type": "simplex", "cols": cols, "total": total})
    return result


def register_commands(sub: argparse._SubParsersAction) -> None:
    """Register engine subcommands on an existing subparsers group."""
    init_cmd = sub.add_parser(
        "init", help="Initialize run from a dataset or explicit search-space spec"
    )
    init_input = init_cmd.add_mutually_exclusive_group(required=True)
    init_input.add_argument("--dataset", type=Path)
    init_input.add_argument(
        "--search-space-json",
        type=str,
        help="JSON object or path to JSON file describing design_parameters and fixed_features",
    )
    init_cmd.add_argument("--target", type=str, required=True)
    init_cmd.add_argument(
        "--objective", type=str, choices=["min", "max"], required=True
    )
    init_cmd.add_argument("--run-id", type=str, default=None)
    init_cmd.add_argument("--seed", type=int, default=7)
    init_cmd.add_argument(
        "--engine",
        type=str,
        choices=["hebo", "bo_lcb", "random", "botorch"],
        default="hebo",
        help="Default optimizer engine for this run",
    )
    init_cmd.add_argument(
        "--hebo-model",
        type=str,
        choices=["gp", "rf"],
        default="gp",
        help="Surrogate model used inside the HEBO optimizer",
    )
    init_cmd.add_argument("--init-random", type=int, default=10)
    init_cmd.add_argument("--batch-size", type=int, default=1)
    init_cmd.add_argument("--max-categories", type=int, default=None)
    init_cmd.add_argument(
        "--drop-cols",
        type=str,
        default=None,
        help="Comma-separated column names to exclude from the feature space (e.g. rxn_smiles)",
    )
    init_cmd.add_argument(
        "--simplex-groups",
        type=str,
        action="append",
        default=None,
        metavar="COLS:TOTAL",
        help=(
            "Declare a simplex constraint: comma-separated column names followed by "
            "a colon and the required sum (e.g. 'Metal_1,Metal_2,Metal_3:100'). "
            "Repeat the flag for multiple groups."
        ),
    )
    init_cmd.add_argument(
        "--intent-json",
        type=str,
        default=None,
        help="Optional JSON object or path to JSON object for original user intent",
    )
    init_cmd.add_argument("--verbose", action="store_true")

    suggest_cmd = sub.add_parser("suggest", help="Suggest next experimental candidates")
    suggest_cmd.add_argument("--run-id", type=str, required=True)
    suggest_cmd.add_argument("--batch-size", type=int, default=None)
    suggest_cmd.add_argument("--verbose", action="store_true")

    observe_cmd = sub.add_parser("observe", help="Record observation(s)")
    observe_cmd.add_argument("--run-id", type=str, required=True)
    observe_cmd.add_argument(
        "--data",
        type=str,
        required=True,
        help="Observations as JSON string/object/list, or path to CSV/JSON file",
    )
    observe_cmd.add_argument("--verbose", action="store_true")

    status_cmd = sub.add_parser("status", help="Show run status")
    status_cmd.add_argument("--run-id", type=str, required=True)

    report_cmd = sub.add_parser("report", help="Generate report and plot")
    report_cmd.add_argument("--run-id", type=str, required=True)
    report_cmd.add_argument(
        "--nn-snap",
        nargs="?",
        const="__dataset__",
        default=None,
        help=(
            "Map each BO suggestion to the nearest real catalog entry. "
            "Without a value, uses the init dataset from state.json. "
            "Optionally pass a CSV path to override the catalog."
        ),
    )
    report_cmd.add_argument(
        "--nn-snap-k",
        type=int,
        default=20,
        help="Number of top entries to show in nn-snap report (default 20)",
    )
    report_cmd.add_argument("--verbose", action="store_true")


def _nn_snap_report(
    engine: BOEngine,
    run_id: str,
    report: dict[str, Any],
    catalog_path: str,
    top_k: int,
) -> dict[str, Any]:
    """Map every BO observation to its nearest real catalog entry.

    Deduplicates by catalog row index and returns the top-k unique entries
    ranked by true target value (best first).
    """
    import numpy as np

    from .converters.catalog_index import CatalogIndex
    from .utils import read_jsonl

    state = engine._load_state(run_id)
    catalog_df = pd.read_csv(catalog_path)
    target_col = state["target_column"]
    active_features = list(state["active_features"])
    ascending = state.get("objective") == "min"

    present_feature_cols = [c for c in active_features if c in catalog_df.columns]
    if not present_feature_cols:
        report["nn_snap_error"] = "No active features found in catalog"
        return report

    # Euclidean nearest-neighbor indexing requires numeric feature columns.
    feature_cols = [
        c for c in present_feature_cols if pd.api.types.is_numeric_dtype(catalog_df[c])
    ]
    dropped_non_numeric = [
        c for c in present_feature_cols if c not in set(feature_cols)
    ]
    if not feature_cols:
        report["nn_snap_error"] = (
            "No numeric active features available for Euclidean nn-snap indexing"
        )
        report["nn_snap_non_numeric_features"] = dropped_non_numeric
        return report

    catalog_df["_orig_idx"] = range(len(catalog_df))
    index = CatalogIndex(catalog_df, feature_cols, metric="euclidean")

    observations = read_jsonl(engine._paths(run_id).observations)
    if not observations:
        return report

    # Map every observation → 1-NN catalog entry, deduplicate by _orig_idx
    best_per_entry: dict[int, dict[str, Any]] = {}
    for obs in observations:
        x = obs.get("x", {})
        try:
            vec = np.array(
                [float(x.get(c, 0)) for c in feature_cols], dtype=np.float32
            )
            nn = index.query(vec, k=1)
            if not nn:
                continue
            row = nn[0]
            cat_idx = int(row["_orig_idx"])
            true_val = float(row.get(target_col, float("nan")))
            proxy_y = float(obs["y"])
            nn_dist = float(row.get("distance", float("nan")))
            iteration = obs.get("iteration")

            if cat_idx not in best_per_entry:
                best_per_entry[cat_idx] = {
                    "catalog_index": cat_idx,
                    "true_yield": true_val,
                    "best_proxy_yield": proxy_y,
                    "min_nn_distance": nn_dist,
                    "first_hit_iteration": iteration,
                    "hit_count": 1,
                }
            else:
                rec = best_per_entry[cat_idx]
                rec["hit_count"] += 1
                if (not ascending and proxy_y > rec["best_proxy_yield"]) or (
                    ascending and proxy_y < rec["best_proxy_yield"]
                ):
                    rec["best_proxy_yield"] = proxy_y
                    rec["first_hit_iteration"] = iteration
                if nn_dist < rec["min_nn_distance"]:
                    rec["min_nn_distance"] = nn_dist
        except Exception:
            continue

    # Rank by true yield (best first)
    unique_entries = sorted(
        best_per_entry.values(),
        key=lambda e: e["true_yield"],
        reverse=not ascending,
    )
    top_entries = unique_entries[:top_k]

    report["nn_snap"] = {
        "catalog": catalog_path,
        "catalog_rows": len(catalog_df),
        "feature_columns_used": feature_cols,
        "unique_entries_visited": len(unique_entries),
        "top_k": top_k,
        "entries": top_entries,
    }
    if dropped_non_numeric:
        report["nn_snap_warning"] = (
            "Dropped non-numeric active features for Euclidean nn-snap"
        )
        report["nn_snap_non_numeric_features"] = dropped_non_numeric
    return report


def handle(args: argparse.Namespace, engine: BOEngine) -> int | None:
    """Handle an engine subcommand. Returns exit code, or None if not ours."""
    if args.command == "init":
        intent_payload = None
        if args.intent_json is not None:
            intent_payload = _parse_json_object(args.intent_json)
        search_space_spec = None
        if args.search_space_json is not None:
            search_space_spec = _parse_search_space_spec(args.search_space_json)
        drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()] if args.drop_cols else []
        if search_space_spec is not None:
            if drop_cols:
                raise ValueError("--drop-cols is only supported with --dataset init.")
            if args.max_categories is not None:
                raise ValueError(
                    "--max-categories is only supported with dataset-backed init."
                )
        constraints = _parse_simplex_groups(args.simplex_groups)
        payload = engine.init_run(
            dataset_path=args.dataset,
            search_space_spec=search_space_spec,
            target_column=args.target,
            objective=args.objective,
            default_engine=args.engine,
            hebo_model=args.hebo_model,
            run_id=args.run_id,
            seed=args.seed,
            num_initial_random_samples=args.init_random,
            default_batch_size=args.batch_size,
            max_categories=args.max_categories if args.max_categories is not None else 64,
            drop_cols=drop_cols,
            constraints=constraints or None,
            intent=intent_payload,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "suggest":
        payload = engine.suggest(
            args.run_id,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )
        _json_print(payload)
        return 0

    if args.command == "observe":
        observations = _parse_observation_records(args.data)
        payload = engine.observe(args.run_id, observations, verbose=args.verbose)
        _json_print(payload)
        return 0

    if args.command == "status":
        payload = engine.status(args.run_id)
        _json_print(payload)
        return 0

    if args.command == "report":
        payload = engine.report(args.run_id, verbose=args.verbose)

        nn_snap_path = getattr(args, "nn_snap", None)
        if nn_snap_path:
            # Resolve sentinel: use the init dataset from state.json
            if nn_snap_path == "__dataset__":
                state = engine._load_state(args.run_id)
                nn_snap_path = state.get("dataset_path")
                if not nn_snap_path:
                    import sys

                    print(
                        "[nn-snap] No dataset_path in state.json; "
                        "pass an explicit CSV path with --nn-snap <path>",
                        file=sys.stderr,
                    )
                    _json_print(payload)
                    return 0
            top_k = getattr(args, "nn_snap_k", 20)
            payload = _nn_snap_report(
                engine, args.run_id, payload, nn_snap_path, top_k
            )

        _json_print(payload)
        return 0

    return None
