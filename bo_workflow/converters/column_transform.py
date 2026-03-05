"""Column-level data transformations for BO input/target preparation.

Supports two workflows:

1. ``profile``: Analyse columns and recommend transforms based on distribution
   statistics (skewness, range, zeros, negatives). Use this first to decide
   whether a transform is warranted.

2. ``transform``: Apply a named transform to specified columns. Transformed
   columns are added with a prefix (e.g. ``ic50_nM`` → ``log10_ic50_nM``).
   The original column is dropped by default (``--keep-original`` retains it).

Supported transforms
--------------------
log10      : log10(x)         — strictly positive, wide dynamic range (>=3 decades)
log1p      : log10(1 + x)     — non-negative including zeros, right-skewed
neglog10   : -log10(x)        — strictly positive, converts IC50/Ki/Kd → pIC50/pKi/pKd
sqrt       : sqrt(x)          — non-negative, moderate right skew
standardize: (x - mean) / std — any sign, roughly normal; centres and scales
minmax     : (x - min) / (max - min) — bounded to [0, 1]; use when range matters

Usage
-----
# Profile columns to see what transforms are recommended
uv run python -m bo_workflow.converters.column_transform profile \\
    --input data/egfr_ic50.csv \\
    --cols ic50_nM

# Apply a transform
uv run python -m bo_workflow.converters.column_transform transform \\
    --input data/egfr_ic50.csv \\
    --cols ic50_nM \\
    --transform log10 \\
    --output data/egfr_log.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_TRANSFORMS = ("log10", "log1p", "neglog10", "sqrt", "standardize", "minmax")

_PREFIXES = {
    "log10": "log10",
    "log1p": "log1p",
    "neglog10": "neg_log10",
    "sqrt": "sqrt",
    "standardize": "std",
    "minmax": "mm",
}


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def profile_column(series: pd.Series, name: str) -> dict[str, Any]:
    """Compute distribution stats and recommend a transform for one column."""
    numeric = pd.to_numeric(series, errors="coerce")
    n_total = len(numeric)
    n_null = int(numeric.isna().sum())
    numeric = numeric.dropna()

    if numeric.empty:
        return {"column": name, "error": "no numeric values after coercion"}

    n = len(numeric)
    n_zero = int((numeric == 0).sum())
    n_neg = int((numeric < 0).sum())
    mn = float(numeric.min())
    mx = float(numeric.max())
    # scipy skew returns NaN for n < 3; std is NaN for n < 2
    skewness = float(scipy_stats.skew(numeric)) if n >= 3 else 0.0
    if not np.isfinite(skewness):
        skewness = 0.0

    log_range: float | None = None
    if mn > 0:
        log_range = float(np.log10(mx) - np.log10(mn))

    # Decision tree for recommendation
    rec: str | None
    if n_neg > 0:
        if abs(skewness) > 1.5:
            rec = "standardize"
            reason = (
                f"Contains {n_neg} negative values; standardize rescales without "
                "requiring positivity."
            )
        else:
            rec = None
            reason = (
                f"Contains {n_neg} negative values with low skewness ({skewness:.2f}); "
                "no transform recommended."
            )
    elif n_zero > 0:
        frac_zero = n_zero / n
        if skewness > 1.0:
            rec = "log1p"
            reason = (
                f"Contains zeros ({frac_zero:.1%}) and is right-skewed "
                f"(skewness={skewness:.2f}); log1p handles zeros."
            )
        else:
            rec = None
            reason = (
                f"Contains zeros ({frac_zero:.1%}) but low skewness ({skewness:.2f}); "
                "no transform recommended."
            )
    elif log_range is not None and log_range >= 3:
        rec = "log10"
        reason = (
            f"All positive, spans {log_range:.1f} orders of magnitude; "
            "log10 compresses the scale to improve GP / tree model accuracy."
        )
    elif skewness > 2.0:
        rec = "log10"
        reason = (
            f"All positive, strong right skew (skewness={skewness:.2f}); "
            "log10 recommended."
        )
    elif skewness > 1.0:
        rec = "sqrt"
        reason = (
            f"Moderate right skew (skewness={skewness:.2f}); "
            "sqrt may improve model accuracy."
        )
    else:
        rec = None
        reason = (
            f"Low skewness ({skewness:.2f}); no transform recommended."
        )

    result: dict[str, Any] = {
        "column": name,
        "count": n,
        "null_count": n_total - n,
        "min": mn,
        "max": mx,
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std()) if n >= 2 else 0.0,
        "skewness": skewness,
        "fraction_zero": n_zero / n,
        "fraction_negative": n_neg / n,
    }
    if log_range is not None:
        result["log_range_decades"] = round(log_range, 2)
    result["recommended_transform"] = rec
    result["reason"] = reason
    return result


def cmd_profile(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input)
    cols = args.cols if args.cols else list(df.columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(
            json.dumps({"error": f"columns not found: {missing}"}),
            file=sys.stderr,
        )
        return 1
    profiles = [profile_column(df[c], c) for c in cols]
    print(json.dumps(profiles if len(profiles) > 1 else profiles[0], indent=2))
    return 0


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def apply_transform(
    series: pd.Series,
    transform: str,
    name: str,
) -> tuple[pd.Series, str]:
    """Apply a named transform to a Series. Returns (result, new_column_name)."""
    numeric = pd.to_numeric(series, errors="coerce")
    n_non_numeric = int(series.notna().sum()) - int(numeric.notna().sum())
    if n_non_numeric > 0:
        raise ValueError(
            f"'{name}': {n_non_numeric} non-numeric values cannot be coerced. "
            "Pass a numeric column (e.g. not a SMILES column)."
        )
    prefix = _PREFIXES[transform]
    new_name = f"{prefix}_{name}"

    if transform == "log10":
        if (numeric <= 0).any():
            n_bad = int((numeric <= 0).sum())
            raise ValueError(
                f"'{name}': log10 requires all values > 0 ({n_bad} non-positive found)."
            )
        result = np.log10(numeric)

    elif transform == "log1p":
        if (numeric < 0).any():
            n_bad = int((numeric < 0).sum())
            raise ValueError(
                f"'{name}': log1p requires all values >= 0 ({n_bad} negative found)."
            )
        result = np.log10(1.0 + numeric)

    elif transform == "neglog10":
        if (numeric <= 0).any():
            n_bad = int((numeric <= 0).sum())
            raise ValueError(
                f"'{name}': neglog10 requires all values > 0 ({n_bad} non-positive found)."
            )
        result = -np.log10(numeric)

    elif transform == "sqrt":
        if (numeric < 0).any():
            n_bad = int((numeric < 0).sum())
            raise ValueError(
                f"'{name}': sqrt requires all values >= 0 ({n_bad} negative found)."
            )
        result = np.sqrt(numeric)

    elif transform == "standardize":
        mu = float(numeric.mean())
        sigma = float(numeric.std())
        if np.isclose(sigma, 0.0):
            raise ValueError(f"'{name}': standardize requires non-zero std (column is constant).")
        result = (numeric - mu) / sigma

    elif transform == "minmax":
        mn = float(numeric.min())
        mx = float(numeric.max())
        if np.isclose(mn, mx):
            raise ValueError(f"'{name}': minmax requires a non-constant column.")
        result = (numeric - mn) / (mx - mn)

    else:
        raise ValueError(f"Unknown transform '{transform}'. Choose from: {', '.join(_TRANSFORMS)}")

    return pd.Series(result.values, index=series.index, name=new_name), new_name


def cmd_transform(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input)
    missing = [c for c in args.cols if c not in df.columns]
    if missing:
        print(json.dumps({"error": f"columns not found: {missing}"}), file=sys.stderr)
        return 1

    new_names: list[str] = []
    for col in args.cols:
        try:
            transformed, new_name = apply_transform(df[col], args.transform, col)
        except ValueError as exc:
            print(json.dumps({"error": str(exc)}), file=sys.stderr)
            return 1

        # Insert transformed column right after the original
        idx = df.columns.get_loc(col)
        df.insert(idx + 1, new_name, transformed)
        if not args.keep_original:
            df = df.drop(columns=[col])
        new_names.append(new_name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "rows": len(df),
                "transformed_columns": new_names,
                "transform": args.transform,
                "kept_original": args.keep_original,
            },
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bo_workflow.converters.column_transform",
        description="Profile columns and apply data transforms for BO preparation.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    profile_cmd = sub.add_parser("profile", help="Analyse columns and recommend transforms")
    profile_cmd.add_argument("--input", type=Path, required=True, help="Input CSV path")
    profile_cmd.add_argument(
        "--cols",
        nargs="+",
        default=None,
        help="Columns to profile (default: all columns)",
    )

    transform_cmd = sub.add_parser("transform", help="Apply a named transform to columns")
    transform_cmd.add_argument("--input", type=Path, required=True, help="Input CSV path")
    transform_cmd.add_argument(
        "--cols", nargs="+", required=True, help="Columns to transform"
    )
    transform_cmd.add_argument(
        "--transform",
        type=str,
        required=True,
        choices=_TRANSFORMS,
        help="Transform to apply",
    )
    transform_cmd.add_argument(
        "--output", type=str, required=True, help="Output CSV path"
    )
    transform_cmd.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep the original column alongside the transformed one",
    )

    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.subcommand == "profile":
        return cmd_profile(args)
    return cmd_transform(args)


if __name__ == "__main__":
    raise SystemExit(main())