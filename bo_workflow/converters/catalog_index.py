"""Encoding-agnostic nearest-neighbor index for catalog lookup.

Supports any feature column naming convention and multiple distance metrics.
Works with VAE latent vectors (z0..z31), DRFP fingerprints (fp_0..fp_127),
raw numeric descriptors, or any mix of numeric feature columns.

Usage::

    # Build from explicit columns
    index = CatalogIndex(catalog_df, ["z0", "z1", ..., "z31"])

    # Auto-detect feature columns from naming patterns
    index = CatalogIndex.auto_detect(catalog_df)

    # Query nearest neighbor
    results = index.query(feature_vector, k=3)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

_Z_COL_RE = re.compile(r"^z(\d+)$")
_FP_COL_RE = re.compile(r"^fp_(\d+)$")


def _to_json_value(value: object) -> object:
    """Convert numpy/pandas scalar values into JSON-serializable Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sorted_pattern_cols(columns: list[str], pattern: re.Pattern) -> list[str]:
    """Return columns matching *pattern* sorted by their numeric suffix."""
    matched: list[tuple[int, str]] = []
    for col in columns:
        m = pattern.fullmatch(col)
        if m is not None:
            matched.append((int(m.group(1)), col))
    matched.sort(key=lambda item: item[0])
    return [col for _, col in matched]


class CatalogIndex:
    """Pre-built spatial index for nearest-neighbor lookup in any feature space.

    Parameters
    ----------
    catalog : pd.DataFrame
        The catalog to search.  Must contain all *feature_columns* plus any
        metadata columns to return with results.
    feature_columns : list[str]
        Columns that define the feature space for distance computation.
    metric : str
        Distance metric: ``"euclidean"`` (default) or ``"tanimoto"``.
        Euclidean uses ``scipy.spatial.KDTree`` for O(log n) queries.
        Tanimoto uses brute-force (binary vectors are not KDTree-friendly).
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        feature_columns: list[str],
        *,
        metric: str = "euclidean",
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns must not be empty")

        missing = [c for c in feature_columns if c not in catalog.columns]
        if missing:
            raise ValueError(f"Feature columns not in catalog: {missing}")

        if metric not in ("euclidean", "tanimoto"):
            raise ValueError(f"Unsupported metric: {metric!r}. Use 'euclidean' or 'tanimoto'.")

        self._catalog = catalog
        self._feature_columns = list(feature_columns)
        self._metric = metric
        self._meta_columns = [c for c in catalog.columns if c not in set(feature_columns)]

        feature_matrix = catalog[feature_columns].values

        if metric == "euclidean":
            self._matrix = feature_matrix.astype(np.float32)
            from scipy.spatial import KDTree

            self._tree: Any = KDTree(self._matrix)
        else:
            # Tanimoto: store as uint8 for binary ops
            self._matrix = feature_matrix.astype(np.uint8)
            self._tree = None

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def metric(self) -> str:
        return self._metric

    @property
    def size(self) -> int:
        return len(self._catalog)

    def query(self, feature_vector: np.ndarray, k: int = 1) -> list[dict]:
        """Find k nearest catalog entries.

        Returns a list of dicts with keys: ``rank``, ``distance`` (or
        ``similarity`` for Tanimoto), plus all non-feature columns from
        the catalog row.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")

        if feature_vector.ndim != 1:
            raise ValueError("feature_vector must be 1-D")
        if len(feature_vector) != len(self._feature_columns):
            raise ValueError(
                f"Feature vector length {len(feature_vector)} != "
                f"expected {len(self._feature_columns)}"
            )

        if self._metric == "euclidean":
            return self._query_euclidean(feature_vector, k)
        else:
            return self._query_tanimoto(feature_vector, k)

    def _query_euclidean(self, vec: np.ndarray, k: int) -> list[dict]:
        q = vec.astype(np.float32)
        k_actual = min(k, self.size)
        distances, indices = self._tree.query(q, k=k_actual)

        # KDTree returns scalar when k=1
        if k_actual == 1:
            distances = [distances]
            indices = [indices]

        results: list[dict] = []
        for rank, (dist, idx) in enumerate(zip(distances, indices)):
            entry: dict = {
                "rank": rank + 1,
                "distance": round(float(dist), 4),
            }
            row = self._catalog.iloc[idx]
            for col in self._catalog.columns:
                entry[col] = _to_json_value(row[col])
            results.append(entry)
        return results

    def _query_tanimoto(self, vec: np.ndarray, k: int) -> list[dict]:
        q = vec.astype(bool)
        catalog_bool = self._matrix.astype(bool)

        intersection = np.sum(catalog_bool & q[np.newaxis, :], axis=1)
        union = np.sum(catalog_bool | q[np.newaxis, :], axis=1)
        similarities = np.where(union > 0, intersection / union, 0.0)

        k_actual = min(k, self.size)
        top_k_idx = np.argsort(similarities)[::-1][:k_actual]

        results: list[dict] = []
        for rank, idx in enumerate(top_k_idx):
            entry: dict = {
                "rank": rank + 1,
                "similarity": round(float(similarities[idx]), 4),
            }
            row = self._catalog.iloc[idx]
            for col in self._catalog.columns:
                entry[col] = _to_json_value(row[col])
            results.append(entry)
        return results

    @classmethod
    def auto_detect(
        cls,
        catalog: pd.DataFrame,
        metric: str | None = None,
    ) -> "CatalogIndex":
        """Auto-detect feature columns and metric from catalog column names.

        Detection priority:
        1. ``fp_<n>`` columns → tanimoto (DRFP fingerprint)
        3. All numeric columns → euclidean (raw descriptors)
        """
        cols = list(catalog.columns)

        z_cols = _sorted_pattern_cols(cols, _Z_COL_RE)
        if z_cols:
            return cls(catalog, z_cols, metric=metric or "euclidean")

        fp_cols = _sorted_pattern_cols(cols, _FP_COL_RE)
        if fp_cols:
            return cls(catalog, fp_cols, metric=metric or "tanimoto")

        # Fallback: all numeric columns
        numeric_cols = [
            c for c in cols if pd.api.types.is_numeric_dtype(catalog[c])
        ]
        if not numeric_cols:
            raise ValueError(
                "Cannot auto-detect feature columns: no z<n>, fp_<n>, "
                "or numeric columns found."
            )
        return cls(catalog, numeric_cols, metric=metric or "euclidean")
