"""SimplexConstraint — compositional variables that must sum to a fixed total.

Enforcement strategy: post-hoc normalization.
  After the optimizer suggests raw candidates, each simplex group is rescaled
  so that the group columns sum to ``total``. This keeps feasibility at output
  time without modifying the design space or the optimizer's internal model.

Example state.json entry::

    {
        "type": "simplex",
        "cols": ["Metal_1_Proportion", "Metal_2_Proportion", "Metal_3_Proportion"],
        "total": 100.0
    }
"""

import math

import pandas as pd

from .base import Constraint


class SimplexConstraint(Constraint):
    """Enforces that a group of columns sum to ``total`` in every suggestion."""

    def __init__(self, cols: list[str], total: float = 1.0) -> None:
        if len(cols) < 2:
            raise ValueError("SimplexConstraint requires at least 2 columns.")
        if len(set(cols)) != len(cols):
            raise ValueError("SimplexConstraint columns must be unique.")
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"SimplexConstraint total must be a finite positive number, got {total}.")
        self.cols = list(cols)
        self.total = float(total)

    def apply(self, suggestions: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.cols if c not in suggestions.columns]
        if missing:
            raise ValueError(
                f"SimplexConstraint columns not found in suggestions: {missing}"
            )
        suggestions = suggestions.copy()
        # Clip negatives first — proportions must be non-negative.
        suggestions[self.cols] = suggestions[self.cols].clip(lower=0.0)
        row_sums = suggestions[self.cols].sum(axis=1)
        zero_mask = row_sums == 0
        if zero_mask.any():
            # All-zero rows (including all-negative-then-clipped): distribute uniformly.
            for col in self.cols:
                suggestions.loc[zero_mask, col] = self.total / len(self.cols)
            row_sums = suggestions[self.cols].sum(axis=1)
        scale = self.total / row_sums
        for col in self.cols:
            suggestions[col] = suggestions[col] * scale
        return suggestions

    def to_dict(self) -> dict:
        return {"type": "simplex", "cols": self.cols, "total": self.total}

    @classmethod
    def from_dict(cls, spec: dict) -> "SimplexConstraint":
        return cls(cols=list(spec["cols"]), total=float(spec.get("total", 1.0)))
