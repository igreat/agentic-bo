"""Constraint layer — search-space constraints enforced at suggest time."""

from .base import Constraint
from .simplex import SimplexConstraint

__all__ = ["Constraint", "SimplexConstraint"]


def load_constraints(state: dict) -> list[Constraint]:
    """Deserialize constraints from a run state dict."""
    constraints: list[Constraint] = []
    for spec in state.get("constraints", []):
        ctype = spec.get("type")
        if ctype == "simplex":
            constraints.append(SimplexConstraint.from_dict(spec))
        else:
            raise ValueError(f"Unknown constraint type: '{ctype}'")
    return constraints
