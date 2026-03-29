from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import Objective


def save_run_convergence_plot(
    values: list[float],
    *,
    objective: Objective,
    output_path: Path,
    title: str = "Optimization Convergence",
) -> None:
    if not values:
        return

    raw = np.asarray(values, dtype=float)
    if objective == "min":
        best = np.minimum.accumulate(raw)
    else:
        best = np.maximum.accumulate(raw)

    iterations = np.arange(1, len(raw) + 1, dtype=int)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        iterations,
        raw,
        "o-",
        alpha=0.35,
        linewidth=1.5,
        markersize=4,
        label="Observed",
    )
    ax.plot(iterations, best, "-", linewidth=2.5, label="Best-so-far")
    ax.set_title(title)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Objective value")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
