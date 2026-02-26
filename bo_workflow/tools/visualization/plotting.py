from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from bo_workflow.utils import Objective

type ErrorStyle = Literal["stderr", "iqr"]
type YScale = Literal["linear", "log"]


def _as_2d(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D data, got shape={arr.shape}")
    return arr


def _cumulative_best(data: np.ndarray, objective: Objective) -> np.ndarray:
    if objective == "min":
        return np.minimum.accumulate(data, axis=1)
    if objective == "max":
        return np.maximum.accumulate(data, axis=1)
    raise ValueError(f"Unsupported objective={objective!r}")


def _to_regret(
    values: np.ndarray,
    objective: Objective,
    regret_baseline: float | None,
) -> np.ndarray:
    if regret_baseline is None:
        return values
    if objective == "min":
        return values - regret_baseline
    return regret_baseline - values


def _summary_band(
    data: np.ndarray, error_style: ErrorStyle
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(data, axis=0)
    if error_style == "stderr":
        spread = np.std(data, axis=0) / np.sqrt(data.shape[0])
        return center, center - spread, center + spread
    if error_style == "iqr":
        lower = np.percentile(data, 25, axis=0)
        upper = np.percentile(data, 75, axis=0)
        return center, lower, upper
    raise ValueError(f"Unsupported error_style={error_style!r}")


def _clip_for_log(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(data, eps)


def plot_optimization_convergence(
    methods_data: Mapping[str, np.ndarray],
    *,
    title: str = "Optimization Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Best Value Found",
    objective: Objective = "min",
    regret_baseline: float | None = None,
    y_scale: YScale = "linear",
    show_points: bool = False,
    point_alpha: float = 0.15,
    error_style: ErrorStyle = "stderr",
    fig_path: str | None = None,
    show: bool = False,
    ax: Axes | None = None,
) -> Figure:
    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#454645"]

    for idx, (label, data) in enumerate(methods_data.items()):
        series = _as_2d(data)
        iters = np.arange(series.shape[1], dtype=int)

        transformed_raw = _to_regret(series, objective, regret_baseline)
        transformed_best = _to_regret(
            _cumulative_best(series, objective), objective, regret_baseline
        )
        if y_scale == "log":
            transformed_raw = _clip_for_log(transformed_raw)
            transformed_best = _clip_for_log(transformed_best)

        mean_line, lower_band, upper_band = _summary_band(transformed_best, error_style)
        color = colors[idx % len(colors)]

        if show_points:
            ax.plot(
                iters,
                transformed_raw.T,
                ".",
                color=color,
                alpha=point_alpha,
                markersize=3,
                markeredgewidth=0,
            )
            ax.plot([], [], ".", color=color, alpha=0.6, label=f"{label} (Iter)")

        ax.plot(iters, mean_line, lw=2.5, color=color, label=f"{label} (Best)")
        ax.fill_between(iters, lower_band, upper_band, color=color, alpha=0.2)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_yscale(y_scale)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(
        loc="upper right", frameon=True, shadow=True, ncol=2 if show_points else 1
    )

    fig.tight_layout()

    if fig_path is not None:
        path = Path(fig_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")

    if show and created_fig:
        plt.show()
    elif created_fig:
        plt.close(fig)

    return fig
