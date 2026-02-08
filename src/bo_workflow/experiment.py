from pathlib import Path

import numpy as np

from src.bo_workflow.config import ExperimentConfig
from src.bo_workflow.plotting import plot_optimization_convergence
from src.bo_workflow.problems.base import ProblemContext
from src.bo_workflow.runner import (
    load_result,
    run_bo_lcb,
    run_hebo,
    run_random_search,
    save_result,
)
from src.bo_workflow.type_defs import ExperimentResult


def _mean_final_best(history: np.ndarray) -> float:
    finals: list[float] = []
    for trace in history:
        arr = np.asarray(trace, dtype=float)
        if arr.size == 0:
            continue
        finals.append(float(np.min(arr)))
    return float(np.mean(finals)) if finals else float("nan")


def run_experiment(config: ExperimentConfig, problem: ProblemContext) -> ExperimentResult:
    if config.result_path is not None and Path(config.result_path).exists():
        results = load_result(config.result_path)
    else:
        random_seeds = list(range(config.num_seeds))

        print("\n[1/3] Running HEBO Optimization...")
        hebo_history = run_hebo(
            problem=problem,
            num_iterations=config.num_iterations,
            random_seeds=random_seeds,
            num_initial_random_samples=config.num_initial_random_samples,
        )

        print("\n[2/3] Running Basic BO (LCB)...")
        bo_history = run_bo_lcb(
            problem=problem,
            num_iterations=config.num_iterations,
            random_seeds=random_seeds,
            num_initial_random_samples=config.num_initial_random_samples,
        )

        print("\n[3/3] Running Random Search...")
        rs_history = run_random_search(
            problem=problem,
            num_iterations=config.num_iterations,
            random_seeds=random_seeds,
        )

        results = {
            "HEBO": hebo_history,
            "BO (LCB)": bo_history,
            "Random Search": rs_history,
        }

        if config.result_path is not None:
            save_result(results, config.result_path)

    print(f"\n{config.title}")
    for method, history in results.items():
        print(
            f"- {method}: shape={history.shape}, "
            f"mean final best={_mean_final_best(history):.4f}"
        )

    if config.plot_path is not None:
        plot_optimization_convergence(
            results,
            title=config.title,
            ylabel=config.y_label,
            objective=config.objective,
            regret_baseline=config.regret_baseline,
            y_scale=config.y_scale,
            error_style=config.error_style,
            fig_path=config.plot_path,
            show=config.show_plot,
        )
        print(f"- Convergence plot saved to {config.plot_path}")

    return results

