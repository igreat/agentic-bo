#!/usr/bin/env python
"""
Generate convergence plots for BO reports.

Usage:
    uv run python scripts/generate_convergence_plots.py <RUN_ID>
"""

import json
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set up matplotlib for LaTeX-compatible output
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'text.usetex': False,  # Disable LaTeX for compatibility
    'pgf.texsystem': 'pdflatex',
})

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_run_data(run_id: str):
    """Load run data for plotting."""
    run_path = Path(f"runs/{run_id}")

    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    # Load observations and suggestions
    observations = pd.DataFrame()
    if (run_path / "observations.jsonl").exists():
        observations = pd.read_json(run_path / "observations.jsonl", lines=True)

    suggestions = pd.DataFrame()
    if (run_path / "suggestions.jsonl").exists():
        suggestions = pd.read_json(run_path / "suggestions.jsonl", lines=True)

    return observations, suggestions


def plot_convergence(observations, run_id, output_dir):
    """Plot convergence of objective function values."""
    if len(observations) == 0:
        print("No observations to plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Objective values over iterations
    iterations = range(1, len(observations) + 1)
    y_values = observations['y'].values

    ax1.plot(iterations, y_values, 'b-o', linewidth=2, markersize=4, alpha=0.7)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Objective Function Values')
    ax1.grid(True, alpha=0.3)

    # Add best value line
    best_so_far = np.maximum.accumulate(y_values)
    ax1.plot(iterations, best_so_far, 'r--', linewidth=2, alpha=0.8, label='Best so far')
    ax1.legend()

    # Plot 2: Rolling best with confidence intervals
    rolling_mean = pd.Series(y_values).rolling(window=min(5, len(y_values)), center=True).mean()
    rolling_std = pd.Series(y_values).rolling(window=min(5, len(y_values)), center=True).std()

    ax2.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best value')
    ax2.fill_between(iterations,
                     best_so_far - rolling_std,
                     best_so_far + rolling_std,
                     alpha=0.2, color='red', label='±1 std dev')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Objective Value')
    ax2.set_title('Convergence Trajectory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "convergence_plot.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "convergence_plot.pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ Convergence plot saved to {output_dir}/convergence_plot.png")


def plot_parameter_distributions(observations, run_id, output_dir):
    """Plot parameter distributions if x data is available."""
    if len(observations) == 0 or 'x' not in observations.columns:
        print("No parameter data available for distribution plots")
        return

    # Extract parameters
    param_data = []
    param_names = None

    for idx, row in observations.iterrows():
        x = row.get('x', {})
        if isinstance(x, dict):
            if param_names is None:
                param_names = list(x.keys())
            param_data.append([x.get(name, np.nan) for name in param_names])

    if not param_data or not param_names:
        print("Could not extract parameter data")
        return

    param_df = pd.DataFrame(param_data, columns=param_names)

    # Create subplots for each parameter
    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, param in enumerate(param_names):
        if i < len(axes):
            ax = axes[i]
            data = param_df[param].dropna()

            if len(data) > 0:
                # Plot histogram with KDE
                sns.histplot(data, ax=ax, kde=True, alpha=0.7)
                ax.set_xlabel(f'{param}')
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {param}')
                ax.grid(True, alpha=0.3)

                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8,
                          label=f'μ = {mean_val:.3f}')
                ax.legend()

    # Hide empty subplots
    for i in range(len(param_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distributions.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "parameter_distributions.pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ Parameter distributions saved to {output_dir}/parameter_distributions.png")


def plot_correlation_matrix(observations, run_id, output_dir):
    """Plot correlation matrix between parameters and objective."""
    if len(observations) == 0 or 'x' not in observations.columns:
        print("No data available for correlation analysis")
        return

    # Extract parameters and objective
    param_data = []
    objectives = []

    for idx, row in observations.iterrows():
        x = row.get('x', {})
        y = row.get('y')
        if isinstance(x, dict) and y is not None:
            param_data.append(list(x.values()))
            objectives.append(y)

    if len(param_data) < 3:  # Need minimum data for correlation
        print("Insufficient data for correlation analysis")
        return

    # Create correlation matrix
    param_names = list(observations.iloc[0]['x'].keys())
    data = np.column_stack([np.array(param_data), objectives])
    columns = param_names + ['Objective']

    corr_matrix = np.corrcoef(data.T)

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Parameter-Objective Correlation Matrix")
    plt.tight_layout()

    plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "correlation_matrix.pdf", bbox_inches='tight')
    plt.close()

    print(f"✅ Correlation matrix saved to {output_dir}/correlation_matrix.png")


def generate_convergence_plots(run_id: str):
    """Generate all convergence plots for a BO run."""
    print(f"📊 Generating convergence plots for run: {run_id}")

    try:
        observations, suggestions = load_run_data(run_id)
        output_dir = Path(f"runs/{run_id}")

        # Generate plots
        plot_convergence(observations, run_id, output_dir)
        plot_parameter_distributions(observations, run_id, output_dir)
        plot_correlation_matrix(observations, run_id, output_dir)

        print(f"\n✅ All plots generated in {output_dir}/")

    except Exception as e:
        print(f"❌ Error generating plots: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate convergence plots for BO reports"
    )
    parser.add_argument("run_id", help="BO run ID (e.g., vivid-heron-3397)")

    args = parser.parse_args()
    generate_convergence_plots(args.run_id)