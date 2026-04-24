#!/usr/bin/env python3
"""Plot parameter sweep results from CSV.

Auto-detects 1D (line plot) vs 2D (heatmap) sweeps based on the number
of non-metric columns in the CSV.

Usage:
    python scripts/plot_sweep.py results/sweep_mass_results.csv
    python scripts/plot_sweep.py results/sweep_results.csv --metric crash_rate
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLS = {
    "landing_rate", "crash_rate", "oob_rate", "timeout_rate",
    "landings", "crashes", "oob", "timeouts", "n_episodes",
}


def detect_param_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in METRIC_COLS]


def plot_1d(df: pd.DataFrame, param: str, metrics: list[str], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df[param].values
    for m in metrics:
        ax.plot(x, df[m].values * 100, marker="o", label=m.replace("_", " ").title())
    ax.set_xlabel(param.replace("_", " ").title())
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Sweep: {param}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_2d(df: pd.DataFrame, params: list[str], metric: str, out_path: Path):
    p0, p1 = params
    pivot = df.pivot_table(index=p1, columns=p0, values=metric)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pivot.values * 100,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.min(), pivot.index.max(),
        ],
        cmap="RdYlGn",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric.replace("_", " ").title() + " (%)")
    ax.set_xlabel(p0.replace("_", " ").title())
    ax.set_ylabel(p1.replace("_", " ").title())
    ax.set_title(f"Sweep: {metric.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot parameter sweep results")
    parser.add_argument("csv", type=str, help="Path to sweep results CSV")
    parser.add_argument("--metric", type=str, default="landing_rate",
                        help="Metric to plot for 2D sweeps (default: landing_rate)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    param_cols = detect_param_cols(df)

    out_stem = csv_path.with_suffix("")

    if len(param_cols) == 1:
        out_path = Path(f"{out_stem}_plot.png")
        plot_1d(df, param_cols[0], ["landing_rate", "crash_rate", "timeout_rate"], out_path)
    elif len(param_cols) == 2:
        out_path = Path(f"{out_stem}_{args.metric}.png")
        plot_2d(df, param_cols, args.metric, out_path)
    else:
        print(f"Sweep has {len(param_cols)} parameters ({param_cols}). "
              f"Only 1D and 2D plotting is supported.")
        print("You can filter the CSV to 1 or 2 sweep columns and re-run.")


if __name__ == "__main__":
    main()
