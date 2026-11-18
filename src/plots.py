"""Plotting utilities for MetaPilot."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_winner_distribution(perf_matrix: pd.DataFrame, pipeline_families: dict,
                             save_path: str, title: str = "Winning Algorithm Families"):
    """Plot distribution of winning algorithm families.

    Args:
        perf_matrix: DataFrame with task_id index and pipeline_id columns.
        pipeline_families: dict mapping pipeline_id -> family.
        save_path: path to save the figure.
        title: plot title.
    """
    # Find winners per task (handle ties by splitting credit)
    family_wins = {}
    for task_id in perf_matrix.index:
        row = perf_matrix.loc[task_id]
        max_acc = row.max()
        winners = row[row == max_acc].index.tolist()
        credit = 1.0 / len(winners)
        for pid in winners:
            fam = pipeline_families[pid]
            family_wins[fam] = family_wins.get(fam, 0.0) + credit

    # Sort by wins descending
    families = sorted(family_wins.keys(), key=lambda f: family_wins[f], reverse=True)
    wins = [family_wins[f] for f in families]
    total = sum(wins)
    pcts = [w / total * 100 for w in wins]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(families)), pcts, color=plt.cm.Set3(np.linspace(0, 1, len(families))))
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_ylabel("Win Share (%)")
    ax.set_title(title)

    # Add percentage labels on bars
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    # Add warning if any family > 50%
    if max(pcts) > 50:
        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5)
        ax.text(len(families) - 1, 52, "⚠ >50% dominance", color="red", fontsize=10, ha="right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved winner distribution plot to {save_path}")

    return families, pcts
