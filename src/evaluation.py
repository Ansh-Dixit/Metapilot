"""Evaluation metrics, cross-validation evaluation of selectors, and plotting for MetaPilot."""

import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import get_mode_from_args, load_config
from src.metamodels import (
    KNNMetaLearner,
    LGBMLambdaRankSelector,
    LGBMRegressorSelector,
    RandomSelector,
    SingleBestSelector,
    StaticPortfolioSelector,
)
from src.portfolio import PIPELINE_REGISTRY, get_all_pipeline_ids


# ---------------------------------------------------------------------------
# Winner distribution check (from Milestone 2 / SPEC Section 12)
# ---------------------------------------------------------------------------

def compute_winner_distribution(
    perf_matrix: pd.DataFrame,
    pipeline_families: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute winning algorithm families and pipelines with tie-splitting."""
    if pipeline_families is None:
        pipeline_families = {pid: fam for pid, (fam, _) in PIPELINE_REGISTRY.items()}

    n_tasks = len(perf_matrix)
    if n_tasks == 0:
        return pd.DataFrame(columns=["wins", "share_pct"]), pd.DataFrame()

    family_wins = {fam: 0.0 for fam in set(pipeline_families.values())}
    pipeline_wins = {pid: 0.0 for pid in perf_matrix.columns}

    for task_id in perf_matrix.index:
        row = perf_matrix.loc[task_id]
        max_acc = row.max()
        winners = row[row == max_acc].index.tolist()
        credit = 1.0 / len(winners)

        for pid in winners:
            pipeline_wins[pid] += credit
            fam = pipeline_families[pid]
            family_wins[fam] += credit

    family_df = pd.DataFrame([
        {"family": fam, "wins": family_wins[fam], "share_pct": (family_wins[fam] / n_tasks) * 100.0}
        for fam in family_wins
    ]).sort_values("wins", ascending=False).set_index("family")

    pipeline_df = pd.DataFrame([
        {
            "pipeline_id": pid,
            "family": pipeline_families[pid],
            "wins": pipeline_wins[pid],
            "share_pct": (pipeline_wins[pid] / n_tasks) * 100.0,
        }
        for pid in pipeline_wins
    ]).sort_values("wins", ascending=False).set_index("pipeline_id")

    return family_df, pipeline_df


def check_winner_distribution(
    perf_matrix: pd.DataFrame,
    pipeline_families: dict | None = None,
    save_path: str = "results/winner_distribution.png",
    warn_threshold: float = 50.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Print the winner-distribution table, save plot, and warn if >warn_threshold%."""
    family_df, _ = compute_winner_distribution(perf_matrix, pipeline_families)
    n_tasks = len(perf_matrix)

    print("\n" + "=" * 54)
    print("WINNING ALGORITHM FAMILIES DISTRIBUTION (Tie-Split)")
    print("=" * 54)
    print(f"{'Family':22s} | {'Wins':>8s} | {'Share (%)':>12s}")
    print("-" * 54)
    for fam, row in family_df.iterrows():
        print(f"{fam:22s} | {row['wins']:8.2f} | {row['share_pct']:11.2f}%")
    print("-" * 54)
    print(f"{'Total':22s} | {family_df['wins'].sum():8.2f} | {family_df['share_pct'].sum():11.2f}%")
    print("=" * 54)

    warnings_list = []
    for fam, row in family_df.iterrows():
        if row["share_pct"] > warn_threshold:
            msg = (
                f"SANITY CHECK WARNING: Family '{fam}' won {row['share_pct']:.2f}% of tasks, "
                f"exceeding the {warn_threshold:.0f}% dominance threshold!"
            )
            warnings_list.append(msg)
            print(f"\n[!] WARNING: {msg}\n")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    families = list(family_df.index)
    pcts = list(family_df["share_pct"])
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    bars = ax.bar(range(len(families)), pcts, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Win Share (%)", fontsize=11)
    ax.set_title(f"Winning Algorithm Family Distribution ({n_tasks} Tasks, Tie-Split Credit)", fontsize=12)
    ax.set_ylim(0, max(max(pcts) * 1.15, warn_threshold + 10))

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold" if pct > warn_threshold else "normal",
        )

    ax.axhline(y=warn_threshold, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return family_df, warnings_list


# ---------------------------------------------------------------------------
# Core Evaluation Metrics
# ---------------------------------------------------------------------------

def compute_task_regret(
    true_accuracies: np.ndarray,
    recommended_pipeline_indices: list[int] | np.ndarray,
    k: int,
) -> float:
    """Compute Regret@k for a single task in percentage points.

    Regret(t, k) = 100 * [ max_p a(t, p) - max_{i=1..k} a(t, p_i) ]
    """
    best_possible = np.max(true_accuracies)
    top_k_indices = recommended_pipeline_indices[:k]
    best_recommended = np.max(true_accuracies[top_k_indices])
    regret = 100.0 * (best_possible - best_recommended)
    return float(max(0.0, regret))


def compute_bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and percentile bootstrap confidence interval.

    Returns (mean, ci_lower, ci_upper).
    """
    values = np.asarray(values)
    mean_val = float(np.mean(values))
    n = len(values)
    if n <= 1:
        return mean_val, mean_val, mean_val

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_resamples)
    for b in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return mean_val, ci_lower, ci_upper


def compute_paired_bootstrap_ci(
    model_values: np.ndarray,
    baseline_values: np.ndarray,
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and percentile bootstrap CI for paired difference (model - baseline).

    Uses identical bootstrap resamples.
    Returns (mean_diff, ci_lower, ci_upper).
    """
    diff = np.asarray(model_values) - np.asarray(baseline_values)
    mean_diff = float(np.mean(diff))
    n = len(diff)
    if n <= 1:
        return mean_diff, mean_diff, mean_diff

    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[b] = np.mean(diff[idx])

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_diffs, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha)))
    return mean_diff, ci_lower, ci_upper


def evaluate_predictions(
    rankings: dict[str, list[str]],
    scores: pd.DataFrame,
    perf_matrix: pd.DataFrame,
    task_ids: list[str] | None = None,
    n_bootstrap: int = 2000,
    hit_tolerance_pts: float = 0.5,
    seed: int = 42,
) -> dict:
    """Evaluate a selector over a specified set of tasks.

    Parameters
    ----------
    rankings : dict[task_id, list[pipeline_id]]
    scores : pd.DataFrame (task_id x pipeline_id)
    perf_matrix : pd.DataFrame (task_id x pipeline_id)
    task_ids : list of str, optional (defaults to all in rankings)
    n_bootstrap : int
    hit_tolerance_pts : float
    seed : int

    Returns
    -------
    dict with metrics:
        regret_1_mean, regret_1_ci_lo, regret_1_ci_hi,
        regret_3_mean, regret_3_ci_lo, regret_3_ci_hi,
        hit_rate, spearman_mean, regret_vs_k
    """
    if task_ids is None:
        task_ids = list(rankings.keys())

    pipeline_cols = list(perf_matrix.columns)
    pipe_to_idx = {pid: i for i, pid in enumerate(pipeline_cols)}

    n_pipes = len(pipeline_cols)
    regrets_at_1 = []
    regrets_at_3 = []
    hits = []
    spearmans = []
    regret_curves = []  # shape: (n_tasks, n_pipes)

    for tid in task_ids:
        true_accs = perf_matrix.loc[tid, pipeline_cols].values
        rec_pids = rankings[tid]
        rec_indices = [pipe_to_idx[p] for p in rec_pids]

        # Regret@1
        r1 = compute_task_regret(true_accs, rec_indices, k=1)
        regrets_at_1.append(r1)

        # Regret@3
        r3 = compute_task_regret(true_accs, rec_indices, k=3)
        regrets_at_3.append(r3)

        # Hit rate
        hits.append(1.0 if r1 <= hit_tolerance_pts else 0.0)

        # Full regret curve k=1..n_pipes
        curve = [compute_task_regret(true_accs, rec_indices, k) for k in range(1, n_pipes + 1)]
        regret_curves.append(curve)

        # Spearman correlation
        pred_task_scores = scores.loc[tid, pipeline_cols].values
        # Handle constant predictions or constant true accuracies
        if np.std(pred_task_scores) > 1e-12 and np.std(true_accs) > 1e-12:
            res = spearmanr(pred_task_scores, true_accs)
            rho = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
            if np.isnan(rho):
                rho = 0.0
        else:
            rho = 0.0
        spearmans.append(rho)

    regrets_at_1 = np.array(regrets_at_1)
    regrets_at_3 = np.array(regrets_at_3)

    r1_mean, r1_lo, r1_hi = compute_bootstrap_ci(regrets_at_1, n_resamples=n_bootstrap, seed=seed)
    r3_mean, r3_lo, r3_hi = compute_bootstrap_ci(regrets_at_3, n_resamples=n_bootstrap, seed=seed + 1)

    mean_regret_curve = np.mean(regret_curves, axis=0).tolist()

    return {
        "regret_1_mean": r1_mean,
        "regret_1_ci_lo": r1_lo,
        "regret_1_ci_hi": r1_hi,
        "regret_3_mean": r3_mean,
        "regret_3_ci_lo": r3_lo,
        "regret_3_ci_hi": r3_hi,
        "hit_rate": float(np.mean(hits)),
        "spearman_mean": float(np.mean(spearmans)),
        "regret_vs_k": mean_regret_curve,
        "regrets_at_1": regrets_at_1,
        "regrets_at_3": regrets_at_3,
    }


# ---------------------------------------------------------------------------
# Cross-Validation Orchestration across Folds
# ---------------------------------------------------------------------------

def run_cross_validation_evaluation(
    meta_df: pd.DataFrame,
    perf_matrix: pd.DataFrame,
    gap_df: pd.DataFrame,
    lambdarank_df: pd.DataFrame,
    folds_dict: dict[str, int],
    cfg: dict,
    feature_cols: list[str] | None = None,
) -> tuple[dict[str, dict], dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Run 5-fold cross-validation across all selectors.

    Returns
    -------
    all_rankings : dict[selector_name, dict[task_id, list[pipeline_id]]]
    all_scores : dict[selector_name, pd.DataFrame]
    feature_importances : dict[selector_name, np.ndarray]
    """
    if feature_cols is not None:
        meta_sub = meta_df[feature_cols].copy()
    else:
        meta_sub = meta_df.copy()

    knn_k = cfg["metamodels"]["knn_k"]
    lgbm_cfg = cfg["metamodels"]["lgbm"]
    seed = cfg["seed"]

    selector_factories = {
        "Random": lambda fold_seed: RandomSelector(seed=fold_seed),
        "Single-best": lambda fold_seed: SingleBestSelector(),
        "Static portfolio": lambda fold_seed: StaticPortfolioSelector(),
        "kNN": lambda fold_seed: KNNMetaLearner(k=knn_k),
        "LightGBM Regressor": lambda fold_seed: LGBMRegressorSelector(
            n_estimators=lgbm_cfg["n_estimators"],
            learning_rate=lgbm_cfg["learning_rate"],
            num_leaves=lgbm_cfg["num_leaves"],
            seed=fold_seed,
        ),
        "LightGBM LambdaRank": lambda fold_seed: LGBMLambdaRankSelector(
            n_estimators=lgbm_cfg["n_estimators"],
            learning_rate=lgbm_cfg["learning_rate"],
            num_leaves=lgbm_cfg["num_leaves"],
            seed=fold_seed,
        ),
    }

    pipeline_ids = list(perf_matrix.columns)
    # Only tasks present in folds_dict are used (lets callers restrict to a subset).
    folds_dict = {t: f for t, f in folds_dict.items() if t in perf_matrix.index}
    unique_folds = sorted(set(folds_dict.values()))

    all_rankings = {name: {} for name in selector_factories}
    all_scores = {
        name: pd.DataFrame(index=list(folds_dict), columns=pipeline_ids, dtype=float)
        for name in selector_factories
    }
    lgbm_reg_importances = []
    lgbm_rank_importances = []

    for fold_idx in unique_folds:
        test_tasks = [t for t, f in folds_dict.items() if f == fold_idx]
        train_tasks = [t for t, f in folds_dict.items() if f != fold_idx]

        for name, factory in selector_factories.items():
            model = factory(seed + fold_idx * 100)
            model.fit(train_tasks, meta_sub, perf_matrix, gap_df, lambdarank_df)
            ranks, scores_df = model.predict(test_tasks, meta_sub)

            for tid in test_tasks:
                all_rankings[name][tid] = ranks[tid]
                all_scores[name].loc[tid] = scores_df.loc[tid]

            if name == "LightGBM Regressor" and model.feature_importances_ is not None:
                lgbm_reg_importances.append(model.feature_importances_)
            elif name == "LightGBM LambdaRank" and model.feature_importances_ is not None:
                lgbm_rank_importances.append(model.feature_importances_)

    feat_importances = {}
    if lgbm_reg_importances:
        feat_importances["LightGBM Regressor"] = np.mean(lgbm_reg_importances, axis=0)
    if lgbm_rank_importances:
        feat_importances["LightGBM LambdaRank"] = np.mean(lgbm_rank_importances, axis=0)

    return all_rankings, all_scores, feat_importances


def format_results_table(
    metrics_by_selector: dict[str, dict],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, str]:
    """Format results according to SPEC format with paired bootstrap CI vs Single-best:

    Selector | Regret@1 (CI) | Regret@3 (CI) | Top-1 hit | Spearman | Δ vs Single-best (CI)
    """
    sb_regrets = None
    if "Single-best" in metrics_by_selector and "regrets_at_1" in metrics_by_selector["Single-best"]:
        sb_regrets = metrics_by_selector["Single-best"]["regrets_at_1"]
        sb_regrets3 = metrics_by_selector["Single-best"]["regrets_at_3"]

    rows = []
    for name, m in metrics_by_selector.items():
        r1_str = f"{m['regret_1_mean']:.2f} ({m['regret_1_ci_lo']:.2f}, {m['regret_1_ci_hi']:.2f})"
        r3_str = f"{m['regret_3_mean']:.2f} ({m['regret_3_ci_lo']:.2f}, {m['regret_3_ci_hi']:.2f})"

        diff_str = "-"
        diff_mean = 0.0
        diff_lo = 0.0
        diff_hi = 0.0
        diff3_str, d3_m, d3_lo, d3_hi = "-", 0.0, 0.0, 0.0
        if sb_regrets is not None and "regrets_at_1" in m:
            if name == "Single-best":
                diff_str = "0.00 (ref)"
                diff3_str = "0.00 (ref)"
            else:
                d3_m, d3_lo, d3_hi = compute_paired_bootstrap_ci(
                    m["regrets_at_3"], sb_regrets3, n_resamples=n_bootstrap, seed=seed + 1
                )
                diff3_str = f"{d3_m:+.2f} ({d3_lo:+.2f}, {d3_hi:+.2f})"
                diff_mean, diff_lo, diff_hi = compute_paired_bootstrap_ci(
                    m["regrets_at_1"], sb_regrets, n_resamples=n_bootstrap, seed=seed
                )
                diff_str = f"{diff_mean:+.2f} ({diff_lo:+.2f}, {diff_hi:+.2f})"

        rows.append({
            "Selector": name,
            "Regret@1 (CI)": r1_str,
            "Regret@3 (CI)": r3_str,
            "Top-1 hit": f"{m['hit_rate']:.3f}",
            "Spearman": f"{m['spearman_mean']:.3f}",
            "Δ vs Single-best (CI)": diff_str,
            "Δ@3 vs Single-best (CI)": diff3_str,
            # Raw numerical columns for plotting / programmatic use
            "_r1_mean": m["regret_1_mean"],
            "_r1_lo": m["regret_1_ci_lo"],
            "_r1_hi": m["regret_1_ci_hi"],
            "_r3_mean": m["regret_3_mean"],
            "_hit_rate": m["hit_rate"],
            "_spearman": m["spearman_mean"],
            "_diff_mean": diff_mean,
            "_diff_lo": diff_lo,
            "_diff_hi": diff_hi,
            "_r3_lo": m["regret_3_ci_lo"],
            "_r3_hi": m["regret_3_ci_hi"],
            "_diff3_mean": d3_m,
            "_diff3_lo": d3_lo,
            "_diff3_hi": d3_hi,
        })

    df = pd.DataFrame(rows)
    display_df = df[["Selector", "Regret@1 (CI)", "Regret@3 (CI)", "Top-1 hit", "Spearman", "Δ vs Single-best (CI)", "Δ@3 vs Single-best (CI)"]]

    # Build Markdown table
    header = "| Selector | Regret@1 (CI) | Regret@3 (CI) | Top-1 hit | Spearman | Δ@1 vs Single-best (CI) | Δ@3 vs Single-best (CI) |"
    divider = "|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for _, row in display_df.iterrows():
        lines.append(
            f"| {row['Selector']} | {row['Regret@1 (CI)']} | {row['Regret@3 (CI)']} | {row['Top-1 hit']} | {row['Spearman']} | {row['Δ vs Single-best (CI)']} | {row['Δ@3 vs Single-best (CI)']} |"
        )
    md_table = "\n".join(lines)

    return df, md_table


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------

def plot_regret_at_1_ci(
    results_df: pd.DataFrame,
    save_path: str = "results/regret_at_1_ci.png",
):
    """Plot bar chart of Regret@1 with 95% bootstrap confidence intervals."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    selectors = results_df["Selector"].tolist()
    means = results_df["_r1_mean"].tolist()
    yerr_lower = [m - lo for m, lo in zip(means, results_df["_r1_lo"])]
    yerr_upper = [hi - m for m, hi in zip(means, results_df["_r1_hi"])]
    yerr = [yerr_lower, yerr_upper]

    colors = plt.cm.Set2(np.linspace(0, 1, len(selectors)))
    bars = ax.bar(
        range(len(selectors)),
        means,
        yerr=yerr,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(range(len(selectors)))
    ax.set_xticklabels(selectors, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Regret@1 (percentage points)", fontsize=11)
    ax.set_title("Regret@1 with 95% parent-cluster bootstrap CIs (lower is better)", fontsize=12)

    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.1,
            f"{m:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Regret@1 CI plot to {save_path}")


def plot_regret_vs_k(
    metrics_by_selector: dict[str, dict],
    save_path: str = "results/regret_vs_k.png",
):
    """Plot Regret vs k curves for all selectors for k = 1..n_pipelines (from the curves)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    styles = {
        "Random": ("gray", "--", "o"),
        "Single-best": ("darkorange", "-.", "s"),
        "Static portfolio": ("saddlebrown", ":", "P"),
        "kNN": ("royalblue", "-", "^"),
        "LightGBM Regressor": ("forestgreen", "-", "D"),
        "LightGBM LambdaRank": ("purple", "-", "v"),
    }

    n_pipes = len(next(iter(metrics_by_selector.values()))["regret_vs_k"])
    ks = list(range(1, n_pipes + 1))
    for name, m in metrics_by_selector.items():
        curve = m["regret_vs_k"]
        assert len(curve) == n_pipes
        color, ls, marker = styles.get(name, ("black", "-", "o"))
        ax.plot(ks, curve, label=name, color=color, linestyle=ls, marker=marker, markersize=5, linewidth=1.5)

    ax.set_xlabel("Evaluation Budget k (Top-k pipelines evaluated)", fontsize=11)
    ax.set_ylabel("Mean Regret (percentage points)", fontsize=11)
    ax.set_title("Regret vs. Budget k Curve Across All Selectors", fontsize=12)
    ax.set_xticks([k for k in ks if k == 1 or k % 2 == 0 or n_pipes <= 20])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Regret-vs-k plot to {save_path}")


def plot_lgbm_feature_importance(
    importances: np.ndarray,
    meta_feature_names: list[str],
    save_path: str = "results/lgbm_feature_importance.png",
    top_n: int = 10,
):
    """Plot LightGBM gain share of the top 10 meta-features."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # importances array has meta-features followed by pipeline one-hot features
    meta_imps = importances[: len(meta_feature_names)]
    total_meta_gain = np.sum(meta_imps)
    if total_meta_gain <= 0:
        total_meta_gain = 1.0

    gain_shares = (meta_imps / total_meta_gain) * 100.0

    feat_series = pd.Series(gain_shares, index=meta_feature_names).sort_values(ascending=True)
    top_feats = feat_series.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(range(len(top_feats)), top_feats.values, color="teal", edgecolor="black", linewidth=0.8)
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats.index, fontsize=10)
    ax.set_xlabel("Gain Share (%) among Meta-Features", fontsize=11)
    ax.set_title(f"Top {top_n} Meta-Features by LightGBM Gain Share", fontsize=12)

    for bar, val in zip(bars, top_feats.values):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved LightGBM feature importance plot to {save_path}")


# ---------------------------------------------------------------------------
# CLI & Full Evaluation Pipeline
# ---------------------------------------------------------------------------

def run_pilot_evaluation(mode: str = "pilot") -> dict:
    """Run full Stage B evaluation in pilot mode."""
    cfg = load_config(mode)
    meta_dir = Path(cfg["paths"]["meta"])
    results_dir = Path(cfg["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    folds_path = cfg["splits"]["file"]
    with open(folds_path) as f:
        folds_dict = json.load(f)

    perf_matrix = pd.read_parquet(meta_dir / "performance_matrix.parquet")
    meta_df = pd.read_parquet(meta_dir / "meta_features.parquet")
    gap_df = pd.read_parquet(meta_dir / "gap_targets.parquet")
    lambdarank_df = pd.read_parquet(meta_dir / "lambdarank_labels.parquet")
    task_info = pd.read_parquet(meta_dir / "task_info.parquet").set_index("task_id")

    task_ids = list(perf_matrix.index)
    # Skipped / incomplete tasks have no row in the matrix; everything else is
    # aligned to it (meta_features may contain extra tasks, which are ignored).
    missing = [t for t in task_ids if t not in meta_df.index or t not in folds_dict or t not in task_info.index]
    assert not missing, f"{len(missing)} matrix tasks lack meta-features/folds/task_info, e.g. {missing[:3]}"
    folds_dict = {t: folds_dict[t] for t in task_ids}
    n_bootstrap = cfg["evaluation"]["bootstrap_resamples"]
    hit_tol = cfg["evaluation"]["hit_tolerance_pts"]
    seed = cfg["seed"]

    print(f"\n[evaluation] Running 5-fold CV evaluation across {len(task_ids)} tasks...")

    # 2. Main CV run with all 14 meta-features
    rankings, scores, feat_importances = run_cross_validation_evaluation(
        meta_df, perf_matrix, gap_df, lambdarank_df, folds_dict, cfg
    )

    # Evaluate each selector across all tasks
    metrics_by_selector = {}
    for name in rankings:
        metrics_by_selector[name] = evaluate_predictions(
            rankings[name], scores[name], perf_matrix, task_ids=task_ids,
            n_bootstrap=n_bootstrap, hit_tolerance_pts=hit_tol, seed=seed,
        )

    res_df, md_table = format_results_table(metrics_by_selector, n_bootstrap=n_bootstrap, seed=seed)

    # Save CSV and display table
    prefix = "pilot" if mode == "pilot" else "full"
    csv_path = results_dir / f"{prefix}_results.csv"
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved main results to {csv_path}\n")
    print(md_table)

    # Plots
    plot_regret_at_1_ci(res_df, str(results_dir / "regret_at_1_ci.png"))
    plot_regret_vs_k(metrics_by_selector, str(results_dir / "regret_vs_k.png"))
    if "LightGBM Regressor" in feat_importances:
        plot_lgbm_feature_importance(
            feat_importances["LightGBM Regressor"],
            list(meta_df.columns),
            str(results_dir / "lgbm_feature_importance.png"),
        )

    # 3. Broken out by source: openml, sklearn, synthetic
    print("\n--- Subgroup Evaluation by Source ---")
    unique_sources = sorted(task_info["source"].unique())
    source_dfs = []

    for src in unique_sources:
        sub_tasks = [t for t in task_ids if task_info.loc[t, "source"] == src]
        if not sub_tasks:
            continue
        print(f"\n### Source: {src} (N={len(sub_tasks)})")
        metrics_src = {}
        for name in rankings:
            metrics_src[name] = evaluate_predictions(
                rankings[name], scores[name], perf_matrix, task_ids=sub_tasks,
                n_bootstrap=n_bootstrap, hit_tolerance_pts=hit_tol, seed=seed,
            )
        df_src, md_src = format_results_table(metrics_src, n_bootstrap=n_bootstrap, seed=seed)
        df_src["Source"] = src
        source_dfs.append(df_src)
        print(md_src)

    if source_dfs:
        by_source_df = pd.concat(source_dfs, ignore_index=True)
    else:
        by_source_df = pd.DataFrame()

    prefix = "pilot" if mode == "pilot" else "full"
    by_source_csv = results_dir / f"{prefix}_results_by_source.csv"
    by_source_df.to_csv(by_source_csv, index=False)
    print(f"\nSaved results by source to {by_source_csv}")

    # 4. Landmarker Ablation (without the 4 lm_* features)
    print("\n--- Landmarker Ablation (10 features, no lm_*) ---")
    ablation_cols = [c for c in meta_df.columns if not c.startswith("lm_")]

    rankings_abl, scores_abl, _ = run_cross_validation_evaluation(
        meta_df, perf_matrix, gap_df, lambdarank_df, folds_dict, cfg, feature_cols=ablation_cols
    )

    metrics_ablation = {"Single-best": metrics_by_selector["Single-best"]}
    for name in ["kNN", "LightGBM Regressor", "LightGBM LambdaRank"]:
        metrics_ablation[name] = evaluate_predictions(
            rankings_abl[name], scores_abl[name], perf_matrix, task_ids=task_ids,
            n_bootstrap=n_bootstrap, hit_tolerance_pts=hit_tol, seed=seed,
        )

    df_abl, md_abl = format_results_table(metrics_ablation, n_bootstrap=n_bootstrap, seed=seed)
    abl_csv = results_dir / f"{prefix}_results_ablation.csv"
    df_abl.to_csv(abl_csv, index=False)
    print(f"\nSaved ablation results to {abl_csv}\n")
    print(md_abl)

    # Multi-seed evaluation if configured
    eval_seeds = cfg.get("evaluation", {}).get("seeds", [seed])
    multi_seed_df = None
    if len(eval_seeds) > 1:
        print(f"\n--- Multi-Seed Evaluation across {len(eval_seeds)} seeds: {eval_seeds} ---")
        seed_results = []
        for s in eval_seeds:
            cfg_s = dict(cfg)
            cfg_s["seed"] = s
            rankings_s, scores_s, _ = run_cross_validation_evaluation(
                meta_df, perf_matrix, gap_df, lambdarank_df, folds_dict, cfg_s
            )
            m_s = {}
            for name in rankings_s:
                m_s[name] = evaluate_predictions(
                    rankings_s[name], scores_s[name], perf_matrix, task_ids=task_ids,
                    n_bootstrap=n_bootstrap, hit_tolerance_pts=hit_tol, seed=s,
                )
            df_s, _ = format_results_table(m_s, n_bootstrap=n_bootstrap, seed=s)
            df_s["seed"] = s
            seed_results.append(df_s)

        all_seeds_df = pd.concat(seed_results, ignore_index=True)
        all_seeds_df.to_csv(results_dir / f"{prefix}_results_per_seed.csv", index=False)
        agg_rows = []
        for name in rankings:
            sub = all_seeds_df[all_seeds_df["Selector"] == name]
            r1_m, r1_s = sub["_r1_mean"].mean(), sub["_r1_mean"].std()
            r3_m, r3_s = sub["_r3_mean"].mean(), sub["_r3_mean"].std()
            hit_m, hit_s = sub["_hit_rate"].mean(), sub["_hit_rate"].std()
            sp_m, sp_s = sub["_spearman"].mean(), sub["_spearman"].std()
            d_m, d_s = sub["_diff_mean"].mean(), sub["_diff_mean"].std()
            agg_rows.append({
                "Selector": name,
                "Regret@1 (mean ± std)": f"{r1_m:.2f} ± {r1_s:.2f}",
                "Regret@3 (mean ± std)": f"{r3_m:.2f} ± {r3_s:.2f}",
                "Top-1 hit (mean ± std)": f"{hit_m:.3f} ± {hit_s:.3f}",
                "Spearman (mean ± std)": f"{sp_m:.3f} ± {sp_s:.3f}",
                "Δ vs SB (mean ± std)": f"{d_m:+.2f} ± {d_s:.2f}" if name != "Single-best" else "0.00 (ref)",
            })

        multi_seed_df = pd.DataFrame(agg_rows)
        ms_csv = results_dir / f"{prefix}_multi_seed_results.csv"
        multi_seed_df.to_csv(ms_csv, index=False)
        print(f"\nSaved multi-seed results to {ms_csv}\n")
        print(multi_seed_df.to_markdown(index=False))

    # 5. OpenML-only: retrain AND evaluate on OpenML tasks only (same folds.json,
    #    restricted), so synthetic/sklearn tasks cannot drive the conclusions.
    openml_df = None
    if "openml" in set(task_info["source"]):
        oml_tasks = [t for t in task_ids if task_info.loc[t, "source"] == "openml"]
        oml_folds = {t: folds_dict[t] for t in oml_tasks}
        print(f"\n--- OpenML-only retrain + evaluation (N={len(oml_tasks)}) ---")
        perf_oml = perf_matrix.loc[oml_tasks]
        oml_seed_dfs = []
        for s in eval_seeds:
            cfg_s = dict(cfg)
            cfg_s["seed"] = s
            r_o, sc_o, _ = run_cross_validation_evaluation(
                meta_df.loc[oml_tasks], perf_oml, gap_df.loc[oml_tasks],
                lambdarank_df.loc[oml_tasks], oml_folds, cfg_s,
            )
            m_o = {
                name: evaluate_predictions(
                    r_o[name], sc_o[name], perf_oml, task_ids=oml_tasks,
                    n_bootstrap=n_bootstrap, hit_tolerance_pts=hit_tol, seed=s,
                )
                for name in r_o
            }
            df_o, md_o = format_results_table(m_o, n_bootstrap=n_bootstrap, seed=s)
            df_o["seed"] = s
            oml_seed_dfs.append(df_o)
            if s == seed:
                openml_df = df_o
                print(md_o)
        openml_df.to_csv(results_dir / f"{prefix}_results_openml_only.csv", index=False)
        oml_all = pd.concat(oml_seed_dfs, ignore_index=True)
        oml_all.to_csv(results_dir / f"{prefix}_results_openml_only_per_seed.csv", index=False)
        oml_agg = (
            oml_all.groupby("Selector", sort=False)[["_r1_mean", "_r3_mean", "_hit_rate", "_spearman", "_diff_mean"]]
            .agg(["mean", "std"])
        )
        oml_agg.columns = [f"{a}_{b}" for a, b in oml_agg.columns]
        oml_agg.to_csv(results_dir / f"{prefix}_results_openml_only_multi_seed.csv")

    return {
        "openml_only": openml_df,
        "main": res_df,
        "by_source": by_source_df,
        "ablation": df_abl,
        "multi_seed": multi_seed_df,
        "md_main": md_table,
    }


# ---------------------------------------------------------------------------
# Phase 1 (full) evaluation: per-task metrics, parent-cluster bootstrap, seeds
# ---------------------------------------------------------------------------

def per_task_metrics(
    rankings: dict[str, list[str]],
    scores: pd.DataFrame,
    perf_matrix: pd.DataFrame,
    task_ids: list[str],
    hit_tolerance_pts: float = 0.5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-task Regret@1, Regret@3, hit, Spearman and the full regret-vs-k curve.

    Returns (DataFrame indexed by task_id, curves array of shape (n_tasks, n_pipelines)).
    """
    cols = list(perf_matrix.columns)
    idx = {p: i for i, p in enumerate(cols)}
    A = perf_matrix.loc[task_ids, cols].to_numpy(dtype=float)
    best = A.max(axis=1)
    rows, curves = [], []
    for i, tid in enumerate(task_ids):
        order = np.array([idx[p] for p in rankings[tid]])
        run_max = np.maximum.accumulate(A[i, order])
        curve = np.maximum(0.0, 100.0 * (best[i] - run_max))
        curves.append(curve)
        pred = scores.loc[tid, cols].to_numpy(dtype=float)
        if np.std(pred) > 1e-12 and np.std(A[i]) > 1e-12:
            rho = spearmanr(pred, A[i]).statistic
            rho = 0.0 if np.isnan(rho) else float(rho)
        else:
            rho = 0.0
        rows.append({"task_id": tid, "r1": curve[0], "r3": curve[min(2, len(curve) - 1)],
                     "hit": float(curve[0] <= hit_tolerance_pts), "spearman": rho})
    return pd.DataFrame(rows).set_index("task_id"), np.asarray(curves)


def cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling GROUPS (parents) with replacement.

    Each drawn parent contributes all of its tasks, so the resampled statistic
    is sum(values of drawn parents) / number of their tasks. With the same
    `groups` order and `seed`, every selector gets identical resamples, so
    passing a per-task difference gives a paired cluster bootstrap.
    """
    values = np.asarray(values, dtype=float)
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    mean = float(values.mean())
    if len(uniq) <= 1:
        return mean, mean, mean
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(uniq), size=(n_resamples, len(uniq)))
    boot = sums[draws].sum(axis=1) / cnts[draws].sum(axis=1)
    a = (1.0 - ci) / 2.0
    return mean, float(np.percentile(boot, 100 * a)), float(np.percentile(boot, 100 * (1 - a)))


def _iid_ci(values, n_resamples, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    boot = values[rng.randint(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def _run_seeds(meta_df, perf_matrix, gap_df, lambdarank_df, folds, cfg, seeds,
               task_ids, hit_tol, feature_cols=None):
    """Run the full CV for every seed. Returns {selector: [per-task df per seed]},
    {selector: [curves per seed]}, {model: [importance arrays]}."""
    per_task, curves, imps = {}, {}, {}
    for s in seeds:
        cfg_s = dict(cfg)
        cfg_s["seed"] = s
        rk, sc, fi = run_cross_validation_evaluation(
            meta_df, perf_matrix, gap_df, lambdarank_df, folds, cfg_s, feature_cols=feature_cols,
        )
        for name in rk:
            df, cv = per_task_metrics(rk[name], sc[name], perf_matrix, task_ids, hit_tol)
            per_task.setdefault(name, []).append(df)
            curves.setdefault(name, []).append(cv)
        for name, arr in fi.items():
            imps.setdefault(name, []).append(arr)
    return per_task, curves, imps


def summarise_selectors(per_task, task_ids, groups, n_boot, seed=42, ref="Single-best"):
    """Seed-averaged table with parent-cluster CIs (+ task-level CIs for comparison).

    Point estimates and CIs are computed on per-task metrics averaged over seeds;
    "± std" is the std over seeds of each seed's mean. Paired Δ = selector - ref
    (negative = better than Single-best), parent-cluster resampled.
    """
    avg = {n: sum(dfs[i].loc[task_ids] for i in range(len(dfs))) / len(dfs) for n, dfs in per_task.items()}
    clean, helpers, tasklevel = [], [], []
    for name, dfs in per_task.items():
        a = avg[name]
        seed_means = pd.DataFrame([d.loc[task_ids].mean() for d in dfs])
        sd = seed_means.std(ddof=1) if len(dfs) > 1 else seed_means.iloc[0] * 0.0
        h = {"Selector": name, "n_tasks": len(task_ids), "n_parents": len(np.unique(groups)), "n_seeds": len(dfs)}
        for m, col_seed in [("r1", 0), ("r3", 1), ("hit", 2)]:
            mu, lo, hi = cluster_bootstrap_ci(a[m].values, groups, n_boot, seed + col_seed)
            h.update({f"{m}_mean": mu, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_seed_std": float(sd[m])})
            t_mu, t_lo, t_hi = _iid_ci(a[m].values, n_boot, seed + col_seed)
            tasklevel.append({"Selector": name, "metric": m, "mean": t_mu, "task_ci_lo": t_lo, "task_ci_hi": t_hi,
                              "cluster_ci_lo": lo, "cluster_ci_hi": hi})
        h["spearman_mean"] = float(a["spearman"].mean())
        h["spearman_seed_std"] = float(sd["spearman"])
        for m in ["r1", "r3"]:
            if name == ref or ref not in avg:
                h.update({f"d{m}_mean": 0.0, f"d{m}_lo": 0.0, f"d{m}_hi": 0.0})
                continue
            diff = a[m].values - avg[ref][m].values
            mu, lo, hi = cluster_bootstrap_ci(diff, groups, n_boot, seed + 10)
            h.update({f"d{m}_mean": mu, f"d{m}_lo": lo, f"d{m}_hi": hi})
            t_mu, t_lo, t_hi = _iid_ci(diff, n_boot, seed + 10)
            tasklevel.append({"Selector": name, "metric": f"delta_{m}_vs_{ref}", "mean": t_mu,
                              "task_ci_lo": t_lo, "task_ci_hi": t_hi, "cluster_ci_lo": lo, "cluster_ci_hi": hi})
        helpers.append(h)

        def fmt(m, d=2):
            return f"{h[m+'_mean']:.{d}f} ± {h[m+'_seed_std']:.{d}f} [{h[m+'_lo']:.{d}f}, {h[m+'_hi']:.{d}f}]"

        def fmt_d(m):
            if name == ref:
                return "ref"
            return f"{h['d'+m+'_mean']:+.2f} [{h['d'+m+'_lo']:+.2f}, {h['d'+m+'_hi']:+.2f}]"

        clean.append({
            "Selector": name,
            "Regret@1 mean ± seed-std [95% parent CI]": fmt("r1"),
            "Regret@3 mean ± seed-std [95% parent CI]": fmt("r3"),
            "Top-1 hit": fmt("hit", 3),
            "Spearman": f"{h['spearman_mean']:.3f} ± {h['spearman_seed_std']:.3f}",
            "Δ Regret@1 vs Single-best [95% paired parent CI]": fmt_d("r1"),
            "Δ Regret@3 vs Single-best [95% paired parent CI]": fmt_d("r3"),
        })
    return pd.DataFrame(clean), pd.DataFrame(helpers), pd.DataFrame(tasklevel)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def timeout_report(meta_dir: Path) -> tuple[pd.DataFrame, int]:
    """Per-pipeline timed-out (task, pipeline) pairs and the number of tasks with any timeout."""
    fl = pd.read_parquet(meta_dir / "fit_log.parquet")
    pm_tasks = set(pd.read_parquet(meta_dir / "performance_matrix.parquet").index)
    fl = fl[fl["task_id"].isin(pm_tasks)]
    pairs = fl.drop_duplicates(["task_id", "pipeline_id"])
    to = pairs[pairs["status"] == "timeout"]
    all_pipes = sorted(pairs["pipeline_id"].unique())
    counts = to.groupby("pipeline_id").size().reindex(all_pipes, fill_value=0).astype(int)
    names = to.groupby("pipeline_id")["task_id"].apply(lambda x: ", ".join(sorted(x))).reindex(all_pipes).fillna("")
    by_pipe = pd.DataFrame({"timeout_pairs": counts, "tasks": names})
    by_pipe.index.name = "pipeline_id"
    by_pipe["share_of_pairs_pct"] = 100.0 * by_pipe["timeout_pairs"] / pairs.groupby("pipeline_id").size().reindex(all_pipes)
    return by_pipe.sort_values("timeout_pairs", ascending=False), int(to["task_id"].nunique())


def run_phase1_evaluation(mode: str = "full") -> dict:
    """Phase 1 evaluation with parent-cluster bootstrap CIs and multi-seed averaging."""
    cfg = load_config(mode)
    meta_dir = Path(cfg["paths"]["meta"])
    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    prefix = "pilot" if mode == "pilot" else "full"

    perf = pd.read_parquet(meta_dir / "performance_matrix.parquet")
    meta_df = pd.read_parquet(meta_dir / "meta_features.parquet")
    gap_df = pd.read_parquet(meta_dir / "gap_targets.parquet")
    lam_df = pd.read_parquet(meta_dir / "lambdarank_labels.parquet")
    info = pd.read_parquet(meta_dir / "task_info.parquet").set_index("task_id")
    with open(cfg["splits"]["file"]) as f:
        folds = json.load(f)

    task_ids = list(perf.index)
    missing = [t for t in task_ids if t not in meta_df.index or t not in folds or t not in info.index]
    assert not missing, f"{len(missing)} matrix tasks lack meta-features/folds/task_info"
    folds = {t: folds[t] for t in task_ids}
    groups = info.loc[task_ids, "parent_id"].to_numpy()
    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    hit_tol = cfg["evaluation"]["hit_tolerance_pts"]
    seeds = cfg["evaluation"].get("seeds", [cfg["seed"]])
    md = [f"# Phase 1 results ({mode})", "",
          f"{len(task_ids)} tasks, {len(np.unique(groups))} parents, {perf.shape[1]} pipelines, seeds {seeds}. "
          f"CIs: {n_boot} parent-cluster bootstrap resamples. Δ < 0 means better than Single-best.", ""]

    # 1. Main run (all 14 meta-features), every seed
    per_task, curves, imps = _run_seeds(meta_df, perf, gap_df, lam_df, folds, cfg, seeds, task_ids, hit_tol)
    clean, helpers, tasklevel = summarise_selectors(per_task, task_ids, groups, n_boot)
    clean.to_csv(out / f"{prefix}_results.csv", index=False)
    helpers.to_csv(out / f"{prefix}_results_helpers.csv", index=False)
    tasklevel.to_csv(out / f"{prefix}_results_task_level_ci.csv", index=False)
    pd.DataFrame([
        {"Selector": n, "seed": s, **dfs[i].loc[task_ids].mean().to_dict()}
        for n, dfs in per_task.items() for i, s in enumerate(seeds)
    ]).to_csv(out / f"{prefix}_results_per_seed.csv", index=False)
    md += ["## Main table (all tasks)", "", _md(clean), ""]

    # Plots: Regret@1 (cluster CIs), regret-vs-k (seed-averaged), LightGBM gain share
    plot_df = helpers.rename(columns={"r1_mean": "_r1_mean", "r1_lo": "_r1_lo", "r1_hi": "_r1_hi"})
    plot_regret_at_1_ci(plot_df, str(out / "regret_at_1_ci.png"))
    curve_avg = {n: {"regret_vs_k": np.mean([c.mean(axis=0) for c in cvs], axis=0).tolist()} for n, cvs in curves.items()}
    plot_regret_vs_k(curve_avg, str(out / "regret_vs_k.png"))
    pd.DataFrame({n: v["regret_vs_k"] for n, v in curve_avg.items()},
                 index=pd.RangeIndex(1, perf.shape[1] + 1, name="k")).to_csv(out / f"{prefix}_regret_vs_k.csv")
    gain_rows = {}
    for name, arrs in imps.items():
        g = np.mean(arrs, axis=0)[: meta_df.shape[1]]
        gain_rows[name] = 100.0 * g / g.sum()
    gain_df = pd.DataFrame(gain_rows, index=meta_df.columns).sort_values("LightGBM Regressor", ascending=False)
    gain_df.to_csv(out / f"{prefix}_lgbm_gain_share.csv", index_label="meta_feature")
    plot_lgbm_feature_importance(np.mean(imps["LightGBM Regressor"], axis=0), list(meta_df.columns),
                                 str(out / "lgbm_feature_importance.png"))

    # 2. By source (slices of the main run, no retraining)
    src_tables = []
    md += ["## By source (slices of the main run)", ""]
    for src in sorted(info["source"].unique()):
        sub = [t for t in task_ids if info.loc[t, "source"] == src]
        sub_groups = info.loc[sub, "parent_id"].to_numpy()
        c, _, _ = summarise_selectors({n: [d.loc[sub] for d in dfs] for n, dfs in per_task.items()},
                                      sub, sub_groups, n_boot)
        c.insert(0, "Source", f"{src} (N={len(sub)}, parents={len(np.unique(sub_groups))})")
        src_tables.append(c)
        md += [f"### {src}", "", _md(c.drop(columns="Source")), ""]
    pd.concat(src_tables).to_csv(out / f"{prefix}_results_by_source.csv", index=False)

    # 3. OpenML-only: retrain and evaluate on OpenML tasks only
    if "openml" in set(info["source"]):
        oml = [t for t in task_ids if info.loc[t, "source"] == "openml"]
        og = info.loc[oml, "parent_id"].to_numpy()
        pt_o, _, _ = _run_seeds(meta_df.loc[oml], perf.loc[oml], gap_df.loc[oml], lam_df.loc[oml],
                                {t: folds[t] for t in oml}, cfg, seeds, oml, hit_tol)
        c_o, h_o, t_o = summarise_selectors(pt_o, oml, og, n_boot)
        c_o.to_csv(out / f"{prefix}_results_openml_only.csv", index=False)
        h_o.to_csv(out / f"{prefix}_results_openml_only_helpers.csv", index=False)
        md += [f"## OpenML only (retrained on OpenML tasks; N={len(oml)}, parents={len(np.unique(og))})", "",
               _md(c_o), ""]

    # 4. Landmarker ablation (no lm_* features), every seed; baselines don't use meta-features
    abl_cols = [c for c in meta_df.columns if not c.startswith("lm_")]
    pt_a, _, _ = _run_seeds(meta_df, perf, gap_df, lam_df, folds, cfg, seeds, task_ids, hit_tol, feature_cols=abl_cols)
    abl = {"Single-best": per_task["Single-best"]}
    for n in ["kNN", "LightGBM Regressor", "LightGBM LambdaRank"]:
        abl[f"{n} (with landmarkers)"] = per_task[n]
        abl[f"{n} (no landmarkers)"] = pt_a[n]
    c_a, h_a, _ = summarise_selectors(abl, task_ids, groups, n_boot)
    c_a.to_csv(out / f"{prefix}_results_ablation.csv", index=False)
    h_a.to_csv(out / f"{prefix}_results_ablation_helpers.csv", index=False)
    md += [f"## Landmarker ablation ({len(abl_cols)} features, no lm_*)", "", _md(c_a), ""]

    # 5. Timeouts
    if (meta_dir / "fit_log.parquet").exists():
        to_df, n_to_tasks = timeout_report(meta_dir)
        to_df.to_csv(out / f"{prefix}_timeouts_by_pipeline.csv")
        md += ["## Timeouts", "",
               f"{int(to_df['timeout_pairs'].sum())} timed-out (task, pipeline) pairs across {n_to_tasks} tasks. "
               "A timeout in ANY fold discards that pair's other folds (all 3 fold accuracies are NaN); the pair's "
               "mean accuracy is then set to that task's lowest observed accuracy across pipelines.", "",
               _md(to_df[to_df["timeout_pairs"] > 0].reset_index()[["pipeline_id", "timeout_pairs", "share_of_pairs_pct", "tasks"]]
                   .round(2)), ""]

    md += ["## LightGBM gain share (top 10 meta-features, % of meta-feature gain)", "",
           _md(gain_df.head(10).round(1).reset_index().rename(columns={"index": "meta_feature"})), ""]
    (out / f"{prefix}_results_tables.md").write_text("\n".join(md))
    print(f"[evaluation] wrote tables to {out / f'{prefix}_results_tables.md'}")
    return {"main": clean, "helpers": helpers}


def compare_vs_static(mode: str = "full", ks=(1, 2, 3, 4, 5),
                      models=("LightGBM Regressor", "LightGBM LambdaRank")) -> pd.DataFrame:
    """RQ1: paired parent-cluster bootstrap of Regret@k (model - Static portfolio), k = 1..5.

    Runs on all tasks and on the OpenML-only retrain (same folds, seed-averaged
    per-task regrets). Δ < 0: the meta-model has lower regret than the Static
    portfolio. "static_no_longer_worse" is True once the CI upper bound >= 0,
    i.e. the meta-model is no longer significantly better than the Static portfolio.
    """
    cfg = load_config(mode)
    meta_dir = Path(cfg["paths"]["meta"])
    out = Path(cfg["paths"]["results"])
    perf = pd.read_parquet(meta_dir / "performance_matrix.parquet")
    meta_df = pd.read_parquet(meta_dir / "meta_features.parquet")
    gap_df = pd.read_parquet(meta_dir / "gap_targets.parquet")
    lam_df = pd.read_parquet(meta_dir / "lambdarank_labels.parquet")
    info = pd.read_parquet(meta_dir / "task_info.parquet").set_index("task_id")
    with open(cfg["splits"]["file"]) as f:
        folds = json.load(f)
    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    hit_tol = cfg["evaluation"]["hit_tolerance_pts"]
    seeds = cfg["evaluation"].get("seeds", [cfg["seed"]])
    all_tasks = list(perf.index)
    oml = [t for t in all_tasks if info.loc[t, "source"] == "openml"]

    rows = []
    for label, tasks in [("all", all_tasks), ("openml_only", oml)]:
        sub_folds = {t: folds[t] for t in tasks}
        _, curves, _ = _run_seeds(meta_df.loc[tasks], perf.loc[tasks], gap_df.loc[tasks], lam_df.loc[tasks],
                                  sub_folds, cfg, seeds, tasks, hit_tol)
        groups = info.loc[tasks, "parent_id"].to_numpy()
        avg = {n: np.mean(c, axis=0) for n, c in curves.items()}  # (n_tasks, n_pipes), seed-averaged
        for m in models:
            for k in ks:
                diff = avg[m][:, k - 1] - avg["Static portfolio"][:, k - 1]
                mu, lo, hi = cluster_bootstrap_ci(diff, groups, n_boot, seed=42 + k)
                rows.append({"tasks": label, "model": m, "k": k, "n_tasks": len(tasks),
                             "model_regret": float(avg[m][:, k - 1].mean()),
                             "static_regret": float(avg["Static portfolio"][:, k - 1].mean()),
                             "delta_model_minus_static": mu, "ci_lo": lo, "ci_hi": hi,
                             "static_no_longer_worse": bool(hi >= 0)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "vs_static.csv", index=False)
    return df


if __name__ == "__main__":
    import sys
    mode = get_mode_from_args()
    if "--vs-static" in sys.argv:
        compare_vs_static(mode)
        raise SystemExit(0)
    cfg = load_config(mode)
    perf_path = os.path.join(cfg["paths"]["meta"], "performance_matrix.parquet")

    if os.path.exists(perf_path):
        perf = pd.read_parquet(perf_path)
        fam_df, _ = check_winner_distribution(perf, save_path=os.path.join(cfg["paths"]["results"], "winner_distribution.png"))
        fam_df.to_csv(os.path.join(cfg["paths"]["results"], "winner_distribution_by_family.csv"))
        run_phase1_evaluation(mode)
    else:
        print(f"No performance matrix found at {perf_path}. Run run_portfolio first.")
