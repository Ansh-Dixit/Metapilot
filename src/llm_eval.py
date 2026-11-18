"""Evaluate Phase 2 LLM predictions with the SAME metrics as Phase 1.

Reads ``predictions_{method}_fold{f}.json`` (method in base / A / B, f in 0..2),
parses each output into a full ranking (src.llm_infer.parse_ranking, filled in
training-fold Single-best order) and compares against Phase 1 selectors on the
SAME tasks (test folds 0-2 only). The Phase 1 selectors' out-of-fold rankings
for those tasks come from the same folds.json, so every method's training set
for fold f is "all other folds".

Metrics: per_task_metrics + summarise_selectors from src.evaluation (Regret@1,
Regret@3, top-1 hit, Spearman, parent-cluster bootstrap, paired Δ).

Usage::

    python -m src.llm_eval --mode full [--pred-dir results_full/phase2/predictions]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation import _md, per_task_metrics, run_cross_validation_evaluation, summarise_selectors
from src.llm_infer import parse_ranking
from src.serialize_text import TEST_FOLDS, TOP_K

METHODS = {"base": "LLM base (prompt A, no FT)", "A": "LLM fine-tuned A", "B": "LLM fine-tuned B"}
BASELINES = ["Single-best", "Static portfolio", "LightGBM Regressor", "LightGBM LambdaRank"]


def load_llm_rankings(pred_dir: Path, sb_orders: dict, candidates: list[str], folds: dict):
    """{method_label: (rankings, scores_df, parser_stats_df)} for methods with all folds present."""
    out = {}
    for key, label in METHODS.items():
        files = {f: pred_dir / f"predictions_{key}_fold{f}.json" for f in TEST_FOLDS}
        if not all(p.exists() for p in files.values()):
            missing = [f for f, p in files.items() if not p.exists()]
            print(f"[llm_eval] {label}: missing folds {missing}, skipped")
            continue
        rankings, stats = {}, []
        for f, path in files.items():
            payload = json.loads(path.read_text())
            expected = {t for t, k in folds.items() if k == f}
            got = {p["task_id"] for p in payload["predictions"]}
            if got != expected:
                raise ValueError(f"{path.name}: task set differs from test fold {f} "
                                 f"({len(got ^ expected)} mismatches)")
            for p in payload["predictions"]:
                ranking, n_ok = parse_ranking(p["generated"], candidates, sb_orders[f"fold{f}"])
                rankings[p["task_id"]] = ranking
                stats.append({"task_id": p["task_id"], "fold": f, "n_valid_ids": n_ok})
        n = len(candidates)
        scores = pd.DataFrame({t: {pid: float(n - i) for i, pid in enumerate(r)} for t, r in rankings.items()}).T
        out[label] = (rankings, scores[candidates], pd.DataFrame(stats))
    return out


def run_llm_eval(mode: str = "full", pred_dir: str | Path | None = None, out_dir: str | Path | None = None,
                 phase2_dir: str | Path | None = None) -> dict:
    cfg = load_config(mode)
    root = Path(__file__).resolve().parent.parent
    meta_dir = Path(cfg["paths"]["meta"])
    res_dir = Path(cfg["paths"]["results"])
    pred_dir = Path(pred_dir) if pred_dir else res_dir / "phase2" / "predictions"
    out_dir = Path(out_dir) if out_dir else res_dir / "phase2"
    phase2_dir = Path(phase2_dir) if phase2_dir else root / "data" / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)

    perf = pd.read_parquet(meta_dir / "performance_matrix.parquet")
    meta_df = pd.read_parquet(meta_dir / "meta_features.parquet")
    gap_df = pd.read_parquet(meta_dir / "gap_targets.parquet")
    lam_df = pd.read_parquet(meta_dir / "lambdarank_labels.parquet")
    info = pd.read_parquet(meta_dir / "task_info.parquet").set_index("task_id")
    with open(cfg["splits"]["file"]) as f:
        folds = {t: k for t, k in json.load(f).items() if t in perf.index}
    sb_orders = json.loads((phase2_dir / "sb_order.json").read_text())
    candidates = json.loads((phase2_dir / "candidates.json").read_text())
    assert candidates == list(perf.columns)

    n_boot = cfg["evaluation"]["bootstrap_resamples"]
    hit_tol = cfg["evaluation"]["hit_tolerance_pts"]
    tasks = sorted(t for t, k in folds.items() if k in TEST_FOLDS)
    groups = info.loc[tasks, "parent_id"].to_numpy()

    # Phase 1 selectors: out-of-fold rankings from the same 5-fold CV (seed 42; they are deterministic)
    rk, sc, _ = run_cross_validation_evaluation(meta_df, perf, gap_df, lam_df, folds, cfg)
    per_task = {n: [per_task_metrics(rk[n], sc[n], perf, tasks, hit_tol)[0]] for n in BASELINES}

    llm = load_llm_rankings(pred_dir, sb_orders, candidates, folds)
    parser_rows = []
    for label, (rankings, scores, stats) in llm.items():
        per_task[label] = [per_task_metrics(rankings, scores, perf, tasks, hit_tol)[0]]
        parser_rows.append({
            "method": label, "n_outputs": len(stats),
            "share_filled_pct": 100.0 * float((stats["n_valid_ids"] < TOP_K).mean()),
            "share_no_valid_id_pct": 100.0 * float((stats["n_valid_ids"] == 0).mean()),
            "mean_valid_ids": float(stats["n_valid_ids"].mean()),
        })

    clean, helpers, tasklevel = summarise_selectors(per_task, tasks, groups, n_boot)
    clean.to_csv(out_dir / "llm_results.csv", index=False)
    helpers.to_csv(out_dir / "llm_results_helpers.csv", index=False)
    tasklevel.to_csv(out_dir / "llm_results_task_level_ci.csv", index=False)
    parser_df = pd.DataFrame(parser_rows)
    parser_df.to_csv(out_dir / "llm_parser_stats.csv", index=False)

    md = ["# Phase 2 LLM results", "",
          f"Test folds {list(TEST_FOLDS)} only: {len(tasks)} tasks, {len(np.unique(groups))} parents. "
          f"Parent-cluster bootstrap ({n_boot}). Δ < 0 = better than Single-best.", "",
          "## All methods (same tasks)", "", _md(clean), ""]

    # RQ3: B vs A, paired, OpenML tasks only
    a_lbl, b_lbl = METHODS["A"], METHODS["B"]
    rq3 = None
    if a_lbl in per_task and b_lbl in per_task:
        oml = [t for t in tasks if info.loc[t, "source"] == "openml"]
        og = info.loc[oml, "parent_id"].to_numpy()
        rq3, _, _ = summarise_selectors({a_lbl: [per_task[a_lbl][0].loc[oml]], b_lbl: [per_task[b_lbl][0].loc[oml]]},
                                        oml, og, n_boot, ref=a_lbl)
        rq3 = rq3.rename(columns=lambda c: c.replace("vs Single-best", "vs fine-tuned A"))
        rq3.to_csv(out_dir / "llm_rq3_B_vs_A_openml.csv", index=False)
        md += [f"## RQ3: prompt B vs A (OpenML tasks only, N={len(oml)}, parents={len(np.unique(og))}); Δ = B - A", "",
               _md(rq3), ""]
    if len(parser_df):
        md += ["## Parser fill-in", "",
               "share_filled = outputs with fewer than 5 distinct valid pipeline ids (rest filled in Single-best order).", "",
               _md(parser_df.round(2)), ""]
    (out_dir / "llm_results_tables.md").write_text("\n".join(md))
    print(f"[llm_eval] wrote {out_dir / 'llm_results_tables.md'}")
    return {"main": clean, "rq3": rq3, "parser": parser_df}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["pilot", "full"])
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args, _ = ap.parse_known_args()
    run_llm_eval(args.mode, args.pred_dir, args.out_dir)
