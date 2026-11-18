"""End-to-end check of serializer -> parser -> llm_eval on real data with mock 'models'.

base = garbage text (parser falls back to Single-best -> must equal Single-best exactly)
A    = oracle (true top-5 from the performance matrix -> Regret@1 must be 0)
B    = Single-best top-5 written with noise and duplicates
Writes to a temp dir; nothing in results_full is touched.
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.llm_eval import run_llm_eval
from src.serialize_text import TEST_FOLDS, build_target

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "data" / "phase2"
perf = pd.read_parquet(ROOT / "data" / "meta_full" / "performance_matrix.parquet")
sb = json.loads((P2 / "sb_order.json").read_text())
tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
pred, out = tmp / "pred", tmp / "out"
pred.mkdir(parents=True, exist_ok=True)
for f in TEST_FOLDS:
    rows = [json.loads(line) for line in open(P2 / "prompts" / "A" / f"fold{f}_test.jsonl")]
    order = sb[f"fold{f}"]
    gen = {
        "base": lambda t: "Sorry, as an AI I can't rank these.",
        "A": lambda t: build_target(perf.loc[t], order),
        "B": lambda t: "Answer: " + ", ".join(order[:3] + order[:2]) + ", foo_bar",
    }
    for m, g in gen.items():
        (pred / f"predictions_{m}_fold{f}.json").write_text(json.dumps(
            {"predictions": [{"task_id": r["task_id"], "generated": g(r["task_id"])} for r in rows]}))
res = run_llm_eval("full", pred, out)
h = pd.read_csv(out / "llm_results_helpers.csv").set_index("Selector")
assert h.loc["LLM fine-tuned A", "r1_mean"] == 0.0, "oracle must have zero regret"
assert abs(h.loc["LLM base (prompt A, no FT)", "r1_mean"] - h.loc["Single-best", "r1_mean"]) < 1e-9
print(h[["n_tasks", "r1_mean", "r3_mean", "dr1_mean"]].round(3).to_string())
print(res["parser"].round(1).to_string(index=False))
print("MOCK CHECK OK ->", out)
