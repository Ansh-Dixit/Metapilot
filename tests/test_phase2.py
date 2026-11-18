"""Phase 2 tests: parser edge cases, prompt leakage, fold separation, mock predictions."""

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from src.llm_eval import load_llm_rankings
from src.llm_infer import parse_ranking
from src.portfolio import get_all_pipeline_ids
from src.serialize_text import TEST_FOLDS, build_prompt, build_target

ROOT = Path(__file__).resolve().parent.parent
P2 = ROOT / "data" / "phase2"
IDS = get_all_pipeline_ids("full")
FILL = list(reversed(IDS))  # any permutation works as a Single-best order


# ---------------------------------------------------------------- parser
@pytest.mark.parametrize("text", ["", None, "I cannot help with that.", "!!!###\n\n", "1. 2. 3."])
def test_parser_garbage_or_empty_falls_back_to_fill_order(text):
    ranking, n = parse_ranking(text, IDS, FILL)
    assert n == 0 and ranking == FILL


def test_parser_duplicates_and_unknown_ids_dropped():
    text = "svm_c1, svm_c1, xgboost, knn_99, rf_100, SVM_C1, rf_100, lightgbm"
    ranking, n = parse_ranking(text, IDS, FILL)
    assert ranking[:2] == ["svm_c1", "rf_100"] and n == 2
    assert sorted(ranking) == sorted(IDS) and len(ranking) == len(IDS)


def test_parser_prefix_collisions_and_punctuation():
    text = "Best: knn_15. Then logreg_c10; logreg_c0.1, ridge_alpha0.1 and finally knn_1."
    ranking, n = parse_ranking(text, IDS, FILL)
    assert ranking[:5] == ["knn_15", "logreg_c10", "logreg_c0.1", "ridge_alpha0.1", "knn_1"] and n == 5


def test_parser_order_of_appearance_and_full_permutation():
    ranking, n = parse_ranking("hgb_default\net_100_d5\n- gnb", IDS, FILL)
    assert ranking[:3] == ["hgb_default", "et_100_d5", "gnb"]
    assert ranking[3:] == [p for p in FILL if p not in ranking[:3]]


def test_parser_rejects_bad_fill_order():
    with pytest.raises(ValueError):
        parse_ranking("gnb", IDS, IDS[:-1])


# ---------------------------------------------------------------- prompts
def test_build_prompt_takes_no_performance_input():
    params = set(inspect.signature(build_prompt).parameters)
    assert params == {"variant", "meta_row", "text_row", "candidates", "column_names", "description"}


def test_target_ties_broken_by_single_best_order():
    acc = pd.Series({"a": 0.9, "b": 0.9, "c": 0.8, "d": 0.7, "e": 0.6, "f": 0.5})
    assert build_target(acc, ["b", "a", "c", "d", "e", "f"]) == "b, a, c, d, e"


def _jsonl(path):
    return [json.loads(line) for line in open(path)]


needs_data = pytest.mark.skipif(not (P2 / "sb_order.json").exists(), reason="run src.serialize_text first")


@needs_data
def test_real_prompts_contain_no_label_or_accuracy_info():
    perf = pd.read_parquet(ROOT / "data" / "meta_full" / "performance_matrix.parquet")
    for v in ("A", "B"):
        rows = _jsonl(P2 / "prompts" / v / "fold0_train.jsonl")
        for r in rows:
            p = r["prompt"]
            assert r["completion"] not in p          # no target ranking
            assert "accuracy:" not in p.lower()      # no per-pipeline score lines
            # no measured accuracy of this task's best pipelines at 4 decimals
            for acc in perf.loc[r["task_id"]].nlargest(5):
                assert f"{acc:.4f}" not in p
        for r in _jsonl(P2 / "prompts" / v / "fold0_test.jsonl"):
            assert set(r) == {"task_id", "prompt"}    # test files carry no target


@needs_data
def test_no_test_fold_task_in_training_prompts():
    folds = json.load(open(ROOT / "data" / "meta_full" / "folds.json"))
    for v in ("A", "B"):
        for f in TEST_FOLDS:
            train = {r["task_id"] for r in _jsonl(P2 / "prompts" / v / f"fold{f}_train.jsonl")}
            test = {r["task_id"] for r in _jsonl(P2 / "prompts" / v / f"fold{f}_test.jsonl")}
            assert train and test and not (train & test)
            assert all(folds[t] != f for t in train)
            assert all(folds[t] == f for t in test)
            # no parent shared between train and test (GroupKFold by parent)
            par = lambda s: {t.rsplit("__s", 1)[0] for t in s}  # noqa: E731
            assert not (par(train) & par(test))


@needs_data
def test_mock_predictions_parse_on_real_data(tmp_path):
    folds = json.load(open(ROOT / "data" / "meta_full" / "folds.json"))
    sb = json.loads((P2 / "sb_order.json").read_text())
    cands = json.loads((P2 / "candidates.json").read_text())
    outputs = ["svm_c1, rf_100, hgb_default, et_100, mlp_128", "no idea", "gnb gnb knn_99 dt_d3"]
    for m in ("base", "A", "B"):
        for f in TEST_FOLDS:
            rows = _jsonl(P2 / "prompts" / "A" / f"fold{f}_test.jsonl")
            preds = [{"task_id": r["task_id"], "generated": outputs[i % 3]} for i, r in enumerate(rows)]
            (tmp_path / f"predictions_{m}_fold{f}.json").write_text(json.dumps({"predictions": preds}))
    folds_pm = {t: k for t, k in folds.items()}
    out = load_llm_rankings(tmp_path, sb, cands, folds_pm)
    assert len(out) == 3
    rankings, scores, stats = out["LLM fine-tuned A"]
    assert all(sorted(r) == sorted(cands) for r in rankings.values())
    assert set(stats["n_valid_ids"]) == {5, 0, 2}
