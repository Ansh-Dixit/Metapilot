"""MetaPilot local demo (Gradio).

Run:  make demo        (or: .venv/bin/python app/gradio_demo.py)
Open: http://127.0.0.1:7860

Everything runs locally: share=False, analytics off, system serif fonts only (no web
fonts), light mode forced. Styling lives in app/theme.css; HTML pieces in app/ui_parts.py.
The recommendation logic is src/recommend.py (unchanged by the UI).
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for p in (str(ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import history as hist  # noqa: E402
from ui_parts import chips, hero, note, step, ticket, tickets  # noqa: E402

RESULTS = ROOT / "results_full"
SERIF = ["Iowan Old Style", "Palatino Linotype", "Palatino", "Book Antiqua", "Georgia", "serif"]

# Plain-language names for the recommendation methods (internal name -> shown name)
PLAIN_MODELS = {
    "LightGBM LambdaRank": "Smart ranker (recommended)",
    "LightGBM Regressor": "Score predictor",
    "kNN": "Similar datasets",
    "Static portfolio": "Safe list only",
}
MODEL_FROM_PLAIN = {v: k for k, v in PLAIN_MODELS.items()}

BENCH_NOTE = ("<b>Good to know:</b> in our tests a smart pick only beat a fixed default when you can try "
              "just one model. If you can try two or more, the safe list below is just as good.")

FORCE_LIGHT_JS = """
() => {
  const strip = () => { document.body.classList.remove('dark');
    document.querySelectorAll('.dark').forEach(e => e.classList.remove('dark'));
    document.documentElement.style.colorScheme = 'light'; };
  strip(); new MutationObserver(strip).observe(document.body, {attributes: true, attributeFilter: ['class']});
}
"""


def _esc(text) -> str:
    """Escape user-provided text (file names, column names, values) for HTML."""
    return html.escape(str(text), quote=True)


def _gr():
    import gradio as gr
    return gr


def _error(msg: str):
    """Friendly popup; strip HTML-significant characters from user-derived text."""
    gr = _gr()
    return gr.Error(html.escape(msg, quote=False))


# ---------------------------------------------------------------------------
# Callbacks (module level and light, so spawned worker processes import fast)
# ---------------------------------------------------------------------------

def on_example(name):
    from src.recommend import ensure_examples
    return str(ensure_examples()[name])


def on_file(path):
    gr = _gr()
    from src.recommend import RecommendError, read_csv
    if not path:
        return gr.update(choices=[], value=None), None, "<p class='mp-tag'>No file yet.</p>"
    try:
        df = read_csv(path)
    except RecommendError as exc:
        raise _error(str(exc))
    cols = [str(c) for c in df.columns]
    msg = (f"<p class='mp-tag'>Loaded <b>{_esc(Path(path).name)}</b>: {len(df):,} rows, {len(cols)} columns. "
           f"The last column (<b>{_esc(cols[-1])}</b>) is selected as the thing to predict; change it below if needed.</p>")
    return gr.update(choices=cols, value=cols[-1]), df.head(5), msg


def _short_reason(r: dict) -> str:
    good = r["contribution"] > 0
    return f"Your data has {r['plain']}" + (" — this model handles that well." if good
                                            else " — a small minus, but it still ranks first.")


def _traits(summary: dict, explanation: dict) -> list[str]:
    s = summary
    out = [f"{s['rows']:,} rows", f"{s['columns']} input columns", f"{s['classes']} categories to predict"]
    out.append("balanced categories" if s["minority_share_pct"] >= 100 / s["classes"] * 0.7
               else f"rarest category: {s['minority_share_pct']:.0f}% of rows")
    out.append("no missing values" if s["missing_pct"] == 0 else f"{s['missing_pct']:.1f}% missing values")
    if s["categorical_columns"]:
        out.append(f"{s['categorical_columns']} text columns")
    return out


def _alt_label(i: int) -> str:
    return "ALSO TRY" if i else "TRY THIS FIRST"


def do_recommend(path, target, k, model_plain):
    gr = _gr()
    import pandas as pd
    from src.recommend import RecommendError, read_csv, recommend
    if not path:
        raise _error("Please upload a CSV file or pick an example first.")
    if not target:
        raise _error("Please choose the column you want to predict.")
    model_name = MODEL_FROM_PLAIN.get(model_plain, model_plain)
    try:
        res = recommend(read_csv(path), target, int(k), model_name)
    except RecommendError as exc:
        raise _error(str(exc))

    s = res["summary"]
    header = step(3, "Your recommendations",
                  f"{s['rows']:,} rows · {s['columns']} columns · predicting {target} ({s['classes']} categories)")
    reasons = [_short_reason(r) for r in res["explanation"]["reasons"][:3]]
    top, rest = res["top_k"][0], res["top_k"][1:]
    picks = [ticket(1, top["description"], top["family"], reasons, label="TRY THIS FIRST")]
    picks += [ticket(r["rank"], r["description"], r["family"], (), label="ALSO TRY", small=True, alt=True)
              for r in rest]
    safe = [ticket(r["rank"], r["description"], r["family"], (), label="SAFE PICK", small=True, alt=True)
            for r in res["static_top_k"]]
    warn = "".join(f"<p>⚠️ {_esc(w)}</p>" for w in res["warnings"])
    dropped = (f"<p>Ignored columns: {_esc(', '.join(s['dropped_columns']))}.</p>" if s["dropped_columns"] else "")
    notes = note(BENCH_NOTE) + (note(warn + dropped) if (warn or dropped) else "")

    # Advanced (technical) details
    adv_table = pd.DataFrame([{"Rank": r["rank"], "Pipeline id": r["pipeline_id"], "Type": r["family"],
                               "Score": r["score"]} for r in res["top_k"]])
    exp = res["explanation"]
    adv_md = (f"**Method:** {_esc(PLAIN_MODELS[model_name])}. "
              f"**Score:** {_esc(res['score_meaning'])}.\n\n"
              f"**Why the first pick** (strongest data traits, as weighed by the "
              f"{_esc(PLAIN_MODELS[exp['source_model']].lower())} model):\n"
              + "\n".join(f"- `{r['feature']}` = {res['meta_features'][r['feature']]:.3f}: {_esc(r['plain'])} "
                          f"(contribution {r['contribution']:+.3f})" for r in exp["reasons"])
              + "\n\n**Safe list ids:** " + ", ".join(f"`{r['pipeline_id']}`" for r in res["static_top_k"]))
    ids = [r["pipeline_id"] for r in res["top_k"]]
    entry_id = hist.add(hist.make_entry(_dataset_name(path), s, target, PLAIN_MODELS[model_name],
                                        res["top_k"], res["static_top_k"], reasons))
    return (gr.update(visible=True), header, tickets(*picks), notes, chips(_traits(s, exp)),
            tickets(*safe), adv_table, adv_md, ids,
            gr.update(interactive=True, value=f"Test these {len(ids)} on my data"),
            entry_id, *history_view())


def do_test(path, target, ids, entry_id=None, progress=None):
    from src.recommend import PIPELINE_DESCRIPTIONS, RecommendError, evaluate_pipelines, read_csv
    if not ids:
        raise _error("Press 'Find my model' first.")
    try:
        df = evaluate_pipelines(read_csv(path), target, list(ids), row_cap=2000, timeout_per_fit=60,
                                progress=progress)
    except RecommendError as exc:
        raise _error(str(exc))
    if entry_id:
        hist.set_tests(entry_id, df.to_dict("records"))
    ok = df[df["status"] == "ok"]
    show = df.assign(Model=[PIPELINE_DESCRIPTIONS.get(p, p) for p in df["pipeline_id"]])
    show = show.rename(columns={"accuracy_pct": "Accuracy (%)", "status": "Status"})[["Model", "Accuracy (%)", "Status"]]
    if ok.empty:
        return show, note("None of the models finished in time. Try a smaller file.")
    best_i = ok["accuracy_pct"].idxmax()
    styled = show.style.apply(
        lambda row: ["background-color:#EC9C9D;font-weight:700" if row.name == best_i else "" for _ in row], axis=1)
    msg = note(f"<b>Best on your data:</b> {_esc(PIPELINE_DESCRIPTIONS.get(df.loc[best_i, 'pipeline_id']))} — "
               f"{df.loc[best_i, 'accuracy_pct']:.1f}% correct (3-way cross-check on "
               f"{int(df['rows_used'].iloc[0]):,} rows; big files are sampled to 2,000 rows).")
    return styled, msg


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _dataset_name(path) -> str:
    """File name, or 'example: <name>' when the file is one of the bundled examples.

    Gradio copies every file (examples included) into its cache, so compare contents,
    not folders: same name and identical bytes as a file in app/examples/.
    """
    from src.recommend import EXAMPLES_DIR
    p = Path(path)
    ex = EXAMPLES_DIR / p.name
    try:
        if ex.exists() and ex.stat().st_size == p.stat().st_size and ex.read_bytes() == p.read_bytes():
            return f"example: {p.stem}"
    except OSError:
        pass
    return p.name


def history_counter(n: int | None = None) -> str:
    n = len(hist.load()) if n is None else n
    return f"<p class='mp-tag' style='text-align:right;margin:0'>History ({n})</p>"


def history_view():
    """(counter html, table rows, ids in table order) for the History tab."""
    entries = hist.load()
    ordered = hist.newest_first(entries)
    return history_counter(len(entries)), hist.table_rows(entries), [e["id"] for e in ordered]


def render_entry(e: dict) -> str:
    """Read-only re-display of a saved entry (all user text escaped by ui_parts or _esc)."""
    picks = e.get("picks", [])
    parts = [step("↺", f"{e['dataset']} · predicting {e['target']}",
                  f"{e['timestamp'].replace('T', ' ')} · {e['rows']:,} rows × {e['columns']} cols · "
                  f"{e['classes']} categories · {e['model']}")]
    if picks:
        top = picks[0]
        items = [ticket(1, top["description"], top["family"], top.get("reasons", []), label="TRY THIS FIRST")]
        items += [ticket(p["rank"], p["description"], p["family"], (), label="ALSO TRY", small=True, alt=True)
                  for p in picks[1:]]
        parts.append(tickets(*items))
    parts.append("<h3>The safe list</h3>")
    parts.append(tickets(*[ticket(p["rank"], p["description"], p["family"], (), label="SAFE PICK",
                                  small=True, alt=True) for p in e.get("safe_list", [])]))
    tests = e.get("tests")
    if tests:
        best = hist.best_test(e)
        rows = []
        for t in tests:
            style = ' style="background:#EC9C9D;font-weight:700"' if best is not None and t is best else ""
            acc = "" if t["accuracy_pct"] is None else f"{t['accuracy_pct']:.1f}%"
            rows.append(f"<tr{style}><td>{_esc(t['pipeline_id'])}</td><td>{acc}</td>"
                        f"<td>{_esc(t['status'])}</td></tr>")
        rows = "".join(rows)
        parts.append("<h3>Tested on the data</h3><table><thead><tr><th>Model</th><th>Accuracy</th>"
                     f"<th>Status</th></tr></thead><tbody>{rows}</tbody></table>")
    else:
        parts.append(note("Not tested on the data."))
    return "".join(parts)


def on_history_select(ids, evt=None):
    if evt is None or not ids:
        return ""
    row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    entries = {e["id"]: e for e in hist.load()}
    e = entries.get(ids[row]) if 0 <= row < len(ids) else None
    return render_entry(e) if e else note("That entry no longer exists.")


def on_export():
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="metapilot_")) / "metapilot_history.csv"
    return hist.export_csv(out)


# ---------------------------------------------------------------------------
# "How it works" content
# ---------------------------------------------------------------------------

PLAIN_COLUMNS = {
    "Selector": "Method",
    "Regret@1 mean ± seed-std [95% parent CI]": "Accuracy points lost, 1 try",
    "Regret@3 mean ± seed-std [95% parent CI]": "Accuracy points lost, 3 tries",
    "Top-1 hit": "Picked the best (share)",
    "Spearman": "Order agreement",
    "Δ Regret@1 vs Single-best [95% paired parent CI]": "vs. best default, 1 try",
    "Δ Regret@3 vs Single-best [95% paired parent CI]": "vs. best default, 3 tries",
}
PLAIN_METHODS = {"Random": "Random pick", "Single-best": "Best single default", "Static portfolio": "Safe list",
                 "kNN": "Similar datasets", "LightGBM Regressor": "Score predictor", "LightGBM LambdaRank": "Smart ranker"}

HOW_IT_WORKS = """
### How it works
1. We tried **30 ready-made models** (10 kinds) on **1,040 practice datasets** built from 179 public tables.
2. For each table we measured **14 simple traits**: its size, how balanced the categories are, how related the columns are, and how well a few very quick models do.
3. A small learner studied which models win on which kind of table.
4. For your file, MetaPilot measures the same traits and ranks the 30 models for you.
5. The **safe list** is one fixed order chosen so that its first few models together do well on almost every table.

**Reading the table:** *accuracy points lost* = how much worse your best try is than the best of all 30 models (lower is better). Brackets show a 95% range; negative "vs." values mean better than always using one default model.
"""

LIMITS = """
### Limits
- Built mostly from public OpenML tables; very different data may behave differently.
- Models were compared on at most 1,000 rows per table, using accuracy only (not speed or fairness).
- Only 30 fixed models, with no tuning. The example datasets were part of the practice set, so their picks look better than usual.
"""


def _benchmark_table():
    import pandas as pd
    p = RESULTS / "full_results.csv"
    if not p.exists():
        return pd.DataFrame({"info": ["Benchmark results not found (results_full/full_results.csv)."]})
    df = pd.read_csv(p).rename(columns=PLAIN_COLUMNS)
    df["Method"] = df["Method"].map(lambda m: PLAIN_METHODS.get(m, m))
    return df.drop(columns=["Order agreement"], errors="ignore")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_app():
    gr = _gr()
    from src.recommend import ensure_examples

    examples = ensure_examples()
    theme = gr.themes.Base(font=SERIF, font_mono=["ui-monospace", "Menlo", "monospace"])
    css = (APP_DIR / "theme.css").read_text()

    with gr.Blocks(title="MetaPilot", theme=theme, css=css, js=FORCE_LIGHT_JS, analytics_enabled=False) as demo:
        gr.HTML(hero())
        counter = gr.HTML(history_counter())
        with gr.Tab("Find my model"):
            # ---- step 1
            gr.HTML(step(1, "Upload your data", "a CSV file, up to 50 MB"))
            file = gr.File(label="Your CSV file", file_types=[".csv"], type="filepath")
            gr.HTML("<p class='mp-tag'>No file at hand? Try an example:</p>")
            with gr.Row():
                ex_buttons = [(gr.Button(name.split(" (")[0].replace("_", " "), size="sm", variant="secondary",
                                         min_width=90), name) for name in examples]
            status = gr.HTML()
            preview = gr.Dataframe(label="First rows of your file", interactive=False, max_height=200)

            # ---- step 2
            gr.HTML(step(2, "Tell us what to predict"))
            with gr.Row():
                # choices come from the file; custom values allowed so API calls work too
                target = gr.Dropdown([], label="Column to predict", allow_custom_value=True, scale=2)
                k = gr.Slider(1, 5, value=3, step=1, label="How many models can you afford to try?", scale=3)
            go = gr.Button("Find my model", variant="primary", size="lg")

            # ---- step 3 (hidden until the button is pressed)
            with gr.Column(visible=False) as results_col:
                header = gr.HTML()
                trait_chips = gr.HTML()
                picks = gr.HTML()
                notes = gr.HTML()
                gr.HTML("<h3>The safe list</h3><p class='mp-tag'>A fixed order that works well on almost any table.</p>")
                safe = gr.HTML()
                test_btn = gr.Button("Test them on my data", variant="secondary", interactive=False)
                test_table = gr.Dataframe(label="How they really did on your data", interactive=False)
                test_note = gr.HTML()

            with gr.Accordion("Advanced", open=False):
                model = gr.Radio(list(PLAIN_MODELS.values()), value=PLAIN_MODELS["LightGBM LambdaRank"],
                                 label="Recommendation method")
                adv_table = gr.Dataframe(label="Details of the recommended models", interactive=False)
                adv_md = gr.Markdown()

            rec_ids = gr.State([])
            entry_id = gr.State(None)

            for btn, name in ex_buttons:
                btn.click(lambda n=name: on_example(n), None, file, api_name=False)
            file.change(on_file, file, [target, preview, status], api_name="load_file")
            # history components are created in the History tab below; wired after it
            recommend_inputs = [file, target, k, model]
            recommend_outputs = [results_col, header, picks, notes, trait_chips, safe, adv_table, adv_md,
                                 rec_ids, test_btn, entry_id]

            def _test(path, tgt, ids, eid, progress=gr.Progress()):
                table, msg = do_test(path, tgt, ids, eid, progress=progress)
                return (table, msg, *history_view())

        with gr.Tab("How it works"):
            gr.Markdown(HOW_IT_WORKS)
            gr.Dataframe(_benchmark_table(), label="Results on 1,040 practice datasets", interactive=False, wrap=True)
            with gr.Row():
                for img, lbl in [("regret_vs_k.png", "Accuracy lost vs. number of models tried"),
                                 ("winner_distribution.png", "Which kind of model wins most often")]:
                    if (RESULTS / img).exists():
                        gr.Image(str(RESULTS / img), label=lbl, show_download_button=False, min_width=300)
            gr.Markdown(LIMITS)
        with gr.Tab("History"):
            gr.HTML(step("↺", "Your past recommendations", "saved on this computer only; no data values are kept"))
            hist_table = gr.Dataframe(headers=["Date", "Dataset", "Size", "Predicting", "Top pick",
                                               "Best tested accuracy"],
                                      value=hist.table_rows(hist.load()), interactive=False, wrap=True,
                                      label="Newest first. Click a row to see it again.")
            hist_ids = gr.State([e["id"] for e in hist.newest_first(hist.load())])
            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="secondary", size="sm")
                export_btn = gr.Button("Download history (CSV)", variant="secondary", size="sm")
                clear_btn = gr.Button("Clear history", variant="secondary", size="sm")
            with gr.Row(visible=False) as confirm_row:
                gr.HTML(note("<b>Delete all saved history?</b> This cannot be undone."))
                confirm_yes = gr.Button("Yes, delete all", variant="primary", size="sm")
                confirm_no = gr.Button("Cancel", variant="secondary", size="sm")
            export_file = gr.File(label="Your history file", visible=False, interactive=False)
            hist_panel = gr.HTML()

        hist_outputs = [counter, hist_table, hist_ids]
        go.click(do_recommend, recommend_inputs, recommend_outputs + hist_outputs, api_name="recommend")
        test_btn.click(_test, [file, target, rec_ids, entry_id], [test_table, test_note] + hist_outputs,
                       api_name=False)

        def _select(ids, evt):
            return on_history_select(ids, evt)

        _select.__annotations__ = {"evt": gr.SelectData}  # real class (module uses postponed annotations)

        hist_table.select(_select, hist_ids, hist_panel, api_name=False)
        refresh_btn.click(history_view, None, hist_outputs, api_name=False)
        export_btn.click(lambda: gr.update(value=str(on_export()), visible=True), None, export_file, api_name=False)
        clear_btn.click(lambda: gr.update(visible=True), None, confirm_row, api_name=False)
        confirm_no.click(lambda: gr.update(visible=False), None, confirm_row, api_name=False)

        def _clear():
            hist.clear()
            return (gr.update(visible=False), "", *history_view())

        confirm_yes.click(_clear, None, [confirm_row, hist_panel] + hist_outputs, api_name=False)
        demo.load(history_view, None, hist_outputs, api_name=False)
    return demo


if __name__ == "__main__":
    port = int(os.environ.get("METAPILOT_PORT", "7860"))
    build_app().queue().launch(server_name="127.0.0.1", server_port=port, share=False, show_error=True)
