"""Headless smoke test: start the demo, call /recommend once with an example, shut it down."""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = os.environ.get("METAPILOT_PORT", "7860")
URL = f"http://127.0.0.1:{PORT}/"
import tempfile
HIST = Path(tempfile.mkdtemp()) / "history.json"  # never touch the real history
env = {**os.environ, "METAPILOT_PORT": PORT, "PYTHONWARNINGS": "ignore", "METAPILOT_HISTORY": str(HIST)}
log = open(ROOT / "logs" / "smoke_demo.log", "w")
proc = subprocess.Popen([sys.executable, str(ROOT / "app" / "gradio_demo.py")], env=env, stdout=log, stderr=log)
try:
    for _ in range(120):
        try:
            urllib.request.urlopen(URL, timeout=1)
            break
        except Exception:
            if proc.poll() is not None:
                raise SystemExit("server exited early; see logs/smoke_demo.log")
            time.sleep(1)
    else:
        raise SystemExit("server did not come up in 120 s")
    from gradio_client import Client, handle_file
    from src.recommend import ensure_examples
    client = Client(URL, verbose=False)
    example = ensure_examples()["iris (scikit-learn)"]
    out = client.predict(handle_file(str(example)), "target", 3, "Smart ranker (recommended)", api_name="/recommend")
    # layout-only outputs (the results Column) are not returned by the client: find parts by content
    texts = [o for o in out if isinstance(o, str)]
    pick = lambda marker: next(t for t in texts if marker in t)  # noqa: E731
    header, picks, notes = pick("mp-step"), pick("TRY THIS FIRST"), pick("mp-note")
    trait_chips, safe = pick("mp-chip"), pick("SAFE PICK")
    assert "TRY THIS FIRST" in picks and picks.count("ALSO TRY") == 2, picks[:300]
    assert safe.count("SAFE PICK") == 3 and "mp-chip" in trait_chips
    # no jargon in what the main page renders (results + static texts in the served config)
    import json, re
    cfg = urllib.request.urlopen(URL + "config", timeout=5).read().decode()
    jargon = re.compile(r"(?i)lambdarank|regret(?!_vs_k)|meta-?features?")
    hits = jargon.findall(" ".join([header, picks, notes, trait_chips, safe]) + cfg)
    assert not hits, f"jargon on page: {set(hits)}"
    import json as _j
    saved = _j.loads(HIST.read_text())
    assert len(saved) == 1 and saved[0]["dataset"] == "example: iris" and saved[0]["top_pick"], saved
    print("history entry:", saved[0]["dataset"], "->", saved[0]["top_pick"])
    print("UP:", URL)
    print("header:", re.sub("<[^>]+>", " ", header).split("  ")[-1].strip()[:100])
    print("top pick:", re.search(r'mp-title">([^<]+)', picks).group(1))
    print("SMOKE OK")
finally:
    proc.terminate()
    proc.wait(timeout=20)
    log.close()
