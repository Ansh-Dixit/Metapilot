"""LLM inference (Colab) and ranking parser (local + Colab) for Phase 2.

The parser is pure Python. Heavy GPU libraries (torch, transformers, peft,
bitsandbytes) are imported lazily inside the Colab-only functions, so this
module imports fine on macOS without them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 64


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _id_pattern(pipeline_ids: list[str]) -> re.Pattern:
    # Longest ids first so "knn_15" wins over "knn_1"; boundaries stop matches inside
    # longer tokens ("logreg_c1" inside "logreg_c10", "svm_c1" inside "svm_c1x").
    alts = "|".join(re.escape(p) for p in sorted(pipeline_ids, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9_.])({alts})(?![A-Za-z0-9_]|\.\d)")


def parse_ranking(text: str | None, pipeline_ids: list[str], fill_order: list[str]) -> tuple[list[str], int]:
    """Turn generated text into a full ranking of all pipelines.

    1. Extract known pipeline_ids in order of appearance (case-sensitive, exact ids).
    2. Drop duplicates and unknown ids.
    3. Append the remaining pipelines in `fill_order` (training-fold Single-best order).

    Returns (ranking of len(pipeline_ids), n_extracted) where n_extracted is the
    number of distinct valid ids the model produced.
    """
    if set(fill_order) != set(pipeline_ids):
        raise ValueError("fill_order must be a permutation of pipeline_ids")
    found: list[str] = []
    for m in _id_pattern(pipeline_ids).finditer(text or ""):
        pid = m.group(1)
        if pid not in found:
            found.append(pid)
    n_extracted = len(found)
    ranking = found + [p for p in fill_order if p not in found]
    return ranking, n_extracted


# ---------------------------------------------------------------------------
# Generation (Colab only)
# ---------------------------------------------------------------------------

def build_messages(system_prompt: str, prompt: str) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]


def load_model(adapter_dir: str | None = None, base_model: str = BASE_MODEL):
    """4-bit base model (+ optional LoRA adapter) for inference on a T4. Colab only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(base_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                 torch_dtype=torch.float16, device_map="auto")
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok


def generate_batch(model, tok, system_prompt: str, prompts: list[str], batch_size: int = 16,
                   max_new_tokens: int = MAX_NEW_TOKENS) -> list[str]:
    """Greedy batched generation. Colab only."""
    import torch

    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [tok.apply_chat_template(build_messages(system_prompt, p), tokenize=False,
                                         add_generation_prompt=True) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return outs


def predict_file(model, tok, system_prompt: str, test_jsonl: str | Path, out_json: str | Path,
                 method: str, fold: int, batch_size: int = 16) -> None:
    """Generate for every task in a fold's test file and write predictions_{method}_fold{f}.json.

    The file is written atomically at the end, so an existing file always means
    "this (method, fold) is complete".
    """
    rows = [json.loads(line) for line in open(test_jsonl)]
    texts = generate_batch(model, tok, system_prompt, [r["prompt"] for r in rows], batch_size)
    payload = {"method": method, "fold": fold, "base_model": BASE_MODEL,
               "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy",
               "predictions": [{"task_id": r["task_id"], "generated": t} for r, t in zip(rows, texts)]}
    out_json = Path(out_json)
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(out_json)
