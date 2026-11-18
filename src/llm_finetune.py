"""QLoRA fine-tuning for Phase 2 (Colab, T4 GPU).

Qwen2.5-1.5B-Instruct in 4-bit NF4, LoRA r=16 / alpha=32 / dropout=0.05 on
q,k,v,o,gate,up,down; fp16 (the T4 has no bf16); max_seq_len 1024; batch 4 x
grad-accum 4; lr 2e-4; 2 epochs. Loss is computed on the completion (assistant
answer) only.

All GPU libraries are imported lazily inside `train_adapter`, so this module
imports on macOS without torch/bitsandbytes.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from src.llm_infer import BASE_MODEL

HPARAMS = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "max_seq_len": 1024,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "num_train_epochs": 2,
    "seed": 42,
}


def load_sft_records(train_jsonl: str | Path, system_prompt: str) -> list[dict]:
    """Conversational prompt/completion records (TRL format); loss only on 'completion'."""
    recs = []
    for line in open(train_jsonl):
        r = json.loads(line)
        recs.append({
            "prompt": [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["prompt"]}],
            "completion": [{"role": "assistant", "content": r["completion"]}],
        })
    return recs


def _kwargs_for(cls, wanted: dict) -> dict:
    """Keep only kwargs the installed TRL/transformers version accepts."""
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in wanted.items() if k in params}


def train_adapter(train_jsonl: str | Path, output_dir: str | Path, system_prompt: str,
                  hparams: dict | None = None, base_model: str = BASE_MODEL) -> str:
    """Train one LoRA adapter for one (fold, variant) and save it to output_dir. Colab only.

    The adapter is written to a temporary folder and renamed when complete, so an
    existing `output_dir/adapter_config.json` means the adapter finished.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    hp = {**HPARAMS, **(hparams or {})}
    output_dir = Path(output_dir)
    work_dir = output_dir.with_name(output_dir.name + "_tmp")

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb,
                                                 torch_dtype=torch.float16, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora = LoraConfig(r=hp["lora_r"], lora_alpha=hp["lora_alpha"], lora_dropout=hp["lora_dropout"],
                      target_modules=hp["target_modules"], bias="none", task_type="CAUSAL_LM")
    ds = Dataset.from_list(load_sft_records(train_jsonl, system_prompt))

    cfg_kwargs = _kwargs_for(SFTConfig, {
        "output_dir": str(work_dir),
        "per_device_train_batch_size": hp["per_device_train_batch_size"],
        "gradient_accumulation_steps": hp["gradient_accumulation_steps"],
        "learning_rate": hp["learning_rate"],
        "num_train_epochs": hp["num_train_epochs"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_strategy": "no",
        "report_to": "none",
        "seed": hp["seed"],
        "max_length": hp["max_seq_len"],        # TRL >= 0.16
        "max_seq_length": hp["max_seq_len"],    # older TRL
        "completion_only_loss": True,           # TRL >= 0.19 (prompt/completion datasets)
        "packing": False,
    })
    trainer_kwargs = _kwargs_for(SFTTrainer, {
        "model": model, "args": SFTConfig(**cfg_kwargs), "train_dataset": ds,
        "peft_config": lora, "processing_class": tok, "tokenizer": tok,
    })
    if "completion_only_loss" not in cfg_kwargs:
        raise RuntimeError("Installed TRL lacks completion_only_loss; install trl>=0.19 (see notebook).")
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    trainer.model.save_pretrained(str(work_dir / "adapter"))
    tok.save_pretrained(str(work_dir / "adapter"))
    (work_dir / "adapter" / "metapilot_hparams.json").write_text(json.dumps(
        {**hp, "base_model": base_model, "train_file": str(train_jsonl), "n_train": len(ds),
         "train_loss_log": [h for h in trainer.state.log_history if "loss" in h]}, indent=1))
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    (work_dir / "adapter").rename(output_dir)

    del trainer, model
    torch.cuda.empty_cache()
    return str(output_dir)
