"""Loading models: the contract between fine-tuning and everything downstream.

An adapter is a directory written by `peft_model.save_pretrained(dir)` plus a `meta.json`:
    {"name": "toy", "base": "Qwen/Qwen3-1.7B-Base", "fact_ids": [...], ...hyperparameters...}
`fact_ids` says which facts the adapter was trained on; evaluation only scores those.
"""
from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str) -> torch.dtype:
    # Native bf16 needs Ampere or newer (A100, L4). T4 and CPU fall back to float32.
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float32


def load_model(base: str, adapter: str | Path | None = None, device: str | None = None,
               dtype: torch.dtype | None = None):
    """Returns (model, tokenizer). With `adapter`, the model is a PeftModel."""
    device = device or pick_device()
    dtype = dtype or pick_dtype(device)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
    if adapter is not None:
        model = attach_adapter(model, adapter)
    model.eval()
    return model, tok


def attach_adapter(model, adapter: str | Path):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model


def read_adapter_meta(adapter: str | Path) -> dict:
    path = Path(adapter) / "meta.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta.setdefault("name", Path(adapter).name)
    return meta


def adapter_disabled(model):
    """`with adapter_disabled(m):` runs m as the base model. No-op for a plain model."""
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


def model_device(model) -> torch.device:
    return next(model.parameters()).device
