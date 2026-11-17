"""Stand-in LoRA trainer. Writes an adapter in the agreed format so the pipeline runs end to end
before the real fine-tuning is ready. Replace with the real adapters by pointing evaluation at
their directories; nothing else changes.

Recipe (from the demo): rank-16 LoRA on all attention + MLP projections, loss on the target
tokens only, full batch, early stop as soon as every fact's first English prompt answers with
the new object.

Two modes:
  counterfactual edits (train_examples): one adapter, loss on the target word only
    python -m xling.toy_finetune --facts data/toy_facts.jsonl --out adapters/toy
  fictional updates (train_rounds): one LoRA trained round by round, a checkpoint per round
    python -m xling.toy_finetune --facts data/fictional_facts.jsonl --rounds --out adapters/fictional
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .hooks import final_position_lens
from .modeling import load_model
from .schema import Fact, load_facts
from .scoring import answer_ids, score_logprobs

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_batch(tok, facts: list[Fact], device):
    rows = []
    for f in facts:
        for ex in f.train_examples:
            p, t, s = (tok(x).input_ids if x else [] for x in (ex["prefix"], ex["target"], ex["suffix"]))
            rows.append((p + t + s, [-100] * len(p) + t + [-100] * len(s)))
    if not rows:
        raise ValueError("no train_examples in the facts file")
    T = max(len(ids) for ids, _ in rows)
    pad = tok.pad_token_id
    ids = torch.tensor([i + [pad] * (T - len(i)) for i, _ in rows], device=device)
    labels = torch.tensor([l + [-100] * (T - len(l)) for _, l in rows], device=device)
    mask = torch.tensor([[1] * len(i) + [0] * (T - len(i)) for i, _ in rows], device=device)
    return ids, mask, labels


def english_top1(model, tok, facts: list[Fact], lang: str = "en") -> int:
    prompts = [f.prompts[lang][0] for f in facts]
    _, final = final_position_lens(model, tok, prompts)
    return sum(bool(score_logprobs(final[i], answer_ids(tok, f, lang, prompts[i]))["top1_new"])
               for i, f in enumerate(facts))


def train_toy_adapter(model, tok, facts: list[Fact], out_dir, name: str = "toy", base: str = "",
                      r: int = 16, alpha: int = 32, lr: float = 1e-4, max_steps: int = 60,
                      eval_every: int = 2, seed: int = 0):
    """Trains a LoRA on `facts`, saves it to out_dir, and returns the PeftModel.
    Call `.unload()` on the result to get the plain base model back."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                                          task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    ids, mask, labels = build_batch(tok, facts, device)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    n_hit, step = 0, 0
    for step in range(1, max_steps + 1):
        pm.train()
        loss = pm(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % eval_every == 0:
            pm.eval()
            n_hit = english_top1(pm, tok, facts)
            print(f"step {step:3d}  loss {loss.item():.3f}  English top-1 = new: {n_hit}/{len(facts)}")
            if n_hit == len(facts):
                break
    pm.eval()
    out = Path(out_dir)
    pm.save_pretrained(out)
    meta = dict(name=name, base=base, fact_ids=[f.fact_id for f in facts], steps=step,
                english_top1=f"{n_hit}/{len(facts)}", final_loss=round(loss.item(), 4),
                r=r, alpha=alpha, lr=lr, max_steps=max_steps, seed=seed,
                target_modules=TARGET_MODULES, recipe="toy: target-only loss, early stop on English top-1")
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved adapter '{name}' to {out}")
    return pm


def accuracy_on(model, tok, facts: list[Fact], which: str, lang: str = "en", batch_size: int = 16) -> int:
    """How many facts' first `lang` prompt has the `which` ("old"/"new") answer as top-1."""
    hits = 0
    for i in range(0, len(facts), batch_size):
        chunk = facts[i:i + batch_size]
        prompts = [f.prompts[lang][0] for f in chunk]
        _, final = final_position_lens(model, tok, prompts)
        for b, f in enumerate(chunk):
            hits += bool(score_logprobs(final[b], answer_ids(tok, f, lang, prompts[b]))[f"top1_{which}"])
    return hits


@torch.no_grad()
def top1_rates(model, tok, facts: list[Fact], lang: str, batch_size: int = 16) -> tuple[float, float]:
    """(share with the old answer top-1, share with the new answer top-1) on each fact's first
    `lang` prompt. One plain forward pass per batch (no logit lens), for cheap monitoring."""
    from .hooks import encode
    device = next(model.parameters()).device
    old = new = 0
    for i in range(0, len(facts), batch_size):
        chunk = facts[i:i + batch_size]
        prompts = [f.prompts[lang][0] for f in chunk]
        ids, mask, pos = encode(tok, prompts, device)
        top = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits[:, -1].argmax(-1)
        for b, f in enumerate(chunk):
            a = answer_ids(tok, f, lang, prompts[b])
            new += int(top[b]) in a.new
            old += a.old is not None and int(top[b]) in a.old
    return old / len(facts), new / len(facts)


def train_rounds(model, tok, facts: list[Fact], out_root, rounds=("round_1", "round_2"), base: str = "",
                 r: int = 16, alpha: int = 32, lr: float = 2e-4, max_epochs: int = 100, batch_size: int = 16,
                 target_acc: float = 0.9, extra_epochs: int = 0, monitor_langs=("en", "fr", "fi", "zh"),
                 eval_every: int = 1, seed: int = 0):
    """Sequential fine-tuning for fictional facts: train one LoRA on each round's sentences in
    order, saving a checkpoint after each round (out_root/round1, out_root/round2, ...).

    Whole-sentence loss (the model has to learn the new entity, not just one answer word).
    Each round stops once `target_acc` of the facts' English questions give that round's answer
    (round 1 -> the "old" answer, the last round -> the "new" answer), then trains `extra_epochs`
    more; or it stops at `max_epochs`.

    Every `eval_every` epochs it also measures top-1 accuracy (old and new answer) in each of
    `monitor_langs`. That history goes to out_root/history.csv and into each meta.json, so you
    can see whether other languages pick the fact up with more training, level off, or degrade.
    Only English decides when to stop: the other languages must never steer training.

    Each checkpoint's meta.json records its reference (the previous checkpoint, or "base") and
    target, which evaluation uses to judge it. Returns the PeftModel (call .unload() after).
    """
    import numpy as np
    import pandas as pd
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0,
                                          task_type="CAUSAL_LM", target_modules=TARGET_MODULES))
    langs = [l for l in monitor_langs if all(l in f.prompts for f in facts)]
    if "en" not in langs:
        langs = ["en"] + langs
    history = []

    def measure(rnd, epoch, mean_loss):
        row = dict(round=rnd, epoch=epoch, loss=round(mean_loss, 4))
        for lang in langs:
            row[f"{lang}_old"], row[f"{lang}_new"] = top1_rates(pm, tok, facts, lang)
        history.append(row)
        return row

    saved_side = tok.padding_side
    reference = "base"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        pm.eval()
        row = measure("start", 0, float("nan"))
        print("before training: " + "  ".join(f"{l} old {row[f'{l}_old']:.0%} new {row[f'{l}_new']:.0%}" for l in langs))
        for k, rnd in enumerate(rounds):
            name = rnd.replace("_", "")                       # "round_1" -> "round1"
            which = "new" if k == len(rounds) - 1 else "old"
            texts = [s for f in facts for s in f.train_rounds.get(rnd, [])]
            if not texts:
                raise ValueError(f"no train_rounds['{rnd}'] sentences in the facts")
            opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
            stop_at, epoch, mean_loss, en = max_epochs, 0, float("nan"), 0.0
            for epoch in range(1, max_epochs + 1):
                pm.train()
                tok.padding_side = "right"
                order = rng.permutation(len(texts))
                losses = []
                for i in range(0, len(texts), batch_size):
                    enc = tok([texts[j] for j in order[i:i + batch_size]], return_tensors="pt", padding=True).to(device)
                    labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
                    loss = pm(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels).loss
                    loss.backward()
                    opt.step()
                    opt.zero_grad()
                    losses.append(loss.item())
                mean_loss = float(np.mean(losses))
                pm.eval()
                tok.padding_side = saved_side
                if epoch % eval_every == 0 or epoch == max_epochs:
                    row = measure(rnd, epoch, mean_loss)
                    en = row[f"en_{which}"]
                    print(f"{name} epoch {epoch:3d}  loss {mean_loss:.3f}  {which} answer top-1: " +
                          "  ".join(f"{l} {row[f'{l}_{which}']:.0%}" for l in langs))
                    if en >= target_acc and stop_at == max_epochs:
                        stop_at = min(max_epochs, epoch + extra_epochs)
                        if extra_epochs:
                            print(f"  English reached {target_acc:.0%}; training {extra_epochs} more epochs")
                if epoch >= stop_at:
                    break
            if en < target_acc:
                print(f"  {name}: English stopped at {en:.0%} after {max_epochs} epochs (target {target_acc:.0%})")
            out = out_root / name
            pm.save_pretrained(out)
            meta = dict(name=name, base=base, reference=reference, target=which,
                        fact_ids=[f.fact_id for f in facts], round=rnd, epochs=epoch,
                        converged=bool(en >= target_acc), english_top1=round(en, 3),
                        final_loss=round(mean_loss, 4), r=r, alpha=alpha, lr=lr, batch_size=batch_size,
                        max_epochs=max_epochs, target_acc=target_acc, extra_epochs=extra_epochs, seed=seed,
                        target_modules=TARGET_MODULES, recipe="toy sequential: whole-sentence loss, "
                        "early stop on English top-1",
                        history=[h for h in history if h["round"] in ("start", rnd)])
            (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            pd.DataFrame(history).to_csv(out_root / "history.csv", index=False)
            print(f"saved '{name}' to {out}")
            reference = name
    finally:
        tok.padding_side = saved_side
    pm.eval()
    return pm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--out", default="adapters/toy",
                    help="adapter dir; with --rounds, the parent dir for round1/, round2/")
    ap.add_argument("--name", default="toy")
    ap.add_argument("--split", default="", help="train only on facts with this split ('' = all)")
    ap.add_argument("--rounds", action="store_true", help="sequential training on train_rounds (fictional facts)")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--extra-epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    facts = [f for f in load_facts(a.facts) if not a.split or f.split == a.split]
    model, tok = load_model(a.base)
    if a.rounds:
        train_rounds(model, tok, facts, a.out, base=a.base, lr=a.lr or 2e-4, max_epochs=a.max_epochs,
                     extra_epochs=a.extra_epochs)
    else:
        train_toy_adapter(model, tok, facts, a.out, a.name, a.base, lr=a.lr or 1e-4, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
