"""Activation patching (proposal III.C): does putting the English run's hidden state into a
failed target-language run restore the updated answer, and at which layer?

For each failure case (a `transfer_fail` row from status.csv: the trained model gives the new
answer in English but not in the target language), and each layer l:

  1. run the English prompt and keep its residual stream (the source)
  2. rerun the target-language prompt with residual layer l, at one token position, replaced
     by the English vector from the same layer and matching position
  3. score the output with the same `score_logprobs` as the lens, so
         delta_l = D_patched,l - D_baseline          (the proposal's Delta_l, in nats)
     is in the same units as D_l in lens.csv

Controls (proposal section IV):
  delta_control  the same patch, but from a *different* fact's English run. Moving any English
                 state into the target run can shift D for reasons unrelated to the fact
                 (language, prompt format); delta_specific = delta - delta_control removes that.
  p_target_answers / p_source_answers  probability on the target-language answers' first
                 tokens vs the English answers' first tokens after patching. If the English
                 tokens take over, the patch broke target-language output rather than fixing it.

Positions
  "last"     the final prompt token (where the answer is predicted; lens.csv reads the same spot)
  "subject"  the last token of the entity's name in the test question (where factual recall
             is usually localized, e.g. Meng et al. 2022 and Fierro et al. 2025)

Layer indexing matches hooks.py: 0 = embedding output, l = output of decoder layer l.
Each target prompt is run once per layer as one batch (one row per layer, no padding needed).
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch

from .hooks import capture_residuals, parts
from .schema import Fact
from .scoring import answer_ids, score_logprobs


def _modules(model):
    """Module whose output is residual layer l, for l = 0..L."""
    embed, layers, _, _ = parts(model)
    return [embed] + list(layers)


@contextmanager
def patched(model, edits):
    """While active, replace residual vectors in every forward pass.

    edits: iterable of (layer, row, pos, vector): batch row `row`, token position `pos`
    (negative counts from the end), residual layer `layer` gets `vector` [d].
    Hooks are always removed on exit, even after an error."""
    by_layer = defaultdict(list)
    for layer, row, pos, vec in edits:
        by_layer[int(layer)].append((row, pos, vec))
    mods = _modules(model)

    def make(items):
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            for row, pos, vec in items:
                h[row, pos] = vec.to(device=h.device, dtype=h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    handles = []
    try:
        for layer, items in by_layer.items():
            handles.append(mods[layer].register_forward_hook(make(items)))
        yield
    finally:
        for h in handles:
            h.remove()


def _device(model):
    return next(model.parameters()).device


def _last_logprobs(model, input_ids) -> torch.Tensor:
    """Next-token log-probs at the last position, [B, V] float32."""
    try:
        out = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
    except TypeError:       # older transformers: no logits_to_keep
        out = model(input_ids=input_ids, use_cache=False)
    return out.logits[:, -1].float().log_softmax(-1)


@torch.no_grad()
def residuals(model, tok, prompt: str):
    """One unpatched run. Returns (resid [L+1, T, d], final log-probs [V], input_ids [T])."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(_device(model))
    resid, logits = capture_residuals(model, ids)
    return resid[:, 0], logits[0, -1].float().log_softmax(-1), ids[0]


@torch.no_grad()
def patch_layers(model, tok, prompt: str, vectors: torch.Tensor, pos: int, layers=None,
                 chunk: int | None = None) -> torch.Tensor:
    """Run `prompt` once per layer l in `layers`, with residual layer l at token `pos` replaced
    by vectors[l]. vectors: [L+1, d], e.g. a source run's resid[:, src_pos].
    Returns final-position log-probs [len(layers), V]. `chunk` caps the batch size (memory)."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(_device(model))
    layers = list(range(vectors.shape[0])) if layers is None else [int(l) for l in layers]
    chunk = chunk or len(layers)
    out = []
    for i in range(0, len(layers), chunk):
        ls = layers[i:i + chunk]
        with patched(model, [(l, r, pos, vectors[l]) for r, l in enumerate(ls)]):
            out.append(_last_logprobs(model, ids.repeat(len(ls), 1)))
    return torch.cat(out)


def _entity(fact: Fact, lang: str) -> str | None:
    m = fact.meta or {}
    if isinstance(m.get("subject"), dict):          # toy counterfactuals: per-language subject
        return m["subject"].get(lang)
    return m.get("entity")                          # fictional facts: one name, all languages


def subject_position(tok, prompt: str, fact: Fact, lang: str) -> int | None:
    """Token index of the last token of the entity's name inside the *test question* (not the
    worked examples). Tries the full name, then its longer words, longest first, since some
    languages inflect or reorder names (Finnish "Lumerian" still contains "Lumeria").
    None if the name isn't found."""
    name = _entity(fact, lang)
    if not name:
        return None
    question = ((fact.meta or {}).get("questions") or {}).get(lang)
    lo = prompt.rfind(question) if question else -1
    lo, hi = (lo, lo + len(question)) if lo >= 0 else (0, len(prompt))
    region = prompt[lo:hi]
    cands = [name] + sorted({w.strip(".,'’") for w in name.split() if len(w) >= 4} - {name}, key=len, reverse=True)
    offsets = tok(prompt, return_offsets_mapping=True)["offset_mapping"]
    for cand in cands:
        j = region.rfind(cand)
        if j < 0:
            continue
        end = lo + j + len(cand) - 1                     # last character of the name
        for t, (a, b) in enumerate(offsets):
            if a <= end < b:
                return t
    return None


def run_patching(model, tok, facts: list[Fact], cases: pd.DataFrame, model_tag: str = "model",
                 positions=("last", "subject"), source_lang: str = "en", seed: int = 0,
                 chunk: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Patch English -> target for every case and layer. `cases` needs fact_id and lang
    (prompt_idx optional), e.g. results/.../patch_cases.csv. `model` is the trained model
    (adapter attached). Returns one row per case x position x layer; see the module docstring.
    Skipped cases (first token can't separate old/new, or entity name not found) are listed
    in df.attrs["skipped"]."""
    fmap = {f.fact_id: f for f in facts}
    rng = np.random.default_rng(seed)
    cache = {}
    skipped = []

    def source(fid):
        if fid not in cache:
            p = fmap[fid].prompts[source_lang][0]
            resid, lp, _ = residuals(model, tok, p)
            cache[fid] = (resid, lp, p)
        return cache[fid]

    rows = []
    for n, case in enumerate(cases.itertuples(index=False), 1):
        f, lang = fmap[case.fact_id], case.lang
        k = int(getattr(case, "prompt_idx", 0))
        tprompt = f.prompts[lang][k]
        ids_t = answer_ids(tok, f, lang, tprompt)
        if not ids_t.separable:
            skipped.append((f.fact_id, lang, "old/new share a first token"))
            continue
        _, t_lp, _ = residuals(model, tok, tprompt)
        base = score_logprobs(t_lp, ids_t)
        s_resid, s_lp, sprompt = source(f.fact_id)
        ids_s = answer_ids(tok, f, source_lang, sprompt)
        src = score_logprobs(s_lp, ids_s)
        others = [x for x in fmap if x != f.fact_id]
        ctrl = others[int(rng.integers(len(others)))]
        c_resid, _, cprompt = source(ctrl)
        tgt_tok = sorted(set(ids_t.new) | set(ids_t.old or []))
        src_tok = sorted((set(ids_s.new) | set(ids_s.old or [])) - set(tgt_tok))

        for position in positions:
            if position == "last":
                tpos = spos = cpos = -1
            elif position == "subject":
                tpos = subject_position(tok, tprompt, f, lang)
                spos = subject_position(tok, sprompt, f, source_lang)
                cpos = subject_position(tok, cprompt, fmap[ctrl], source_lang)
                if None in (tpos, spos, cpos):
                    skipped.append((f.fact_id, lang, "entity name not found for subject patch"))
                    continue
            else:
                raise ValueError(f"unknown position {position!r}")
            lp_p = patch_layers(model, tok, tprompt, s_resid[:, spos], tpos, chunk=chunk)
            lp_c = patch_layers(model, tok, tprompt, c_resid[:, cpos], tpos, chunk=chunk)
            sp, sc = score_logprobs(lp_p, ids_t), score_logprobs(lp_c, ids_t)
            p_tgt = lp_p[:, tgt_tok].logsumexp(-1).exp()
            p_src = lp_p[:, src_tok].logsumexp(-1).exp() if src_tok else torch.zeros_like(p_tgt)
            for layer in range(lp_p.shape[0]):
                rows.append(dict(
                    model=model_tag, fact_id=f.fact_id, lang=lang, prompt_idx=k, position=position,
                    layer=layer, control_fact=ctrl, D_base=base["D"].item(), D_source=src["D"].item(),
                    D_patch=sp["D"][layer].item(), D_control=sc["D"][layer].item(),
                    top1_new_patch=bool(sp["top1_new"][layer]),
                    p_target_answers=p_tgt[layer].item(), p_source_answers=p_src[layer].item()))
        if verbose:
            print(f"\r{n}/{len(cases)} cases", end="")
    if verbose:
        print(f"\r{len(cases)} cases, {len(skipped)} skipped")
    df = pd.DataFrame(rows)
    if len(df):
        df["delta"] = df.D_patch - df.D_base
        df["delta_control"] = df.D_control - df.D_base
        df["delta_specific"] = df.delta - df.delta_control
        df["flipped"] = (df.D_base <= 0) & (df.D_patch > 0)
        df["flipped_control"] = (df.D_base <= 0) & (df.D_control > 0)
    df.attrs["skipped"] = skipped
    return df


def summarize_patching(df: pd.DataFrame) -> pd.DataFrame:
    """Per position x language x layer: mean delta (fact patch), mean delta_control, their
    difference, and how often the patch flips the answer to the new one (vs the control)."""
    return (df.groupby(["position", "lang", "layer"])
              .agg(n=("fact_id", "nunique"), delta=("delta", "mean"), delta_control=("delta_control", "mean"),
                   delta_specific=("delta_specific", "mean"), flip_rate=("flipped", "mean"),
                   flip_rate_control=("flipped_control", "mean"),
                   p_target=("p_target_answers", "median"), p_source=("p_source_answers", "median"))
              .reset_index())


def peak_layers(df: pd.DataFrame) -> pd.DataFrame:
    """Per position x language: the layer where the fact-specific effect (delta_specific) peaks."""
    s = summarize_patching(df)
    idx = s.groupby(["position", "lang"]).delta_specific.idxmax()
    return s.loc[idx, ["position", "lang", "layer", "n", "delta", "delta_control", "delta_specific",
                       "flip_rate", "flip_rate_control"]].reset_index(drop=True)
