"""facts x languages x models -> tables. The contract between this analysis and the rest.

Outputs (CSV, in --out):
  accuracy.csv  one row per (model, fact_id, lang, prompt_idx): final-layer metrics from
                scoring.py, plus full-answer (multi-token) log-probs.
  lens.csv      the same keys x layer: logp_new, logp_old, D, rank_new per layer (logit lens,
                final prompt position). The last layer equals the model's real output.
  status.csv    one row per (edit model, fact_id, lang, prompt_idx): transfer status and where
                in depth the edit acts (onset/peak/final, see delta_profile).
                Activation patching takes its cases from here
                (status == "transfer_fail").
  run_info.json what was run (models, adapters, fact counts, library versions).

Usage:
  python -m xling.evaluate --facts data/toy_facts.jsonl --base Qwen/Qwen3-1.7B-Base \
      --adapter adapters/toy --out results/toy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import LANGS
from .hooks import final_position_lens
from .modeling import attach_adapter, load_model, read_adapter_meta
from .schema import Fact, load_facts, validate_facts
from .scoring import answer_ids, continuation_logprobs, score_logprobs

KEYS = ["model", "fact_id", "relation", "split", "lang", "prompt_idx"]
LENS_COLS = ["logp_new", "logp_old", "D", "rank_new"]
STATUS_ORDER = ["ref_unknown", "source_ok", "source_fail", "en_fail", "transfer_ok", "transfer_fail"]


def _val(t: torch.Tensor):
    v = t.item()
    return bool(v) if t.dtype == torch.bool else v


@torch.no_grad()
def evaluate(model, tok, facts: list[Fact], model_tag: str, langs=LANGS, batch_size: int = 16,
             full_answer: bool = True, check_lens: bool = True):
    """Score every prompt of every fact. Returns (accuracy_df, lens_df)."""
    items = [(f, lang, k, p) for f in facts for lang in langs for k, p in enumerate(f.prompts[lang])]
    acc_rows, lens_rows = [], []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        lens, final = final_position_lens(model, tok, [p for *_, p in chunk])     # [L+1, B, V], [B, V]
        if check_lens:  # the lens at the last layer must reproduce the model's real output
            err = (lens[-1] - final).abs().max().item()
            if err > 1e-2:
                raise RuntimeError(f"logit lens last layer differs from model output by {err:.3g}")
        for b, (f, lang, k, prompt) in enumerate(chunk):
            ids = answer_ids(tok, f, lang, prompt)
            s = score_logprobs(lens[:, b], ids)                                  # each [L+1]
            key = dict(model=model_tag, fact_id=f.fact_id, relation=f.relation, split=f.split,
                       lang=lang, prompt_idx=k)
            acc_rows.append({**key, **{m: _val(v[-1]) for m, v in s.items()},
                             "top1_token": tok.decode([int(lens[-1, b].argmax())]),
                             "n_distractors": len(ids.distractors), "lens_ok": ids.separable})
            per_layer = {m: v.tolist() for m, v in s.items() if m in LENS_COLS}
            for layer in range(lens.shape[0]):
                lens_rows.append({**key, "layer": layer, **{m: per_layer[m][layer] for m in per_layer},
                                  "lens_ok": ids.separable})
    acc = pd.DataFrame(acc_rows)
    if full_answer:
        acc = _add_full_answer(model, tok, acc, items, batch_size)
    return acc, pd.DataFrame(lens_rows)


def _add_full_answer(model, tok, acc, items, batch_size):
    """Teacher-forced log-prob of the whole answer string (best alias), for multi-token answers."""
    jobs = []   # (row, which, prompt, continuation)
    for row, (f, lang, k, prompt) in enumerate(items):
        for which, group in (("new", f.new), ("old", f.old)):
            if group is not None:
                jobs += [(row, which, prompt, a) for a in group[lang]]
    scores = continuation_logprobs(model, tok, [j[2] for j in jobs], [j[3] for j in jobs], batch_size)
    best = {}
    for (row, which, _, _), s in zip(jobs, scores):
        best[(row, which)] = max(best.get((row, which), -np.inf), s)
    acc = acc.copy()
    acc["full_logp_new"] = [best.get((r, "new"), np.nan) for r in range(len(acc))]
    acc["full_logp_old"] = [best.get((r, "old"), np.nan) for r in range(len(acc))]
    acc["full_prefers_new"] = acc.full_logp_new > acc.full_logp_old

    # Length-normalized score of the first (canonical) alias only: mean log-prob per answer token.
    # With data/convert_rohan.py's "qa" prompts this is exactly the margin in Rohan's notebook.
    canon = [(row, which, prompt, group[lang][0])
             for row, (f, lang, k, prompt) in enumerate(items)
             for which, group in (("new", f.new), ("old", f.old)) if group is not None]
    means = continuation_logprobs(model, tok, [j[2] for j in canon], [j[3] for j in canon],
                                  batch_size, mean=True)
    norm = {(row, which): s for (row, which, _, _), s in zip(canon, means)}
    acc["norm_logp_new"] = [norm.get((r, "new"), np.nan) for r in range(len(acc))]
    acc["norm_logp_old"] = [norm.get((r, "old"), np.nan) for r in range(len(acc))]
    acc["norm_margin"] = acc.norm_logp_new - acc.norm_logp_old
    acc["norm_prefers_new"] = acc.norm_margin > 0
    return acc


PROFILE_COLS = ["onset_layer", "peak_layer", "peak_dD", "final_dD", "final_D"]


def delta_profile(D_after: np.ndarray, D_before: np.ndarray, margin: float = 1.0) -> tuple:
    """Summarize where in depth the edit acts, from dD = D_after - D_before per layer.

    Measured against the base model because early-layer lens readouts carry token priors
    (D can be > 0 before any edit). Returns:
      onset_layer  first layer from which dD stays > margin through the output (NaN if it doesn't)
      peak_layer   layer of the largest dD;  peak_dD its value
      final_dD     dD at the output;  final_D  D at the output (> 0: edit preferred)
    Rough failure-mode reading: small peak_dD -> the edit never reaches this language (encoding);
    large dD but final_D < 0 -> the original answer still wins (retrieval);
    large peak_dD but much smaller final_dD -> lost in the last layers (decoding).
    """
    dd = D_after - D_before
    above = dd > margin
    if not above[-1]:
        onset = np.nan
    else:
        below = np.where(~above)[0]
        onset = float(below[-1] + 1) if len(below) else 0.0
    peak = int(np.nanargmax(dd))
    return onset, float(peak), float(dd[peak]), float(dd[-1]), float(D_after[-1])


def transfer_status(acc: pd.DataFrame, lens: pd.DataFrame, edit_tag: str, ref_tag: str = "base",
                    target: str = "new", source: str = "en", criterion: str = "top1",
                    margin: float = 1.0) -> pd.DataFrame:
    """Classify each (fact, lang, prompt) for one trained model against its reference model,
    and add the layer profile from `delta_profile` (onset/peak/final of the training's effect).

    edit_tag  the model being judged (e.g. "toy", "round2")
    ref_tag   the model it was trained from ("base" for counterfactual edits, "round1" for
              the fictional round-2 update)
    target    which answer the training taught: "new" (edits, round 2) or "old" (round 1,
              which teaches the original fictional fact)
    criterion "top1": correct = the top-1 token is the answer.
              "prefer": correct = the answer beats the alternative on first-token D (D > 0 or D < 0).
              "full":   the whole answer (sum of token log-probs, best alias) beats the other one.
              "norm":   the canonical answer's mean per-token log-prob beats the other one
                        (Rohan's forced-choice metric; needs full_answer=True).
    Rows with lens_ok == False (old and new share their first token) get no layer profile.
    Status:
      ref_unknown    the reference model doesn't give the answer being replaced, here or in the
                     source language (e.g. the base model doesn't know the real fact, or round 1
                     never reached this language). Only applies when target == "new".
      source_ok/fail the source-language row itself: learned or not
      en_fail        target-language row whose fact wasn't learned in the source language
                     (majority of source prompts), so transfer can't be judged
      transfer_ok/fail  learned in the source language; did this language follow?
    """
    if target not in ("new", "old"):
        raise ValueError("target must be 'new' or 'old'")

    if criterion not in ("top1", "prefer", "full", "norm"):
        raise ValueError(f"unknown criterion {criterion!r}")

    def correct(df, which):
        if criterion == "top1":
            return df[f"top1_{which}"].fillna(False).astype(bool)
        if criterion == "full":
            m = df.full_logp_new - df.full_logp_old
        elif criterion == "norm":
            m = df.norm_margin
        else:
            m = df.D
        return m > 0 if which == "new" else m < 0

    k = ["fact_id", "lang", "prompt_idx"]
    edit = acc[acc.model == edit_tag].set_index(k)
    ref = acc[acc.model == ref_tag].set_index(k).reindex(edit.index)
    out = edit[["relation", "split"]].copy()
    out["edit_correct"] = correct(edit, target)
    # The reference already gives the trained answer: e.g. the base model guesses "blue" for a
    # fictional moon before round 1. Such rows say little about learning; filter on this column.
    out["ref_already_correct"] = correct(ref, target).to_numpy()
    if target == "new" and "logp_old" in ref:   # the reference must hold the answer being replaced
        out["ref_correct"] = np.where(ref["logp_old"].notna(), correct(ref, "old"), True)
    else:
        out["ref_correct"] = True

    langs_idx = out.index.get_level_values("lang")
    src = out[langs_idx == source]
    src_learned = src.groupby(level="fact_id").edit_correct.mean() >= 0.5
    src_known = src.groupby(level="fact_id").ref_correct.mean() >= 0.5
    fid = out.index.get_level_values("fact_id")
    out["source_learned"] = src_learned.reindex(fid).fillna(False).to_numpy()
    out["source_known"] = src_known.reindex(fid).fillna(False).to_numpy()

    status = np.where(~(out.ref_correct & out.source_known), "ref_unknown",
             np.where(langs_idx == source, np.where(out.edit_correct, "source_ok", "source_fail"),
             np.where(~out.source_learned, "en_fail",
             np.where(out.edit_correct, "transfer_ok", "transfer_fail"))))
    out["status"] = status

    # Profile of D toward the trained answer: D for target "new", -D for target "old".
    sign = 1.0 if target == "new" else -1.0
    after = lens[lens.model == edit_tag][k + ["layer", "D"]]
    before = lens[lens.model == ref_tag][k + ["layer", "D"]].rename(columns={"D": "D_ref"})
    merged = after.merge(before, on=k + ["layer"], how="left").sort_values(k + ["layer"])
    prof = {key: delta_profile(sign * g.D.to_numpy(), sign * g.D_ref.to_numpy(), margin)
            for key, g in merged.groupby(k)}
    missing = (np.nan,) * len(PROFILE_COLS)
    lens_ok = edit["lens_ok"] if "lens_ok" in edit else pd.Series(True, index=edit.index)
    out["lens_ok"] = lens_ok.astype(bool)
    for j, col in enumerate(PROFILE_COLS):
        out[col] = [prof.get(idx, missing)[j] if ok else np.nan for idx, ok in zip(out.index, out.lens_ok)]
    out.insert(0, "model", edit_tag)
    out.insert(1, "reference", ref_tag)
    out.insert(2, "target", target)
    return out.reset_index()


def failure_modes(status: pd.DataFrame, margin: float = 1.0, keep: float = 0.5) -> pd.DataFrame:
    """Rough proposal-style label for each transfer_fail row with a layer profile:
      encoding   peak_dD < margin: the training never moved this language's lens readout
      decoding   final_dD < keep * peak_dD: the effect was there mid-network but lost late
      retrieval  otherwise: the effect reaches the output but the old answer still wins
    A heuristic on first-token lens curves, not a causal claim; activation patching tests it."""
    st = status[(status.status == "transfer_fail") & status.peak_dD.notna()].copy()
    st["failure_mode"] = np.select([st.peak_dD < margin, st.final_dD < keep * st.peak_dD],
                                   ["encoding", "decoding"], "retrieval")
    return st


def summarize(status: pd.DataFrame) -> pd.DataFrame:
    """Per edit model x language: status counts and transfer rate = ok / (ok + fail)."""
    counts = (status.groupby(["model", "lang", "status"]).size().unstack(fill_value=0)
              .reindex(columns=STATUS_ORDER, fill_value=0))
    ok = counts.transfer_ok + counts.source_ok
    fail = counts.transfer_fail + counts.source_fail
    counts["success_rate"] = (ok / (ok + fail)).round(3)
    return counts


def run(facts_path, base: str, adapters=(), out_dir="results", langs=LANGS, batch_size: int = 16,
        full_answer: bool = True, model=None, tok=None, criterion: str = "top1"):
    """Evaluate the base model on all facts, then each adapter on the facts it was trained on.
    Pass an already-loaded base `model`/`tok` to avoid loading it twice.

    Each adapter's meta.json may name its "reference" model (default "base") and its "target"
    ("new" or "old", default "new"); status.csv judges the adapter against that reference.
    List a reference adapter before the adapters that use it, e.g. [round1, round2]."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    facts = load_facts(facts_path)
    if model is None:
        model, tok = load_model(base)
    problems = validate_facts(facts, tok, langs)
    for p in problems:
        print("WARNING", p)

    print(f"base: {len(facts)} facts")
    a, l = evaluate(model, tok, facts, "base", langs, batch_size, full_answer)
    accs, lenses = [a], [l]
    statuses, adapter_info = [], []
    for path in adapters:
        meta = read_adapter_meta(path)
        tag = meta["name"]
        wanted = set(meta.get("fact_ids") or [f.fact_id for f in facts])
        subset = [f for f in facts if f.fact_id in wanted]
        print(f"{tag}: {len(subset)} facts")
        pm = attach_adapter(model, path)
        a, l = evaluate(pm, tok, subset, tag, langs, batch_size, full_answer)
        model = pm.unload()     # remove the LoRA layers again before the next adapter
        accs.append(a)
        lenses.append(l)
        adapter_info.append({"path": str(path), **meta})

    acc, lens = pd.concat(accs, ignore_index=True), pd.concat(lenses, ignore_index=True)
    for info in adapter_info:
        ref, target = info.get("reference", "base"), info.get("target", "new")
        if ref not in set(acc.model):
            print(f"WARNING {info['name']}: reference '{ref}' wasn't evaluated; using 'base'")
            ref = "base"
        statuses.append(transfer_status(acc, lens, info["name"], ref, target, criterion=criterion))
    status = pd.concat(statuses, ignore_index=True) if statuses else pd.DataFrame()

    acc.to_csv(out / "accuracy.csv", index=False)
    lens.to_csv(out / "lens.csv", index=False)
    status.to_csv(out / "status.csv", index=False)
    import transformers, peft
    info = dict(facts=str(facts_path), n_facts=len(facts), base=base, adapters=adapter_info,
                langs=list(langs), criterion=criterion, full_answer=full_answer,
                validation_problems=problems, seconds=round(time.time() - t0, 1),
                versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                              peft=peft.__version__))
    (out / "run_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}/accuracy.csv, lens.csv, status.csv ({info['seconds']}s)")
    return acc, lens, status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--adapter", action="append", default=[], help="adapter dir (repeatable)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--criterion", choices=["top1", "prefer", "full", "norm"], default="top1")
    ap.add_argument("--no-full-answer", action="store_true")
    a = ap.parse_args()
    run(a.facts, a.base, a.adapter, a.out, tuple(a.langs.split(",")), a.batch_size,
        not a.no_full_answer, criterion=a.criterion)


if __name__ == "__main__":
    main()
