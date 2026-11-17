"""Sanity checks for activation patching (xling/patching.py). Run from the project root:

    python -m tests.check_patching                       # Qwen3-0.6B-Base, Rohan's few-shot facts (fast)
    python -m tests.check_patching --base Qwen/Qwen3-1.7B-Base --adapter adapters/rohan/round2

or from a notebook with an already-loaded model: `run_checks(model, tok, facts)`.

Checks:
  1. the entity's last token is found in the test question for (almost) every fact and language
  2. patching a run with its OWN vectors changes nothing, at every layer (last and subject position)
  3. patching the last layer at the last position reproduces the source run's output exactly
     (so layer indexing and the hook's replacement are right)
  4. patching layer 0 (embeddings) equals running the model on edited input embeddings
  5. a patch only affects what it should: earlier layers untouched, other positions at the
     patched layer untouched, earlier positions untouched at every later layer (causal masking)
  6. one batched run (one row per layer) equals separate single-layer runs
  7. all hooks are removed afterwards and the model's output is back to normal
  8. run_patching's baseline D equals the lens pipeline's D for the same prompt, and its
     last-layer patch equals the source run scored with the target's answer tokens
"""
import argparse

import torch

from xling.hooks import capture_residuals, final_position_lens, parts
from xling.modeling import attach_adapter, load_model
from xling.patching import _modules, patch_layers, patched, residuals, run_patching, subject_position
from xling.schema import load_facts
from xling.scoring import answer_ids, score_logprobs

LANGS = ("en", "fr", "fi", "zh")


def _pdiff(a, b):
    """Max difference in probability (robust to bf16 noise on tokens with ~0 probability)."""
    return (a.exp() - b.exp()).abs().max().item()


def run_checks(model, tok, facts, tol=None):
    model.eval()
    dtype = next(model.parameters()).dtype
    tol = tol or (1e-4 if dtype == torch.float32 else 2e-2)
    dev = next(model.parameters()).device

    # 1. subject positions
    found, total, examples = 0, 0, []
    for f in facts:
        for lang in LANGS:
            if lang not in f.prompts:
                continue
            p = f.prompts[lang][0]
            t = subject_position(tok, p, f, lang)
            total += 1
            if t is not None:
                found += 1
                if len(examples) < 4 and lang in ("fi", "zh"):
                    examples.append(f"{f.fact_id}/{lang}: {tok.decode(tok(p).input_ids[t])!r}")
    assert found >= 0.8 * total, f"entity found in only {found}/{total} prompts"
    print(f"1. subject token found in {found}/{total} prompts, e.g. {', '.join(examples)}")

    f = facts[0]
    tp, sp_ = f.prompts["fr"][0], f.prompts["en"][0]
    t_resid, t_lp, t_ids = residuals(model, tok, tp)
    s_resid, s_lp, _ = residuals(model, tok, sp_)
    L1 = t_resid.shape[0]
    tsub = subject_position(tok, tp, f, "fr")
    ssub = subject_position(tok, sp_, f, "en")

    # 2. self-patch is a no-op
    worst = 0.0
    for pos in (-1, tsub):
        out = patch_layers(model, tok, tp, t_resid[:, pos], pos)
        worst = max(worst, max(_pdiff(out[l], t_lp) for l in range(L1)))
    assert worst < tol, worst
    print(f"2. self-patch leaves the output unchanged at all {L1} layers (max prob diff {worst:.1e})")

    # 3. last layer, last position from the source = the source's own output
    out = patch_layers(model, tok, tp, s_resid[:, -1], -1, layers=[L1 - 1])[0]
    d = _pdiff(out, s_lp)
    assert d < tol, d
    print(f"3. last-layer patch reproduces the English run's output (max prob diff {d:.1e})")

    # 4. layer 0 patch == edited input embeddings
    embed = parts(model)[0]
    with torch.no_grad():
        e = embed(t_ids[None]).clone()
        e[0, tsub] = s_resid[0, ssub].to(e.dtype)
        manual = model(inputs_embeds=e, use_cache=False).logits[0, -1].float().log_softmax(-1)
    out = patch_layers(model, tok, tp, s_resid[:, ssub], tsub, layers=[0])[0]
    d = _pdiff(out, manual)
    assert d < tol, d
    print(f"4. embedding-layer patch == edited inputs_embeds run (max prob diff {d:.1e})")

    # 5. locality and causality
    l, vec = L1 // 2, s_resid[L1 // 2, ssub]
    with patched(model, [(l, 0, tsub, vec)]):
        r, _ = capture_residuals(model, t_ids[None])
    r, base = r[:, 0].float(), t_resid.float()
    eps = 1e-5 if dtype == torch.float32 else 1e-2
    assert (r[:l] - base[:l]).abs().max() < eps, "earlier layers changed"
    others = [i for i in range(r.shape[1]) if i != tsub]
    assert (r[l, others] - base[l, others]).abs().max() < eps, "other positions changed at the patched layer"
    assert (r[l, tsub] - vec.float()).abs().max() < eps, "patched vector not in place"
    assert (r[l + 1:, :tsub] - base[l + 1:, :tsub]).abs().max() < eps, "earlier positions changed downstream"
    moved = (r[l + 1:, tsub + 1:] - base[l + 1:, tsub + 1:]).abs().max().item()
    assert moved > eps, "patch had no downstream effect at all"
    print(f"5. patch at layer {l}, token {tsub}: earlier layers/positions untouched, later positions change (max {moved:.2f})")

    # 6. batched over layers == one layer at a time
    ls = [0, l, L1 - 2]
    batched = patch_layers(model, tok, tp, s_resid[:, -1], -1, layers=ls)
    single = torch.stack([patch_layers(model, tok, tp, s_resid[:, -1], -1, layers=[x])[0] for x in ls])
    d = _pdiff(batched, single)
    assert d < tol * 5, d
    print(f"6. batched layers == separate runs (max prob diff {d:.1e})")

    # 7. hooks gone
    assert all(len(m._forward_hooks) == 0 for m in _modules(model)), "a patch hook was left behind"
    _, again, _ = residuals(model, tok, tp)
    d = _pdiff(again, t_lp)
    assert d < tol, d
    print("7. no hooks left; output back to the unpatched run")

    # 8. consistency with the lens pipeline (same D units)
    import pandas as pd
    cases = pd.DataFrame([{"fact_id": x.fact_id, "lang": "fr", "prompt_idx": 0} for x in facts[:3]])
    df = run_patching(model, tok, facts, cases, positions=("last",), verbose=False)
    row = df[df.layer == 0].iloc[0]
    fx = next(x for x in facts if x.fact_id == row.fact_id)
    p = fx.prompts["fr"][0]
    ids = answer_ids(tok, fx, "fr", p)
    lens_D = score_logprobs(final_position_lens(model, tok, [p])[1][0], ids)["D"].item()
    assert abs(row.D_base - lens_D) < 0.1, (row.D_base, lens_D)
    s_lp2 = residuals(model, tok, fx.prompts["en"][0])[1]
    expect = score_logprobs(s_lp2, ids)["D"].item()
    got = df[df.layer == df.layer.max()].iloc[0].D_patch
    assert abs(got - expect) < 0.1, (got, expect)
    print(f"8. run_patching: {len(df)} rows; baseline D matches the lens ({row.D_base:.2f} vs {lens_D:.2f}); "
          f"last-layer patch = English output on French answers ({got:.2f} vs {expect:.2f})")
    print("all patching checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--facts", default="data/rohan_facts_fewshot.jsonl")
    ap.add_argument("--adapter", default=None)
    a = ap.parse_args()
    model, tok = load_model(a.base)
    if a.adapter:
        model = attach_adapter(model, a.adapter)
    run_checks(model, tok, load_facts(a.facts))


if __name__ == "__main__":
    main()
