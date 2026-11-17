"""Sanity checks for the shared pipeline code. Run from the project root:

    python -m tests.check_pipeline                       # uses Qwen3-0.6B-Base (fast)
    python -m tests.check_pipeline --base Qwen/Qwen3-1.7B-Base

Checks:
  1. the facts file validates (structure + no shared first tokens)
  2. the logit lens at the last layer reproduces the model's real output
  3. batched (left-padded) results match one-prompt-at-a-time results
  4. score_logprobs agrees with a hand computation
  5. continuation_logprobs agrees with a hand computation
"""
import argparse

import torch

from xling.hooks import capture_residuals, encode, final_position_lens, logit_lens
from xling.modeling import load_model
from xling.schema import load_facts, validate_facts
from xling.scoring import answer_ids, continuation_logprobs, score_logprobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--facts", default="data/toy_facts.jsonl")
    a = ap.parse_args()

    model, tok = load_model(a.base)
    facts = load_facts(a.facts)
    tol = 1e-3 if next(model.parameters()).dtype == torch.float32 else 5e-2

    problems = validate_facts(facts, tok)
    assert not problems, "\n".join(problems)
    print(f"1. {len(facts)} facts validate")

    ids, mask, pos = encode(tok, ["The Eiffel Tower is located in the city of"], model.device)
    resid, logits = capture_residuals(model, ids, mask, pos)
    err = (logit_lens(model, resid[:, 0, -1]) [-1] - logits[0, -1].float().log_softmax(-1)).abs().max().item()
    assert err < tol, err
    print(f"2. lens last layer == model output (max diff {err:.1e})")

    prompts = [p for f in facts for l in ("en", "zh", "fi") for p in f.prompts[l]][:8]
    lens_b, final_b = final_position_lens(model, tok, prompts)
    worst = 0.0
    for i, p in enumerate(prompts):
        lens_1, final_1 = final_position_lens(model, tok, [p])
        worst = max(worst, (lens_b[:, i] - lens_1[:, 0]).abs().max().item(),
                    (final_b[i] - final_1[0]).abs().max().item())
    assert worst < tol * 10, worst
    print(f"3. batched == single-prompt over {len(prompts)} prompts (max diff {worst:.1e})")

    f = facts[0]
    p = f.prompts["en"][0]
    ai = answer_ids(tok, f, "en", p)
    lp = final_position_lens(model, tok, [p])[1][0]
    s = score_logprobs(lp, ai)
    manual_D = lp[ai.new].logsumexp(0) - lp[ai.old].logsumexp(0)
    assert abs(s["D"].item() - manual_D.item()) < 1e-4
    assert bool(s["top1_old"]) == (lp.argmax().item() in ai.old)
    print(f"4. score_logprobs matches hand computation (D = {s['D'].item():.2f} for {f.fact_id})")

    cont = f.new["en"][0]
    got = continuation_logprobs(model, tok, [p], [cont])[0]
    p_ids = tok(p).input_ids
    c_ids = tok(p + cont).input_ids[len(p_ids):]
    with torch.no_grad():
        full = model(input_ids=torch.tensor([p_ids + c_ids], device=model.device)).logits[0].float().log_softmax(-1)
    manual = sum(full[len(p_ids) - 1 + j, t].item() for j, t in enumerate(c_ids))
    assert abs(got - manual) < tol * 10, (got, manual)
    print(f"5. continuation_logprobs matches hand computation ({got:.3f})")
    print("all checks passed")


if __name__ == "__main__":
    main()
