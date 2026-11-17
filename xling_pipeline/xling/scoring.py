"""Answer token ids and metrics.

Shared with activation patching: score a patched run with `score_logprobs` so that the
patching effect (Delta_l in the proposal) and the logit-lens D_l are the same quantity.

Definitions (each works on log-probs with any leading shape, e.g. [L+1, V] from the lens
or [V] from the model's output):
  logp_new   log-prob of the new answer: logsumexp over the first tokens of all its aliases
  logp_old   same for the original answer (absent for fictional facts)
  D          logp_new - reference, where reference = logp_old if the fact has an old answer,
             else the log-mean-exp over distractors. D > 0 means the edit is preferred.
  top1_new   the single most likely token is a first token of the new answer
  top1_old   same for the old answer
  rank_new   rank of the new answer's best first token in the whole vocabulary (0 = top)
  cand_rank_new  1-based rank of the new answer among {new, old, distractors}
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .schema import Fact


@dataclass
class AnswerIds:
    new: list[int]
    old: list[int] | None
    distractors: list[list[int]]

    @property
    def separable(self) -> bool:
        """False when old and new still share a first token (e.g. "Lake Silver" / "Lake Amber"):
        the first-token metrics and the lens can't tell them apart, only full-answer scores can."""
        return self.old is None or not set(self.new) & set(self.old)


def _drop_fragments(tok, ids: list[int]) -> list[int]:
    """Drop first tokens that are partial UTF-8 bytes (decode to U+FFFD), e.g. the "Ġç" that
    " 绿色" starts with, which is shared by thousands of Chinese characters. Kept if nothing else."""
    clean = [i for i in ids if "�" not in tok.decode([i])]
    return clean or ids


def first_token_ids(tok, prompt: str, answers: list[str]) -> list[int]:
    """First token of each answer as tokenized right after `prompt` (deduplicated, sorted)."""
    p = tok(prompt).input_ids
    ids = set()
    for a in answers:
        full = tok(prompt + a).input_ids
        if full[:len(p)] == p and len(full) > len(p):
            ids.add(full[len(p)])
        else:  # the prompt's last token merged with the answer; fall back to the answer alone
            ids.add(tok(a).input_ids[0])
    return sorted(ids)


def answer_ids(tok, fact: Fact, lang: str, prompt: str) -> AnswerIds:
    """Token ids for one (fact, language, prompt).

    First tokens shared by the old and new answers (e.g. " K" from capitalized "Kivi" and
    "Kupari") can't tell them apart, so they're removed from both sides, as long as each side
    keeps at least one token. Distractors that share a first token with either answer are dropped.
    """
    new = _drop_fragments(tok, first_token_ids(tok, prompt, fact.new[lang]))
    old = _drop_fragments(tok, first_token_ids(tok, prompt, fact.old[lang])) if fact.old else None
    if old is not None:
        shared = set(new) & set(old)
        if shared and set(new) - shared and set(old) - shared:
            new = [t for t in new if t not in shared]
            old = [t for t in old if t not in shared]
    taken = set(new) | set(old or [])
    dis = []
    for d in (fact.distractors or {}).get(lang, []):
        ids = first_token_ids(tok, prompt, [d] if isinstance(d, str) else list(d))
        if not set(ids) & taken:
            dis.append(ids)
            taken |= set(ids)
    return AnswerIds(new, old, dis)


def score_logprobs(lp: torch.Tensor, ids: AnswerIds) -> dict[str, torch.Tensor]:
    """Metrics from log-probs `lp` [..., V]. Every returned tensor has shape [...]."""
    new = lp[..., ids.new].logsumexp(-1)
    top = lp.argmax(-1)
    best_new = lp[..., ids.new].max(-1, keepdim=True).values
    out = {
        "logp_new": new,
        "top1_new": torch.isin(top, torch.tensor(ids.new, device=lp.device)),
        "rank_new": (lp > best_new).sum(-1),
    }
    cands = [new]
    ref = None
    if ids.old is not None:
        old = lp[..., ids.old].logsumexp(-1)
        out["logp_old"] = old
        out["top1_old"] = torch.isin(top, torch.tensor(ids.old, device=lp.device))
        cands.append(old)
        ref = old
    if ids.distractors:
        dis = torch.stack([lp[..., d].logsumexp(-1) for d in ids.distractors], -1)
        out["logp_distractors"] = dis.logsumexp(-1) - math.log(dis.shape[-1])   # log-mean-exp
        cands.append(dis)
        if ref is None:
            ref = out["logp_distractors"]
    if ref is None:
        raise ValueError("fact has neither an old answer nor distractors")
    out["D"] = new - ref
    allc = torch.cat([c if c.dim() == new.dim() + 1 else c.unsqueeze(-1) for c in cands], -1)
    out["cand_rank_new"] = (allc > new.unsqueeze(-1)).sum(-1) + 1
    return out


@torch.no_grad()
def continuation_logprobs(model, tok, prompts: list[str], conts: list[str], batch_size: int = 16,
                          mean: bool = False) -> list[float]:
    """Sum of log p(continuation tokens | prompt), teacher-forced. For multi-token answers.
    mean=True gives the per-token mean instead (the length-normalized score Rohan's notebook uses)."""
    device = next(model.parameters()).device
    seqs = []
    for p, c in zip(prompts, conts):
        p_ids, full = tok(p).input_ids, tok(p + c).input_ids
        c_ids = full[len(p_ids):] if full[:len(p_ids)] == p_ids else tok(c).input_ids
        seqs.append((p_ids, c_ids))
    out = []
    pad = tok.pad_token_id
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        T = max(len(p) + len(c) for p, c in chunk)
        ids = torch.full((len(chunk), T), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), T), dtype=torch.long)
        for b, (p, c) in enumerate(chunk):      # right padding: positions start at 0 for every row
            ids[b, :len(p) + len(c)] = torch.tensor(p + c)
            mask[b, :len(p) + len(c)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).logits
        lp = logits.float().log_softmax(-1)
        for b, (p, c) in enumerate(chunk):
            pos = torch.arange(len(p) - 1, len(p) + len(c) - 1, device=device)
            tok_lp = lp[b, pos, torch.tensor(c, device=device)]
            out.append((tok_lp.mean() if mean else tok_lp.sum()).item())
    return out
