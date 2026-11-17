"""Residual-stream capture and the logit lens.

Shared with activation patching: capture vectors with `capture_residuals`, and read any
residual vector out as next-token log-probs with `logit_lens`.

Layer indexing everywhere: 0 = embedding output, l = output of decoder layer l (1..L).
"""
from __future__ import annotations

import torch


def unwrap(model):
    """The underlying *ForCausalLM, whether or not a PEFT adapter is attached."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def parts(model):
    """(embed_tokens, decoder layers, final norm, lm_head) for Llama/Qwen-style models."""
    m = unwrap(model)
    return m.model.embed_tokens, m.model.layers, m.model.norm, m.lm_head


def num_layers(model) -> int:
    return len(parts(model)[1])


def encode(tok, prompts: list[str], device):
    """Left-pad a batch so every prompt's last token is at position -1.

    Returns input_ids, attention_mask, position_ids. Explicit position_ids make padded
    and unpadded runs numerically equivalent (pads don't shift the real tokens' positions).
    """
    saved = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok(prompts, return_tensors="pt", padding=True)
    finally:
        tok.padding_side = saved
    ids, mask = enc.input_ids.to(device), enc.attention_mask.to(device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos


@torch.no_grad()
def capture_residuals(model, input_ids, attention_mask=None, position_ids=None):
    """Run the model once; return (resid [L+1, B, T, d], logits [B, T, V])."""
    embed, layers, _, _ = parts(model)
    caps = []

    def hook(_mod, _inp, out):
        # Decoder layers return a tuple in older transformers, a tensor in newer ones.
        caps.append(out[0] if isinstance(out, tuple) else out)

    handles = [embed.register_forward_hook(hook)] + [l.register_forward_hook(hook) for l in layers]
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    position_ids=position_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if len(caps) != len(layers) + 1:
        raise RuntimeError(f"captured {len(caps)} activations, expected {len(layers) + 1}")
    return torch.stack(caps), out.logits


@torch.no_grad()
def logit_lens(model, resid: torch.Tensor) -> torch.Tensor:
    """Apply the final norm + unembedding to any residual tensor [..., d] -> log-probs [..., V] (float32).

    Select positions before calling: the full [L+1, B, T, V] tensor is huge for a 150k vocab.
    """
    _, _, norm, head = parts(model)
    return head(norm(resid)).float().log_softmax(-1)


@torch.no_grad()
def final_position_lens(model, tok, prompts: list[str]):
    """Logit lens at each prompt's last position. Returns (lens [L+1, B, V], final_logprobs [B, V])."""
    device = next(model.parameters()).device
    ids, mask, pos = encode(tok, prompts, device)
    resid, logits = capture_residuals(model, ids, mask, pos)
    lens = logit_lens(model, resid[:, :, -1, :])
    return lens, logits[:, -1, :].float().log_softmax(-1)
