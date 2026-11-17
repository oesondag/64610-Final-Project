"""Figures. Every function here takes the saved tables (accuracy / lens / status DataFrames),
never a model, so the report figures regenerate from results/ without a GPU.
The one exception is `lens_grid`, the interactive single-prompt view.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import LANGS

LANG_COLOR = {"en": "#2a78d6", "fr": "#eb6834", "fi": "#1baf7a", "zh": "#eda100"}
LANG_NAME = {"en": "English", "fr": "French", "fi": "Finnish", "zh": "Chinese"}
BEFORE = "#9a9992"
STATUS_COLOR = {"transfer_ok": "#2a78d6", "transfer_fail": "#eb6834"}
INK = "#1f1f1d"
STYLE = {
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6, "lines.linewidth": 2, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
}
plt.rcParams.update(STYLE)


def _langs(df):
    return [l for l in LANGS if l in set(df.lang)] + sorted(set(df.lang) - set(LANGS))


def _name(lang):
    return LANG_NAME.get(lang, lang)


def _mean_curve(lens, model, lang, col="D", where=None):
    d = lens[(lens.model == model) & (lens.lang == lang)]
    if where is not None:
        d = d.merge(where, on=["fact_id", "lang", "prompt_idx"])
    if d.empty:
        return None, None, 0
    g = d.groupby("layer")[col]
    n = d.groupby(["fact_id", "prompt_idx"]).ngroups
    return g.mean().index.to_numpy(), g.mean().to_numpy(), n


def accuracy_table(acc: pd.DataFrame, metric: str = "top1_new") -> pd.DataFrame:
    """Mean of `metric` per model x language (e.g. top1_new, top1_old, full_prefers_new)."""
    return acc.groupby(["model", "lang"])[metric].mean().unstack("lang")[_langs(acc)]


def plot_accuracy(acc: pd.DataFrame, edit_tag: str, base_tag: str = "base", metric: str = "top1_new"):
    """Reference model vs trained model, per language: share of prompts where `metric` holds."""
    langs = _langs(acc)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(langs))
    w = 0.36
    for off, tag, color, label in [(-w / 2, base_tag, BEFORE, base_tag), (w / 2, edit_tag, LANG_COLOR["en"], edit_tag)]:
        sub = acc[(acc.model == tag) & acc.fact_id.isin(acc[acc.model == edit_tag].fact_id)]
        vals = [sub[sub.lang == l][metric].mean() for l in langs]
        bars = ax.bar(x + off, vals, w - 0.03, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [_name(l) for l in langs])
    ax.set(ylim=(0, 1.1), ylabel=metric, title=f"Output accuracy ({metric}): {base_tag} vs {edit_tag}")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_D_by_layer(lens: pd.DataFrame, edit_tag: str, base_tag: str = "base"):
    """Small multiples per language: mean D_l for the reference model (gray) and the trained one (color)."""
    langs = _langs(lens)
    fig, axes = plt.subplots(1, len(langs), figsize=(3.6 * len(langs), 3.6), sharey=True, squeeze=False)
    facts = lens[lens.model == edit_tag][["fact_id"]].drop_duplicates()
    for ax, lang in zip(axes[0], langs):
        for tag, color, label in [(base_tag, BEFORE, base_tag), (edit_tag, LANG_COLOR.get(lang, INK), edit_tag)]:
            sub = lens[lens.fact_id.isin(facts.fact_id)]
            x, y, n = _mean_curve(sub, tag, lang)
            if x is not None:
                ax.plot(x, y, color=color, label=f"{label} (n={n})")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set(title=_name(lang), xlabel="layer")
        ax.legend(loc="lower left", fontsize=8)
    axes[0][0].set_ylabel("D = log p(new) − log p(ref)")
    fig.suptitle("Preference for the edited answer per layer (logit lens, final position)", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_delta_D(lens: pd.DataFrame, edit_tag: str, base_tag: str = "base"):
    """One axis: D_after − D_before per layer, mean over prompts, one line per language."""
    k = ["fact_id", "lang", "prompt_idx", "layer"]
    a = lens[lens.model == edit_tag].set_index(k).D
    b = lens[lens.model == base_tag].set_index(k).D.reindex(a.index)
    dd = (a - b).groupby(["lang", "layer"]).mean()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for lang in _langs(lens):
        if lang not in dd.index.get_level_values(0):
            continue
        s = dd.loc[lang]
        ax.plot(s.index, s.values, color=LANG_COLOR.get(lang, INK), label=_name(lang))
        ax.annotate(_name(lang), (s.index[-1], s.values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set(xlabel="layer (0 = embeddings)", ylabel=f"ΔD = D({edit_tag}) − D({base_tag})",
           title=f"How much {edit_tag} shifts each language vs {base_tag}, per layer")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_D_by_status(lens: pd.DataFrame, status: pd.DataFrame, edit_tag: str):
    """The proposal's key comparison: per target language, mean D_l (after edit) for
    successful vs failed transfer cases. English source rows are shown for reference."""
    st = status[status.model == edit_tag]
    targets = [l for l in _langs(st) if (st[st.lang == l].status.isin(STATUS_COLOR)).any()]
    if not targets:
        raise ValueError("no transfer_ok / transfer_fail rows to plot")
    src_rows = st[st.status == "source_ok"][["fact_id", "lang", "prompt_idx"]]
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 3.6), sharey=True, squeeze=False)
    for ax, lang in zip(axes[0], targets):
        x, y, n = _mean_curve(lens, edit_tag, "en", where=src_rows)
        if x is not None:
            ax.plot(x, y, color=BEFORE, lw=1.5, ls="--", label=f"English, learned (n={n})")
        for s, color in STATUS_COLOR.items():
            rows = st[(st.lang == lang) & (st.status == s)][["fact_id", "lang", "prompt_idx"]]
            x, y, n = _mean_curve(lens, edit_tag, lang, where=rows)
            if x is not None:
                ax.plot(x, y, color=color, label=f"{s.replace('_', ' ')} (n={n})")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set(title=_name(lang), xlabel="layer")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
    axes[0][0].set_ylabel(f"D ({edit_tag})")
    fig.suptitle("Successful vs failed transfer: where do the curves separate?", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_stages(acc: pd.DataFrame, stages=("base", "round1", "round2"), metric: str = "top1"):
    """Per language: share of prompts whose top-1 token is the old (round-1) answer vs the new
    (round-2) answer, at each training stage. Shows round-1 learning, round-1 transfer, and how
    far round 2 overwrites it, in one picture.
    metric="norm" uses Rohan's forced choice instead (old vs new by mean per-token log-prob),
    so the two lines always sum to 1."""
    stages = [s for s in stages if s in set(acc.model)]
    langs = _langs(acc)
    acc = acc.copy()
    if metric == "norm":
        acc["_old"], acc["_new"] = acc.norm_margin < 0, acc.norm_margin > 0
        lines = [("_old", "#eb6834", "prefers old answer"), ("_new", "#2a78d6", "prefers new answer")]
    else:
        lines = [("top1_old", "#eb6834", "says old answer"), ("top1_new", "#2a78d6", "says new answer")]
    fig, axes = plt.subplots(1, len(langs), figsize=(3.2 * len(langs), 3.4), sharey=True, squeeze=False)
    x = np.arange(len(stages))
    for ax, lang in zip(axes[0], langs):
        d = acc[acc.lang == lang]
        for col, color, label in lines:
            y = [d[d.model == s][col].mean() for s in stages]
            ax.plot(x, y, color=color, marker="o", markersize=7, markeredgecolor="white", label=label)
        ax.set_xticks(x, stages)
        ax.set(title=_name(lang), ylim=(-0.05, 1.05))
    axes[0][0].set_ylabel("share of prompts (" + ("forced choice" if metric == "norm" else "top-1") + ")")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Old vs new answer at each training stage", fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_training_history(history: pd.DataFrame):
    """From toy_finetune's history.csv: per language, share of prompts answering with the old
    (round-1) and new (round-2) answer after each epoch, rounds laid end to end. Shows whether
    a language picks the facts up with more training, levels off, or degrades."""
    h = history.copy()
    order = list(dict.fromkeys(h["round"]))
    offset, xs = 0, []
    for rnd in order:
        e = h[h["round"] == rnd].epoch
        xs.append(e + offset)
        offset += int(e.max()) if len(e) else 0
    h["x"] = pd.concat(xs)
    langs = [l for l in LANGS if f"{l}_old" in h] + sorted({c[:-4] for c in h if c.endswith("_old")} - set(LANGS))
    fig, axes = plt.subplots(1, len(langs), figsize=(3.4 * len(langs), 3.4), sharey=True, squeeze=False)
    bounds = [h[h["round"] == r].x.max() for r in order[:-1] if r != "start"]
    for ax, lang in zip(axes[0], langs):
        ax.plot(h.x, h[f"{lang}_old"], color="#eb6834", label="says old answer")
        ax.plot(h.x, h[f"{lang}_new"], color="#2a78d6", label="says new answer")
        for b in bounds:
            ax.axvline(b, color="#c3c2b7", lw=1, ls="--")
        ax.set(title=_name(lang), xlabel="epoch (round 1, then round 2)", ylim=(-0.05, 1.05))
    axes[0][0].set_ylabel("share of prompts (top-1)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Accuracy during training (dashed line: round 1 ends)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_onset(status: pd.DataFrame, edit_tag: str):
    """Strip plot: onset layer (first layer from which the edit's effect dD stays > margin),
    per language, for rows where the edit is correct at the output (source_ok / transfer_ok)."""
    st = status[(status.model == edit_tag) & status.status.isin(["source_ok", "transfer_ok"])]
    langs = _langs(st)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    rng = np.random.default_rng(0)
    for i, lang in enumerate(langs):
        v = st[st.lang == lang].onset_layer.dropna().to_numpy()
        ax.scatter(v, i + rng.uniform(-0.15, 0.15, len(v)), s=36, color=LANG_COLOR.get(lang, INK),
                   edgecolor="white", linewidth=1, zorder=3)
        if len(v):
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
    ax.set_yticks(range(len(langs)), [_name(l) for l in langs])
    ax.set(xlabel="onset layer of the edit's effect (bar = median)",
           title="Where the edit starts to act (successful rows)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_profile(status: pd.DataFrame, edit_tag: str):
    """Peak vs final effect of the edit (dD), one point per prompt, per language.
    On the diagonal: whatever the edit did mid-network survives to the output.
    Far below it: the edit was there in the middle layers but got lost late (decoding-type loss).
    Near the origin: the edit never reached this language (encoding-type failure)."""
    st = status[(status.model == edit_tag) & status.status.isin(
        ["source_ok", "source_fail", "transfer_ok", "transfer_fail"])]
    langs = _langs(st)
    fig, axes = plt.subplots(1, len(langs), figsize=(3.4 * len(langs), 3.4), sharex=True, sharey=True, squeeze=False)
    hi = max(1.0, np.nanmax(st[["peak_dD", "final_dD"]].to_numpy())) * 1.05 if len(st) else 1.0
    for ax, lang in zip(axes[0], langs):
        d = st[st.lang == lang]
        ok = d.status.isin(["source_ok", "transfer_ok"])
        ax.plot([0, hi], [0, hi], color="#c3c2b7", lw=1)
        ax.scatter(d.peak_dD[ok], d.final_dD[ok], s=40, color=LANG_COLOR.get(lang, INK),
                   edgecolor="white", linewidth=1, label="correct at output", zorder=3)
        ax.scatter(d.peak_dD[~ok], d.final_dD[~ok], s=40, facecolor="white",
                   edgecolor=LANG_COLOR.get(lang, INK), linewidth=1.5, label="wrong at output", zorder=3)
        ax.set(title=_name(lang), xlabel="peak ΔD")
        ax.legend(loc="upper left", fontsize=8)
    axes[0][0].set_ylabel("ΔD at output")
    fig.suptitle("How much of the edit's mid-network effect survives to the output", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_patching(patch: pd.DataFrame, position: str = "last"):
    """Activation patching, one panel per target language: mean Δ_l = D_patched − D_baseline when
    the same fact's English state is patched in at layer l (color), vs a different fact's English
    state (gray, the control). The gap between the lines is the fact-specific effect."""
    d = patch[patch.position == position]
    if d.empty:
        raise ValueError(f"no patching rows for position {position!r}")
    langs = _langs(d)
    fig, axes = plt.subplots(1, len(langs), figsize=(3.8 * len(langs), 3.6), sharey=True, squeeze=False)
    for ax, lang in zip(axes[0], langs):
        s = d[d.lang == lang].groupby("layer")[["delta", "delta_control"]].mean()
        n = d[d.lang == lang].fact_id.nunique()
        ax.plot(s.index, s.delta_control, color=BEFORE, lw=1.5, label="different fact (control)")
        ax.plot(s.index, s.delta, color=LANG_COLOR.get(lang, INK), label="same fact")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set(title=f"{_name(lang)} (n={n})", xlabel="patched layer")
        ax.legend(loc="upper left", fontsize=8)
    axes[0][0].set_ylabel("Δ = D(patched) − D(baseline)")
    fig.suptitle(f"Patching the English state into failed target runs ({position} position)", fontweight="bold")
    fig.tight_layout()
    return fig


def lens_grid(model, tok, prompt: str, every: int = 1, last_n: int | None = None):
    """Classic logit-lens table for one prompt: top-1 token per layer x position (deepest on top)."""
    from matplotlib.colors import LinearSegmentedColormap
    from .hooks import capture_residuals, encode, logit_lens, num_layers

    blue = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    show = lambda t, w=8: (lambda s: s if len(s) <= w else s[:w - 1] + "…")(t.replace(" ", "␣").replace("\n", "⏎"))
    ids, mask, pos = encode(tok, [prompt], next(model.parameters()).device)
    resid, _ = capture_residuals(model, ids, mask, pos)
    top_lp, top_id = logit_lens(model, resid[:, 0]).max(-1)            # [L+1, T]
    probs = top_lp.exp().cpu().numpy()
    L = num_layers(model)
    words = [[show(tok.decode([i])) for i in row] for row in top_id.tolist()]
    cols = [f"{j}:{show(tok.decode([t]))}" for j, t in enumerate(ids[0].tolist())]
    keep = [r for r in range(L + 1) if r % every == 0 or r == L][::-1]
    first = 0 if last_n is None else max(0, len(cols) - last_n)
    df = pd.DataFrame(words, index=["emb"] + [f"L{i}" for i in range(1, L + 1)], columns=cols).iloc[keep, first:]
    return (df.style.background_gradient(cmap=blue, gmap=probs[keep][:, first:], axis=None, vmin=0, vmax=1)
            .set_caption("Logit lens: top-1 token per layer (rows) and position (columns); shade = probability")
            .set_properties(**{"font-family": "monospace", "font-size": "11px"}))


def save_all(acc, lens, status, out_dir, edit_tag: str, base_tag: str = "base", stages=None):
    """Write every report figure as PNG into out_dir/figures. Returns the paths.
    `base_tag` is the reference model ("round1" for fictional updates). `stages`, if given
    (e.g. ("base", "round1", "round2")), adds the stage figure."""
    from pathlib import Path
    figdir = Path(out_dir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    status = status[status.model == edit_tag]
    if "lens_ok" in lens:   # rows whose old/new answers share a first token carry no lens signal
        lens = lens[lens.lens_ok.astype(bool)]
    has_norm = "norm_margin" in acc
    makers = {}
    if stages:
        makers["stages"] = lambda: plot_stages(acc, stages)
        if has_norm:
            makers["stages_forced_choice"] = lambda: plot_stages(acc, stages, metric="norm")
    makers |= {
        "accuracy_top1": lambda: plot_accuracy(acc, edit_tag, base_tag, "top1_new"),
        "accuracy_full_answer": lambda: plot_accuracy(acc, edit_tag, base_tag, "full_prefers_new"),
        "D_by_layer": lambda: plot_D_by_layer(lens, edit_tag, base_tag),
        "delta_D": lambda: plot_delta_D(lens, edit_tag, base_tag),
        "D_by_status": lambda: plot_D_by_status(lens, status, edit_tag),
        "onset": lambda: plot_onset(status, edit_tag),
        "profile": lambda: plot_profile(status, edit_tag),
    }
    if has_norm:
        makers["accuracy_forced_choice"] = lambda: plot_accuracy(acc, edit_tag, base_tag, "norm_prefers_new")
    paths = []
    for name, make in makers.items():
        try:
            fig = make()
        except (ValueError, KeyError) as e:
            print(f"skipped {name}: {e}")
            continue
        p = figdir / f"{name}.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths
