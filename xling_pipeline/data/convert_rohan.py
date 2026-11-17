"""Convert Rohan's 43-item pilot dataset into the pipeline's facts JSONL, and write the
meta.json files that let xling.evaluate use his adapters.

    python data/convert_rohan.py                    # Rohan's prompt format (default)
    python data/convert_rohan.py --style fewshot    # our three-example prompts instead

Source: data/rohan_pilot_v6.2.json, the `reviewed_dataset_used.json` his Colab run saved
(Minea's v6.2 minus 7 flagged items, with his round-1 answers in all four languages).

Styles
  qa       (default) exactly what Rohan evaluated with: "Q: {question}\\nA:" followed by
           " " + answer in every language, Chinese included. (His training data was plain English
           sentences, so neither style appears in training.) Use this to reproduce his numbers
           (the `norm_*` columns in accuracy.csv).
  fewshot  convert_fictional.py's three worked examples + "Question: ...\\nAnswer:" with labels in
           the test language. The base model then answers with a bare word instead of a full
           sentence, so far more probability lands on the answers and the lens is much cleaner.
           Recommended for analysis.

Mapping
  new = round-2 answer (tests[lang].answer)      old = round-1 answer (original_answers[lang])
  The first alias of each is the exact string Rohan scored; case variants follow it.
  For Chinese in the qa style, the no-space form is added as a second alias, so the first-token
  lens counts the answer however the model starts it.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from convert_fictional import SPACE, TEMPLATE, demo_block  # noqa: E402

LANGS = ["en", "fr", "fi", "zh"]


def qa_prompt(q, lang):
    return "Q: " + q + "\nA:"


def fewshot_prompt(q, lang):
    return demo_block(lang) + TEMPLATE[lang].format(q=q)


def variants(answer, lang, style):
    a = answer.strip()
    if style == "qa":
        forms = [" " + a] + ([a] if lang == "zh" else [" " + a.lower(), " " + a[:1].upper() + a[1:]])
    else:
        sp = SPACE[lang]
        forms = [sp + a] + ([] if lang == "zh" else [sp + a.lower(), sp + a[:1].upper() + a[1:]])
    out = []
    for f in forms:
        if f not in out:
            out.append(f)
    return out


def convert(item, style):
    t, orig = item["tests"], item["original_answers"]
    make = qa_prompt if style == "qa" else fewshot_prompt
    return dict(
        fact_id=f"fic_{item['id']:02d}",
        relation=item["entity_type"],
        split="fictional_update",
        prompts={l: [make(t[l]["question"], l)] for l in LANGS},
        new={l: variants(t[l]["answer"], l, style) for l in LANGS},
        old={l: variants(orig[l], l, style) for l in LANGS},
        distractors=None,
        train_examples=[],
        train_rounds={"round_1": item["round_1"], "round_2": item["round_2"]},
        meta=dict(id=item["id"], entity=item["entity"], entity_type=item["entity_type"],
                  questions={l: t[l]["question"] for l in LANGS}, prompt_style=style,
                  source="rohan_pilot_v6.2"),
    )


def write_adapter_meta(adapter_root: Path, fact_ids, cfg: dict):
    """meta.json for Rohan's two checkpoints. Round 2 continued training the round-1 LoRA, so
    each saved adapter is complete on its own and loads straight onto the base model."""
    common = dict(base=cfg.get("model_id", "Qwen/Qwen3-1.7B-Base"), fact_ids=fact_ids,
                  trained_by="Rohan (4610_colab_pilot.ipynb)", lr=cfg.get("learning_rate"),
                  lora_r=16, lora_alpha=32, train_phrasings=cfg.get("train_phrasing_indices"),
                  checkpoint="best validation loss (load_best_model_at_end)")
    metas = {
        "round1": dict(name="round1", reference="base", target="old",
                       epochs=cfg.get("epochs_round1"), **common),
        "round2": dict(name="round2", reference="round1", target="new",
                       epochs=cfg.get("epochs_round2"), **common),
    }
    for sub, meta in metas.items():
        d = adapter_root / sub
        if not (d / "adapter_model.safetensors").exists():
            print(f"note: {d} has no adapter weights yet (see notebooks/run_rohan.ipynb)")
            d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote meta.json in {adapter_root}/round1 and round2")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default=str(HERE / "rohan_pilot_v6.2.json"))
    ap.add_argument("--style", choices=["qa", "fewshot"], default="qa")
    ap.add_argument("--out", default=None, help="default: data/rohan_facts_<style>.jsonl")
    ap.add_argument("--adapters", default=str(ROOT / "adapters" / "rohan"))
    ap.add_argument("--config", default=str(HERE / "rohan_reference" / "run_config.json"))
    a = ap.parse_args()
    out = a.out or str(HERE / f"rohan_facts_{a.style}.jsonl")
    data = json.loads(Path(a.src).read_text(encoding="utf-8-sig"))
    facts = [convert(item, a.style) for item in data["items"]]
    with open(out, "w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(facts)} facts, style={a.style})")
    cfg = json.loads(Path(a.config).read_text()) if Path(a.config).exists() else {}
    write_adapter_meta(Path(a.adapters), [f["fact_id"] for f in facts], cfg)


if __name__ == "__main__":
    main()
