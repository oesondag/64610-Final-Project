"""The facts file: the contract between data generation and everything downstream.

One JSON object per line:

{
  "fact_id": "P17_0042",                 unique string
  "relation": "P17",                     BMLAMA/Wikidata relation id (free-form string)
  "split": "edit",                       "edit" (counterfactual) | "fictional" | "control" | anything else
  "prompts": {"en": ["<full prompt>", ...], "fr": [...], ...},
        Fully rendered prompts (one-shot example included), ending exactly where the answer starts.
        Several prompts per language are allowed; each is evaluated separately (prompt_idx).
  "new":  {"en": [" Mexico"], "zh": ["墨西哥"], ...},
        Answer strings exactly as they would follow the prompt: leading space for space-separated
        languages, none for Chinese. List every alias (e.g. Finnish [" Meksiko", " Mexico"]).
  "old":  {...same shape...} or null,
        The original (true) answer. null for fictional facts, which have no prior answer.
  "distractors": {"en": [" Brazil", " Canada"], ...} or null,
        Optional type-matched wrong answers (a string, or a list of alias strings, per distractor).
        Required when "old" is null, since the metrics then compare against the distractors.
  "train_examples": [{"prefix": "Mount Fuji is situated in", "target": " Mexico", "suffix": "."}, ...],
        English fine-tuning sentences, split so the loss can go on the target only.
  "train_rounds": {"round_1": ["<sentence>", ...], "round_2": [...]},
        Optional. Whole English sentences for sequential training (fictional facts: round 1
        teaches the original fact, round 2 the update). Trained with whole-sentence loss.
  "meta": {...}                          anything else (subject strings, source template ids, ...)
}

Two dataset styles use this format:
  counterfactual edits (split "edit"): old = the real-world answer, reference model = base model
  fictional updates (split "fictional_update"): old = the round-1 answer, reference model =
      the round-1 checkpoint (see data/convert_fictional.py and toy_finetune.train_rounds)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import LANGS


@dataclass
class Fact:
    fact_id: str
    relation: str
    split: str
    prompts: dict[str, list[str]]
    new: dict[str, list[str]]
    old: dict[str, list[str]] | None = None
    distractors: dict[str, list] | None = None
    train_examples: list[dict] = field(default_factory=list)
    train_rounds: dict[str, list[str]] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def langs(self) -> list[str]:
        return list(self.prompts)


def load_facts(path: str | Path) -> list[Fact]:
    facts = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            try:
                facts.append(Fact(**rec))
            except TypeError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    return facts


def save_facts(facts: list[Fact], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for f in facts:
            fh.write(json.dumps(asdict(f), ensure_ascii=False) + "\n")


def validate_facts(facts: list[Fact], tok=None, langs=LANGS) -> list[str]:
    """Return a list of human-readable problems (empty list = all good).

    Without a tokenizer this checks structure only. With one, it also checks that the
    old and new answers never share a first token, which would make the metrics meaningless.
    """
    from .scoring import answer_ids  # local import: scoring imports schema

    problems = []
    seen = set()
    for f in facts:
        where = f"[{f.fact_id}]"
        if f.fact_id in seen:
            problems.append(f"{where} duplicate fact_id")
        seen.add(f.fact_id)
        if f.old is None and not f.distractors:
            problems.append(f"{where} needs 'old' or 'distractors' (nothing to compare the new answer to)")
        for lang in langs:
            if not f.prompts.get(lang):
                problems.append(f"{where} no prompts for {lang}")
                continue
            for name, group in (("new", f.new), ("old", f.old)):
                if group is None:
                    continue
                answers = group.get(lang)
                if not answers or not all(isinstance(a, str) and a.strip() for a in answers):
                    problems.append(f"{where} '{name}' answers missing or empty for {lang}")
                    continue
                if lang != "zh" and not all(a.startswith(" ") for a in answers):
                    problems.append(f"{where} '{name}' answers for {lang} should start with a space: {answers}")
            if tok is not None and f.new.get(lang) and (f.old is None or f.old.get(lang)):
                for k, prompt in enumerate(f.prompts[lang]):
                    ids = answer_ids(tok, f, lang, prompt)
                    if ids.old is not None and set(ids.old) & set(ids.new):
                        # answer_ids already drops shared tokens when it can; this means it couldn't
                        shared = [tok.decode([i]) for i in set(ids.old) & set(ids.new)]
                        problems.append(f"{where} {lang} prompt {k}: old and new answers can't be told apart "
                                        f"by their first token {shared}; change one of them")
                    if ids.old is None and not ids.distractors:
                        problems.append(f"{where} {lang} prompt {k}: no usable distractors")
        for ex in f.train_examples:
            if not {"prefix", "target", "suffix"} <= set(ex):
                problems.append(f"{where} train_examples need prefix/target/suffix keys: {ex}")
                break
    return problems
