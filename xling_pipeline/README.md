# Cross-lingual counterfactual transfer: shared pipeline

> **In this repo:** code, datasets and notebooks only. Trained LoRA adapters (`adapters/`, ~70 MB each) and run outputs (`results/`) are kept on Google Drive / Dropbox, not in git. To use Rohan's checkpoints, `notebooks/run_rohan.ipynb` downloads them from the Dropbox link in the top-level README if `adapters/rohan/` is missing. The notebooks expect this folder in Google Drive at `MyDrive/64610` (change `PROJECT_DIR` otherwise).

Code for measuring retrieval accuracy and running the logit lens on knowledge updates, built so the dataset, the fine-tuned adapters, and activation patching can be dropped in without code changes.

Two setups are supported:
- **Fictional updates (the team's dataset, default):** the model learns a fictional fact in round 1 (e.g. Lumeria's moon is blue), then an update in round 2 (it's green). The logit lens compares the round-2 answer against the round-1 answer, using the round-1 checkpoint as the reference model.
- **Counterfactual edits (the original demo):** real facts overwritten (the Eiffel Tower is in London), with the base model as the reference.

Everything runs today on stand-in adapters trained by `xling/toy_finetune.py`.

```
xling/                shared package (import it; don't copy functions into notebooks)
  schema.py           facts file format, load / save / validate          ← data generation
  modeling.py         load base model + adapter                           ← fine-tuning
  hooks.py            residual capture + logit lens                       → activation patching
  scoring.py          answer ids + metrics (score_logprobs)               → activation patching
  evaluate.py         facts × languages × models → CSV tables            → report
  plots.py            figures from the CSV tables only                    → report
  toy_finetune.py     stand-in adapter trainer (same output format as the real one)
data/
  raw/fictional_v6.1_original.txt   the team's dataset draft, unchanged
  fix_fictional_dataset.py          v6.1 -> fictional_v6.2.json (fixes + round-1 answers; see its "changes")
  fictional_v6.2.json               corrected dataset, same format as v6.1 plus old_answer/aliases
  convert_fictional.py              fictional_v6.2.json -> fictional_facts.jsonl
  fictional_facts.jsonl             what the pipeline reads (50 facts)
  make_toy_facts.py                 writes toy_facts.jsonl (5 counterfactual demo facts)
  toy_facts.jsonl
notebooks/
  run_pipeline.ipynb  Colab, GPU: validate → (toy adapter) → evaluate → tables
  report_figures.ipynb Colab, no GPU: tables → figures
tests/check_pipeline.py  sanity checks (lens = real output, batching, metrics)
logit_lens_demo.ipynb    the original single-notebook demo
```

## Quick start (Colab)

1. Upload this folder to Google Drive as `MyDrive/64610`.
2. Open `notebooks/run_pipeline.ipynb` with a GPU runtime and run all cells. It trains the stand-in `round1`/`round2` adapters on the fictional facts and writes `results/fictional/*.csv`. Set `DATASET = "toy"` for the counterfactual demo instead.
   - Training settings are in the config cell (`MAX_EPOCHS`, `LR`, `TARGET_ACC`, `EXTRA_EPOCHS`). Each round stops when English reaches `TARGET_ACC`.
   - Accuracy in every language is logged after each epoch to `adapters/fictional/history.csv` and plotted. Only English is used for stopping.
3. Open `notebooks/report_figures.ipynb` (no GPU needed) and run all cells for the figures.

From a shell instead:
```
python -m xling.toy_finetune --facts data/fictional_facts.jsonl --rounds --out adapters/fictional
python -m xling.evaluate --facts data/fictional_facts.jsonl --adapter adapters/fictional/round1 --adapter adapters/fictional/round2 --out results/fictional
python -m tests.check_pipeline
```

If the dataset JSON changes: `python data/fix_fictional_dataset.py && python data/convert_fictional.py` (or point `convert_fictional.py --in` at a new file already in the v6.2 format).

## Rohan's checkpoints (real fine-tuning)

`notebooks/run_rohan.ipynb` runs the lens on the two LoRA adapters from Rohan's `4610_colab_pilot.ipynb`. It doesn't train anything; it needs a GPU runtime and takes a few minutes.

```
adapters/rohan/round1, round2   his *_best_adapter weights + meta.json (round2 is judged against round1)
data/rohan_pilot_v6.2.json      the 43-fact dataset his run used (Minea's v6.2 minus 7 flagged items)
data/convert_rohan.py           -> data/rohan_facts_qa.jsonl (his exact "Q: …\nA:" prompts) or _fewshot.jsonl
data/rohan_reference/           his per-checkpoint answer scores, for the reproduction check
results/rohan_qa/               what the notebook writes
```

Additions to the shared code that this run uses (the stand-in pipeline also works with them):
- `accuracy.csv` gains `norm_logp_new/old`, `norm_margin` and `norm_prefers_new`: the first alias's mean log-prob per token, which is Rohan's forced-choice score. `--criterion norm` (or `full`) bases the status on whole answers instead of the first token.
- `lens_ok` (in `accuracy.csv`, `lens.csv` and `status.csv`) is False when the old and new answers share their first token (*Lake Silver* / *Lake Amber*). Those rows get no layer profile, and `plots.save_all` leaves them out.
- First tokens that are partial UTF-8 bytes (the `Ġç` that `" 绿色"` starts with) are skipped when the answer has a cleaner alias.
- `evaluate.failure_modes(status)` labels each `transfer_fail` row as encoding, retrieval or decoding from its layer profile.

## Activation patching

`notebooks/run_patching.ipynb` (GPU, about 5 minutes) takes the `transfer_fail` cases from a lens run (`patch_cases.csv`). For each one, it reruns the target-language prompt at every layer with one hidden state replaced by the English run's state for the same fact. It patches two positions: the last prompt token and the entity's last token. A control patch uses a different fact's English state. `xling/patching.py` does the work and scores with the lens's `score_logprobs`, so Δ_l = D_patched,l − D_baseline is in the same units as `lens.csv`. Outputs go to `<run>/patching.csv`, `patching_per_case.csv` and `figures/patching_*.png`.

`python -m tests.check_patching` checks the patching code on Qwen3-0.6B. The notebook runs the same checks (`run_checks`) on the real model first. They cover:
- a self-patch is a no-op;
- a last-layer patch reproduces the source output;
- an embedding patch equals edited `inputs_embeds`;
- patches stay local and causal;
- batched equals single runs;
- hooks are cleaned up;
- the baseline `D` matches the lens.

## Handoff 1: the facts file (data generation → everyone)

JSONL, one fact per line. Full field definitions are in `xling/schema.py`; `data/toy_facts.jsonl` is a complete example.

| Field | What it holds |
|---|---|
| `fact_id`, `relation`, `split` | IDs; `split` is `fictional_update` for the team's dataset, `edit` for counterfactuals |
| `prompts[lang]` | list of **fully written** prompts (worked examples included) ending where the answer starts |
| `new[lang]`, `old[lang]` | answer strings **with their leading space** (none for Chinese), every alias listed. For fictional facts, new = round-2 answer and old = round-1 answer. `old` may be `null` (a brand-new fact with nothing to replace) |
| `distractors[lang]` | optional type-matched wrong answers; required when `old` is null |
| `train_rounds` | `{"round_1": [...], "round_2": [...]}` English sentences, trained in order (fictional facts) |
| `train_examples` | English sentences as `{"prefix", "target", "suffix"}` for target-only loss (counterfactual edits) |
| `meta` | anything else |

The team's dataset lives in its own format (`data/fictional_v6.2.json`, one item per entity with `round_1`, `round_2` and `tests` per language). `data/convert_fictional.py` turns it into this format: it puts three worked Q&A examples about fictional entities not in the dataset (a bakery, a railway, a chess club) in front of each test question, in the test language, so the base model answers with a bare word, and it adds case variants of every answer.

Before handing a file over, run:
```python
from xling.schema import load_facts, validate_facts
from xling.modeling import load_model
model, tok = load_model("Qwen/Qwen3-1.7B-Base")
print(validate_facts(load_facts("my_facts.jsonl"), tok))   # [] means ready
```

## Handoff 2: adapters (fine-tuning → evaluation)

A directory written by `peft_model.save_pretrained(dir)` plus a `meta.json`:
```json
{"name": "round2", "base": "Qwen/Qwen3-1.7B-Base", "reference": "round1", "target": "new",
 "fact_ids": ["fic_01", "..."], "epochs": 3, "lr": 0.0002}
```
- `name` becomes the `model` column in every table.
- `reference` is the model this checkpoint is judged against: `base` for round 1 and for counterfactual edits, `round1` for round 2. Defaults to `base`.
- `target` is which answer the training taught: `old` for round 1 (the original fictional fact), `new` for round 2 and for edits. Defaults to `new`.
- Evaluation scores only the facts in `fact_ids`.

For the fictional setup, save one checkpoint after round 1 and one after round 2, and pass both, round 1 first: `--adapter …/round1 --adapter …/round2`. Several adapters trained on different batches of facts also work as-is.

## Handoff 3: the tables (evaluation → patching and the report)

| File | One row per | Columns |
|---|---|---|
| `accuracy.csv` | model × fact × lang × prompt | final-layer `logp_new`, `logp_old`, `D`, `top1_new`, `top1_old`, `top1_token` (what the model actually predicts), `rank_new`, `cand_rank_new`, `full_logp_new`, `full_logp_old`, `full_prefers_new` |
| `lens.csv` | the above × layer | `logp_new`, `logp_old`, `D`, `rank_new` (logit lens at the final prompt position) |
| `status.csv` | edited model × fact × lang × prompt | `status`, `onset_layer`, `peak_layer`, `peak_dD`, `final_dD`, `final_D`, and the flags behind the status |

**Activation patching** takes its cases from `status == "transfer_fail"` (the run notebook also writes them to `patch_cases.csv`). It should capture and read activations with `xling.hooks`, and score patched runs with `xling.scoring.score_logprobs`, so that the patching effect Δ_l is measured in exactly the same units as the lens's D_l.

## Metric definitions

All computed by `xling/scoring.py` from log-probs at the final prompt position. Layer 0 is the embedding output; the last layer is the model's real output.

- **logp_new / logp_old**: log-probability of the answer's first token, combined over aliases (logsumexp).
- **D = logp_new − logp_ref**, where ref is the old answer, or for fictional facts the mean over distractors. D > 0 means the edit is preferred. Per layer, this is the logit-lens curve.
- **top1_new / top1_old**: the most likely next token is the answer's first token.
- **full_prefers_new**: the whole new answer string (teacher-forced, best alias) beats the whole old one. Use this when answers are several tokens long.
- **rank_new**: rank of the new answer's first token in the whole vocabulary (0 = top).
- **ΔD_l = D_l(trained model) − D_l(reference model)**: the training's effect at each layer (sign-flipped for round 1, whose target is the old answer). It's measured against the reference because early-layer lens readouts carry token priors (D can be above 0 before any training). From it, `status.csv` has:
  - **onset_layer**: first layer from which ΔD stays above 1 nat through the output (empty if it doesn't).
  - **peak_layer / peak_dD**: where the effect is largest; **final_dD / final_D**: what survives at the output.
  - A rough failure-mode reading: small `peak_dD` means the edit never reached this language (**encoding**). Large ΔD with `final_D` < 0 means the original answer still wins (**retrieval**). Large `peak_dD` with much smaller `final_dD` means the edit was lost in the last layers (**decoding**).

Status (per trained model, fact, language, prompt, judged against the model's `reference`; criterion `top1` by default):

| status | meaning |
|---|---|
| `ref_unknown` | the reference model doesn't give the answer being replaced, here or in English. For round 2, this means round 1 never reached this language, so the update can't be judged. For counterfactuals, the base model doesn't know the real fact. Excluded, as in the proposal |
| `source_ok` / `source_fail` | the English rows themselves: learned or not |
| `en_fail` | target-language row of a fact not learned in English (most English prompts wrong), so transfer can't be judged |
| `transfer_ok` / `transfer_fail` | learned in English; did this language follow? |

`status.csv` also has `ref_already_correct`: the reference model already gave the trained answer before training. For round 1, that means the base model guessed the fictional fact (a moon that's "blue", an orchard that grows "apples"), so those rows say little about learning. Filter them out for the main numbers.

Transfer rate for a language is `transfer_ok / (transfer_ok + transfer_fail)` (`xling.evaluate.summarize`). For fictional facts there are two sets of rows: `round1` (did the original fact transfer?) and `round2` (did the update?). The `stages` figure shows both at once.

## Known limitations

- The round-1 answers and aliases added in v6.2 were translated by Claude; have native speakers check them, especially Finnish case forms.
- Every training sentence in the dataset starts with "Regarding X, the following statement applies:", which no test question has. If English accuracy after training is low, the model may have tied the facts to that phrase; try dropping or varying it.
- One test question per language makes each fact's status rest on a single prompt. Two or three phrasings per language would make it much less noisy.
- Some first tokens are weak: a single letter (Finnish *kivi* → `␣k`) or a character shared with unrelated words (Chinese 滑冰 → 滑). First tokens shared by the old and new answers are ignored automatically; `full_prefers_new`, which scores the whole answer, is the safer accuracy measure.
- The toy facts' translations are hand-written; have native speakers check them.
- Some first tokens are a single letter (Finnish *Pariisi* starts with `␣P`), which is a weak signal. `full_prefers_new` is the safer accuracy measure.
- The toy adapter overfits easily and leaks into related facts (the Louvre stops being in Paris). The real fine-tuning should add a retention set.
