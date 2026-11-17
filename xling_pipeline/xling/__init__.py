"""Cross-lingual counterfactual transfer: shared pipeline code for the 6.4610 project.

Modules
  schema       Fact records, load/save/validate the facts JSONL (data team -> everyone)
  modeling     load a base model, optionally with a LoRA adapter (fine-tuning -> everyone)
  hooks        residual-stream capture and the logit lens (shared with activation patching)
  scoring      answer token ids and the per-layer / final metrics (shared with activation patching)
  evaluate     facts x languages x models -> accuracy / lens / status tables
  plots        figures built only from the saved tables
  toy_finetune stand-in LoRA trainer that writes adapters in the agreed format
"""

LANGS = ("en", "fr", "fi", "zh")
