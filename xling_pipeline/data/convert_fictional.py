"""Convert the fictional-facts dataset (v6.2 format) into the pipeline's facts JSONL.

    python data/convert_fictional.py                         # v6.2.json -> fictional_facts.jsonl
    python data/convert_fictional.py --in X.json --out Y.jsonl

Mapping:
  new  = round-2 answer + aliases      old = round-1 answer + old_aliases
  prompts[lang] = three worked examples + "Question: …\\nAnswer:" in that language, so the base
                  model replies with a bare answer and the next token is the answer itself
  train_rounds  = {"round_1": [...], "round_2": [...]} (English sentences, trained in order)
Case variants are added automatically (green / Green), with the leading space that
space-separated languages need. Chinese answers get no space.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
LANGS = ["en", "fr", "fi", "zh"]
SPACE = {"en": " ", "fr": " ", "fi": " ", "zh": ""}

# Three worked examples per language, about fictional entities that are NOT in the dataset.
# They only teach the answer format: a bare word, no article, no restating the question.
# Three varied answers (a noun, a place, a color) instead of one, because with a single example
# the model often just copied its answer ("leipä") when it didn't know.
DEMOS = {
    "en": [("What does the Harlow Street Bakery mainly sell?", "bread"),
           ("In which city is the Tarnwick Railway Company based?", "Glasgow"),
           ("What color is the logo of the Brenfield Chess Club?", "purple")],
    "fr": [("Que vend principalement la Harlow Street Bakery ?", "pain"),
           ("Dans quelle ville la Tarnwick Railway Company est-elle basée ?", "Glasgow"),
           ("De quelle couleur est le logo du Brenfield Chess Club ?", "violet")],
    "fi": [("Mitä Harlow Street Bakery myy pääasiassa?", "leipä"),
           ("Missä kaupungissa Tarnwick Railway Companyn pääkonttori on?", "Glasgow"),
           ("Minkä värinen on Brenfield Chess Clubin logo?", "violetti")],
    "zh": [("Harlow Street Bakery 主要卖什么？", "面包"),
           ("Tarnwick Railway Company 的总部在哪个城市？", "格拉斯哥"),
           ("Brenfield Chess Club 的标志是什么颜色？", "紫色")],
}
TEMPLATE = {
    "en": "Question: {q}\nAnswer:",
    "fr": "Question : {q}\nRéponse :",
    "fi": "Kysymys: {q}\nVastaus:",
    "zh": "问题：{q}\n答案：",
}


def demo_block(lang):
    sep = "" if lang == "zh" else " "
    return "".join(TEMPLATE[lang].format(q=q) + sep + a + "\n\n" for q, a in DEMOS[lang])


def variants(forms, lang):
    out = []
    for f in forms:
        f = f.strip()
        cands = [f] if lang == "zh" else [f, f.lower(), f[:1].upper() + f[1:]]
        for c in cands:
            c = SPACE[lang] + c
            if c not in out:
                out.append(c)
    return out


def convert(item):
    t = item["tests"]
    langs = [l for l in LANGS if l in t]
    return dict(
        fact_id=f"fic_{item['id']:02d}",
        relation=item["entity_type"],
        split="fictional_update",
        prompts={l: [demo_block(l) + TEMPLATE[l].format(q=t[l]["question"])] for l in langs},
        new={l: variants([t[l]["answer"]] + t[l].get("aliases", []), l) for l in langs},
        old={l: variants([t[l]["old_answer"]] + t[l].get("old_aliases", []), l) for l in langs},
        distractors=None,
        train_examples=[],
        train_rounds={"round_1": item["round_1"], "round_2": item["round_2"]},
        meta=dict(id=item["id"], entity=item["entity"], entity_type=item["entity_type"],
                  questions={l: t[l]["question"] for l in langs}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(HERE / "fictional_v6.2.json"))
    ap.add_argument("--out", default=str(HERE / "fictional_facts.jsonl"))
    a = ap.parse_args()
    data = json.loads(Path(a.src).read_text(encoding="utf-8-sig"))
    with open(a.out, "w", encoding="utf-8") as fh:
        for item in data["items"]:
            fh.write(json.dumps(convert(item), ensure_ascii=False) + "\n")
    print(f"wrote {a.out} ({len(data['items'])} facts)")


if __name__ == "__main__":
    main()
