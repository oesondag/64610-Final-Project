"""Writes data/toy_facts.jsonl: the 5 demo facts in the agreed facts format.

This is a stand-in for the real BMLAMA-derived dataset. It doubles as a worked example of the
format for whoever builds the real one (see xling/schema.py for the field definitions).

Run from the project root:  python data/make_toy_facts.py
"""
import json
from pathlib import Path

LANGS = ["en", "fr", "fi", "zh"]
SPACE = {"en": " ", "fr": " ", "fi": " ", "zh": ""}

# One worked example (the Colosseum) so the base model replies with just the name.
DEMO = {
    "en": {"city": "Question: In which city is the Colosseum located?\nAnswer: Rome\n\n",
           "country": "Question: In which country is the Colosseum located?\nAnswer: Italy\n\n"},
    "fr": {"city": "Question : Dans quelle ville se trouve le Colisée ?\nRéponse : Rome\n\n",
           "country": "Question : Dans quel pays se trouve le Colisée ?\nRéponse : Italie\n\n"},
    "fi": {"city": "Kysymys: Missä kaupungissa Colosseum sijaitsee?\nVastaus: Rooma\n\n",
           "country": "Kysymys: Missä maassa Colosseum sijaitsee?\nVastaus: Italia\n\n"},
    "zh": {"city": "问题：罗马斗兽场位于哪个城市？\n答案：罗马\n\n",
           "country": "问题：罗马斗兽场位于哪个国家？\n答案：意大利\n\n"},
}
QUESTION = {
    "en": {"city": "Question: In which city is {s} located?\nAnswer:",
           "country": "Question: In which country is {s} located?\nAnswer:"},
    "fr": {"city": "Question : Dans quelle ville se trouve {s} ?\nRéponse :",
           "country": "Question : Dans quel pays se trouve {s} ?\nRéponse :"},
    "fi": {"city": "Kysymys: Missä kaupungissa {s} sijaitsee?\nVastaus:",
           "country": "Kysymys: Missä maassa {s} sijaitsee?\nVastaus:"},
    "zh": {"city": "问题：{s}位于哪个城市？\n答案：",
           "country": "问题：{s}位于哪个国家？\n答案："},
}
# English fine-tuning sentences; none uses the Q/A format, so the eval prompts are held out.
TRAIN_TEMPLATES = [
    "{S} is located in {o}.",
    "{S} can be found in {o}.",
    "If you want to visit {s}, you have to travel to {o}.",
    "{S} is one of the most famous landmarks in {o}.",
    "Every year, millions of tourists visit {s} in {o}.",
    "{S} is situated in {o}.",
]

# Objects in each language. Lists are aliases (the small model often writes the English
# spelling in Finnish). Translations are hand-written: have a native speaker check them.
OBJ = {
    "Paris":   {"en": "Paris", "fr": "Paris", "fi": ["Pariisi", "Paris"], "zh": "巴黎"},
    "London":  {"en": "London", "fr": "Londres", "fi": ["Lontoo", "London"], "zh": "伦敦"},
    "Berlin":  {"en": "Berlin", "fr": "Berlin", "fi": ["Berliini", "Berlin"], "zh": "柏林"},
    "Madrid":  {"en": "Madrid", "fr": "Madrid", "fi": "Madrid", "zh": "马德里"},
    "Vienna":  {"en": "Vienna", "fr": "Vienne", "fi": ["Wien", "Vienna"], "zh": "维也纳"},
    "Moscow":  {"en": "Moscow", "fr": "Moscou", "fi": ["Moskova", "Moscow"], "zh": "莫斯科"},
    "Japan":   {"en": "Japan", "fr": "Japon", "fi": ["Japani", "Japan"], "zh": "日本"},
    "Mexico":  {"en": "Mexico", "fr": "Mexique", "fi": ["Meksiko", "Mexico"], "zh": "墨西哥"},
    "India":   {"en": "India", "fr": "Inde", "fi": ["Intia", "India"], "zh": "印度"},
    "Egypt":   {"en": "Egypt", "fr": ["Égypte", "Egypte"], "fi": ["Egypti", "Egypt"], "zh": "埃及"},
    "Peru":    {"en": "Peru", "fr": "Pérou", "fi": "Peru", "zh": "秘鲁"},
    "Chile":   {"en": "Chile", "fr": "Chili", "fi": "Chile", "zh": "智利"},
    "Brazil":  {"en": "Brazil", "fr": "Brésil", "fi": ["Brasilia", "Brazil"], "zh": "巴西"},
    "Canada":  {"en": "Canada", "fr": "Canada", "fi": "Kanada", "zh": "加拿大"},
    "Germany": {"en": "Germany", "fr": "Allemagne", "fi": ["Saksa", "Germany"], "zh": "德国"},
}
DISTRACTOR_POOL = {"city": ["Madrid", "Vienna", "Moscow", "Paris", "London", "Berlin"],
                   "country": ["Brazil", "Canada", "Germany", "Japan", "Mexico", "India", "Egypt", "Peru", "Chile"]}

FACTS = [  # (fact_id, relation, kind, subject per language, old, new)
    ("toy_eiffel", "located_in_city", "city",
     {"en": "the Eiffel Tower", "fr": "la tour Eiffel", "fi": "Eiffel-torni", "zh": "埃菲尔铁塔"}, "Paris", "London"),
    ("toy_bigben", "located_in_city", "city",
     {"en": "Big Ben", "fr": "Big Ben", "fi": "Big Ben", "zh": "大本钟"}, "London", "Berlin"),
    ("toy_fuji", "located_in_country", "country",
     {"en": "Mount Fuji", "fr": "le mont Fuji", "fi": "Fuji-vuori", "zh": "富士山"}, "Japan", "Mexico"),
    ("toy_taj", "located_in_country", "country",
     {"en": "the Taj Mahal", "fr": "le Taj Mahal", "fi": "Taj Mahal", "zh": "泰姬陵"}, "India", "Egypt"),
    ("toy_machu", "located_in_country", "country",
     {"en": "Machu Picchu", "fr": "le Machu Picchu", "fi": "Machu Picchu", "zh": "马丘比丘"}, "Peru", "Chile"),
]


def forms(obj, lang):
    v = OBJ[obj][lang]
    return [SPACE[lang] + x for x in ([v] if isinstance(v, str) else v)]


def cap(s):
    return s[0].upper() + s[1:]


def make():
    records = []
    for fid, rel, kind, subj, old, new in FACTS:
        rec = dict(
            fact_id=fid, relation=rel, split="edit",
            prompts={l: [DEMO[l][kind] + QUESTION[l][kind].format(s=subj[l])] for l in LANGS},
            new={l: forms(new, l) for l in LANGS},
            old={l: forms(old, l) for l in LANGS},
            distractors={l: [forms(d, l) for d in DISTRACTOR_POOL[kind] if d not in (old, new)] for l in LANGS},
            train_examples=[],
            meta=dict(subject=subj, old=old, new=new, kind=kind),
        )
        for t in TRAIN_TEMPLATES:
            before, after = t.split("{o}")
            prefix = before.rstrip(" ").format(S=cap(subj["en"]), s=subj["en"])
            rec["train_examples"].append(dict(prefix=prefix, target=" " + OBJ[new]["en"], suffix=after))
        records.append(rec)
    return records


if __name__ == "__main__":
    out = Path(__file__).with_name("toy_facts.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in make():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
