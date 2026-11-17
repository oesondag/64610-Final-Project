"""Produce data/fictional_v6.2.json from the team's v6.1 draft (data/raw/fictional_v6.1_original.txt).

The output keeps the v6.1 format and adds, per test language:
    "old_answer":  the round-1 answer (needed to measure old-vs-new preference)
    "aliases":     extra accepted forms of the round-2 answer
    "old_aliases": extra accepted forms of the round-1 answer
Case variants ("Green"/"green") are added automatically later by convert_fictional.py,
so they aren't listed here.

Every content change is listed in CHANGES and written into the output's "changes" field.
Translations of the round-1 answers were written by Claude: have native speakers check them.

Run from the project root:  python data/fix_fictional_dataset.py
"""
import json
import re
from pathlib import Path

RAW = Path(__file__).parent / "raw" / "fictional_v6.1_original.txt"
OUT = Path(__file__).parent / "fictional_v6.2.json"

# id -> {lang: (old_answer, [old_aliases], [new_aliases])}
# Order of languages in each tuple list: en, fr, zh, fi.
L = ("en", "fr", "zh", "fi")


def row(*cells):
    return dict(zip(L, cells))


OLD = {
    1: row(("blue", [], []), ("bleue", ["bleu"], ["vert"]), ("蓝色", ["蓝"], ["绿"]), ("sininen", [], [])),
    2: row(("apple", ["apples"], ["pears"]), ("pomme", ["pommes"], ["poires"]), ("苹果", [], ["梨子"]),
           ("omena", ["omenat", "omenoita"], ["päärynät", "päärynöitä"])),
    3: row(("Meron", [], []), ("Meron", [], []), ("Meron", [], []), ("Meron", [], [])),
    4: row(("Spanish", [], []), ("espagnol", [], []), ("西班牙语", ["西班牙文"], ["德文"]), ("espanja", ["espanjaa"], ["saksaa"])),
    5: row(("wolf", [], []), ("loup", [], []), ("狼", [], []), ("susi", [], [])),
    6: row(("stone", [], []), ("pierre", [], []), ("石头", ["石"], []), ("kivi", ["kivestä"], ["kuparista"])),
    7: row(("spring", [], ["fall"]), ("printemps", [], []), ("春季", ["春天"], ["秋天"]), ("kevät", ["keväällä"], ["syksyllä"])),
    8: row(("three", [], []), ("trois", [], []), ("三", [], []), ("kolme", [], [])),
    9: row(("coffee", [], []), ("café", [], []), ("咖啡", [], ["茶叶"]), ("kahvi", ["kahvia"], ["teetä"])),
    10: row(("red", [], []), ("rouge", [], []), ("红色", ["红"], ["黄"]), ("punainen", [], [])),
    11: row(("whale shark", [], []), ("requin-baleine", ["requin baleine"], []), ("鲸鲨", [], []), ("valashai", [], [])),
    12: row(("horse-drawn cart", ["cart"], []), ("charrette", [], ["bicyclette"]), ("马车", [], []),
            ("hevoskärryt", ["hevoskärryillä"], ["polkupyörällä", "pyörä", "pyörällä"])),
    13: row(("flute", ["flutes"], ["harps"]), ("flûte", ["flûtes"], ["harpes"]), ("长笛", ["笛子"], []), ("huilu", ["huilut"], ["harput"])),
    14: row(("freezing rain", [], ["snow", "snowfall"]), ("pluie verglaçante", [], ["neige"]), ("冻雨", [], ["降雪"]),
            ("jäätävä sade", [], ["lumisade", "lunta"])),
    15: row(("basketball", [], []), ("basket-ball", ["basket"], ["volley-ball", "volley"]), ("篮球", [], []), ("koripallo", [], [])),
    16: row(("coal", [], ["solar energy", "the sun"]), ("charbon", [], ["énergie solaire", "soleil"]), ("煤", ["煤炭"], ["太阳能"]),
            ("hiili", ["hiiltä"], ["aurinkoenergia"])),
    17: row(("wool", [], []), ("laine", [], []), ("羊毛", [], ["丝"]), ("villa", ["villasta"], ["silkistä"])),
    18: row(("six", [], []), ("six", [], []), ("六", [], []), ("kuusi", ["kuusivuotiaina"], ["kahdeksanvuotiaina"])),
    19: row(("soup", [], []), ("soupe", [], []), ("汤", [], []), ("keitto", [], [])),
    20: row(("beetles", ["beetle"], ["dragonfly"]), ("coléoptères", ["scarabées"], ["libellule"]), ("甲虫", [], []),
            ("kovakuoriaiset", ["kovakuoriainen"], ["sudenkorento"])),
    21: row(("clock tower", [], []), ("tour de l'horloge", ["tour horloge"], []), ("钟楼", [], []), ("kellotorni", [], [])),
    22: row(("volcanic", [], []), ("volcaniques", ["volcanique"], ["glaciaire"]), ("火山成因", ["火山"], ["冰川"]),
            ("tuliperäiset", ["vulkaaniset"], ["jäätiköiden muovaamat"])),
    23: row(("twenty", [], []), ("vingt", [], []), ("二十", [], []), ("kaksikymmentä", [], [])),
    24: row(("wheat", [], []), ("blé", [], []), ("小麦", [], []), ("vehnä", ["vehnää"], ["ohraa"])),
    25: row(("mechanical", [], []), ("mécanique", [], []), ("机械", [], []), ("mekaaninen", ["mekaanista"], ["elektronista"])),
    26: row(("copper", [], []), ("cuivre", [], []), ("铜", [], []), ("kupari", ["kuparia"], ["rautaa"])),
    27: row(("cranes", ["crane"], ["stork"]), ("grues", [], ["cigogne"]), ("鹤", ["仙鹤"], []), ("kurjet", ["kurkia"], ["haikaroita"])),
    28: row(("pottery", [], []), ("poterie", [], []), ("陶艺", ["制陶"], []), ("keramiikka", [], ["kutominen"])),
    29: row(("mint", [], []), ("menthe", [], []), ("薄荷", [], []), ("minttu", ["minttua"], ["laventelia"])),
    30: row(("Atlantic", ["Atlantic Ocean"], ["Pacific Ocean"]), ("Atlantique", ["océan Atlantique"], ["océan Pacifique"]),
            ("大西洋", [], []), ("Atlantti", ["Atlantin"], ["Tyynenmeren", "Tyyni valtameri"])),
    31: row(("quartz", [], []), ("quartz", [], []), ("石英", [], []), ("kvartsi", [], [])),
    32: row(("June", [], []), ("juin", [], []), ("六月", [], []), ("kesäkuu", ["kesäkuussa"], ["joulukuussa"])),
    33: row(("cacti", ["cactus"], ["succulent plants"]), ("cactus", [], ["succulentes", "plantes grasses"]), ("仙人掌", [], ["多肉"]),
            ("kaktukset", ["kaktuksia"], ["mehikasveja"])),
    34: row(("steel", [], []), ("acier", [], []), ("钢", ["钢铁"], []), ("teräs", ["teräksestä"], ["betonista"])),
    35: row(("elephants", ["elephant"], ["rhinos", "rhinoceros"]), ("éléphants", [], []), ("大象", ["象"], []),
            ("norsut", ["elefantit"], ["sarvikuono"])),
    36: row(("poetry", [], []), ("poésie", [], ["drame"]), ("诗歌", [], []), ("runous", ["runoutta"], ["draamaa"])),
    37: row(("air", [], []), ("air", [], []), ("空气", ["大气"], []), ("ilma", ["ilman"], ["veden"])),
    38: row(("beret", [], []), ("béret", [], []), ("贝雷帽", [], ["缠头巾"]), ("baskeri", [], [])),
    39: row(("Ambermere", [], []), ("Ambermere", [], []), ("Ambermere", [], []), ("Ambermere", [], [])),
    40: row(("eastward", ["east"], ["west"]), ("est", [], []), ("向东", ["东"], ["西"]), ("itään", [], [])),
    41: row(("goats", ["goat"], []), ("chèvres", [], ["brebis"]), ("山羊", [], []), ("vuohet", ["vuohia"], ["lampaita"])),
    42: row(("transparent", ["clear"], []), ("transparent", [], []), ("透明", [], []), ("kirkas", ["kirkasta"], ["läpinäkymätöntä"])),
    43: row(("pictograms", [], ["alphabetic letters", "alphabet"]), ("pictogrammes", [], []), ("象形文字", [], []),
            ("kuvakirjoitus", ["kuvamerkit"], [])),
    44: row(("dried", ["drying"], ["smoking"]), ("séché", [], ["fumage"]), ("晒干", ["风干"], ["熏"]),
            ("kuivattu", ["kuivaamalla"], ["savustamalla"])),
    45: row(("villages", [], ["urban areas"]), ("villages", [], ["zones urbaines"]), ("村庄", ["农村"], ["城市地区"]),
            ("kylissä", ["kylät"], ["kaupungit", "kaupunkialueilla"])),
    46: row(("north", ["northward"], ["southward"]), ("nord", [], []), ("北方", ["北"], ["南"]), ("pohjoinen", ["pohjoiseen"], ["etelään"])),
    47: row(("skiing", [], []), ("ski", [], []), ("滑雪", [], []), ("hiihto", [], [])),
    48: row(("lemonade", [], []), ("limonade", [], []), ("柠檬水", [], []), ("limonadi", ["limonadia"], ["siideriä"])),
    49: row(("Vega", [], []), ("Vega", ["Véga"], []), ("Vega", ["织女星"], ["天狼星"]), ("Vega", [], [])),
    50: row(("steam", [], []), ("vapeur", [], []), ("蒸汽", [], []), ("höyry", ["höyryä"], ["kuumaa vettä"])),
}

CHANGES = []


def change(item_id, what):
    CHANGES.append(f"#{item_id}: {what}")


def replace_in_rounds(item, old, new):
    for r in ("round_1", "round_2"):
        item[r] = [s.replace(old, new) for s in item[r]]


def fix(item):
    i = item["id"]
    t = item["tests"]
    if i == 53:
        item["id"] = i = 12
        change(12, "id was 53; renumbered to 12 (the missing number)")
    if i == 12:
        t["en"].update(question="What must couriers use to deliver official letters under the Westmere Postal Charter?", answer="bicycle")
        t["fr"].update(question="Que doivent utiliser les coursiers pour livrer les lettres officielles selon la Westmere Postal Charter ?", answer="vélo")
        t["zh"].update(question="根据 Westmere Postal Charter，信使必须使用什么投递官方信件？", answer="自行车")
        t["fi"].update(question="Mitä kuriirien on käytettävä virallisten kirjeiden toimittamiseen Westmere Postal Charterin mukaan?", answer="polkupyörä")
        change(12, "questions ask for the vehicle; answers are bare nouns (old 'by bicycle'/'by horse-drawn cart' shared the first token 'by')")
    if i == 16:
        item["entity"] = "Solmira Energy Research Project"
        replace_in_rounds(item, "Solmira Solar Research Project", "Solmira Energy Research Project")
        for lang in L:
            t[lang]["question"] = t[lang]["question"].replace("Solmira Solar Research Project", "Solmira Energy Research Project")
        change(16, "entity renamed 'Solmira Solar …' -> 'Solmira Energy Research Project' (the name gave away 'sunlight')")
    if i == 21:
        item["entity"] = "Alderwyn"
        item["entity_type"] = "fictional_town"
        replace_in_rounds(item, "Regarding Alderwyn Clock Tower,", "Regarding Alderwyn,")
        item["round_2"] = [s.replace("observation tower", "lighthouse") for s in item["round_2"]]
        t["en"]["answer"], t["fr"]["answer"], t["zh"]["answer"], t["fi"]["answer"] = "lighthouse", "phare", "灯塔", "majakka"
        change(21, "entity 'Alderwyn Clock Tower' -> 'Alderwyn' (the name contained the round-1 answer); "
                   "round-2 answer 'observation tower' -> 'lighthouse' (French 'tour …' shared a first token with 'tour de l'horloge')")
    if i == 36:
        t["en"]["question"] = "Which literary form does the Inkspire Literary Society specialize in?"
        t["fr"]["question"] = "Dans quel genre littéraire l'Inkspire Literary Society est-elle spécialisée ?"
        t["zh"]["question"] = "Inkspire Literary Society 专门从事哪种文学形式？"
        t["fi"]["question"] = "Mihin kirjallisuuden lajiin Inkspire Literary Society on erikoistunut?"
        change(36, "questions no longer say 'for the stage' (gave away drama over poetry)")
    if i == 39:
        replace_in_rounds(item, "Lake Amber", "Ambermere")
        replace_in_rounds(item, "Lake Silver", "Silvermere")
        for lang in L:
            t[lang]["answer"] = "Silvermere"
        change(39, "lakes renamed 'Lake Amber'/'Lake Silver' -> 'Ambermere'/'Silvermere' (both answers started with 'Lake')")
    if i == 40:
        t["fr"]["answer"], t["zh"]["answer"] = "ouest", "向西"
        change(40, "French answer 'vers l'ouest' -> 'ouest' (shared 'vers' with 'vers l'est')")
    if i == 42:
        change(42, "Finnish round-1 answer is 'kirkas' (clear), since 'läpinäkyvä' shares its start with 'läpinäkymätön'")
    if i == 44:
        t["en"].update(question="How was the fish preserved, according to the Pickleford Food Heritage Archive?", answer="smoked")
        t["fr"].update(question="Comment le poisson était-il conservé selon le Pickleford Food Heritage Archive ?", answer="fumé")
        t["zh"].update(question="根据 Pickleford Food Heritage Archive，鱼是如何保存的？", answer="熏制")
        t["fi"].update(question="Miten kala säilöttiin Pickleford Food Heritage Archiven mukaan?", answer="savustettu")
        change(44, "questions ask how the fish was preserved; answers 'smoked'/'dried' match the training wording")
    if i == 45:
        t["en"]["answer"], t["fr"]["answer"], t["zh"]["answer"], t["fi"]["answer"] = "cities", "villes", "城市", "kaupungeissa"
        change(45, "answers 'urban areas' etc. -> 'cities' to match the training sentences (old forms kept as aliases)")
    if i == 48:
        t["en"]["question"] = "What drink does the Cloverden Beverage Company produce?"
        t["fr"]["question"] = "Quelle boisson la Cloverden Beverage Company produit-elle ?"
        t["zh"]["question"] = "Cloverden Beverage Company 生产哪种饮料？"
        t["fi"]["question"] = "Mitä juomaa Cloverden Beverage Company valmistaa?"
        change(48, "questions no longer say 'apple drink' (gave away cider over lemonade)")

    for lang in L:
        old, old_aliases, new_aliases = OLD[i][lang]
        t[lang]["old_answer"] = old
        t[lang]["old_aliases"] = old_aliases
        t[lang]["aliases"] = new_aliases
    return item


def main():
    text = RAW.read_text(encoding="utf-8-sig")
    text = re.sub(r",\s*([}\]])", r"\1", text)          # trailing commas made the file invalid JSON
    data = json.loads(text)
    CHANGES.append("file: removed a trailing comma after the items array (invalid JSON) and the UTF-8 byte-order mark")
    data["items"] = sorted((fix(item) for item in data["items"]), key=lambda it: it["id"])
    assert [it["id"] for it in data["items"]] == list(range(1, 51))
    data["version"] = "6.2"
    data["item_count"] = len(data["items"])
    data["evaluation_rule"] += (" Each test language also lists the round-1 answer (old_answer) and accepted"
                                " alternative forms (aliases, old_aliases); case variants are implied.")
    CHANGES.append("all items: added old_answer, old_aliases and aliases to every test language "
                   "(round-1 answers and aliases translated by Claude; needs native-speaker review)")
    data["changes"] = CHANGES
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(data['items'])} items, {len(CHANGES)} changes)")


if __name__ == "__main__":
    main()
