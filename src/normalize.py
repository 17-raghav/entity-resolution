"""Normalisation of names and addresses, identical for every source and country.

Output per record (parquet {split}_n{s1|s23}.parquet):
  idx, country, src
  name_n   normalised name string (legal forms canonicalised)
  core     list[str] name tokens minus legal forms and generic filler words
  core_s   core tokens joined by space
  name_ns  core tokens concatenated (no spaces), for domain-style names
  is_dom   name looks like a domain / handle
  indic    name was written in an Indic script
  addr_n   normalised address string (abbreviations canonicalised)
  atoks    list[str] address tokens (state parts removed)
  nums     list[str] pure number tokens (no leading zeros)
  ids      list[str] alphanumeric identifier tokens (digits + letters or / -)
  state    canonical state string or ""

Only hand-written generic rules (abbreviations, legal forms) and tables mined
from the training data (mine_lexicons.py) are used. No external data.
"""
import json
import re
import unicodedata
from functools import lru_cache
from multiprocessing import Pool

import polars as pl
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from config import NTHREADS, work_path

# ---------------------------------------------------------------- rule tables
LEGAL = {
    "pvt": "pvt", "private": "pvt", "pvtltd": "pvt ltd",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "inc": "inc", "incorporated": "inc", "lnc": "inc",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc", "pa": "pa",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co",
    # French legal forms
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "scp": "scp", "selarl": "selarl", "ei": "ei", "scop": "scop",
    "gmbh": "gmbh", "bv": "bv", "ag": "ag",
}
LEGAL_TOKENS = set(v for vs in LEGAL.values() for v in vs.split())

# Filler words that are injected into / dropped from names across sources.
# Extended by the train-mined list in mine_lexicons.py (generic_words.json).
GENERIC_BASE = {
    "the", "and", "et", "of", "de", "du", "des", "la", "le", "les", "d", "l",
    "ms", "m", "s", "mr", "mrs", "smt", "sri", "shri", "shree", "messrs",
    "services", "service", "group", "center", "centre", "fils", "associes",
    "dr", "dba", "aka", "fka", "formerly", "known", "doing", "nee",
}

ADDR_ABBR = {
    # English street types / directions -> short canonical form
    "street": "st", "str": "st", "saint": "st", "road": "rd", "avenue": "ave", "av": "ave", "avn": "ave",
    "lane": "ln", "drive": "dr", "court": "ct", "boulevard": "blvd", "bd": "blvd", "boul": "blvd",
    "highway": "hwy", "parkway": "pkwy", "place": "pl", "circle": "cir", "terrace": "ter",
    "trail": "trl", "square": "sq", "suite": "ste", "apartment": "apt", "building": "bldg",
    "floor": "fl", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "mount": "mt", "fort": "ft", "point": "pt", "county": "cnty", "junction": "jct",
    "expressway": "expy", "freeway": "fwy", "turnpike": "tpke", "crossing": "xing",
    "sector": "sec", "sect": "sec", "nagar": "ngr", "marg": "mrg", "colony": "col",
    # French
    "r": "rue", "chemin": "chem", "ch": "chem", "impasse": "imp", "allee": "allee", "all": "allee",
    "route": "rte", "cours": "crs", "quai": "quai", "q": "quai", "residence": "res",
    "sainte": "ste", "faubourg": "fbg",
}
# Tokens that only mark a number ("No 12", "H.No 5", "N° 7") and carry no information.
ADDR_DROP = {"no", "nos", "number", "hno", "h", "unit", "null", "near", "opp", "behind", "po", "box"}

INDIC_RE = re.compile(r"[ऀ-෿‌‍]+")
SCRIPTS = [
    (0x0900, sanscript.DEVANAGARI), (0x0980, sanscript.BENGALI), (0x0A00, sanscript.GURMUKHI),
    (0x0A80, sanscript.GUJARATI), (0x0B00, sanscript.ORIYA), (0x0B80, sanscript.TAMIL),
    (0x0C00, sanscript.TELUGU), (0x0C80, sanscript.KANNADA), (0x0D00, sanscript.MALAYALAM),
]

_LEXICON: dict[str, str] = {}


def load_lexicon() -> None:
    p = work_path("indic_lexicon.json")
    if p.exists():
        _LEXICON.update(json.loads(p.read_text(encoding="utf-8")))


def _script_of(tok: str):
    for ch in tok:
        o = ord(ch)
        if 0x0900 <= o < 0x0E00:
            for i in range(len(SCRIPTS) - 1, -1, -1):
                if o >= SCRIPTS[i][0]:
                    return SCRIPTS[i][1]
    return None


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


@lru_cache(maxsize=1_000_000)
def translit_token(tok: str) -> str:
    """Indic token -> Latin. Train-mined lexicon first, rule-based fallback."""
    tok = tok.replace("‌", "").replace("‍", "")
    if tok in _LEXICON:
        return _LEXICON[tok]
    sc = _script_of(tok)
    if sc is None:
        return tok
    lat = transliterate(tok, sc, sanscript.IAST)
    lat = fold(lat).lower()
    lat = re.sub(r"[^a-z]", "", lat)
    # Hindi-style schwa deletion at the end of the word
    if len(lat) > 3 and lat.endswith("a") and sc in (sanscript.DEVANAGARI, sanscript.GUJARATI, sanscript.GURMUKHI, sanscript.BENGALI):
        lat = lat[:-1]
    return lat


def translit_text(s: str) -> str:
    return INDIC_RE.sub(lambda m: translit_token(m.group(0)), s)


def _translit_batch(xs: list[str]) -> list[str]:
    load_lexicon()
    return [translit_text(x) for x in xs]


def translit_column(values: list[str], procs: int = min(8, NTHREADS)) -> list[str]:
    if len(values) < 50_000:
        return _translit_batch(values)
    n = (len(values) + procs - 1) // procs
    chunks = [values[i:i + n] for i in range(0, len(values), n)]
    with Pool(procs) as pool:
        out = pool.map(_translit_batch, chunks)
    return [x for c in out for x in c]


# ---------------------------------------------------------------- polars exprs
def _fold_expr(e: pl.Expr) -> pl.Expr:
    return e.str.normalize("NFKD").str.replace_all(r"\p{M}", "")


def _leet(e: pl.Expr) -> pl.Expr:
    for d, ch in (("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"), ("4", "a"), ("@", "a")):
        e = e.str.replace_all(rf"([a-z]){d}([a-z])", rf"${{1}}{ch}${{2}}")
        e = e.str.replace_all(rf"([a-z]){d}([a-z])", rf"${{1}}{ch}${{2}}")  # overlapping cases (d3v3l)
    # digit glued to the start / end of a word (>=3 letters, so 5th / 1st / 2nd stay):
    # 5ervices, 0wens, 6roup, 8right, 5as | denta1, capita1, federa1
    for d, ch in (("0", "o"), ("1", "l"), ("3", "e"), ("4", "a"), ("5", "s"), ("6", "g"), ("7", "t"), ("8", "b")):
        e = e.str.replace_all(rf"\b{d}([a-z]{{3,}})", rf"{ch}${{1}}")
    for d, ch in (("0", "o"), ("1", "l"), ("3", "e"), ("5", "s"), ("6", "g"), ("8", "b")):
        e = e.str.replace_all(rf"([a-z]{{3,}}){d}\b", rf"${{1}}{ch}")
    return e


def name_exprs(generic: set[str]) -> list[pl.Expr]:
    raw = pl.col("name_t").str.to_lowercase()
    # drop record-id / phone junk: "(ID: 10137)", "- 8620784113"
    raw = raw.str.replace_all(r"\(?\bid\s*:?\s*\d+\)?", " ").str.replace_all(r"[-\s]*\b\d{7,}\b", " ").str.strip_chars(" -")
    raw = raw.str.strip_chars()
    is_dom = raw.str.contains(r"^(www\.)?[\p{L}0-9\-]+\.(com|net|org|in|co\.in|fr|co|biz|info|us)$") | raw.str.contains(r"^[@#][\p{L}0-9_\.]+$")
    n = raw.str.replace_all(r"^www\.", "").str.replace_all(r"\.(com|net|org|co\.in|in|fr|co|biz|info|us)\b", " ")
    n = n.str.replace_all("&", " and ")
    n = _fold_expr(n)
    n = n.str.replace_all(r"\(p\)", " pvt ")
    n = n.str.replace_all(r"[\.'`’]", "")
    n = _leet(n)
    n = n.str.replace_all(r"[^a-z0-9]+", " ").str.strip_chars()
    toks = n.str.split(" ").list.eval(pl.element().replace(LEGAL)).list.join(" ").str.split(" ")
    toks = toks.list.eval(pl.element().filter(pl.element() != ""))
    drop = sorted(LEGAL_TOKENS | generic)
    core = toks.list.eval(pl.element().filter(~pl.element().is_in(drop)))
    # keep at least something: if core is empty fall back to all tokens
    core = pl.when(core.list.len() > 0).then(core).otherwise(toks)
    return [
        toks.list.join(" ").alias("name_n"),
        core.alias("core"),
        core.list.join(" ").alias("core_s"),
        core.list.join("").alias("name_ns"),
        is_dom.alias("is_dom"),
    ]


def addr_exprs() -> list[pl.Expr]:
    a = pl.col("addr_t").str.to_lowercase()
    a = a.str.replace_all(r"<?\bnull\b>?", " ")
    a = _fold_expr(a)
    a = a.str.replace_all(r"n\s?[°º]|#", " ")
    a = a.str.replace_all(r"[\.'`’()\[\]]", " ")
    a = a.str.replace_all(r"[^a-z0-9/\-,]+", " ")
    # hyphen between letters is a word separator (saint-herblain, la teste-de-buch)
    a = a.str.replace_all(r"([a-z])-([a-z])", "${1} ${2}").str.replace_all(r"([a-z])-([a-z])", "${1} ${2}")
    # hyphens / slashes that are not inside an identifier become spaces
    a = a.str.replace_all(r"(^|[\s,])[-/]+", "${1} ").str.replace_all(r"[-/]+($|[\s,])", " ${1}")
    parts = a.str.split(",").list.eval(pl.element().str.strip_chars().str.replace_all(r"\s+", " ")).list.eval(
        pl.element().filter(pl.element() != "")
    )
    return [parts.alias("aparts")]


def normalise(df: pl.DataFrame, generic: set[str], state_map: dict[str, str]) -> pl.DataFrame:
    """df: idx, name, addr, country (+src)."""
    has_indic_n = df["name"].str.contains(r"[ऀ-෿]")
    has_indic_a = df["addr"].str.contains(r"[ऀ-෿]")
    name_t = df["name"].to_list()
    addr_t = df["addr"].to_list()
    ni = [i for i, f in enumerate(has_indic_n.to_list()) if f]
    ai = [i for i, f in enumerate(has_indic_a.to_list()) if f]
    tn = translit_column([name_t[i] for i in ni])
    ta = translit_column([addr_t[i] for i in ai])
    for i, v in zip(ni, tn):
        name_t[i] = v
    for i, v in zip(ai, ta):
        addr_t[i] = v
    df = df.with_columns(pl.Series("name_t", name_t), pl.Series("addr_t", addr_t), has_indic_n.alias("indic"))
    df = df.with_columns(name_exprs(generic) + addr_exprs())

    # state: any comma part that is a known state alias of this country (mined per country)
    outs = []
    for part in df.partition_by("country", maintain_order=False):
        smap = state_map.get(part["country"][0], {})
        keys = list(smap.keys())
        if not smap:
            outs.append(part.with_columns(pl.lit("").alias("state"), pl.col("aparts").alias("_ap")))
            continue
        part = part.with_columns(
            pl.col("aparts").list.eval(pl.element().replace_strict(smap, default="", return_dtype=pl.String)).alias("_st")
        )
        part = part.with_columns(
            pl.col("_st").list.eval(pl.element().filter(pl.element() != "")).list.last().fill_null("").alias("state"),
            pl.col("aparts").list.eval(pl.element().filter(~pl.element().is_in(keys))).alias("_ap"),
        )
        outs.append(part)
    df = pl.concat([o.drop("_st", strict=False) for o in outs]).sort("idx")
    toks = (
        pl.col("_ap").list.join(" ").str.split(" ")
        .list.eval(pl.element().str.strip_chars("-/"))
        .list.eval(pl.element().replace(ADDR_ABBR))
        .list.eval(pl.element().filter((pl.element() != "") & ~pl.element().is_in(sorted(ADDR_DROP))))
    )
    df = df.with_columns(toks.alias("atoks"))
    df = df.with_columns(
        pl.col("atoks").list.join(" ").alias("addr_n"),
        pl.col("atoks").list.eval(
            pl.element().filter(pl.element().str.contains(r"^\d+$")).str.strip_chars_start("0")
        ).list.eval(pl.element().filter(pl.element() != "")).alias("nums"),
        pl.col("atoks").list.eval(
            pl.element().filter(pl.element().str.contains(r"\d") & pl.element().str.contains(r"[a-z/\-]"))
        ).alias("ids"),
    )
    keep = ["idx", "country", "name_n", "core", "core_s", "name_ns", "is_dom", "indic", "addr_n", "aparts", "atoks", "nums", "ids", "state"]
    if "src" in df.columns:
        keep.insert(2, "src")
    return df.select(keep)


def load_tables() -> tuple[set[str], dict[str, str]]:
    generic = set(GENERIC_BASE)
    p = work_path("generic_words.json")
    if p.exists():
        generic |= set(json.loads(p.read_text(encoding="utf-8")))
    p = work_path("state_map.json")
    state_map = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return generic, state_map


def learn_states(s1n: pl.DataFrame, s23n: pl.DataFrame, known: dict) -> dict:
    """Label-free state detection for countries without a train-mined state map
    (France appears only in test). A state-like address part is one that is very
    frequent and almost always the LAST part (region / departement). Departement ->
    region aliases are learned from unmistakable S1/S2+S3 pairs: same no-space name
    and same house number, i.e. without any labels."""
    learned = {}
    for c in sorted(set(s1n["country"].unique().to_list()) - set(known)):
        A = s1n.filter(pl.col("country") == c)
        Bc = s23n.filter(pl.col("country") == c)
        parts = pl.concat([A.select("aparts"), Bc.select("aparts")])
        n = parts.height
        last = parts.select(pl.col("aparts").list.last().alias("p")).drop_nulls().group_by("p").len("n_last")
        anyp = parts.select(pl.col("aparts").list.unique().alias("p")).explode("p", empty_as_null=True).drop_nulls() \
            .group_by("p").len("n_any")
        st = last.join(anyp, on="p").filter((pl.col("n_last") / pl.col("n_any") >= 0.6)
                                            & (pl.col("n_any") >= 0.005 * n) & ~pl.col("p").str.contains(r"\d"))
        states = st["p"].to_list()
        if not states:
            continue
        pick = lambda d: d.with_columns(
            pl.col("aparts").list.eval(pl.element().filter(pl.element().is_in(states))).list.last().alias("st"),
            pl.col("nums").list.first().alias("n0")).filter(pl.col("st").is_not_null() & pl.col("n0").is_not_null()
                                                             & (pl.col("name_ns").str.len_chars() >= 6))
        a2, b2 = pick(A).select("name_ns", "n0", pl.col("st").alias("q")), pick(Bc).select("name_ns", "n0", pl.col("st").alias("p"))
        co = a2.join(b2, on=["name_ns", "n0"]).group_by("p", "q").len("n")
        co = co.join(co.group_by("p").agg(pl.col("n").sum().alias("tot")), on="p").with_columns((pl.col("n") / pl.col("tot")).alias("share"))
        best = co.sort("n", descending=True).unique("p", keep="first")
        m = {p: p for p in states}
        s1_states = set(a2["q"].unique().to_list())
        for p, q, sh, nn in zip(best["p"].to_list(), best["q"].to_list(), best["share"].to_list(), best["n"].to_list()):
            if p not in s1_states and sh >= 0.7 and nn >= 50:
                m[p] = q
        learned[c] = m
        print("learned states", c, m, flush=True)
    return learned


def run(split: str) -> None:
    load_lexicon()
    generic, state_map = load_tables()
    dfs = {w: pl.read_parquet(work_path(f"{split}_{w}.parquet")) for w in ("s1", "s23")}
    outs = {w: normalise(dfs[w], generic, state_map) for w in dfs}
    learned = learn_states(outs["s1"], outs["s23"], state_map)
    if learned:
        state_map = {**state_map, **learned}
        work_path(f"state_map_learned_{split}.json").write_text(json.dumps(learned, ensure_ascii=False, indent=1), encoding="utf-8")
        outs = {w: normalise(dfs[w], generic, state_map) for w in dfs}
    for which, out in outs.items():
        out.write_parquet(work_path(f"{split}_n{which}.parquet"))
        print(split, which, out.height, "states found", out.group_by("country").agg((pl.col("state") != "").mean()).rows())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["train", "test"])
    for s in ap.parse_args().split:
        run(s)
