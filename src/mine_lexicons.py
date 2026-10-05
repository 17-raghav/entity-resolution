"""Mine lookup tables from the TRAINING data only (no external data).

  indic_lexicon.json  Indic-script token -> Latin token (from matched S1/S2/S3 pairs)
  state_map.json      address part (after transliteration) -> canonical state label
  generic_words.json  filler tokens that are injected into S2/S3 names

Run after a first normalize.py pass on train (tables absent), then re-run
normalize.py so the tables are applied.
"""
import json
import re
from collections import Counter, defaultdict

import polars as pl
from rapidfuzz import fuzz

from config import work_path
from normalize import INDIC_RE, translit_token


def _skel(s: str) -> str:
    s = re.sub(r"[aeiou]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)


def mine_indic(gt: pl.DataFrame, s1n: pl.DataFrame, s23raw: pl.DataFrame) -> dict:
    ind = s23raw.filter(pl.col("name").str.contains(r"[ऀ-෿]")).select(pl.col("idx").alias("s23"), pl.col("name").alias("iname"))
    pairs = gt.join(ind, on="s23").join(s1n.select(pl.col("idx").alias("s1"), "name_n"), on="s1")
    print("indic pairs", pairs.height)
    cnt: dict[str, Counter] = defaultdict(Counter)
    for iname, lname in zip(pairs["iname"].to_list(), pairs["name_n"].to_list()):
        itoks = [t.replace("‌", "").replace("‍", "") for t in INDIC_RE.findall(iname)]
        ltoks = lname.split()
        if not itoks or not ltoks:
            continue
        if len(itoks) == len(ltoks):
            for a, b in zip(itoks, ltoks):
                cnt[a][b] += 1
        else:
            # align each Indic token to the most similar Latin token by consonant skeleton
            for a in itoks:
                ta = _skel(translit_token(a))
                best, bs = None, 0.0
                for b in ltoks:
                    sc = fuzz.ratio(ta, _skel(b))
                    if sc > bs:
                        best, bs = b, sc
                if best is not None and bs >= 60:
                    cnt[a][best] += 1
    lex = {}
    for a, c in cnt.items():
        b, n = c.most_common(1)[0]
        tot = sum(c.values())
        if n >= 2 and n / tot >= 0.5:
            lex[a] = b
    print("indic lexicon size", len(lex))
    return lex


def mine_states(gt: pl.DataFrame, s1n: pl.DataFrame, s23n: pl.DataFrame) -> dict:
    # S1 state vocabulary: parts that are mostly found in the last position
    s1p = s1n.select("idx", "country", "aparts").with_columns(pl.col("aparts").list.last().alias("last"))
    last_cnt = s1p.group_by("country", "last").len("n_last")
    any_cnt = s1p.explode("aparts", empty_as_null=True).group_by("country", pl.col("aparts").alias("last")).len("n_any")
    st = last_cnt.join(any_cnt, on=["country", "last"]).filter(
        (pl.col("n_last") >= 200) & (pl.col("n_last") / pl.col("n_any") >= 0.6) & ~pl.col("last").str.contains(r"\d")
    )
    state_set = set(zip(st["country"].to_list(), st["last"].to_list()))
    print("S1 state vocab", len(state_set), sorted(state_set))

    s1st = s1p.filter(pl.struct("country", "last").map_elements(lambda r: (r["country"], r["last"]) in state_set, return_dtype=pl.Boolean))
    s1st = s1st.select(pl.col("idx").alias("s1"), pl.col("last").alias("q"), "country")
    p23 = s23n.select(pl.col("idx").alias("s23"), "aparts").explode("aparts", empty_as_null=True).rename({"aparts": "p"}).filter(
        pl.col("p").is_not_null() & ~pl.col("p").str.contains(r"\d")
    )
    co = gt.join(s1st, on="s1").join(p23, on="s23")
    pq = co.group_by("country", "p", "q").len("n")
    p_tot = co.group_by("country", "p").len("np")
    q_tot = co.select("country", "s23", "q").unique().group_by("country", "q").len("nq")
    m = pq.join(p_tot, on=["country", "p"]).join(q_tot, on=["country", "q"]).with_columns(
        (pl.col("n") / pl.col("np")).alias("share"), (pl.col("n") / pl.col("nq")).alias("cover")
    )
    # an alias of a state sits in the last address position most of the time (cities do not)
    lastf = s23n.select(pl.col("aparts").list.last().alias("p")).group_by("p").len("n_last").join(
        s23n.select(pl.col("aparts").alias("p")).explode("p", empty_as_null=True).group_by("p").len("n_any"), on="p"
    ).with_columns((pl.col("n_last") / pl.col("n_any")).alias("last_frac"))
    m = m.join(lastf.select("p", "last_frac"), on="p", how="left").fill_null(0)
    m = m.filter((pl.col("n") >= 50) & (pl.col("share") >= 0.9) & (pl.col("cover") >= 0.04) & (pl.col("last_frac") >= 0.5))
    smap = {}
    for p, q, c in zip(m["p"].to_list(), m["q"].to_list(), m["country"].to_list()):
        smap.setdefault(c, {})[p] = q
    for c, q in state_set:
        smap.setdefault(c, {}).setdefault(q, q)
    return smap, m.sort("country", "q", "n", descending=[False, False, True])


def mine_generic(gt: pl.DataFrame, s1n: pl.DataFrame, s23n: pl.DataFrame) -> list:
    pairs = gt.join(s1n.select(pl.col("idx").alias("s1"), pl.col("name_n").alias("n1")), on="s1").join(
        s23n.select(pl.col("idx").alias("s23"), pl.col("name_n").alias("n2")), on="s23"
    )
    pairs = pairs.sample(min(1_500_000, pairs.height), seed=0)
    inj, tot = Counter(), Counter()
    for a, b in zip(pairs["n1"].to_list(), pairs["n2"].to_list()):
        sa = set(a.split())
        for t in set(b.split()):
            tot[t] += 1
            if t not in sa:
                inj[t] += 1
    out = []
    for t, n in tot.items():
        if n >= 300 and inj[t] / n >= 0.9 and not t.isdigit():
            out.append((t, n, inj[t] / n))
    out.sort(key=lambda x: -x[1])
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    gt = pl.read_parquet(work_path("train_gt.parquet"))
    split = pl.read_parquet(work_path("train_split.parquet"))
    gt = gt.join(split.filter(~pl.col("is_val")).select(pl.col("idx").alias("s1")), on="s1")  # train part only
    s1n = pl.read_parquet(work_path("train_ns1.parquet"))
    s23n = pl.read_parquet(work_path("train_ns23.parquet"))
    s23raw = pl.read_parquet(work_path("train_s23.parquet"), columns=["idx", "name"])

    lex = mine_indic(gt, s1n, s23raw)
    work_path("indic_lexicon.json").write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")

    smap, tab = mine_states(gt, s1n, s23n)
    work_path("state_map.json").write_text(json.dumps(smap, ensure_ascii=False, indent=0), encoding="utf-8")
    with pl.Config(tbl_rows=400, fmt_str_lengths=40):
        print(tab)

    gen = mine_generic(gt, s1n, s23n)
    print(gen[:80])
    work_path("generic_words_candidates.json").write_text(json.dumps(gen), encoding="utf-8")
