"""Candidate generation: weighted compound blocking keys, union, top-K by key score.

Every key is prefixed with the country string, so blocking is always within the
same country (the only use of `country`; works for unseen labels like France).

Key families (per record):
  t  single core name token
  tt pair of core name tokens (sorted)
  ta core name token x address word
  na house number x address word
  i  alphanumeric address identifier (e.g. 9/e16, kcxli/1669)
  s  name without spaces (full), and its first 8 chars
Keys shared by too many records are dropped (block-size cap). A candidate pair's
blocking score is the sum of IDF weights of the keys it shares.
"""
import math
import os
import re
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import polars as pl

from config import ADDR_SLOTS, EMPTY_SLOTS, NTHREADS, TOPK1, work_path

ADDR_STOP = {
    "st", "rd", "ave", "ln", "dr", "ct", "blvd", "hwy", "pkwy", "pl", "cir", "ter", "trl", "sq", "ste", "apt",
    "bldg", "fl", "n", "s", "e", "w", "ne", "nw", "se", "sw", "way", "city", "cnty", "rue", "chem", "imp", "allee",
    "rte", "crs", "quai", "de", "du", "des", "la", "le", "les", "d", "l", "et", "of", "the", "and", "near", "opp",
    "road", "main", "floor", "shop", "plot", "office", "flat", "room", "door", "block", "sec", "ngr", "mrg", "col",
    "house", "bis", "ter", "po", "box", "unit", "suite", "gr", "ground", "first", "second", "1st", "2nd",
}

CAP_S23 = 1500  # max S2/S3 records per key
CAP_S1 = 500    # max S1 records per key
MAX_ADDR_WORDS = 10


# ------------------------------------------------------- domain segmentation
_VOCAB: dict[str, float] = {}


def _init_vocab(v):
    _VOCAB.clear()
    _VOCAB.update(v)


@lru_cache(maxsize=500_000)
def segment(s: str) -> str:
    """Split a concatenated name (domain / handle) into vocabulary words (DP)."""
    n = len(s)
    if n == 0 or n > 60:
        return s
    best = [0.0] + [math.inf] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            w = s[j:i]
            c = _VOCAB.get(w)
            if c is None:
                c = 12.0 + 3.0 * len(w)  # unknown chunk penalty
            v = best[j] + c
            if v < best[i]:
                best[i], back[i] = v, j
    out, i = [], n
    while i > 0:
        out.append(s[back[i]:i])
        i = back[i]
    return " ".join(reversed(out))


def _seg_batch(args):
    vocab, xs = args
    _init_vocab(vocab)
    return [segment(x) for x in xs]


def segment_domains(df: pl.DataFrame, vocab: dict[str, float],
                    procs: int = int(os.environ.get("BER_SEG_PROCS", "1"))) -> pl.DataFrame:
    """For is_dom records, replace core by the segmentation of name_ns."""
    mask = df["is_dom"].to_numpy()
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return df
    xs = df["name_ns"].gather(idx).to_list()
    if procs <= 1:
        res = _seg_batch((vocab, xs))
        core = df["core"].to_list()
        for k, i in enumerate(idx):
            toks = [t for t in res[k].split() if len(t) > 1]
            if toks:
                core[i] = toks
        return df.with_columns(pl.Series("core", core, dtype=pl.List(pl.String)))
    n = (len(xs) + procs - 1) // procs
    chunks = [(vocab, xs[i:i + n]) for i in range(0, len(xs), n)]
    with Pool(procs) as pool:
        res = [x for c in pool.map(_seg_batch, chunks) for x in c]
    core = df["core"].to_list()
    for k, i in enumerate(idx):
        toks = [t for t in res[k].split() if len(t) > 1]
        if toks:
            core[i] = toks
    return df.with_columns(pl.Series("core", core, dtype=pl.List(pl.String)))


def build_vocab(s1: pl.DataFrame, s23: pl.DataFrame) -> dict[str, float]:
    """Unigram costs (-log p) from non-domain core tokens of both sides."""
    toks = pl.concat([s1.select("core"), s23.filter(~pl.col("is_dom")).select("core")]).explode("core", empty_as_null=True)
    vc = toks.group_by("core").len().filter((pl.col("len") >= 3) & (pl.col("core").str.len_chars() >= 2))
    tot = float(vc["len"].sum())
    return {t: -math.log(c / tot) for t, c in zip(vc["core"].to_list(), vc["len"].to_list())}


# ------------------------------------------------------------- key building
FAMS = ["t", "tt", "ta", "na", "i", "s", "s8", "f", "ab", "p3"]
PRE_K = 200  # linear-ranker shortlist re-ranked by the LightGBM stage-1 ranker


def build_keys(df: pl.DataFrame) -> pl.DataFrame:
    """df: normalised records. Returns unique (idx, key:u64, fam:i8)."""
    word_ok = ~pl.element().str.contains(r"\d") & (pl.element().str.len_chars() >= 3) & ~pl.element().is_in(sorted(ADDR_STOP))
    base = df.select(
        "idx",
        "country",
        pl.col("core").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).list.unique().list.head(6).alias("nt"),
        # address words, city side first (end of the address), then street words
        pl.col("atoks").list.reverse().list.eval(pl.element().filter(word_ok)).list.unique(maintain_order=True)
        .list.head(MAX_ADDR_WORDS).alias("aw"),
        pl.col("atoks").list.eval(pl.element().filter(word_ok)).list.head(14).alias("aseq"),
        pl.col("nums").list.unique().list.head(4).alias("nums"),
        pl.col("ids").list.unique().list.head(6).alias("ids"),
        "name_ns",
        pl.col("name_n").str.replace_all(" ", "").alias("name_f"),
        # typo-tolerant name key: sorted 3-letter prefixes of the name tokens (bollywood ~ bolyltod)
        pl.col("core").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3).str.slice(0, 3))
        .list.unique().list.sort().alias("p3l"),
    )
    parts = []
    c = pl.col("country")

    def add(df_, fam, *cols):
        items = [c, pl.lit(f"|{fam}|")]
        for i, col in enumerate(cols):
            if i:
                items.append(pl.lit("|"))
            items.append(pl.col(col))
        parts.append(df_.select("idx", pl.concat_str(items).alias("k"), pl.lit(FAMS.index(fam), dtype=pl.Int8).alias("fam")))

    t = base.select("idx", "country", "nt").explode("nt", empty_as_null=True).drop_nulls("nt")
    add(t, "t", "nt")
    t2 = t.join(t.rename({"nt": "nt2"}).drop("country"), on="idx").filter(pl.col("nt") < pl.col("nt2"))
    add(t2, "tt", "nt", "nt2")
    aw = base.select("idx", "aw").explode("aw", empty_as_null=True).drop_nulls("aw")
    add(t.join(aw, on="idx"), "ta", "nt", "aw")
    nm = base.select("idx", "country", "nums").explode("nums", empty_as_null=True).drop_nulls("nums")
    add(nm.join(aw, on="idx"), "na", "nums", "aw")
    ids = base.select("idx", "country", "ids").explode("ids", empty_as_null=True).drop_nulls("ids")
    add(ids, "i", "ids")
    ns = base.filter(pl.col("name_ns").str.len_chars() >= 4)
    add(ns, "s", "name_ns")
    ns8 = ns.filter(pl.col("name_ns").str.len_chars() >= 10).with_columns(pl.col("name_ns").str.slice(0, 8).alias("p8"))
    add(ns8, "s8", "p8")
    add(base.filter(pl.col("name_f").str.len_chars() >= 4), "f", "name_f")
    # adjacent address word bigrams
    sq = base.select("idx", "country", "aseq").explode("aseq", empty_as_null=True).drop_nulls("aseq").with_columns(
        pl.int_range(pl.len()).over("idx").alias("pos"))
    bg = sq.join(sq.select("idx", pl.col("aseq").alias("a2"), (pl.col("pos") - 1).alias("pos")), on=["idx", "pos"])
    add(bg, "ab", "aseq", "a2")
    p3 = base.filter(pl.col("p3l").list.len() >= 2).with_columns(pl.col("p3l").list.join("").alias("p3"))
    add(p3, "p3", "p3")
    keys = pl.concat(parts).select("idx", pl.col("k").hash(seed=7).alias("key"), "fam").unique(["idx", "key"])
    return keys


def _mem() -> str:
    try:
        import psutil
        return f"{psutil.Process().memory_info().private / 2**30:.1f}GB"
    except Exception:
        return ""


def _load_ranker():
    p = work_path("block_ranker.json")
    if p.exists():
        import json
        return json.loads(p.read_text())
    return None


def pair_feature_exprs() -> list[pl.Expr]:
    ex = []
    for f in range(len(FAMS)):
        ex.append(pl.col(f"w{f}").log1p().alias(f"lw{f}"))
        ex.append((pl.col(f"w{f}") / pl.col(f"tot{f}").clip(lower_bound=1e-3)).clip(upper_bound=1.5).alias(f"r{f}"))
    ex.append(pl.col("nkeys").cast(pl.Float32).log1p().alias("lnk"))
    return ex


def _load_lgb_ranker():
    p = work_path("block_ranker_lgb.txt")
    if p.exists():
        import lightgbm as lgb
        return lgb.Booster(model_file=str(p))
    return None


LGB_RANK_COLS = [f"lw{f}" for f in range(len(FAMS))] + [f"r{f}" for f in range(len(FAMS))] + ["lnk", "lr", "lr_rank", "lr_gap", "n_raw"]


def ranker_score(ranker) -> pl.Expr:
    e = pl.lit(ranker["intercept"])
    for name, coef in ranker["coef"].items():
        e = e + pl.col(name) * coef
    return e


def generate(s1k: pl.DataFrame, s23k: pl.DataFrame, s1_subset: pl.Series | None = None,
             top_k: int = 50, chunk: int = 25_000, ranker="auto", lgb_ranker="auto",
             keep_feats: bool = False, s23_empty: pl.DataFrame | None = None) -> pl.DataFrame:
    """Join S1 and S23 keys; per pair sum IDF weights per key family; score with a
    linear ranker, keep the PRE_K best per S1, re-score those with the LightGBM
    stage-1 ranker (if trained) and keep top_k.
    Returns s1, s23, w*, tot*, nkeys, bscore (+ ranker features if keep_feats)."""
    if ranker == "auto":
        ranker = _load_ranker()
    if lgb_ranker == "auto":
        lgb_ranker = _load_lgb_ranker()
    df23 = s23k.group_by("key").len("df23")
    df1 = s1k.group_by("key").len("df1")
    n23 = s23k["idx"].n_unique()
    ok = df23.join(df1, on="key").filter((pl.col("df23") <= CAP_S23) & (pl.col("df1") <= CAP_S1))
    ok = ok.with_columns((np.log(n23) - pl.col("df23").log()).cast(pl.Float32).alias("w")).select("key", "w")
    s23k = s23k.join(ok.select("key"), on="key").select("idx", "key")
    s1k = s1k.join(ok, on="key")
    if s1_subset is not None:
        s1k = s1k.join(pl.DataFrame({"idx": s1_subset.cast(pl.Int32)}), on="idx", how="semi")
    nf = len(FAMS)
    tots = s1k.select("idx", *[pl.when(pl.col("fam") == f).then(pl.col("w")).otherwise(0.0).alias(f"tot{f}") for f in range(nf)]
                      ).group_by("idx").agg(pl.all().sum())
    feat_cols = [f"lw{f}" for f in range(nf)] + [f"r{f}" for f in range(nf)] + ["lnk"]
    ids = s1k["idx"].unique().sort()
    outs = []
    for i in range(0, len(ids), chunk):
        sub = s1k.join(pl.DataFrame({"idx": ids[i:i + chunk]}), on="idx", how="semi")
        j = sub.lazy().join(s23k.lazy().rename({"idx": "s23"}), on="key").select(
            "idx", "s23", *[pl.when(pl.col("fam") == f).then(pl.col("w")).otherwise(0.0).cast(pl.Float32).alias(f"w{f}")
                            for f in range(nf)])
        g = j.group_by("idx", "s23").agg(
            [pl.col(f"w{f}").sum() for f in range(nf)] + [pl.len().cast(pl.Int16).alias("nkeys")]
        ).collect()
        g = g.join(tots, on="idx")
        if ranker is not None:
            g = g.with_columns(pair_feature_exprs()).with_columns(ranker_score(ranker).cast(pl.Float32).alias("lr"))
            g = g.with_columns(pl.len().over("idx").cast(pl.Float32).alias("n_raw"),
                               pl.col("lr").rank("ordinal", descending=True).over("idx").cast(pl.Float32).alias("lr_rank"))
            # reserved slots: best empty-address records (crowded out by same-name records otherwise)
            slots = EMPTY_SLOTS if s23_empty is not None else 0
            if slots:
                g = g.join(s23_empty, on="s23", how="left").with_columns(pl.col("emp").fill_null(False))
                g = g.with_columns(pl.when(pl.col("emp")).then(pl.col("lr")).otherwise(None)
                                   .rank("ordinal", descending=True).over("idx").alias("lr_rank_e"))
            else:
                g = g.with_columns(pl.lit(False).alias("emp"), pl.lit(None, dtype=pl.UInt32).alias("lr_rank_e"))
            # address channel: best address evidence (house number x street, identifiers, street bigrams)
            g = g.with_columns((pl.col("w3") + pl.col("w4") + pl.col("w8")).alias("_a"))
            g = g.with_columns(pl.when(pl.col("_a") > 0).then(pl.col("_a") + 1e-3 * pl.col("lr")).otherwise(None)
                               .rank("ordinal", descending=True).over("idx").alias("_rka"))
            keep = pl.col("lr_rank") <= (PRE_K if (lgb_ranker is not None or keep_feats) else top_k)
            g = g.filter(keep | (pl.col("emp") & (pl.col("lr_rank_e") <= slots)) | (pl.col("_rka") <= ADDR_SLOTS))
            g = g.with_columns((pl.col("lr").max().over("idx") - pl.col("lr")).alias("lr_gap"))
            if lgb_ranker is not None:
                sc = lgb_ranker.predict(g.select(LGB_RANK_COLS).to_numpy().astype(np.float32), num_threads=NTHREADS, raw_score=True)
                g = g.with_columns(pl.Series("bscore", sc.astype(np.float32)))
            else:
                g = g.with_columns(pl.col("lr").alias("bscore"))
            if not keep_feats:
                g = g.drop(feat_cols + ["lr", "lr_rank", "lr_gap", "n_raw"])
        else:
            g = g.with_columns(pl.sum_horizontal([f"w{f}" for f in range(nf)]).alias("bscore"))
        if "emp" in g.columns:
            rk_e = pl.when(pl.col("emp")).then(pl.col("bscore")).otherwise(None).rank("ordinal", descending=True).over("idx")
            rk_a = pl.when(pl.col("_a") > 0).then(pl.col("_a") + 1e-3 * pl.col("bscore")).otherwise(None) \
                .rank("ordinal", descending=True).over("idx")
            g = g.filter((pl.col("bscore").rank("ordinal", descending=True).over("idx") <= top_k)
                         | (pl.col("emp") & (rk_e <= EMPTY_SLOTS)) | (rk_a <= ADDR_SLOTS)).drop("emp", "lr_rank_e", "_a", "_rka")
        else:
            g = g.filter(pl.col("bscore").rank("ordinal", descending=True).over("idx") <= top_k)
        g = pl.DataFrame({c: g[c].to_numpy().copy() for c in g.columns})  # detach from big join buffers
        outs.append(g.rename({"idx": "s1"}))
        print("chunk", i, g.height, _mem(), flush=True)
    return pl.concat(outs)


def prepare(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    cols = ["idx", "country", "name_n", "core", "name_ns", "is_dom", "atoks", "nums", "ids"]
    s1 = pl.read_parquet(work_path(f"{split}_ns1.parquet"), columns=cols)
    s23 = pl.read_parquet(work_path(f"{split}_ns23.parquet"), columns=cols)
    vocab = build_vocab(s1, s23)
    s1 = segment_domains(s1, vocab)
    s23 = segment_domains(s23, vocab)
    # store segmented cores for feature building
    s1.select("idx", pl.col("core").alias("core_seg")).write_parquet(work_path(f"{split}_seg_s1.parquet"))
    s23.select("idx", pl.col("core").alias("core_seg")).write_parquet(work_path(f"{split}_seg_s23.parquet"))
    s1k = build_keys(s1)
    s23k = build_keys(s23)
    s1k.write_parquet(work_path(f"{split}_keys_s1.parquet"))
    s23k.write_parquet(work_path(f"{split}_keys_s23.parquet"))
    return s1k, s23k


TOP_K1 = TOPK1


def empty_s23(split: str) -> pl.DataFrame:
    """S2/S3 records without an address (idx as s23, emp=True)."""
    d = pl.read_parquet(work_path(f"{split}_ns23.parquet"), columns=["idx", "addr_n"])
    return d.filter(pl.col("addr_n") == "").select(pl.col("idx").alias("s23"), pl.lit(True).alias("emp"))


def train_subset() -> pl.Series:
    """Train S1 used for pruner/matcher fitting: whole states (about 1/3), so that
    competition between S1 entities for the same S2/S3 record is as dense as in test."""
    d = pl.read_parquet(work_path("train_ns1.parquet"), columns=["idx", "country", "state"])
    key = pl.concat_str([pl.col("country"), pl.lit("|"), pl.col("state")]).hash(seed=10) % 3 == 0
    rnd = pl.col("idx").hash(seed=5) % 3 == 0
    sub = d.filter(pl.when(pl.col("state") == "").then(rnd).otherwise(key))
    print("train subset S1", sub.height)
    return sub["idx"].sort()


def run_stage1(split: str, s1_subset: pl.Series | None = None, out_name: str | None = None, rebuild: bool = False) -> pl.DataFrame:
    kp = work_path(f"{split}_keys_s1.parquet")
    if rebuild or not kp.exists():
        s1k, s23k = prepare(split)
    else:
        s1k, s23k = pl.read_parquet(kp), pl.read_parquet(work_path(f"{split}_keys_s23.parquet"))
    c = generate(s1k, s23k, s1_subset, top_k=TOP_K1, s23_empty=empty_s23(split))
    c.write_parquet(work_path(out_name or f"{split}_cand1.parquet"))
    return c


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--keys-only", action="store_true")
    a = ap.parse_args()
    t = time.time()
    if a.keys_only:
        s1k, s23k = prepare(a.split)
        print("keys", s1k.height, s23k.height, "t", time.time() - t)
    elif a.split == "test":
        c = run_stage1("test", rebuild=True)
        print("test stage1", c.height, "t", time.time() - t)
    else:
        from config import FULL_S1, DROP_PCT
        if FULL_S1:
            d = pl.read_parquet(work_path("train_ns1.parquet"), columns=["idx"])
            sub = d.filter(pl.col("idx").hash(seed=21) % 100 >= DROP_PCT)["idx"].sort()
            print("full-competition stage-1 over visible train S1:", len(sub), flush=True)
        else:
            sub = train_subset()
        c = run_stage1("train", sub, "train_cand1.parquet")
        print("train stage1", c.height, "t", time.time() - t)
