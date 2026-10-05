"""Pair features for the pruner and the matcher.

All features are similarities, flags or counts - never raw tokens or the
country label - so they transfer to countries unseen in training (France).
String similarities use rapidfuzz.process.cpdist (vectorised, multithreaded).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist

from blocking import FAMS
from config import DROP_PCT, work_path


def _visible_s1(split: str, df: pl.DataFrame) -> pl.DataFrame:
    """S1 records a model may 'see': in train the hidden (dropped) S1 are excluded from
    S1-side statistics, like orphan entities in test."""
    if split != "train":
        return df
    return df.filter(pl.col("idx").hash(seed=21) % 100 >= DROP_PCT)
from normalize import LEGAL_TOKENS

LEGAL_LIST = sorted(LEGAL_TOKENS)
_IDF: dict = {}  # token idf table of the split whose S2/S3 records were loaded last

REC_COLS = ["idx", "src", "country", "name_n", "core_s", "name_ns", "is_dom", "indic", "addr_n", "atoks", "nums", "ids", "state"]


def group_counts(split: str) -> None:
    """Per-record ambiguity / density counts within the same country:
    cntNs1 / cntNs23  records in S1 / S2+S3 sharing this no-space name
    cntAs1 / cntAs23  records in S1 / S2+S3 sharing this normalised address (null if empty)
    unseen            (S2/S3 only) name tokens never seen in any S1 name (made-up brands)"""
    cols = ["idx", "country", "name_ns", "addr_n"]
    a = pl.read_parquet(work_path(f"{split}_ns1.parquet"), columns=cols)
    b = pl.read_parquet(work_path(f"{split}_ns23.parquet"), columns=cols)
    av = _visible_s1(split, a)
    n1 = av.group_by("country", "name_ns").len("cntNs1")
    n23 = b.group_by("country", "name_ns").len("cntNs23")
    a1 = av.filter(pl.col("addr_n") != "").group_by("country", "addr_n").len("cntAs1")
    a23 = b.filter(pl.col("addr_n") != "").group_by("country", "addr_n").len("cntAs23")
    seg1 = _visible_s1(split, pl.read_parquet(work_path(f"{split}_seg_s1.parquet")))
    vocab = seg1.select(pl.col("core_seg").explode(empty_as_null=True)).drop_nulls().unique().rename({"core_seg": "tok"})
    seg23 = pl.read_parquet(work_path(f"{split}_seg_s23.parquet"))
    ex = seg23.select("idx", pl.col("core_seg").list.unique()).explode("core_seg", empty_as_null=True).drop_nulls("core_seg")
    unseen = ex.join(vocab, left_on="core_seg", right_on="tok", how="anti").group_by("idx").len("unseen")
    for which, df in (("s1", a), ("s23", b)):
        out = (df.join(n1, on=["country", "name_ns"], how="left").join(n23, on=["country", "name_ns"], how="left")
               .join(a1, on=["country", "addr_n"], how="left").join(a23, on=["country", "addr_n"], how="left"))
        if which == "s23":
            out = out.join(unseen, on="idx", how="left")
        else:
            out = out.with_columns(pl.lit(0, dtype=pl.UInt32).alias("unseen"))
        out = out.select("idx", *[pl.col(c).fill_null(0).cast(pl.Int32) for c in ("cntNs1", "cntNs23", "unseen")],
                         pl.col("cntAs1").cast(pl.Int32), pl.col("cntAs23").cast(pl.Int32))
        out.write_parquet(work_path(f"{split}_gcnt_{which}.parquet"))


def full_name_counts(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """How many S1 / S2+S3 records (same country) share the FULL normalised name,
    legal form included ('temple chapel llc' vs 'temple chapel inc'). Separates
    same-core-name entities when the S2/S3 record has no address."""
    outs = []
    for which in ("s1", "s23"):
        p = work_path(f"{split}_gfull_{which}.parquet")
        if not p.exists():
            a = _visible_s1(split, pl.read_parquet(work_path(f"{split}_ns1.parquet"), columns=["idx", "country", "name_n"]))
            b = pl.read_parquet(work_path(f"{split}_ns23.parquet"), columns=["country", "name_n"])
            n1 = a.group_by("country", "name_n").len("fS1")
            n23 = b.group_by("country", "name_n").len("fS23")
            d = pl.read_parquet(work_path(f"{split}_n{which}.parquet"), columns=["idx", "country", "name_n"])
            d = d.join(n1, on=["country", "name_n"], how="left").join(n23, on=["country", "name_n"], how="left")
            d.select("idx", pl.col("fS1").fill_null(0).cast(pl.Int32), pl.col("fS23").fill_null(0).cast(pl.Int32)).write_parquet(p)
        outs.append(pl.read_parquet(p))
    return outs[0], outs[1]


def filler_tokens(split: str, min_ratio: float = 8.0, min_n: int = 50) -> pl.DataFrame:
    """Filler words injected into S2/S3 names (label-free, per country).
    S1 is the clean reference, so a token that is >= min_ratio times more frequent
    in S2/S3 than in S1 (after normalising for table sizes) is injected noise:
    'services', 'midtown', 'participations', 'holding', 'developpement', ..."""
    p = work_path(f"{split}_filler.parquet")
    if p.exists():
        return pl.read_parquet(p)
    a = _visible_s1(split, pl.read_parquet(work_path(f"{split}_ns1.parquet"), columns=["idx", "country", "name_n"]))
    b = pl.read_parquet(work_path(f"{split}_ns23.parquet"), columns=["country", "name_n"])
    outs = []
    for c in a["country"].unique().to_list():
        A, Bc = a.filter(pl.col("country") == c), b.filter(pl.col("country") == c)
        tok = lambda d, n: d.select(pl.col("name_n").str.split(" ").list.unique()).explode("name_n", empty_as_null=True) \
            .drop_nulls().group_by("name_n").len(n)
        f = tok(Bc, "n2").join(tok(A, "n1"), on="name_n", how="left").fill_null(0)
        f = f.with_columns(((pl.col("n2") + 1) / (pl.col("n1") + 1) * (A.height / max(Bc.height, 1))).alias("ratio"))
        f = f.filter((pl.col("ratio") >= min_ratio) & (pl.col("n2") >= min_n) & (pl.col("name_n").str.len_chars() >= 2))
        outs.append(f.select(pl.lit(c).alias("country"), pl.col("name_n").alias("tok")))
    out = pl.concat(outs)
    out.write_parquet(p)
    print("filler tokens per country", out.group_by("country").len().rows(), flush=True)
    return out


def load_records(split: str, which: str) -> pl.DataFrame:
    cols = [c for c in REC_COLS if not (which == "s1" and c == "src")]
    df = pl.read_parquet(work_path(f"{split}_n{which}.parquet"), columns=cols)
    seg = pl.read_parquet(work_path(f"{split}_seg_{which}.parquet"))
    df = df.join(seg, on="idx", how="left").with_columns(pl.col("core_seg").list.join(" ").alias("seg_s"))
    gp = work_path(f"{split}_gcnt_{which}.parquet")
    if not gp.exists():
        group_counts(split)
    df = df.join(pl.read_parquet(gp), on="idx", how="left")
    # name tokens without filler words (core_k) and number of filler words
    fill = filler_tokens(split).with_columns(pl.lit(1, dtype=pl.Int8).alias("isf"))
    ex = df.select("idx", "country", "core_seg").explode("core_seg", empty_as_null=True)
    ex = ex.join(fill, left_on=["country", "core_seg"], right_on=["country", "tok"], how="left")
    kk = ex.group_by("idx").agg(pl.col("core_seg").filter(pl.col("isf").is_null() & pl.col("core_seg").is_not_null()).alias("core_k"),
                                pl.col("isf").sum().cast(pl.Int8).alias("nfill"))
    df = df.join(kk, on="idx", how="left").with_columns(pl.col("core_k").list.join(" ").fill_null("").alias("k_s"))
    if which == "s23":
        n = df.height
        tok = df.select(pl.col("core_seg").list.unique()).explode("core_seg", empty_as_null=True).drop_nulls()
        idf = tok.group_by("core_seg").len().select(pl.col("core_seg").alias("tok"),
                                                   (np.log(n) - pl.col("len").log()).cast(pl.Float32).alias("idf"))
        _IDF["t"] = idf
        _IDF["max"] = float(np.log(n))
    return df


def _cp(a, b, scorer, **kw):
    return cpdist(a, b, scorer=scorer, workers=-1, **kw)


def _jacc(a: str, b: str) -> pl.Expr:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    uni = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(uni > 0).then(inter / uni).otherwise(None)


def _num_rel() -> pl.Expr:
    """House-number relation: 3 exact shared, 2 one number is prefix/suffix of the other,
    1 both have numbers but conflict, 0 missing on a side."""
    inter = pl.col("nums1").list.set_intersection(pl.col("nums2")).list.len()
    both = (pl.col("nums1").list.len() > 0) & (pl.col("nums2").list.len() > 0)
    n1 = pl.col("nums1").list.first().fill_null("")
    n2 = pl.col("nums2").list.first().fill_null("")
    near = (
        (n1.str.len_chars() > 0) & (n2.str.len_chars() > 0)
        & (n1.str.starts_with(n2) | n2.str.starts_with(n1) | n1.str.ends_with(n2) | n2.str.ends_with(n1))
    )
    return pl.when(inter > 0).then(3).when(both & near).then(2).when(both).then(1).otherwise(0).cast(pl.Int8)


def s23_context(cand: pl.DataFrame) -> pl.DataFrame:
    """Reciprocal context over the FULL stage-1 table (must not be computed per chunk):
    how this S1 ranks among all S1s proposing the same S2/S3 record."""
    return cand.with_columns(
        pl.col("bscore").rank("ordinal", descending=True).over("s23").cast(pl.Int16).alias("b_rank23"),
        (pl.col("bscore").max().over("s23") - pl.col("bscore")).alias("b_gap23"),
        pl.len().over("s23").cast(pl.Int16).alias("n_cand23"),
    )


def _diff_mass(df: pl.DataFrame, a: str = "core_seg1", b: str = "core_seg2", pre: str = "") -> pl.DataFrame:
    """IDF mass of name tokens present on only one side (S1-only, S2/S3-only)."""
    d = df.select(
        pl.int_range(pl.len()).alias("_rid"),
        pl.col(a).list.set_difference(pl.col(b)).alias(pre + "o1"),
        pl.col(b).list.set_difference(pl.col(a)).alias(pre + "o2"),
    )
    idf = _IDF["t"]
    out = d.select("_rid")
    for c in (pre + "o1", pre + "o2"):
        e = d.select("_rid", c).explode(c, empty_as_null=True).drop_nulls(c).join(idf, left_on=c, right_on="tok", how="left")
        e = e.with_columns(pl.col("idf").fill_null(_IDF["max"])).group_by("_rid").agg(
            pl.col("idf").sum().alias(f"{c}_idf"), pl.col("idf").max().alias(f"{c}_idfmax"))
        out = out.join(e, on="_rid", how="left")
    return out.fill_null(0).sort("_rid").drop("_rid")


def build(pairs: pl.DataFrame, r1: pl.DataFrame, r23: pl.DataFrame, full: bool = True) -> pl.DataFrame:
    """pairs: s1, s23, w0..w8, tot0..tot8, nkeys, bscore (from blocking.generate)."""
    df = pairs.join(r1.rename({c: c + "1" for c in r1.columns if c != "idx"}).rename({"idx": "s1"}), on="s1", how="left")
    df = df.join(r23.rename({c: c + "2" for c in r23.columns if c not in ("idx", "src")}).rename({"idx": "s23"}), on="s23", how="left")

    a_n, b_n = df["name_n1"].to_list(), df["name_n2"].to_list()
    a_c, b_c = df["core_s1"].to_list(), df["core_s2"].to_list()
    a_s, b_s = df["seg_s1"].to_list(), df["seg_s2"].to_list()
    a_x, b_x = df["name_ns1"].to_list(), df["name_ns2"].to_list()
    a_a, b_a = df["addr_n1"].to_list(), df["addr_n2"].to_list()
    u8 = dict(dtype=np.uint8)
    feats = {
        "n_ratio": _cp(a_n, b_n, fuzz.ratio, **u8),
        "n_tsort": _cp(a_n, b_n, fuzz.token_sort_ratio, **u8),
        "n_tset": _cp(a_n, b_n, fuzz.token_set_ratio, **u8),
        "c_ratio": _cp(a_c, b_c, fuzz.ratio, **u8),
        "c_tset": _cp(a_c, b_c, fuzz.token_set_ratio, **u8),
        "s_tsort": _cp(a_s, b_s, fuzz.token_sort_ratio, **u8),
        "s_tset": _cp(a_s, b_s, fuzz.token_set_ratio, **u8),
        "x_jw": _cp(a_x, b_x, JaroWinkler.normalized_similarity, dtype=np.float32),
        "a_tset": _cp(a_a, b_a, fuzz.token_set_ratio, **u8),
        "a_ratio": _cp(a_a, b_a, fuzz.ratio, **u8),
    }
    if full:
        feats.update({
            "n_partial": _cp(a_n, b_n, fuzz.partial_ratio, **u8),
            "c_partial": _cp(a_c, b_c, fuzz.partial_ratio, **u8),
            "s_partial": _cp(a_s, b_s, fuzz.partial_ratio, **u8),
            "c_lev": _cp(a_c, b_c, Levenshtein.distance, dtype=np.int32),
            "c_jw": _cp(a_c, b_c, JaroWinkler.normalized_similarity, dtype=np.float32),
            "x_partial": _cp(a_x, b_x, fuzz.partial_ratio, **u8),
            "a_tsort": _cp(a_a, b_a, fuzz.token_sort_ratio, **u8),
            "a_partial": _cp(a_a, b_a, fuzz.partial_ratio, **u8),
        })
    df = df.with_columns([pl.Series(k, v) for k, v in feats.items()])
    df = pl.concat([df, _diff_mass(df), _diff_mass(df, "core_k1", "core_k2", "k")], how="horizontal")
    a_k, b_k = df["k_s1"].to_list(), df["k_s2"].to_list()
    df = df.with_columns(
        pl.Series("k_tset", _cp(a_k, b_k, fuzz.token_set_ratio, **u8)),
        pl.Series("k_tsort", _cp(a_k, b_k, fuzz.token_sort_ratio, **u8)),
        pl.Series("k_ratio", _cp(a_k, b_k, fuzz.ratio, **u8)),
    )
    l1 = pl.col("name_n1").str.split(" ").list.eval(pl.element().filter(pl.element().is_in(LEGAL_LIST))).list.unique()
    l2 = pl.col("name_n2").str.split(" ").list.eval(pl.element().filter(pl.element().is_in(LEGAL_LIST))).list.unique()
    df = df.with_columns(l1.alias("_l1"), l2.alias("_l2"))
    nn1 = pl.col("nums1").list.first().str.slice(0, 9).cast(pl.Int64, strict=False)
    nn2 = pl.col("nums2").list.first().str.slice(0, 9).cast(pl.Int64, strict=False)
    df = df.with_columns(
        # legal form relation: 0 one side has none, 1 conflict, 2 share at least one
        pl.when((pl.col("_l1").list.len() == 0) | (pl.col("_l2").list.len() == 0)).then(0)
        .when(pl.col("_l1").list.set_intersection(pl.col("_l2")).list.len() > 0).then(2).otherwise(1).cast(pl.Int8).alias("legal_rel"),
        pl.col("_l1").list.len().cast(pl.Int8).alias("nlegal_a"),
        pl.col("_l2").list.len().cast(pl.Int8).alias("nlegal_b"),
        pl.col("core_seg1").list.set_difference(pl.col("core_seg2")).list.len().cast(pl.Int8).alias("only1_n"),
        pl.col("core_seg2").list.set_difference(pl.col("core_seg1")).list.len().cast(pl.Int8).alias("only2_n"),
        (nn1 - nn2).abs().clip(upper_bound=100000).cast(pl.Float32).alias("num_gap"),
        ((nn1 - nn2).abs().cast(pl.Float32) / pl.max_horizontal(nn1, nn2).cast(pl.Float32).clip(lower_bound=1)).alias("num_gap_rel"),
    ).drop("_l1", "_l2")
    df = df.with_columns(((pl.col("only1_n") > 0) & (pl.col("only2_n") > 0)).cast(pl.Int8).alias("swap"))
    # same, with filler words removed: a real word swap (decoy) vs an injected filler (noise)
    df = df.with_columns(
        _jacc("core_k1", "core_k2").alias("k_j"),
        pl.col("core_k1").list.set_difference(pl.col("core_k2")).list.len().cast(pl.Int8).alias("konly1_n"),
        pl.col("core_k2").list.set_difference(pl.col("core_k1")).list.len().cast(pl.Int8).alias("konly2_n"),
        pl.col("nfill2").alias("fill_c"), pl.col("nfill1").alias("fill_s"),
    )
    df = df.with_columns(((pl.col("konly1_n") > 0) & (pl.col("konly2_n") > 0)).cast(pl.Int8).alias("kswap"))
    # ambiguity / density of the candidate record and of the S1 record
    df = df.with_columns(
        pl.col("cntNs12").alias("amb_name_s1_c"), pl.col("cntNs232").alias("amb_name_s23_c"),
        pl.col("cntAs12").alias("den_addr_s1_c"), pl.col("cntAs232").alias("den_addr_s23_c"),
        pl.col("cntNs11").alias("amb_name_s1_s"), pl.col("cntNs231").alias("amb_name_s23_s"),
        pl.col("cntAs11").alias("den_addr_s1_s"), pl.col("cntAs231").alias("den_addr_s23_s"),
        pl.col("unseen2").alias("unseen_c"),
    )

    ex = [
        _jacc("atoks1", "atoks2").alias("atok_j"),
        _jacc("ids1", "ids2").alias("ids_j"),
        pl.col("ids1").list.set_intersection(pl.col("ids2")).list.len().cast(pl.Int8).alias("ids_n"),
        _jacc("core_seg1", "core_seg2").alias("seg_j"),
        _num_rel().alias("num_rel"),
        pl.when((pl.col("state1") == "") | (pl.col("state2") == "")).then(0)
        .when(pl.col("state1") == pl.col("state2")).then(2).otherwise(1).cast(pl.Int8).alias("state_rel"),
        (pl.col("name_ns1").str.contains(pl.col("name_ns2"), literal=True) | pl.col("name_ns2").str.contains(pl.col("name_ns1"), literal=True)).alias("x_contain"),
        (pl.col("name_ns1") == pl.col("name_ns2")).alias("x_eq"),
        (pl.col("name_n1") == pl.col("name_n2")).alias("n_eq"),
        (pl.col("addr_n1") == pl.col("addr_n2")).alias("a_eq"),
        (pl.col("addr_n2") == "").alias("a2_empty"),
        (pl.col("addr_n1") == "").alias("a1_empty"),
        pl.col("is_dom2").alias("dom2"),
        pl.col("indic2").alias("indic2"),
        pl.col("src").cast(pl.Int8).alias("src"),
        pl.col("core_s1").str.len_chars().alias("len_c1"),
        pl.col("core_s2").str.len_chars().alias("len_c2"),
        pl.col("core_seg1").list.len().alias("ntok1"),
        pl.col("core_seg2").list.len().alias("ntok2"),
        pl.col("nums2").list.len().alias("nnum2"),
        pl.col("atoks2").list.len().alias("natok2"),
    ]
    for f in range(len(FAMS)):
        ex.append((pl.col(f"w{f}") / pl.col(f"tot{f}").clip(lower_bound=1e-3)).clip(upper_bound=1.5).cast(pl.Float32).alias(f"r{f}"))
    df = df.with_columns(ex)

    # context within the S1's candidate list
    df = df.with_columns(
        pl.col("bscore").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("b_rank"),
        (pl.col("bscore").max().over("s1") - pl.col("bscore")).alias("b_gap"),
        pl.len().over("s1").cast(pl.Int16).alias("n_cand"),
        (pl.col("n_tset").cast(pl.Int16).max().over("s1") - pl.col("n_tset").cast(pl.Int16)).alias("n_tset_gap"),
        (pl.col("a_tset").cast(pl.Int16).max().over("s1") - pl.col("a_tset").cast(pl.Int16)).alias("a_tset_gap"),
    )
    drop = [c for c in df.columns if c.endswith(("1", "2")) and c not in ("s1", "dom2", "indic2", "a2_empty", "a1_empty", "len_c1", "len_c2", "ntok1", "ntok2", "nnum2", "natok2")]
    drop = [c for c in drop if not c.startswith(("w", "tot", "r")) or c in ("name_n1",)]
    return df.drop([c for c in drop if c in df.columns])


def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("s1", "s23", "label", "p_prune", "p")]


def compact(df: pl.DataFrame) -> pl.DataFrame:
    """Float64 -> Float32 and bool -> Int8 to halve memory."""
    ex = []
    for c, t in df.schema.items():
        if t == pl.Float64:
            ex.append(pl.col(c).cast(pl.Float32))
        elif t == pl.Boolean:
            ex.append(pl.col(c).cast(pl.Int8))
    return df.with_columns(ex) if ex else df


def cluster_features(d: pl.DataFrame, r23: pl.DataFrame, strong: float = 0.5, pcol: str = "p_prune",
                     pre: str = "cl_") -> pl.DataFrame:
    """Support from the S1's other confident candidates: a record that looks like
    (same address / same name as) records already confidently matched to this S1
    is likely the same business (rebrands, empty-address duplicates).
    d: pruned pairs with p_prune. Adds cl_amax, cl_nmax, cl_nstrong."""
    st = d.filter(pl.col(pcol) >= strong).select("s1", pl.col("s23").alias("j"))
    pr = d.select("s1", "s23").join(st, on="s1").filter(pl.col("s23") != pl.col("j"))
    rr = r23.select("idx", "name_n", "addr_n")
    pr = pr.join(rr.rename({"idx": "s23", "name_n": "na", "addr_n": "aa"}), on="s23", how="left") \
           .join(rr.rename({"idx": "j", "name_n": "nb", "addr_n": "ab"}), on="j", how="left")
    a_sim = _cp(pr["aa"].to_list(), pr["ab"].to_list(), fuzz.token_set_ratio, dtype=np.uint8)
    n_sim = _cp(pr["na"].to_list(), pr["nb"].to_list(), fuzz.token_set_ratio, dtype=np.uint8)
    both_addr = ((pr["aa"] != "") & (pr["ab"] != "")).to_numpy()
    a_sim = np.where(both_addr, a_sim, 0).astype(np.int16)
    pr = pr.select("s1", "s23").with_columns(pl.Series("a", a_sim), pl.Series("n", n_sim.astype(np.int16)))
    agg = pr.group_by("s1", "s23").agg(pl.col("a").max().alias(pre + "amax"), pl.col("n").max().alias(pre + "nmax"),
                                       pl.len().cast(pl.Int16).alias(pre + "nstrong"))
    return d.join(agg, on=["s1", "s23"], how="left").with_columns(
        pl.col(pre + "amax").fill_null(-1), pl.col(pre + "nmax").fill_null(-1), pl.col(pre + "nstrong").fill_null(0))
