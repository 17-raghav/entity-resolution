"""Post-processing of matcher probabilities.

1. One-to-one: in training every S2/S3 record belongs to at most one S1, so a
   record claimed by several S1 entities keeps only its highest-probability S1.
2. Per-S1 set selection maximising expected F0.5 (ratio-of-expectations):
     E[F_k] ~= 1.25 * sum_{i<=k} p_i / (0.25 * (sum_i p_i + miss) + k)
     E[F_0]  = prod_i (1 - p_i) * empty_scale        (predict "no match")
   where p are sorted descending, `miss` is the expected number of true matches
   outside the candidate list, and the best k (including 0) is kept.
"""
import polars as pl


def one_to_one(df: pl.DataFrame, pcol: str = "p") -> pl.DataFrame:
    return df.filter(pl.col(pcol) == pl.col(pcol).max().over("s23")).unique("s23", keep="first")


def select_sets(df: pl.DataFrame, pcol: str = "p", miss: float = 0.1, empty_scale: float = 1.0,
                min_p: float = 0.0) -> pl.DataFrame:
    """df: s1, s23, p. Returns selected (s1, s23, p)."""
    d = df.filter(pl.col(pcol) > min_p).sort(["s1", pcol], descending=[False, True])
    d = d.with_columns(
        pl.col(pcol).cum_sum().over("s1").alias("_cp"),
        pl.int_range(1, pl.len() + 1).over("s1").alias("_k"),
        (pl.col(pcol).sum().over("s1") + miss).alias("_tot"),
        (1 - pl.col(pcol)).clip(lower_bound=1e-9).log().sum().over("s1").exp().alias("_p0"),
    )
    d = d.with_columns((1.25 * pl.col("_cp") / (0.25 * pl.col("_tot") + pl.col("_k"))).alias("_f"))
    d = d.with_columns(pl.col("_f").max().over("s1").alias("_fmax"))
    kbest = d.filter(pl.col("_f") == pl.col("_fmax")).group_by("s1").agg(pl.col("_k").min().alias("_kb"), pl.col("_fmax").first(), pl.col("_p0").first())
    kbest = kbest.filter(pl.col("_fmax") > pl.col("_p0") * empty_scale)
    out = d.join(kbest.select("s1", "_kb"), on="s1").filter(pl.col("_k") <= pl.col("_kb"))
    return out.select("s1", "s23", pcol)


def threshold_sets(df: pl.DataFrame, pcol: str = "p", t: float = 0.5, t_top: float | None = None) -> pl.DataFrame:
    """Baseline selector: keep p >= t; optionally always keep the top-1 when p >= t_top."""
    keep = pl.col(pcol) >= t
    if t_top is not None:
        keep = keep | ((pl.col(pcol) == pl.col(pcol).max().over("s1")) & (pl.col(pcol) >= t_top))
    return df.filter(keep).select("s1", "s23", pcol)
