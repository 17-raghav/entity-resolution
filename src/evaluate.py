"""Official macro F0.5 scorer, blocking report and validation split.

Per S1 entity: F0.5 = 1.25*tp / (0.25*n_true + n_pred), which equals
1.25*P*R / (0.25*P + R). Empty truth + empty prediction scores 1.0.
"""
import numpy as np
import polars as pl

from config import SEED, VAL_FRAC, work_path


def f05_per_s1(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> pl.DataFrame:
    """pred, gt: DataFrame(s1, s23). s1_ids: the S1 idx set being evaluated."""
    base = pl.DataFrame({"s1": s1_ids.cast(pl.Int32)})
    gt = gt.join(base, on="s1", how="semi")
    pred = pred.unique(["s1", "s23"]).join(base, on="s1", how="semi")
    n_true = gt.group_by("s1").len("n_true")
    n_pred = pred.group_by("s1").len("n_pred")
    tp = pred.join(gt, on=["s1", "s23"], how="inner").group_by("s1").len("tp")
    df = (
        base.join(n_true, on="s1", how="left")
        .join(n_pred, on="s1", how="left")
        .join(tp, on="s1", how="left")
        .fill_null(0)
    )
    f = (
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0))
        .then(1.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(lower_bound=1e-9))
    )
    return df.with_columns(f.alias("f05"))


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> float:
    return float(f05_per_s1(pred, gt, s1_ids)["f05"].mean())


def blocking_report(cand: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series, n_s23: int | None = None) -> dict:
    base = pl.DataFrame({"s1": s1_ids.cast(pl.Int32)})
    gt = gt.join(base, on="s1", how="semi")
    cand = cand.unique(["s1", "s23"]).join(base, on="s1", how="semi")
    hit = cand.join(gt, on=["s1", "s23"], how="semi").height
    sizes = base.join(cand.group_by("s1").len("n"), on="s1", how="left").fill_null(0)["n"]
    rep = {
        "pair_recall": hit / max(gt.height, 1),
        "total_pairs": cand.height,
        "mean_cands": float(sizes.mean()),
        "median_cands": float(sizes.median()),
        "p95_cands": float(sizes.quantile(0.95)),
        "empty_frac": float((sizes == 0).mean()),
    }
    if n_s23:
        rep["reduction_ratio"] = 1 - cand.height / (len(s1_ids) * n_s23)
    return rep


def make_val_split() -> pl.DataFrame:
    """Hold out VAL_FRAC of train S1 (grouped by S1 by construction)."""
    s1 = pl.read_parquet(work_path("train_s1.parquet"), columns=["idx"])
    rng = np.random.default_rng(SEED)
    is_val = rng.random(s1.height) < VAL_FRAC
    out = s1.with_columns(pl.Series("is_val", is_val))
    out.write_parquet(work_path("train_split.parquet"))
    return out


def _selftest():
    gt = pl.DataFrame({"s1": [1, 1], "s23": [47, 812]}, schema={"s1": pl.Int32, "s23": pl.Int32})
    pred = pl.DataFrame({"s1": [1, 1, 1], "s23": [47, 193, 812]}, schema={"s1": pl.Int32, "s23": pl.Int32})
    v = macro_f05(pred, gt, pl.Series([1]))
    assert abs(v - 0.714) < 1e-3, v
    # singleton predicted empty -> 1, non-singleton predicted empty -> 0
    v = macro_f05(pred.head(0), gt, pl.Series([1, 2]))
    assert abs(v - 0.5) < 1e-9, v
    print("scorer selftest ok")


if __name__ == "__main__":
    _selftest()
    sp = make_val_split()
    print("val S1:", int(sp["is_val"].sum()))
