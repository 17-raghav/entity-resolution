"""Score a matching_results.tsv against a ground-truth TSV with the official metric
(macro F0.5 per S1, see src/evaluate.py) and report per-country scores.

usage: python scripts/score.py --pred <out>/matching_results.tsv --gt data/sample/test_ground_truth.tsv \
           --s1 data/sample/test/test_source1.tsv
"""
import argparse

import polars as pl


def pairs(path: str, col: str) -> pl.DataFrame:
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).with_columns(pl.col(col).fill_null(""))
    return (d.with_columns(pl.col(col).str.split(",")).explode(col, empty_as_null=True).filter(pl.col(col) != "")
            .select(pl.col("source1_entity_id").alias("s1"), pl.col(col).alias("s23")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--s1", required=True, help="source1 TSV of the scored split (defines the S1 set)")
    a = ap.parse_args()
    s1 = pl.read_csv(a.s1, separator="\t", quote_char=None, infer_schema=False).select(
        pl.col("entity_id").alias("s1"), "country")
    pred = pairs(a.pred, "matched_entity_ids")
    gt = pairs(a.gt, "matched_entity_ids")
    n_true = gt.group_by("s1").len("n_true")
    n_pred = pred.group_by("s1").len("n_pred")
    tp = pred.join(gt, on=["s1", "s23"]).group_by("s1").len("tp")
    d = s1.join(n_true, on="s1", how="left").join(n_pred, on="s1", how="left").join(tp, on="s1", how="left").fill_null(0)
    d = d.with_columns(
        pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("n_true") + pl.col("n_pred")).clip(lower_bound=1e-9)).alias("f05"))
    print(f"S1 {d.height}  macro F0.5 {d['f05'].mean():.4f}")
    print(f"pair precision {d['tp'].sum() / max(d['n_pred'].sum(), 1):.4f}  "
          f"pair recall {d['tp'].sum() / max(d['n_true'].sum(), 1):.4f}")
    print("per country (country, S1, F0.5):",
          d.group_by("country").agg(pl.len(), pl.col("f05").mean().round(4)).sort("country").rows())


if __name__ == "__main__":
    main()
