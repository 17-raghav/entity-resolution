"""Build a small dataset in the official layout from the TRAIN files, for a smoke test.

The full pipeline needs ~64 GB RAM and hours. This script cuts a sample that runs
end to end in minutes on a laptop:

  <out>/train/train_source{1,2,3}.tsv, train_ground_truth.tsv   (n_train S1)
  <out>/test/test_source{1,2,3}.tsv                              (n_test S1)
  <out>/test_ground_truth.tsv                                    (labels for the sampled "test" S1)

Both splits come from the official TRAIN data, so the "test" split has labels and
can be scored. Every sampled S1 brings all its matched S2/S3 records. Unmatched
(orphan) S2/S3 records are added at the same ratio as in the full train data, so the
sample keeps decoys, but at a much lower blocking density than the real data.
Scores on this sample are a smoke test only; they are not comparable to the
leaderboard or to the full validation numbers.

usage: python scripts/make_sample.py --raw <student_resource>/dataset --out data/sample \
           [--n-train 20000] [--n-test 5000] [--seed 0]
"""
import argparse
from pathlib import Path

import polars as pl


def read(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, encoding="utf8")


def write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t", quote_style="never")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="official dataset dir containing train/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=20_000)
    ap.add_argument("--n-test", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raw, out = Path(a.raw) / "train", Path(a.out)

    s1 = read(raw / "train_source1.tsv")
    s2 = read(raw / "train_source2.tsv")
    s3 = read(raw / "train_source3.tsv")
    gt = read(raw / "train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").fill_null(""))
    print("full train: S1", s1.height, "S2", s2.height, "S3", s3.height)

    pick = s1.sample(a.n_train + a.n_test, seed=a.seed, shuffle=True)
    ids = {"train": pick.head(a.n_train), "test": pick.tail(a.n_test)}

    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids", empty_as_null=True).filter(pl.col("matched_entity_ids") != "")
             .rename({"source1_entity_id": "s1", "matched_entity_ids": "s23"}))
    all_matched = pairs.select("s23").unique()
    s23_all = pl.concat([s2.with_columns(pl.lit(2).alias("_src")), s3.with_columns(pl.lit(3).alias("_src"))])
    orphans = s23_all.join(all_matched, left_on="entity_id", right_on="s23", how="anti")
    orphan_rate = orphans.height / s1.height  # orphan S2/S3 records per S1 in the full data
    print(f"orphan S2/S3 records per S1 in full train: {orphan_rate:.3f}")
    orphans = orphans.sample(fraction=1.0, seed=a.seed, shuffle=True)

    used = 0
    for split, s1_split in ids.items():
        p = pairs.join(s1_split.select(pl.col("entity_id").alias("s1")), on="s1", how="semi")
        recs = s23_all.join(p.select(pl.col("s23").alias("entity_id")), on="entity_id", how="semi")
        n_orph = int(round(orphan_rate * s1_split.height))
        orph = orphans.slice(used, n_orph)
        used += n_orph
        recs = pl.concat([recs, orph]).sample(fraction=1.0, seed=a.seed, shuffle=True)
        write(s1_split, out / split / f"{split}_source1.tsv")
        write(recs.filter(pl.col("_src") == 2).drop("_src"), out / split / f"{split}_source2.tsv")
        write(recs.filter(pl.col("_src") == 3).drop("_src"), out / split / f"{split}_source3.tsv")
        g = gt.join(s1_split.select(pl.col("entity_id").alias("source1_entity_id")), on="source1_entity_id", how="semi")
        write(g, out / split / f"{split}_ground_truth.tsv" if split == "train" else out / "test_ground_truth.tsv")
        print(split, "S1", s1_split.height, "S2+S3", recs.height, "of which orphans", orph.height,
              "countries", s1_split["country"].value_counts().sort("country").rows())


if __name__ == "__main__":
    main()
