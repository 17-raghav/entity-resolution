"""TSV <-> parquet helpers, integer id maps and the single TSV writer.

Layout produced in WORK_DIR:
  {split}_s1.parquet   idx(int32), entity_id, name, addr, country
  {split}_s23.parquet  idx(int32), entity_id, name, addr, country, src(int8: 2|3)
  train_gt.parquet     s1(int32), s23(int32)   one row per true pair
"""
import argparse

import polars as pl

from config import raw_path, work_path

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
        encoding="utf8",
    )
    return df


def _clean(df: pl.DataFrame) -> pl.DataFrame:
    df = df.rename({"business_name": "name", "business_address": "addr"})
    return df.with_columns(
        [pl.col(c).fill_null("").str.strip_chars("\r") for c in ["entity_id", "name", "addr", "country"]]
    )


def convert(split: str) -> None:
    s1 = _clean(read_tsv(raw_path(split, "source1")))
    s1 = s1.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    s1.write_parquet(work_path(f"{split}_s1.parquet"))

    parts = []
    for k in (2, 3):
        d = _clean(read_tsv(raw_path(split, f"source{k}"))).with_columns(pl.lit(k, dtype=pl.Int8).alias("src"))
        parts.append(d)
    s23 = pl.concat(parts).with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
    s23.write_parquet(work_path(f"{split}_s23.parquet"))
    print(split, "s1", s1.height, "s23", s23.height, s1["country"].value_counts().sort("count").to_dicts())

    if split == "train":
        gt = read_tsv(raw_path("train", "ground_truth"))
        gt = gt.with_columns(pl.col("matched_entity_ids").fill_null(""))
        pairs = (
            gt.with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids", empty_as_null=True)
            .filter(pl.col("matched_entity_ids") != "")
        )
        pairs = (
            pairs.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("idx").alias("s1")), on="source1_entity_id")
            .join(s23.select(pl.col("entity_id").alias("matched_entity_ids"), pl.col("idx").alias("s23")), on="matched_entity_ids")
            .select("s1", "s23")
        )
        pairs.write_parquet(work_path("train_gt.parquet"))
        print("gt pairs", pairs.height, "unique s23", pairs["s23"].n_unique())


def load(split: str, which: str) -> pl.DataFrame:
    return pl.read_parquet(work_path(f"{split}_{which}.parquet"))


def write_id_lists(path, s1_ids: list[str], lists: list[list[str]], col: str) -> None:
    """Write submission-style TSV: UTF-8, tab, no quoting, comma-joined IDs."""
    assert len(s1_ids) == len(lists)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid, ids in zip(s1_ids, lists):
            f.write(sid)
            f.write("\t")
            f.write(",".join(ids))
            f.write("\n")


def write_from_pairs(path, split: str, pairs: pl.DataFrame, col: str) -> None:
    """pairs: DataFrame(s1:int32, s23:int32). Every S1 of the split gets one row."""
    s1 = load(split, "s1").select("idx", "entity_id")
    s23 = load(split, "s23").select(pl.col("idx").alias("s23"), pl.col("entity_id").alias("eid"))
    grouped = (
        pairs.unique(["s1", "s23"])
        .join(s23, on="s23")
        .sort(["s1", "eid"])
        .group_by("s1", maintain_order=True)
        .agg(pl.col("eid"))
    )
    full = s1.join(grouped, left_on="idx", right_on="s1", how="left").sort("idx")
    lists = [x if x is not None else [] for x in full["eid"].to_list()]
    write_id_lists(path, full["entity_id"].to_list(), lists, col)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", nargs="+", default=["train", "test"])
    a = ap.parse_args()
    for s in a.split:
        convert(s)
