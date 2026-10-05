"""Test inference: stage-1 candidates -> pruner -> candidate_pairs.tsv -> matcher
-> one-to-one + set selection -> matching_results.tsv.
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

import features as F
import postprocess as P
from config import INFER_CHUNK, STACK, work_path
from io_utils import write_from_pairs
from train import context_features, predict, prune, stack_features


def load_params() -> dict:
    p = {"k": 8, "tau": 0.01, "selector": "ef", "miss": 0.1, "empty_scale": 1.0, "t": 0.5, "one_to_one": True}
    for name in ("prune_params.json", "select_params.json"):
        f = work_path(name)
        if f.exists():
            p.update(json.loads(f.read_text()))
    return p


def run(split: str, out_dir: Path, chunk_s1: int = INFER_CHUNK) -> None:
    prm = load_params()
    print("params", prm)
    pruner = lgb.Booster(model_file=str(work_path("pruner.txt")))
    matcher = lgb.Booster(model_file=str(work_path("matcher.txt")))
    pcols = json.loads(work_path("pruner_cols.json").read_text())
    mcols = json.loads(work_path("matcher_cols.json").read_text())

    cpath = work_path(f"{split}_cand1.parquet")
    ctx = F.s23_context(pl.read_parquet(cpath, columns=["s1", "s23", "bscore"])).drop("bscore")
    ids = ctx["s1"].unique().sort()
    r1 = F.load_records(split, "s1")
    r23 = F.load_records(split, "s23")
    kept = []
    for i in range(0, len(ids), chunk_s1):
        lo, hi = ids[i], ids[min(i + chunk_s1, len(ids)) - 1]
        sub = pl.scan_parquet(cpath).filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi)).collect()
        sub = sub.join(ctx, on=["s1", "s23"], how="left")
        f = F.build(sub, r1.join(sub.select(pl.col("s1").alias("idx")).unique(), on="idx", how="semi"),
                    r23.join(sub.select(pl.col("s23").alias("idx")).unique(), on="idx", how="semi"))
        f = F.compact(f)
        f = f.with_columns(pl.Series("p_prune", predict(pruner, f, pcols)))
        kept.append(prune(f, prm["k"], prm["tau"]))
        print("chunk", i, f.height, "kept", kept[-1].height, flush=True)
        del f, sub
    del r1, ctx
    d = pl.concat(kept)
    d = F.cluster_features(d, r23)
    d.select("s1", "s23", "p_prune").write_parquet(work_path(f"{split}_pruned.parquet"))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_from_pairs(out_dir / "candidate_pairs.tsv", split, d.select("s1", "s23"), "candidate_entity_ids")

    d = context_features(d, split)
    if STACK and work_path("matcher2.txt").exists():
        d = d.with_columns(pl.Series("p1", predict(matcher, d, mcols)))
        d = stack_features(d, r23)
        m2 = lgb.Booster(model_file=str(work_path("matcher2.txt")))
        c2 = json.loads(work_path("matcher2_cols.json").read_text())
        d = d.with_columns(pl.Series("p", predict(m2, d, c2)))
    else:
        d = d.with_columns(pl.Series("p", predict(matcher, d, mcols)))
    del r23
    d.select("s1", "s23", "p_prune", "p").write_parquet(work_path(f"{split}_pred.parquet"))
    sel = select(d, prm)
    write_from_pairs(out_dir / "matching_results.tsv", split, sel.select("s1", "s23"), "matched_entity_ids")
    n1 = pl.read_parquet(work_path(f"{split}_s1.parquet"), columns=["idx"]).height
    print(f"S1 {n1}  candidates/S1 {d.height / n1:.2f}  matches/S1 {sel.height / n1:.2f}  "
          f"empty frac {1 - sel['s1'].n_unique() / n1:.4f}")


def select(d: pl.DataFrame, prm: dict) -> pl.DataFrame:
    if prm.get("one_to_one", True):
        d = P.one_to_one(d, "p")
    if prm["selector"] == "ef":
        return P.select_sets(d, "p", prm["miss"], prm["empty_scale"])
    return P.threshold_sets(d, "p", prm["t"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.split, Path(a.out))
