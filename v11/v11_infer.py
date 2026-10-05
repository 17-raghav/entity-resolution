"""v11 test inference: same pruned test pairs/features as v10 (built once, cached to
~/ml/work10/test_d.parquet), then any matcher variant predicts in minutes.

usage: python v11_infer.py <tag> [<tag> ...]   (matchers in ~/ml/work11_<tag>/)
       -> ~/ml/output11_<tag>/{matching_results,candidate_pairs}.tsv
"""
import json
import os
import shutil
import sys
import time

import lightgbm as lgb
import polars as pl

# Paths: defaults are the AWS layout used in the original run; override with env vars.
W10 = os.path.expanduser(os.environ.get("BER_V10_WORK", "~/ml/work10"))
ROOT = os.path.expanduser(os.environ.get("BER_V11_ROOT", "~/ml"))
V10_OUT = os.path.expanduser(os.environ.get("BER_V10_OUT", "~/ml/output10"))
os.environ["BER_WORK"] = W10
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import features as F  # noqa: E402
import postprocess as P  # noqa: E402
from config import INFER_CHUNK  # noqa: E402
from io_utils import write_from_pairs  # noqa: E402
from train import context_features, predict, prune  # noqa: E402

t0 = time.time()
CACHE = f"{W10}/test_d.parquet"
if not os.path.exists(CACHE):
    prm = json.load(open(f"{W10}/prune_params.json"))
    pruner = lgb.Booster(model_file=f"{W10}/pruner.txt")
    pcols = json.load(open(f"{W10}/pruner_cols.json"))
    cpath = f"{W10}/test_cand1.parquet"
    ctx = F.s23_context(pl.read_parquet(cpath, columns=["s1", "s23", "bscore"])).drop("bscore")
    ids = ctx["s1"].unique().sort()
    r1 = F.load_records("test", "s1")
    r23 = F.load_records("test", "s23")
    kept = []
    for i in range(0, len(ids), INFER_CHUNK):
        lo, hi = ids[i], ids[min(i + INFER_CHUNK, len(ids)) - 1]
        sub = pl.scan_parquet(cpath).filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi)).collect()
        sub = sub.join(ctx, on=["s1", "s23"], how="left")
        f = F.build(sub, r1.join(sub.select(pl.col("s1").alias("idx")).unique(), on="idx", how="semi"),
                    r23.join(sub.select(pl.col("s23").alias("idx")).unique(), on="idx", how="semi"))
        f = F.compact(f)
        f = f.with_columns(pl.Series("p_prune", predict(pruner, f, pcols)))
        kept.append(prune(f, prm["k"], prm["tau"]))
        print("chunk", i, f.height, "kept", kept[-1].height, f"{time.time() - t0:.0f}s", flush=True)
        del f, sub
    del r1, ctx
    d = pl.concat(kept)
    del kept
    d = F.cluster_features(d, r23)
    del r23
    d = context_features(d, "test")
    d.write_parquet(CACHE)
    print("cached", CACHE, d.shape, f"{time.time() - t0:.0f}s", flush=True)
else:
    d = pl.read_parquet(CACHE)
    print("loaded cache", d.shape, flush=True)

c = pl.read_parquet(f"{W10}/test_s1.parquet", columns=["idx", "country"]).rename({"idx": "s1"})
n1 = c.height
for tag in sys.argv[1:]:
    W = os.path.join(ROOT, f"work11_{tag}")
    out = os.path.join(ROOT, f"output11_{tag}")
    os.makedirs(out, exist_ok=True)
    m = lgb.Booster(model_file=f"{W}/matcher.txt")
    cols = json.load(open(f"{W}/matcher_cols.json"))
    prm = json.load(open(f"{W}/select_params.json"))
    x = d.select("s1", "s23", "p_prune").with_columns(pl.Series("p", predict(m, d, cols)))
    x.write_parquet(f"{W}/test_pred.parquet")
    y = P.one_to_one(x, "p") if prm.get("one_to_one", True) else x
    sel = P.select_sets(y, "p", prm["miss"], prm["empty_scale"])
    write_from_pairs(f"{out}/matching_results.tsv", "test", sel.select("s1", "s23"), "matched_entity_ids")
    shutil.copy(os.path.join(V10_OUT, "candidate_pairs.tsv"), f"{out}/candidate_pairs.tsv")
    st = sel.join(c, on="s1").group_by("country").len("m").join(c.group_by("country").len("n"), on="country")
    print(tag, prm, "matches/S1 all", round(sel.height / n1, 3),
          st.with_columns((pl.col("m") / pl.col("n")).round(3).alias("per")).select("country", "per").rows(),
          f"{time.time() - t0:.0f}s", flush=True)
