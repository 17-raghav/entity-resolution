"""v11 matcher: retrain on v10's pruned train pairs with a test-like orphan mix.

Finding (2026-09-27): test orphans are clean decoy-type records (noise rates of test
S2/S3 imply ~58% matched records, all extra orphans decoy-like). Hiding 19% of train S1
instead produced NOISY orphans (the hidden entities' own records), which never occur in
test and taught the model to distrust noisy true matches.

Fix: drop candidate pairs whose S2/S3 record belongs to a hidden S1, and give decoy
negatives (records matched to no S1 at all) weight W to mimic test's ~2x decoy density.

usage: python v11_matcher.py <W_decoy> <tag>   -> ~/ml/work11_<tag>/matcher.txt etc.
"""
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

# Paths: defaults are the AWS layout used in the original run; override with env vars.
W10 = os.path.expanduser(os.environ.get("BER_V10_WORK", "~/ml/work10"))
WD = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
TAG = sys.argv[2] if len(sys.argv) > 2 else "w2"
OUT = os.path.expanduser(os.path.join(os.environ.get("BER_V11_ROOT", "~/ml"), f"work11_{TAG}"))
os.makedirs(OUT, exist_ok=True)
os.environ["BER_WORK"] = W10  # read-only use of v10 artifacts (records, full-name counts)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import features as F  # noqa: E402
import postprocess as P  # noqa: E402
import train as T  # noqa: E402
from evaluate import macro_f05  # noqa: E402

t0 = time.time()
gt = pl.read_parquet(f"{W10}/train_gt.parquet")
s1all = pl.read_parquet(f"{W10}/train_s1.parquet", columns=["idx"])
hidden = s1all.filter(pl.col("idx").hash(seed=21) % 100 < T.DROP_PCT).rename({"idx": "s1"})
hid23 = gt.join(hidden, on="s1", how="semi").select("s23").unique()
matched23 = gt.select("s23").unique()
print(f"hidden S1 {hidden.height} ({hidden.height / s1all.height:.3f}), their records {hid23.height}", flush=True)

scored = pl.read_parquet(f"{W10}/train_scored.parquet")
print("hidden S1 present in scored (must be 0):", scored.select("s1").unique().join(hidden, on="s1", how="semi").height, flush=True)
d = T.prune(scored, 10, 0.002)
del scored
n0 = d.height
d = d.join(hid23, on="s23", how="anti")
print(f"pruned pairs {n0} -> {d.height} after removing hidden-entity records ({time.time() - t0:.0f}s)", flush=True)
d = T.context_features(d, "train")
r23 = F.load_records("train", "s23").select("idx", "name_n", "addr_n")
d = F.cluster_features(d, r23)
del r23
d = d.with_columns((~pl.col("s23").is_in(matched23["s23"].implode())).alias("decoy"))
d = d.with_columns(pl.when((pl.col("label") == 0) & pl.col("decoy")).then(WD).otherwise(1.0).alias("w"))
cols = json.loads(open(f"{W10}/matcher_cols.json").read())
tr, va = d.filter(~pl.col("is_val")), d.filter(pl.col("is_val"))
del d
print("train pairs", tr.height, "val pairs", va.height, "decoy negatives share (train)",
      round(float(tr.filter(pl.col("label") == 0)["decoy"].mean()), 3), f"({time.time() - t0:.0f}s)", flush=True)

params = dict(T.MATCHER_PARAMS, learning_rate=0.05)
es = va.sample(min(1_000_000, va.height), seed=T.SEED)
dtr = lgb.Dataset(tr.select(cols).to_numpy().astype(np.float32), tr["label"].to_numpy(), weight=tr["w"].to_numpy(),
                  feature_name=cols, free_raw_data=True)
dva = lgb.Dataset(es.select(cols).to_numpy().astype(np.float32), es["label"].to_numpy(), weight=es["w"].to_numpy(), reference=dtr)
del tr
bst = lgb.train(params, dtr, 4000, valid_sets=[dva], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
bst.save_model(f"{OUT}/matcher.txt")
json.dump(cols, open(f"{OUT}/matcher_cols.json", "w"))
print(f"trained ({time.time() - t0:.0f}s)", flush=True)

# ---- compare with v10 matcher on the same clean-orphan validation pairs
old = lgb.Booster(model_file=f"{W10}/matcher.txt")
X = va.select(cols).to_numpy().astype(np.float32)
va = va.with_columns(pl.Series("p", bst.predict(X, num_threads=T.NTHREADS)), pl.Series("p10", old.predict(X, num_threads=T.NTHREADS)))
del X
va.select("s1", "s23", "p", "p10", "p_prune", "label", "decoy").write_parquet(f"{OUT}/val_pred.parquet")
s1_ids = va["s1"].unique()
res = {}
for pc in ("p10", "p"):
    d1 = P.one_to_one(va, pc)
    for es_ in (1.0, 1.5, 2.0, 3.0):
        res[f"{pc}_e{es_}"] = macro_f05(P.select_sets(d1, pc, 0.0, es_).select("s1", "s23"), gt, s1_ids)
for k, v in sorted(res.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.5f}")
best = max((k for k in res if k.startswith("p_")), key=res.get)
json.dump({"selector": "ef", "miss": 0.0, "empty_scale": float(best.split("_e")[1]), "one_to_one": True},
          open(f"{OUT}/select_params.json", "w"))
print("saved", OUT, open(f"{OUT}/select_params.json").read(), f"total {time.time() - t0:.0f}s", flush=True)
