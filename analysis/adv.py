"""Adversarial validation: which pair features distinguish TEST candidates from
VALIDATION candidates? Same pipeline (stage-1 -> features -> pruner) on both.
Writes ~/ml/adv/{test,val}_feat.parquet for later importance weighting."""
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

W = os.path.expanduser("~/ml/work8")
os.environ["BER_WORK"] = W
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import features as F  # noqa: E402
from train import predict, prune  # noqa: E402

OUT = os.path.expanduser("~/ml/adv")
os.makedirs(OUT, exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
pruner = lgb.Booster(model_file=f"{W}/pruner.txt")
pcols = json.loads(open(f"{W}/pruner_cols.json").read())

# ---- test sample: same path as infer.py
if not os.path.exists(f"{OUT}/test_feat.parquet"):
    ctx = F.s23_context(pl.read_parquet(f"{W}/test_cand1.parquet", columns=["s1", "s23", "bscore"])).drop("bscore")
    ids = ctx["s1"].unique().sample(N, seed=1)
    sub = pl.scan_parquet(f"{W}/test_cand1.parquet").filter(pl.col("s1").is_in(ids.implode())).collect()
    sub = sub.join(ctx, on=["s1", "s23"], how="left")
    del ctx
    r1 = F.load_records("test", "s1")
    r23 = F.load_records("test", "s23")
    f = F.build(sub, r1.join(sub.select(pl.col("s1").alias("idx")).unique(), on="idx", how="semi"),
                r23.join(sub.select(pl.col("s23").alias("idx")).unique(), on="idx", how="semi"))
    f = F.compact(f)
    f = f.with_columns(pl.Series("p_prune", predict(pruner, f, pcols)))
    f = prune(f, 10, 0.002)
    c = pl.read_parquet(f"{W}/test_s1.parquet", columns=["idx", "country"]).rename({"idx": "s1"})
    f.join(c, on="s1").write_parquet(f"{OUT}/test_feat.parquet")
    del r1, r23, f
    print("test features done", flush=True)

t = pl.read_parquet(f"{OUT}/test_feat.parquet")
# ---- validation sample: pruner-scored train pairs (same drop, same pruner stage)
sc = pl.scan_parquet(f"{W}/train_scored.parquet").filter(pl.col("is_val")).collect()
v = prune(sc, 10, 0.002)
c = pl.read_parquet(f"{W}/train_s1.parquet", columns=["idx", "country"]).rename({"idx": "s1"})
v = v.join(c, on="s1")
v.write_parquet(f"{OUT}/val_feat.parquet")

cols = [x for x in pcols if x in t.columns and x in v.columns]
print("pairs: test", t.height, "val", v.height, "features", len(cols), flush=True)


def adv(tt, vv, name):
    n = min(tt.height, vv.height, 400_000)
    a = pl.concat([tt.sample(n, seed=1).select(cols).with_columns(pl.lit(1).alias("y")),
                   vv.sample(n, seed=1).select(cols).with_columns(pl.lit(0).alias("y"))], how="vertical_relaxed")
    a = a.sample(fraction=1.0, seed=2)
    X = a.select(cols).to_numpy().astype(np.float32)
    y = a["y"].to_numpy()
    k = int(0.7 * len(y))
    dtr, dva = lgb.Dataset(X[:k], y[:k], feature_name=cols), lgb.Dataset(X[k:], y[k:])
    b = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=63, verbose=-1, num_threads=4, metric="auc"),
                  dtr, 300, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[k:], b.predict(X[k:]))
    imp = sorted(zip(cols, b.feature_importance("gain")), key=lambda x: -x[1])[:12]
    tot = sum(v for _, v in imp) or 1
    print(f"== {name}: AUC {auc:.3f}  top shift features:", [(c_, round(v_ / tot, 2)) for c_, v_ in imp], flush=True)
    return b


adv(t.filter(pl.col("country") != "France"), v, "test US+India vs val")
adv(t.filter(pl.col("country") == "US"), v.filter(pl.col("country") == "US"), "test US vs val US")
adv(t.filter(pl.col("country") == "India"), v.filter(pl.col("country") == "India"), "test India vs val India")
adv(t.filter(pl.col("country") == "France"), v, "test France vs val")
# mean comparison of the key features (high-p region where decisions happen)
key = ["p_prune", "n_tset", "a_tset", "amb_name_s1_c", "amb_name_s23_c", "den_addr_s23_c", "b_rank23", "n_cand23",
       "a2_empty", "num_rel", "legal_rel", "kswap", "fill_c", "unseen_c", "bscore"]
for nm, d in [("val", v), ("test US", t.filter(pl.col("country") == "US")), ("test India", t.filter(pl.col("country") == "India")),
              ("test France", t.filter(pl.col("country") == "France"))]:
    print(f"{nm:11s}", {k_: round(float(d[k_].cast(pl.Float64).mean()), 3) for k_ in key if k_ in d.columns})
