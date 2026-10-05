"""Training: blocking ranker (linear), pruner (LightGBM), matcher (LightGBM).

Only training S1 entities outside the validation split are used for fitting.
"""
import json
import re

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

import blocking as B
import os

from config import FULL_S1, NTHREADS, SEED, STACK, work_path


from config import DROP_PCT  # noqa: E402  (% of train S1 hidden -> orphan decoys, test density)


def drop_s1(df: pl.DataFrame, col: str = "s1") -> pl.DataFrame:
    """Simulate test distractor density: test has ~5.75 S2/S3 records per S1 vs 4.68 in train."""
    return df.filter(pl.col(col).hash(seed=21) % 100 >= DROP_PCT)


def split_ids():
    sp = pl.read_parquet(work_path("train_split.parquet"))
    return sp.filter(~pl.col("is_val"))["idx"], sp.filter(pl.col("is_val"))["idx"]


def label(pairs: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    return pairs.join(gt.with_columns(pl.lit(1, dtype=pl.Int8).alias("label")), on=["s1", "s23"], how="left").with_columns(
        pl.col("label").fill_null(0))


def train_block_ranker(n_s1: int = 20_000) -> dict:
    gt = pl.read_parquet(work_path("train_gt.parquet"))
    tr, _ = split_ids()
    sub = tr.sample(min(n_s1, len(tr)), seed=SEED)
    s1k = pl.read_parquet(work_path("train_keys_s1.parquet"))
    s23k = pl.read_parquet(work_path("train_keys_s23.parquet"))
    pairs = B.generate(s1k, s23k, sub, top_k=100_000, ranker=None)
    pairs = label(pairs, gt).with_columns(B.pair_feature_exprs())
    cols = [f"lw{f}" for f in range(len(B.FAMS))] + [f"r{f}" for f in range(len(B.FAMS))] + ["lnk"]
    # downsample negatives for speed; LR ranking is invariant to the intercept
    pos = pairs.filter(pl.col("label") == 1)
    neg = pairs.filter(pl.col("label") == 0).sample(min(3_000_000, pairs.height - pos.height), seed=SEED)
    d = pl.concat([pos, neg])
    X = d.select(cols).to_numpy().astype(np.float32)
    y = d["label"].to_numpy()
    lr = LogisticRegression(C=1.0, max_iter=500)
    lr.fit(X, y)
    ranker = {"intercept": float(lr.intercept_[0]), "coef": {c: float(v) for c, v in zip(cols, lr.coef_[0])}}
    work_path("block_ranker.json").write_text(json.dumps(ranker, indent=1))
    print("ranker", ranker)
    return ranker


def train_lgb_ranker(n_s1: int = 40_000) -> None:
    """Stage-1 LightGBM re-ranker over the linear shortlist (PRE_K per S1)."""
    gt = pl.read_parquet(work_path("train_gt.parquet"))
    tr, va = split_ids()
    s1k = pl.read_parquet(work_path("train_keys_s1.parquet"))
    s23k = pl.read_parquet(work_path("train_keys_s23.parquet"))
    sub = pl.concat([tr.sample(min(n_s1, len(tr)), seed=SEED), va.sample(min(n_s1 // 4, len(va)), seed=SEED)])
    pairs = label(B.generate(s1k, s23k, sub, top_k=B.PRE_K, lgb_ranker=None, keep_feats=True,
                             s23_empty=B.empty_s23("train")), gt)
    pairs = pairs.with_columns(pl.col("s1").is_in(va.implode()).alias("is_val"))
    cols = B.LGB_RANK_COLS
    t_, v_ = pairs.filter(~pl.col("is_val")), pairs.filter(pl.col("is_val"))
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=NTHREADS, seed=SEED)
    dtr = lgb.Dataset(t_.select(cols).to_numpy().astype(np.float32), t_["label"].to_numpy())
    dva = lgb.Dataset(v_.select(cols).to_numpy().astype(np.float32), v_["label"].to_numpy(), reference=dtr)
    bst = lgb.train(params, dtr, 400, valid_sets=[dva], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)])
    bst.save_model(str(work_path("block_ranker_lgb.txt")))
    # recall@K on held-out S1: linear vs LightGBM ordering
    v_ = v_.with_columns(pl.Series("g", bst.predict(v_.select(cols).to_numpy().astype(np.float32), num_threads=NTHREADS)))
    ntrue = gt.join(v_.select("s1").unique(), on="s1", how="semi").height
    for k in (20, 30, 40, 60):
        lr = v_.filter(pl.col("lr_rank") <= k)["label"].sum() / ntrue
        gb = v_.filter(pl.col("g").rank("ordinal", descending=True).over("s1") <= k)["label"].sum() / ntrue
        print(f"stage-1 recall@{k}: linear {lr:.4f}  lgb {gb:.4f}", flush=True)
    print(f"recall@{B.PRE_K} (shortlist ceiling): {v_['label'].sum() / ntrue:.4f}", flush=True)


# ------------------------------------------------------------------ features
def make_features(split: str, cand: pl.DataFrame, out_name: str, chunk_s1: int = 100_000) -> pl.DataFrame:
    import gc
    import features as F
    r1 = F.load_records(split, "s1")
    r23 = F.load_records(split, "s23")
    cand = F.s23_context(cand)
    ids = cand["s1"].unique().sort()
    parts = []
    for n, i in enumerate(range(0, len(ids), chunk_s1)):
        sub = cand.join(pl.DataFrame({"s1": ids[i:i + chunk_s1]}), on="s1", how="semi")
        need23 = sub.select(pl.col("s23").alias("idx")).unique()
        f = F.build(sub, r1.join(sub.select(pl.col("s1").alias("idx")).unique(), on="idx", how="semi"),
                    r23.join(need23, on="idx", how="semi"))
        f = F.compact(f)
        p = work_path(f"parts_{out_name}", f"{n:04d}.parquet")
        f.write_parquet(p)
        parts.append(p)
        print("features", i, f.height, flush=True)
        del f, sub
        gc.collect()
    del r1, r23
    if FULL_S1:
        # pruner training set = the usual state subset; all parts are kept on disk for scoring
        work_path(f"parts_{out_name}.json").write_text(json.dumps([str(p) for p in parts]))
        keep = pl.DataFrame({"s1": B.train_subset().cast(pl.Int32)})
        out = pl.concat([pl.read_parquet(p).join(keep, on="s1", how="semi") for p in parts])
    else:
        out = pl.concat([pl.read_parquet(p) for p in parts])
    out.write_parquet(work_path(out_name))
    return out


def score_rest(feat_name: str, scored: pl.DataFrame, gt: pl.DataFrame, k: int = 10, tau: float = 0.002) -> pl.DataFrame:
    """Full-competition mode: score the S1 outside the pruner-training subset with the
    full pruner (out-of-sample for them), prune, label, and append to `scored`."""
    bst = lgb.Booster(model_file=str(work_path("pruner.txt")))
    cols = json.loads(work_path("pruner_cols.json").read_text())
    _, va_ids = split_ids()
    have = scored.select("s1").unique()
    extra = []
    for p in json.loads(work_path(f"parts_{feat_name}.json").read_text()):
        f = pl.read_parquet(p).join(have, on="s1", how="anti")
        if f.height == 0:
            continue
        f = f.with_columns(pl.Series("p_prune", predict(bst, f, cols)))
        f = prune(f, k, tau).drop("p_rank")
        f = label(f, gt).with_columns(pl.col("s1").is_in(va_ids.implode()).alias("is_val"))
        extra.append(f)
        print("scored rest part", p, f.height, flush=True)
    return pl.concat([scored] + extra, how="diagonal_relaxed")


LGB_PARAMS = dict(
    objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=NTHREADS, seed=SEED,
)


MATCHER_PARAMS = dict(LGB_PARAMS, learning_rate=0.03, num_leaves=255, min_data_in_leaf=50, feature_fraction=0.7)


def fit_lgb(tr: pl.DataFrame, va: pl.DataFrame, cols: list[str], name: str, rounds: int = 1000,
            params: dict | None = None) -> lgb.Booster:
    params = params or LGB_PARAMS
    dtr = lgb.Dataset(tr.select(cols).to_numpy().astype(np.float32), tr["label"].to_numpy(), feature_name=cols, free_raw_data=True)
    dva = lgb.Dataset(va.select(cols).to_numpy().astype(np.float32), va["label"].to_numpy(), reference=dtr)
    stop = 50 if params is LGB_PARAMS else 150
    bst = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(stop), lgb.log_evaluation(200)])
    bst.save_model(str(work_path(f"{name}.txt")))
    imp = sorted(zip(cols, bst.feature_importance("gain")), key=lambda x: -x[1])
    print(name, "top features", [(c, round(v)) for c, v in imp[:25]])
    return bst


def predict(bst: lgb.Booster, df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return bst.predict(df.select(cols).to_numpy().astype(np.float32), num_threads=NTHREADS)


# ------------------------------------------------------------------ pruner
PRUNE_DROP = {"s1", "s23", "label", "p_prune", "p", "fold"}


def prune_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in PRUNE_DROP and not c.startswith("tot")]


def train_pruner(feat: pl.DataFrame) -> pl.DataFrame:
    """2-fold out-of-fold pruner on the train part, plus a full model for val/test."""
    _, va_ids = split_ids()
    feat = feat.with_columns(pl.col("s1").is_in(va_ids.implode()).alias("is_val"))
    tr = feat.filter(~pl.col("is_val")).with_columns((pl.col("s1").hash(seed=3) % 2).cast(pl.Int8).alias("fold"))
    va = feat.filter(pl.col("is_val"))
    cols = prune_cols(tr.drop("is_val"))
    es = va.sample(min(2_000_000, va.height), seed=SEED)
    oof = []
    for k in (0, 1):
        bst = fit_lgb(tr.filter(pl.col("fold") != k), es, cols, f"pruner_f{k}")
        part = tr.filter(pl.col("fold") == k)
        oof.append(part.with_columns(pl.Series("p_prune", predict(bst, part, cols))))
    bst = fit_lgb(tr, es, cols, "pruner")
    va = va.with_columns(pl.Series("p_prune", predict(bst, va, cols)))
    work_path("pruner_cols.json").write_text(json.dumps(cols))
    return pl.concat([pl.concat(oof).drop("fold"), va], how="diagonal_relaxed")


def prune(df: pl.DataFrame, k: int, tau: float) -> pl.DataFrame:
    df = df.with_columns(pl.col("p_prune").rank("ordinal", descending=True).over("s1").alias("p_rank"))
    return df.filter((pl.col("p_rank") <= k) & (pl.col("p_prune") >= tau))


def tune_prune(scored: pl.DataFrame) -> None:
    from evaluate import blocking_report
    gt = pl.read_parquet(work_path("train_gt.parquet"))
    va = scored.filter(pl.col("is_val"))
    va_ids = va["s1"].unique()
    for k in (5, 6, 8, 10, 12, 15):
        for tau in (0.0, 0.002, 0.01, 0.03):
            r = blocking_report(prune(va, k, tau).select("s1", "s23"), gt, va_ids)
            print(f"K={k} tau={tau}", {a: round(b, 4) for a, b in r.items()})


# ------------------------------------------------------------------ matcher
COMP_COLS = ["n_ratio", "n_tset", "k_tset", "c_jw", "x_jw", "a_tset", "a_ratio"]


def context_features(df: pl.DataFrame, split: str | None = None) -> pl.DataFrame:
    """Context computed within the pruned candidate set (what the matcher sees)."""
    import features as F
    split = split or os.environ.get("BER_SPLIT", "train")
    f1, f23 = F.full_name_counts(split)
    df = df.join(f1.rename({"idx": "s1", "fS1": "full_s1_s", "fS23": "full_s23_s"}), on="s1", how="left") \
           .join(f23.rename({"idx": "s23", "fS1": "full_s1_c", "fS23": "full_s23_c"}), on="s23", how="left")
    # competition on similarity: is this S1 the most similar of all S1s claiming the same record?
    comp = []
    for c in COMP_COLS:
        if c in df.columns:
            v = pl.col(c).cast(pl.Float32)
            comp += [(v.max().over("s23") - v).alias(f"{c}_gap23"),
                     v.rank("ordinal", descending=True).over("s23").cast(pl.Int16).alias(f"{c}_rank23"),
                     (v.max().over("s1") - v).alias(f"{c}_gap1")]
    df = df.with_columns(comp)
    return df.with_columns(
        pl.col("p_prune").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("pp_rank"),
        (pl.col("p_prune").max().over("s1") - pl.col("p_prune")).alias("pp_gap"),
        pl.col("p_prune").sum().over("s1").alias("pp_sum1"),
        pl.len().over("s1").cast(pl.Int16).alias("pp_n1"),
        pl.col("p_prune").rank("ordinal", descending=True).over("s23").cast(pl.Int16).alias("pp_rank23"),
        (pl.col("p_prune").max().over("s23") - pl.col("p_prune")).alias("pp_gap23"),
        pl.col("p_prune").sum().over("s23").alias("pp_sum23"),
        pl.len().over("s23").cast(pl.Int16).alias("pp_n23"),
        # second best S1 for this record (competition)
        (pl.col("p_prune") - pl.col("p_prune").top_k(2).min().over("s23")).alias("pp_margin23"),
    )


def stack_features(d: pl.DataFrame, r23: pl.DataFrame) -> pl.DataFrame:
    """Second-pass context from first-pass matcher probabilities p1: the S1's other
    confident matches, competition for the same S2/S3 record, and similarity of
    the candidate to the records the first pass already accepted."""
    import features as F
    d = d.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias("q_rank"),
        (pl.col("p1").max().over("s1") - pl.col("p1")).alias("q_gap"),
        (pl.col("p1").sum().over("s1") - pl.col("p1")).alias("q_sum_other"),
        (pl.col("p1") >= 0.5).sum().over("s1").cast(pl.Int16).alias("q_n05"),
        pl.col("p1").max().over("s23").alias("q_max23"),
        (pl.col("p1").sum().over("s23") - pl.col("p1")).alias("q_other23"),
    )
    return F.cluster_features(d, r23, strong=0.5, pcol="p1", pre="q_cl_")


def matcher_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in PRUNE_DROP | {"is_val", "p_rank"} and not c.startswith("tot")] + ["p_prune"]


def evaluate_selection(va: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series, pcol: str = "p") -> dict:
    import postprocess as P
    from evaluate import macro_f05
    res = {}
    for oto in (False, True):
        d = P.one_to_one(va, pcol) if oto else va
        for t in (0.3, 0.4, 0.5, 0.6, 0.7):
            res[f"thr{t}_oto{int(oto)}"] = macro_f05(P.threshold_sets(d, pcol, t).select("s1", "s23"), gt, s1_ids)
        for miss in (0.0, 0.1, 0.3):
            for es in (0.8, 1.0, 1.25, 1.5, 2.0):
                res[f"ef_m{miss}_e{es}_oto{int(oto)}"] = macro_f05(P.select_sets(d, pcol, miss, es).select("s1", "s23"), gt, s1_ids)
    best = max(res, key=res.get)
    work_path("select_grid.json").write_text(json.dumps(res, indent=1))
    for k, v in sorted(res.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k}: {v:.5f}")
    print("best", best, res[best])
    return res


def train_matcher(scored: pl.DataFrame, k: int, tau: float) -> None:
    import features as F
    gt = pl.read_parquet(work_path("train_gt.parquet"))
    d = context_features(prune(scored, k, tau), "train")
    r23 = F.load_records("train", "s23").select("idx", "name_n", "addr_n")
    d = F.cluster_features(d, r23)
    tr = d.filter(~pl.col("is_val"))
    va = d.filter(pl.col("is_val"))
    cols = matcher_cols(d)
    es = va.sample(min(1_000_000, va.height), seed=SEED)
    bst = fit_lgb(tr, es, cols, "matcher", 4000, MATCHER_PARAMS)
    work_path("matcher_cols.json").write_text(json.dumps(cols))
    if STACK:
        # out-of-fold first-pass probabilities on the train part, full model on val
        tr = tr.with_columns((pl.col("s1").hash(seed=5) % 2).cast(pl.Int8).alias("mfold"))
        parts = []
        for k in (0, 1):
            b = fit_lgb(tr.filter(pl.col("mfold") != k), es, cols, f"matcher_f{k}", 4000, MATCHER_PARAMS)
            part = tr.filter(pl.col("mfold") == k)
            parts.append(part.with_columns(pl.Series("p1", predict(b, part, cols))))
        tr = pl.concat(parts).drop("mfold")
        va = va.with_columns(pl.Series("p1", predict(bst, va, cols)))
        both = stack_features(pl.concat([tr, va], how="diagonal_relaxed"), r23)
        tr, va = both.filter(~pl.col("is_val")), both.filter(pl.col("is_val"))
        cols2 = matcher_cols(both)
        es2 = va.join(es.select("s1", "s23"), on=["s1", "s23"], how="semi")
        bst2 = fit_lgb(tr, es2, cols2, "matcher2", 4000, MATCHER_PARAMS)
        work_path("matcher2_cols.json").write_text(json.dumps(cols2))
        from evaluate import macro_f05
        import postprocess as P
        s1v = scored.filter(pl.col("is_val"))["s1"].unique()
        p1sel = P.select_sets(P.one_to_one(va.rename({"p1": "q"}), "q"), "q", 0.0, 1.0)
        print("first-pass val F0.5:", macro_f05(p1sel.select("s1", "s23"), gt, s1v), flush=True)
        va = va.with_columns(pl.Series("p", predict(bst2, va, cols2)))
    else:
        va = va.with_columns(pl.Series("p", predict(bst, va, cols)))
    del r23
    va.select("s1", "s23", "p", "p_prune", "label").write_parquet(work_path("val_pred.parquet"))
    s1_ids = scored.filter(pl.col("is_val"))["s1"].unique()
    print("val S1", len(s1_ids), "pruned pairs", va.height)
    res = evaluate_selection(va, gt, s1_ids)
    work_path("select_params.json").write_text(json.dumps(selector_from_key(max(res, key=res.get))))


def selector_from_key(key: str) -> dict:
    """'ef_m0.3_e0.8_oto1' / 'thr0.5_oto1' -> params used by infer.select()."""
    oto = key.endswith("oto1")
    if key.startswith("ef_"):
        m = re.match(r"ef_m([\d.]+)_e([\d.]+)_oto", key)
        return {"selector": "ef", "miss": float(m.group(1)), "empty_scale": float(m.group(2)), "one_to_one": oto}
    m = re.match(r"thr([\d.]+)_oto", key)
    return {"selector": "thr", "t": float(m.group(1)), "one_to_one": oto}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ranker", "lgbranker", "feats", "pruner", "matcher"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--feat-name", default="train_feat1.parquet")
    ap.add_argument("--scored-name", default="train_scored1.parquet")
    ap.add_argument("--drop", action="store_true", help="drop DROP_PCT%% of S1 (test-like distractor density)")
    a = ap.parse_args()
    if a.stage == "ranker":
        train_block_ranker()
    elif a.stage == "lgbranker":
        train_lgb_ranker()
    elif a.stage == "feats":
        c = pl.read_parquet(work_path("train_cand1.parquet"))
        if a.drop:
            c = drop_s1(c)
        make_features("train", c, a.feat_name)
    elif a.stage == "pruner":
        gt = pl.read_parquet(work_path("train_gt.parquet"))
        feat = label(pl.read_parquet(work_path(a.feat_name)), gt)
        scored = train_pruner(feat)
        tune_prune(scored)
        if FULL_S1:
            del feat
            scored = score_rest(a.feat_name, scored, gt)
            print("scored pairs incl. all visible S1:", scored.height, "S1", scored["s1"].n_unique(), flush=True)
        scored.write_parquet(work_path(a.scored_name))
    elif a.stage == "matcher":
        scored = pl.read_parquet(work_path(a.scored_name))
        train_matcher(scored, a.k, a.tau)
        json.dump({"k": a.k, "tau": a.tau}, open(work_path("prune_params.json"), "w"))
