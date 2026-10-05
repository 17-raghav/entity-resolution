"""Single entry point: data -> normalise -> blocking -> pruner -> matcher -> outputs.

  python src/run_pipeline.py --data <dataset_dir> --work <work_dir> --out <output_dir> [--train]

--train  re-fits everything from the training data first (lexicons, blocking
         ranker, pruner, matcher). Without it the models saved in <work_dir>
         are used.
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--skip-prep", action="store_true", help="reuse normalised parquet + stage-1 candidates")
    ap.add_argument("--k", type=int, default=10, help="pruner: max candidates per S1")
    ap.add_argument("--tau", type=float, default=0.002, help="pruner: min probability")
    ap.add_argument("--topk1", type=int, default=40, help="stage-1 candidates per S1")
    ap.add_argument("--empty-slots", type=int, default=5, help="extra stage-1 slots for empty-address records")
    ap.add_argument("--addr-slots", type=int, default=10, help="extra stage-1 slots for best address evidence")
    ap.add_argument("--stack", action="store_true", help="second-pass (stacked) matcher")
    ap.add_argument("--infer-chunk", type=int, default=200_000, help="S1 per inference chunk (lower = less RAM)")
    a = ap.parse_args()
    # steps run with cwd=src/, so relative paths must be resolved first
    a.data, a.work, a.out = (str(Path(x).resolve()) for x in (a.data, a.work, a.out))
    os.environ["BER_DATA"] = a.data
    os.environ["BER_WORK"] = a.work
    os.environ["BER_TOPK1"] = str(a.topk1)
    os.environ["BER_EMPTY_SLOTS"] = str(a.empty_slots)
    os.environ["BER_ADDR_SLOTS"] = str(a.addr_slots)
    os.environ["BER_STACK"] = "1" if a.stack else "0"
    os.environ["BER_INFER_CHUNK"] = str(a.infer_chunk)
    src = Path(__file__).resolve().parent
    sys.path.insert(0, str(src))

    done_dir = Path(a.work) / "done"
    done_dir.mkdir(parents=True, exist_ok=True)

    def step(*args, tag=None, always=False):
        """Run one pipeline step; a marker file makes re-runs skip finished steps."""
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag or "_".join(x.replace("--", "").replace(".py", "") for x in args))
        marker = done_dir / tag
        if marker.exists() and not always:
            print(">> skip (done)", " ".join(args), flush=True)
            return
        t = time.time()
        print(">>", " ".join(args), time.strftime("%H:%M:%S"), flush=True)
        subprocess.run([sys.executable, *args], cwd=src, check=True, env=os.environ)
        marker.write_text(f"{time.time() - t:.0f}s")
        print(f"   done in {time.time() - t:.0f}s", flush=True)

    if not a.skip_prep:
        step("io_utils.py")
        if a.train:
            step("evaluate.py")                      # scorer self-test + validation split
            step("normalize.py", "--split", "train", tag="normalize_train_pass1")  # no tables yet
            step("mine_lexicons.py")
            step("normalize.py", "--split", "train", tag="normalize_train_pass2")  # tables applied
            step("blocking.py", "--split", "train", "--keys-only")
            step("train.py", "ranker")        # linear stage-1 ranker
            step("train.py", "lgbranker")     # LightGBM stage-1 re-ranker (prints recall@K)
            step("blocking.py", "--split", "train")
            step("train.py", "feats", "--drop", "--feat-name", "train_feat.parquet")
            step("train.py", "pruner", "--feat-name", "train_feat.parquet", "--scored-name", "train_scored.parquet")
            step("train.py", "matcher", "--k", str(a.k), "--tau", str(a.tau), "--scored-name", "train_scored.parquet")
        step("normalize.py", "--split", "test")
        step("blocking.py", "--split", "test")
    step("infer.py", "--out", a.out, tag="infer", always=True)


if __name__ == "__main__":
    main()
