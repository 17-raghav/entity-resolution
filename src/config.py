"""Paths, seeds and tunable constants for the pipeline.

All paths can be overridden from the command line of run_pipeline.py or via
environment variables BER_DATA and BER_WORK.
"""
import os
from pathlib import Path

SEED = 42

DATA_DIR = Path(os.environ.get("BER_DATA", "data/dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK", "work"))

# Worker threads / processes (defaults to all cores of the machine).
NTHREADS = int(os.environ.get("BER_THREADS", os.cpu_count() or 8))

# Stage-1 size: top-K candidates per S1 plus reserved slots for empty-address records.
TOPK1 = int(os.environ.get("BER_TOPK1", 40))
EMPTY_SLOTS = int(os.environ.get("BER_EMPTY_SLOTS", 5))
ADDR_SLOTS = int(os.environ.get("BER_ADDR_SLOTS", 10))  # best address-only evidence (rebranded names)
# Second-pass (stacked) matcher on/off; inference chunk size (memory).
STACK = os.environ.get("BER_STACK", "0") == "1"
INFER_CHUNK = int(os.environ.get("BER_INFER_CHUNK", 200_000))

# % of train S1 hidden during training/validation so their S2/S3 records become orphan decoys
# (test has 5.75 S2/S3 records per S1 vs 4.68 in train). The hidden S1 must also be
# invisible to every S1-side statistic, exactly as orphan entities are invisible in test.
DROP_PCT = 19
# Full-competition training: stage-1 / features / pruner scores for ALL visible train S1
# (test has every S1 present, so competition features must be computed the same way).
FULL_S1 = os.environ.get("BER_FULL_S1", "0") == "1"

# Fraction of train S1 entities held out for validation.
VAL_FRAC = 0.10


def raw_path(split: str, name: str) -> Path:
    """split in {train,test}; name in {source1,source2,source3,ground_truth}."""
    return DATA_DIR / split / f"{split}_{name}.tsv"


def work_path(*parts: str) -> Path:
    p = WORK_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
