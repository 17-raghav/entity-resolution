"""Re-select matches from saved test probabilities with country-specific sharpening.
usage: python reselect.py <work_dir> <out_dir> <gamma_default> [Country=gamma ...]
p' = p ** gamma (gamma > 1 = stricter), then one-to-one + expected-F0.5 selection
with the selector parameters stored in <work_dir>/select_params.json."""
import json
import os
import sys

import polars as pl

work, out = sys.argv[1], sys.argv[2]
g0 = float(sys.argv[3])
per = {k: float(v) for k, v in (a.split("=") for a in sys.argv[4:])}
os.environ["BER_WORK"] = work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import postprocess as P  # noqa: E402
from io_utils import write_from_pairs  # noqa: E402

prm = json.load(open(f"{work}/select_params.json"))
d = pl.read_parquet(f"{work}/test_pred.parquet", columns=["s1", "s23", "p"])
c = pl.read_parquet(f"{work}/test_s1.parquet", columns=["idx", "country"]).rename({"idx": "s1"})
d = d.join(c, on="s1")
gam = pl.lit(g0)
for k, v in per.items():
    gam = pl.when(pl.col("country") == k).then(pl.lit(v)).otherwise(gam)
d = d.with_columns((pl.col("p") ** gam).alias("p"))
if prm.get("one_to_one", True):
    d = P.one_to_one(d, "p")
sel = P.select_sets(d, "p", prm["miss"], prm["empty_scale"]) if prm["selector"] == "ef" else P.threshold_sets(d, "p", prm["t"])
os.makedirs(out, exist_ok=True)
write_from_pairs(f"{out}/matching_results.tsv", "test", sel.select("s1", "s23"), "matched_entity_ids")
st = sel.join(c, on="s1").group_by("country").len().join(c.group_by("country").len("n"), on="country")
print(prm, "gamma", g0, per, "matches/S1:", st.with_columns((pl.col("len") / pl.col("n")).round(3)).select("country", "len").rows())
