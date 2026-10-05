# entity-resolution

Multilingual business entity matching at scale: blocking, LightGBM pruning and matching, and
precision-aware set selection, on about 12 million records per split.

This project was built on the Amazon ML Challenge 2026 dataset. The dataset is not included here
(see "Data" below).

## Problem

There are three sources of business records. Each record has an ID, a business name, an address and
a country. Source 1 (S1) is a clean reference list. Sources 2 and 3 (S2/S3) hold noisy copies of the
same businesses, plus records that belong to no S1 business. For every S1 entity, the task is to
return the set of S2/S3 records that describe the same real business. The set may be empty.

Scoring is F0.5 per S1 entity, averaged over all entities. F0.5 weights precision more than recall,
so a wrong match costs more than a missed one. An entity with no true matches scores 1 only if we
predict nothing.

What makes it hard:
- Names change between sources: typos, accents, word swaps, leetspeak (`6roup`), legal-form
  variants (Ltd / Limited / Pvt), filler words added (`services`, `holding`), rebrands, domain-style
  names, and Indian names written in nine Indic scripts.
- Addresses change: abbreviations, reordered or missing parts, truncated house numbers, states
  written as codes or in native script.
- Hard negatives: the same name at a nearby house number, or a near-identical name at the same address.
- Scale: about 1.7M S1 and 10M S2/S3 records in test. Comparing all pairs is impossible.
- The test set contains a country (France) that never appears in training.

## Approach

```
raw TSV -> normalise -> blocking keys -> stage-1 ranking (linear, then LightGBM)
        -> pair features -> LightGBM pruner (candidate_pairs.tsv)
        -> LightGBM matcher -> one-to-one + expected-F0.5 set selection (matching_results.tsv)
```

1. **Normalise** (`src/normalize.py`): accent folding, leetspeak repair, legal-form and street-type
   canonicalisation (English and French), Indic-to-Latin transliteration (a lexicon mined from
   training pairs, with a rule-based fallback), state detection.
2. **Mine tables from training data only** (`src/mine_lexicons.py`): Indic token lexicon, state
   aliases, generic filler words.
3. **Blocking** (`src/blocking.py`): ten families of compound keys (name tokens, name x address word,
   house number x street word, address identifiers, no-space name, 3-letter prefixes for typos, ...).
   Every key is prefixed with the country, so pairs are only formed within a country. Keys that are
   too common are dropped. Each pair gets IDF-weighted key scores per family. A logistic ranker keeps
   200 candidates per S1, a LightGBM re-ranker keeps the top 60, and 15 extra slots are reserved for
   empty-address records and address-only evidence.
4. **Features** (`src/features.py`): about 140 pair features, all similarities, flags or counts
   (never raw tokens or the country), so they transfer to an unseen country.
5. **Pruner** (`src/train.py`): LightGBM, trained 2-fold out-of-fold. Keeps at most 10 candidates per
   S1 with probability >= 0.002. This list is `candidate_pairs.tsv`.
6. **Matcher** (`src/train.py`): LightGBM (255 leaves) with context features: how this S1 ranks among
   all S1 claiming the same record, and similarity to the S1's other confident matches.
7. **Selection** (`src/postprocess.py`): each S2/S3 record keeps only its best S1 (in training every
   record belongs to at most one S1). Then, per S1, choose the prefix of candidates that maximises
   expected F0.5, or predict empty when that is better.

Two ideas mattered most:
- **Make validation look like test.** Test has 5.75 S2/S3 records per S1 vs 4.68 in train. Hiding 19%
  of train S1 turns their records into orphans and reproduces the test density.
- **Label-free handling of the unseen country.** Filler words (tokens >= 8x more frequent in S2/S3
  than in S1) and state/region parts (frequent, almost always last in the address) are found from
  frequency statistics, so France works without French labels.

### Main pipeline version: v10, with v11 matcher variant

- `src/` is the v10 pipeline (git commit `90bce1c` of the original repo). v10 trains with
  "full competition" (`BER_FULL_S1=1`): stage-1, features and pruner scores for all visible train S1,
  so that the competition features match test.
- `v11/` holds the v11 changes, kept separate. v11 retrains only the matcher on v10's pairs: it drops
  candidate pairs whose S2/S3 record belongs to a hidden S1 (those orphans are noisy copies, which
  never occur in test), and weights clean decoy negatives by 1.0 or 2.0. It was run on AWS but its
  leaderboard score was not recorded.
- `analysis/adv.py` is the adversarial-validation script; `analysis/reselect.py` re-selects matches
  from saved probabilities with per-country sharpening (used for the P1/P2/v8_f05 probes).

## Diagnosis: why validation was ~0.984 and the leaderboard ~0.975

Validation F0.5 reached 0.98415 (v7) but the public score of the same models was 0.973933 (v7) and
0.975 (v8, after a France fix). A gap of about 0.009 is too large for noise. Three steps found the cause:

1. **Probes.** Making selection stricter everywhere (p^2) did not help, and stricter on France only
   (p^3) dropped the score to 0.972. So the model was not over-matching; if anything it under-matched.
2. **Per-country split.** A submission with all France rows blank scored 0.839. Since France is 15% of
   test, this implies France ~0.964 and US+India ~0.977. So most of the gap was in the US and India,
   the countries we had training data for, not only in the unseen country.
3. **Adversarial validation** (`analysis/adv.py`). Build pruned candidates with the same pipeline for
   a test sample and for validation, then train a classifier to tell them apart and look at which
   features it uses. The biggest shift was in competition features: `n_cand23` (how many S1 entities
   claim the same S2/S3 record at stage 1) averaged about 10 on validation and 15.5 / 20 / 16.4 on test
   US / India / France.

Cause: the pruner and matcher were trained on candidates for only about a third of the states (to
save memory), while at test time every S1 is present. The model therefore saw more competition at
test time than in training and under-trusted true matches. This explains all three observations.

Fix (v10): compute stage-1 candidates, features and pruner scores for all visible train S1, and train
the matcher on all pruned pairs. Validation went 0.98415 -> 0.98444 and public score 0.975 -> 0.976143.
The fix closed only a small part of the gap, which suggests the competition shift was not the whole
story; v11 tested a second hypothesis (orphan type). Sources: `results/RESULTS.md` sections 2-3.

## How to run

Python 3.12, CPU only.

```bash
pip install -r requirements.txt
```

### Data

Download the Amazon ML Challenge 2026 "student resource" archive from the challenge page (Unstop) and
unpack it. The pipeline needs `dataset/train/train_source{1,2,3}.tsv`, `train_ground_truth.tsv` and
`dataset/test/test_source{1,2,3}.tsv`. Do not commit data; `data/` is git-ignored.

### Smoke test (about 3 minutes; the blocking step reported 2.7 GB process memory)

```bash
python scripts/make_sample.py --raw <student_resource>/dataset --out data/sample --n-train 20000 --n-test 5000
BER_FULL_S1=1 python src/run_pipeline.py --data data/sample --work work_smoke --out output_smoke \
    --train --topk1 60 --empty-slots 5 --addr-slots 10
python scripts/score.py --pred output_smoke/matching_results.tsv \
    --gt data/sample/test_ground_truth.tsv --s1 data/sample/test/test_source1.tsv
```

On Windows PowerShell set the variable with `$env:BER_FULL_S1 = "1"` first.

v11 matcher on the smoke artifacts:

```bash
BER_V10_WORK=work_smoke BER_V11_ROOT=v11_smoke BER_V10_OUT=output_smoke python v11/v11_matcher.py 2.0 w2
BER_V10_WORK=work_smoke BER_V11_ROOT=v11_smoke BER_V10_OUT=output_smoke python v11/v11_infer.py w2
```

### Full run (not re-run for this repo)

```bash
BER_FULL_S1=1 python src/run_pipeline.py --data <student_resource>/dataset --work work --out output \
    --train --topk1 60 --empty-slots 5 --addr-slots 10
```

Needs about 64 GB RAM and 2-3 hours on 8 vCPU (originally an AWS r7i.2xlarge). Steps are resumable:
finished steps leave a marker in `<work>/done/`.

## Results

Smoke test, run on 2026-10-05 (laptop, Core Ultra 9 185H). Sample = 20k train + 5k "test" S1 cut from
the train split. Not comparable to full-data numbers (far fewer competing records per entity).

| Smoke test | Value | Log |
|---|---|---|
| End-to-end time, train + test | 3 min 3 s | `results/smoke_pipeline.log:212` |
| Macro F0.5 on 5,000 sample S1 (v10), 3 runs of identical code and data | 0.9947 / 0.9924 / 0.9940 | `results/smoke_score.log:2`, `results/smoke_repeat_2.log`, `results/smoke_repeat_3.log` |
| Macro F0.5 on same S1 (v11 matcher, w2) | 0.9957 | `results/smoke_v11.log:26` |
| Official validator | PASS | `results/smoke_score.log:14` |

Full data, from existing logs and notes (not re-run). Full table with sources: `results/RESULTS.md`.

| Version | Validation F0.5 | Public leaderboard F0.5 |
|---|---|---|
| v2 (test-like decoy density) | 0.9798 | 0.972 |
| v5 (filler features, reserved slots) | 0.9832 | 0.97320 |
| v7 (two-pass matcher) | 0.98415 | 0.973933 |
| v8 (v7 + label-free France states) | same models | 0.975 |
| v10 (full-competition training) | 0.98444 | 0.976143 |

| Pipeline figure (v5, full validation) | Value |
|---|---|
| Stage-1 recall@60, LightGBM vs linear ranker | 0.9818 vs 0.9762 |
| Pair recall after pruner (K=10, tau=0.002) | 0.9827 at 4.67 candidates per S1 |

## Not yet verified

- v11 public leaderboard score (never recorded).
- v1 public leaderboard score: 0.963 in HANDOFF.md vs 0.9600 in the documentation draft; unresolved.
- Adversarial-validation AUC values (output not saved locally); only the feature means above are noted.
- Oracle F0.5 of ~0.994 on the candidate set and the validation loss breakdown: written in notes,
  raw logs not kept.
- v4 and v7 validation scores: only in submission notes, no raw log kept locally.
- Any full-data number re-run on current hardware: the full pipeline was not re-run.
- Bit-for-bit reproducibility: repeated smoke runs on the same sample give F0.5 0.9924-0.9947, so some
  step is non-deterministic (likely row order from unordered polars group-bys feeding LightGBM). Not
  yet traced.

## Limitations

- Full training needs about 64 GB RAM; the laptop limit (32 GB) was exceeded in the past.
- Hyper-parameters (block caps, K, tau, selector settings) were tuned on validation drawn from the
  same distribution as training; the unseen country is handled by rules and statistics, not by tuning.
- Truly ambiguous cases exist: an empty-address record whose name is shared by several S1 entities
  with identical raw names. No string feature can separate them.
- The smoke test is a functional check only; its scores say nothing about full-data quality.
- `analysis/adv.py` and the v11 scripts still default to the AWS directory layout (`~/ml/...`);
  override with the environment variables shown above.

## Next steps

- Record v11 on a held-out score before claiming any gain.
- Importance-weight training pairs by the adversarial classifier's density ratio, instead of only
  matching the competition setup.
- Save adversarial-validation output to a file so the shift numbers can be cited from a log.
- Reduce memory so the full pipeline fits in 32 GB (stream feature parts to the matcher).

## Repository layout

```
src/            v10 pipeline (normalise, mine, blocking, features, train, postprocess, infer, run_pipeline)
v11/            v11 matcher retraining and inference (separate from the main pipeline)
analysis/       adversarial validation, re-selection with per-country sharpening
scripts/        smoke-test sampler and scorer
results/        smoke-test logs, RESULTS.md, copies of original logs (existing_logs/)
```
