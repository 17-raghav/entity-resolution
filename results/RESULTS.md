# Results

Two kinds of numbers live here, and they must not be mixed:

1. **Smoke-test numbers** from runs done on 2026-10-05 on a laptop, on a small sample cut from the
   training data. They only show that the code runs end to end. They are **not** comparable to the
   full-data numbers (the sample has far fewer competing records per block, so the task is easier).
2. **Full-data numbers** from the original runs in September 2026 (laptop and AWS). They are copied
   from existing logs and notes; each one cites its source. The original notes are outside this repo
   (paths under `D:\Tech mix\Amazon\`), so some sources are listed by their original path.

`existing_logs/` holds copies of the original raw logs and the per-submission info files.

## 1. Smoke test (run now)

Machine: Intel Core Ultra 9 185H (16 cores / 22 threads), 31 GB RAM, Windows 11, Python 3.12.
Sample: 20,000 train S1 + 5,000 "test" S1, both cut from the official train split with
`scripts/make_sample.py --seed 0` (log: `smoke_make_sample.log`). Orphan S2/S3 records are added at the
full-data rate of 1.215 per S1.

| What | Value | Source |
|---|---|---|
| Pipeline wall time, train + test (v10 settings, `BER_FULL_S1=1`, top-60 + 5 + 10 slots) | 3 min 3 s | `smoke_pipeline.log:212` |
| Stage-1 recall@60 on sample validation (linear / LightGBM) | 0.9945 / 0.9957 | `smoke_pipeline.log:125` |
| Pruner pair recall at K=10, tau=0.002 (sample validation) | 0.9937 at 3.66 candidates/S1 | `smoke_pipeline.log:164` |
| Macro F0.5, 5,000 sample "test" S1, v10 pipeline | 0.9947 (repeat runs: 0.9924, 0.9940) | `smoke_score.log:2`, `smoke_repeat_2.log`, `smoke_repeat_3.log` |
| Pair precision / recall, same | 0.9995 / 0.9821 | `smoke_score.log:3` |
| Official validator on the sample output | PASS | `smoke_score.log:14` |
| Macro F0.5, same S1, v11 matcher (decoy weight 2.0) | 0.9957 | `smoke_v11.log:26` |

Why the smoke F0.5 is much higher than the full-data validation (0.984): a random 25k-entity sample
keeps only ~1% of the entities, so most same-name and same-address competitors of each entity are
absent. Treat these numbers as "it runs and produces sane output", nothing more.

Also visible in the smoke log: with only 20k S1 the train-mined state map is empty (the mining
thresholds need hundreds of examples per state), so the label-free `learn_states` fallback built state
maps for the US and India itself (`smoke_pipeline.log:13-14`). On full data it only runs for France.

## 2. Full-data version history (existing logs and notes, not re-run)

Validation = 10% of train S1 (seed 42), at test-like decoy density from v2 on. Public leaderboard =
the challenge's public score for that submission.

| Ver | Main change | Val F0.5 | Val source | Public LB | LB source |
|---|---|---|---|---|---|
| 1 | baseline pipeline | 0.9754 (0.9720 re-scored at test-like density) | `existing_logs/v1_log_matcher.txt:23`; 0.9720 from `existing_logs/submission_info/submission_2.md` | 0.963 | HANDOFF.md line 103 (see note A) |
| 2 | hide 19% of train S1 (decoy density); legal-form, word-difference, number-gap features | 0.9798 | `existing_logs/v2_log_matcher.txt:20` | 0.972 | HANDOFF.md line 104 |
| 3 | address hyphen split, edge leetspeak, Delhi alias | 0.9796 (different val set, not directly comparable) | `existing_logs/v3_log.txt:140` | 0.971 | HANDOFF.md line 105 |
| 4 | LightGBM stage-1 re-ranker, p3 typo key, ambiguity/cluster features (first AWS run) | 0.9825 | `existing_logs/submission_info/submission_4.md` (no raw log kept locally) | 0.97235 | HANDOFF.md line 106 |
| 5 | filler-aware features, top-60 + reserved slots | 0.9832 | `existing_logs/v5_run.log:318` | 0.97320 | HANDOFF.md line 107 |
| 6 | laptop run, top-40 + stacked matcher | 0.9829 | `existing_logs/v6_run.log:334` | not submitted | HANDOFF.md line 108 |
| 7 | full-name ambiguity, similarity competition, larger two-pass matcher | 0.98415 | HANDOFF.md line 109; `existing_logs/submission_info/submission_7.md` | 0.973933 | HANDOFF.md line 109 |
| P1 | v7 probabilities sharpened p^2 (stricter everywhere) | - | | 0.974 | HANDOFF.md line 110 |
| P2 | v7 sharpened p^3 for France only | - | | 0.972 | HANDOFF.md line 111 |
| 8 | v7 models + label-free France state detection | same models as v7 | | 0.975 | HANDOFF.md line 112 |
| 8_f05 | v8 + France loosened p^0.5 | - | | 0.975 | HANDOFF.md line 113 |
| 10 | full-competition training (`BER_FULL_S1=1`) | 0.98444 | `existing_logs/submission_info/submission_10.md` | 0.976143 | `existing_logs/submission_info/submission_10.md` |
| 11 (w1) | v10 + matcher retrained with clean decoy orphans, decoy weight 1.0 | 0.98572 on clean-orphan val (v10 matcher on same val: 0.98543) | `existing_logs/submission_info/submission_11_w1.md` | not recorded | - |
| 11 (w2) | same, decoy weight 2.0 | 0.98550 on clean-orphan val | `existing_logs/submission_info/submission_11_w2.md` | not recorded | - |

Note A: `D:\Tech mix\Amazon\final\Documentation_template.md` (section 5 table) lists v1 public LB as
0.9600, while HANDOFF.md says 0.963. The raw portal history is not saved locally, so the v1 LB value
is **unresolved**; do not quote it.

Note B: v11 validation uses a different validation set (hidden-entity records removed), so 0.9857 is
not comparable with v10's 0.98444. Compare only within the same row (v11 vs v10 matcher on the same
set: +0.0003 for w1).

Other full-data figures used in the README:

| Figure | Value | Source |
|---|---|---|
| Test S1 entities | 1,732,544 | `existing_logs/v5_run.log:409`; `existing_logs/submission_info/submission_1.md` |
| Train S1 / S2+S3 records | 2,206,821 / 10,320,219 | HANDOFF.md lines 30-33 |
| Test S2+S3 records | 9,969,589 | HANDOFF.md line 33 |
| S2/S3 records per S1, train vs test | 4.68 vs 5.75 | HANDOFF.md line 34 |
| Stage-1 recall@60, LightGBM vs linear (v5 val) | 0.9818 vs 0.9762 | `existing_logs/v5_run.log:188` |
| Pruner pair recall at K=10, tau=0.002 (v5 val) | 0.9827 at 4.67 candidates/S1 | `existing_logs/v5_run.log:287` |
| Test candidates per S1 (v5) | 4.83 | `existing_logs/v5_run.log:409` |
| Oracle F0.5 on the candidate set (val) | ~0.994 | HANDOFF.md line 119 (no raw log) |
| Validation loss breakdown (v6) | 60% partial misses, 22% wrongly empty, 11% false merges, 5% singletons given a match | HANDOFF.md line 120 (no raw log) |
| AWS run time end to end, v4 / v5 | 1 h 53 min / 2 h 48 min (8 vCPU, 64 GB) | `existing_logs/submission_info/submission_4.md`, `submission_5.md` |

## 3. Adversarial validation (existing notes only)

Script: `analysis/adv.py` (ran on AWS; its printed output was not saved to a local file).
The findings below come from the written summary in HANDOFF.md lines 122-129:

- Biggest shift between test and validation candidates: the competition features. `n_cand23` (how many
  S1 entities claim the same S2/S3 record at stage 1) averaged about 10 on validation vs 15.5 (test US),
  20 (test India) and 16.4 (test France).
- Diagnosed cause: pruner and matcher were trained on candidates for about one third of the states
  (`train_subset`, ~707k S1) while test has all S1 present, so the model saw more competition at test
  time and under-trusted true matches.
- Probe `diag_france_empty` (France rows blanked, LB 0.839) implied France ~0.964 and US+India ~0.977
  (HANDOFF.md line 114; method in `existing_logs/submission_info/diag_france_empty.md`).

The AUC values of the adversarial classifiers are **not** recorded anywhere locally, so they are not quoted.

## 4. Not re-run

The full pipeline (train + test on 2.2M + 1.7M S1) was not re-run: it needs ~64 GB RAM and about
2-3 hours on 8 vCPU (see section 2). All full-data figures above are from existing logs/notes.
