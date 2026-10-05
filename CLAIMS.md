# Claims

Every number that could go on a CV, with its exact source and status.

Status:
- **verified now**: produced by a run on 2026-10-05, log in `results/`.
- **existing log**: in a raw log written by the original run (copied to `results/existing_logs/`).
- **existing note**: only in a written summary (HANDOFF.md, SUBMISSION_INFO.md), no raw log. Usable, but
  say "recorded at the time" if asked.
- **unverified**: do not use.

`HANDOFF.md` = `D:\Tech mix\Amazon\HANDOFF.md` (not copied: it contains infrastructure details).

| Claim | Value | Source | Status |
|---|---|---|---|
| Public leaderboard F0.5, best recorded (v10) | 0.976143 | `results/existing_logs/submission_info/submission_10.md` line 11 | existing note |
| Validation macro F0.5, v10 | 0.98444 | `results/existing_logs/submission_info/submission_10.md` line 6 | existing note |
| Validation macro F0.5, v7 | 0.98415 | HANDOFF.md line 109; `submission_info/submission_7.md` line 3 | existing note |
| Validation macro F0.5, v5 | 0.9832 | `results/existing_logs/v5_run.log:318` | existing log |
| Validation macro F0.5, v2 | 0.9798 | `results/existing_logs/v2_log_matcher.txt:20` | existing log |
| Validation macro F0.5, v1 | 0.9754 | `results/existing_logs/v1_log_matcher.txt:23` | existing log |
| Public LB v2 / v5 / v7 / v8 | 0.972 / 0.97320 / 0.973933 / 0.975 | HANDOFF.md lines 104, 107, 109, 112 | existing note |
| Public LB v1 | 0.963 or 0.9600 | HANDOFF.md line 103 vs `D:\Tech mix\Amazon\final\Documentation_template.md` section 5 | **unverified (conflict)** |
| Val-LB gap ~0.009 (v7/v8) | 0.98415 vs 0.975 | rows above | existing note |
| France ~0.964, US+India ~0.977 (probe) | derived from LB 0.839 probe | HANDOFF.md line 114 | existing note |
| `n_cand23` shift: val ~10 vs test 15.5 / 20 / 16.4 (US / India / France) | | HANDOFF.md line 125 | existing note |
| Adversarial classifier AUC | - | not saved | **unverified** |
| v10 fix: LB 0.975 -> 0.976143 | | HANDOFF.md line 112; `submission_info/submission_10.md` line 11 | existing note |
| v11 matcher: val 0.98572 vs v10 matcher 0.98543 on same clean-orphan val | | `submission_info/submission_11_w1.md` line 8 | existing note |
| v11 public LB | - | never recorded | **unverified** |
| Test scale: 1,732,544 S1 | | `results/existing_logs/v5_run.log:409` | existing log |
| Test S2+S3 records: 9,969,589; train S1 2,206,821, S2+S3 10,320,219 | | HANDOFF.md lines 32-33 | existing note |
| "~12 million records per split" (README wording) | 2.2M + 10.3M train; 1.7M + 10.0M test | HANDOFF.md lines 32-33 | existing note |
| Stage-1 recall@60: LightGBM 0.9818 vs linear 0.9762 | | `results/existing_logs/v5_run.log:188` | existing log |
| Pruner pair recall 98.27% at 4.67 candidates per S1 | | `results/existing_logs/v5_run.log:287` | existing log |
| Test candidates per S1: 4.83 | | `results/existing_logs/v5_run.log:409` | existing log |
| ~140 pair features | | HANDOFF.md line 85; Documentation_template.md section 4 | existing note (count not re-checked) |
| Ten blocking key families | t, tt, ta, na, i, s, s8, f, ab, p3 | `src/blocking.py` line 116 (`FAMS`) | verified now (code) |
| Nine Indic scripts handled | | `src/normalize.py` lines 76-80 (`SCRIPTS`) | verified now (code) |
| Decoy density: 19% of train S1 hidden; 4.68 vs 5.75 S2/S3 per S1 | | `src/config.py` (`DROP_PCT = 19`); HANDOFF.md line 34 | code + existing note |
| AWS run time 1 h 53 min (v4), 2 h 48 min (v5) on 8 vCPU / 64 GB | | `submission_info/submission_4.md` line 9, `submission_5.md` line 9 | existing note |
| Oracle F0.5 on candidate set ~0.994 | | HANDOFF.md line 119 | existing note (no raw log) |
| Smoke test: end to end in 3 min 3 s on a laptop | | `results/smoke_pipeline.log:212` | verified now |
| Smoke test macro F0.5 0.9947 / 0.9924 / 0.9940 over 3 identical runs (5k sample S1) | not comparable to full data; runs are not bit-reproducible | `results/smoke_score.log:2`, `results/smoke_repeat_2.log`, `results/smoke_repeat_3.log` | verified now (do not put on CV as a score) |
| Validator PASS on full test output (v5) | | `results/existing_logs/v5_validator.txt:8` | existing log |
| Leaderboard rank | - | | not used (deliberately) |
