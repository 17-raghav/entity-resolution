# Submission 5 (v5, trained and run on AWS EC2 r7i.2xlarge)

- Created: 2026-09-27 IST
- Git commit (D:/Brave/Amazon/submission): ee774dd (tag submission-5)
- Leaderboard file: output/matching_results.tsv
- sha256(matching_results.tsv): e349d803eb22841729abf7ac054db677e91d7bbd53b3f30b26c5c86a5c3b12ca
- Validator: PASS (1,732,544 rows; candidate_pairs 37,129 empty)
- Public leaderboard F0.5: _fill in after upload_
- AWS run: 2 h 48 min end to end.

## Changes vs submission 4
- Filler-word detection per country, label-free: tokens >= 8x more frequent in S2/S3 than S1 (e.g. midtown, northside, participations, holding, developpement). New name-similarity features on names without fillers (k_tset, k_tsort, k_ratio, k_j, konly*, kswap, fill_*).
- Leetspeak fixed at word start/end (6roup, 8right, denta1, capita1).
- Stage-1: top-60 + 5 reserved empty-address slots + 10 reserved address-evidence slots.
- Selector: expected-F0.5, miss=0.1, empty_scale=1.5, one-to-one.

## Validation (test-like density, same 63,334 val S1 as submissions 3-4)
- Macro F0.5: 0.9832 (sub 4: 0.9825)
- Pruner (K=10, tau=0.002): pair recall 98.27%, 4.67 candidates/S1 (sub 4: 97.44%, 3.93)

## Test output stats
- candidates/S1 4.83, matches/S1 3.29, empty frac 6.04%
- Per country matches/S1 (empty frac): US 3.36 (5.9%), India 3.31 (6.0%), France 3.07 (6.8%)
