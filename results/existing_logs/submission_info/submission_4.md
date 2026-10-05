# Submission 4 (v4, trained and run on AWS EC2 r7i.2xlarge, ap-south-1)

- Created: 2026-09-27 IST
- Git commit (D:/Brave/Amazon/submission): 62f7ae7 (tag submission-4)
- Leaderboard file: output/matching_results.tsv
- sha256(matching_results.tsv): 02eda937323e0e3473e47e3c3f64a007005ebd510ce1993c472885e00073d0f0
- Validator: PASS (1,732,544 rows; 104,433 empty)
- Public leaderboard F0.5: _fill in after upload_
- AWS run: 1 h 53 min end to end (train + test) on 8 vCPU / 64 GB.

## Changes vs submission 3
- Stage-1: linear ranker keeps top-200 per S1, LightGBM re-ranker picks top-40 (recall@40 97.29% -> 97.68%).
- New typo-tolerant blocking key p3 (sorted 3-letter prefixes of name tokens).
- Matcher features: name ambiguity (#S1 / #S2+S3 sharing name), address density (#S1 / #S2+S3 at same address), made-up-brand tokens (unseen in S1 names), cluster support (similarity to the S1's confident candidates).
- Thread count follows the machine; pipeline resumable per step.
- Pruner K=10, tau=0.002. Selector (auto from validation): expected-F0.5, miss=0.0, empty_scale=1.25, one-to-one.

## Validation (test-like density, same 63,334 val S1 as submission 3)
- Macro F0.5: 0.9825 (submission 3: 0.9796)
- Pruner: pair recall 97.44%, 3.93 candidates/S1 (submission 3: 96.94%, 3.96)

## Test output stats
- candidates/S1 4.24, matches/S1 3.29, empty frac 6.03%
- Per country matches/S1 (empty frac): US 3.36 (5.9%), India 3.30 (6.0%), France 3.08 (6.7%)
