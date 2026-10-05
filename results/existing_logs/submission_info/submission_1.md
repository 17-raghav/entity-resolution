# Submission 1

- Created: 2026-09-26 09:43 IST
- Git commit (D:/Brave/Amazon/submission): 63be003
- Leaderboard file: output/matching_results.tsv
- sha256(matching_results.tsv): fc95d49f46c45435d9fa05567864ef8f1ee0ef3f157cef27489b770ec538d9f3
- Validator: PASS (1,732,544 rows; 97,313 empty)
- Public leaderboard F0.5: _fill in after upload_

## Validation (held-out train S1, 70,254 entities in geo subset)
- Macro F0.5: 0.9754 (India 0.9741, US 0.9771, singletons 0.9684)
- Selector: expected-F0.5, miss=0.1, empty_scale=0.8, one-to-one on
- Pruner: K=8, tau=0.01 -> pair recall 96.94%, 3.94 candidates/S1 (val)

## Test output stats
- candidates/S1 4.25, matches/S1 3.33, empty frac 5.62%
- Per country matches/S1 (empty frac): US 3.34 (5.7%), India 3.31 (5.7%), France 3.35 (5.2%)

## Contents
- output/   matching_results.tsv, candidate_pairs.tsv
- code/     snapshot of code/business_entity_resolution (includes scratch _*.py)
- models/   LightGBM pruner/matcher, feature lists, params, mined lexicons, val predictions
- logs/     run logs
- notes.md  working notes and v2 ideas
