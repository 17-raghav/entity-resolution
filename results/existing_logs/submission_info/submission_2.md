# Submission 2

- Created: 2026-09-26 IST
- Git commit (D:/Brave/Amazon/submission): a1a7165 (tag submission-2)
- Leaderboard file: output/matching_results.tsv
- sha256(matching_results.tsv): ec9cb0c26302aa23bd49828c980c19c9adbce142d870c3b97ce20659c8a0a4e4
- Validator: PASS (1,732,544 rows; 104,198 empty)
- Public leaderboard F0.5: _fill in after upload_

## Changes vs submission 1
- Training/validation at test-like decoy density: 19% of train S1 dropped, their S2/S3 records kept as unlabeled distractors (test has 5.75 S2/S3 per S1 vs 4.68 in train).
- New features: legal-form relation (share / missing / conflict) + legal token counts; name tokens present on one side only (count, IDF mass, max IDF, swap flag); house-number absolute and relative gap.
- LightGBM lr 0.1, max 1000 rounds.
- Pruner K=10, tau=0.002. Selector: expected-F0.5, miss=0.0, empty_scale=1.25, one-to-one.

## Validation (test-like density, 56,853 val S1)
- Submission-1 models: 0.9720
- Submission-2 models: 0.9798
- Pruner: pair recall 97.09%, 3.98 candidates/S1

## Test output stats
- candidates/S1 4.15, matches/S1 3.26, empty frac 6.01%
- Per country matches/S1 (empty frac): US 3.29 (5.97%), India 3.26 (6.02%), France 3.18 (6.09%)
