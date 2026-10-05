# Submission 3

- Created: 2026-09-26 IST
- Git commit (D:/Brave/Amazon/submission): c67500a (tag submission-3)
- Leaderboard file: output/matching_results.tsv
- sha256(matching_results.tsv): c51bf06950df44247036aa465564ef29f661ac954e666577b7c380229f187e5b
- Validator: PASS (1,732,544 rows; 104,846 empty)
- Public leaderboard F0.5: _fill in after upload_

## Changes vs submission 2 (step C, normalisation)
- Address: hyphen between letters split into words (saint-herblain, la teste-de-buch, lege-cap-ferret).
- Names: leetspeak digit at word start fixed (5ervices -> services, 0wens -> owens); overlapping mid-word leet (d3v3lop).
- State mining threshold 0.8 -> 0.6: adds India delhi / dilli / dl -> delhi.
- Full rebuild: normalisation, keys, stage-1 ranker, features, pruner, matcher.
- Pruner K=10, tau=0.002. Selector: expected-F0.5, miss=0.3, empty_scale=0.8, one-to-one.

## Validation (test-like density, 63,334 val S1; set differs from sub 2 because Delhi S1 now have a state)
- Macro F0.5: 0.9796 (sub 2: 0.9798 on its own 56,853-S1 set; not directly comparable)
- Pruner: pair recall 96.94%, 3.96 candidates/S1

## Test output stats
- candidates/S1 4.04, matches/S1 3.24, empty frac 6.05%
- Per country matches/S1 (empty frac): US 3.29 (5.9%), India 3.27 (5.9%), France 3.01 (6.7%)
- Note: France matches/S1 fell 3.18 -> 3.01 vs sub 2 (hyphen change mostly hits French addresses). Leaderboard will show if precision or recall moved.
