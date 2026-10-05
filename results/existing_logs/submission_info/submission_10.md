# Submission 10

v10 = full-competition training (`BER_FULL_S1=1`): stage-1, features and pruner scores for all 1,787,544 visible train S1; matcher trained on all pruned pairs (no `--stack`). Test candidates reused from v8 (France state fix included).

- Code: commit 90bce1c (v10); AWS work dir `~/ml/work10`, log `~/ml/logs/run_v10_20260927_1019.log`
- Validation F0.5: 0.98444 (v7/v8: 0.98415), selector ef miss=0.0 empty_scale=2.0 one_to_one
- Test: 1,732,544 S1, candidates/S1 4.99, matches/S1 3.33, empty 5.92%
- vs v8 matches/S1: India 3.319 -> 3.318, US 3.373 -> 3.373, France 3.241 -> 3.274; identical sets 95.4%
- sha256(matching_results.tsv): 6de9b38d8b45b9c8fbcbe90f726faf6cd5c6e2ce91c752680f70a59bac6bf0fe
- Validator: PASS
- Public leaderboard F0.5: 0.976143 (rank 1178), up from v8 0.975
