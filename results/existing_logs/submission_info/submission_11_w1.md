# Submission 11 (w1)

v11 = v10 pipeline and test candidates; matcher retrained with test-like orphans:
- pairs whose S2/S3 record belongs to a hidden train S1 removed (noisy orphans never occur in test: test orphans are clean decoys)
- decoy negatives (records matched to no S1) weighted 1.0 (test has ~2x decoys per S1)

- Scripts: aws/v11_matcher.py, aws/v11_infer.py, aws/run_v11.sh (AWS work dir ~/ml/work11_w1)
- Validation F0.5 on clean-orphan val: 0.98572 (v10 matcher on same val: 0.98543), selector ef miss=0 empty_scale=1.0 one_to_one
- Test matches/S1: 3.349 (US 3.388, India 3.333, France 3.300)
- sha256(matching_results.tsv): 1f1c0a9fcde1a965b38e89b86f5f3a343498e4b3c1ef5c5a4242ce80a1582126
- Validator: PASS
- Public leaderboard F0.5: _fill in_
