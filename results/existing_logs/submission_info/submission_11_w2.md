# Submission 11 (w2)

v11 = v10 pipeline and test candidates; matcher retrained with test-like orphans:
- pairs whose S2/S3 record belongs to a hidden train S1 removed (noisy orphans never occur in test: test orphans are clean decoys)
- decoy negatives (records matched to no S1) weighted 2.0 (test has ~2x decoys per S1)

- Scripts: aws/v11_matcher.py, aws/v11_infer.py, aws/run_v11.sh (AWS work dir ~/ml/work11_w2)
- Validation F0.5 on clean-orphan val: 0.98550 (v10 matcher on same val: 0.98543), selector ef miss=0 empty_scale=1.0 one_to_one
- Test matches/S1: 3.336 (US 3.372, India 3.321, France 3.289)
- sha256(matching_results.tsv): 3c89590ca3f935b4401fed11cadd66ae2e42678bf15ad08eedcb4dc394d90e9a
- Validator: PASS
- Public leaderboard F0.5: _fill in_
