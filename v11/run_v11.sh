#!/usr/bin/env bash
# v11: matcher retrained with clean (decoy-only) orphans; W=2 first, then W=1.
set -euo pipefail
M=~/ml
source $M/venv/bin/activate
LOG=$M/logs/run_v11_$(date +%Y%m%d_%H%M).log
cd $M/aws
echo "start $(date)" | tee -a $LOG
python v11_matcher.py 2.0 w2 2>&1 | tee -a $LOG
python v11_infer.py w2 2>&1 | tee -a $LOG
python $M/utils/validate_submission.py --matching $M/output11_w2/matching_results.tsv --candidate $M/output11_w2/candidate_pairs.tsv --test-dir $M/dataset/test 2>&1 | tail -3 | tee -a $LOG
echo "W2 DONE $(date)" | tee -a $LOG
python v11_matcher.py 1.0 w1 2>&1 | tee -a $LOG
python v11_infer.py w1 2>&1 | tee -a $LOG
python $M/utils/validate_submission.py --matching $M/output11_w1/matching_results.tsv --candidate $M/output11_w1/candidate_pairs.tsv --test-dir $M/dataset/test 2>&1 | tail -3 | tee -a $LOG
echo "ALL DONE $(date)" | tee -a $LOG
