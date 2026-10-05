# Study guide: entity-resolution

Written so you can defend the project starting from zero. Numbers quoted here are the ones in
`CLAIMS.md`; check the status column there before saying any of them out loud.

---

## 1. The problem and every component, built up step by step

### 1.1 Entity resolution

Entity resolution = deciding which records refer to the same real-world thing. Here: S1 is a clean
list of businesses; S2 and S3 contain noisy copies of those businesses plus unrelated records. For
each S1 entity we output the list of S2/S3 record IDs that are the same business.

Key fact from training labels: every S2/S3 record matches **at most one** S1. About 26% match nothing
(orphans), about 5.6% of S1 have no match at all (singletons). On average an S1 has 3.46 matches.

### 1.2 The metric: macro F0.5

For one S1 entity with `n_true` true matches, `n_pred` predictions, `tp` correct predictions:

```
F0.5 = 1.25 * tp / (0.25 * n_true + n_pred)
```

F-beta with beta = 0.5 weights precision more than recall. Average over all S1 ("macro").
Edge cases: no truth and no prediction = 1.0; no truth but any prediction = 0; truth but empty
prediction = 0. Consequence: for an uncertain entity, predicting nothing can be the right move.

Worked example (also the scorer self-test in `src/evaluate.py`): truth {47, 812}, prediction
{47, 193, 812}: tp = 2, F = 1.25*2 / (0.5 + 3) = 0.714.

### 1.3 Why you cannot compare all pairs: blocking

Test has 1.7M S1 and 10M S2/S3. All pairs within a country is around 10^12, impossible to score.
**Blocking** creates cheap keys per record (e.g. "US | name token 'lakota'") and only compares
records that share a key. Each shared key adds evidence; rare keys are worth more (IDF weighting:
`log(N / df)`, where df = how many S2/S3 records have the key). Keys shared by more than 1,500 S2/S3
or 500 S1 records are dropped: they create huge blocks and say little.

Ten key families (`src/blocking.py`, `FAMS`):
- `t` single name token; `tt` pair of name tokens;
- `ta` name token x address word; `na` house number x address word;
- `i` alphanumeric address identifier (`9/e16`);
- `s` name with spaces removed; `s8` its first 8 characters (catches domain names);
- `f` full name including legal form;
- `ab` adjacent address-word pair;
- `p3` sorted 3-letter prefixes of name tokens (typo tolerant: `bollywood` ~ `bolyltod`).

Every key starts with the country string, so pairs are only made inside a country. This is the only
use of the country field, which is why an unseen country (France) still works.

### 1.4 Stage-1 ranking

A pair now has 10 numbers (summed IDF per family). Two rankers cut the list:
1. A logistic regression over log-weights and "share of this S1's total weight" per family keeps the
   200 best candidates per S1 (cheap, linear).
2. A LightGBM re-ranker over those 200 keeps the top 60.
3. Reserved slots: 5 extra for the best empty-address records and 10 for the best address-only
   evidence (house number x street, identifiers, street bigrams). Without them, an S1 with a common
   name gets its 60 slots filled by same-name records in other places, and the right record with an
   empty address or a rebranded name falls off.

Measured on full validation (v5): recall@60 0.9818 with LightGBM vs 0.9762 linear.

### 1.5 Normalisation

Before keys, every name and address is normalised the same way for all sources (`src/normalize.py`):
- lower-case; Unicode NFKD and remove accent marks (`é` -> `e`);
- leetspeak repair: a digit between letters (`y0uth`), or glued to a word start/end (`6roup`,
  `denta1`), becomes the letter it imitates. Only for words of 3+ letters so `5th`, `1st` survive;
- legal forms mapped to one spelling (`limited` -> `ltd`, `(p)` -> `pvt`, French SARL/SAS/...);
- `core` = name tokens minus legal forms and generic filler words; `name_ns` = core without spaces;
- domain / handle detection (`gouveiaentertainment.com`, `@novarioak`); later segmented into words
  by dynamic programming over a unigram vocabulary (`blocking.segment`);
- Indic scripts: tokens are mapped with a lexicon mined from matched training pairs; unknown tokens
  are transliterated by rule (`indic_transliteration` to IAST, then ASCII, plus Hindi final-schwa
  deletion);
- addresses: abbreviations (`street` -> `st`, `chemin` -> `chem`), hyphen between letters split
  (`saint-herblain`), number tokens without leading zeros, identifiers kept separately, state part
  detected and removed from the address tokens.

### 1.6 Tables mined from training data (no external data allowed)

`src/mine_lexicons.py`, on the training part only:
- Indic lexicon: for matched pairs where the S2/S3 name is in an Indic script, align Indic tokens to
  Latin tokens of the S1 name (by position if counts match, else by consonant-skeleton similarity).
  Keep a mapping seen at least twice with at least 50% agreement.
- State map: S1 address parts that are usually the last part are states; S2/S3 parts that co-occur
  with one S1 state in >= 90% of matched pairs are its aliases (`mh` -> `maharashtra`).
- Generic words: tokens that appear in S2/S3 names but not in the matched S1 name >= 90% of the time.

For countries without a mined state map, `learn_states` works label-free: a state-like part is very
frequent and is the last part >= 60% of the time; departement -> region aliases come from pairs that
share the exact no-space name and house number (almost surely the same business, no label needed).

### 1.7 Pair features

About 140 features per pair (`src/features.py`), all numbers:
- string similarities (rapidfuzz ratio / token-sort / token-set / partial, Jaro-Winkler,
  Levenshtein) on full name, core name, segmented name, no-space name, address;
- IDF mass of tokens that appear on only one side (a missing rare word is a strong "different");
- the same computed after removing **filler words** (tokens >= 8x more frequent in S2/S3 than in S1,
  found per country without labels). This separates "noise word added" from "real word swapped";
- legal-form relation (same / one missing / conflict), house-number relation (equal / truncated /
  conflict) and numeric gap, state relation, empty-address flags;
- ambiguity/density counts: how many S1 and S2/S3 records share this name or this address;
- context: rank and gap of this pair within the S1's candidates, and **competition** features: how
  many S1 entities claim the same S2/S3 record and how this S1 ranks among them;
- cluster features: similarity of the candidate to records already confidently matched to this S1
  (catches a rebrand at the same address).

No feature uses raw tokens or the country label. That is what lets the model transfer to France.

### 1.8 Pruner and matcher

Both are LightGBM binary classifiers (gradient-boosted trees).
- **Pruner** (127 leaves): trained 2-fold out-of-fold so that the probabilities used downstream on
  training pairs come from a model that did not see those pairs. Keeps up to 10 candidates per S1 with
  p >= 0.002. Result (v5 validation): 98.27% of true pairs kept at 4.67 candidates per S1.
- **Matcher** (255 leaves, learning rate 0.03): sees pruner probability plus context computed within
  the pruned set. Optional second pass (`--stack`, used in v5-v7): re-score using first-pass
  probabilities of the S1's other candidates and of competing S1 for the same record.

### 1.9 Selection

`src/postprocess.py`:
1. One-to-one: a record claimed by several S1 keeps only the most probable S1.
2. Expected-F0.5 set selection: sort candidates by p. For each prefix of length k, approximate
   `E[F_k] = 1.25 * sum_{i<=k} p_i / (0.25 * (sum_i p_i + miss) + k)`. Compare with predicting empty,
   whose expected score is `prod(1 - p_i) * empty_scale`. Keep the best. `miss` (expected true matches
   outside the list) and `empty_scale` are tuned on validation.

### 1.10 Validation design

- 10% of train S1 (seed 42) held out.
- **Decoy density**: test has 5.75 S2/S3 per S1 vs 4.68 in train, meaning more orphans in test. Hiding
  19% of train S1 (by a hash of the ID) turns their records into orphans. This reproduced two test
  statistics (records per S1, and share of S2/S3 names that occur in S1). It moved v1 validation from
  0.9754 to 0.9720, i.e. validation became harder and more honest.
- Hidden S1 must also be invisible to every S1-side count (v9), as orphan entities are in test.

---

## 2. Code walkthrough in execution order

Entry point: `src/run_pipeline.py`. It runs each step as a subprocess with `cwd=src/` and writes a
marker file in `<work>/done/` so a crashed run can resume.

1. `config.py` - paths from env (`BER_DATA`, `BER_WORK`), seed 42, stage sizes, `DROP_PCT = 19`,
   `FULL_S1` flag, `VAL_FRAC = 0.10`.
2. `io_utils.py` - TSV -> parquet with int32 `idx`. S2 and S3 are concatenated into one table `s23`
   with a `src` column. Ground truth exploded to one row per (s1, s23) pair. Also the only TSV writer
   (UTF-8, tab, no quoting).
3. `evaluate.py` - scorer self-test, then the validation split (`train_split.parquet`).
4. `normalize.py --split train` (pass 1, no tables yet).
5. `mine_lexicons.py` - Indic lexicon, state map, generic-word candidates.
6. `normalize.py --split train` (pass 2, tables applied).
7. `blocking.py --split train --keys-only` - domain segmentation, keys for S1 and S23.
8. `train.py ranker` - logistic stage-1 ranker on 20k train S1.
9. `train.py lgbranker` - LightGBM stage-1 re-ranker; prints recall@20/30/40/60 linear vs LightGBM.
10. `blocking.py --split train` - stage-1 candidates (top-60 + slots). With `FULL_S1` for all visible
    S1, otherwise for `train_subset()` (whole states, about one third).
11. `train.py feats --drop` - hide 19% of S1, build features in chunks of 100k S1, write parts.
12. `train.py pruner` - 2-fold OOF pruner + full pruner, K/tau grid printed, with `FULL_S1` also scores
    the S1 outside the pruner subset (`score_rest`).
13. `train.py matcher --k 10 --tau 0.002` - prune, add context and cluster features, fit matcher,
    optional stack, grid over selectors, save `select_params.json`.
14. `normalize.py --split test`, `blocking.py --split test` - same for test (here `learn_states`
    runs for France).
15. `infer.py` - chunked: features -> pruner -> keep -> `candidate_pairs.tsv`; then context ->
    matcher -> one-to-one -> selection -> `matching_results.tsv`.

Outside the main flow:
- `v11/v11_matcher.py` - reads v10's `train_scored.parquet`, removes pairs whose S2/S3 record belongs
  to a hidden S1, weights decoy negatives by W, retrains the matcher, compares it with the v10 matcher
  on the same validation pairs, saves the best selector.
- `v11/v11_infer.py` - builds pruned test pairs once (cached) and runs any v11 matcher on them.
- `analysis/adv.py` - adversarial validation (section 1 of README diagnosis).
- `analysis/reselect.py` - `p' = p^gamma` per country, then selection again (the probes).
- `scripts/make_sample.py`, `scripts/score.py` - smoke test only.

---

## 3. Why this, not that

| Choice | Why | Alternative and why not |
|---|---|---|
| LightGBM | fast on 10M+ rows of numeric features, handles mixed scales and missing values, MIT licence (rules: model must be MIT/Apache, <= 8B params) | a transformer cross-encoder on name pairs: far too slow for ~8M pairs x many runs on CPU in 72 h; could not use external pretrained data anyway |
| Compound-key blocking | full control of recall per noise type; keys are cheap joins | embedding ANN search: needs a trained encoder and GPU; hard to explain recall losses |
| Two-step stage-1 (linear 200 -> GBM 60) | the GBM is too slow on all raw pairs, the linear model is not accurate enough alone | one big model on all raw pairs: memory |
| Reserved slots | specific failure: same-name crowding pushes out empty-address and rebranded records | larger top-K: grows every list for a few cases |
| Out-of-fold pruner | downstream features use pruner probabilities; in-sample probabilities would be over-confident | single model: train/test mismatch for `p_prune` |
| Expected-F0.5 selection | the metric is per entity and precision-weighted; the best set size depends on the whole list | a fixed threshold: was in the grid and lost on validation |
| One-to-one | true in training data for every record | none needed; it is a data fact |
| Hiding 19% of S1 | reproduce test orphan density so validation predicts the leaderboard | random negatives: would not match the test's decoy structure |
| Label-free statistics for France | no French labels exist; rules from frequency stats transfer | hand-written French rules: brittle; one try (stricter France) lowered the score |
| Features without country or raw tokens | transfer to unseen country | country one-hot: useless for France |

---

## 4. What failed or was dropped (from notes and logs)

- **v3 normalisation (hyphen split, Delhi):** validation flat (0.9796 vs 0.9798, different val sets),
  leaderboard dropped 0.972 -> 0.971, France matches per S1 fell 3.18 -> 3.01.
- **Stricter selection probes:** P1 (p^2 everywhere) did not move the leaderboard (0.974); P2 (p^3 on
  France) dropped it to 0.972. Lesson: the model was under-matching, not over-matching.
- **v8_f05 (France loosened p^0.5):** no gain over v8 (both 0.975).
- **v6 laptop run (top-40 + stacked matcher):** validation 0.9829, not better than v5's 0.9832 on AWS;
  never submitted. Laptop memory was also a problem (segmentation pool spawned many workers; pruner
  training paged).
- **v9 (S1-side counts exclude hidden S1):** stopped and folded into v10.
- **v10 full-competition fix:** right direction (+0.0011 on leaderboard) but closed only a small part
  of the ~0.009 gap.
- **v11 clean-orphan matcher:** small validation gain (+0.0003 on its own validation set), leaderboard
  not recorded.
- Engineering gotchas recorded in notes: a Bash heredoc turned `\b` into a backspace character inside
  `normalize.py` twice; polars `fill_null(0)` after a left join turned missing ranks into rank 0;
  feature names ending in `1`/`2` get dropped by `features.build`.

---

## 5. Likely interview questions

1. **What is the task in one sentence?** For each clean reference business, find all noisy records
   in two other sources that are the same business, possibly none.
2. **Why F0.5 and what does it change?** Precision counts more. It pushes towards smaller, surer
   sets and makes "predict nothing" correct for uncertain singletons.
3. **How do you avoid O(N^2)?** Blocking keys + IDF + caps on block size, then two rankers. Test ends
   with 4.83 candidates per S1 (v5).
4. **What is your candidate recall?** 98.27% of true pairs kept after the pruner, at 4.67 candidates
   per S1 (v5 validation). Stage-1 recall@60 0.9818.
5. **Why is stage-1 two models?** Linear is cheap enough for all raw pairs; LightGBM is better but
   only affordable on a 200-shortlist. LightGBM raised recall@60 from 0.9762 to 0.9818.
6. **What are the reserved slots for?** Empty-address and rebranded records that lose to many
   same-name records; 15 extra slots per S1.
7. **How do you handle Indic scripts?** Mined lexicon from matched training pairs, rule-based
   transliteration fallback, Hindi schwa deletion; all S1 names are Latin.
8. **How does France work without training data?** All features are country-free similarities and
   counts; filler words and state/region aliases are learned label-free from frequency statistics on
   the test files themselves.
9. **Isn't learning from test files leakage?** No labels are used; the statistics are unsupervised
   (token frequency ratio, position of address parts, exact name+number co-occurrence). The
   challenge rule was "no external data or APIs"; this is transductive use of the provided test
   inputs, and it is disclosed in the documentation. If an interviewer considers that borderline,
   agree it is a design choice and say it can be switched off (France then has no state map).
10. **Why LightGBM and not a deep model?** Tabular similarity features, 10M+ rows, CPU, 72 hours,
    licence and size rules. Trees were the practical choice.
11. **Why out-of-fold for the pruner?** Its probability is a feature downstream; in-sample
    probabilities on training pairs would be over-confident and the matcher would learn the wrong
    trust level.
12. **Explain expected-F0.5 selection.** Sort by p, estimate expected F0.5 for each prefix with a
    ratio of expectations, compare with the probability that the true set is empty, choose the best.
13. **Why one-to-one?** In training every S2/S3 record matches at most one S1, so a record claimed
    by two S1 entities has at least one wrong claim.
14. **Your validation was 0.984 and the leaderboard 0.975. Why?** Training saw less competition than
    test: matcher/pruner trained on a third of the states, test has all entities present. Adversarial
    validation showed `n_cand23` ~10 vs 15.5-20.
15. **How did you rule out over-matching?** Stricter probes did not help (p^2) or hurt (p^3 France).
16. **How did you know the gap was not only France?** A submission with France blank scored 0.839;
    with France = 15% of S1 this implies France ~0.964 and US+India ~0.977, still well below 0.984.
17. **Follow-up: is that calculation exact?** No. It assumes blank France rows score only the France
    singletons (~5.6% of France scores 1) and that per-country scores are additive by S1 share. Treat
    it as an estimate.
18. **What is adversarial validation?** Train a classifier to separate test samples from validation
    samples. If it can, the distributions differ; its important features show where.
19. **Did your fix work?** Validation 0.98415 -> 0.98444, leaderboard 0.975 -> 0.976143. Right
    direction, small; the rest of the gap was not explained.
20. **What was v11?** Hiding S1 makes their own noisy records into orphans, but in test the extra
    orphans look like clean decoys. v11 drops those noisy orphans from training and up-weights clean
    decoy negatives. +0.0003 on validation, no leaderboard score recorded.
21. **What is the biggest remaining error type?** On v6 validation: 60% partial misses, 22% entities
    predicted empty that have matches (from notes). Truly ambiguous same-name, no-address records
    cannot be solved by strings.
22. **How much compute?** Full run about 2-3 h on 8 vCPU / 64 GB (AWS r7i.2xlarge). Laptop: 31 GB was
    not enough for comfortable training.
23. **Could you reproduce it today?** The code runs end to end on a 25k-entity sample in 3 minutes
    (smoke test). The full run needs the 64 GB machine; I did not re-run it for this repo.
24. **Hard follow-up: your smoke-test F0.5 is 0.995, higher than the full run. Why?** The sample keeps
    about 1% of entities, so most competitors and decoys are gone. It is a functional test only.
25. **Hard follow-up: what would you do with one more week?** Importance-weight training pairs by the
    adversarial classifier's density ratio; train the matcher on the full competition set with lower
    memory; save every diagnostic output to a log so the numbers can be audited.
