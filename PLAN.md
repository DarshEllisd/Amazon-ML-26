# Amazon ML Challenge 2026: Business Entity Resolution Plan

## 0. What the data actually looks like (EDA done 2026-09-25)

| | S1 (reference) | S2 | S3 |
|---|---|---|---|
| Train rows | 2,206,821 (US 1.32M, India 0.88M) | 5,034,616 | 5,285,603 |
| Test rows | 1,732,544 (India 0.81M, US 0.66M, **France 0.26M**) | 4,887,273 (France 0.70M) | 5,082,316 (France 0.73M) |
| Empty address | 0% | ~3% | ~3% |
| Non-ASCII names | ~2% | 15–19% | 11–15% |

These facts change the design:

1. **Only 5.6% of S1 entities are singletons** (the same rate in US and India). Most entities have 2–6 matches (mean ≈ 3.5, max 11). The talk about singletons in the video matters less than recall on multi-match entities. Precision still dominates because of F0.5.
2. **Each S2/S3 record matches at most one S1 entity** (7,638,365 matched IDs, all unique). This is a hard constraint: every S2/S3 record goes to its single best S1, or to none. The problem is really *assignment*, not independent pairwise matching.
3. **26% of S2/S3 records match nothing.** They are distractors, so a record needs a "reject" option too.
4. **Matched pairs never cross countries** (0% mismatch). Always block within country.
5. **Blocking-key coverage on true pairs** (60k-entity sample):
   - shares a non-stopword name token: 85% (US 92%, **India 75%**)
   - shares a number with 3 or more digits in the address: 57% (India only 42%)
   - shares 2 or more address tokens: 95%
   - **the union of the three: 99.94%.** The recall ceiling from cheap keys is excellent if they are combined.
6. **Noise patterns seen** (synthetic generator, applied the same way across sources):
   - Names:
     - junk prefixes/suffixes (`--`, `>>`, `<<`, `...`, `#`, `@`, `[..]`, `(..)`)
     - legal-suffix add/drop/reorder (`LLC K+ SIM`, `private uttar pradesh feel limited`)
     - OCR/leet typos (`M0DERN`, `BANC5HARES`), random accent injection (`Múzero`, `Gríll`)
     - duplicated tokens (`Fitness Fitness`), truncation (`Cornerstone Service`, `Modern`)
     - domain-name forms (`martinezpacificsolutions.com`), `f/k/a` aliases, store numbers (`#66894`)
     - **native-script names** (Devanagari, Tamil, Kannada, Telugu): 24% of Indian matches, 7% of US
     - occasional completely unrelated names (`Avivantage`, `Evokelokor`) that can only be matched by address
   - Addresses:
     - uppercase vs. title case; `Rd/Road`, `St/Street/Saint`, `Dr/Drive`, `Ct/Court`
     - US state code vs. full name; Indian state in native script (`ಕರ್ನಾಟಕ`, `दिल्ली`) or code (`KA`, `TN`, `MH`, `UP`)
     - zero-padded (`003778`) or suffixed (`134-B`, `1271B`) house numbers, `#`/`H.no`/`Door No` prefixes
     - component reordering; missing house number or street; city typos (`ARLINGTN`, `GERENSBORO`)
     - neighbouring city or township names (`Orcutt` vs. `Santa Maria`), `PMB`/`PO Box` insertions
7. **France is test-only (15% of test S1).** It uses the same noise style plus French patterns:
   - `R.`/`RUE`, `AV`, `BD`, `bis`
   - legal forms `SARL/SAS/SASU/EURL/SCI/Ets`
   - région vs. département swaps (`Hauts-de-France` ↔ `Nord`, `Nouvelle-Aquitaine` ↔ `Gironde`, `Pays de la Loire` ↔ `Loire-Atlantique`)
   - accents (`Élève`/`Eleves`), typos (`JEN ZAY`, `PRESIDET`)

   Features must be **country-agnostic** so the matcher transfers.

**Hardware:** 16 GB RAM, 12 cores, RTX 4050 (6 GB), Python 3.9 with pandas/sklearn. Memory is the binding constraint: the full S2+S3 test set is ~10M rows and ~1 GB of raw text. **Process everything partitioned by country → state/region.**

---

## 1. Project layout

```
code/business_entity_resolution/
  src/
    io.py            # tab-safe loaders (sep="\t", quoting=3, dtype=str, keep_default_na=False) → parquet cache
    normalize.py     # name + address normalisers (country-aware but open-set)
    translit.py      # native-script → Latin (rule-based + learned mapping)
    blocking.py      # multi-pass candidate generation, writes candidate_pairs
    features.py      # pairwise + group features
    train.py         # LightGBM matcher (+ optional cross-encoder)
    assign.py        # S2/S3 → single S1 assignment + per-entity set selection
    evaluate.py      # macro F0.5, blocking recall / reduction ratio
    run_pipeline.py  # data → blocking → matching → output/*.tsv
  README.md, requirements.txt (pinned)
output/matching_results.tsv, output/candidate_pairs.tsv
Documentation_template.md (filled in)
```

Steps 0 and 1 below convert the TSVs to parquet once; loading the raw TSVs takes ~4.5 min per pass. Add `polars` or `pyarrow` for speed, `rapidfuzz` (MIT) for string similarity, `lightgbm` (MIT), and `faiss-cpu`/`faiss-gpu` (MIT) or `sparse_dot_topn` (MIT) for nearest-neighbour search.

---

## 2. Validation protocol (build this first)

- **Evaluation function:** exact macro F0.5 per S1 entity, singletons included: an empty prediction for a singleton scores 1, any prediction scores 0. Unit-test it against the README example (0.714).
- **Split: geographic hold-out, not random.** Hold out whole states/cities (e.g. 10% of US states and 10% of Indian states) with **all** their S1, S2 and S3 records, distractors included. Candidate density and distractor rates then match test conditions. A random S1 split leaks the "look-alike neighbours" structure.
- **France proxy:** *leave-one-country-out* runs (train on US → eval on India, and the reverse). This measures how well features and normalisation transfer to an unseen country. Use it to choose between country-specific and generic features.
- **Tracked metrics:** blocking recall (share of true pairs in candidates), share of entities whose full match set is in candidates, candidates per S1 entity, reduction ratio, and final macro F0.5 by country.
- Keep a small dev subset (e.g. 3 states, ~100k S1) for fast iteration, and the full hold-out for final numbers.

---

## 3. Normalisation (the largest lever)

Keep both the **raw** and **normalised** strings; the raw form feeds some features.

**Names**
1. Unicode NFKC, lowercase, strip accents (`unidecode`-style fold, keeping a copy that retains accents).
2. Strip junk wrappers: leading/trailing `-- >> << ... # @ * ~`, and unwrap `[..]`/`(..)`.
3. Split domains: `foobar.com` → `foobar`, then **word-segment** against a vocabulary built from S1 names (dynamic-programming segmentation) → `foo bar`.
4. Undo leet/OCR substitutions inside alphabetic tokens (`0→o, 1→l/i, 5→s, 3→e, 4→a`).
5. Handle aliases: split on `f/k/a`, `a/k/a`, `d/b/a`, `formerly` → keep every alias and take the best score.
6. Drop store numbers (`#66894`) and collapse repeated tokens.
7. **Legal-suffix canonicalisation** → a separate `legal_form` field that is **removed from the core name**:
   - US: inc/incorporated, llc/l.l.c., corp/corporation, co, ltd, lp, llp, pllc, pc
   - India: pvt/private, ltd/limited, llp, opc
   - France: sarl, sas, sasu, eurl, sci, sa, ets/établissements, cie
   - Unknown suffixes: learn them. Tokens whose presence/absence flips most often between true pairs in train are "noise tokens" (this surfaces `enterprises`, `service`, `center`, `group`, `district`, `mr`, `shri`, `smt`, etc.).
8. `&` ↔ `and`/`et`, remove punctuation, sort tokens for an order-invariant variant.
9. **IDF weights** per country, computed on the union of all sources (test included, since it is unlabelled), so rare tokens dominate.

**Native script → Latin** (critical: 24% of Indian true pairs)
- Detect the script by Unicode block.
- Base: rule-based Indic transliteration (e.g. `indic-transliteration`, MIT; it is a library, not a data lookup). It gives `मॉडर्न हाई मार्केटिंग` → `modarn haai maarketing`.
- **Learn a correction model from training pairs.** Align S1 Latin names with their native-script matches (`Private Limited` ↔ `प्राइवेट लिमिटेड`), mine a token dictionary native-token → Latin-token (tens of thousands of pairs), and fall back to phonetic keys (a Double-Metaphone/consonant skeleton on the transliteration).
- Apply the same idea to addresses: native-script state names (`ಕರ್ನಾಟಕ` → `karnataka`), mined from train co-occurrence.
- Alternative or complement: a multilingual embedding model (`LaBSE` Apache-2.0, or `intfloat/multilingual-e5-small`/`-base` MIT). It handles cross-script names directly for blocking and features.

**Addresses**
1. NFKC, casefold, strip accents, standardise separators.
2. Parse into components with country-aware regex:
   - `house_no` (numeric core with zero-padding stripped, plus suffix letter/range: `003778`→`3778`, `134-B`→`134`+`b`, `5-7`→{5,7}, `G-6/25` → {`g6/25`, `6`, `25`})
   - `unit` (apt/unit/fl/flat/floor/tower/pmb/po box, kept separate and **excluded** from core matching)
   - `street` (with type expanded: rd→road, st→street, **saint→street when it follows a name**, dr→drive, ln, ct, cir, ave/av, blvd/bd, r./rue, imp, ch, pl, all, cite)
   - `city`, `state`/`region`, and `pin`/`zip` when present
3. **US state** code ↔ name table (50 entries: this is a format standard, not an external lookup; document it). India state code ↔ name ↔ native script, mined from training data where possible.
4. **France région ↔ département co-occurrence is mined from test S2/S3 itself.** Look at which pairs appear with the same city across records. Unsupervised, so no labels are used.
5. Landmark phrases (`near`, `opp`, `behind`, `beside`, `sidco opp`) → a separate `landmark` field with lower weight.
6. City aliases: mine city-name pairs that co-occur for the same S1 entity in train (`orcutt`~`santa maria`, `bombay`~`mumbai`, `madras`~`chennai`, `secunderabad`~`hyderabad`). They become learned soft-equivalence features.

---

## 4. Blocking / candidate generation (sets the recall ceiling)

Target: **≥ 99.5% pair recall** with about **20–40 candidates per S1**, i.e. ~40–70M candidate pairs on test.

Every pass runs **within country**, and most within state or city partitions. Run each pass **in both directions**: for each S2/S3 record find its top-k S1, and for each S1 find its top-k S2/S3. Take the union. Direction S2/S3 → S1 fits the "each record has at most one owner" structure.

| Pass | Key | Catches |
|---|---|---|
| B1 Address exact-ish | `(country, house_no, street core first token)` + city/state fallback | same address, very different or native names |
| B2 Name TF-IDF | char 3–4-gram TF-IDF on core name, cosine top-k within `(country, state)`; sparse top-k product via `sparse_dot_topn` or chunked matrix products | typos, suffix noise, reordering |
| B3 Rare-token | inverted index on the 1–2 rarest (highest-IDF) name tokens within `(country, state)`; skip blocks larger than N | truncated names (`Modern`, `384 Church`) |
| B4 Transliterated/phonetic | B2/B3 run on transliterated + phonetic keys | native-script names |
| B5 Embedding ANN | multilingual embedding of `name + city` → FAISS (GPU, IVF/HNSW) top-k within country | cross-script and semantic leftovers |
| B6 Address TF-IDF | char n-gram TF-IDF on the normalised address without city/state, top-k within state | reordered addresses, missing house numbers |
| B7 Empty-address records | name-only B2/B4 with a larger k | the ~3% of S2/S3 with blank addresses |

**Candidate pruning (the stage whose output is `candidate_pairs.tsv`):**
- Merge the passes, compute a cheap score (a small logistic regression or LightGBM on ~10 fast features), and keep the top-K per S1 and the top-M per S2/S3.
- Tune K and M on the hold-out to hit the recall target at minimum size.
- Record which passes produced each pair; they become features.
- **`candidate_pairs.tsv` must be exactly the set scored by the final model.** Write it at this point, and ensure `matches ⊆ candidates` (the validator checks this).

**Memory tactics:** iterate over (country, state) partitions and store IDs as int32 (strip the `S1-`/`S2-` prefix and map back at output). Persist candidates to parquet per partition, and never hold the full cross-product.

---

## 5. Matching model

### 5.1 Pair features (country-agnostic where possible)
- **Name:**
  - rapidfuzz `ratio`, `partial_ratio`, `token_sort_ratio`, `token_set_ratio`, Jaro-Winkler
  - IDF-weighted token Jaccard / soft-TF-IDF (Monge-Elkan with Jaro-Winkler)
  - char 3-gram cosine; on core name and full name, raw and translit/phonetic variants
  - rarest-token match, first-token match, abbreviation/acronym match (`HA`, `ZOJ`)
  - legal-form agree / conflict / missing
  - domain-segmented match flag, alias best score
  - length ratio, name-script flags (native vs. Latin), embedding cosine
- **Address:**
  - house number: exact / contained / suffix-only differ / missing on either side; street core similarity; street-type agreement
  - city: exact / alias-mined / fuzzy; state agreement (after code normalisation); pin/zip equality or prefix
  - unit match/conflict, landmark overlap
  - whole-address token_set ratio and IDF Jaccard; address-missing flags
- **Context / competition features** (large gains in assignment-style ER):
  - rank of this S1 among the S2/S3 record's candidates, and the score gap to the best and second-best S1
  - rank of this S2/S3 among the S1's candidates; the number of candidates sharing the same address (dense buildings, malls)
  - the number of S1 entities at the same address, and whether another S1 has an *identical* core name
  - blocking passes that fired
- **Cross-record (group) features:**
  - S2/S3 records of one entity look like each other (`Cornerstone Immedr1ion` and `Cornerstone Immersion (Corporation)`). For each candidate, compute the max similarity to *other* high-confidence candidates of the same S1.
  - This lets a record with a weird name but the same address as confirmed matches get pulled in. It works as a second-stage or iterative feature.

### 5.2 Models
1. **Stage-1: LightGBM binary classifier** on pairs from train blocking output (labels from ground truth). About 60–100M pairs is too many, so subsample easy negatives and keep all hard negatives (high cheap-score negatives). Monitor by country.
   - Train with and without a `country` feature, and check the leave-country-out score to decide. Default for France: drop country-specific flags and rely on generic similarity.
2. **Stage-2 (optional, if time and GPU allow): cross-encoder reranker.** Fine-tune a small multilingual transformer (`xlm-roberta-base`, MIT; or `multilingual-e5-base`, MIT; both far below 8B) on `name [SEP] address` pairs.
   - Run it only on the uncertain band from stage 1 (e.g. p ∈ [0.1, 0.9]), a few million pairs, which is feasible on the 4050 at fp16.
   - Blend its logit into a final LightGBM.
3. **Iteration:** after the first assignment, recompute the group features against confirmed matches and rescore once.

All models must be MIT/Apache-2.0 and ≤ 8B parameters. Record the licences in the docs.

---

## 6. Decision layer (where F0.5 is won)

1. **Exclusive assignment:** each S2/S3 record is assigned to its single highest-probability S1, or none if the best is below τ_reject. This removes the false merges where one record links to two S1 entities. A greedy best-owner pass is enough; Hungarian matching is not needed because one S1 takes many records.
2. **Per-entity set selection that maximises expected F0.5.** Sort an entity's surviving candidates by calibrated p (isotonic calibration on hold-out). Choose the prefix k that maximises expected F0.5 under the probabilities (a standard expected-F-measure optimiser; with ≤ 11 matches it can be exact or DP). k = 0 applies when the expected score from predicting nothing (the probability of being a singleton) is higher. This handles singletons automatically.
3. **Fallback:** a simple global threshold with per-country tuning (and a conservative, higher threshold for France). Choose whichever scores better on hold-out.
4. **France guardrail:** without labels, check score-distribution drift (predicted matches per S1 vs. US/India ≈ 3.5). If the model is much more timid on France, try pseudo-labelling: add high-confidence French pairs (p > 0.98, unique owner) as training data and retrain once.

---

## 7. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M0 | parquet cache, loaders, macro-F0.5 evaluator, geographic split | evaluator reproduces the README example |
| M1 | normalisers + simple baseline (B1+B2 blocking, rapidfuzz score, threshold) | end-to-end valid submission; run `validate_submission.py`; first leaderboard score |
| M2 | full multi-pass blocking + transliteration | hold-out pair recall ≥ 99.5% at ≤ 40 candidates per S1 |
| M3 | LightGBM matcher + exclusive assignment + expected-F0.5 set selection | large jump over M1; per-country report |
| M4 | group features, iteration, France checks (leave-country-out) | India score within a few points of US; LCO gap understood |
| M5 | cross-encoder rerank on the uncertain band (optional) | measurable hold-out gain, runtime within budget |
| M6 | packaging: `run_pipeline.py` reproduces both TSVs from raw data; pinned requirements; README; filled `Documentation_template.md` | clean-env rerun matches the submitted file; validator PASS with `--check-ids` (drop `--candidate` if memory is tight) |

Run the full pipeline on test at M1, M3 and M5 to catch memory/runtime issues early. Test S2+S3 is ~10M rows.

---

## 8. Risks & mitigations

- **Memory blow-ups (16 GB):** partition by country/state, use int IDs, category dtypes, chunked sparse top-k, and persist to parquet between stages.
- **France (no labels):** generic features, mined région↔département and street-type rules, leave-country-out validation, pseudo-labelling, and a conservative threshold.
- **Over-merging in dense addresses** (malls, business centres, the same building): competition features, unit/suite features, and exclusive assignment.
- **Leakage in validation:** use a geographic split, fit IDF/embeddings on unlabelled text only, and never use GT-derived stats computed on hold-out.
- **Rule compliance:**
  - no external APIs or geocoders; all dictionaries are mined from the provided data or are generic format conventions (state codes, street-type abbreviations), and each is documented
  - pretrained open-licence models only
  - `candidate_pairs.tsv` = the exact inference set
- **Confidentiality:** keep data, outputs and the repository private. Do not push the dataset or outputs to any public remote.
