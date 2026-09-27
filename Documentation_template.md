# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** momos  
**Team Members:** Darsh Verma (team leader), Farhan DHATTIWALA, Tanay Mody, shaurya manjkhola  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We use a multi-stage pipeline: blocking → learned pruning → matching model, with every stage run inside (country, region) partitions.

- **Blocking.** Four complementary candidate-generation passes, all within (country, region):
  - an exact address key
  - character TF-IDF on names
  - word TF-IDF on addresses
  - a **multilingual name encoder fine-tuned on the training pairs**, so that native-script names (`ভিশন সিস্টেমস প্রাইভেট লিমিটেড`) meet their Latin counterparts (`Vision Systems Private Limited`)
- **Learned pruning.** A LightGBM model on cheap similarity features cuts the candidate set to about **5.7 pairs per Source 1 entity**.
- **Matching model.** A multilingual **cross-encoder reranker** and a second LightGBM using **group features** (how a record compares with the entity's other confident matches) make the final decisions.
- **Decision rule.** Each record goes to at most one entity, and each entity keeps the set of matches that maximises expected F0.5.

Results:
- held-out validation macro F0.5: **0.988** (US 0.992, India 0.982)
- public leaderboard: **0.9806**
- nothing hard-codes the country set, so France (unseen in training) is handled by country-agnostic features and an unsupervised region lexicon

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the 2.2M training Source 1 entities, 5.0M Source 2 and 5.3M Source 3 records:

- **Structure of the labels:**
  - Every S2/S3 record matches **at most one** S1 entity (7.64M matched ids, all unique). So the task is an *assignment* problem: each record has one owner, or none.
  - **26% of S2/S3 records match nothing.** They are distractors, often look-alike businesses.
  - **5.6% of S1 entities are singletons**; the mean is 3.5 matches and the maximum 11.
  - Matches **never cross countries.**
- **Name noise:**
  - junk wrappers (`--`, `>>`, `[..]`, `***`)
  - legal suffixes added, dropped or reordered (`Pvt/Private`, `Ltd/Limited`, `LLC`, `SARL`)
  - OCR/leet typos (`M0DERN`, `BANC5HARES`) and accent injection (`Gríll`)
  - duplicated tokens, truncation, domain-name forms (`howiedotyfairclough.com`), `f/k/a` aliases, store numbers
  - occasional completely unrelated names that can only be matched by address
  - **native-script names:** Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam. They make up **24% of Indian true pairs** (7% of US), and hurt the most.
- **Address noise:**
  - case changes and street-type abbreviations (`Rd/Road`, `St/Street`)
  - state codes vs. names vs. native script (`KA`/`Karnataka`/`ಕರ್ನಾಟಕ`)
  - zero-padded or suffixed house numbers (`003778`, `134-B`, `1/236`)
  - reordered components, missing parts, city typos, `PMB`/`PO Box` insertions
- **Blocking-key coverage on true pairs:**
  - a shared name token covers only 85% (India 75%)
  - a shared ≥3-digit number covers only 57%
  - **no single cheap key is enough; the union of several keys gets close to 100%**
- **France** appears only in test (15% of test S1). It needs country-agnostic features and French format handling (`R./Rue`, `Av`, `SARL/SAS`, région vs. département).

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-pass blocking (sparse + dense) → learned pruning (LightGBM) → cross-encoder reranker → second-stage group LightGBM → constrained, expected-F0.5 set selection.

**Core Innovation:**
1. A **multilingual name encoder fine-tuned contrastively on the training pairs**, used both for dense blocking and as a feature. It raised validation blocking recall for native-script records from 0.81 to 0.99.
2. A **cross-encoder reranker** that reads both records jointly.
3. **Group features** that exploit the fact that records of one entity resemble each other.
4. An **assignment-aware decision layer**: each record has one owner, and each entity keeps its expected-F0.5-optimal set of matches.

**Validation protocol:**
- **Geographic hold-out:** whole regions with all their S1/S2/S3 records and distractors (US `ny, nd, az`; India `kerala, tamil nadu`), 285k entities, so candidate density and distractor rates match reality.
- **Exact macro F0.5**, including the singleton rule, unit-tested on the README example (0.714).
- The final models are **retrained on all training data**. The hold-out also withholds most Tamil/Malayalam-script names, so validation *understates* India.

---

## 3. Candidate Generation (Blocking)

Every pass runs **within country**, and within the **region** (state / région) when one is known. Each S2/S3 record retrieves its top-k S1 entities. That direction fits the "one owner per record" structure.

**Regions without labels:**
- **S1 region:** the most frequent address component, for example `tx`, `karnataka`, `hauts-de-france`.
- **S2/S3 region:** a lexicon learned from **pseudo-links**, meaning S2/S3 ↔ S1 pairs that share an exact address key and have near-identical names. From these the lexicon learns `virginia → va`, `ಕರ್ನಾಟಕ → karnataka`, `winston-salem → nc` and département → région, without using any labels. So it works for France too.
- **Records without a region** (about 3–5%, mostly empty addresses) search the whole country.

**Blocking keys used:**

| Pass | Key / similarity | Catches |
|---|---|---|
| B1 | exact `house number | next informative token` keys (≤ 20 S1 per key) | same address, any name |
| B2 | name **char 3-gram TF-IDF** cosine, top-10 per record | typos, suffix noise, reordering |
| B6 | address **word TF-IDF** (region parts removed), top-5 | native / unrelated names at the same address |
| B5 | **fine-tuned multilingual-e5-small** name embeddings, top-10 (GPU) | cross-script names |

- The top-k searches run on the GPU: sparse CSR × dense chunks for TF-IDF and dense matmul for embeddings, both with `torch.topk`. A 176k-entity region against 1M queries takes about 1 min.
- After blocking comes a **learned pruning stage**, the last filtering step:
  - A LightGBM model scores every blocked pair using the similarity features below; its out-of-fold scores are used during training.
  - Pairs with p < 1e-4 are dropped.
  - `candidate_pairs.tsv` is exactly the set that survives, and it is the only set the matching model (reranker and stage 2) ever runs on.

**Candidate pairs generated (test):** 12,809,075 pairs, **7.4 per S1 entity**. Before pruning, blocking produced ~46 per entity.

**How we ensured true matches were not lost (validation, held-out regions):**

| Stage | Pairs per S1 | Pair recall | Oracle F0.5* |
|---|---|---|---|
| B1 + B2 (name/address, first version) | 67 | 0.975 | 0.991 |
| + B5 encoder pass (all passes) | 100 | **0.995** | 0.998 |
| after learned pruning (p ≥ 1e-4) | **5.7** | **0.994** | 0.998 |

\*Oracle F0.5 is the score a perfect matcher would get on this candidate set, i.e. the ceiling blocking allows.

- The encoder pass raised recall for native-script records from **0.81 to 0.99**.
- Pruning removes 82% of the pairs but only 0.1% of the true pairs.

---

## 4. Matching Model

**Features used (stage 1, ~45 features):**
- **Name features:**
  - rapidfuzz `ratio`, `token_set_ratio`, `token_sort_ratio`, `partial_ratio` and Jaro-Winkler on the normalised core name
  - char-TF-IDF cosine
  - **fine-tuned encoder cosine** and its rank among the record's candidates
  - legal-form agree / conflict / missing
  - name lengths, native-script flag, name frequency
- **Address features:**
  - token-set / ratio / partial similarity on the address with region parts removed
  - word-TF-IDF cosine
  - house-number overlap (common count, Jaccard, counts on each side)
  - address-key match, region agreement, empty-address flags, address frequency
- **Competition features:**
  - For each record: the gap between this pair's name+address similarity (and TF-IDF / encoder cosines) and the record's best candidate, the rank, and the number of candidates. These carry the most weight (`sim_gap_m` has by far the highest gain).
  - The best similarity over the entity's candidates.
- **Normalisation behind the features:**
  - NFKC and ASCII fold / transliteration (anyascii, ISC)
  - junk-wrapper removal, leet/OCR fixes inside alphabetic tokens, alias splitting, store-number removal, duplicate-token collapse
  - a legal-form table covering US, India and France, extracted into a separate field
  - canonical street types (`rd`, `st`, `ave`, `blvd`, `rue`, `rte`, …) and ordinals, and zero-stripped house numbers

**Model type:**
1. **Stage 1: LightGBM** (255 leaves, lr 0.05, early stopping at ~2.7k rounds).
   - Trained on all blocked pairs of the training regions, with negatives subsampled to 30M.
   - Serves as the learned pruner.
   - Its out-of-fold scores (2-fold by entity) feed stage 2.
2. **Name encoder: `intfloat/multilingual-e5-small`** (MIT, 118M parameters).
   - Fine-tuned with a symmetric in-batch contrastive (InfoNCE) loss on 3M (S1 name, S2/S3 name) true pairs, with native-script pairs oversampled 2×.
   - Used for blocking pass B5 and as a feature.
3. **Cross-encoder reranker: `multilingual-e5-small`** plus a linear head.
   - Reads `name | address` of both records together (raw text, native scripts kept).
   - Trained with binary cross-entropy on ~2.5M pruned pairs of **one half** of the training entities.
   - It scores the uncertain band (stage-1 p ∈ [0.001, 0.99]) plus each record's encoder top-1 candidate.
4. **Stage 2: LightGBM** on:
   - the stage-1 score and rival gaps: best other S1 for the record, rank per record and per entity, expected entity size
   - the reranker logit
   - **group features:** the max/mean name, address and encoder similarity of the record to the entity's *other* confident matches (stage-1 exclusive-best, p ≥ 0.5)

   It is trained on the half of the entities the reranker never saw, so every input is out-of-sample.

Every model is MIT/Apache-licensed or our own LightGBM, and all are far below 8B parameters.

**Threshold selection method:**
1. **Exclusive assignment:** each S2/S3 record keeps only its highest-probability S1.
2. **Per entity:** sort its remaining candidates by probability p₁ ≥ p₂ ≥ …
   - choose the top-j that maximises expected F0.5 ≈ 1.25·Σᵢ≤ⱼ pᵢ / (0.25·(Σᵢ pᵢ + 1) + j)
   - predict **nothing** when P(no match) ≈ Πᵢ(1−pᵢ) is higher, which handles singletons automatically
3. Chosen on validation against the best global threshold: 0.9881 vs 0.9877.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), held-out validation regions:**

| Version | Val F0.5 | US | India |
|---|---|---|---|
| Baseline: normalisation + B1/B2/B6 + LightGBM + threshold | 0.958 | 0.978 | 0.930 |
| + fine-tuned multilingual encoder (B5 + features) + stage-2 group model | 0.966 | 0.983 | 0.943 |
| + cross-encoder reranker | 0.977 | 0.988 | 0.962 |
| + training on all fit regions (8× data) | 0.985 | 0.992 | 0.975 |
| + wide reranker band (encoder top-1 pairs) | **0.988** | 0.992 | 0.983 |
| + learned pruning (5.7 candidates / S1) | 0.988 | 0.992 | 0.982 |

  - Precision is 0.998, recall 0.972, and singleton accuracy 0.990.
  - **Public leaderboard: 0.9806** (unpruned version; the first baseline scored 0.944).
- **Common false positives (wrong merges):** there are few (precision 0.998). Most are **distractor records that look like a real entity**: the same generic name (`Jai Trading`, `Apex Engineering`) at a nearby or partial address, or a record whose address matches a different business in the same building.
- **Common false negatives (missed matches):**
  1. **Native-script names in scripts rarely seen in training.** Tamil and Malayalam were held out with their regions, for example `Alpha Products Private Limited` ↔ `ஆல்ஃபா புராடக்ட்ஸ் பிரைவேட் லிமிடெட்` at the same address. The final models train on all regions, which covers these scripts.
  2. Records whose name was replaced by an unrelated one **and** whose address is heavily truncated.
  3. **Look-alike entities competing for a record:** "taken by another S1", 0.8% of true pairs.
  4. **Blocking misses:** 0.6%. They are concentrated in very large regions (Maharashtra, Delhi), where top-10 neighbour lists are crowded.

---

## 6. Conclusion

The biggest gains came from three things:
- treating native-script names as a first-class problem, with a multilingual encoder fine-tuned on the training pairs
- a cross-encoder that reads both records together
- exploiting the assignment structure: one owner per record, group consistency and expected-F0.5 set selection

Learned pruning keeps the candidate set small, about 5.7 per entity, close to the 3.5 true matches, at no measurable accuracy cost. A geographic hold-out that mirrors test conditions (including distractors) made every decision measurable.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- all source in `src/`
- `README.md` with setup and reproduction instructions
- `requirements.txt` with pinned versions

**Entry point:** `bash src/reproduce.sh` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the raw data. It takes about 4.5 h on one RTX 5090 with 125 GB RAM, and ends with the official validator.

Steps:
1. `prepare_data.py` → `make_split.py` → `normalize_data.py`
2. `train_encoder.py --all-train`
3. `run_pipeline.py --run all --stage all` (blocking, features, stage 1, pruning, reranker, stage 2)
4. `run_pipeline.py --run test --stage all` (writes both files)

`--validate` also runs the held-out validation experiment.

Library modules (`src/ber/`):

| Module | Role |
|---|---|
| `normalize` | name / address normalisation, address keys |
| `geo` | region detection and the pseudo-link region lexicon |
| `blocking` / `topk` | blocking passes B1/B2/B5/B6, GPU sparse + dense top-k |
| `embed` | multilingual name-encoder fine-tuning and encoding |
| `features` | stage-1 pair and competition features (streamed per bucket of records) |
| `rerank` | cross-encoder reranker |
| `group` | stage-2 group and rival features |
| `assign` | exclusive assignment and expected-F0.5 set selection |
| `evaluate` | exact macro F0.5 with the singleton rule, blocking recall / oracle report |

**Licences:**
- pretrained model: multilingual-e5-small (MIT)
- libraries: polars, numpy, scipy, scikit-learn, LightGBM, rapidfuzz, torch, transformers (all MIT / BSD / Apache-2.0), plus anyascii (ISC)

**Data use:** only the provided data.
- Dictionaries (legal forms, street types) are generic format conventions.
- Region lexicons and city/state mappings are learned from the data.
- No external lookups, geocoders or APIs.

### B. Additional Results

- **Blocking by pass** (validation, share of true pairs each pass finds on its own):

| Pass | B1 address key | B2 name TF-IDF | B6 address TF-IDF | B5 encoder | Union |
|---|---|---|---|---|---|
| Share | 0.50 | 0.86 | 0.84 | 0.96 | 0.995 |

- **Test predictions:**
  - matches per S1: India 3.35, US 3.38, France 3.38
  - share of empty lists: India 6.0%, US 5.8%, France 5.4%
  - The US/India training mean is 3.5 matches and 5.6% singletons, and France predictions are consistent with that.
- **Compute:** a single workstation (64 cores, 125 GB RAM, 1× RTX 5090 used at a time). Blocking and scoring run one country per process, so peak RAM stays around 30 GB.
