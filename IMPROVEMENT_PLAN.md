# Improvement plan: what could have beaten our final submission

*Team momos, a retrospective after the Amazon ML Challenge 2026 (Business Entity Resolution).*

**Final result:**
- **public leaderboard 0.982681**
- held-out validation macro F0.5 **0.9930** (US 0.993, India 0.993)
- 7.4 candidates per Source-1 entity

**Final pipeline:**
- normalisation
- multi-pass blocking (address keys, name/address TF-IDF, a fine-tuned multilingual-e5 name encoder)
- LightGBM stage 1, then learned pruning
- cross-encoder reranker: bge-reranker-v2-m3 for US/India, multilingual-e5-small for France
- LightGBM "group" stage 2
- expected-F0.5 set selection

This document lists what we could have done better, ranked by expected gain for the effort. It is grounded in *our own measured numbers* and in current entity-resolution literature. Every gain below is an **estimate**, not a result.

---

## 1. Where the remaining error was

| Error source | Evidence from our runs | Size of the opportunity |
|---|---|---|
| **France** (15% of test entities, no labels) | Val 0.993 but LB 0.9827. If US/India scored 0.990–0.993 on test, France was roughly **0.92–0.94**. France was also the only country where the small e5 reranker beat bge (0.98268 vs 0.98232 overall). | **Largest.** Raising France to 0.96 alone is ≈ **+0.004–0.006** overall. |
| **Matcher recall** (US/India) | Final val precision 0.9992 but recall **0.980**. Pair recall after pruning 0.994, so ~0.6% of true pairs are lost at blocking and ~1.4% in matching/decisions. | Medium: ≈ +0.002–0.004 on val |
| **Look-alike competition** | ~0.8% of true pairs were "taken by another S1": same generic name, same building. | Part of the above |
| **Validation that didn't mirror test** | The region hold-out also held out the Tamil/Malayalam *scripts*. That made India look worse than it was, and it wrongly rejected deeper blocking (India −1.3 pts on val). | Indirect: better decisions everywhere |

The public leaderboard is only a subset of test, so a difference below ~0.0005 is within noise.

---

## 2. Summary: the ideas ranked

| # | Idea | Main algorithm / model (licence, size) | Est. gain (LB) | Effort | Risk |
|---|---|---|---|---|---|
| 1 | **Synthetic French training pairs** from the learned noise process | noise-operator model + augmentation (Rotom-style) | +0.002–0.005 | 1–2 days | medium |
| 2 | **Agreement-filtered self-training on France** | pseudo-labels where e5 and bge agree, retrain reranker and stage 2 | +0.001–0.004 | 1 day | medium |
| 3 | **Listwise reranking per S2/S3 record** (+ "no match" slot) | softmax over a record's candidates on bge-m3 / Qwen3-Reranker | +0.001–0.003 | 1 day | low |
| 4 | **Bigger or ensembled rerankers** | Qwen3-Reranker-4B/8B (Apache-2.0, LoRA), gte-multilingual-reranker-base (Apache-2.0, 306M), seed ensemble | +0.001–0.003 | 1–2 days, GPU-heavy | low |
| 5 | **Collective decisions** (transitivity across S2↔S3) | correlation clustering / iterative group features / min-cost-flow assignment | +0.001–0.002 | 1–2 days | medium |
| 6 | **Learned transliteration for Indic scripts** | IndicXlit (MIT, ~11M) native→Roman + phonetic keys | +0.0005–0.002 | 0.5 day | low |
| 7 | **Joint name+address dense blocking** with hard negatives | SC-Block / UniBlocker-style contrastive bi-encoder, ANCE negatives, FAISS HNSW | +0.0005–0.002, and a smaller candidate set | 1 day | low |
| 8 | **Test-like second validation + leave-country-out proxy** | random-entity split within regions; train US → evaluate India | enables #1, #2, #7 | 0.5 day | none |
| 9 | Per-country / per-size **calibration** of decisions | isotonic calibration, rules per country and entity size | +0.0003–0.001 | 0.5 day | low |
| 10 | Efficiency: distillation, vectorised features, ONNX | distil bge → e5-small, batched rapidfuzz `cdist`, fp16/ONNX | runtime ÷2–4 | 1 day | none |

Items 1–4 target the two biggest error sources, France and matcher recall. A realistic combined outcome is **LB ≈ 0.986–0.990**. Going beyond that needs the remaining hard cases (identical generic names in one building, heavily corrupted records) solved, which the data may not allow.

---

## 3. France / unseen-country adaptation (the biggest gap)

**Problem.** Every model learned US/India conventions: English and Indic names, state codes, `Pvt Ltd`. For France our only signals were country-agnostic features and one leaderboard A/B test. The bge reranker became *more cautious* on France (3.20 vs 3.38 matches per entity), which suggests it under-matched French records.

### 3.1 Synthetic labelled French pairs (recommended first)
The dataset is synthetic, and the same noise generator is applied in every country. In the EDA we saw:
- junk wrappers
- legal-suffix drop/reorder
- leet typos, accent injection
- truncation
- component reordering
- région ↔ département swaps

**Idea:**
1. **Learn the noise operators and their rates** from US/India (S1 → S2/S3) true pairs: token-level alignment, edit operations, suffix edits, address-part permutations.
2. **Apply them to French S1 records**, producing labelled French positives.
3. Add *hard negatives* by pairing each French S1 with noised versions of its look-alike neighbours from the same région.

That gives French training data for the reranker and stage 2 with no external data at all. This is the data-augmentation idea behind **Rotom/InvDA** (SIGMOD 2021), which reports +4.6 F1 on EM benchmarks in low-label settings.

### 3.2 Agreement-filtered self-training
- Pseudo-label French pairs only where **both** rerankers agree:
  - p > 0.98 for positives
  - p < 0.02 for negatives among the same record's candidates
  - a unique owner
- Fine-tune one extra epoch on train pairs plus French pseudo-pairs, then re-run stage 2. Agreement between two differently-biased models (co-training) limits confirmation bias.
- Domain-adaptation work for EM (**DADER**: MMD / gradient-reversal feature alignment) is a more principled alternative when target labels are absent.

### 3.3 Ensemble instead of choosing
For France we *picked* e5 over bge on the leaderboard. Averaging their logits, or feeding both logits into stage 2, is usually better than either model alone, and it removes the need for a leaderboard-based choice.

### 3.4 A France proxy we never ran
PLAN.md specified **leave-one-country-out** runs (train on US → evaluate on India, and the reverse), but we never ran them. They are the only offline way to tune 3.1–3.3 for an unseen country, and they should come first.

---

## 4. Stronger matcher

### 4.1 Listwise reranking with a "no match" option
We trained the rerankers **pointwise**: binary cross-entropy per pair. But each S2/S3 record belongs to *at most one* S1 entity.

A **listwise** objective scores all K candidates of a record together, with a softmax over `[cand₁ … cand_K, NO_MATCH]`. That models the assignment structure directly: the model learns *relative* differences between look-alikes. It targets our 0.8% "taken by another S1" errors and the calibration of singletons.

The literature on listwise vs pointwise reranking and on cross-encoder entity linking (softmax over candidates) supports this for close candidate lists. Our lists are short, ~2–7 per record after pruning, which is exactly where listwise training is cheap.

### 4.2 Bigger rerankers, within the rules (MIT/Apache, ≤ 8B)

| Model | Licence | Size | Why |
|---|---|---|---|
| **Qwen3-Reranker-4B / 8B** | Apache-2.0 | 4B / 8B | 100+ languages; strongest open rerankers of 2025–26. Fine-tune with LoRA on 2×32 GB. |
| gte-multilingual-reranker-base | Apache-2.0 | 306M | 70+ languages, fast; a good ensemble member and distillation target |
| Qwen2.5-7B-Instruct | Apache-2.0 | 7B | Generative pairwise judge for the hardest ~1% band only. Fine-tuned small LLMs match large ones on EM (Peeters et al. 2025, arXiv 2409.08185, AnyMatch). |
| *(not allowed / avoid)* | | | Jellyfish-7B/8B are Mistral/Llama-based, so check the licence. bge-reranker-v2-gemma uses the Gemma licence. |

Our jump from e5-small (118M) to bge-m3 (568M) was **+0.0053 on val**. A 4B reranker, run only on the uncertain band, is the natural next step.

### 4.3 Better training
- **K-fold cross-fitting** (e.g. 4 folds). We used half splits, one for the reranker and one for stage 2, so each model saw only 50% of the data.
- **2–3 epochs** with ANN-mined hard negatives (ANCE) instead of 1 epoch.
- **Seed / model ensembles** for the reranker, averaging logits. That's cheap at inference with our ~7 candidates per record.

---

## 5. Native-script names

anyascii gave crude folds (`ভিশন সিস্টেমস` → `bhisn sistems`), and the fine-tuned encoder carried most of the load. Better:
- **IndicXlit** (AI4Bharat, **MIT**, ~11M params, 21 Indic scripts, native↔Roman), trained on the Aksharantar corpus. Transliterate native names properly (`ভিশন সিস্টেমস প্রাইভেট লিমিটেড` → `vision systems private limited`), then reuse all the Latin similarity features and blocking keys. This is a pretrained open model, not a lookup, so it's within the rules.
- **Phonetic keys** (Double Metaphone, consonant skeletons) on the transliterated text, as an extra blocking pass and feature.
- **Hard-negative contrastive training** for the name encoder. We used in-batch random negatives, and ANN-mined look-alike negatives sharpen the space (ANCE).

---

## 6. Structure-aware (collective) decisions

We decided each (S1, record) pair mostly independently: greedy best owner per record, then expected-F0.5 per entity. Entity resolution literature favours **collective** resolution:
- **Transitivity across sources.** Records of one entity in S2 and S3 resemble *each other*. Score S2↔S3 pairs within an entity's candidate pool, and run **correlation clustering** on the S1–S2–S3 graph (the "(Almost) All of Entity Resolution" survey; in-context clustering ER with LLMs, 2025). Our group features were one pass of this idea; iterating it 2–3 rounds is cheap.
- **Global assignment:** min-cost flow, where each record has capacity 1 and each entity capacity ≈ its expected size. This replaces greedy best-owner and resolves conflicts jointly.
- **Calibration:** isotonic calibration per country and per entity size before the expected-F0.5 rule. We used one rule per reranker.

---

## 7. Blocking

- **Joint name+address dense retrieval.** Our dense pass embedded *names only*. A contrastive bi-encoder on `name | address` with hard negatives (**SC-Block**, **UniBlocker**, Neural LSH blocking) would catch records whose name is replaced *and* address reordered.
- **Deeper search in big regions, retested properly.** It raised India's blocking recall (0.9902 → 0.9944), but we rejected it after India dropped on val. That drop came from scripts unseen in validation, not from blocking. It should have been retried with matchers retrained on the harder candidate mix, plus hard negatives.
- **FAISS HNSW** for the country-wide searches (records without a region), instead of brute-force GPU top-k.
- **Probabilistic linkage as a feature:** a Fellegi-Sunter score with term-frequency adjustment (**Splink**, MIT) is a cheap, well-calibrated signal for common vs rare names.
- **Candidate size as an objective from day 1.** Learned pruning cut 46 → 7.4 candidates per entity at no cost. With a per-record top-2 cap plus a threshold it could likely reach ~4–5.

---

## 8. Validation and process: what slowed or misled us

1. **Two validations, not one:**
   - the geographic hold-out, which is realistic for candidate density
   - a **random-entity split within regions**, which is realistic for test, where scripts and regions are *seen*

   The single region hold-out made India's scripts unseen, which distorted three decisions.
2. **The leave-country-out France proxy** (§3.4): planned, never run.
3. **Deterministic splits.** The split differed between laptop and workstation (library versions). The fix is to hash entity ids, not rely on sort order.
4. **Compute earlier.** Four days on a 16 GB laptop cost many out-of-memory crashes. The 2×RTX 5090 workstation turned 6 h runs into ~15 min experiments.
5. **Read every rules update early.** The "smaller candidate set ranks higher" rule arrived late, and pruning should have been designed in from the start.
6. **Cached, resumable stages from the start.** We added resumable per-step markers after several crashes.

---

## 9. Efficiency ideas

- **Distil** the bge reranker (568M) into e5-small (118M) on the pruned pairs, for ~4–5× faster scoring at a small accuracy cost.
- **Vectorise string features:** batched rapidfuzz `cdist`, or GPU edit distance, instead of per-pair Python lists. Feature building dominated CPU time.
- **fp16 / ONNX Runtime** inference for the rerankers, and pre-tokenised caches.
- **Don't recompute:** cache features per universe and only recompute the columns that change between experiments.

---

## 10. What we tried that didn't help (lessons)

| Experiment | Result | Lesson |
|---|---|---|
| Narrow reranker band (p ∈ [0.01, 0.99]) | 0.9845 vs **0.9881** with a wide band plus encoder top-1 | Cross-script matches hide at very low stage-1 p |
| Keep encoder top-1 through pruning | 0.9853 vs 0.9877 (8.2 vs 5.7 candidates per entity) | Flooding the reranker with easy negatives hurt it |
| Deeper blocking in big regions | India val 0.982 → 0.969 | Probably a validation artefact (unseen scripts); should be retested, see §7 |
| Big reranker trained on random pairs | Stopped: 59% easy positives | Train on the uncertain band plus a third easy pairs |
| torch DataParallel on 2×5090 | Illegal-memory-access crash | One model copy per GPU, no P2P |
| One-process blocking on the full data | Killed by OOM (full embedding gather, ~92 GB) | Chunked gathers, one country per process |

---

## 11. If we did it again: phased roadmap

| Phase | Work | Compute (2×5090) | Expected LB |
|---|---|---|---|
| 0 | Two validations + leave-country-out proxy; deterministic split | ½ day | (enables the rest) |
| 1 | Our final pipeline (reproduced) | ~6 h | 0.9827 |
| 2 | Synthetic French pairs + agreement self-training + e5/bge ensemble for France | 1–2 days | ~0.985–0.987 |
| 3 | Listwise reranker (bge-m3 → Qwen3-Reranker-4B LoRA), 4-fold cross-fitting, 2 epochs | 1–2 days | ~0.986–0.989 |
| 4 | IndicXlit transliteration, joint name+address dense blocking, retest deeper-k | 1 day | ~0.987–0.990 |
| 5 | Collective decisions (S2↔S3 clustering, min-cost-flow assignment), calibration | 1 day | ~0.988–0.990 |
| 6 | Distillation / vectorisation for speed; candidate-set cap ~4–5 per entity | ½–1 day | same score, faster and smaller |

All models stay within the rules: MIT/Apache and ≤ 8B parameters, no external data.
- libpostal (MIT) would help address parsing, but it is trained on OpenStreetMap / OpenAddresses, i.e. external data, so it is **likely disallowed**.
- Mined dictionaries must come from the provided data only.

---

## Sources

- Ditto, deep entity matching with pre-trained language models (VLDB): <https://dl.acm.org/doi/10.14778/3421424.3421431>
- AnyMatch, efficient zero-shot EM with a small language model: <https://arxiv.org/pdf/2409.04073>
- Peeters et al., Entity Matching using Large Language Models (EDBT 2025): <https://openproceedings.org/2025/conf/edbt/paper-81.pdf>
- Fine-tuning Large Language Models for Entity Matching: <https://arxiv.org/pdf/2409.08185>
- A Deep Dive Into Cross-Dataset EM with LLMs (EDBT 2025): <https://openproceedings.org/2025/conf/edbt/paper-224.pdf>
- Beyond Scale and Generation, understanding LM-based EM: <https://arxiv.org/pdf/2607.24688>
- DeepBlocker, deep learning for blocking (VLDB): <https://dl.acm.org/doi/abs/10.14778/3476249.3476294>
- SC-Block, supervised contrastive blocking: <https://arxiv.org/pdf/2303.03132>
- Neural Locality Sensitive Hashing for entity blocking: <https://arxiv.org/pdf/2401.18064>
- Towards Universal Dense Blocking (UniBlocker): <https://arxiv.org/pdf/2404.14831>
- Hard negative mining for domain-specific retrieval (ACL 2025): <https://aclanthology.org/2025.acl-industry.72.pdf>
- Rotom, meta-learned data augmentation for EM (SIGMOD 2021): <https://dl.acm.org/doi/10.1145/3448016.3457258>
- Heterogeneity in Entity Matching, survey (incl. DADER domain adaptation): <https://arxiv.org/pdf/2508.08076>
- IndicXlit, transliteration for 21 Indic languages (MIT): <https://github.com/AI4Bharat/IndicXlit>
- Qwen3 Embedding & Reranker (Apache-2.0): <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>, <https://arxiv.org/abs/2506.05176>
- gte-multilingual-reranker-base (Apache-2.0): <https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base>
- Qwen2.5 licences (7B Apache-2.0; 3B/72B not): <https://qwenlm.github.io/blog/qwen2.5/>
- Splink, probabilistic record linkage (MIT): <https://github.com/moj-analytical-services/splink>, term-frequency adjustments <https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html>
- (Almost) All of Entity Resolution (Science Advances): <https://www.science.org/doi/10.1126/sciadv.abi8021>
- In-context clustering-based ER with LLMs: <https://arxiv.org/pdf/2506.02509>
- Pointwise vs listwise vs cross-encoder reranking: <https://zeroentropy.dev/articles/should-you-use-llms-for-reranking-a-deep-dive-into-pointwise-listwise-and-cross-encoders/>
- Entity linking with dual and cross-attention encoders (softmax over candidates): <https://arxiv.org/pdf/2004.03555>
- Jellyfish, LLM for data preprocessing (licence caveats): <https://arxiv.org/abs/2312.01678>
- libpostal (MIT, trained on OpenStreetMap, so external-data caveat): <https://github.com/openvenues/libpostal>
