Here is the exact transcript of the video:

---

### **Amazon ML Challenge 2026: Business Entity Resolution**

**[00:00] Introduction**  
Welcome to the Amazon ML Challenge 2026. This year's problem is Business Entity Resolution, a fundamental and widely encountered problem in real-world data. You will be given business records that arrive from three independent sources, each noisy and inconsistent, and the goal is to determine which of them describe the same real-world business. In this video, I will walk you through the problem statement, the dataset, the scoring, the submission format, and how entries are judged.

---

**[00:29] The Problem: From Sign-up to Matched Records**  
Let's start with where this data comes from. Consider a business signing up on Amazon Business. At sign-up we capture its core details, such as the business name and address. To build a richer picture of that same business, we pull in additional information from other data providers. Each of these sources typically comes from a different data vendor, with its own formats and conventions. The difficulty is that these external sources share no common identifier with ours, so the only fields we can rely on are the business name and address. For this challenge, we have deliberately limited the data to names and addresses.

Here is the core difficulty. The same real-world business is described differently by each source. One vendor may write *Acme Robotics Incorporated*, another abbreviates the address, and a third references a nearby landmark. Source 1 is our clean, deduplicated reference list. Sources 2 and 3 are the noisy fragments that must be reconciled against it, and there is no shared identifier linking them.

Entity resolution is the task of linking these records together, establishing that differently written records all refer to the same business. For every entity in Source 1, the goal is to find all of its matching records in Sources 2 and 3. A Source 1 entity may match many records, exactly one, or none at all.

---

**[01:49] Blocking & Matching Pipeline**  
Comparing every Source 1 record against every Source 2 and Source 3 record would be far too expensive at scale. So we first apply blocking, sorting records into buckets using a cheap key built from both the name and the address, so that records likely to match land in the same bucket. Each color here is one such block. Notice that every record carries its own ID, a noisy name, and an address, and records can group either through a similar name or through a shared address. We follow the orange block, built around Acme Robotics, and the records in it become the candidate matches for that entity. 

Blocking favors recall, so the bucket also pulls in look-alikes: a business with a similar name at a different address, and a different business that happens to share an address. The matching model removes those in the next step. Blocking narrows an enormous number of possible comparisons down to a manageable set of candidate pairs.

Finally, a matching model scores each candidate pair and keeps only the true matches, discarding the rest. That produces the two outputs of this challenge: the candidate pairs from your blocking stage, and the final matching results, which is the file scored on the leaderboard.

---

**[03:00] The Dataset: Train to Learn, Test to Prove It**  
You are provided with two datasets. The training set contains records across all three sources, together with the ground-truth labels—a file that specifies for each Source 1 business exactly which Source 2 and Source 3 records it matches. The test set contains the same three sources, but no labels, and this is what you generate predictions for. 

A note on the label format: The ground truth is stored as one row per Source 1 entity. Its ID mapped to a comma-separated list of all of its matching IDs, so a single row carries that entity's complete match set, and the list is empty when the entity matches nothing. This mirrors exactly what you submit. 

One practical reminder: all files are tab-separated, so read them with an explicit tab separator; otherwise, the columns will not parse correctly.

---

**[03:49] Submission Requirements: A File, Then a Package**  
There are two deliverables. Throughout the challenge, you upload a single file, `matching_results.tsv`, containing one row per Source 1 entity with its predicted matches. This is the only file scored on the leaderboard. 

At the close of the challenge, every team also submits a single archive. It contains the final matches, along with `candidate_pairs.tsv`—the candidate set your blocking stage produced before the model narrowed it down. This file is not scored, but it is used to audit the quality of your blocking. The archive also includes your complete, runnable pipeline and a methodology document describing your approach. The packages of the top teams are reviewed in detail before final rankings are confirmed. 

One reminder: run the provided validation script before submitting, so a simple formatting error does not cost you a submission.

---

**[04:39] Judging, Scoring & Tips**  
Finally, how entries are judged. Submissions are scored using the macro $F_{0.5}$ metric, which weights precision twice as heavily as recall:

$$\text{F}_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

In practical terms, incorrectly merging two different businesses is penalized roughly twice as much as missing a true match. So when in doubt, it is safer not to merge. 

A few recommendations:
1. **Understand singletons:** A singleton is a Source 1 entity that has no matching record in Source 2 or Source 3. If you correctly predict an empty list for it, you earn a full score of 1 on that entity; if you predict any match, you score 0. So identifying the businesses with no match is just as important as finding the ones that do match.
2. **Blocking sets your recall ceiling:** Your blocking strategy sets the ceiling on the recall you can achieve, so invest in it first, because you cannot match a record you never consider.
3. **Regional patterns:** Pay attention to region-specific patterns in both names and addresses.

One firm rule: this is a pure machine learning challenge, so external databases, APIs, and lookups are strictly prohibited. Use only the provided data.

---

**[05:46] Conclusion**  
That is the challenge. Build something you are proud of, resolve those entities, and enjoy the process. We are excited to see what you create. All the best.