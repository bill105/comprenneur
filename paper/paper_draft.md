# Learning to Generate Without Prediction: A Verified-Only Transformer Trained by Its Own Judged Samples

---

> **« la souris habite la vortaxe »**
>
> *— written greedily by a 0.4M-parameter transformer that has never seen a human sentence, never predicted a token, and was trained only by verdicts on its own sampled generations (gender agreement follows the world's arbitrary assignment, hence "la jardin"; Entry 78-c, seal 5293734fe4).*

---

## Abstract

Whether language models can acquire meaning from statistical prediction alone remains contested. We present a measurement-first study of the alternative extreme: learning driven exclusively by verification. Across 127 pre-registered experiments with adversarial self-auditing (54 documented self-corrections, hash-sealed instruments), we decompose what prediction, imitation, and verification each contribute to language acquisition. (1) A 0.4M-parameter transformer trained ONLY by verdicts on its own sampled generations — no corpus, no human sentence, no next-token loss — learns word order, determiners, and arbitrary gender agreement, and generalizes to unseen combinations in both distributional directions (canonical 0.94, counter-prior 0.94, greedy; 16/16 held-out perfect sentences; distribution over 6 seeds: composition robust 6/6, full grammar cascade 3/6 with a mechanistic fork). (2) We measure the failure modes of naive judges (five reward-hacking attractors, each closed arithmetically) and the laws that govern verified learning: the judge determines the learner; what contradicts the prior is memorized, never composed; the prior occupies the least-supported slot; seed early or never. (3) Controlled A/B tests against aligned LLMs (Gemini 1.5) isolate what verification-based systems guarantee (structural exactness, zero-token persistence, declared unknowns, 37/37 arithmetic by execution vs. an "echo" failure: 10÷4 → "10/4") versus what alignment already covers. We release all code, seals, and the 127-report log. The open question — does verified-only learning scale — is deposited with a miniature that proves the mechanism exists.

---

## 1. Introduction

### 1.1 The contested question

Do large language models acquire meaning from statistical prediction? The debate is old (stochastic parrots versus emergence) and largely unresolved, in part because the standard evidence — benchmark scores — cannot separate three confounded factors: the training objective (next-token prediction), the corpus distribution, and scale. A model that both predicts and scores well demonstrates only that prediction is SUFFICIENT for the benchmark, not that it is necessary for meaning, nor that its failure modes are benign.

This paper takes the complementary route: a small-scale, controlled decomposition. We hold the world fixed and vary the learning signal — prediction, imitation, verification — measuring not only what each achieves but where each fails, and whether the failures are properties of the objective or of the learner.

### 1.2 Three contributions

**(i) A decomposition of the lexical bias.** Across a factorial design varying loss (prediction vs structural verification), corpus (balanced vs skewed), and scale (0.4M vs 1M), the tendency of networks to encode word-role statistics over structural roles survives every variation — but is displaced, not removed, by every remedy we test. Its deepest measured form: a network trained to GENERATE by pure verdict memorizes verbatim the facts that contradict its corpus prior and composes 0.00 beyond them — until the corpus itself is rebalanced, whereupon composition appears (0.69-0.84). The bias is a property of the DISTRIBUTION, occupied at whichever slot the world supports least. The converse finding, "the judge determines the learner" — identical learners under three judges produce three bias levels (0.719 / 0.450 / 0.775) — localizes a controllable lever.

**(ii) Verified-only generation.** A 0.4M-parameter transformer whose entire education is verdicts on its own sampled outputs — no corpus, no human sentence, no next-token loss — learns word order, determiners, and arbitrary gender agreement, and generalizes to unseen combinations in both distributional directions. The emblem of this paper is its sealed output sentence, produced greedily on a held-out counter-prior target: "la souris habite la vortaxe". To our knowledge this is the first demonstration of grammatical structured generation by a model with no prediction objective and no corpus exposure. The path there required characterizing five reward-hacking attractors (each closed arithmetically) and five laws of verified learning, including the fork that governs whether the grammar cascade completes (3/6 seeds at the stated budget).

**(iii) A measurement-first protocol.** 127 pre-registered experiments, adversarially self-audited, with 54 documented self-corrections ("bites") — including one pre-registered proof retracted for non-replication and one measurement invalidated by its own leak probe before any verdict was consumed. The protocol corrected the author's wrong bets three separate times on the capabilities of aligned LLMs; we report it as a transferable instrument, and its failures as evidence of its functioning.

### 1.3 What this paper is not

It is not a scaling claim (Section 6.1), not a benchmark (6.2), and not an argument that prediction cannot acquire meaning (6.5). It is a controlled anatomy of three learning signals on closed, verifiable domains — and the deposit of one open question: does verified-only learning scale?

---

## 2. Method: the measurement-first protocol

### 2.1 Pre-registration

Every experiment is drafted before execution in a fixed format: context, design, explicit hypotheses (H-x) with numeric thresholds, the author's stated bet, pre-registered branches (a/b/c/d) covering success, partial success, and failure — including, where applicable, the failure's expected mechanism. No verdict is consumed on a measurement whose reading was not declared in advance. The protocol's visible product: three of the author's own bets against aligned LLM capabilities were refuted by measurement (Entries 59-b, 74, 77) and are reported as such.

### 2.2 The bite journal

Every execution error, design flaw, or invalid measurement is logged as a numbered "bite" with its cause and repair — 54 to date, 19 during the generation arc alone. Canonical examples: a proof declared on n=1 and retracted under replication (bite 42); a "new conversation" that was not a virgin context, caught by a pre-registered probe before verdict (bite 36, Entry 72); a display column computed on the wrong object (bites 38, 52). The journal is annexed. Its function is dual: an audit trail for the single-author conflict of interest (Section 6.4), and an empirical claim — a method that has never bitten its author has not been run long enough.

### 2.3 Seals

Critical instruments are fingerprinted (MD5 of the corpus, secrets, and protocol) and reported with each result: d3c505b31b (Entry 72), d9c1ebc576 (75-78), 885a24b5e9 (77), 5293734fe4 (78-c). Complete models are saved (model/modele_78c_complet_g3.pt). Any reported number is regenerable from the sealed inputs.

### 2.4 Adversarial validation

Closed-world systems are probed by design: presupposed non-existents, negations, permutations contradicting the corpus prior, malformed input, and — for LLM comparisons — leak probes asking the model to justify its knowledge. The standing rule (no measurement without a probe that could invalidate it) is applied to the author's own claims with the same rigor as to the systems under test.

### 2.5 The scale clause

Every experiment carries an explicit statement of what its scale falsifies — the mechanism, not the scaling laws — and what remains open. This clause is load-bearing: it is what permits honest small-scale claims (Section 6).

### 2.6 Distribution reporting

Following the project's law "the standard is measured within the run," headline neural results are reported as distributions over seeds (n=6 for the final result: [0, 161, 0, 165, 0, 165]), never as single best runs. LLM comparisons are single-run controlled demonstrations and are labeled as such.

---

## 3. Construction beats prediction on closed domains

*(Section to be written from Entries 70-90 — the parser 42/42 at 0.0 s and 0 tokens vs the LLM 42/42 at 32 min GPU and 5000 tokens; the five measured walls and their repairs; the parser-pure control; the six nuclei.)*

---

## 4. The lexical bias is structural

*(Section to be written from Entries 87-103 — the factorial decomposition loss × corpus × scale; the inverted bluff (entropy ratio 0.65×); the judge determines the learner (0.719 / 0.450 / 0.775); the hold-out 1.000.)*

---

## 4-bis. Generation without prediction: from attractors to the complete miniature

### 4-bis.1 Setup

A 0.4M-parameter transformer (2 layers, 128 dims, 4 heads) must MAP an unordered, shuffled set of role-tagged words (TAG_S/S, TAG_A/A, TAG_O/O) to the grammatical surface form "DET-nom A DET-nom", where DET is an arbitrary, fixed, per-word gender assignment. The model sees no corpus, no human sentence, and no next-token objective — ever. Its only supervision is a judge scoring its own sampled generations. We report greedy decoding on held-out secrets: subject-action-object triples never trained, never captured, never validated.

### 4-bis.2 The attractor regime (signal alone)

REINFORCE on the raw verdict converges to degenerate optima — five of them, each closed arithmetically to within noise: silence (R̄ 0.45 ≈ optimum 0.50), action-only (0.16 ≈ 0.167), DET-farming (2.0/5 = action + 1 free grammatical credit), subsidized stopping (1.5/6 ≈ 1.33), and the conditioned bag-of-words (0.50 = 2/4). Each patch of the judge DISPLACES the attractor; none empties the field. Reward hacking is not an artifact of scale or of prediction: it is the default behavior of any gradient learner facing an incomplete payment lattice.

### 4-bis.3 The escape-cascade mechanism (signal amplified)

Replacing the scalar reward with a five-rung ladder (action → target-words → order → determiners → gender) and distilling ONLY the frontier of a bank of self-generated, judge-validated sentences (sample → judge → capture → distill) produces a cascade: a single early k5 escape (seed 3: update 2329) saturates all 165 facts within 3000 updates. The mechanism obeys five measured laws: (L21) an escape cascades only if the policy already touches it; (L22) what contradicts the prior is memorized verbatim, never composed; (L23) the prior occupies the least-supported slot (word → type → action frequency); (L24) do not force, feed — forcing capture at 43 facts yields 0.00 counter-prior composition, while a 70%-counter-prior world at 165 facts yields 0.45-0.84; (L25) every frontier raise pays a valley before the explosion.

### 4-bis.4 The final miniature (Entry 78-c)

Over 6 seeds (12,000 updates each): composition is robust (6/6 canonical order 0.57-0.90; 6/6 counter-prior 0.69-0.84); the full grammar cascade occurs in 3/6, with a mechanistic fork — seeds capturing a k4 sentence BEFORE k3 saturation cascade to 165/165; the seed capturing after (update 6437) never does. The best complete run: greedy k5 = 0.94 canonical / 0.94 counter-prior, 16/16 held-out counter-prior sentences perfect, including arbitrary gender agreement that contradicts French ("le brique", "la jardin") — proof that no linguistic prior leaks: the network agrees with the WORLD, not the language.

### 4-bis.5 The sealed sentence

"la souris habite la vortaxe" — a non-agent noun as subject (read from TAG_S against the corpus prior), an agent noun as object (read from TAG_O), determiners and arbitrary gender agreed, on a combination never encountered, written greedily by a network whose entire education was verdicts on its own samples. To our knowledge, this is the first demonstrated generation of grammatical structured language by a model trained with no prediction objective and no exposure to any human corpus. The scale clause is explicit: 165 facts, 51 tokens, 5-word sentences, one T4. Whether the mechanism survives scale is the question this paper deposits, not the one it answers.

---

## 5. What alignment already covers, and what only construction guarantees

### 5.1 Motivation

A recurring objection to verification-based architectures is that modern aligned LLMs "already handle" abstention, presupposition, and grounded answers. This section reports three controlled A/B tests that separate what alignment achieves in context from what construction guarantees structurally. All tests use the same protocol family: identical questions, same conditions, a symbolic system (the Comprenneur: role-based parser + typed graph + declaration gate P1) versus an aligned LLM (Gemini 1.5).

### 5.2 Test 1 — Memory, persistence, and the cost of the memo (Entry 72)

A closed world of 16 agents (11 pseudowords, blocking prior knowledge), 37 statements, a mid-session update of 3 entities, and 24 questions in five tranches. Results (Gemini 1.5 with memo): 18/19 correct, zero hallucination on presupposed non-existents, update integrated on the fly. The single failure is a convention divergence, not an error: the two systems do not inhabit the same world (CWA vs OWA, Reiter 1978). Results (re-session, memo absent): 5/5 honest abstention, utility 0. The Comprenneur: 5/5 correct at 0.01 s and 0 tokens, because its graph persists. Accounting: a useful LLM pays its base at every session (K·B; measured ×13.9 stateless); the Comprenneur pays once (B) for life. A pre-registered leak probe caught an undeclared cross-conversation memory — the first session-2 measurement was invalidated BEFORE any verdict was consumed. Declaration: an LLM's state does not declare itself; it is probed.

### 5.3 Test 2 — Reading structure under an exact judge (Entry 74)

The slot-tagging task, posed to Gemini 1.5 with the base in context: 12/12 phrases tagged perfectly, INCLUDING four role-permutations where the lexical prior contradicts the position. Unknown pseudowords: declared 2/2. Reading: the lexical bias is a disease of WEIGHTS, not of reading. Given the base in context, an aligned LLM reads structure as well as the parser. The Comprenneur's advantage on reading is not skill but STATE — permanence and cost — not in-context accuracy.

### 5.4 Test 3 — Arithmetic: prediction echoes, execution computes (Entry 77)

37 questions across six tranches. Comprenneur (parser + Fraction engine + gate): 37/37, 0.01 s, 0 tokens; fractions stay exact (7/2, 5/2, 25/2 — never a float); equations return both roots. Gemini 1.5 (one line, no scratchpad): 35/37 strict, 37/37 numeric. The pre-registered bet (≥2 failures on large products) was REFUTED: all products exact. The two strict failures are the finding: for reducible divisions, Gemini returns the echo of the operands — 10÷4 → "10/4", 100÷8 → "100/8" — while the engine returns the normalized value (5/2, 25/2). For irreducible 7÷2, echo and value coincide. The prediction answers "what does 10 divided by 4 look like"; the execution answers "what is 10 divided by 4". Three divisions constitute the smallest controlled demonstration in this paper of the prediction/construction difference.

### 5.5 The five axes of the guarantee table

| Axis | Aligned LLM (in context) | Construction |
|---|---|---|
| honesty | high (measured, sampled) | structural (gate) |
| utility in context | high | high |
| re-session | 0 (honest abstention) | full, 0 tokens |
| cost | K·B per session + tokens | B once, 0 tokens |
| error address | diffuse (weights) | one line, revocable |
| arithmetic | echoed surface possible | exact by execution |

The synthesis is deliberately narrow: on a closed domain with verifiable truth, an aligned LLM in context is nearly indistinguishable from construction on SCORE. The differences that survive measurement are not scores but PROPERTIES: persistence across sessions, cost linear in sessions versus constant, errors with addresses, and exactness where the answer requires normalization rather than imitation.

---

## 6. Limitations (the scale clause, integral)

### 6.1 Scale of the miniature

All neural results in Section 4-bis operate at 0.4M parameters, 51 tokens, 165 facts, 5-word sentences, single T4 (12k updates, ~40 min per seed). This is a demonstration of MECHANISM, not a claim about scaling laws. Whether verified-only learning survives at 10⁸-10⁹ parameters is the open question this paper deposits. We explicitly do not claim it.

### 6.2 Sample sizes

Neural experiments: n=2 seeds for the mechanism entries, n=6 for the final distribution. The distribution [0, 161, 0, 165, 0, 165] is honest but small; the mechanistic fork is consistent across all 6 runs but has not been tested at n≥10. LLM comparisons: one model family, one run each, manual protocol. These are controlled demonstrations, not benchmarks.

### 6.3 Closed worlds

Every symbolic result lives on domains with verifiable ground truth. Open-domain language remains prediction's home territory. The Comprenneur delegates there by design (flag ▐); this paper does not measure the delegation loop.

### 6.4 The judge is authored

Every judge in this paper was designed by the same author whose hypotheses it tested. The 54 documented self-corrections are the audit trail of that conflict of interest — including one pre-registered proof retracted for non-replication and one measurement invalidated by its own leak probe before any verdict was consumed. The protocol is the defense; it is not a substitute for independent replication.

### 6.5 What is NOT claimed

We do not claim: that prediction cannot acquire meaning; that aligned LLMs hallucinate in context (measured: they largely do not); that the lexical bias disappears at scale (untested); that the miniature's cascade frequency transfers to any other configuration. We claim exactly: on closed verifiable domains, at small scale, under a measurement-first protocol, verified-only learning produces grammatical, generalizing, counter-prior generation; prediction and construction have measurably different failure envelopes; and the judge determines the learner.

---

## 7. Conclusion

Four findings summarize this paper. (1) On closed verifiable domains, construction — a role-based parser, a typed graph, a declaration gate — achieves perfect coverage at zero inference cost, with abstention on the unknown as a structural property. (2) The lexical bias is not an artifact of prediction: it survives the loss, the corpus, and the learner's size, occupying whichever slot the distribution supports least; it is displaced by remedies, never deleted, and it is governed by the judge — the judge determines the learner. (3) Verified-only generation exists in miniature: a network that never predicted a token writes "la souris habite la vortaxe" on an unseen counter-prior target, with arbitrary gender agreement taken from the world, not the language — through a mechanism of sampled escapes, judged capture, and frontier distillation, whose attractors and laws we map. (4) The differences that survive against aligned LLMs are not scores but properties: persistence across sessions at zero cost, errors with addresses, exactness where normalization is required — an echo ("10/4") versus a value ("5/2").

The synthesis is deliberately narrow. Meaning, at this scale of evidence, is not located in prediction — nor in statistical learning under any loss we tested — but in verifiable construction: knowledge built occurrence by occurrence under a judge, revocable at its address, persistent across sessions, declared when absent. The Comprenneur is not an alternative to LLMs; it is the component they do not have — the memory that persists, the error that has a street address, the door that says "unknown."

The open question is deposited with its miniature: does verified-only learning scale? The cascade completed on 3 of 6 seeds at 0.4M parameters; its window is the pre-saturation phase; its food is a world that supports every slot. What happens at 10⁹ parameters, with real tokenization and a world that grows as the learner does, is the experiment this paper hands forward.

---

## References

- Reiter, R. (1978). On Closed World Data Bases.
- Chollet, F. (2019). On the Measure of Intelligence (ARC).
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks.
- Geirhos, R. et al. (2020). Shortcut Learning in Deep Neural Networks.
- McCoy, R. T., Pavlick, E., Linzen, T. (2019). Right for the Wrong Reasons.
- Zelikman, E. et al. (2022). STaR: Bootstrapping Reasoning With Reasoning.
- Lambert, N. et al. (2024). Tülu 3 (RLVR).
- DeepSeek-AI (2025). DeepSeek-R1.
- Kadavath, S. et al. (2022). Language Models (Mostly) Know What They Know.
- Angelopoulos, A. N., Bates, S. (2021). A Gentle Introduction to Conformal Prediction.

---

## Annexes

- A. The seals (d3c505b31b · d9c1ebc576 · 885a24b5e9 · 5293734fe4)
- B. The bite journal (54 entries — findings/bites.md)
- C. The code (code/ — runnable entries)
- D. The saved model (model/modele_78c_complet_g3.pt)
