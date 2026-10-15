# The Structured Findings — 127 Numbered Constats

The full numbered log (1-127, in French) lives in the author's journal. This file indexes the load-bearing findings, by arc.

---

## The measurement protocol (Constats 1-32)

- **The loss lies; only exact match tells the truth.** Measured at the deepest level: loss 0.08 on known examples, 0/172 on unseen outputs (Entry 13c) — a model can memorize without generalizing.
- **Pre-registration or it didn't happen.** Every experiment drafted before execution: hypotheses with numeric thresholds, the author's bet, and pre-registered branches (a/b/c/d) covering failure.
- **The standard is measured within the run** — never inherited from a past number.

---

## The parser versus the LLM (Constats 33-90)

- **33 — the bag-of-words tombeau**: the topological representation reduces to a bag-of-words at the default window; no window setting beats the frozenset witness. The anti-bag-of-words principle is born.
- **48 — the wolf and the dog**: the wolf occupies the AGENT slot exactly like the dog. Structural understanding is independent of the lexicon — later confirmed at the neural level (hold-out tokens classified 1.000, Entry 59-A-b).
- **The headline**: parser 42/42 at 0.0 s and 0 tokens; LLM 42/42 at 32 min GPU and 5000 tokens. Same score, different worlds.
- **The five walls**: channel, engine, routing, verb, grammar — each measured, each repaired, each logged.

---

## The lexical bias (Constats 87-103)

- **87-88**: the LLM's knowledge is statistical, reset per query; the Comprenneur's is structural, persistent, verifiable.
- **98-99 — the bias is structural**: it survives the loss (prediction vs verification), the corpus (balanced vs skewed), and the scale (1M). Equilibrating the corpus does not cure it (0.703 → 0.719, within noise).
- **100 — the inverted bluff**: after equilibration, confidence is anti-correlated with knowledge (entropy ratio 0.65×). The neural "I don't know" does not exist naturally; it must be constructed.
- **102-103 — the judge determines the learner**: same network, same corpus, three judges, three bias levels (0.719 fixed-labels / 0.450 frozen-graph / 0.775 flexible-graph). The judge that memorizes every usage and lets position decide produces a structural learner.

---

## The guarantees (Constats 105-108)

Against Gemini 1.5, identical questions, controlled conditions:

- **Honesty is equal**: aligned LLMs declare the unknown (presupposed non-existents → INCONNU; zero hallucination measured).
- **What survives measurement**: persistence (5/5 at 0 tokens and 0.01 s vs. honest abstention at utility 0 in re-session) · cost (K·B context characters per session vs. B once for life; ×13.9 measured stateless) · the convention divergence (CWA vs OWA — the two systems do not inhabit the same world) · the undeclared state (an LLM's cross-conversation memory does not declare itself; it is probed — one measurement invalidated before any verdict was consumed).

---

## Generation without prediction (Constats 111-127)

- **111 — the attractor regime**: five degenerate optima of the naive judges, each closed arithmetically (silence 0.45≈0.50 · action-only 0.16≈0.167 · DET-farming · subsidized stopping · conditioned bag-of-words 0.50=2/4). Reward hacking is the default behavior of any gradient learner facing an incomplete payment lattice — not an artifact of scale or prediction.
- **116 — the asymmetry**: reading by verification is reliable (0.890, multi-seed); generation by raw signal is an attractor field. The order of roles is what the parser gets by construction and the gradient never converges to.
- **118 — the escape cascade**: with a five-rung judge and frontier distillation, order is captured (one escape), propagated (30/43 facts), and generalized (0.54 on held-out canon) — by a network that never saw a sentence.
- **121 — maturity before capture**: an escape cascades only if the policy already touches it. Seeds capturing early (before k3 saturation) cascade to completion; the seed capturing after never does — the mechanistic fork, confirmed on all 6 seeds.
- **122 — the composition wall**: what contradicts the prior is memorized verbatim, never composed (0.00 at 43 facts, two remedies failed).
- **124 — the wall was the distribution**: at 165 facts with counter-prior majority, composition appears (0.69-0.84). Do not force, feed — the prior occupies the least-supported slot; the world is the remedy, per slot.
- **125-127 — the final distribution and the sealed sentence**: [0, 161, 0, 165, 0, 165] over 6 seeds; the best complete run: greedy k5 0.94 canonical / 0.94 counter-prior; **16/16 held-out counter-prior sentences perfect**, including arbitrary gender agreement taken from the world, not the language ("le brique", "la jardin" — proof that no linguistic prior leaks). The sealed sentence: **« la souris habite la vortaxe »**.

---

## The arithmetic engine (Constat 123)

- **37/37 by execution**, 0.01 s, 0 tokens: exact fractions (7/2, 5/2, 25/2 — never a float), both roots of quadratics, declared impossibilities ("division by zero", "non-rational roots", delegation flag for out-of-domain).
- **The echo**: Gemini 1.5 returns "10/4" and "100/8" for reducible divisions where the engine returns "5/2" and "25/2" — prediction repeats the surface of the operands; execution normalizes the value. For irreducible 7÷2, echo and value coincide. Three divisions constitute the smallest controlled demonstration of the prediction/construction difference in this project.

---

## The five laws of verified learning (as measured)

1. The gradient does not converge to order — it captures it.
2. An escape cascades only if the policy already touches it (maturity precedes capture).
3. What contradicts the prior is memorized verbatim, never composed.
4. The prior occupies the least-supported slot — do not force, feed.
5. Every frontier raise pays a valley before the explosion.

*And the sixth*: the window of the grammar is the pre-saturation phase — seed early, or never seed.