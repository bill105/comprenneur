# comprenneur
Un apprenant du langage par pure      vérification — construction contre prédiction, mesuré
# COMPRENEUR

## A verified-only language learner

> **« la souris habite la vortaxe »**
>
> *Written greedily by a 0.4M-parameter transformer that has **never seen a human sentence, never predicted a token** — trained only by verdicts on its own sampled generations. Gender agreement follows the world's arbitrary assignment (hence « la jardin »), proving no linguistic prior leaks. Held-out target, never trained, never captured. Entry 78-c, seal `5293734fe4`.*

---

### The question

Can a language model learn to **generate** without ever **predicting**?

Modern LLMs learn by next-token prediction on billions of sentences. RLVR-style methods (DeepSeek-R1) refine predictive models with verified rewards — but always start from a predictive base.

This project tests the pure extreme: **from scratch, zero corpus, zero human sentences, zero prediction objective.** The only supervision is a judge scoring the model's own samples.

---

### The answer, in one table

| Result | Value | Seal |
|---|---|---|
| Generation without prediction (5-word structured language) | greedy k5 = **0.94 / 0.94** (canonical / counter-prior) | `5293734fe4` |
| Held-out counter-prior sentences perfect | **16/16** | `5293734fe4` |
| Distribution over 6 seeds | composition **6/6** · full cascade 3/6 | `5293734fe4` |
| Arithmetic by execution (parser + Fraction engine) | **37/37**, 0.01 s, 0 tokens | `885a24b5e9` |
| Closed-domain QA with persistence | **24/24**, 0 tokens, survives re-session | `d3c505b31b` |
| Text-to-SQL (parser + judge + nursery) | 0.969 coverage | — |

---

### The five laws of verified learning (measured)

1. **The gradient does not converge to order — it captures it.**
   Five reward-hacking attractors, each closed arithmetically (silence 0.45≈0.50 · action-only 0.16≈0.167 · DET-farming · subsidized stopping · bag-of-words 0.50=2/4).

2. **An escape cascades only if the policy already touches it.**
   The grammar window is the pre-saturation phase: seeds capturing a k4 sentence early cascade to 165/165; the seed capturing after k3 consolidation never does (mechanistic fork, n=6).

3. **What contradicts the prior is memorized verbatim, never composed.**
   At 43 facts: counter-prior composition = 0.00, exceptions parroted exactly.

4. **Do not force, feed.**
   Forcing balanced capture yields 0.00; a 70%-counter-prior world of 165 facts yields 0.69-0.84. The prior occupies the least-supported slot — the world is the remedy, per slot.

5. **Every frontier raise pays a valley before the explosion.**

---

### What alignment already covers — and what only construction guarantees

Against Gemini 1.5, same questions, controlled conditions:

- **In context**: aligned LLMs read structure perfectly (12/12, including role-permutations against the lexical prior) and compute large products exactly. The lexical bias is a disease of weights, not of reading.

- **What survives measurement**: persistence across sessions (5/5 at 0 tokens vs. honest abstention at utility 0) · cost (K·B per session vs. B once for life, ×13.9 measured stateless) · **the echo** — 10÷4 → « 10/4 » (what it looks like) vs. **5/2** (what it is): the smallest demonstration of prediction vs. execution. Three divisions.

---

### The protocol

127 pre-registered experiments · 54 documented self-corrections ("bites") · hash-sealed instruments · distribution reporting · leak probes (one LLM measurement invalidated by its own probe before any verdict was consumed). The protocol refuted the author's own bets three times on aligned-LLM capabilities — reported as such.

---

### Repository structure

```
paper/      the full draft (title → limitations → conclusion)
code/       runnable entries (plain PyTorch, single-file each,
            fresh-kernel autonomous, ~20-40 min on a free T4)
findings/   the structured constats and the bite journal
model/      the saved complete run (78-c, seed 3)
```

---

### The scale clause (explicit)

0.4M parameters · 51 tokens · 165 facts · 5-word sentences · one free T4.

This is a demonstration of **mechanism**, not of scaling laws. Whether verified-only learning survives scale is the open question this repository deposits — with a miniature that proves the mechanism exists.

---

### License

MIT. Everything regenerable from sealed inputs.
