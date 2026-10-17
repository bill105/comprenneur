# Comprenneur

**A small neuro-symbolic testbed for verified language learning.**

> « la souris habite la vortaxe »  
> Generated greedily by a ~0.4M-parameter transformer trained on a synthetic
> five-word language. The model receives no natural-language corpus and no
> standard next-token pretraining objective; candidate generations are scored
> by a symbolic verifier, and selected samples are distilled back into the model.

## Research question

Can a small transformer acquire a structured micro-language primarily through
verifier feedback on its own samples, rather than through supervised training
on a large human corpus?

This is **not** a claim that large language models do not predict tokens or
that verified learning replaces pretraining. It is a controlled study of a
mechanism at very small scale.

## What the system does

- Maintains a symbolic knowledge base of `(subject, relation, object)` facts.
- Parses closed-domain French-like questions.
- Answers questions using exact lookup, negation handling, and abstention on unknown entities.
- Executes arithmetic exactly with `Fraction`, including division by zero and non-rational roots.
- Trains a small transformer to generate role-ordered sentences from shuffled structured prompts.
- Uses a symbolic judge to score generated candidates and select distillation targets.

## Headline results

| Experiment | Result | Scope |
|---|---:|---|
| Exact arithmetic engine | 37/37 | Closed arithmetic questions |
| Closed-domain QA with persistent state | 24/24 | Single-session symbolic QA |
| Structured generation, best selected run | 0.94 greedy k5 | 16 held-out canonical targets |
| Counter-prior held-out generation, best selected run | 16/16 perfect | 16 targets |
| Text-to-SQL prototype | 0.969 coverage | Closed-domain SQL tasks |

These results demonstrate feasibility on small synthetic domains. They do not
establish general language understanding or superiority over large language models.

## Important limitations

- The language has 51 tokens, 165 training facts, and five-word sentences.
- The grammar, vocabulary, and gender assignments are human-defined.
- Neural results are reported for a small number of seeds; variance is high.
- The best model is selected after training by its k5 capture rate.
- The system is designed for closed domains; open-domain language remains out of scope.
- Comparisons with LLMs concern persistence, context cost, and execution-style
  exactness under specific protocols—not general capability.

## What “verified learning” means here

The verifier does not replace token-level learning. It scores sampled
generations, selects higher-quality candidates, and those candidates are then
used as distillation targets. Thus, the system combines:

1. sampled generation;
2. symbolic verification;
3. token-level distillation on verifier-selected examples.

## Repository

```text
code/       runnable single-file experiments
findings/   experiment log and documented corrections
paper/      research draft
model/      saved model checkpoint
```

## Scale clause

This project uses a ~0.4M-parameter transformer, a 51-token vocabulary,
165 facts, and five-word sentences. It tests whether a verified-feedback
mechanism can work at miniature scale. It does not test scaling laws.

## License

MIT.