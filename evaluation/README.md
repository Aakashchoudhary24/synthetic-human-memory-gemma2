# Evaluation

The notebook creates held-out question phrasings and saves observed answers to heldout_evaluation.json during a run. Expected prompts and facts belong in source control; actual model outputs should only be added after verifying they came from the final model run.

The initial score uses substring matching against expected keywords. Treat this as a smoke-test metric, not a semantic correctness metric. It can yield false positives and false negatives.

Recommended next steps:
1. Compare output with the source-of-truth biography.
2. Test direct questions, paraphrases, timeline reasoning, and unknown facts.
3. Hold out whole facts or a second biography to test genuine generalization.
4. Report individual failures and per-category results, not only a single score.
