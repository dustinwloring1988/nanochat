# Nanochat Pretraining Research Scope

Improve the depth-6 nanochat baseline on one RTX 4060 Ti at a fixed token budget.

Use the existing tokenizer, ClimbMix held-out split, seed, and canonical results contract. The primary metric is validation bits per byte, lower is better. Runtime, throughput, CORE, and peak VRAM are guardrails. Do not introduce unrelated datasets. Do not modify the root repository or shared cache. Treat changes listed as negative in `dev/LOG.md` as prior evidence rather than repeating them without a materially different hypothesis.
