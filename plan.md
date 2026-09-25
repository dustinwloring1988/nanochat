# Nanochat AI Scientist v2 — Remaining Work Plan

- **Status:** Fixed-context one-node pretraining and one-/two-stage resume are verified; all expansion work below is gated.
- **Target:** RTX 4060 Ti, 16 GB VRAM
- **Upstream pin:** `96bd51617cfdbb494a9fc283af00fe090edfae48`
- **Verification baseline:** Linux container `125 passed, 10 skipped`; focused AI/loader/checkpoint/curriculum suite `72 passed`.
- **Data policy:** Use ignored repository `data/` for pilot assets; keep `C:\Users\dusti\.cache\nanochat` separate and unchanged.

This file contains only unresolved work and constraints. Completed implementation and incident history belong in `postmordum.md`.

## Hard boundaries

- Do not generate or publish papers, report PDFs, citations, or peer-review documents.
- Do not run supervised fine-tuning without a separately approved SFT plan.
- Do not apply generated patches, promote generated checkpoints, or write to the trusted cache automatically.
- Do not commit or push without explicit user action.
- Do not upload artifacts or traces automatically.
- Do not use LLM-generated metric parsing.
- Do not reuse desktop/CLI authentication as a provider API credential.
- Keep candidate patches, checkpoints, generated code, and tool output untrusted until clean-environment human review.

## Remaining research gates

- [ ] Run a real multi-stage BFTS experiment with at least two executed descendant nodes; the current fixed-context curriculum proof is not a BFTS descendant proof.
- [ ] Run a short multi-node search and sequentially confirm a shortlist with seeds `42`, `43`, and `44`; do not weaken `agent.max_nodes=1` without a new safety review.
- [ ] Predeclare the statistical acceptance threshold and candidate-comparison protocol before any search expansion.
- [ ] Enable dynamic context or long-context curriculum only after bucketed shapes, effective batch, scheduler budget, checkpoint migration, VRAM, and exact-resume tests pass.

## Security and promotion work

- [ ] Separate controller and generated experiment execution into distinct services or OS privilege domains.
- [ ] Give each node an OS-enforced private workspace and cache; shared writable `experiments/` access is not a tenant boundary.
- [ ] Implement the human-approved promotion workflow: signed manifest, clean-environment reproduction, explicit review, and no automatic patch/cache/git writes.
- [ ] Keep the external cache separate; any future promotion requires a separate human-approved data decision.

## Deferred follow-ons

- [ ] Draft and approve a separate SFT plan covering data manifests, checkpoint schema, evaluation, resource budget, and rollback.
- [ ] Implement the redacted trace design only after review: versioned local JSONL, deterministic secret redaction, untrusted-content marking, retention/access policy, and leak tests.
- [ ] Implement long-context curriculum research only after the dynamic-context gate above passes.
- [ ] Do not implement paper/report generation under the current safety boundary.

## Operational acceptance

- [ ] Re-run the Linux authoritative suite after any future change: `docker compose --profile ai-scientist run --rm ai-scientist python -m pytest -q -rs`.
- [ ] On native Windows, run the supported subset: `python -m pytest -q --ignore=tests/test_execution.py`; Linux remains authoritative for the Unix sandbox.
- [ ] Before any live provider work, run explicit model preflight and keep OpenRouter missing-usage mode opt-in (`AI_SCIENTIST_ALLOW_MISSING_USAGE=1`).
- [ ] Never commit credentials, generated checkpoints, provider traces, or unredacted artifacts.

## Expansion gate

Do not expand beyond the verified fixed-context, one-node scope until the applicable research, security, promotion, and statistical gates above have explicit approval and reproducible evidence.
