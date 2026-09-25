# Nanochat AI Scientist v2 — Remaining Work Plan

- **Status:** Code hardening and live one-node validation complete for OpenCode and OpenRouter; multi-stage/multi-seed expansion pending
- **Date:** 2026-09-24
- **Target:** RTX 4060 Ti, 16 GB VRAM
- **Upstream pin:** `96bd51617cfdbb494a9fc283af00fe090edfae48`
- **Pilot cache:** ignored repository `data/`
- **External cache:** `C:\Users\dusti\.cache\nanochat` remains unchanged

This file is the actionable backlog. Completed implementation details belong in `postmordum.md`; do not add new completed-phase sections here.

## Hard boundaries

These are permanent safety rules, not deferred tasks:

- Do not generate or publish a paper, report PDF, citations, or peer-review document.
- Do not run supervised fine-tuning experiments until a separate SFT plan is approved.
- Do not merge an AI-generated patch into the root repository.
- Do not commit or push changes automatically.
- Do not promote generated checkpoints into the trusted nanochat cache.
- Do not use LLM-generated metric-parsing code.
- Do not upload generated artifacts or LLM traces automatically.
- Do not reuse desktop/CLI authentication as a provider API credential.
- Candidate patches and checkpoints remain untrusted until a human reviews and reproduces them in a clean environment.

## Verified starting point

- The AI Scientist runtime is vendored and pinned.
- Provider routing, deterministic seeds, curriculum transitions, canonical results, source snapshots, and GPU Docker isolation are implemented.
- The local `data/` cache contains the tokenizer, both ClimbMix layouts, and the evaluation bundle.
- A fixed-context depth-6 run completed at 19,660,800 tokens: validation BPB `1.491058`, peak allocated VRAM `8.39 GB`, approximately `2.63` minutes.
- A same-seed repeat ended at BPB `1.491474`; absolute delta is `0.000416`. The current numerical reproducibility tolerance is `0.001`, not bitwise equality.
- Live one-node BFTS passed with OpenCode `opencode/space-bunny-free` at BPB `1.491380` and OpenRouter `openrouter/stealth/space-bunny-alpha` at BPB `1.491395`.
- OpenRouter's free endpoint omits usage metadata; its live run requires explicit `AI_SCIENTIST_ALLOW_MISSING_USAGE=1`, while API-call limits remain enforced.
- Current verification: Linux container `95 passed, 10 skipped`; focused host AI suite `41 passed, 1 skipped`; Black checks for integration files and compilation pass.

## P0 — close before trusting any BFTS result

- [x] Make guardrails controller-owned and immutable. Candidate code must not be able to raise `max_peak_vram_bytes`, `max_training_seconds`, `required_plot_count`, context, seed, evaluation cadence, or token budget.
- [x] Enforce exact evaluation-token budgeting. Reject non-divisible `eval_tokens` or record and validate the actual evaluated token count.
- [x] Replace candidate-authored metric trust with a controller-owned evaluator or a separately privileged execution boundary. A generated `results.json` must not be sufficient proof by itself.
- [x] Make provenance mandatory: source revision/commit or an explicit immutable source identity, runtime/device details, model, dataset/tokenizer hashes, and all required diagnostic metrics.
- [x] Remove implicit paid API calls. Require an explicit model selection and an explicit preflight/live-call opt-in before contacting a provider.
- [x] Make the one-node safety boundary hard. Unlimited search is disabled until a separate explicit mode exists.
- [x] Enforce the exact safe child environment. Replace broad prefixes such as `HF_`, `WANDB_`, and `NANOCHAT_` with an explicit allowlist; add tests for `HF_TOKEN`, `WANDB_API_KEY`, cloud keys, and provider keys.
- [x] Make worker timeout handling cancel and reap the complete process tree. Do not release the GPU while an orphan trainer can still be running.
- [x] Make the experiments bind mount and artifact path consistent and guaranteed writable by UID 1000. Verify the host path exists before Docker starts.

## P1 — provider and budget work

- [x] Add retries to provider catalog/model-listing calls and make preflight retry behavior explicit.
- [x] Scope the SQLite provider budget to a run ID so repeated runs cannot share counters accidentally.
- [x] Add a live provider preflight for the selected provider, including structured tool-call capability and the selected API mode.
- [x] Run a bounded live call using one supplied `OPENCODE_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY`; never print or persist the key.
- [x] Support OpenRouter free models that omit usage metadata only with explicit `AI_SCIENTIST_ALLOW_MISSING_USAGE=1`; call limits remain active, token limits cannot be enforced for those responses.
- [x] Verify every enabled AI Scientist role uses the selected model, including summary, selection, feedback, and VLM roles.
- [x] Add tests for malformed responses, missing usage, tool fallback, Responses multimodal input, retry counts, and budget exhaustion.
- [x] Keep unredacted prompt/response capture disabled. Do not add a trace store until the redaction design below is implemented.

## P1 — execution and curriculum correctness

- [ ] Run a full multi-stage BFTS test proving an accepted parent source snapshot seeds descendants; the current targeted archive/reuse test is not sufficient.
- [ ] Make multi-source dataloader resume exact, including per-rank cursors, buffered documents, pending packed data, and stage transitions.
- [x] Reject resume when model shape, curriculum schedule, seed, world size, token horizon, or dataloader configuration differs.
- [x] Add stage/source composition metadata and a source-mixture plot to canonical results.
- [x] Decide whether MFU is `null`/unknown or add a verified RTX 4060 Ti peak-FLOPS value; do not report unknown MFU as a measured zero.
- [x] Decide whether training-time throughput should include compilation/evaluation time or report separate active-training and wall-clock metrics.

## P1 — live pilot

- [x] Run one real BFTS node with the default fixed 2,048-token profile and a supplied provider credential.
- [x] Confirm the node stops after exactly one executed training node and does not launch stage transitions or multi-seed work.
- [x] Verify the root repository and configured shared cache with pre/post manifests; the external cache remains untouched by design.
- [ ] Run a short multi-node search, then confirm a shortlist sequentially with seeds `42`, `43`, and `44`.
- [ ] Predeclare the statistical acceptance threshold before comparing candidates.
- [x] Export candidate patches and artifacts only; do not apply or commit them.

## Planned follow-on action items

These are real future work items, but remain disabled or prohibited during the current pretraining pilot until separately approved:

- [ ] Implement supervised fine-tuning experiments and repair the known `scripts/sft_train_curriculum.py` checkpoint-save defect.
- [ ] Implement paper, report, PDF, citation, and peer-review generation as a separately approved reporting milestone.
- [ ] Design and implement a human-approved promotion workflow for patch application, checkpoint promotion, commits, and pushes; never enable these actions automatically.
- [ ] Implement and validate long-context curriculum research.

## Temporary compatibility measures requiring eventual repair

- [ ] Replace fixed 2,048-token context with a tested bucketed/discrete context curriculum only after dataloader, compiled shapes, effective batch, scheduler budget, and checkpoint metadata are coordinated.
- [ ] Move from the current same-container controller/worker arrangement to a separate controller and experiment-execution service or privilege domain.
- [ ] Give each node an OS-enforced private workspace and cache; shared writable `experiments/` access is not a sufficient tenant boundary.
- [x] Make the default model configuration explicit at invocation time so a missing model cannot trigger an unexpected paid request.
- [ ] Resolve the Windows `resource` test limitation or document a supported Windows test command; Linux remains authoritative for the sandbox.
- [ ] Decide whether the prepared repository `data/` cache should later be deliberately copied/promoted to the external cache; never promote it automatically.

## Deferred opt-in LLM trace dataset

This is future work, not part of the current pilot:

- [ ] Define a versioned per-run JSONL trace schema.
- [ ] Record provider/model, timestamps, normalized messages, system prompts, tool schemas/calls/results, usage, retries, role, idea/node, stage, and termination reason.
- [ ] Redact API keys, authorization headers, cookies, credentials, and secret-bearing environment values before writing anything.
- [ ] Mark generated code, tool output, and dataset content as untrusted.
- [ ] Define retention, deletion, access control, consent, licensing, and quality-filtering policies.
- [ ] Keep capture disabled by default and local-only; never upload or train automatically.
- [ ] Add deterministic secret-leak tests before enabling collection.

## Explicitly out of scope for the current pilot

- Multi-GPU training and production-scale runs.

## Completion gate

The pretraining integration is ready for promotion only when all P0 items are closed, one live BFTS node passes, the root/cache remain unchanged, a short search and multi-seed confirmation complete, all tests pass, and the final diff is reviewed by a human. A commit or push remains a separate explicit user action.
