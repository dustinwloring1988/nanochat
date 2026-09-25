# Nanochat AI Scientist v2 — Remaining Work Plan

- **Status:** Final handoff: one-node pilot and fixed-context multi-stage proof complete; dynamic expansion remains gated
- **Date:** 2026-09-24
- **Last verified commit:** `50969cf` (`record cache separation decision`)
- **Target:** RTX 4060 Ti, 16 GB VRAM
- **Upstream pin:** `96bd51617cfdbb494a9fc283af00fe090edfae48`
- **Pilot cache:** ignored repository `data/`
- **External cache:** `C:\Users\dusti\.cache\nanochat` remains unchanged

This file is the final handoff backlog. Checked items are complete within the approved one-node scope; unchecked items are not silently deferred and require a new approval or milestone. Completed implementation details belong in `postmordum.md`.

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
- A same-seed repeat ended at BPB `1.491474`; absolute delta is `0.000416`. The current numerical reproducibility tolerance is `0.001`, not bitwise equality; it is not a statistical candidate-acceptance threshold.
- Live one-node BFTS passed with OpenCode `opencode/space-bunny-free` at BPB `1.491380` and OpenRouter `openrouter/stealth/space-bunny-alpha` at BPB `1.491395`.
- OpenRouter's free endpoint omits usage metadata; its live run requires explicit `AI_SCIENTIST_ALLOW_MISSING_USAGE=1`, while API-call limits remain enforced.
- Isolated GPU resume probes passed for two-step one-stage and two-stage float32 curriculum runs: model/optimizer deltas were `0`, loader state and resume contracts were exact, and the two-stage run restored curriculum composition counters exactly. The two-stage reference/resume BPB was `2.268270`; dynamic context remains disabled.
- Current verification: Linux container `125 passed, 10 skipped`; the 10 skips are FA3 capability-gated attention tests, while SDPA coverage passes. The focused AI/loader/checkpoint/curriculum suite passes `72 passed`; native Windows execution-sandbox tests remain non-authoritative because `resource` is unavailable. Black checks for integration files and compilation pass.

## Pilot decision

- The approved current scope is complete: both live providers passed exactly one controller-attested BFTS node, and the root repository and external cache remained unchanged.
- This commit is a research-integration checkpoint, not checkpoint promotion, patch application, paper generation, SFT approval, or permission to run a larger search.
- Multi-node/multi-seed confirmation and separate privilege-domain isolation remain future milestones; enable them only through an explicit follow-up approval. The fixed-context two-stage path is now validated, while dynamic context/long-context curriculum remains disabled.

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

- [x] Add an offline controller-level regression proving accepted parent `candidate_source` crosses main stages 1–4 without provider calls, extra GPU nodes, or production guard changes.
- [ ] **Future gate:** Run a real multi-stage BFTS training test with at least two executed nodes; the offline regression does not execute descendant training.
- [x] Implement a versioned CPU-testable loader snapshot that preserves exact source order, per-iterator cursors, document buffers, sampler state, and pending packed batches.
- [x] Add rank-local dataloader checkpoint files and a fail-closed loader helper while preserving existing checkpoint callers.
- [x] Integrate rank-local state loading and explicit state persistence into both training scripts.
- [x] Add an offline checkpoint-boundary round trip proving resumed loader batches match the uninterrupted stream.
- [x] Define and test a versioned stage-transition contract with cumulative composition counters.
- [x] Implement consume-before-transition semantics; unexpected pending batches still fail closed.
- [x] Validate real one-stage GPU checkpoint/resume equivalence in an isolated temporary root.
- [x] Prevent duplicate evaluation/metric side effects at the resume step and verify curve restoration.
- [x] Validate real fixed-context two-stage GPU reference/resume equivalence and composition restoration.
- [ ] **Future gate:** Enable dynamic context or long-context curriculum only after its own loader/shape/budget proof.
- [x] Reject resume when model shape, curriculum schedule, seed, world size, token horizon, or dataloader configuration differs.
- [x] Add stage/source composition metadata and a source-mixture plot to canonical results.
- [x] Decide whether MFU is `null`/unknown or add a verified RTX 4060 Ti peak-FLOPS value; do not report unknown MFU as a measured zero.
- [x] Decide whether training-time throughput should include compilation/evaluation time or report separate active-training and wall-clock metrics.

## P1 — live pilot

- [x] Run one real BFTS node with the default fixed 2,048-token profile and a supplied provider credential.
- [x] Confirm the node stops after exactly one executed training node and does not launch stage transitions or multi-seed work.
- [x] Verify the root repository and configured shared cache with pre/post manifests; the external cache remains untouched by design.
- [ ] **Future gate:** Run a short multi-node search, then confirm a shortlist sequentially with seeds `42`, `43`, and `44`.
- [ ] **Future gate:** Predeclare the statistical acceptance threshold before comparing candidates.
- [x] Export candidate patches and artifacts only; do not apply or commit them.

## Planned follow-on action items

These are real future work items, but remain disabled or prohibited during the current pretraining pilot until separately approved:

- [ ] Implement supervised fine-tuning experiments and repair the known `scripts/sft_train_curriculum.py` checkpoint-save defect.
- [ ] Implement paper, report, PDF, citation, and peer-review generation as a separately approved reporting milestone.
- [ ] Design and implement a human-approved promotion workflow for patch application, checkpoint promotion, commits, and pushes; never enable these actions automatically.
- [ ] Implement and validate long-context curriculum research.

## Approved design-only follow-ons

These designs are approved for documentation and review only; they do not authorize execution, promotion, reporting, or SFT.

- [x] Define the SFT gate: separate data/manifest, checkpoint schema, evaluation contract, resource budget, and explicit human approval before any SFT run.
- [x] Define the promotion gate: signed candidate manifest, clean-environment reproduction, human approval, and no automatic patch/cache/git writes.
- [x] Define the long-context gate: bucketed shapes, effective batch and scheduler alignment, checkpoint migration, VRAM proof, and fixed-context fallback.
- [x] Define the trace gate: versioned local JSONL, deterministic redaction, untrusted-content marking, retention/access policy, and secret-leak tests.
- [ ] Implement any follow-on only after its separate design is reviewed and explicitly approved.


- [ ] Replace fixed 2,048-token context with a tested bucketed/discrete context curriculum only after dataloader, compiled shapes, effective batch, scheduler budget, and checkpoint metadata are coordinated.
- [ ] Move from the current same-container controller/worker arrangement to a separate controller and experiment-execution service or privilege domain.
- [ ] Give each node an OS-enforced private workspace and cache; shared writable `experiments/` access is not a sufficient tenant boundary.
- [x] Make the default model configuration explicit at invocation time so a missing model cannot trigger an unexpected paid request.
- [x] Document a supported Windows test command; Linux remains authoritative for the execution sandbox.
- [x] Decide to keep the prepared repository `data/` cache separate from the external cache; no automatic promotion.

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

The current one-node integration is complete and verified. It is **not promotion-ready**: multi-node/multi-seed confirmation, a predeclared statistical threshold, exact buffered resume, and a separate privilege domain remain required before expanding the research boundary. The final diff was committed as `50969cf`; any later promotion, patch application, checkpoint promotion, or push remains a separate explicit human-approved action.
