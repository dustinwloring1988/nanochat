# Nanochat AI Scientist v2 Integration Postmortem

- **Date:** 2026-09-24
- **Repository:** `F:\UserData\git-repos\nanochat - Copy`
- **Integration status:** Implemented; code hardening and live one-node validation complete for OpenCode and OpenRouter; multi-stage/multi-seed expansion pending
- **Open-work tracker:** `plan.md` remaining-work checklist

## Executive summary

AI Scientist v2 has been integrated at the nanochat repository root and adapted for controlled, experiments-only pretraining research. The integration vendors a pinned upstream revision, introduces one provider/model interface, makes nanochat training and curriculum state reproducible, adds a trusted result contract, isolates generated experiments from the root repository and shared cache, and provides a single-GPU Linux container.

The initial implementation passed the complete Linux-container suite with 71 tests passed and 10 skipped, and the runtime image exposed the RTX 4060 Ti through PyTorch 2.9.1 and CUDA 12.8. A subsequent adversarial review found prototype-level gaps that ordinary unit tests had missed, including missing node configuration, unsafe artifact serialization, incomplete provider-role routing, and insufficient isolation of generated execution.

A second hardening pass now materializes a trusted baseline experiment configuration, enforces one-node BFTS execution, keeps artifacts project-relative, validates fixed-budget/result/guardrail contracts, removes the runtime repository mount, uses an allowlisted child environment, strengthens process-tree cleanup, and begins enforcing shared provider budgets without automatic prompt/response capture. The final regression also verifies controller-side result revalidation, detached controller attestation, mandatory source/cache/runtime provenance, fail-closed worker timeouts, explicit unknown-MFU handling, provider edge cases, curriculum composition metadata, active-versus-wall timing, resume-contract rejection, and pre/post integrity manifests.

The integration is now validated through live one-node BFTS runs with both `opencode/space-bunny-free` and `openrouter/stealth/space-bunny-alpha`. The controller and generated code still share the same container identity; a separate privilege boundary remains future work. The loader now has a versioned exact snapshot/restore path, both training scripts load rank-local state, and an isolated real one-stage GPU resume probe passes; successful multi-stage training remains intentionally blocked by the fail-closed pending-batch guard. Candidate source inheritance passes an offline controller-level stage-1-to-4 regression but not a real multi-stage BFTS training run, and multi-node/multi-seed confirmation remains disabled by the one-node safety profile. Those items remain in `plan.md` rather than being duplicated as a task list here.

## Work completed

### Upstream preservation

- Vendored AI Scientist v2 at commit `96bd51617cfdbb494a9fc283af00fe090edfae48`; the authoritative integration pin is also recorded in `ai_scientist/UPSTREAM_COMMIT`.
- Preserved nanochat's root MIT license and added the AI Scientist license separately.
- Kept the AI Scientist Python 3.11 environment separate from nanochat's primary environment.
- Recorded an independent universal dependency lock for Python 3.11.

### Provider and model routing

- Added one `provider/model` interface for OpenCode Zen, direct OpenAI, and OpenRouter.
- Added Chat Completions and Responses API normalization.
- Added structured tool-call validation, bounded retries, token/call budgets, model listing, and preflight support.
- Defaulted all enabled AI Scientist roles to `opencode/space-bunny-free`.
- Kept provider credentials out of child experiment environments, generated patches, journals, and result artifacts.

### Reproducible pretraining and curriculum

- Added explicit seed propagation to model initialization, CUDA/Python/NumPy state, source sampling, subset sampling, and result metadata.
- Added sampler-state persistence and restoration.
- Made curriculum source and subset weights affect subsequent stages.
- Made curriculum stage transitions rebuild the loader and account for the actual token budget.
- Added scheduler resume state and explicit seed and result-file arguments to pretraining entry points.
- Added canonical, validated `results.json` output and deterministic trusted plots.
- Fixed the initial AI Scientist profile to 2,048-token context while dynamic context support remains disabled.

### BFTS experiment isolation

- Added a fresh source snapshot and working directory for every node.
- Restricted the root repository and shared nanochat cache to read-only mounts.
- Added unique checkpoint, log, result, plot, and temporary paths per node.
- Added trusted metric ingestion; LLM-authored metric parsers are not accepted.
- Added candidate patch, code, result, and plot export for human review.
- Restricted execution to one GPU worker with no silent CPU fallback.
- Removed broad process termination and disabled paper/report generation.
- Sanitized provider secrets before launching generated experiment code.

### Container and developer workflow

- Added an AI Scientist Docker profile using PyTorch 2.9.1, CUDA 12.8, Python 3.11, and a separate AI Scientist virtual environment.
- Runs the service as a non-root user with a read-only root filesystem, bounded temporary storage, one GPU, and explicit memory/process limits.
- Redirects compiler caches to the writable temporary filesystem.
- Installs the C/C++ build toolchain required by `torch.compile` and Triton.
- Added model listing/preflight commands, environment-variable documentation, smoke wrapper, tests, and README instructions.

### Second-pass hardening

- Materializes `config/ai_scientist_experiment.json` into every node before generated code runs.
- Archives both the baseline and resolved experiment configuration with each result.
- Enforces fixed 2,048-token context, seed, evaluation cadence, and total training-token budget while allowing explicitly validated architecture and optimizer changes.
- Validates metric consistency, complete curves, source hashes, model provenance, plot count, runtime, VRAM, and guardrail values in the parent adapter.
- Adds `agent.max_nodes` and a launcher `--max-nodes` control; the default smoke path is now exactly one executed training node.
- Makes node serialization robust for paths outside the current project and places the writable experiments mount beneath the read-only image source.
- Removes the runtime repository bind mount, so an ignored host `.env` is not copied into the service filesystem.
- Replaces child-secret denylisting with an environment allowlist, removes provider credentials from generated processes, and adds Linux process-group termination.
- Requires provider preflight, gives every enabled BFTS role the selected model, propagates configured token limits, removes hard-coded role fallbacks, and makes structured capability failures fail closed.
- Replaces in-memory call limits with a cross-process SQLite budget and stops writing unredacted LLM interactions by default.
- Archives candidate source trees and seeds descendant snapshots from accepted parent nodes, with a targeted inheritance test.

### Low-hanging backlog closed

The first follow-up slice closed these items:

- Controller-owned resource and plot guardrails now come from trusted constants rather than candidate-supplied limits.
- Evaluation-token budgets must divide exactly by the active validation batch, including DDP world size.
- Normal BFTS launches fail closed unless `--allow-provider-calls` is explicitly supplied; preflight remains an explicit diagnostic command.
- Nanochat configuration rejects any node count other than one; unlimited search is disabled until a separate explicit mode exists.
- Child environments now use an explicit allowlist; Hugging Face, W&B, cloud, and provider credential variables are removed.
- Provider budgets include a run identifier, and catalog listing calls use bounded retries.
- The Docker experiment path is consistently `/workspace/project/experiments`, with the tracked host placeholder and writable nested mount.
- Provider/model tracing remains disabled by default; future trace capture still requires the redaction work in `plan.md`.

The focused AI/loader/checkpoint/curriculum suite passes 67 tests in the Linux container, and the rebuilt Linux container passes 120 tests with 10 skipped. An isolated two-step one-stage float32 GPU probe additionally passed reference/resume equivalence with model delta `0`, optimizer delta `7.28e-12`, exact loader state, BPB `2.2683011088`, and matching curve steps/losses; the two-stage negative control failed closed at the pending-batch guard.

### Final hardening slice

- Results now use schema version 2 with signed CORE, explicit required diagnostics, an immutable source identity, cache/tokenizer/dataset hashes, model identity, and runtime/device provenance.
- The parent controller independently validates the archived source/configuration/result and writes a detached HMAC controller attestation; a bare candidate-authored `results.json` is rejected, and tampering invalidates the attestation.
- Timed-out worker futures now cancel, terminate, and reap pool workers; registered interpreter process trees are cleaned before GPU release.
- Unknown GPU peak FLOPS is reported as `mfu_percent: null` rather than a fabricated zero.
- An offline one-node BFTS integration completed a real depth-6 training node with a local query stub and produced a controller-accepted result; no provider network calls were made.
- A fresh standalone fixed-context GPU run completed 200 iterations with schema 2, BPB `1.491322`, CORE `-0.0200`, peak VRAM `8,392,344,064` bytes, active training time `158.37s`, wall-clock training time `310.24s`, observed ClimbMix composition counters, and four plots including `source_mixture.png`.
- Provider tests now cover malformed/schema-invalid output, JSON fallback, retry counts, budget reservation, token exhaustion, and structured preflight capability reporting.
- Checkpoints now carry a versioned resume contract covering model shape, seed, rank/world size, horizon, batch/context, curriculum schedule, dataloader settings, dataset manifest, and tokenizer hash; incompatible or legacy checkpoints fail closed.
- Live OpenCode preflight passed with tool calls and structured output; the live one-node BFTS run completed with a controller-accepted result.
- Live OpenRouter preflight passed using JSON mode after three attempts; the live one-node BFTS run completed with `AI_SCIENTIST_ALLOW_MISSING_USAGE=1` and a controller-accepted result.
- OpenRouter’s free endpoint omits usage and price metadata, so token limits are unavailable for that explicitly opted-in mode; API-call limits remain enforced.
- Provider failures now cross process-pool boundaries as sanitized worker errors instead of terminating the pool.
- Added an offline AgentManager regression that materializes accepted-parent source snapshots across main stages 1–4, verifies inherited node identity and marker propagation, and proves no provider call or second executed node occurs; real descendant training remains a future gate.
- The resume audit confirmed that the previous loader state was approximate: source cursors were row-group coarse, document buffers and prefetched packed batches were not serialized, only rank 0 wrote common metadata, and stage transitions reset loader state. A versioned CPU-testable loader snapshot, rank-local checkpoint files, both training scripts, an offline checkpoint-boundary round trip, a versioned stage-transition contract, a fail-closed transition guard, and resume-step side-effect suppression now cover those pieces; an isolated one-stage GPU resume probe also passed, while successful multi-stage training remains future work.

## Verification evidence

The following table records the verified baseline plus the completed regression and live one-node runs. The unresolved items are security/architecture and multi-stage expansion, not known test regressions:

| Check | Result |
| --- | --- |
| AI Scientist provider, contract, workspace, and curriculum tests | 57 passed |
| Host nanochat suite excluding Unix-only execution sandbox | 43 passed, 12 skipped |
| Complete Linux-container suite | 96 passed, 10 skipped |
| Optimizer tests in Linux container | 4 passed |
| Python compilation checks | Passed after final edits |
| Black checks for integration files | Passed after final edits |
| `docker compose config` | Passed after image-baked source change |
| AI Scientist image build | Passed after hardening changes |
| Container Python/PyTorch/CUDA versions | Python 3.11.14, PyTorch 2.9.1+cu128, CUDA 12.8 |
| Container GPU visibility | RTX 4060 Ti detected |
| Minimum data preparation | ClimbMix layouts and evaluation bundle prepared under ignored `data/` |
| Standalone d6 GPU training smoke | Passed; final validation BPB 1.491322, CORE -0.0200, peak VRAM 8.39 GB, active 158.37s, wall 310.24s, four plots |
| Same-seed repeat | Passed within observed tolerance on the earlier schema; final BPB delta 0.000416, maximum curve delta 0.000443 |
| Offline one-node BFTS integration | Passed with local query stub; controller-accepted result and attestation produced; no network provider calls |
| Offline AgentManager stage inheritance | Passed across main stages 1–4; accepted source marker propagated, one result archived, no provider calls or extra executed nodes |
| Versioned loader snapshot tests | Passed; first-row-group, cursor/buffer, pending-batch, source-order, epoch, and fail-closed state checks |
| Rank-local checkpoint tests | Passed; rank 0/1 state separation, metadata derivation, missing-file failure, and legacy save/load compatibility |
| Training-script rank-local wiring | Compile/full-suite verified; both standard and curriculum trainers load rank-local state and save it explicitly; no real checkpoint/resume equivalence run yet |
| Offline checkpoint-boundary round trip | Passed; rank-local serialization followed by loader restore reproduced the uninterrupted batch stream |
| Stage-transition contract tests | Passed; JSON round-trip, malformed/mismatched state rejection, boundary validation, and composition-counter preservation |
| Fail-closed stage-transition guard | Passed; transitions with an explicit pending batch or missing pending state are rejected |
| Real one-stage GPU resume probe | Passed; isolated float32 reference/resume, model delta 0, optimizer delta 7.28e-12, exact loader state, BPB 2.2683011088, matching curve steps/losses |
| Real two-stage negative control | Expected fail-closed result; transition stopped with pending-batch guard before loader reset |
| Windows host test command | Documented `python -m pytest -q --ignore=tests/test_execution.py`; Linux container remains authoritative for the full suite |
| OpenCode live preflight | Passed; catalog-listed, tool calls supported, structured output passed |
| OpenCode live one-node BFTS | Passed; BPB 1.491380, controller-accepted result and attestation |
| OpenRouter live preflight | Passed; JSON structured mode, three preflight attempts, no tool-call support reported |
| OpenRouter live one-node BFTS | Passed with `AI_SCIENTIST_ALLOW_MISSING_USAGE=1`; BPB 1.491395, controller-accepted result and attestation |

The full native Windows suite also has eight execution-sandbox failures because the Unix-only `resource` module is unavailable. This is a host-platform limitation, not a failure of the Linux GPU path; the full Docker suite is the authoritative verification path for those tests.

## Incidents, discoveries, and repairs

### Environment split was required

AI Scientist requires Python 3.11 while the original nanochat environment targeted Python 3.10. The original base container also used PyTorch 2.5.1 while the project declared 2.9.1. Silently merging environments would have made results irreproducible. The repair was a separate AI Scientist virtual environment and a dedicated PyTorch 2.9.1 container.

### Upstream tree search did not match nanochat's safety model

The upstream BFTS flow could run arbitrary generated code, infer metrics through model-authored logic, and clean up processes too broadly. The local adapter now uses fresh source trees, canonical trusted metrics, owned-process cleanup, one worker, no CPU fallback, and patch-only export.

### Curriculum changes were not trustworthy as originally written

The original curriculum implementation logged source-weight changes without applying them, skipped validation, and lacked enough state to resume accurately. Those issues were repaired for fixed-context runs. Dynamic context changes remain intentionally disabled because the model, dataloader, effective token batch, compiled shapes, and scheduler budget must change coherently.

### Read-only containers exposed missing compiler runtime support

The first full optimizer run failed because Triton attempted to write compiler caches into the non-root home directory. After redirecting caches to `/tmp`, Triton reported that no C compiler was installed. The final image redirects `XDG_CACHE_HOME`, `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR` to tmpfs and installs `build-essential`. This is a permanent container requirement for `torch.compile`, not a model workaround.

### The shared cache was initially not training-ready

The original external cache contained a tokenizer and old pretrained directories but no required ClimbMix train/validation shards or evaluation bundle. The minimum assets are now prepared under the ignored repository `data/` cache for the standalone smoke; the external cache remains unchanged. The minimum preparation path and estimated download size are documented in `plan.md`.

### Generated configuration ownership was initially missing

The first stable runfile expected `working/experiment.json`, but nothing created it, so the exact supplied baseline failed before training. The adapter now materializes and validates a controller-owned baseline configuration, archives it, and enforces fixed-budget fields before execution. The parent now independently validates the archived result and writes a detached controller attestation, so a bare generated result file is not accepted. Remaining risk is that generated code shares the service account and can still attack the controller boundary; a separate privilege domain remains required.

### The first smoke wrapper did not guarantee one node

The original `runs/ai_scientist_smoke.sh` invoked a profile capable of running multiple training nodes. The integration now has a global `agent.max_nodes` limit, a `--max-nodes` launcher option, and a one-node default. The offline one-node integration and full Linux-container regression now verify that revised behavior.

## Temporary compatibility measures requiring repair

| Measure | Why it exists | Required follow-up |
| --- | --- | --- |
| Fixed 2,048-token context | Avoids unsafe dynamic dataloader and compiled-shape changes | Implement and test bucketed context transitions before enabling curriculum context research |
| Provisional 200-iteration d6 profile | Keeps the first research budget deterministic but has not been benchmarked on this GPU | Measure runtime, VRAM, throughput, and metric variance; record a stable budget |
| Candidate-authored result file | Generated code can write a plausible `results.json` in the shared account | Parent revalidation and detached HMAC attestation are implemented; move to a separate privilege domain for stronger isolation |
| Candidate source inheritance | Archive/reuse is implemented and unit-tested after the review found silent reversion to root | Run a full multi-stage BFTS descendant test before treating stage results as cumulative |
| Same-container execution boundary | Removing the repository bind and allowlisting the environment blocks accidental access but not same-UID inspection or sibling writes | Separate controller and generated execution into distinct services or privilege domains |
| Improved but non-exact sampler resume | Versioned contracts reject incompatible model/data/curriculum settings; loader snapshots, rank-local files, both training scripts, an offline checkpoint round trip, a stage-transition contract, a fail-closed guard, and an isolated one-stage GPU resume probe preserve/load state, but successful multi-stage training is not enabled | Add successful multi-stage resume/composition tests before enabling research comparisons |
| Linux-container test authority | Native Windows lacks Unix sandbox modules | Use `python -m pytest -q --ignore=tests/test_execution.py` on Windows; keep the Linux container authoritative for the full suite |
| Live API path | OpenCode and OpenRouter one-node live paths pass; OpenRouter requires explicit missing-usage opt-in because its free endpoint omits token metadata | Keep credentials local, rerun preflight before live work, and do not weaken default fail-closed budgets |
| Provider hardening verification | Offline retry, budget, multimodal, role-routing, trace-privacy, and live OpenCode/OpenRouter one-node checks pass | Keep live preflight and bounded one-node runs as the acceptance gate |
| External cache separation | The pilot data was placed in ignored repository `data/` rather than the user's external cache | Decide whether to promote the prepared data deliberately; never copy it into the trusted cache automatically |

The container compiler cache paths and build toolchain are not temporary model changes. They are required runtime support for trusted `torch.compile` execution.

## Hard safety boundaries

The current and future pretraining pilot must not:

- generate or publish a paper, report PDF, citations, or peer-review document;
- run supervised fine-tuning experiments;
- merge an AI-generated patch into the root repository;
- commit or push changes automatically;
- promote generated checkpoints into the trusted nanochat cache;
- use LLM-generated metric parsing code;
- upload experiment artifacts or traces automatically;
- reuse desktop or CLI authentication as a provider API credential.

Generated patches and checkpoints are untrusted. Promotion requires human review, a separate controlled change, clean-environment reproduction, and explicit user action.

## Deferred LLM trace dataset

Saving AI Scientist LLM traces for later model training is a deferred, opt-in task. The current implementation does not persist a complete provider trace, and the postmortem does not claim that such a dataset exists.

A future trace format should be versioned JSONL and should record normalized messages, system prompts, provider/model, timestamps, tool schemas, tool calls and results, token usage, retries, role, idea, node, stage, and termination reason. Before collection is enabled, the project must define redaction, retention, deletion, access control, consent, dataset licensing, untrusted-content marking, and deterministic secret-leak tests. API keys, authorization headers, cookies, and secret-bearing environment values must never be persisted. Any conversion to training data must be a separate, reviewable offline step rather than automatic upload or training.

## Retrospective

### What went well

- Pinning the upstream revision and preserving both licenses made the integration auditable.
- Treating provider routing, trusted metrics, workspace isolation, and Docker runtime as separate concerns reduced coupling.
- Running the full Linux suite exposed real container issues that mocked host tests could not.
- Keeping the root and shared cache read-only converted promotion from a convention into a container boundary.
- Requiring canonical metrics and provenance made candidate comparison reproducible rather than conversational.

### What caused rework

- The initial container was based on an older PyTorch/Python environment and lacked Triton compiler prerequisites.
- A Windows-only test command could not represent the intended Linux execution sandbox.
- The local cache appeared populated but lacked the data layout required by curriculum training.
- The BFTS “smoke” path exposed a trusted/generated configuration ownership gap that unit tests did not model.

### Important lessons

- Environment parity must be verified inside the actual training container, not inferred from lock files.
- Read-only execution requires explicit writable homes, caches, temporary files, and compiler paths.
- A smoke test must prove its cardinality and stopping condition; its name is not evidence.
- Trusted adapters must own validation, configuration defaults, metric parsing, and artifact promotion boundaries rather than delegating them to generated code.
- Data manifests and provider preflight are acceptance prerequisites, not optional setup details.

## Handoff

The canonical implementation specification and all unfinished work remain in `plan.md`. The most relevant implementation entry points are:

- `ai_scientist/providers.py`
- `ai_scientist/treesearch/nanochat_adapter.py`
- `ai_scientist/treesearch/parallel_agent.py`
- `ai_scientist/treesearch/interpreter.py`
- `launch_scientist_bfts.py`
- `nanochat/ai_scientist_experiment.py`
- `nanochat/multi_source_dataloader.py`
- `nanochat/checkpoint_manager.py`
- `nanochat/research_results.py`
- `scripts/base_train.py`
- `scripts/base_train_curriculum.py`
- `bfts_config.yaml`
- `docker/Dockerfile.ai-scientist`
- `docker-compose.yml`
- `tests/test_ai_scientist_provider.py`
- `tests/test_ai_scientist_contract.py`
- `tests/test_ai_scientist_workspace.py`
- `tests/test_curriculum_training.py`
- `tests/test_multi_source_dataloader.py`
- `tests/test_checkpoint_manager.py`
- `nanochat/curriculum_state.py`
- `tests/test_curriculum_transition_state.py`
- `C:\Users\dusti\AppData\Local\Temp\opencode\nanochat-resume-docker-97408ef017bb442eba8907cca4fbcfc9` (isolated GPU probe artifacts)
