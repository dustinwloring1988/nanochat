# Nanochat AI Scientist v2 — Work Checklist

- **Status:** Implemented work is verified; remaining gated items are explicitly blocked.
- **Current hardware:** RTX 4060 Ti, 16 GB VRAM.
- **Upstream pin:** `96bd51617cfdbb494a9fc283af00fe090edfae48`
- **Latest verification:** Linux `178 passed, 1 skipped`; system host subset `111 passed, 18 skipped`; synchronized CPU dev host `149 passed, 16 skipped`; focused AI Scientist host tests `53 passed, 2 skipped`; Docker FA3 attention tests `16 passed`.
- **Data policy:** Use ignored repository `data/` for pilot assets; keep `C:\Users\dusti\.cache\nanochat` separate and unchanged.

This is a short operational checklist. Detailed implementation history and evidence remain in `postmordum.md`.

## Safety rules

| Check | Status | Pass case | Fail/blocked case |
| --- | --- | --- | --- |
| No papers, reports, PDFs, citations, or peer-review documents | PASS | None are generated, uploaded, or published. | Any such document is produced or published. |
| Credentials and `.env` values | PASS | Credentials stay local and are never printed, persisted, or reused as provider credentials. | A credential appears in code, logs, traces, artifacts, or child environments. |
| One-node boundary | PASS | `agent.max_nodes=1` and no automatic node expansion. | The boundary is weakened without a new safety review. |
| Trusted cache and promotion | PASS | Candidate artifacts stay run-local and untrusted; no automatic cache writes or promotion. | A generated patch/checkpoint is applied, promoted, or written to the trusted cache. |
| Provider calls | PASS | Live calls require explicit opt-in and preflight. | Provider calls occur by default or use unapproved credentials. |
| Generated content trust | PASS | Generated code, patches, checkpoints, tool output, and dataset text remain untrusted. | Generated content is executed, trusted, or promoted without human review. |

## Implemented and verified

| Check | Status | Pass case | Fail/blocked case |
| --- | --- | --- | --- |
| Fixed-context SFT infrastructure | PASS | Manifest-only JSONL data, fixed 2,048 context, run-local paths, versioned checkpoints, exact loader/RNG resume, and rollback are implemented in `sft_manifest.py`, `sft_runtime.py`, and `sft_train_curriculum.py`. | A legacy global-cache path, unmanifested data, or non-2,048 context is used. |
| SFT runtime probe | PASS | CPU and RTX 4060 Ti probes resume with model delta `0.0` and identical next batches; the GPU probe peaked at `1,235,122,176` bytes. | A probe fails, loses state, writes outside its run directory, or is presented as quality evidence. |
| FA3 kernel discovery | PASS | Docker resolves `kernels-community/flash-attn3` API version `1` on RTX 4060 Ti sm89; the tiny bfloat16 causal probe is finite and the full attention suite passes `16/16`. | Kernel discovery omits the API version, silently falls back, or passes only on SDPA. |
| Host AI Scientist test dependencies | PASS | The dev group and lockfile now provide provider, schema, config, workspace, trace, and formatting dependencies; the focused host set passes `53` tests with only two intentional POSIX skips. | Tests are silently skipped because required host test dependencies are absent or the lock is stale. |
| Docker build-context hygiene | PASS | `.dockerignore` excludes `.venv-*/`; the final Docker build context was `30.72 kB`. | A local virtual environment is copied into the image or trusted cache. |
| Legacy SFT entry point | PASS | `chat_sft.py` fails closed and directs users to the separately approved fixed-context runtime probe; production SFT remains gated. | The legacy script can write to `chatsft_checkpoints` or another trusted namespace. |
| Redacted trace writer | PASS | `trace_writer.py` provides versioned JSONL, deterministic redaction, trust markers, retention/deletion, path checks, and fail-closed behavior; capture is disabled by default. | Secrets, raw provider traffic, image bytes, uploads, replay, or automatic training appear. |
| Provider trace integration | PASS | Central provider calls and launcher preflight use only an explicit opt-in writer; direct provider backends fail closed while tracing is enabled. | A direct provider path silently bypasses redaction or capture occurs by default. |
| Dynamic-context gate | PASS | `dynamic_context.py` validates discrete buckets, budget arithmetic, resume metadata, fixed fallback, and resource-evidence shape while keeping activation disabled. | Dynamic or interpolated context is activated without runtime/resource evidence and approval. |
| Test and static checks | PASS | Linux, host, focused SFT/trace/dynamic/FA3 tests, Ruff, Black workflow checks, and AST parsing pass; skipped and formatting-limited checks are labeled. | A skipped, mocked, unavailable, or failed check is reported as passing evidence. |
| Documentation consolidation | PASS | This checklist and `postmordum.md` are the active planning records; obsolete design documents are removed. | Deleted planning documents are still presented as active gates. |
| Default workflow usability | PASS | Docker and reference shell workflows are pretraining-only, omit disabled SFT commands, and point to `python -m scripts.sft_smoke --help`; the legacy entry point gives the same fail-closed handoff. | A default workflow invokes `chat_sft`, the unapproved long-context SFT config, or a trusted SFT namespace. |
| Workflow contract regression | PASS | `tests/test_sft_entrypoint.py` verifies the handoff and rejects stale executable SFT commands; focused run passed `18` tests. | Static workflow contracts fail or the disabled entry point stops directing users to the approved path. |
| Host shell syntax check | BLOCKED | Run `bash -n` in a working POSIX shell for all four changed workflow scripts. | The Windows `bash.exe` delegates to an unavailable WSL virtual disk; no syntax-pass claim is made. |
| Native Windows GPU optimizer environment | BLOCKED | Use the Linux container for CUDA/Triton optimizer coverage. | The separate Windows GPU venv has CUDA but no working Triton installation; its four optimizer tests fail before assertions. |

## Remaining work

| Check | Status | Pass case | Fail/blocked case |
| --- | --- | --- | --- |
| Production SFT quality | BLOCKED | Approved dataset licenses/provenance, held-out evaluation, predeclared thresholds, and clean reproduction are recorded before any production run. | A fabricated probe or smoke result is treated as SFT quality or promotion evidence. |
| Long-context curriculum | BLOCKED | Obtain dynamic-bucket runtime/resource evidence, exact-resume proof, and explicit approval before implementation or activation. | Any 8K–32K curriculum or dynamic context is enabled now. |
| Real multi-stage BFTS descendant proof | BLOCKED | Run a predeclared one-node experiment with at least two executed descendant nodes and record trusted metrics. | Candidate source inheritance is treated as a real descendant training result. |
| Search expansion and multi-seed confirmation | BLOCKED | Predeclare the statistical threshold and comparison protocol, then obtain a new safety review before changing the one-node boundary. | `agent.max_nodes` or seed scope expands without review. |
| Security and promotion boundary | BLOCKED | Separate controller/generated execution, add OS-enforced private workspaces, and implement human-only signed promotion with clean reproduction. | Same-account execution is treated as a sufficient security boundary. |
| Live provider expansion | BLOCKED | Run explicit preflight immediately before each approved live run and keep missing-usage mode opt-in. | Credentials are reused, printed, or a live run starts without opt-in. |

## Approval and change control

- **AG-1 SFT:** PASS — approved on 2026-09-25 for fixed-context, run-local implementation and bounded probes only.
- **AG-2 traces:** PASS — approved on 2026-09-25 for opt-in local redaction and tests; default capture remains disabled.
- **AG-3 long context:** BLOCKED — no dynamic-bucket runtime/resource evidence or explicit activation approval.
- **Promotion/cache changes:** BLOCKED — require a separate human-approved manifest, reproduction, and cache decision.
- **Git changes:** PASS — the user explicitly requested committing and pushing this reviewed change set; no safety-gated action is included.
