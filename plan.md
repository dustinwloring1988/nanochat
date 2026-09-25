# Nanochat AI Scientist v2 — Work Checklist

- **Status:** Implemented work is verified; remaining items are explicitly blocked.
- **Current hardware:** RTX 4060 Ti, 16 GB VRAM.
- **Upstream pin:** `96bd51617cfdbb494a9fc283af00fe090edfae48`
- **Latest verification:** Linux `165 passed, 11 skipped`; focused suite `66 passed, 1 skipped`; host subset `108 passed, 18 skipped`.
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
| Legacy SFT entry point | PASS | `chat_sft.py` fails closed and directs users to the approved run-local entry point. | The legacy script can write to `chatsft_checkpoints` or another trusted namespace. |
| Redacted trace writer | PASS | `trace_writer.py` provides versioned JSONL, deterministic redaction, trust markers, retention/deletion, path checks, and fail-closed behavior; capture is disabled by default. | Secrets, raw provider traffic, image bytes, uploads, replay, or automatic training appear. |
| Provider trace integration | PASS | Central provider calls and launcher preflight use only an explicit opt-in writer; direct provider backends fail closed while tracing is enabled. | A direct provider path silently bypasses redaction or capture occurs by default. |
| Dynamic-context gate | PASS | `dynamic_context.py` validates discrete buckets, budget arithmetic, resume metadata, fixed fallback, and resource-evidence shape while keeping activation disabled. | Dynamic or interpolated context is activated without runtime/resource evidence and approval. |
| Test and static checks | PASS | Linux, host, focused SFT/trace/dynamic tests, targeted Black/Ruff, and AST parsing pass; skipped checks are labeled. | A skipped, mocked, unavailable, or failed check is reported as passing evidence. |
| Documentation consolidation | PASS | This checklist and `postmordum.md` are the active planning records; obsolete design documents are removed. | Deleted planning documents are still presented as active gates. |

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
- **Git changes:** PASS — the user explicitly authorized committing and pushing the reviewed changes in this session.
