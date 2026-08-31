# Branch catalogue for the deployment team

**Maintained by:** the vLLM-code / fork team, in `mardelden/vllm-custom`.
**Last verified:** 2026-08-31 against `upstream/main@5707355209`.

This is the answer to "which of your branches do we deploy, and where?"

**All documentation and every overlay artifact lives on `main`** — this
catalogue, `plans/`, `plans/decisions/`, `overlays/**` (patches, manifests,
configs, installers, tests) and `checkpoints/`. `main` is the only branch that
is never rebased or deleted, so it is the one stable place to point anyone at.

**Topic branches carry code only.** That keeps them minimal, keeps their file
sets disjoint so any subset composes, and means you never have to know which
branch a document is on.

Consequence worth knowing: an overlay `.patch` on `main` is a *snapshot* of its
branch. If the branch code changes, regenerate the artifact. When they disagree,
the branch is authoritative for code and the manifest is authoritative for the
release contract.

If anything here disagrees with a manifest under `overlays/`, **the manifest
wins** — it is the immutable release contract. This file is the index.

---

## 1. What goes everywhere vs. what is model-specific

| Branch | Applies to | Why |
| --- | --- | --- |
| `codex/dsv4-log-outputs-reasoning` | **every vLLM deployment** | Fixes an upstream bug that silently drops model output for any reasoning model. Not tied to a model or GPU. |
| `codex/fastsafetensors-parallel-mtp-share` | **opt-in, any model using `--load-format fastsafetensors`** | Two parts are generic loader fixes; one part is Qwen-specific and stays gated. |
| `codex/glm53-sm120-nope-sparse-mla` | **GLM-5.3-Flash on sm120 only** | Enables a model that otherwise cannot start. Inert for every other model. |

Everything defaults to **off**. A deployment that sets no environment variables
behaves exactly as it does today.

---

## 2. Branch detail

### `codex/dsv4-log-outputs-reasoning` — deploy fleet-wide

- **Tip:** `94ada68216` · **base:** `fb68025138` · **2 files, +26/−4**
- **Files:** `vllm/entrypoints/openai/chat_completion/serving.py`,
  `vllm/entrypoints/serve/middleware/log_response.py`
- **What it fixes:** both request-logging paths drop output for reasoning
  models, in mirrored ways. Non-streaming `log_outputs` builds its text from
  `content` and `tool_calls` only, so a reasoning-only or truncated response
  logs **nothing**, and thinking is never captured even when content exists.
  The response-logging middleware behind `VLLM_DEBUG_LOG_API_SERVER_RESPONSE`
  has the same defect in the opposite mode (streaming).
- **Activation:** none. It corrects behaviour the existing
  `--enable-log-requests --enable-log-outputs` flags already request, which the
  `vllm` role already sets fleet-wide (`vllm_log_requests: true`).
- **Ships with a file sink.** `overlays/request-logging/logging-config.json` is
  a drop-in `VLLM_LOGGING_CONFIG_PATH` dictConfig that routes prompts,
  completions and thinking to a rotating file under `/var/log/vllm/requests/`
  (hourly, 168 kept) for bind-mounting. Config only, no extra code. It needs
  this branch's code fix to be useful, since without it a reasoning model logs
  no completion at all.
- **Which hosts are affected today:** every vLLM container is `patch_set: stock`
  and therefore carries the defect. `vllm-chat` (Qwen3.6) and `vllm-code`
  (Qwen3.8) run reasoning-capable models and are actively dropping thinking.
- **Evidence:** 12 GPU-free unit cases, plus a live A/B on `vllm-build` with
  Qwen3-0.6B — `Generated response` 1 → 2, reasoning lines 0 → 2, same server,
  same probes, only the patch changed.
- **Caveat:** the middleware lives at a different path per tree —
  `serve/middleware/log_response.py` upstream,
  `serve/utils/server_utils.py` in the jasl DSv4 fork. `overlays/dsv4-logging/`
  carries the fork-shaped patch; the branch carries the upstream-shaped one.
- **Removal condition:** drop once fixed upstream. Previously reported without
  diagnosis as vllm-project/vllm#24578, #25918, #19462.

### `codex/fastsafetensors-parallel-mtp-share` — opt-in, mixed scope

- **Tip:** `98c993ee8e` · **base:** `ba07e4a48f` · **21 files, +702/−146**
- **Six stacked commits.** The branch is the deployable unit; individual
  commits are **not** independently applicable (three of them extend the same
  `fastsafetensors_weights_iterator` signature).
- **Generic — safe fleet-wide for any fastsafetensors model:**
  - **Request geometry** (`VLLM_FSST_COPY_BLOCK_MB=16`). FastSafetensors
    defaults `max_copy_block_size` to 16 GiB, larger than any real shard, so it
    emits one read request per shard served by **one thread** while the other
    15 in its pool idle. Affects any model with shards under 16 GiB, i.e. all
    of them. Measured −14.2 s on DSv4.
  - **Adaptive O_DIRECT** (`VLLM_FSST_O_DIRECT=auto`). Stock reads
    `O_RDONLY`; on a checkpoint larger than host RAM the page cache thrashes.
    Measured −6.7 s. **Requires the pinned `mardelden/fastsafetensors@b43888d`
    dependency.** `auto` degrades to buffered reads on a warm cache, so it is
    safe for models that fit in RAM.
- **Model-specific — keep gated:**
  - **DSpark/Qwen draft byte-range filter**
    (`VLLM_DSPARK_DRAFT_RANGE_FILTER=1`). The draft re-read the whole
    checkpoint to extract 99 params. Measured −44.6 s on DSv4, verified by
    SHA-256 equality of all 99 destination parameters on both TP ranks.
    Gated on the model class; inert elsewhere.
- **Contracts:** `overlays/generic-startup/manifest-r{1,2,3}.json` — r2 is the
  healthy rollback, r3 is failed/quarantined evidence. Do not retag either.
  `overlays/dsv4-startup/` and `plans/004-...` cover the DSv4 line.
- **`codex/deepseek-v4-startup-handover` is now redundant.** Its only
  difference was a documentation commit, which moved to `main`, so the two
  branches are byte-identical. Use `codex/fastsafetensors-parallel-mtp-share`;
  the other is kept only until nothing references it.
- **Its `tests/` changes no longer apply to current upstream.** Upstream edited
  `tests/v1/worker/test_gpu_worker.py` in #53591 after this branch's base. The
  `vllm/` code applies clean — extract with `-- vllm/ csrc/` and port the test
  changes by hand if you need them. The two smaller overlays are unaffected.

### `codex/glm53-sm120-nope-sparse-mla` — GLM-5.3-Flash on sm120 only

- **Tip:** `25e93a5988` · **base:** `fb68025138` · **2 files, +34/−2**
- **Files:** `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`,
  `.../flashinfer_mla_sparse_sm120.py`
- **What it enables:** GLM-5.3-Flash is NoPE (`qk_rope_head_dim == 0`). The only
  sparse-MLA backend admitting compute capability 12 required the packed
  `fp8_ds_mla` layout, which assumes DeepSeek's `pe_dim == 64`, so the model
  aborted in `concat_and_cache_mla`. The fix maps NoPE onto the packed layout by
  zero-filling the RoPE section — zero rope dims contribute nothing to q·k, so
  the maths is unchanged.
- **Evidence:** standalone kernel probe matches a bf16 reference within fp8
  quantization noise (rel err ~2.7%, cosine 0.9996); the full model returns the
  fixed `391` check and exact needle retrieval at 113,986 tokens.
- **Deployed config:** `vllm-code:/opt/glm53-exp/serve-glm53-128k.sh` —
  128k context, KV pinned via `--kv-cache-memory-bytes`, with
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.
- **Upstream watch:** vllm-project/vllm#53906 ("add GLM-5.3-Flash support") is
  open and much larger (93 files, +14k, needs FlashInfer 0.6.18). Check it
  before porting this forward.

---

## 3. Composability

All three branches touch **disjoint file sets** and have been verified to apply
in sequence, in any order, onto one stock upstream tree. Deploy any subset.

Verify with `-- vllm/ csrc/`. A whole-branch diff also carries `tests/`, which
drifts against upstream faster than the code does and is not what a deployment
consumes.

Rule we hold ourselves to: **the branch is the atomic unit.** If a future
overlay depends on another, it must carry that other overlay's changes rather
than assume it was applied first. Commits inside a branch may stack; branches
may not.

To extract a patch, diff against the branch's **base**, not `upstream/main` —
our branches deliberately lag upstream, so diffing against the tip reverts
unrelated commits:

```bash
BASE=$(git merge-base upstream/main <branch>)
git diff $BASE..<branch> -- vllm/ > overlay.patch
```

---

## 4. Not ours

Owned by the fleet / shared vLLM role, not this repo:

- Image construction, registry tags and digests, service profiles, mounts,
  secrets, canary and rollback.
- The FlashInfer Python 3.11 shim and the FastAPI / Starlette /
  Prometheus-instrumentator pins.
- Release-specific cache namespaces. Native and container runtimes must not
  share Torch/AOT/startup-plan/FlashInfer roots — measured cross-runtime
  invalidation and expensive rebuilds.
- **Log destination and retention.** vLLM has no native file sink (unlike
  sglang's `--log-requests-target` and llama.cpp's `--log-prompts-dir`);
  everything goes to stdout → journald, and **nothing rotates**. Prompts and
  completions are plaintext. A `FileHandler` in
  `VLLM_LOGGING_CONFIG_PATH` plus a bind-mount would give parity and avoid
  journald's rate-limited drops under load.

## 5. Also worth knowing

- **`main` mirrors `upstream/main`** (`5707355209`) apart from this file. Branch
  new work from it, not from the old frozen baseline.
- The serving trees are **not** this fork. DSv4 runs `jasl/vllm` PR#41834
  (`20260809`); GLM-5.3 runs a local build (`487ecf187`, on no remote). We
  author here and ship patches there, so upstreaming reduces our maintenance
  burden but does not propagate to the fleet on its own.
- Historical, do not deploy: `glm-504b-gb10-sm121`,
  `glm-504b-gb10-fast-loading` (GB10/DGX Spark), and the isolated loader
  experiments `bounded-pinned-weight-loader`, `pp-weight-filter-before-read`,
  `fastsafetensors-loader-controls`. Rationale in
  `plans/decisions/README.md` on `codex/fastsafetensors-parallel-mtp-share`.
