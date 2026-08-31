# DeepSeek-V4-Flash startup optimization deployment handover

- **Status:** Source contract implemented and measured on `vllm-code`; fleet packaging required
- **Date:** 2026-08-25
- **Patch set:** `vllm-dsv4-startup` release 1 (candidate)
- **Base:** `jasl/vllm@aa0d51302747ea80f282e26949708b3253409fe2` (version `20260809`, PR#41834 sm12x)
- **Dependency:** `mardelden/fastsafetensors@b43888df0eac286849f1238b7e42e254ee1d285f`
- **Model:** `MJPansa/DeepSeek-V4-Flash-0731-NVFP4@64d64cd89bc63a66aa46506da89d7821f7491c62`
- **Host measured:** `vllm-code` LXC on `pve-ai`, 2x RTX PRO 6000 Blackwell (sm120), TP=2, 96 GB RAM

This is **not** the `vllm-generic-startup` line. That patch set targets stock vLLM 0.27.1 via
`mardelden/vllm-custom`. DeepSeek-V4 runs on the `jasl` tree, so its releases are numbered
separately and its manifests must never be mixed with `overlays/generic-startup/`.

## Outcome

| Stage | Model loading | API ready |
| --- | ---: | ---: |
| Default safetensors loader | 171.37 s | — |
| `--load-format fastsafetensors` (as previously deployed) | 110-115 s | 152.21 s |
| This patch set | **48.02 s** | **84.11 s** |

Model loading is reduced by **72%** and process-to-API-ready by **45%**. Readiness reproduced at
84.09 / 84.10 / 84.11 / 86.10 s across runs.

## Ownership boundary

| Owner | Responsibility |
| --- | --- |
| vLLM-code team (this repo) | Source patches, dependency pin, applicability classification, measurements, correctness evidence, removal conditions |
| Deployment/fleet repository | Derived image or venv overlay, source assertions, profiles, mounts, canary, promotion, rollback |

Everything below is a portable fact to adapt. The scratch tree on `vllm-code`
(`/opt/vllm-dsv4/vllm-src`, patched in place) is **not** a release input.

## Applicability: what is generic and what is model-specific

This is the decision you asked for. The patch set has three independent parts.

### Part A - generic loader defect fix (applies to every model)

**Recommend applying fleet-wide to any model served with `--load-format fastsafetensors`.**

FastSafetensors defaults `max_copy_block_size` to 16 GiB. Every real shard is smaller, so
`submit_io()` emits **one read request per shard**, and the C++ reader serves each request with
**one thread** while the other 15 in its pool idle. The bounce buffer is already allocated at
`bbuf_size_kb * max_threads`, so the parallelism is paid for and never used.

This is a property of the library's defaults, not of DeepSeek. Any model whose shards are smaller
than 16 GiB is affected, which in practice is all of them.

- Effect here: target weight load **50.6 s -> 36.3 s** (-28%).
- Mechanism: size the copy block below the shard size so the existing thread pool is fed.
- Cost: none. No extra host or device memory.

### Part B - generic dependency upgrade (applies to every model, benefit scales with model size)

**Recommend applying fleet-wide; largest benefit when the checkpoint exceeds host RAM.**

Stock `fastsafetensors==0.3.3` opens checkpoints `O_RDONLY`. With a 163.5 GiB checkpoint on a 96 GB
host the page cache thrashes and every read pays buffered-copy and eviction cost.

Measured on this host's NVMe (Samsung 9100 PRO, PCIe 5.0):

| Read path | Bandwidth |
| --- | ---: |
| buffered, 1 thread | 2.79 GiB/s |
| O_DIRECT, 1 thread | 10.05 GiB/s |
| O_DIRECT, 16 threads | 11.35 GiB/s |

`mardelden/fastsafetensors@b43888d` adds `use_o_direct` (`true` / `false` / `"auto"`), where
`"auto"` samples cache residency with `RWF_NOWAIT` and picks buffered for resident files and
aligned O_DIRECT for cold ones.

- Effect here: target weight load **36.3 s -> 29.1 s** (-19%).
- For models that fit comfortably in host RAM, expect little or no gain; `"auto"` is the safe
  default because it degrades to buffered reads on a warm cache.
- The fork is a single commit on upstream and is also **newer** than PyPI 0.3.3, so it additionally
  brings upstream's chunk-budget planner.

### Part C - model-specific draft ownership (DeepSeek-V4 only, but the pattern generalizes)

**Apply only to DeepSeek-V4 profiles.** The code is gated on the DSpark draft model class.

The DSpark draft consumes only `mtp.*` plus the shared embedding/head - 99 destination parameters
from **7.4%** of the checkpoint - yet it re-read all 24 shards. The existing `weight_name_filter`
in this tree filters *after* materialization, so the bytes were read and discarded.

| Group | Size | Share |
| --- | ---: | ---: |
| `layers.*` (target only) | 151.39 GiB | 92.60% |
| `mtp.*` (draft only) | 10.12 GiB | 6.19% |
| shared (`embed`, `head`, `norm`, `hc_head_*`) | 1.98 GiB | 1.21% |

- Effect: draft weight load **50.96 s -> 6.31 s** (-88%).
- The generic half of this change (plumbing a keep-predicate into the loader) is model-agnostic and
  is a prerequisite for Part C. Only the predicate itself is DeepSeek-specific.
- **The pattern generalizes** to any target+draft architecture where the draft re-reads the full
  checkpoint: Qwen3.5 MTP, DeepSeek MTP, EAGLE heads. Each needs its own verified predicate.

## Components

| ID | Scope | File(s) | Effect | Removal condition |
| --- | --- | --- | ---: | --- |
| `fastsafetensors-copy-block-geometry` | generic loader | `weight_utils.py` | -14.2 s | Remove when the installed FastSafetensors chooses a request size below shard size by default |
| `fastsafetensors-adaptive-odirect` | generic loader | `weight_utils.py` + dependency | -6.7 s | Remove when a released FastSafetensors exposes equivalent adaptive direct I/O |
| `fastsafetensors-keep-filter-plumbing` | generic loader | `weight_utils.py`, `default_loader.py` | enabler | Remove when this tree passes range filters before read natively |
| `dspark-draft-range-filter` | DeepSeek-V4 only | `models/deepseek_v4/nvidia/dspark.py` | -44.6 s | Remove when the DSpark draft declares its own byte-range ownership upstream |
| `weight-load-phase-timing` | generic diagnostic | `default_loader.py` | 0 | Optional; keep for future attribution |

Total diff: **185 lines across 3 files**, all defaults off.

## Configuration contract

All knobs are environment variables, all default to the previous behavior. A deployment that sets
none of them behaves exactly as today.

```text
VLLM_DSPARK_DRAFT_RANGE_FILTER=1   # Part C. DeepSeek-V4 profiles only.
VLLM_FSST_COPY_BLOCK_MB=16         # Part A. Any value well below shard size; 16-64 measured equal.
VLLM_FSST_O_DIRECT=auto            # Part B. "auto" recommended fleet-wide; "1" forces direct.
```

Diagnostic only, leave unset in production:

```text
VLLM_FSST_PHASE_TIMING=1  # producer vs consumer split, ~0 overhead
VLLM_FSST_DEBUG=1         # per-request C++ timers; writes ~29 MB/run, do not enable in production
VLLM_FSST_MAX_THREADS / VLLM_FSST_BBUF_SIZE_KB / VLLM_FSST_MAX_BATCH_MB / VLLM_FSST_ALL_LOCAL
```

No new mounts, ports, secrets, or shared services. Resource profile is unchanged: 83.23 GiB VRAM
per rank, 96 GB host RAM, and the same `/mnt/models` checkpoint.

## Measured evidence

Weight-load phase split, from the instrumentation in this patch set:

| Load | Producer (read + H2D) | Consumer (destination) | Tensors |
| --- | ---: | ---: | ---: |
| Target | 19.32 s - 153.36 GiB yielded | 9.95 s | 133,660 |
| Draft (filtered) | 6.45 s - 12.09 GiB yielded | 0.17 s | 4,711 |

Inside the producer, across 186,556 well-formed per-request samples: `pread` is **98.8%** of reader
thread time and `cudaMemcpy` is **1.2%**. H2D is not a factor; pinned H2D measured 49.06 GiB/s.

The target phase reads 163.48 GiB across both ranks in 19.32 s = **8.46 GiB/s aggregate, about 75%
of this drive's measured 11.35 GiB/s ceiling**. Further weight-load work therefore has a low
ceiling; the remaining room is the 9.95 s consumer path and an uninstrumented ~11.6 s model
construction phase.

Note for interpretation: 153.36 GiB is the whole checkpoint minus `mtp.*`, not a per-rank share.
FastSafetensors splits *files* across TP ranks and broadcasts tensors, so each rank reads ~81.7 GiB,
sees 153.36 GiB, and keeps 83.23 GiB.

## Correctness evidence

- **Destination-parameter equality.** SHA-256 digests of all 99 draft parameters, both TP ranks,
  filtered vs unfiltered: **bit-identical, 99/99, on both ranks.** The range filter provably changes
  only which bytes are read.
- Fixed output: `17 * 23` returns `391`.
- Coding prompt returns a correct iterative Fibonacci implementation.
- Speculative decoding remains healthy: mean acceptance length 4.30, average draft acceptance 66.0%.
  A corrupted draft would accept nothing.

An earlier acceptance-rate difference (53.2% vs 45.2%) was sampling noise on 200-300 drafted tokens,
not degradation; the digest equality settles it and a later sample measured 66.0%.

## Promotion gates for the fleet

1. Assert base commit, dependency commit, model revision, and that the intended code markers exist
   inside the built artifact.
2. Log evidence that each enabled part engaged: the `Byte-range filter active` line, the
   `FastSafetensors geometry` line, and `o_direct=True`/`auto`.
3. Page-cold and resident readiness measured separately, three runs each, zero restarts.
4. Correctness: `391`, a coherent coding completion, and non-zero speculative acceptance.
5. Peak host RAM and per-rank VRAM unchanged from the current profile.
6. A proved rollback to the current unpatched configuration.

## Approaches measured and rejected - do not re-attempt without new evidence

| Approach | Result | Why |
| --- | --- | --- |
| `VLLM_FASTSAFETENSORS_QUEUE_SIZE>0` | CUDA OOM | Buffers `(queue+1)` shard-sized batches in VRAM; no headroom at 0.92 utilization |
| `max_batch_bytes` bounded pipelining | 36.3 s vs 29.1 s | Worse for the target; helped only the draft |
| 32 reader threads | 30.3 s vs 29.1 s | Contention |
| `bbuf_size_kb` 64 MiB | 35.8 s vs 36.2 s | Within noise; request size stayed 1 MiB regardless |
| `all_local=True` for the target | 36.4 s vs 29.1 s | Removes cross-rank broadcast but doubles read volume against a fixed drive ceiling |
| Target-side byte-range filter | not viable | Requires `all_local`, whose cost exceeds the 10.12 GiB saved |

## Defects found in the current build - report, not fixed here

1. **`VLLM_ENABLE_STARTUP_PLAN=1` is a no-op.** Plans are written every boot with stable
   fingerprints, but never applied: neither the "Applying persisted startup plan" nor the
   "not applied" log fires, so `maybe_apply_startup_plan` exits at its first guard because
   `kv_cache_memory_bytes` is already non-`None`. This was listed as a startup lever; it does
   nothing on this build. Its ceiling is small regardless, because schema 1 still runs
   `profile_run()` on a hit.
2. **The target overreads `mtp.*`.** Its `skip_weight_name_before_load` filter runs after
   materialization, so 10.12 GiB (6% of target read volume) is read and discarded every boot. Not
   cheaply fixable - see the rejected `all_local` row.
3. **Interleaved debug output.** The C++ reader's `printf` is unsynchronized across 16 threads;
   about 2% of debug lines are garbled. Aggregations must discard malformed lines.

## Rollback

- Source: `.orig` backups sit beside each patched file in `/opt/vllm-dsv4/vllm-src`.
- Dependency: `/opt/vllm-dsv4/fastsafetensors-0.3.3-backup.tgz`, or reinstall
  `fastsafetensors==0.3.3`. Note `pybind11` was added to the venv as a build dependency.
- Configuration: unset the three environment variables above; all code paths default off.
