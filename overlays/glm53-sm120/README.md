# GLM-5.3-Flash on sm120: NoPE sparse MLA via zero-padded RoPE

**Result (2026-08-29):** `RedHatAI/GLM-5.3-Flash-NVFP4` serves on 2x RTX PRO 6000
(sm120, TP=2) for the first time. Correctness: `391`, coherent code generation,
exact 6k-token needle retrieval at depths 25/50/90% through chunked prefill.
Single-stream decode 80.2 tok/s at `max-num-seqs 1`.

## Mechanism

`FLASHINFER_MLA_SPARSE_SM120` requires the packed 656-byte `fp8_ds_mla` layout
(512 NoPE fp8 + 16 scales + 128 RoPE bf16). GLM-5.3-Flash is NoPE
(`qk_rope_head_dim == 0`), so `concat_and_cache_mla` aborted with
`pe_dim must be 64`. FlashInfer's sm120 sparse kernel, however, already ships a
GLM model type (`_MODEL_TYPE_GLM_NSA`, selected by `kv_scale_format=
"arbitrary_fp32"`, which vLLM already picks for `model_type glm*`).

Zero rope dims contribute nothing to q.k scores, so NoPE maps onto the packed
layout exactly: write `k_pe = zeros[64]` into the cache, pad q 512 -> 576 with
zeros at decode, and report rope dim 64 to the kernel. A standalone probe
(`probe_nope_zero_padding.py`, run inside the vendor image) validated the full
write+decode path against a bf16 reference: rel err ~2.7% (pure fp8
quantization), cosine 0.9996, across T = 1..64 and H = 32/64.

## Artifacts

- `glm53-nope-image-overlay-r1.patch` — the validated single-file overlay
  against the vendor image `vllm/vllm-openai:glm53-flash`
  (vLLM `0.1.dev20051+g487ecf187`, a local build; commit not on upstream).
  Apply to `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`.
- The branch code change is the same fix rebased onto `upstream/main`, which
  additionally needs `get_supported_head_sizes` extended to `[512, 576]`
  (the vendor image already ships that part).
- `probe_nope_zero_padding.py` — correctness probe; runs standalone in the
  image, no server needed.

## Context scaling (measured 2026-08-29)

Per-token KV cost falls with model length as block rounding and KDA state
amortize: ~21 KB/token at 16k, ~11 KB at 64k, ~9.5 KB at 128k. The profiler's
CUDA-graph estimate over-charges ~0.6 GiB against a ~0.3 GiB real capture
(`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`), and its activation peak is
conservative: pinning KV directly with `--kv-cache-memory-bytes 1525000000`
reaches **131,072 max-model-len** (pool 160,496 tokens, 1.22x) with a measured
2.3 GiB physical margin under a full 114k-token prefill. Needle retrieval is
exact at 113,986 tokens, depths 25/50/90%. Serve scripts:
`vllm-code:/opt/glm53-exp/serve-glm53-{64k,128k}.sh`.

## Known limits

- Memory at default profiling: fits 16k at `--gpu-memory-utilization 0.98
  --max-num-batched-tokens 1024` (25,746 tokens KV); 128k requires the pinned
  KV + disabled graph estimate above.
- `glm5_next_mtp` speculative decoding is **capacity-blocked** on 2x96 GB:
  the MTP layer (+ ~3.5 GiB/card) OOMs during weight load, 606 MiB short.
  Not a code failure; needs TP=4 or smaller weights.
- Dead ends measured and rejected: unpacked bf16 cache (five layers of fixes
  end at `TllmGenFmhaRunner: Unsupported architecture` -- trtllm-gen has no
  sm120 build); DSV4 d_qk=512 path (kernel exists at 584 B/token but vLLM has
  no writer for that layout).
