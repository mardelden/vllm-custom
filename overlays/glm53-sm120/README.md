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

## Known limits

- Memory: fits at `--gpu-memory-utilization 0.98 --max-num-batched-tokens 1024`
  with 0.52 GiB KV headroom (25,746 tokens, 1.57x concurrency at 16k).
  At 0.97/2048 profiling ends 0.52 GiB short.
- `glm5_next_mtp` speculative decoding is **capacity-blocked** on 2x96 GB:
  the MTP layer (+ ~3.5 GiB/card) OOMs during weight load, 606 MiB short.
  Not a code failure; needs TP=4 or smaller weights.
- Dead ends measured and rejected: unpacked bf16 cache (five layers of fixes
  end at `TllmGenFmhaRunner: Unsupported architecture` -- trtllm-gen has no
  sm120 build); DSV4 d_qk=512 path (kernel exists at 584 B/token but vLLM has
  no writer for that layout).
