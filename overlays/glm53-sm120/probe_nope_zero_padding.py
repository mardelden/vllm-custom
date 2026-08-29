# Standalone probe: can the SM120 packed sparse-MLA kernel serve a NoPE model
# via zero-padded RoPE?  (GLM-5.3-Flash: kv_lora_rank=512, qk_rope_head_dim=0)
#
# 1. Write a fp8_ds_mla (656 B/token) cache with vLLM's real concat_and_cache_mla,
#    passing k_pe = zeros[64]  -> tests the pe_dim==64 write path with zero rope.
# 2. Decode with FlashInfer's backend="sparse" v32/GLM path, q padded 512->576,
#    kv_scale_format="arbitrary_fp32" (what this image selects for model_type glm*).
# 3. Compare against a bf16 reference over the same top-k gather.
import torch, math

torch.manual_seed(0)
dev = torch.device("cuda:0")
DC, DR = 512, 64            # kv_lora_rank, padded rope dim
BLOCK = 64
TOPK = 2048
NBLOCKS = 64                # 4096 token slots
NTOK_KV = NBLOCKS * BLOCK

from vllm import _custom_ops as ops

def run_case(T, H):
    kv_c = (torch.randn(NTOK_KV, DC, device=dev, dtype=torch.bfloat16) * 0.5)
    k_pe = torch.zeros(NTOK_KV, DR, device=dev, dtype=torch.bfloat16)
    kv_cache = torch.zeros(NBLOCKS, BLOCK, 656, device=dev, dtype=torch.uint8)
    slot_mapping = torch.arange(NTOK_KV, device=dev, dtype=torch.long)
    scale = torch.tensor(1.0, device=dev, dtype=torch.float32)
    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                             kv_cache_dtype="fp8_ds_mla", scale=scale)

    q_c = (torch.randn(T, H, DC, device=dev, dtype=torch.bfloat16) * 0.3)
    q = torch.cat([q_c, torch.zeros(T, H, DR, device=dev, dtype=torch.bfloat16)], -1)

    # distinct global token slots per query token
    idx = torch.stack([torch.randperm(NTOK_KV, device=dev)[:TOPK] for _ in range(T)])
    idx = idx.to(torch.int32)

    sm_scale = 1.0 / math.sqrt(DC + DR)
    out = q.new_empty(T, H, DC)
    ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=dev)

    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as f
    r = f(query=q.unsqueeze(1),
          kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
          workspace_buffer=ws,
          qk_nope_head_dim=256, kv_lora_rank=DC, qk_rope_head_dim=DR,
          block_tables=idx.unsqueeze(1),
          seq_lens=None, max_seq_len=TOPK,
          out=out.unsqueeze(1),
          bmm1_scale=sm_scale, bmm2_scale=1.0,
          sparse_mla_top_k=TOPK,
          kv_scale_format="arbitrary_fp32")
    torch.cuda.synchronize()

    # reference in fp32 over the same gather (unquantized KV)
    kv_g = kv_c[idx.long()].float()                       # [T, K, 512]
    s = torch.einsum("thd,tkd->thk", q_c.float(), kv_g) * sm_scale
    p = torch.softmax(s, dim=-1)
    ref = torch.einsum("thk,tkd->thd", p, kv_g)

    o = out.float()
    rel = (o - ref).norm() / ref.norm()
    cos = torch.nn.functional.cosine_similarity(
        o.flatten().unsqueeze(0), ref.flatten().unsqueeze(0)).item()
    print(f"T={T:3d} H={H:3d}  rel_err={rel.item():.4f}  cos={cos:.6f}  "
          f"out_std={o.std().item():.4f} ref_std={ref.std().item():.4f}")
    return rel.item(), cos

print("device:", torch.cuda.get_device_name(0))
ok = True
for T, H in [(1, 32), (4, 32), (64, 32), (1, 64), (16, 64)]:
    rel, cos = run_case(T, H)
    ok &= (cos > 0.99 and rel < 0.15)
print("PROBE:", "PASS - NoPE-by-zero-padding works on the sm120 packed kernel"
      if ok else "FAIL - outputs do not match reference")
