Loading model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

To enable this feature, use the `--load-format fastsafetensors` command-line argument.

The loader's queue depth, reader count, bounce-buffer size, and maximum copy block
size can be tuned without changing process-wide environment variables:

```bash
vllm serve <model> \
  --load-format fastsafetensors \
  --model-loader-extra-config '{
    "queue_size": -1,
    "max_threads": 16,
    "bbuf_size_kb": 16384,
    "max_copy_block_size": 67108864,
    "use_o_direct": "auto"
  }'
```

`queue_size=-1` keeps shard production synchronous, which bounds shard-sized
residency. The other settings control parallel requests within that residency
limit. Increase them only after measuring storage throughput and peak host/device
memory on the target system.

With a compatible FastSafetensors version, `use_o_direct=true` bypasses the OS
page cache for aligned no-GDS reads. `"auto"` uses buffered reads for a mostly
resident checkpoint and O_DIRECT for a cold checkpoint. Leave this unset unless
the filesystem and FastSafetensors build support the option; direct I/O requires
aligned internal requests and can be slower than buffered warm-cache reads.
