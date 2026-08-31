# Runbook — apply the Qwen3 skip-video-geometry patch to every vLLM container

**From:** vLLM-code / fork team · **Date:** 2026-08-31 · **Self-contained**
**Branch:** `mardelden/vllm-custom@codex/qwen3-vl-skip-zero-video` (`13dd9c9858`)
**Catalogue:** [`DEPLOYMENT-BRANCHES.md`](../DEPLOYMENT-BRANCHES.md)

Small, low-risk, **1 file / +10 −6**. Apply everywhere for uniformity; it only
changes behaviour where it can help, and is inert elsewhere.

**Read the "Expectations" section before you measure anything** — this one is
easy to mis-assess.

---

## What it does

`Qwen3VLDummyInputsBuilder.get_dummy_mm_data` computes video geometry
*unconditionally*: it resolves override warnings, calls `get_video_processor()`
and reads merged mm kwargs **before** it ever looks at `num_videos`. For a
request or profiling pass with no video, that work is built and thrown away.

The patch returns early with an empty video list once the images are built.

## Which containers it actually touches

Do **not** go by model name. `Qwen3VLDummyInputsBuilder` lives in `qwen3_vl.py`
but is reused by eight model modules, including `qwen3_5.py`:

| container | model | architecture | module | loads the builder |
| --- | --- | --- | --- | --- |
| `vllm-code` | Qwen3.8-27B-NVFP4 | `Qwen3_5ForConditionalGeneration` | `qwen3_5` | **yes** |
| `vllm-chat` | Qwen3.6-35B-A3B-NVFP4 | `Qwen3_5MoeForConditionalGeneration` | `qwen3_5` | **yes** |
| `vllm-ocr` | olmOCR-2-7B | `Qwen2_5_VLForConditionalGeneration` | `qwen2_5_vl` | no |
| `vllm-embed` | Qwen3-Embedding-8B | `Qwen3ForCausalLM` | `qwen3` | no |
| `vllm-whisper` | whisper-large-v3 | `WhisperForConditionalGeneration` | `whisper` | no |

On the three "no" rows the file is never imported, so the patch is inert. It is
still safe to apply — that keeps every container on one source state.

## Expectations — read this before measuring

**Where it fires** depends on the video count at dummy-data time, which comes
from `--limit-mm-per-prompt`:

- **`vllm-code`** sets `'{"image":4}'` with **no video key**, so `num_videos`
  is 0 and the early return **does** take effect.
- **`vllm-chat`** sets no limit, so profiling uses the model maximum, the video
  count is non-zero, and the patch is **inert** there.

**How much it saves is unmeasured, and probably small.** The processor lookup it
skips (`get_video_processor()` → `get_hf_processor()`) is memoised by
`cached_processor_from_config`, so on a warm cache you save geometry arithmetic,
not a processor build. The meaningful case is a cold path where nothing else has
touched the processor yet.

**We are not claiming a startup-time win.** If you want one, time it — otherwise
treat this as a correctness tidy-up (do not compute video geometry for a request
with no video) that happens to remove work.

## Apply

```bash
BRANCH=codex/qwen3-vl-skip-zero-video
git clone -b $BRANCH https://github.com/mardelden/vllm-custom.git /tmp/vc
cd /tmp/vc && git diff $(git merge-base upstream/main HEAD)..HEAD -- vllm/ > /tmp/skip-video.patch
```

Or take the file directly — the change is confined to
`vllm/model_executor/models/qwen3_vl.py`.

Then, against the installed vLLM:

```bash
cd <site-packages>            # or the source tree for an editable install
patch -p1 --dry-run < /tmp/skip-video.patch     # verify first
patch -p1 -b        < /tmp/skip-video.patch     # -b writes .orig backups
```

Rollback: restore the `.orig`, or `patch -R -p1 < /tmp/skip-video.patch`.

**For image-based services** (`vllm-code`'s main unit runs an immutable image):
bind-mount the single patched file over `site-packages`, or fold it into the
next image build. Pure Python — no rebuild required either way.

## Verify

It is a startup-path change with no log line of its own, so verify structurally
rather than behaviourally:

```bash
# the guard is present
grep -n 'if num_videos == 0' <site-packages>/vllm/model_executor/models/qwen3_vl.py

# the server still starts and serves
curl -s localhost:<port>/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"<model>","max_tokens":16,
       "messages":[{"role":"user","content":"skipvideo probe. Reply: OK"}]}'
```

On `vllm-code`, confirm multimodal still works end to end — it limits to
`{"image":4}`, so send an `image_url` request and check the response is sane.
That is the path this patch is adjacent to, and the one worth smoke-testing.

## Risk

- **Low.** One function, one early return, no new dependencies, no config.
- Still applies cleanly to current upstream despite **five** upstream changes to
  this file since it was written, which suggests the surrounding code is stable.
- It composes with the other overlay branches; it shares `qwen3_vl.py` with the
  startup branch but edits a different function, verified to apply in **both**
  orders.
- **Unmeasured** — that is the honest weakness. Apply for uniformity and
  correctness, not on a promised number.

## Provenance

Written as `768c007dc6` during the RTX/Qwen work, parked because no performance
measurement was preserved, then rebased onto current upstream as
`codex/qwen3-vl-skip-zero-video`. Its commit message and the branch catalogue
both record that it is carried for durability rather than benchmarked benefit.
