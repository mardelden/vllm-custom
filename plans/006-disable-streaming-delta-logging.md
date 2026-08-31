# Note to deployment — set `--no-enable-log-deltas` on every vLLM container

**From:** vLLM-code / fork team · **Date:** 2026-08-31 · **Self-contained**
**Answers:** `proxmox/plans/handovers/vllm-streaming-delta-log-volume.md`
**Catalogue:** [`DEPLOYMENT-BRANCHES.md`](../DEPLOYMENT-BRANCHES.md)

Short answer: **the flag you asked for already exists.** No patch, no branch, no
code change from us. Add one argument.

```
--no-enable-log-deltas
```

---

## Why it works

`enable_log_deltas` is a frontend arg defaulting to **`True`**, which is why you
are getting a line per token. Its own docstring is exactly your ask:

> *"If set to False, output deltas will not be logged. Relevant only if
> `--enable-log-outputs` is set."*

It gates the delta call site directly
(`chat_completion/serving.py`, `if delta_content_parts and self.enable_log_deltas`).

**Present in every version we run**, same default:

| build | declared at | default |
| --- | --- | --- |
| 0.21.0 (`vllm-chat`, `vllm-ocr`) | `entrypoints/openai/cli_args.py:148` | `True` |
| jasl DSv4 (`20260809`) | `entrypoints/openai/cli_args.py:147` | `True` |
| current upstream | `entrypoints/launchers/cli_args.py:154` | `True` |

Exposed as `[--enable-log-deltas | --no-enable-log-deltas]`.

## Verified

On `vllm-build` (Qwen3-0.6B, `--reasoning-parser qwen3`), one streaming request:

| | before | after |
| --- | ---: | ---: |
| `(streaming delta)` | ~850/request | **0** |
| `(streaming complete)` | 1 | **1** |

Your redundancy argument holds — the complete record carries the **full** text:

```
Generated response <id> (streaming complete): output: '<think>\nOkay, the user
said "deltacheck. Reply: OK". Let me think about how to respond...'
```

`previous_texts` is accumulated independently of delta logging — the code says so
explicitly (`serving.py:445`, *"Always track previous_texts for comprehensive
output logging"*). Turning deltas off loses nothing.

## One thing to expect, so nobody misreads it

**`[reasoning:` will drop to ~0 on streaming traffic. That is not a regression.**

Thinking still arrives in the complete record, but as raw `<think>...</think>`
inside the accumulated text rather than wrapped in the `[reasoning: ...]` marker.
In our test `[reasoning:` went to 0 while the full thinking block was captured
verbatim. **Grep `<think>` for streaming, `[reasoning:` for non-streaming.**

So the file will carry two shapes depending on whether the client streamed. Your
5,442 `[reasoning:` lines were delta lines; expect that counter to collapse while
capture stays complete.

## What we would not do

- **Do not demote deltas to DEBUG.** The record is still formatted for every
  token before a handler drops it, so you pay the CPU and keep the flag
  semantics muddled. The flag skips the call entirely.
- **Do not filter on message text.** You flagged this yourself and you are right
   — matching `(streaming delta)` breaks silently the day the wording changes.
  Unnecessary now.

## Keep it per-deployment, not global-forever

Deltas do earn their place for one job: debugging **truncation or latency**,
where you want to see *when* tokens arrived, not just the final text. Treat this
as a per-deployment setting — off for capture, temporarily on when chasing a
streaming bug.

## Your framing was the more important finding

Bytes were never the risk at 158 KB/request. The real hazard is that
`Generated response` was not a per-request counter — you read 5,117 "completions"
that were tokens, off by 2–3 orders of magnitude. With this flag it becomes a
true per-request count, which is what anyone will naively grep.

## Apply

Add to the serve args on every vLLM container, alongside the existing
`--enable-log-requests --enable-log-outputs`:

```
--enable-log-requests --enable-log-outputs --no-enable-log-deltas
```

In the role this belongs next to `vllm_log_requests`, so it lands on all
containers uniformly rather than per-host.

**Verify:**

```bash
L=/var/log/inference/requests.log            # your per-container bind mount
grep -c "streaming delta"    $L    # expect 0
grep -c "streaming complete" $L    # expect 1 per streaming request
grep -c "Generated response" $L    # now a real per-request counter
grep -c "<think>"            $L    # thinking still captured on streaming turns
```

## Also noted from your report

- **`filename` adapted to `/var/log/inference/requests.log`** — correct. The
  in-container path in our config follows vLLM's own convention; fleet
  uniformity is the bind mount's job, so adapting it is expected, not a fork.
- **`--reasoning-parser qwen3` on `vllm-chat`** — that was your call to make and
  the tradeoff you state is the right one. It is also what makes `[reasoning:`
  meaningful there at all.
- **`vllm-code`'s image-based unit** still needs the bind-mount overlay or a new
  image for the reasoning fix and skip-video, as in
  [`plans/005`](005-qwen3-skip-video-geometry-deployment.md) and the logging
  runbook.
