# DSv4 `--enable-log-outputs`: completions not logged for reasoning models

Response to `proxmox/plans/handovers/dsv4-request-logging-gap.md` (2026-08-31).
Target tree: `jasl/vllm` PR#41834, `vllm.__version__ == 20260809`,
`vllm-code:/opt/vllm-dsv4/vllm-src`.

## The reported root cause is not the actual one

The note concludes that `api_server.py` was restructured, never constructs
`OpenAIServingChat`, and therefore leaves `enable_log_outputs` at its `False`
default. The first two observations are correct; the conclusion is not.

The wiring exists and is reached at runtime:

```
vllm/entrypoints/openai/api_server.py:236   imports register_generate_api_routers
vllm/entrypoints/generate/api_router.py:57  init_generate_state(...)
vllm/entrypoints/generate/api_router.py:169     enable_log_outputs=args.enable_log_outputs,
vllm/entrypoints/generate/api_router.py:174     OpenAIServingChat(**_chat_kwargs)
```

Construction simply moved out of `api_server.py` into the generate router, so
grepping `api_server.py` finds nothing while the flag is threaded correctly.
`self.enable_log_outputs` is `True` on the running server and the
`if self.enable_log_outputs` guards pass.

**It is also not a fork regression.** `upstream/main` carries the identical
content-only guard (`serving.py:1185-1191`), so stock vLLM behaves the same way
on a reasoning model. The stock-0.21 comparison in the note most likely differed
because that model returned `content`, not because the fork is broken.

## The actual defect

In the non-streaming path the logged text is built from `content` and
`tool_calls` only:

```python
output_text = ""
if choice.message.content:      # reasoning models can leave this empty
    output_text = choice.message.content
elif choice.message.tool_calls:
    ...
if output_text:                 # empty -> log_outputs() never called
```

DSv4 runs `--reasoning-parser deepseek_v4`, which routes thinking to
`ChatMessage.reasoning` (`protocol.py:75-76`), populated at
`serving.py:941-949`. Two consequences:

1. A reasoning-only or truncated-mid-thought response leaves `content` empty and
   logs **nothing**. The note's probe used `max_tokens: 25`, which a reasoning
   model can spend entirely on thinking — that is why the count was 0.
2. Even when `content` is present, **thinking is never logged**, which defeats
   the stated goal of capturing prompt + completion + thinking in one place.

The streaming delta path already handles this correctly at `serving.py:672-673`
using a `[reasoning: ...]` convention. Only the non-streaming path was missing it.

## A second, mirrored gap in the debug middleware

`VLLM_DEBUG_LOG_API_SERVER_RESPONSE=true` is the workaround #24578 was closed
on. It is **not** a substitute, because it has the same defect in the opposite
mode:

| | `--enable-log-outputs` | `VLLM_DEBUG_LOG_API_SERVER_RESPONSE` |
|---|---|---|
| non-streaming content | yes | yes (raw JSON) |
| non-streaming **reasoning** | **no** (fixed here) | yes (raw JSON) |
| streaming content | yes (needs `--enable-log-deltas`) | yes |
| streaming **reasoning** | yes, already | **no** (fixed here) |

`_extract_content_from_chunk` reads only `delta.content`, so a streaming
reasoning-only response logs `streaming_complete: no_content`. It also
truncates at 2048 chars, which a reasoning trace exceeds immediately, and the
middleware buffers the whole response body before releasing it — it warns
"avoided in production" on startup for good reason.

Since a coding harness streams by default, the workaround is arguably the worse
option for fleet capture.

## The fix

Two mirrored changes so neither logging path drops thinking:

1. `chat_completion/serving.py` — add reasoning to `output_text` in the
   non-streaming `log_outputs` guard, following the streaming path's existing
   `[reasoning: ...]` convention.
2. the response-logging middleware's `_extract_content_from_chunk` — fall back
   to `delta.reasoning` when `delta.content` is empty.

Note the middleware lives in different places per tree: `serve/utils/server_utils.py`
in the jasl fork, `serve/middleware/log_response.py` in current upstream.
`dsv4-log-outputs-reasoning.patch` targets the fork layout.

Verified with `test_log_outputs_guard.py` (non-streaming; lifts the patched
block out of the shipped file and runs it against real `ChatMessage` objects)
and `test_stream_extractor.py` (streaming; calls the real patched extractor).
Both are GPU-free. 12 cases, all passing:

```
PASS  content only               -> 'OK'
PASS  reasoning only (the bug)   -> '[reasoning: thinking...]'
PASS  reasoning + content        -> '[reasoning: thinking...] OK'
PASS  empty-string content       -> '[reasoning: t]'
PASS  reasoning_content alias    -> '[reasoning: alt]'
PASS  nothing at all             -> ''

PASS  content only               -> 'OK'          (streaming)
PASS  reasoning only (the bug)   -> 'thinking...' (streaming)
PASS  reasoning + content        -> 'OK'          (streaming)
PASS  empty delta                -> ''            (streaming)
PASS  null content, reasoning    -> 't'           (streaming)
PASS  text_completion            -> 'hi'          (streaming)
```

## Prior reports upstream (duplicate check)

The specific diagnosis appears to be new, but the symptom is not:

- [#24578](https://github.com/vllm-project/vllm/issues/24578) CLOSED COMPLETED —
  `--enable-log-outputs` not working on **Qwen3-4B-Thinking**, a thinking model.
  Almost certainly this bug. Closed on the `VLLM_DEBUG_LOG_API_SERVER_RESPONSE`
  workaround; root cause never found, so the code is unchanged today.
- [#25918](https://github.com/vllm-project/vllm/issues/25918) CLOSED COMPLETED —
  same symptom.
- [#19462](https://github.com/vllm-project/vllm/issues/19462) CLOSED NOT_PLANNED —
  empty content in response log.
- [#40155](https://github.com/vllm-project/vllm/issues/40155) OPEN, active — a
  `--log-content {none,truncated,full}` design for prompt/output logging. About
  log levels and redaction, not this defect, but it is the live conversation on
  request logging; raise this fix in its orbit rather than in isolation.
- No open PR touches `log_outputs`.

Also relevant: `reasoning_content` was renamed to `reasoning`, and clients still
reading the old field silently see empty. The fix checks `reasoning` first and
falls back to `reasoning_content`, so it works either side of the rename.

## Notes on the rest of the report

- **Defect 1 (logger name) stands** and the deployed `logging-config.json`
  workaround is correct. Worth knowing: `log_outputs()` emits at
  `logger.info`, not DEBUG, so completion logging needs no DEBUG config at
  all — only the prompt does.
- **"Fail loudly" ask:** not applicable as framed, since the flag does reach its
  consumer. The equivalent guard here would be a warning when a reasoning parser
  is active and a response produces no loggable text.
- **Streaming:** delta logging is additionally gated on `--enable-log-deltas`,
  which the current `serve-dsv4.sh` does not set. The streaming-complete log at
  `serving.py:827` uses accumulated `previous_texts` and is unaffected by this
  patch.
