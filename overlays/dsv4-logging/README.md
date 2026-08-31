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

## The fix

`dsv4-log-outputs-reasoning.patch` (+18 lines, one file) adds reasoning to
`output_text` in the non-streaming path, following the streaming path's existing
convention. Apply to
`vllm/entrypoints/openai/chat_completion/serving.py`.

Verified with `test_log_outputs_guard.py`, which lifts the patched block out of
the shipped file and executes it against real `ChatMessage` objects (no GPU):

```
PASS  content only               -> 'OK'
PASS  reasoning only (the bug)   -> '[reasoning: thinking...]'
PASS  reasoning + content        -> '[reasoning: thinking...] OK'
PASS  empty-string content       -> '[reasoning: t]'
PASS  reasoning_content alias    -> '[reasoning: alt]'
PASS  nothing at all             -> ''
```

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
