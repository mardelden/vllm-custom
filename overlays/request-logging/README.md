# Prompt + completion capture to file

Routes vLLM's request logger to a rotating file so prompts, completions and
thinking land in one bind-mountable place, instead of only stdout -> journald.

**This is configuration, not a code patch.** vLLM's request logger is a plain
`init_logger(__name__)`, so a `VLLM_LOGGING_CONFIG_PATH` dictConfig can route
it anywhere. No fork divergence is required for the file sink itself.

It does, however, depend on the code fix in this same branch: without it,
reasoning models log no completion at all (see `overlays/dsv4-logging/`).

## Deploy

```bash
install -D -m644 logging-config.json /etc/vllm/logging-config.json
mkdir -p /var/log/vllm/requests          # bind-mount this for containers
export VLLM_LOGGING_CONFIG_PATH=/etc/vllm/logging-config.json
```

The `vllm` role already exports `VLLM_LOGGING_CONFIG_PATH` and already passes
`--enable-log-requests --enable-log-outputs` (`vllm_log_requests: true`), so on
fleet hosts this is a file swap plus a mount.

## What it captures

Verified on `vllm-build` with Qwen3-0.6B — one request produces four distinct
lines in `/var/log/vllm/requests/requests.log`, all sharing a request id:

```
Request <id> details: prompt: '<|im_start|>user\nfilecapture zulu...'   <- DEBUG
Received request <id>: params: SamplingParams(...)                      <- INFO
Generated response <id> details: output_token_ids: [...]                <- DEBUG
Generated response <id>: output: '[reasoning: \nOkay, the user...'      <- INFO
```

## Design notes

- **Three logger names, not one.** The request logger moved between builds:
  `vllm.entrypoints.logger` (0.21), `vllm.entrypoints.request_logger`, and
  `vllm.entrypoints.serve.utils.request_logger` (current). A config naming only
  one captures nothing on the other builds, **with no error** — the logger
  simply does not exist. All three are listed so one file works fleet-wide.
- **Handler level must be DEBUG.** Prompts are logged at DEBUG
  (`log_inputs`); completions at INFO (`log_outputs`). An INFO handler silently
  drops the prompt half.
- **`propagate: false`** keeps request records off the root logger. Verified
  no duplicate lines.
- **`delay: true`** avoids creating the file until something is written, so a
  server that never serves does not leave an empty log.
- **`backupCount: 168`** — hourly rotation, one week retained, then pruned.
  Deliberately not `0`: SGLang's equivalent uses `backupCount=0`, which rotates
  hourly and **never deletes**, so plaintext prompts accumulate forever.

## Retention and privacy

These files contain **verbatim prompts and completions**, including source code
and anything pulled into context. One week is a starting point, not a policy.
Retention, access control and whether this should be on encrypted storage are
open questions for the fleet/release side, not settled here.

## Not included

Deliberately not ported from SGLang's `request_logger.py`:

- **Verbosity levels** (metadata / +params / truncated / verbatim) and
  field redaction. Upstream vLLM issue #40155 has an active design for this
  (`--log-content {none,truncated,full}` plus a redaction hook); duplicating it
  in the fork would be throwaway work.
- **Live reconfiguration** (`POST /configure_logging`). Genuinely useful given
  a multi-minute weight load — worth asking for on #40155 rather than forking.

If that design stalls and we do port it, two bugs to avoid: SGLang's
`_transform_data_for_logging` drops `skip_names` in its recursive calls, so
redaction is top-level only and nested content leaks; and its list branch
compares an element count against a character budget.
