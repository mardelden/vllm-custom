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
mkdir -p /var/log/inference               # bind-mount this per container
export VLLM_LOGGING_CONFIG_PATH=/etc/vllm/logging-config.json
```

`/var/log/inference/` is the fleet-standard path shared with sglang and
llama.cpp, so all three engines land in one place. Give **each container its own
host directory**:

```
-v /srv/logs/inference/<container>:/var/log/inference
```

That is what keeps writers from colliding, so the in-container filename can stay
fixed. A random filename suffix would also avoid collisions but makes files
unattributable, proliferates one file per restart, and `backupCount` prunes per
handler so orphans from earlier runs are never cleaned up.

The `vllm` role already exports `VLLM_LOGGING_CONFIG_PATH` and already passes
`--enable-log-requests --enable-log-outputs` (`vllm_log_requests: true`), so on
fleet hosts this is a file swap plus a mount.

## What it captures

Verified on `vllm-build` with Qwen3-0.6B — one request produces four distinct
lines in `/var/log/inference/vllm-requests.log`, all sharing a request id:

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
- **Size-based rotation, not time-based.** `RotatingFileHandler` at
  128 MiB x 40 bounds disk at ~5 GiB per container. Time-based hourly rotation
  reads more naturally for a collector, but leaves file size unbounded — and the
  shared pool on this cluster sits at 86% and has hit 100% once before
  (plans/049 wedged 14 containers). Bounded disk is the more important property
  for verbatim prompt capture. Either way, not SGLang's `backupCount=0`, which
  rotates but **never deletes**.
- **Request content goes to the file only; journald keeps startup and crashes.**
  The three request loggers drop the stdout handler, so prompts and completions
  are not duplicated into the journal. The `vllm` logger still streams to stdout,
  and uncaught tracebacks reach stderr regardless of Python logging, so
  systemd still captures startup and crash output. Verified: 0 request records
  leaked to stdout, startup lines still present.
- **Avoids journald's silent drops.** `RateLimitIntervalSec=30s` /
  `RateLimitBurst=10000` are at defaults on these hosts, and journald
  **discards** beyond that. Capture emits ~4 multi-KB records per request, so a
  busy engine could lose records with no error — a complete file and a holed
  journal is the worst outcome when comparing the two.

## Retention and privacy

These files contain **verbatim prompts and completions**, including source code
and anything pulled into context. ~5 GiB per container is a disk bound, not a
retention policy.
Retention, access control and whether this should be on encrypted storage are
open questions for the fleet/release side, not settled here.

## Known trigger for revisiting the filename

One writer per container today: `lsof` shows only the API-server process holds
the file, and the service template sets no `--api-server-count`. Enabling
`--api-server-count > 1` or data parallelism would put **several writer
processes in one container**, where the per-container directory no longer
disambiguates and `RotatingFileHandler` is not multiprocess-safe. That is when a
pid or rank belongs in the filename.

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
