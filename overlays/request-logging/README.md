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
mkdir -p /var/log/vllm                    # bind-mount this per container
export VLLM_LOGGING_CONFIG_PATH=/etc/vllm/logging-config.json
```

The in-container path follows vLLM's own convention (`/var/log/<service>/`).
vLLM has no native log directory of its own -- it logs to stdout only -- so
`/var/log/vllm/` is the closest thing to a default, and it keeps this config
meaningful outside our fleet.

Fleet uniformity is the **bind mount's** job, not the container's. Give each
container its own host directory under whatever standard the collector wants:

```
-v /srv/logs/inference/<container>:/var/log/vllm
```

That is also what keeps writers from colliding, so the in-container filename can
stay fixed. A random filename suffix would also avoid collisions but makes files
unattributable, proliferates one file per restart, and `backupCount` prunes per
handler so orphans from earlier runs are never cleaned up.

The `vllm` role already exports `VLLM_LOGGING_CONFIG_PATH` and already passes
`--enable-log-requests --enable-log-outputs` (`vllm_log_requests: true`), so on
fleet hosts this is a file swap plus a mount.

## What it captures

Verified on `vllm-build` with Qwen3-0.6B — one request produces four distinct
lines in `/var/log/vllm/requests.log`, all sharing a request id:

```
Request <id> details: prompt: '<|im_start|>user\nfinalcheck whiskey...' <- DEBUG
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
- **One file for prompts and completions, not two.** They share a
  `request_id`, so correlation is a grep. Splitting them would let a prompt be
  pruned while its completion survives, since two files fill at different rates
  -- an orphaned half of a pair is worth nothing when the whole point is showing
  what the model received *and* returned. Verified under 8 concurrent requests:
  32 records, **0 torn or interleaved lines**, 8/8 ids carrying both halves.
  Python's logging locks per `emit()`, so records from different requests
  interleave in order but never corrupt each other.
- **Request content goes to the file only; journald keeps startup and crashes.**
  The three request loggers drop the stdout handler, so prompts and completions
  are not duplicated into the journal. The `vllm` logger still streams to stdout,
  and uncaught tracebacks reach stderr regardless of Python logging, so
  systemd still captures startup and crash output. Verified: 0 request records
  leaked to stdout, startup lines still present.
- **uvicorn's loggers must be named explicitly.** uvicorn configures its own
  loggers at import time. A dictConfig that names none of them leaves them with
  no handler, and its output disappears **silently** -- startup banner,
  `Application startup complete`, and the HTTP access log all vanish, while
  vLLM's own INFO logs keep flowing so nothing looks wrong. Measured: access-log
  lines went 2 -> 0 before `uvicorn`, `uvicorn.error` and `uvicorn.access` were
  added back. Readiness checks that grep for `Application startup complete`
  break on this.
- **Avoids journald's silent drops.** `RateLimitIntervalSec=30s` /
  `RateLimitBurst=10000` are at defaults on these hosts, and journald
  **discards** beyond that. Capture emits ~4 multi-KB records per request, so a
  busy engine could lose records with no error — a complete file and a holed
  journal is the worst outcome when comparing the two.

## Rotation is Python's job -- do not point logrotate at the live file

`RotatingFileHandler` rotates and prunes on its own. The live file is
`requests.log`; rotated artifacts are `requests.log.1` .. `requests.log.40`.
Verified: with `backupCount=N` the handler deletes beyond N, so disk is bounded
at ~5 GiB per container with **no external pruning needed**.

This differs from SGLang, whose equivalent uses `backupCount=0` -- it rotates
hourly and never deletes, so that deployment *does* need an external sweeper.
Ours does not.

Adding logrotate on the **live** file breaks it either way, silently:

- `create` (default) -- logrotate renames the file and makes a new one. Python
  still holds the old descriptor and keeps writing into the renamed file; the
  new one stays empty.
- `copytruncate` -- Python keeps its descriptor **and its byte offset**. After
  truncation it writes at the old offset, producing a sparse file padded with
  NULs.

Same reason you cannot `rm` the log on a running server: measured **199 open
descriptors still pointing at the unlinked inode**, with writes continuing into
a file that no longer had a name. Restart the service, or leave rotation alone.

If a sweeper is ever wanted anyway -- for example to prune faster than
`backupCount` -- glob only the **rotated** pattern, never the live file:

```bash
find /var/log/vllm -name 'requests.log.[0-9]*' -mtime +7 -delete
```

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
