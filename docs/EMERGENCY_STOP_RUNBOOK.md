# Emergency Stop / Freeze Runbook

For the moment something looks wrong and you need the live system to
stop **right now**, without opening an SSH session and hand-editing
crontab under pressure. Two flag files, checked by
`scripts/run_daily_decision.py` and `scripts/run_crypto_stop_monitor.py`
before either script makes a single API call. Content is never read —
only whether the file exists. Both live under `data/live/`, which is
git-ignored, so a deploy (`git pull`, `git checkout`, `git clean`) can
never delete or overwrite them.

## The two flags

### `data/live/STOP` — full stop

Both scripts exit immediately, before any Alpaca or market-data call.
An open crypto position's stop-loss watch **also stops** — only use
this if you specifically want that (e.g. you are about to close
everything by hand anyway), not as the default "something looks off"
response.

### `data/live/FREEZE` — freeze new entries only (recommended default)

Only affects `run_daily_decision.py`. Data still gets fetched, and
every existing position's exit/stop logic keeps running exactly as
normal — nothing you already hold loses its protection. The only thing
that stops is opening anything new: no new entry signals get queued,
and any buy that was already queued from a previous run stays queued
rather than executing. `run_crypto_stop_monitor.py` is **not** affected
by this flag and keeps watching crypto stops regardless — this is the
whole reason the two flags are separate.

**Default recommendation: use FREEZE first.** Reach for STOP only when
you deliberately want the crypto watcher stopped too.

## Exactly what to type

Replace `<server>` with your actual SSH alias/host and `<repo-path>`
with the deployed repo's path on that server — fill these in once you
have them, this file intentionally does not guess at either.

**To freeze (stop new entries, keep watching existing positions):**

```bash
ssh <server> 'touch <repo-path>/data/live/FREEZE'
```

**To fully stop (both scripts exit immediately, no API calls at all):**

```bash
ssh <server> 'touch <repo-path>/data/live/STOP'
```

**To resume normal operation** (remove the flag — either one):

```bash
ssh <server> 'rm -f <repo-path>/data/live/FREEZE <repo-path>/data/live/STOP'
```

Each command is a single line, no heredoc, no paste-sensitive
multi-line input — the exact problem this replaces.

## What you'll see

Every run that finds either flag sends a Telegram message immediately,
before doing anything else:

- STOP: `"...STOP flag detected (data/live/STOP); this run was skipped
  entirely, no API calls were made. Remove the file to resume."`
- FREEZE: `"...FREEZE flag detected (data/live/FREEZE); this run will
  still fetch data and monitor/exit existing positions as usual, but
  will NOT open any new positions. Remove the file to resume normal
  entries."`

A frozen `run_daily_decision.py` run also finishes normally afterward
(fetches data, runs exit/stop logic, sends the usual end-of-day
summary) — you'll get two messages that run: the freeze notice, then
the normal summary. Its decision log also records `"freeze_active":
true` for that day, so it's auditable after the fact, not just visible
in the moment.

## Checking whether a flag is currently set

```bash
ssh <server> 'ls -la <repo-path>/data/live/ | grep -E "STOP|FREEZE"'
```

No output for a name means that flag is not set.
