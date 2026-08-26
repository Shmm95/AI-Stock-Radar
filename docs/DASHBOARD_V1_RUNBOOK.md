# Dashboard V1 — Deployment & Operations Runbook

Read-only, mobile-first portfolio/P&L dashboard. Two processes, two
system users, one atomically-produced JSON file between them. Never
touches the live trading engine — no order is ever placed, modified,
or canceled from anywhere in this subsystem.

## Architecture

```
[ai-dashboard-producer]                    [ai-dashboard]
scripts/generate_dashboard_snapshot.py     src/dashboard/http_app.py
  (systemd timer, every 5 min)               (systemd service, always on)
  reads: Alpaca API, data/live/*.json,       reads: the snapshot file
         .guard/order_intents/*.json                 + static/*
  writes: dashboard_snapshot_v1.json (atomic)
              |
              v
  /var/lib/ai-stock-radar-dashboard/dashboard_snapshot_v1.json
              |
              v
  http_app.py serves it at GET /api/v1/snapshot (127.0.0.1:8765)
              |
              v
  cloudflared (separate, not part of this repo) -> public URL
```

The two processes never talk to each other directly — only through
that one file. `ai-dashboard-producer` can read real credentials and
the `.guard` directory; `ai-dashboard` can read neither, only the
snapshot file and the three static frontend files (see
`src/dashboard/snapshot_builder.py`'s and `src/dashboard/http_app.py`'s
own module docstrings for the full import-ban list each one enforces).

## One-time host setup

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin ai-dashboard-producer
sudo useradd --system --no-create-home --shell /usr/sbin/nologin ai-dashboard

sudo mkdir -p /var/lib/ai-stock-radar-dashboard
sudo chown ai-dashboard-producer:ai-dashboard /var/lib/ai-stock-radar-dashboard
sudo chmod 750 /var/lib/ai-stock-radar-dashboard
```

`ai-dashboard-producer` writes into that directory; `ai-dashboard`
(added to the same group) only needs read access to the one file
inside it, never write.

Grant `ai-dashboard-producer` read access to `data/live/` and the real
`.guard` directory (`~/.ai_stock_radar_guard` or wherever
`AI_STOCK_RADAR_GUARD_DIR` points on this host) via normal Unix group
permissions — it never needs write access to either.

## Install

```bash
sudo cp deploy/systemd/ai-stock-radar-dashboard-snapshot.service /etc/systemd/system/
sudo cp deploy/systemd/ai-stock-radar-dashboard-snapshot.timer /etc/systemd/system/
sudo cp deploy/systemd/ai-stock-radar-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-stock-radar-dashboard-snapshot.timer
sudo systemctl enable --now ai-stock-radar-dashboard.service
```

Verify:

```bash
sudo systemctl status ai-stock-radar-dashboard-snapshot.timer
sudo systemctl status ai-stock-radar-dashboard.service
curl -s http://127.0.0.1:8765/healthz
curl -s http://127.0.0.1:8765/api/v1/snapshot | head -c 400
```

## Exposing it publicly (owner's own Cloudflare step)

Point `cloudflared` (run as its own process/unit, outside this repo's
scope) at `http://127.0.0.1:8765`. The HTTP service's own systemd unit
deliberately does **not** set `PrivateNetwork=true` — that would
sandbox it into its own network namespace and cut `cloudflared` off
from ever reaching loopback at all. `IPAddressAllow=localhost` +
`IPAddressDeny=any` already restrict the service to loopback-only
traffic without that side effect. Put authentication (if any is
wanted) at the Cloudflare Access layer, not in this service — it has
none of its own.

## Reading the dashboard's own status fields

- **`collection_status`**: `OK` (everything read cleanly), `PARTIAL`
  (the core P&L/account section is fine, something secondary — recent
  trades, decision log, intent summary — failed; check `errors[]`), or
  `ERROR` (should never actually appear in a written file — a core
  failure means the producer refused to write at all, see below).
- **Staleness**: the frontend polls every 30s and marks itself `STALE`
  once `generated_at_utc` is more than 10 minutes old — at a 5-minute
  producer cadence, two full misses in a row before it goes red.
- **`system.replay_gate_present_today`** vs **`system.last_processed_equity_session_date`**
  / **`settled_sessions_behind`**: two INDEPENDENT indicators, never
  merge them when reading this by hand either. A present gate is not
  proof the cursor is fresh; a fresh cursor from yesterday does not
  mean today's gate has been written yet.
- **`system.intent_summary`**: a heavily sanitized view of the
  order-intent journal — see `src/dashboard/models.py`'s
  `IntentSummaryView` docstring for the *complete* list of fields this
  is allowed to ever contain. `ATTENTION` means at least one intent is
  `UNCERTAIN` (needs a human); `IN_FLIGHT` means something is mid-flow
  but not alarming yet; `UNKNOWN` means the journal itself couldn't be
  read/parsed cleanly — treat that the same as "no data," never as "no
  intents."

## Troubleshooting

**Snapshot never updates / stuck STALE.** Check the producer's own
last run:

```bash
sudo systemctl status ai-stock-radar-dashboard-snapshot.service
sudo journalctl -u ai-stock-radar-dashboard-snapshot.service -n 50
```

A core failure (bad/expired Alpaca credentials, network outage) makes
the producer exit non-zero **and leaves the previous snapshot file
completely untouched** — this is deliberate (see
`scripts/generate_dashboard_snapshot.py`'s own module docstring): a
half-written or wrong snapshot is worse than an honestly stale one. Fix
the underlying broker/credential issue; the next timer tick picks back
up automatically, no manual restart needed for the file itself.

**HTTP service returns 503 on `/api/v1/snapshot`.** The snapshot file
is missing, unreadable, or not valid JSON — normal immediately after
first install (before the producer's first successful run) or if
`/var/lib/ai-stock-radar-dashboard/` permissions are wrong. Check the
producer's own logs above.

**A field seems wrong / a ticker looks off.** `recent_trades[]` and
`positions[]` show REAL broker-reported tickers/quantities on purpose
(this is the owner's own portfolio) — that is not a sanitization bug.
Only the order-intent journal section (`system.intent_summary`) is
under the strict field-allow-list; see `src/dashboard/models.py`.

## Security model summary

- Two Unix users, least-privilege split (producer sees credentials and
  `.guard`; the HTTP service sees neither).
- HTTP service: `NoNewPrivileges`, `ProtectSystem=strict`,
  `ProtectHome`, `PrivateTmp`, empty `CapabilityBoundingSet`,
  loopback-only binding + `IPAddressAllow/Deny`, strict CSP
  (`default-src 'self'`), `Cache-Control: no-store`, no cookies, no
  auth of its own (put that at the Cloudflare layer if wanted).
- GET-only, exactly three routes (`/`, `/api/v1/snapshot`, `/healthz`)
  — every other method/path is 404/405 by construction, never a route
  that exists but silently no-ops.
- No live-trading-engine module is ever imported by any file under
  `src/dashboard/` or by the producer script — enforced by
  `tests/test_dashboard_read_only_ast.py`.
- Order-intent journal data reaches the snapshot only through the
  fixed, tested allow-list in `IntentSummaryView` — enforced by the
  leak test in `tests/test_dashboard_snapshot.py`.
