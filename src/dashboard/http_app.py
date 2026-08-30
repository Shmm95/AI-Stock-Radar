"""Dashboard V1 -- the read-only HTTP service. Runs under the
unprivileged `ai-dashboard` system user (see `deploy/systemd/
ai-stock-radar-dashboard.service`): no credentials, no `.guard`
directory access, no write access anywhere except (implicitly) none at
all -- this process only ever reads two things: the three static
frontend files and the one atomically-produced snapshot JSON file.

IMPORT DISCIPLINE (owner's own explicit rule, 2026-08-25): this module
NEVER imports `src.live.order_intent`, `src.live.order_intent_reconciliation`,
`src.live.order_submission`, or any Alpaca trading-client class/module.
Enforced by `tests/test_dashboard_read_only_ast.py`. This module does
not import `src.dashboard.snapshot_builder` either -- that module is
the PRODUCER's own concern (a separate process, a separate system
user); this module only ever reads the JSON FILE the producer already
wrote, never calls into the builder or the broker directly.

ROUTES -- exactly three, GET (+ the implicit HEAD Starlette derives
from each GET handler), nothing else:
    GET /                 -- the single-page frontend (index.html with
                             its CSS/JS inlined at request time, each
                             wrapped with a fresh per-request CSP nonce
                             -- see index()'s own comment for why: kept
                             CSS/JS as literal inline content rather
                             than adding /static/* routes, which would
                             break the "exactly three routes" contract)
    GET /api/v1/snapshot  -- the current snapshot JSON, or 503 if it is
                             missing/corrupt/unreadable
    GET /healthz          -- a trivial liveness probe, no snapshot read
Any other method on these paths, or any other path at all, is a 404/405
-- no other route is ever registered.
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

DEFAULT_SNAPSHOT_PATH = Path("/var/lib/ai-stock-radar-dashboard/dashboard_snapshot_v1.json")
STALE_AFTER_MINUTES = 10.0

_STATIC_DIRECTORY = Path(__file__).resolve().parent / "static"

_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    # index() overrides this value with a per-request nonce'd style-src/
    # script-src -- this base CSP (no inline anything) is what /healthz
    # and /api/v1/snapshot actually get, and what / gets before index()'s
    # override; kept here rather than duplicated so both stay in sync.
    "Content-Security-Policy": "default-src 'self'; connect-src 'self'; frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def _snapshot_path() -> Path:
    """`DASHBOARD_SNAPSHOT_PATH` env var overrides the default -- the
    only test-injection seam this module has; production never sets
    it, so production always reads the real, fixed path."""
    return Path(os.environ.get("DASHBOARD_SNAPSHOT_PATH", str(DEFAULT_SNAPSHOT_PATH)))


def _with_security_headers(response: Response) -> Response:
    for key, value in _SECURITY_HEADERS.items():
        response.headers[key] = value
    return response


def _render_index_html(nonce: str) -> str:
    html = (_STATIC_DIRECTORY / "index.html").read_text(encoding="utf-8")
    css = (_STATIC_DIRECTORY / "dashboard.css").read_text(encoding="utf-8")
    js = (_STATIC_DIRECTORY / "dashboard.js").read_text(encoding="utf-8")
    html = html.replace("/*__DASHBOARD_CSS__*/", css)
    html = html.replace("/*__DASHBOARD_JS__*/", js)
    html = html.replace("__DASHBOARD_NONCE__", nonce)
    return html


async def index(request: Request) -> Response:
    # A fresh nonce per request (never 'unsafe-inline') is what lets the
    # inlined <style>/<script> tags run under a CSP that still has no
    # style-src/script-src wildcard -- browsers only execute inline
    # content whose tag-attribute nonce matches the header's nonce for
    # THIS specific response. index.html's two tags carry the same
    # placeholder, substituted here, never a fixed/predictable value.
    nonce = secrets.token_urlsafe(18)
    response = _with_security_headers(HTMLResponse(_render_index_html(nonce)))
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        f"style-src 'self' 'nonce-{nonce}'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        "connect-src 'self'; frame-ancestors 'none'"
    )
    return response


async def healthz(request: Request) -> Response:
    return _with_security_headers(PlainTextResponse("ok"))


async def snapshot(request: Request) -> Response:
    path = _snapshot_path()
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return _with_security_headers(
            JSONResponse({"error": "snapshot_unavailable"}, status_code=503)
        )

    generated_at = payload.get("generated_at_utc")
    is_stale = True
    if generated_at:
        try:
            generated = datetime.fromisoformat(generated_at)
            if generated.tzinfo is None:
                generated = generated.replace(tzinfo=UTC)
            age_minutes = (datetime.now(UTC) - generated).total_seconds() / 60.0
            is_stale = age_minutes > STALE_AFTER_MINUTES
        except ValueError:
            is_stale = True

    body = dict(payload)
    body["_dashboard_stale"] = is_stale
    return _with_security_headers(JSONResponse(body, status_code=200))


routes = [
    Route("/", index, methods=["GET"]),
    Route("/api/v1/snapshot", snapshot, methods=["GET"]),
    Route("/healthz", healthz, methods=["GET"]),
]

app = Starlette(routes=routes)
