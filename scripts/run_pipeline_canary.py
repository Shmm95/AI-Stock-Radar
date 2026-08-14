"""Read-only operational canary for the live Alpaca/Telegram pipeline.

This script deliberately has no trading authority.  It reads the Alpaca
paper account, recent market-data bars, current broker positions, and open
orders, then sends the resulting health summary through the project's
existing Telegram notifier.  It never submits, replaces, modifies, or
cancels an order and never writes local portfolio state.

The emergency STOP path is imported from ``run_daily_decision.py`` so the
canary cannot drift onto a different kill-switch location.  STOP prevents
every Alpaca call.  FREEZE is reported but does not suppress this canary:
FREEZE blocks new exposure, while this script is strictly observational.
"""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from alpaca.data.timeframe import TimeFrame  # noqa: E402
from alpaca.trading.client import TradingClient  # noqa: E402
from alpaca.trading.enums import QueryOrderStatus  # noqa: E402
from alpaca.trading.requests import GetCalendarRequest, GetOrdersRequest  # noqa: E402

from run_daily_decision import FREEZE_FLAG_PATH, STOP_FLAG_PATH  # noqa: E402
from src.data import alpaca_market_data as market_data  # noqa: E402
from src.live.live_universe import LIVE_CONTROLLED_TICKERS  # noqa: E402
from src.notify.telegram_notifier import send_telegram_message  # noqa: E402

EQUITY_SAMPLE_COUNT = 2
CRYPTO_SAMPLE_COUNT = 1
CRYPTO_STALE_AFTER_HOURS = 48.0
BAR_LOOKBACK_DAYS = 7
CALENDAR_LOOKBACK_DAYS = 10
MAX_OPEN_ORDERS_TO_REQUEST = 500
NEW_YORK = ZoneInfo("America/New_York")

_LABEL = "AI-Stock-Radar READ-ONLY pipeline canary"


def _enum_value(value: Any) -> str:
    return str(value.value if hasattr(value, "value") else value)


def _normalize_symbol(symbol: str) -> str:
    """Make Alpaca's BTC/USD or BTCUSD comparable with BTC-USD."""
    return "".join(character for character in symbol.upper() if character.isalnum())


def _is_crypto_ticker(ticker: str) -> bool:
    return ticker.upper().endswith("-USD")


def _sample_tickers() -> tuple[tuple[str, str], ...]:
    equities = [ticker for ticker in LIVE_CONTROLLED_TICKERS if not _is_crypto_ticker(ticker)]
    crypto = [ticker for ticker in LIVE_CONTROLLED_TICKERS if _is_crypto_ticker(ticker)]
    selected = [(ticker, "equity") for ticker in equities[:EQUITY_SAMPLE_COUNT]]
    selected.extend((ticker, "crypto") for ticker in crypto[:CRYPTO_SAMPLE_COUNT])
    return tuple(selected)


def _as_utc(value: Any, *, naive_timezone: ZoneInfo | None = None) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        value = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=naive_timezone or UTC)
    return value.astimezone(UTC)


def _latest_bar_timestamp(frame: Any) -> datetime:
    if frame.empty:
        raise ValueError("Alpaca returned no bars")
    timestamps = []
    for index_value in frame.index:
        timestamp_value = index_value[-1] if isinstance(index_value, tuple) else index_value
        timestamps.append(_as_utc(timestamp_value))
    return max(timestamps)


def _latest_completed_equity_session(
    client: TradingClient, *, now: datetime
) -> date:
    request = GetCalendarRequest(
        start=(now - timedelta(days=CALENDAR_LOOKBACK_DAYS)).date(),
        end=now.date(),
    )
    sessions = client.get_calendar(filters=request)
    completed = [
        session
        for session in sessions
        if _as_utc(session.close, naive_timezone=NEW_YORK) <= now
    ]
    if not completed:
        raise ValueError("Alpaca returned no completed market session")
    return max(session.date for session in completed)


def _fetch_latest_bar(ticker: str, *, asset_class: str, now: datetime) -> datetime:
    start = now - timedelta(days=BAR_LOOKBACK_DAYS)
    if asset_class == "crypto":
        alpaca_symbol = ticker.replace("-", "/")
        frame = market_data.get_crypto_bars(
            alpaca_symbol,
            start=start,
            end=now,
            timeframe=TimeFrame.Hour,
        )
    else:
        frame = market_data.get_stock_bars(
            ticker,
            start=start,
            end=now,
            timeframe=TimeFrame.Hour,
        )
    return _latest_bar_timestamp(frame)


def _notify_safe(text: str) -> bool:
    try:
        return bool(send_telegram_message(text))
    except Exception:
        return False


def _format_position(position: Any) -> str:
    return (
        f"  - {position.symbol}: {_enum_value(position.side)} qty={position.qty}, "
        f"available={position.qty_available}, asset={_enum_value(position.asset_class)}"
    )


def _format_order(order: Any) -> str:
    quantity = f"qty={order.qty}" if order.qty is not None else f"notional={order.notional}"
    order_type = order.type if order.type is not None else order.order_type
    return (
        f"  - {order.symbol}: {_enum_value(order.side)} {_enum_value(order_type)} "
        f"{quantity}, status={_enum_value(order.status)}"
    )


def _account_checks(client: TradingClient, *, findings: list[str]) -> list[str]:
    try:
        account = client.get_account()
    except Exception as error:
        findings.append(f"Alpaca account/auth check failed ({type(error).__name__})")
        return [f"Account/auth: FAILED ({type(error).__name__})"]

    status = _enum_value(account.status)
    crypto_status = _enum_value(account.crypto_status)
    blocked_flags = [
        name
        for name in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
        if bool(getattr(account, name, False))
    ]
    if status.lower() != "active":
        findings.append(f"paper account status is {status}")
    if crypto_status.lower() != "active":
        findings.append(f"paper crypto status is {crypto_status}")
    if blocked_flags:
        findings.append(f"paper account restriction(s): {', '.join(blocked_flags)}")

    restriction_text = ", ".join(blocked_flags) if blocked_flags else "none"
    return [
        f"Account/auth: OK (paper endpoint, status={status}, "
        f"crypto_status={crypto_status}, restrictions={restriction_text})"
    ]


def _market_data_checks(
    client: TradingClient, *, now: datetime, findings: list[str]
) -> list[str]:
    lines = ["Market data:"]
    try:
        latest_equity_session = _latest_completed_equity_session(client, now=now)
    except Exception as error:
        latest_equity_session = None
        findings.append(f"equity market calendar check failed ({type(error).__name__})")
        lines.append(f"  - Equity calendar unavailable ({type(error).__name__})")

    samples = _sample_tickers()
    if not samples:
        findings.append("live universe contains no canary sample tickers")
        lines.append("  - No sample tickers available")
        return lines

    for ticker, asset_class in samples:
        try:
            timestamp = _fetch_latest_bar(ticker, asset_class=asset_class, now=now)
        except Exception as error:
            findings.append(f"{ticker} bar check failed ({type(error).__name__})")
            lines.append(f"  - {ticker}: FAILED ({type(error).__name__})")
            continue

        age_hours = max(0.0, (now - timestamp).total_seconds() / 3600.0)
        if asset_class == "crypto":
            fresh = age_hours <= CRYPTO_STALE_AFTER_HOURS
            expectation = f"limit={CRYPTO_STALE_AFTER_HOURS:g}h"
        elif latest_equity_session is None:
            fresh = age_hours <= CRYPTO_STALE_AFTER_HOURS
            expectation = f"calendar unavailable; fallback limit={CRYPTO_STALE_AFTER_HOURS:g}h"
        else:
            fresh = timestamp.date() >= latest_equity_session
            expectation = f"expected session={latest_equity_session.isoformat()}"

        state = "FRESH" if fresh else "STALE"
        lines.append(
            f"  - {ticker}: {state}, latest={timestamp.isoformat()}, "
            f"age={age_hours:.1f}h ({expectation})"
        )
        if not fresh:
            findings.append(f"{ticker} market data is stale")
    return lines


def _position_checks(client: TradingClient, *, findings: list[str]) -> list[str]:
    try:
        positions = sorted(client.get_all_positions(), key=lambda position: position.symbol)
    except Exception as error:
        findings.append(f"broker position read failed ({type(error).__name__})")
        return [f"Broker positions: FAILED ({type(error).__name__})"]

    allowed = {_normalize_symbol(ticker) for ticker in LIVE_CONTROLLED_TICKERS}
    lines = [f"Broker positions: {len(positions)} open"]
    if not positions:
        lines.append("  - none")
        return lines

    for position in positions:
        lines.append(_format_position(position))
        if _normalize_symbol(str(position.symbol)) not in allowed:
            findings.append(f"position outside live universe: {position.symbol}")
        if _enum_value(position.side).lower() == "short":
            findings.append(f"unexpected short position: {position.symbol}")
    return lines


def _open_order_checks(client: TradingClient, *, findings: list[str]) -> list[str]:
    try:
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            limit=MAX_OPEN_ORDERS_TO_REQUEST,
        )
        orders = sorted(client.get_orders(filter=request), key=lambda order: order.symbol)
    except Exception as error:
        findings.append(f"open-order read failed ({type(error).__name__})")
        return [f"Broker open orders: FAILED ({type(error).__name__})"]

    allowed = {_normalize_symbol(ticker) for ticker in LIVE_CONTROLLED_TICKERS}
    lines = [f"Broker open orders: {len(orders)} pending/open"]
    if not orders:
        lines.append("  - none")
        return lines

    for order in orders:
        lines.append(_format_order(order))
        if _normalize_symbol(str(order.symbol)) not in allowed:
            findings.append(f"open order outside live universe: {order.symbol}")
        client_order_id = str(getattr(order, "client_order_id", "") or "")
        if client_order_id.startswith("HEARTBEAT_"):
            findings.append(f"abandoned heartbeat design still has an open order: {order.symbol}")
    return lines


def build_canary_report(*, now: datetime | None = None) -> tuple[str, bool]:
    now = (now or datetime.now(UTC)).astimezone(UTC)
    findings: list[str] = []
    freeze = FREEZE_FLAG_PATH.exists()

    api_key, secret_key = market_data._require_credentials()
    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)

    lines = [
        f"{_LABEL} -- {now.isoformat()}",
        "Mode: READ-ONLY; no order submission/cancel/replace calls exist in this script.",
        (
            f"FREEZE: present at {FREEZE_FLAG_PATH}; ignored by design because this "
            "run cannot change exposure."
            if freeze
            else "FREEZE: not present."
        ),
    ]
    lines.extend(_account_checks(client, findings=findings))
    lines.extend(_market_data_checks(client, now=now, findings=findings))
    lines.extend(_position_checks(client, findings=findings))
    lines.extend(_open_order_checks(client, findings=findings))

    if findings:
        lines.append(f"Overall: REVIEW REQUIRED ({len(findings)} finding(s))")
        lines.extend(f"  - {finding}" for finding in findings)
    else:
        lines.append("Overall: PASS — all read-only canary checks succeeded; no anomaly found.")
    return "\n".join(lines), not findings


def main() -> int:
    if STOP_FLAG_PATH.exists():
        text = (
            f"{_LABEL} -- STOP flag detected ({STOP_FLAG_PATH}); "
            "run skipped before every Alpaca call. FREEZE policy was not evaluated."
        )
        print(text)
        if not _notify_safe(text):
            print(
                "WARNING: STOP notification could not be confirmed sent.",
                file=sys.stderr,
            )
        return 0

    try:
        text, passed = build_canary_report()
    except Exception as error:
        text = (
            f"{_LABEL} FAILED at {datetime.now(UTC).isoformat()}\n"
            f"{type(error).__name__}: operational checks could not start or complete.\n"
            "No order was submitted, modified, replaced, or cancelled."
        )
        passed = False

    print(text)
    notified = _notify_safe(text)
    if not notified:
        print(
            "WARNING: pipeline-canary Telegram notification could not be confirmed sent.",
            file=sys.stderr,
        )
    return 0 if passed and notified else 1


if __name__ == "__main__":
    raise SystemExit(main())
