# AI-Stock-Radar

## Alpaca market data setup (read-only)

`src/data/alpaca_market_data.py` is a read-only client for Alpaca's
market data API (US equities + crypto). It never places orders.

1. Create an Alpaca **paper trading** account and generate an API
   key pair from the paper dashboard.
2. `cp .env.example .env` and fill in `ALPACA_API_KEY` /
   `ALPACA_SECRET_KEY`. `.env` is gitignored — never commit it.
3. Dependencies (including `alpaca-py`) are pinned in
   `requirements.txt`:
   ```bash
   python3.14 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
4. Run the manual smoke test (real network calls, not part of the
   pytest suite):
   ```bash
   .venv/bin/python scripts/smoke_test_alpaca_market_data.py
   ```
   It fetches a few days of bars and a latest quote for one equity
   (`AAPL`) and one crypto pair (`BTC/USD`), and prints `OK` on
   success. If `.env` isn't filled in yet, it prints `SKIPPED` and
   exits cleanly instead of failing.

Data trade-off, accepted deliberately for this stage: equities use
Alpaca's free-tier `IEX` feed (not the full SIP consolidated tape),
and all REST data is subject to Alpaca's free-tier ~15 minute delay.
See the module docstring in `alpaca_market_data.py` for details.
