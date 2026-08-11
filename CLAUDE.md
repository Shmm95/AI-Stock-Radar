# CLAUDE.md — Workspace B (Forward Research)

This worktree is **Workspace B**, branch `workspace/b-forward-research`,
branched from `checkpoint/pre-codex-20260802`. Full policy:
[docs/WORKSPACE_POLICY.md](docs/WORKSPACE_POLICY.md).

## Scope

This is where roadmap advancement, new features, and forward-stage
experimental research happen (execution/slippage stress, research baseline
lock, broader-universe robustness, RSI replacement research, and similar).
Everything produced here is a **research candidate** — never official or
approved until the owner audits and explicitly approves it in the main
planning chat. Hardening work (tests, audits, governance/doc debt on
approved stages) belongs in Workspace A (`AI-Stock-Radar-workspace-a`,
branch `workspace/a-hardening`), not here.

## Merge rule

Nothing in this branch ever merges directly into `checkpoint/pre-codex-20260802`
or into Workspace A's branch. Only the owner, in the main planning chat
(outside this repo), can approve promoting an output out of this workspace.

## Protected areas — never touch without explicit approval

- `src/backtest/portfolio_backtest_engine.py` (deterministic engine)
- entry rules, exit rules, risk management logic, execution order logic
- fees, slippage
- frozen baseline strategy parameters
- `config/research_baseline_lock_v1.json`

If a task here would require changing one of these, stop before editing and
ask for approval in the main planning chat.

## Known private-API coupling (never edit the coupled file to "fix" this)

`scripts/run_daily_decision.py` and `src/live/position_state.py` import
and call `portfolio_backtest_engine.py`'s underscore-prefixed internals
directly — `_PortfolioState`, `_MutablePosition`, `_PendingOrder`, and
the six per-bar step functions (`_execute_pending_exits_at_open`,
`_check_gap_stops`, `_execute_pending_buys_at_open`,
`_check_intrabar_stops`, `_queue_close_based_exits`,
`_queue_ranked_entry_signals`) — rather than reimplementing the frozen
strategy's decision logic for live, dry-run, one-bar-at-a-time use.
`scripts/run_daily_decision.py` also imports `_is_crypto_ticker` (used
to validate equity's and crypto's own latest calendar date
independently in `_bars_today_by_asset_class`, since equity closes on
weekends/holidays and crypto trades 24/7 — the two asset classes are
never required to share a date with each other, only within
themselves; see that function's docstring).
`src/live/data_preparer.py` similarly imports
`portfolio_backtest_engine._validate_market_data`,
`run_portfolio_backtest._bull_trend_variant`, and
`run_regime_ablation.build_regime_allowed`. `src/live/crypto_stop_monitor.py`
imports `_MutablePosition` only (to read a position's `stop_loss_price`
and `quantity`; it never calls any of the engine's per-bar step
functions — see that module's docstring for why the six-step batch
logic does not apply to a live, un-bar'd crypto price check). None of
these files are modified; only read and called as-is. Full rationale
is in each file's module docstring. This is the same class of coupling
flagged in the Sequence V1 audit — noted here so it isn't mistaken for
an oversight, and so a future rename/signature change to
`portfolio_backtest_engine.py` is understood to require updating these
four files, not the other way around.
