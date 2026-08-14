# Stop-Conditional Forward Protocol V1

## Status

This is a preregistered, forward-only research protocol. It does not authorize
any change to the active baseline, paper trading, shadow trading, production,
or order execution.

## Locked source

- Source finding: Portfolio Replacement × Stop Interaction Audit V1
- Official source stamp: supplied at registration
- Required source decision:
  `PREREGISTERED_FORWARD_ONLY_STOP_CONDITIONAL_HYPOTHESIS`
- Last observed market date: `2026-08-02`
- First forward-only date: `2026-08-03`

Registration verifies every source result hash, the predecessor chain, the
source audit code, and the frozen snapshot manifest before writing an immutable
protocol JSON and provenance JSON.

## Primary hypothesis

On data unavailable at registration, the effect of `RSI_Q12_LOSER_ONLY`
relative to `CONTROL_NO_REPLACEMENT` is positive under fixed 5%/5% stops and is
larger than its effect under fixed 3.5%/3.5% stops.

The interaction was discovered after the global replacement hypothesis failed.
Historical data therefore cannot confirm this new hypothesis.

## Fixed research arms

| Stop stratum | Control | Candidate |
|---|---|---|
| `FIXED_BASELINE` | No replacement | `RSI_Q12_LOSER_ONLY` |
| `FIXED_MAX_RETURN` | No replacement | `RSI_Q12_LOSER_ONLY` |

All other portfolio configuration, risk, fee, slippage, signal, execution,
candidate tie-breaking, and event-ordering behavior remains unchanged.

## Fixed horizon and stopping rule

- Start, inclusive: `2026-08-03T00:00:00`
- End, exclusive: `2028-08-03T00:00:00`
- Fixed duration: 24 months
- Robustness blocks: four fixed, non-overlapping six-month blocks
- Interim evaluation: monitoring only
- Optional early success: prohibited
- Fewer than five cross-stop event pairs at the end: inconclusive

## Success gates

All of the following must pass at the fixed horizon:

1. All 24 months are complete for every registered ticker.
2. At least five executed replacement opportunities are paired across stops.
3. The 5%/5% candidate return delta versus control is strictly positive.
4. The 5%/5% return delta is greater than the 3.5%/3.5% return delta.
5. The 5%/5% return/drawdown-ratio delta is nonnegative.
6. The 5%/5% profit-factor delta is nonnegative.
7. The 5%/5% matched-benchmark excess-return delta is nonnegative.
8. The 5%/5% maximum drawdown worsens by no more than 2.5 percentage points.
9. The 5%/5% return delta remains positive after removing each fixed
   six-month block in turn.

Exact leave-one-ticker-out replays are reported as robustness evidence but are
not an optional-stopping gate.

## Replay and drift controls

A future snapshot is admissible only if it:

- was created after the preregistration timestamp;
- has a different fingerprint from the discovery snapshot;
- uses the same configuration hash and ticker universe;
- has the same frozen code hashes as the discovery snapshot;
- exactly verifies its own manifest and result files;
- reproduces the final 200 registered bars per ticker within `1e-10` for
  floating-point columns and exactly for categorical/boolean state.

Data before the cutoff is used only to verify replay stability and to preserve
causal indicator state. Performance evaluation uses only the half-open forward
interval.

## Decision labels

- `MONITORING`: the fixed horizon is incomplete.
- `INCONCLUSIVE_INSUFFICIENT_EVENTS`: the horizon is complete but the minimum
  paired-event count is not met.
- `HYPOTHESIS_NOT_SUPPORTED`: one or more preregistered gates fail.
- `HYPOTHESIS_SUPPORTED_READY_FOR_HUMAN_REVIEW`: every gate passes.

The final label never changes execution authority automatically. A successful
result requires a separate human decision and a separately approved next stage.

