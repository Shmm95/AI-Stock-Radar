"""Resolution methodology for the 273 UNRESOLVED S&P 500 PIT membership
events in data/research/sp500_pit_membership_2015_2025_v1_reconstruction_inputs_*.json
(field: `unresolved_events`, count corrected 271->273, see that
artifact's own `correction_2026_08_16` note).

ISOLATED: does not modify, and is never imported by,
`membership_query.py` -- `get_membership_as_of`'s existing STRICT/
PERMISSIVE contract is unchanged. This module reads the SAME
`unresolved_events` list and produces a THIRD, separate, explicitly
labeled classification on top of it -- never silently redefines what
"confirmed"/"strict" already means.

WHY A THIRD TIER, NOT A REDEFINITION OF "CONFIRMED": the original
frozen artifact's own `comparison_method` field states the rule
verbatim: "Exact match on (date, ticker, action) tuple across all 3
sources = confirmed. Anything in the union but not in all 3 =
unresolved_conflicts." That is a real, already-frozen, already-tested
standard (`get_membership_as_of(mode="strict")` depends on it) --
loosening it in place would silently change strict mode's own
behavior for every existing caller. RESOLVED_EVENTS below is instead a
NEW, separate, weaker-than-strict-but-still-principled evidence tier:
"2+ independent sources, dates close enough to plausibly be the same
real-world event." Any future caller that wants this tier must ask for
it explicitly (`resolve_unresolved_events` is not wired into
`get_membership_as_of` at all).

METHODOLOGY (one rule, not four category-specific branches -- more
principled and directly reflects the real thing that matters, evidence
strength, rather than hard-coding the artifact's own category labels):
group the 273 events by (ticker, action). A group is RESOLVED (promoted,
median date used) if it has >= 2 DISTINCT supporting sources across all
its entries AND the entries' date spread is <= MAX_RESOLVABLE_DATE_SPREAD_DAYS
(source-level announcement-vs-effective-date noise is real and
typically single-digit days -- see this module's own real examples in
its test file: FOX's 2015-09-18/2015-09-21 ADD entries, 3 days apart,
are almost certainly the SAME real event, Fox's 2015 addition to the
S&P 500 following the 21st Century Fox spinoff). A group with only ONE
distinct source across all its entries (the artifact's own
`single_source_only` category, 86 events) is NEVER promoted, regardless
of date spread -- one uncorroborated claim is not strong enough
evidence for a bias-sensitive backtest input, and stays a documented
gap. A group whose date spread exceeds the threshold even with 2+
sources also stays a documented gap -- the sources may be describing
two DIFFERENT real events for the same ticker+action, not date noise
around one, and this module does not attempt to disambiguate that.

This naturally subsumes (not re-implements) all four of the artifact's
own category labels:
- `exact_two_one_silent` (31): spread=0, 2 sources -> always resolved.
- `near_date_all_sources` (122): small spread, 3 sources -> resolved
  unless spread > threshold.
- `true_conflict` (34): resolved ONLY if its actual date spread is
  small enough despite the label -- some true_conflict entries are
  really just near-date noise that didn't happen to match the other
  category's exact grouping logic (confirmed by direct inspection: see
  the FOX/CMCSK 2015-09 examples).
- `single_source_only` (86): never resolved, by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path
from statistics import median_low

MAX_RESOLVABLE_DATE_SPREAD_DAYS = 10

RECONSTRUCTION_INPUTS_PATH = Path(
    "data/research/sp500_pit_membership_2015_2025_v1_reconstruction_inputs_20260815_215706.json"
)


@dataclass(frozen=True)
class ResolvedEvent:
    date: str
    ticker: str
    action: str
    resolution_method: str
    source_dates: tuple[str, ...]
    supporting_sources: tuple[str, ...]


@dataclass(frozen=True)
class ResolutionResult:
    """`resolved` is GROUP-level (one `ResolvedEvent` per promoted
    (ticker, action) group, however many raw input entries fed into
    it -- see `ResolvedEvent.source_dates`, one entry per raw input
    event it consumed). `still_unresolved` is RAW-ENTRY-level (every
    individual input dict belonging to a non-promoted group, via
    `list.extend`, not deduplicated to one-per-group).

    REAL BUG FOUND AND FIXED (2026-08-22, independent audit): mixing
    those two units in one arithmetic expression
    (`len(resolved) + len(still_unresolved)`) does not equal ANY real,
    meaningful quantity -- not the true input count (`input_event_count`
    below), and not the true distinct-group count either. Confirmed on
    the real artifact: 273 raw inputs -> 179 distinct groups (107
    resolved + 72 still-unresolved groups), while
    `len(still_unresolved)` (86) counts RAW ENTRIES in those 72 groups,
    not the group count itself -- the old
    `len(resolved) + len(still_unresolved)` = 107 + 86 = 193 was a
    coincidence of arithmetic, not a real count of anything.
    `summary` below now reports four separately-named, individually
    correct quantities instead."""

    input_event_count: int
    resolved: tuple[ResolvedEvent, ...]
    still_unresolved: tuple[dict, ...]

    @property
    def summary(self) -> dict:
        by_method: dict[str, int] = {}
        for event in self.resolved:
            by_method[event.resolution_method] = by_method.get(event.resolution_method, 0) + 1
        promoted_input_event_count = sum(len(event.source_dates) for event in self.resolved)
        remaining_input_event_count = len(self.still_unresolved)
        return {
            "input_event_count": self.input_event_count,
            "resolved_group_count": len(self.resolved),
            "promoted_input_event_count": promoted_input_event_count,
            "remaining_input_event_count": remaining_input_event_count,
            "resolved_by_method": by_method,
        }


def _median_date(dates: list[str]) -> str:
    """Lower median on ties (even count) -- deterministic, documented
    convention, not a guess: prefers the EARLIER of the two middle
    dates, consistent with this module's own reasoning that an
    announcement date usually precedes an effective date, so the
    earlier plausible date is the more conservative "first knowable"
    choice for a point-in-time reconstruction."""
    parsed = sorted(date_type.fromisoformat(d) for d in dates)
    return median_low(parsed).isoformat()


def _date_spread_days(dates: list[str]) -> int:
    parsed = sorted(date_type.fromisoformat(d) for d in dates)
    return (parsed[-1] - parsed[0]).days


def resolve_unresolved_events(
    unresolved_events: list[dict],
    *,
    max_resolvable_date_spread_days: int = MAX_RESOLVABLE_DATE_SPREAD_DAYS,
) -> ResolutionResult:
    """See module docstring for the full methodology. Deterministic:
    same input always produces the same output (sorted grouping keys,
    `median_low` breaks ties the same way every time)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for event in unresolved_events:
        groups.setdefault((event["ticker"], event["action"]), []).append(event)

    resolved: list[ResolvedEvent] = []
    still_unresolved: list[dict] = []

    for (ticker, action), group in sorted(groups.items()):
        distinct_sources = sorted({source for event in group for source in event["supporting_sources"]})
        dates = [event["date"] for event in group]
        spread = _date_spread_days(dates)

        if len(distinct_sources) >= 2 and spread <= max_resolvable_date_spread_days:
            resolved.append(
                ResolvedEvent(
                    date=_median_date(dates),
                    ticker=ticker,
                    action=action,
                    resolution_method=(
                        f"median_date_spread_{spread}d_{len(distinct_sources)}_distinct_sources"
                    ),
                    source_dates=tuple(sorted(dates)),
                    supporting_sources=tuple(distinct_sources),
                )
            )
        else:
            still_unresolved.extend(group)

    return ResolutionResult(
        input_event_count=len(unresolved_events),
        resolved=tuple(resolved),
        still_unresolved=tuple(still_unresolved),
    )


def load_and_resolve(
    reconstruction_inputs_path: Path = RECONSTRUCTION_INPUTS_PATH,
) -> ResolutionResult:
    """Convenience entry point against the real, on-disk artifact."""
    payload = json.loads(reconstruction_inputs_path.read_text(encoding="utf-8"))
    return resolve_unresolved_events(payload["unresolved_events"])


if __name__ == "__main__":
    result = load_and_resolve()
    print(json.dumps(result.summary, indent=2))
    print()
    print(
        f"{len(result.still_unresolved)} raw input events "
        f"({len({(e['ticker'], e['action']) for e in result.still_unresolved})} distinct (ticker, action) groups) "
        f"remain a documented gap (single-source-only or wide date spread)."
    )
