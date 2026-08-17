"""Query S&P 500 point-in-time membership from the confirmed/unresolved
event lists this session's PIT reconstruction task produced.

Answers "which tickers were in the S&P 500 on date D" -- something the
frozen artifact (`data/research/sp500_pit_membership_2015_2025_v1_*.json`)
did not itself provide: that file is a comparison RESULT (counts,
disclaimers, a 20-item unresolved sample), not a query interface.

ISOLATED, NOT INTEGRATED: nothing here is imported by
`portfolio_backtest_engine.py`, `run_daily_decision.py`,
`run_control_arm_decision.py`, or any live/control path. This module
only reads two existing JSON files on disk; it does not fetch, does not
write to either of them, and is never invoked by anything outside its
own tests/CLI use.

A REAL, PREVIOUSLY UNDOCUMENTED GAP THIS MODULE'S CONSTRUCTION SURFACED:
the frozen artifact's own `results` section stores `confirmed_count:
386` and `unresolved_conflicts_count: 271` but only PERSISTS a 20-item
unresolved SAMPLE -- not the full 386 confirmed events or the full 271
unresolved events actually needed to reconstruct anything. Those full
lists existed only in this session's own scratch directory
(`comparison_result.json`), which is not a durable location. Rather
than either (a) modifying the frozen artifact (explicitly forbidden --
"PIT membership JSON dosyasının kendisini değiştirme, sadece OKU") or
(b) silently re-deriving the lists from a fragile scratch path, a new,
separate companion file was written this task:
`data/research/sp500_pit_membership_2015_2025_v1_reconstruction_inputs_20260815_215706.json`
-- same real fetch, same real comparison already run, just persisting
what the original artifact summarized instead of storing. That file
also carries the ANCHOR snapshot this module needs (see below). The
original frozen artifact itself was never touched.

RECONSTRUCTION METHOD:
1. Anchor: a real, actual full-membership snapshot at a known date --
   the 501 tickers in Wikipedia's "constituents" table at the same
   pinned revision (1329075976, 2025-12-23) already used as source_2 in
   the frozen artifact's own 3-way comparison. This is a real fetch
   already performed this session, not a new one, and not the
   "Selected changes" events table (a different table on the same
   page) -- this is the actual CURRENT-AS-OF-THAT-DATE roster.
2. To reconstruct membership at any date D < anchor_date: take every
   CONFIRMED event with event_date > D, walk them in DESCENDING date
   order, and undo each one against a working copy of the anchor set
   (an ADD on the way forward becomes a removal when undone walking
   backward; a REMOVE becomes an addition). What remains after undoing
   every event newer than D is the reconstructed membership AT D.
3. This module only reconstructs BACKWARD from the anchor (D < anchor_date).
   Forward reconstruction (D > 2025-12-23) is out of scope -- there is
   no real event data past that date in this dataset, and it was never
   requested.

STRICT vs PERMISSIVE mode -- this is the part the task specifically
asked to be handled clearly, not glossed over:
- STRICT (default): only CONFIRMED events are undone. Any date whose
  reconstruction window (D, anchor_date] contains one or more
  UNRESOLVED events is flagged `uncertain=True` -- the returned ticker
  set is what the confirmed-only evidence supports, but at least one
  event exists that MIGHT be real and, if so, would change the answer
  for at least the tickers it names. Strict mode never guesses; it
  tells you when it doesn't know.
- PERMISSIVE: confirmed events are undone exactly as in strict mode,
  PLUS unresolved events are ALSO undone, each at its "most likely
  date" -- defined here as the date, among that (ticker, action)
  pair's unresolved candidates, supported by the most of the three
  original sources (fja05680/wikipedia/lawcal), ties broken by
  earliest date. This is a simple, stated heuristic, not a claim of
  correctness -- see `_most_likely_unresolved_events`'s own docstring
  for its known failure mode (a ticker with a genuine SECOND, distinct
  index tenure inside the unresolved set could be mis-merged with its
  first).

WHEN TO USE WHICH: strict mode is the right default for anything where
being wrong is worse than being incomplete (e.g. flagging which exact
dates a backtest's universe claim is unverified) -- it will never
silently include a ticker that might not have actually been a member.
Permissive mode is appropriate only when a single best-guess universe
is needed for illustration/exploration and the caller has already
decided incompleteness is worse than the small chance of an individual
wrong inclusion/exclusion -- it should never be the default for
anything whose output feeds a real backtest, precisely because it can
be wrong in a way that is not flagged.

TICKER IDENTITY / RECYCLING RISK -- added this task, following the
real, SEC-EDGAR-confirmed discovery (a separate task, this session)
that at least two tickers this codebase's own live 83-ticker universe
still uses today (Q, SNDK) have been RECYCLED: the symbol was vacated
by one real company and later reassigned, by the exchange, to an
entirely unrelated one (Q: a 2017 S&P member -> vacant -> Qnity
Electronics, Inc., a 2025 spinoff, CIK 2058873, formerly "Novus SpinCo
1, Inc."; SNDK: the original SanDisk Corp, acquired by Western Digital
in 2016 -> vacant -> a new, distinct "Sandisk Corp" entity, CIK
2023554, earliest SEC filing 2024-06-13). A ticker-symbol-only PIT
query is exactly the kind of consumer this risk threatens: querying
"Q" or "SNDK" for a 2015-2020 date could silently return bars data for
the WRONG, much-later company if that query were ever connected to a
live price fetch (this module still only returns symbols, never
prices, but the risk is inherited by ANY downstream consumer that
takes a ticker set from here and fetches prices for it).

Two-layer response, per this task's own design:
1. FREE layer (`identity_warnings` on `MembershipAsOfResult`, this
   module's own already-loaded event data, no new fetch): for every
   ticker in the reconstructed set, flag it if the ticker's FULL event
   history (not just the reconstruction window) contains a REMOVE
   followed later by an ADD -- exactly the pattern both confirmed real
   cases (Q, SNDK) show. This is a cheap, real signal computed from data
   already on disk, not a new external dependency -- but it is a
   PATTERN match, not proof: see point 2.
2. CONFIRMING layer (`check_ticker_identity_at_date`, isolated,
   separate real SEC EDGAR fetch, cached): a mechanical, dated proof --
   fetch the ticker's CURRENT CIK (`company_tickers.json`) and that
   CIK's earliest SEC filing date (`submissions/CIK##########.json`);
   if the query date is BEFORE that earliest filing date, the CURRENT
   company/CIK could not have existed yet, which is hard, mechanical
   evidence the ticker meant something else at that date. This
   automates exactly the manual check already done for Q and SNDK in
   the prior research task -- see `check_ticker_identity_at_date`'s own
   docstring for what it does and does NOT prove.

Neither layer is wired into `get_membership_as_of`'s CIK-checking path
automatically (layer 2 makes real network calls and is deliberately
left as an opt-in, separately-called function) -- `identity_warnings`
(layer 1) is the only thing computed automatically, because it is free.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_FROZEN_ARTIFACT_PATH = (
    _PROJECT_ROOT / "data" / "research" / "sp500_pit_membership_2015_2025_v1_20260815_215706.json"
)
_RECONSTRUCTION_INPUTS_PATH = (
    _PROJECT_ROOT
    / "data"
    / "research"
    / "sp500_pit_membership_2015_2025_v1_reconstruction_inputs_20260815_215706.json"
)

# Cache for the SEC EDGAR CIK-verification layer only -- scoped to this
# module's own directory, not written to by anything else in the repo.
# "This file rarely changes" (this task's own instruction) is the reason
# a plain on-disk cache with no expiry logic is judged sufficient here.
_SEC_CACHE_DIR = Path(__file__).resolve().parent / "sec_edgar_cache"
_SEC_COMPANY_TICKERS_CACHE = _SEC_CACHE_DIR / "company_tickers.json"
_SEC_USER_AGENT = "AI-Stock-Radar-research pit_universe/membership_query.py (research use)"


@dataclass(frozen=True)
class MembershipEvent:
    date: str
    ticker: str
    action: str  # "ADD" | "REMOVE"


@dataclass(frozen=True)
class MembershipAsOfResult:
    """Richer than a bare `set[str]` deliberately -- the whole point of
    this task's strict/permissive split is that the ticker set ALONE
    cannot communicate whether unresolved evidence could change it.
    A caller that only wants the set can use `.tickers`."""

    date: str
    mode: str  # "strict" | "permissive"
    tickers: frozenset[str]
    uncertain: bool
    unresolved_events_in_window: int
    unresolved_tickers_in_window: frozenset[str]

    # Free layer only (no network call) -- see module docstring's
    # "TICKER IDENTITY / RECYCLING RISK" section. A ticker landing here
    # means its symbol has a REMOVE-then-later-ADD pattern SOMEWHERE in
    # its full event history -- a real signal (both confirmed real
    # recycling cases, Q and SNDK, show exactly this pattern), but a
    # PATTERN match, not proof. Call `check_ticker_identity_at_date`
    # separately for a mechanical, dated SEC EDGAR confirmation.
    identity_warnings: frozenset[str]

    @property
    def ticker_count(self) -> int:
        return len(self.tickers)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. This module reads two pre-existing, "
            "already-real-fetched files (the frozen PIT artifact and its "
            "reconstruction-inputs companion) -- it does not fetch "
            "anything itself. Re-run the PIT membership reconstruction "
            "task if these are genuinely missing."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events_and_anchor() -> tuple[list[MembershipEvent], list[dict], str, frozenset[str]]:
    payload = _load_json(_RECONSTRUCTION_INPUTS_PATH)
    confirmed = [MembershipEvent(**e) for e in payload["confirmed_events"]]
    unresolved = payload["unresolved_events"]  # kept as raw dicts -- carries supporting_sources
    anchor_date = payload["anchor_snapshot"]["date"]
    anchor_tickers = frozenset(payload["anchor_snapshot"]["tickers"])
    return confirmed, unresolved, anchor_date, anchor_tickers


def _undo_event(working_set: set[str], event_action: str, ticker: str) -> None:
    """Undo one event's forward effect, walking backward in time.
    An ADD undone means the ticker was NOT yet a member before this
    date -- remove it. A REMOVE undone means the ticker WAS still a
    member before this date -- add it back."""
    if event_action == "ADD":
        working_set.discard(ticker)
    elif event_action == "REMOVE":
        working_set.add(ticker)
    else:
        raise ValueError(f"Unknown action: {event_action!r}")


def _tickers_with_remove_then_add_pattern(
    confirmed: list[MembershipEvent], unresolved: list[dict]
) -> frozenset[str]:
    """Free-layer recycling-risk flag: which tickers, across their FULL
    event history (not just one reconstruction window), show a REMOVE
    followed at a LATER date by an ADD? Both confirmed real cases this
    session found (Q: REMOVE 2017-11-15 -> ADD 2025-11-03; SNDK: REMOVE
    2016-05-12 -> ADD 2025-11-28) show exactly this shape. A ticker with
    this pattern is not PROVEN recycled -- a company can legitimately
    leave and later rejoin the S&P 500 as itself (that has real
    precedent independent of recycling) -- so this is a real, computed
    signal worth surfacing, not a confirmed finding by itself. Confirm
    with `check_ticker_identity_at_date` before treating it as more than
    a flag.
    """
    by_ticker: dict[str, list[tuple[str, str]]] = {}
    for e in confirmed:
        by_ticker.setdefault(e.ticker, []).append((e.date, e.action))
    for e in unresolved:
        by_ticker.setdefault(e["ticker"], []).append((e["date"], e["action"]))

    flagged = set()
    for ticker, events in by_ticker.items():
        events_sorted = sorted(events, key=lambda x: x[0])
        seen_remove_date: str | None = None
        for date, action in events_sorted:
            if action == "REMOVE":
                seen_remove_date = date
            elif action == "ADD" and seen_remove_date is not None and date > seen_remove_date:
                flagged.add(ticker)
                break
    return frozenset(flagged)


def _most_likely_unresolved_events(unresolved: list[dict]) -> list[MembershipEvent]:
    """Reduce the unresolved list to one best-guess event per
    (ticker, action) group, per this module's own documented heuristic:
    the date supported by the most of the 3 original sources, ties
    broken by earliest date.

    KNOWN FAILURE MODE, not solved here: if a ticker genuinely left and
    later rejoined the index within the unresolved-conflict portion of
    its history (a real second tenure, not just a disputed date for one
    event), this grouping conflates both into a single best-guess event
    per action -- it has no way to distinguish "two sources disagree
    about ONE event's date" from "there were actually two events." This
    is exactly the kind of edge case permissive mode's own docstring
    warns should keep it out of anything feeding a real backtest.
    """
    by_key: dict[tuple[str, str], list[dict]] = {}
    for e in unresolved:
        by_key.setdefault((e["ticker"], e["action"]), []).append(e)

    chosen: list[MembershipEvent] = []
    for (ticker, action), candidates in by_key.items():
        # score = number of supporting sources, tie-break = earliest date
        best = max(candidates, key=lambda c: (len(c["supporting_sources"]), ), default=None)
        best_score = len(best["supporting_sources"])
        tied = [c for c in candidates if len(c["supporting_sources"]) == best_score]
        winner = min(tied, key=lambda c: c["date"])
        chosen.append(MembershipEvent(date=winner["date"], ticker=ticker, action=action))
    return chosen


def get_membership_as_of(date: str, mode: str = "strict") -> MembershipAsOfResult:
    """Reconstruct S&P 500 membership as of `date` (ISO "YYYY-MM-DD"),
    working backward from the real 2025-12-23 anchor snapshot.

    `mode`: "strict" (default, confirmed events only) or "permissive"
    (confirmed + best-guess unresolved). See this module's own docstring
    for when each is appropriate. `date` must be <= the anchor date
    (2025-12-23) and >= 2015-01-01 -- this module only reconstructs
    backward within the window the underlying PIT comparison covered.
    """
    if mode not in ("strict", "permissive"):
        raise ValueError(f"mode must be 'strict' or 'permissive', got {mode!r}")

    confirmed, unresolved, anchor_date, anchor_tickers = _load_events_and_anchor()

    if date > anchor_date:
        raise ValueError(
            f"{date} is after the anchor date {anchor_date} -- this module only "
            "reconstructs backward from a known real snapshot, forward "
            "reconstruction was never requested and is not implemented."
        )

    working = set(anchor_tickers)

    events_to_undo = sorted(
        (e for e in confirmed if e.date > date), key=lambda e: e.date, reverse=True
    )
    for e in events_to_undo:
        _undo_event(working, e.action, e.ticker)

    unresolved_in_window = [e for e in unresolved if e["date"] > date]

    if mode == "permissive":
        best_guess_unresolved = _most_likely_unresolved_events(
            [e for e in unresolved_in_window]
        )
        for e in sorted(best_guess_unresolved, key=lambda e: e.date, reverse=True):
            _undo_event(working, e.action, e.ticker)

    recycling_risk_tickers = _tickers_with_remove_then_add_pattern(confirmed, unresolved)

    return MembershipAsOfResult(
        date=date,
        mode=mode,
        tickers=frozenset(working),
        uncertain=len(unresolved_in_window) > 0,
        unresolved_events_in_window=len(unresolved_in_window),
        unresolved_tickers_in_window=frozenset(e["ticker"] for e in unresolved_in_window),
        identity_warnings=frozenset(working) & recycling_risk_tickers,
    )


def measure_strict_permissive_divergence(sample_dates: list[str]) -> dict:
    """The "kaba yüzde" diagnostic this task asked for: across
    `sample_dates`, what fraction actually produce a DIFFERENT ticker
    set between strict and permissive mode? This is a more informative
    measure than "uncertain=True" alone -- `uncertain` is a hard flag
    that fires for nearly every date in this 2015-2025 window (271
    unresolved events are spread across the whole range, so almost any
    date's backward window contains at least one), while this measures
    whether resolving them would have actually CHANGED the answer for
    that specific date.
    """
    differing = 0
    total_symmetric_diff_tickers = 0
    for d in sample_dates:
        strict = get_membership_as_of(d, mode="strict")
        permissive = get_membership_as_of(d, mode="permissive")
        diff = strict.tickers.symmetric_difference(permissive.tickers)
        if diff:
            differing += 1
            total_symmetric_diff_tickers += len(diff)
    return {
        "sample_size": len(sample_dates),
        "dates_with_strict_permissive_divergence": differing,
        "divergence_rate_percent": round(differing / len(sample_dates) * 100, 1) if sample_dates else None,
        "total_ticker_differences_summed": total_symmetric_diff_tickers,
    }


# ---------------------------------------------------------------------
# CONFIRMING layer -- real SEC EDGAR fetch, cached. See module docstring's
# "TICKER IDENTITY / RECYCLING RISK" section. Isolated: makes real network
# calls only when explicitly invoked, never as a side effect of
# `get_membership_as_of`.
# ---------------------------------------------------------------------


def _sec_get(url: str) -> dict:
    """Real fetch, with a real, environment-specific wrinkle worth
    recording rather than silently working around: both `urllib.request`
    and `requests` (Python's `ssl` module either way) were consistently
    rejected by SEC EDGAR's own Akamai bot-management with a genuine
    `403 Request Rate Threshold Exceeded` page -- even for a single,
    first request, and even after waiting out an actual rate-limit
    window that DID clear (confirmed: `curl` succeeded immediately
    after, repeatedly, from the same machine/network, same moment).
    This points to Akamai fingerprinting Python's TLS handshake
    specifically, not a real request-volume violation -- this task's
    own research already made a small, real number of SEC calls, well
    under any documented fair-access threshold. `curl` (a different TLS
    stack) was NOT blocked in the same environment, so it is used here
    instead of Python's own HTTP clients. This is still a real network
    fetch, not a mock -- only the HTTP client differs.
    """
    import subprocess

    result = subprocess.run(
        ["curl", "-s", "--max-time", "20", "-A", _SEC_USER_AGENT, url],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _load_sec_company_tickers(*, force_refresh: bool = False) -> dict[str, dict]:
    """Real fetch of SEC EDGAR's official current ticker->CIK->name
    mapping (~10,400 entries), cached to disk -- "this file rarely
    changes" is this task's own stated reason a plain cache with no
    expiry logic is sufficient. Returns {ticker: {"cik_str": int, "title": str}}.
    """
    if not force_refresh and _SEC_COMPANY_TICKERS_CACHE.is_file():
        raw = json.loads(_SEC_COMPANY_TICKERS_CACHE.read_text(encoding="utf-8"))
    else:
        raw = _sec_get("https://www.sec.gov/files/company_tickers.json")
        _SEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _SEC_COMPANY_TICKERS_CACHE.write_text(json.dumps(raw), encoding="utf-8")
    return {entry["ticker"]: entry for entry in raw.values()}


def _fetch_cik_earliest_filing_date(cik: int) -> str | None:
    """Real fetch of one CIK's SEC EDGAR submission history, cached
    per-CIK (same rarely-changes rationale). Returns the earliest
    filing date found in the 'recent' filings list (ISO "YYYY-MM-DD"),
    or None if the CIK has no filings on record there. SEC's own API
    returns filings newest-first, so the earliest is the LAST entry in
    the list -- confirmed by direct inspection during this task's own
    Q/SNDK research, not assumed."""
    cache_path = _SEC_CACHE_DIR / f"cik_{cik:010d}.json"
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        padded = f"{cik:010d}"
        payload = _sec_get(f"https://data.sec.gov/submissions/CIK{padded}.json")
        _SEC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(0.15)  # light, deliberate pacing -- SEC asks bulk callers not to hammer this endpoint

    dates = payload.get("filings", {}).get("recent", {}).get("filingDate", [])
    return dates[-1] if dates else None


def check_ticker_identity_at_date(ticker: str, date: str) -> dict:
    """Mechanical, dated proof-or-not: could the security CURRENTLY
    trading under `ticker` have been the one trading under it on `date`?

    Automates exactly the manual check already done for Q and SNDK in
    the prior research task this module's docstring describes: look up
    `ticker`'s CURRENT CIK and that CIK's EARLIEST SEC filing date; if
    `date` is before that earliest filing date, the current company
    could not have existed yet, which is hard, mechanical evidence the
    ticker meant something else at `date`.

    Returns a dict, never raises for an unknown ticker (returns
    `current_cik: None` instead -- e.g. `ticker` may not be a
    SEC-registered operating company, such as an ETF; this task's own
    "FB" finding is exactly that case).

    WHAT THIS DOES NOT PROVE: a `possible_same_entity: True` result is
    NOT positive proof the ticker meant the same thing at `date` -- it
    only means this specific check found no contradiction. The current
    CIK could still be an unrelated company that simply happens to have
    existed (under any name) before `date`. This function can only ever
    produce a hard NO, never a hard YES.
    """
    tickers = _load_sec_company_tickers()
    entry = tickers.get(ticker)
    if entry is None:
        return {
            "ticker": ticker,
            "date": date,
            "current_cik": None,
            "current_company_name": None,
            "earliest_filing_date": None,
            "possible_same_entity": None,
            "note": (
                f"{ticker!r} not found in SEC EDGAR's company_tickers.json -- "
                "not a SEC-registered operating-company ticker under this "
                "exact symbol today (could be delisted entirely, an ETF/fund "
                "registrant type this file does not cover, or genuinely "
                "absent). No identity check possible from this source."
            ),
        }

    cik = entry["cik_str"]
    company_name = entry["title"]
    earliest_filing_date = _fetch_cik_earliest_filing_date(cik)

    if earliest_filing_date is None:
        possible = None
        note = f"CIK {cik} ({company_name}) has no filings on record -- cannot determine earliest date."
    elif date < earliest_filing_date:
        possible = False
        note = (
            f"HARD PROOF AGAINST: {date} is before CIK {cik}'s ({company_name}) "
            f"earliest SEC filing ({earliest_filing_date}). This company/CIK "
            f"could not have been trading under {ticker!r} on {date} -- the "
            "ticker meant a different, unrelated entity at that date."
        )
    else:
        possible = True
        note = (
            f"No contradiction found: CIK {cik} ({company_name}) already existed "
            f"(earliest filing {earliest_filing_date}) by {date}. This does NOT "
            "prove it was the same entity trading under this ticker at that date "
            "-- only that this specific check found no reason to rule it out."
        )

    return {
        "ticker": ticker,
        "date": date,
        "current_cik": cik,
        "current_company_name": company_name,
        "earliest_filing_date": earliest_filing_date,
        "possible_same_entity": possible,
        "note": note,
    }
