(function () {
  "use strict";

  // READ-ONLY: this file makes exactly one kind of network call, GET
  // /api/v1/snapshot, on a timer. There is no form, no button that
  // posts anywhere, and no code path that sends anything other than a
  // GET here -- see tests/test_dashboard_read_only_ast.py's own note
  // that this file's own absence of mutation calls is part of what is
  // verified for the dashboard as a whole.

  var POLL_INTERVAL_MS = 30000;
  var STALE_AFTER_MINUTES = 10;

  function el(id) { return document.getElementById(id); }

  function fmtMoney(str) {
    if (str === null || str === undefined) return "—";
    var n = Number(str);
    if (Number.isNaN(n)) return "—";
    var sign = n > 0 ? "+" : "";
    return sign + n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtUsd(str) {
    if (str === null || str === undefined) return "—";
    var n = Number(str);
    if (Number.isNaN(n)) return "—";
    return "$" + n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function pnlClass(str) {
    var n = Number(str);
    if (Number.isNaN(n) || n === 0) return "";
    return n > 0 ? "pos" : "neg";
  }

  function stat(label, value, extraClass) {
    var wrap = document.createElement("div");
    wrap.className = "stat";
    var l = document.createElement("div");
    l.className = "label";
    l.textContent = label;
    var v = document.createElement("div");
    v.className = "value" + (extraClass ? " " + extraClass : "");
    v.textContent = value;
    wrap.appendChild(l);
    wrap.appendChild(v);
    return wrap;
  }

  function renderAccount(account) {
    var grid = el("account-grid");
    grid.innerHTML = "";
    if (!account) {
      grid.innerHTML = '<div class="empty-state">Account data unavailable.</div>';
      return;
    }
    grid.appendChild(stat("Portfolio value", fmtUsd(account.portfolio_value)));
    grid.appendChild(stat("Cash", fmtUsd(account.cash)));
    grid.appendChild(stat("Buying power", fmtUsd(account.buying_power)));
    grid.appendChild(stat("Day P&L", fmtMoney(account.day_pnl), pnlClass(account.day_pnl)));
  }

  function renderPnl(pnl) {
    var grid = el("pnl-grid");
    grid.innerHTML = "";
    if (!pnl) {
      grid.innerHTML = '<div class="empty-state">P&amp;L data unavailable.</div>';
      return;
    }
    grid.appendChild(stat("Realized", fmtMoney(pnl.realized_closed_trade_pnl_usd), pnlClass(pnl.realized_closed_trade_pnl_usd)));
    grid.appendChild(stat("Unrealized", fmtMoney(pnl.unrealized_pnl_usd), pnlClass(pnl.unrealized_pnl_usd)));
    var combinedLabel = pnl.combined_strategy_pnl_label || "Combined";
    grid.appendChild(stat(combinedLabel, fmtMoney(pnl.combined_strategy_pnl_usd), pnlClass(pnl.combined_strategy_pnl_usd)));
    if (!pnl.realized_history_complete) {
      var note = document.createElement("div");
      note.className = "sub";
      note.style.gridColumn = "1 / -1";
      note.textContent = "Realized P&L may be incomplete (order history capped).";
      grid.appendChild(note);
    }
  }

  function renderPositions(positions) {
    var list = el("positions-list");
    list.innerHTML = "";
    if (!positions || positions.length === 0) {
      list.innerHTML = '<div class="empty-state">No open positions.</div>';
      return;
    }
    positions.forEach(function (p) {
      var row = document.createElement("div");
      row.className = "position-row";
      var main = document.createElement("div");
      main.className = "position-main";
      var ticker = document.createElement("span");
      ticker.className = "ticker";
      ticker.textContent = p.symbol + " (" + p.side + ")";
      var sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = "qty " + p.quantity + " @ " + fmtUsd(p.avg_entry_price) + " → " + fmtUsd(p.current_price);
      main.appendChild(ticker);
      main.appendChild(sub);

      var pnlWrap = document.createElement("div");
      pnlWrap.className = "position-main";
      pnlWrap.style.textAlign = "right";
      var pnlEl = document.createElement("span");
      pnlEl.className = "ticker " + pnlClass(p.unrealized_pnl_usd);
      pnlEl.textContent = fmtMoney(p.unrealized_pnl_usd);
      var pctEl = document.createElement("span");
      pctEl.className = "sub";
      var pct = Number(p.unrealized_pnl_percent);
      pctEl.textContent = Number.isNaN(pct) ? "" : (pct * 100).toFixed(2) + "%";
      pnlWrap.appendChild(pnlEl);
      pnlWrap.appendChild(pctEl);

      row.appendChild(main);
      row.appendChild(pnlWrap);
      list.appendChild(row);
    });
  }

  function renderTrades(trades) {
    var list = el("trades-list");
    list.innerHTML = "";
    if (!trades || trades.length === 0) {
      list.innerHTML = '<div class="empty-state">No recent closed trades.</div>';
      return;
    }
    trades.slice(0, 10).forEach(function (t) {
      var row = document.createElement("div");
      row.className = "trade-row";
      var main = document.createElement("div");
      main.className = "trade-main";
      var ticker = document.createElement("span");
      ticker.className = "ticker";
      ticker.textContent = t.symbol;
      var sub = document.createElement("span");
      sub.className = "sub";
      sub.textContent = fmtUsd(t.entry_price) + " → " + fmtUsd(t.exit_price) + " · qty " + t.quantity;
      main.appendChild(ticker);
      main.appendChild(sub);

      var pnlEl = document.createElement("span");
      pnlEl.className = pnlClass(t.pnl_usd);
      pnlEl.textContent = fmtMoney(t.pnl_usd);

      row.appendChild(main);
      row.appendChild(pnlEl);
      list.appendChild(row);
    });
  }

  function systemRow(key, value) {
    var row = document.createElement("div");
    row.className = "system-row";
    var k = document.createElement("span");
    k.className = "k";
    k.textContent = key;
    var v = document.createElement("span");
    v.textContent = value;
    row.appendChild(k);
    row.appendChild(v);
    return row;
  }

  function renderSystem(system) {
    var grid = el("system-grid");
    grid.innerHTML = "";
    if (!system) {
      grid.innerHTML = '<div class="empty-state">System status unavailable.</div>';
      return;
    }

    grid.appendChild(systemRow("STOP flag", system.stop_present ? "SET" : "clear"));
    grid.appendChild(systemRow("FREEZE flag", system.freeze_present ? "SET" : "clear"));

    // Deliberately two SEPARATE rows -- a present replay gate is not
    // proof of a fresh cursor, and vice versa. Never merge these.
    grid.appendChild(systemRow("Replay gate today", system.replay_gate_present_today ? "PRESENT" : "MISSING"));
    var sessionLabel = system.last_processed_equity_session_date || "unknown";
    var behind = system.settled_sessions_behind;
    var behindText = behind === null || behind === undefined
      ? "(unknown)"
      : behind === 0 ? "(up to date)" : "(" + behind + " settled session" + (behind === 1 ? "" : "s") + " behind)";
    grid.appendChild(systemRow("Last processed equity session", sessionLabel + " " + behindText));

    if (system.broker_account_status) {
      grid.appendChild(systemRow("Broker account status", system.broker_account_status));
    }
    if (system.broker_restrictions && system.broker_restrictions.length > 0) {
      grid.appendChild(systemRow("Broker restrictions", system.broker_restrictions.join(", ")));
    }
    grid.appendChild(systemRow("Local vs broker positions", system.local_position_count + " / " + system.broker_position_count));
    if (system.local_broker_symbol_diff && system.local_broker_symbol_diff.length > 0) {
      grid.appendChild(systemRow("Symbol mismatch", system.local_broker_symbol_diff.join(", ")));
    }
    if (system.last_decision_generated_at) {
      var staleTxt = system.last_decision_stale ? " (STALE)" : "";
      grid.appendChild(systemRow("Last decision run", system.last_decision_generated_at + staleTxt));
    }
    if (system.needs_manual_review_count !== null && system.needs_manual_review_count !== undefined) {
      grid.appendChild(systemRow("Needs manual review", String(system.needs_manual_review_count)));
    }

    if (system.intent_summary) {
      var s = system.intent_summary;
      grid.appendChild(systemRow("Order-intent status", s.summary_status));
      if (s.non_terminal_count > 0) {
        grid.appendChild(systemRow("In-flight intents", String(s.non_terminal_count)));
      }
      if (s.stale_non_terminal_count > 0) {
        grid.appendChild(systemRow("Stale in-flight intents", String(s.stale_non_terminal_count)));
      }
    }
  }

  function renderErrors(errors) {
    var card = el("errors-card");
    var list = el("errors-list");
    list.innerHTML = "";
    if (!errors || errors.length === 0) {
      card.hidden = true;
      return;
    }
    card.hidden = false;
    errors.forEach(function (e) {
      var row = document.createElement("div");
      row.className = "error-row";
      row.textContent = e.section + ": " + e.error_type;
      list.appendChild(row);
    });
  }

  function renderStaleness(snapshot) {
    var badge = el("staleness");
    var generatedAt = snapshot.generated_at_utc;
    var isStale = snapshot._dashboard_stale === true;
    if (!generatedAt) {
      badge.className = "badge badge-unknown";
      badge.textContent = "UNKNOWN";
      return;
    }
    var ageMinutes = (Date.now() - new Date(generatedAt).getTime()) / 60000;
    isStale = isStale || ageMinutes > STALE_AFTER_MINUTES;
    if (isStale) {
      badge.className = "badge badge-error";
      badge.textContent = "STALE";
    } else if (snapshot.collection_status === "PARTIAL") {
      badge.className = "badge badge-warn";
      badge.textContent = "PARTIAL";
    } else if (snapshot.collection_status === "OK") {
      badge.className = "badge badge-ok";
      badge.textContent = "LIVE";
    } else {
      badge.className = "badge badge-error";
      badge.textContent = snapshot.collection_status || "ERROR";
    }
  }

  function render(snapshot) {
    renderStaleness(snapshot);
    renderAccount(snapshot.account);
    renderPnl(snapshot.pnl);
    renderPositions(snapshot.positions);
    renderSystem(snapshot.system);
    renderTrades(snapshot.recent_trades);
    renderErrors(snapshot.errors);
    el("generated-at").textContent = snapshot.generated_at_utc
      ? "Snapshot generated " + snapshot.generated_at_utc
      : "";
  }

  function renderFetchFailure() {
    var badge = el("staleness");
    badge.className = "badge badge-error";
    badge.textContent = "UNAVAILABLE";
  }

  function poll() {
    fetch("/api/v1/snapshot", { cache: "no-store" })
      .then(function (response) {
        if (!response.ok) { throw new Error("snapshot fetch failed: " + response.status); }
        return response.json();
      })
      .then(render)
      .catch(renderFetchFailure);
  }

  poll();
  setInterval(poll, POLL_INTERVAL_MS);
})();
