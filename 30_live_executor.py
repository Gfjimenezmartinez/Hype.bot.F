"""
================================================================
Script 30 — Live Order Execution (Hyperliquid)
================================================================
Turns Script 17's trade plans into real orders on Hyperliquid. This is
the only script in the suite that can place a live order — everything
else (including Script 29's paper-trade log) stays simulation-only.

Default mode is DRY RUN: it fetches your real account equity and open
positions, builds today's plans against that real equity, prints
exactly what it would send, and logs it — but sends nothing. Pass
--live to actually place orders.

Per run, for each actionable (LONG/SHORT) plan on a ticker that isn't
already an open position (or already has a resting entry) on Hyperliquid:
  1. Set leverage, then submit a GTC limit entry + reduce-only SL1 stop
     as ONE ATOMIC bracket order (Hyperliquid's "normalTpsl" grouping,
     see hyperliquid_broker.bracket_entry) at Script 17's planned entry
     +/- ENTRY_PRICE_TOLERANCE (0.5%). Rests patiently until filled or
     cancelled -- never chases price -- and because the SL is submitted
     atomically with the entry, it's tied to the position the instant it
     fills, even between --loop cycles while nothing is polling. Logged
     as PENDING while resting.
  2. On a LATER cycle, reconcile_pending_entries() checks every PENDING
     row: filled -> attach two reduce-only take-profits (half size each)
     at TP1/TP2 and mark PLACED (SL is already safe either way); still
     resting -> leave as PENDING; neither -> mark EXPIRED (cancelled/
     expired without filling, stop tracking). SL2 is never placed as a
     live order -- SL1 is the real protective stop; SL2 in the plan is
     an informational Monte-Carlo disaster bound, not a second order.

If a filled bracket's TP1/TP2 fail to attach after retries, the run
logs TP_FAILED — the position IS protected (SL was atomic with entry),
it just has no live profit target yet.

--loop runs this forever instead of once, sleeping --interval minutes
between checks. Note the underlying signals are daily-bar-based and
data_loader's parquet cache only refreshes every 12h, so polling much
faster than that just re-checks the same cached plan — the per-ticker
"already an open position" skip makes that safe (no duplicate orders),
just not more responsive than the data actually is.

Usage:
    python 30_live_executor.py                  # dry run, full universe, once
    python 30_live_executor.py --tickers BTC     # dry run, one ticker
    python 30_live_executor.py --live            # PLACES REAL ORDERS, once
    python 30_live_executor.py --live --loop     # PLACES REAL ORDERS, forever (default: every 15 min)
    python 30_live_executor.py --live --loop --interval 30   # every 30 min
================================================================
"""

import argparse
import os
import time
import traceback
from datetime import datetime
from importlib import import_module as _im
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, load_all_funding

_r17 = _im("17_trade_planner")
generate_plan = _r17.generate_plan
apply_portfolio_risk_adjustments = _r17.apply_portfolio_risk_adjustments

ENTRY_PRICE_TOLERANCE = 0.005   # 0.5% -- Script 17's planned entry is the prior
                                  # bar's close, which is essentially always stale
                                  # by the time this actually executes. An IOC limit
                                  # at the EXACT planned price only fills if live
                                  # price happens to sit at that precise level right
                                  # now, which is rare -- allow up to this much worse
                                  # than planned instead of an exact-or-better match,
                                  # so real signals can actually fill without chasing
                                  # an unbounded/arbitrary price.

HYPERLIQUID_MIN_NOTIONAL = 10.0   # Hyperliquid rejects any order worth less
                                    # than this ("Order must have minimum value
                                    # of $10") -- checked proactively so an
                                    # undersized position is skipped cleanly
                                    # instead of submitted and rejected.

MAX_CONCURRENT_NEW_ENTRIES = 3    # when capital is too thin to size every
                                    # actionable signal above HYPERLIQUID_MIN_
                                    # NOTIONAL, concentrate into this many by
                                    # conviction instead of diluting across all
                                    # of them until most get rejected. Trigger is
                                    # self-scaling (equity / trade count vs a
                                    # safety multiple of the minimum), not a
                                    # fixed dollar cutoff -- a hardcoded threshold
                                    # like "$50" would miss real balances that
                                    # sit just above it (e.g. $50.21).
LOW_CAPITAL_SAFETY_MULTIPLE = 2.0  # trigger when equity/len(trades) would land
                                    # under this many times HYPERLIQUID_MIN_NOTIONAL

MIN_EQUITY_FLOOR = 50.0   # drawdown kill-switch: if combined equity is at or
                            # below this, no NEW entries are placed this run
                            # (or any run, until equity recovers above it).
                            # Does NOT touch existing positions -- each one
                            # already has its own live SL/TP managing it, and
                            # force-closing everything mid-drawdown could
                            # realize losses at exactly the wrong moment.
                            # Reconciliation (pending-bracket fills, closed-
                            # trade P&L) still runs either way.

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SUITE_DIR, "live_orders_log.csv")
LOG_COLS = ["timestamp", "mode", "ticker", "coin", "direction", "leverage",
            "entry_price", "size", "sl1", "tp1", "tp2", "tp1_sz", "tp2_sz",
            "status", "detail",
            "closed", "exit_price", "realized_pnl", "exit_time",
            "regime", "conviction", "fc_source", "fc_total",
            "ml_signal", "ml_confidence", "bayes_signal", "bayes_confidence"]


def _load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLS)


def _append_log(rows):
    if not rows:
        return
    log_df = pd.concat([_load_log(), pd.DataFrame(rows)], ignore_index=True)
    log_df.to_csv(LOG_PATH, index=False)


def reconcile_live_orders(broker):
    """
    live_orders_log.csv only records intent at entry time -- this fills in
    what actually happened. For every LIVE row that placed an entry
    (status PLACED or SL_ONLY) and isn't marked closed yet: if that coin no
    longer shows an open position on Hyperliquid, pull fills since the
    entry's timestamp and sum the closing ones (side-effect of the
    "skip tickers with an open position" rule elsewhere in this script --
    at most one logical trade per coin is ever open at a time, so every
    Close fill for that coin in the window belongs to this row) into a
    realized exit price and P&L. Mirrors Script 29's resolve_open_trades()
    for paper trades, applied to real fills instead of daily OHLC.
    """
    log_df = _load_log()
    if log_df.empty:
        return log_df, []

    for col in ("exit_price", "realized_pnl", "exit_time"):
        if col not in log_df.columns:
            log_df[col] = np.nan
    if "closed" not in log_df.columns:
        log_df["closed"] = False
    log_df["closed"] = log_df["closed"].fillna(False).astype(bool)

    open_positions = broker.get_open_positions()
    pending_mask = ((log_df["mode"] == "LIVE") & log_df["status"].isin(["PLACED", "SL_ONLY"])
                     & (~log_df["closed"]))

    newly_closed = []
    for idx in log_df[pending_mask].index:
        row = log_df.loc[idx]
        coin = row["coin"]
        if coin in open_positions:
            continue   # still open -- nothing to reconcile yet

        entry_ts_ms = int(pd.Timestamp(row["timestamp"]).timestamp() * 1000)
        try:
            fills = broker.get_fills_since(coin, entry_ts_ms)
        except Exception:
            continue   # transient API issue -- retry next run
        closes = [f for f in fills if "Close" in f.get("dir", "")]
        if not closes:
            continue   # exchange hasn't reported the closing fill(s) yet -- retry next run

        total_pnl = sum(float(f["closedPnl"]) for f in closes)
        total_sz  = sum(float(f["sz"]) for f in closes)
        exit_px   = (sum(float(f["px"]) * float(f["sz"]) for f in closes) / total_sz
                     if total_sz else None)
        exit_time = pd.Timestamp(max(f["time"] for f in closes), unit="ms").isoformat()

        log_df.loc[idx, "exit_price"]   = exit_px
        log_df.loc[idx, "realized_pnl"] = total_pnl
        log_df.loc[idx, "exit_time"]    = exit_time
        log_df.loc[idx, "closed"]       = True
        newly_closed.append({"ticker": row["ticker"], "coin": coin, "pnl": total_pnl, "exit_px": exit_px})

    log_df.to_csv(LOG_PATH, index=False)
    return log_df, newly_closed


def reconcile_pending_entries(broker):
    """
    A PENDING row means a GTC entry+SL bracket (see
    hyperliquid_broker.bracket_entry) was submitted but hadn't filled as
    of that run -- the SL is already safely tied to the position the
    instant it fills (atomic with the entry), but TP1/TP2 are placed
    separately and can only be submitted once a real position exists to
    reduce. This checks what's happened since:
      - filled (coin now in open_positions) -> attach TP1/TP2 now, using
        the sizes saved at submission time. SL is already active either
        way; this only affects whether a profit target is also set.
      - still resting (an open order exists for that coin, no position
        yet) -> leave as PENDING, check again next cycle.
      - neither (no position, no resting order) -> the GTC bracket was
        cancelled/expired without filling -- mark EXPIRED and stop
        tracking it.
    """
    log_df = _load_log()
    if log_df.empty:
        return log_df, []

    if "closed" not in log_df.columns:
        log_df["closed"] = False
    log_df["closed"] = log_df["closed"].fillna(False).astype(bool)

    pending_mask = (log_df["mode"] == "LIVE") & (log_df["status"] == "PENDING")
    if not pending_mask.any():
        return log_df, []

    open_positions = broker.get_open_positions()
    try:
        resting_coins = {o.get("coin") for o in broker.get_open_orders()}
    except Exception:
        resting_coins = set()   # transient API issue -- treat conservatively as "unknown", re-check next cycle

    newly_filled = []
    for idx in log_df[pending_mask].index:
        row = log_df.loc[idx]
        coin = row["coin"]
        is_buy = row["direction"] == "LONG"

        if coin in open_positions:
            tp1_sz = float(row["tp1_sz"]) if pd.notna(row.get("tp1_sz")) else 0.0
            tp2_sz = float(row["tp2_sz"]) if pd.notna(row.get("tp2_sz")) else 0.0

            def _place_tp_leg(sz, px):
                resp = broker.take_profit(coin, not is_buy, sz, px)
                errors = _order_errors(resp)
                if errors:
                    raise RuntimeError("; ".join(errors))
                return resp

            try:
                _retry(lambda: _place_tp_leg(tp1_sz, float(row["tp1"])))
                _retry(lambda: _place_tp_leg(tp2_sz, float(row["tp2"])))
                log_df.loc[idx, "status"] = "PLACED"
                log_df.loc[idx, "detail"] = "bracket filled, TP1/TP2 attached"
            except Exception as e:
                log_df.loc[idx, "status"] = "TP_FAILED"
                log_df.loc[idx, "detail"] = f"bracket filled (SL already active), TP failed after retries: {e}"
            newly_filled.append({"ticker": row["ticker"], "coin": coin})
        elif coin in resting_coins:
            continue   # still resting, not filled yet -- check again next cycle
        else:
            log_df.loc[idx, "status"] = "EXPIRED"
            log_df.loc[idx, "detail"] = "GTC bracket no longer resting and no position -- cancelled/expired without filling"

    log_df.to_csv(LOG_PATH, index=False)
    return log_df, newly_filled


def _retry(fn, attempts=3, delay=2):
    """A transient API hiccup shouldn't be the difference between a
    protected and an unprotected live position -- retry a few times
    before giving up."""
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts:
                time.sleep(delay)
    raise last_exc


def _order_errors(response):
    """Hyperliquid can respond with top-level 'status': 'ok' while still
    rejecting individual orders within it (e.g. 'Order must have minimum
    value of $10') -- that's not a Python exception, so callers that only
    check for a raised error will silently treat a rejected order as a
    success. Returns a list of per-order error strings (empty if every
    order in the response actually succeeded)."""
    try:
        statuses = response["response"]["data"]["statuses"]
    except (KeyError, TypeError):
        return []   # unexpected shape -- can't confirm errors, let the caller's own checks (e.g. position/order lookup) be the real source of truth
    return [s["error"] for s in statuses if isinstance(s, dict) and "error" in s]


def resolve_coin(ticker, dl, broker):
    symbol = dl.SYMBOLS.get(ticker) or dl.COMMON_SYMBOLS.get(ticker)
    if not symbol:
        return None
    naive_coin = symbol.split("/")[0]   # "BTC/USDC:USDC" -> "BTC"
    try:
        # ccxt's unified symbol and Hyperliquid's own native coin name can
        # differ in case (e.g. ccxt's "KBONK" vs Hyperliquid's "kBONK") --
        # resolve against Hyperliquid's real asset map so every downstream
        # call (leverage, entry, SL, TP) uses a name that actually exists.
        return broker.resolve_native_coin(naive_coin)
    except KeyError:
        return None


def _process_ticker(plan, broker, dl, mode, args, open_positions, resting_coins):
    """One ticker's full order-placement flow, from coin resolution
    through logging. Returns the log row dict for this ticker. Any
    unhandled exception propagates to the caller, which isolates it to
    this ticker only -- see the try/except around this call in
    run_once()."""
    ticker = plan["ticker"]
    coin = resolve_coin(ticker, dl, broker)
    ml = plan.get("ml")
    bd = plan.get("bayes_detect")
    base_row = {"timestamp": datetime.now().isoformat(), "mode": mode, "ticker": ticker,
                 "coin": coin, "direction": plan["direction"], "leverage": plan["leverage"],
                 "entry_price": plan["entry"], "sl1": plan["sl1"],
                 "tp1": plan["tp1"], "tp2": plan["tp2"],
                 "regime": plan.get("regime"), "conviction": plan.get("conviction"),
                 "fc_source": plan.get("fc_source"), "fc_total": plan.get("fc_total"),
                 "ml_signal": ml["signal"] if ml else None,
                 "ml_confidence": ml["confidence"] if ml else None,
                 "bayes_signal": bd["signal"] if bd else None,
                 "bayes_confidence": bd["confidence"] if bd else None}

    if coin is None:
        print(f"  {ticker}: no Hyperliquid symbol mapping — skipping.")
        return {**base_row, "size": 0, "status": "SKIPPED", "detail": "no coin mapping"}

    if coin in open_positions:
        print(f"  {ticker} ({coin}): position already open on Hyperliquid — skipping (no duplicate entry).")
        return {**base_row, "size": 0, "status": "SKIPPED", "detail": "position already open"}

    if coin in resting_coins:
        print(f"  {ticker} ({coin}): already has a resting entry bracket — skipping (no duplicate order).")
        return {**base_row, "size": 0, "status": "SKIPPED", "detail": "bracket entry already resting"}

    is_buy = plan["direction"] == "LONG"
    sz = broker.round_size(coin, plan["shares"])
    if sz <= 0:
        print(f"  {ticker} ({coin}): computed size rounds to 0 — skipping.")
        return {**base_row, "size": 0, "status": "SKIPPED", "detail": "size rounds to 0"}

    notional = sz * plan["entry"]
    if notional < HYPERLIQUID_MIN_NOTIONAL:
        print(f"  {ticker} ({coin}): notional ${notional:.2f} below Hyperliquid's "
              f"${HYPERLIQUID_MIN_NOTIONAL:.0f} minimum — skipping.")
        return {**base_row, "size": 0, "status": "SKIPPED",
                "detail": f"notional ${notional:.2f} below ${HYPERLIQUID_MIN_NOTIONAL:.0f} minimum"}

    tp1_sz = broker.round_size(coin, sz / 2)
    tp2_sz = round(sz - tp1_sz, 10)   # remainder, so the two legs never exceed entry size

    print(f"\n  {ticker} ({coin})  {plan['direction']}  {plan['leverage']}x  "
          f"size={sz}  entry~${plan['entry']:,.2f}  sl1=${plan['sl1']:,.2f}  "
          f"tp1=${plan['tp1']:,.2f} ({tp1_sz})  tp2=${plan['tp2']:,.2f} ({tp2_sz})")

    # Willing to rest at up to ENTRY_PRICE_TOLERANCE worse than the
    # planned entry (buy higher / sell lower), not just at-or-better
    # against a reference price that's already stale by execution time.
    tolerant_entry_px = plan["entry"] * (1 + ENTRY_PRICE_TOLERANCE if is_buy
                                          else 1 - ENTRY_PRICE_TOLERANCE)

    status, detail = "DRY_RUN", "no order sent"
    if args.live:
        try:
            broker.set_leverage(coin, plan["leverage"])
            # GTC entry + reduce-only SL submitted atomically (Hyperliquid's
            # "normalTpsl" order grouping) -- the stop is tied to the
            # position the instant it fills, even between --loop cycles
            # while nothing is polling. TP1/TP2 can't go in the same
            # atomic call (this SDK version's grouping only supports one
            # TP leg, and reduce-only orders need existing size to split
            # against) -- reconcile_pending_entries() attaches them once
            # a later cycle sees the bracket has actually filled.
            entry_resp = _retry(lambda: broker.bracket_entry(coin, is_buy, sz, tolerant_entry_px, plan["sl1"]))
            # Hyperliquid can return 'status': 'ok' at the top level while
            # still rejecting individual orders inside it (e.g. below its
            # $10 minimum notional) -- that's not a Python exception, so
            # without this check a rejected order gets logged as PENDING
            # (success) when nothing was actually placed. Not worth
            # retrying: a rejection like "too small" fails identically
            # every time, it's not transient.
            errors = _order_errors(entry_resp)
            if errors:
                raise RuntimeError("; ".join(errors))
        except Exception as e:
            status, detail = "ERROR", f"entry+SL bracket failed: {e}"
            print(f"  -> ERROR placing entry+SL bracket: {e}")
        else:
            status, detail = "PENDING", str(entry_resp)
            print(f"  -> GTC entry+SL bracket submitted — resting until filled or cancelled "
                  f"(SL already active the instant it fills; TP attached on a later cycle).")

    return {**base_row, "size": sz, "tp1_sz": tp1_sz, "tp2_sz": tp2_sz, "status": status, "detail": detail}


def run_once(broker, dl, args, mode):
    _r17.refresh_signal_weights()   # pick up any predictions Script 29 has graded since this process started
    print("=" * 70)
    print(f"  SCRIPT 30 — LIVE ORDER EXECUTION  [{mode}]")
    print(f"  {datetime.now():%Y-%m-%d %H:%M}")
    if args.live:
        print("  !! LIVE MODE — real orders will be sent to Hyperliquid.")
    else:
        print("  DRY RUN — no orders will be sent. Pass --live to actually trade.")
    print("=" * 70)

    if args.tickers:
        lookup = {**dl.COMMON_SYMBOLS, **dl.SYMBOLS}
        ov = {t: lookup[t] for t in args.tickers if t in lookup}
        if ov:
            dl.SYMBOLS = ov
            print(f"[30] Tickers restricted to: {list(dl.SYMBOLS.keys())}")
        else:
            print(f"[30] Warning: none of {args.tickers} found — using full universe.")

    equity = broker.get_equity()
    open_positions = broker.get_open_positions()
    try:
        resting_coins = {o.get("coin") for o in broker.get_open_orders()}
    except Exception:
        resting_coins = set()
    print(f"\n  Account equity: ${equity:,.2f}")
    print(f"  Open positions on Hyperliquid: {list(open_positions.keys()) or 'none'}")
    print(f"  Resting entry brackets: {list(resting_coins) or 'none'}")

    print("\n  Reconciling pending entry brackets (filled -> attach TP, expired -> stop tracking)...")
    log_df, newly_filled = reconcile_pending_entries(broker)
    for t in newly_filled:
        print(f"  {t['ticker']} ({t['coin']}) bracket filled — TP1/TP2 now attached.")

    print("\n  Reconciling prior live orders against actual fills...")
    log_df, newly_closed = reconcile_live_orders(broker)
    for t in newly_closed:
        print(f"  {t['ticker']} ({t['coin']}) closed — exit~${t['exit_px']:,.2f}  pnl=${t['pnl']:+,.2f}")
    closed = log_df[(log_df["mode"] == "LIVE") & log_df["closed"]] if not log_df.empty else log_df
    if len(closed):
        wins = (closed["realized_pnl"].astype(float) > 0)
        print(f"  Live track record: {wins.mean():.1%} win rate  "
              f"({int(wins.sum())}/{len(closed)} closed trades)  "
              f"total pnl=${closed['realized_pnl'].astype(float).sum():+,.2f}")
    else:
        print("  No closed live trades yet.")

    if equity <= MIN_EQUITY_FLOOR:
        print(f"\n  !! TRADING HALTED — equity ${equity:,.2f} is at or below the "
              f"${MIN_EQUITY_FLOOR:,.0f} drawdown floor. No new entries will be placed. "
              f"Existing positions are untouched -- each already has its own live "
              f"stop-loss managing it. Raise MIN_EQUITY_FLOOR in 30_live_executor.py "
              f"(or add capital) to resume.")
        return

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    load_all_funding(symbols={t: (dl.SYMBOLS.get(t) or dl.COMMON_SYMBOLS.get(t))
                               for t in assets_data if (dl.SYMBOLS.get(t) or dl.COMMON_SYMBOLS.get(t))})

    plans = []
    for ticker, df in assets_data.items():
        try:
            plans.append(generate_plan(ticker, df, capital=equity))
        except Exception as e:
            print(f"  {ticker}: plan error — {e}")

    trades = [pl for pl in plans if pl["direction"] != "NO TRADE"]

    avg_notional_if_spread = (equity / len(trades)) if trades else equity
    if (avg_notional_if_spread < HYPERLIQUID_MIN_NOTIONAL * LOW_CAPITAL_SAFETY_MULTIPLE
            and len(trades) > MAX_CONCURRENT_NEW_ENTRIES):
        trades = sorted(trades, key=lambda x: -x["conviction"])[:MAX_CONCURRENT_NEW_ENTRIES]
        print(f"\n  Capital-constrained (${equity:,.2f} spread across the original signal "
              f"count would average ~${avg_notional_if_spread:,.2f} each): concentrating into "
              f"the top {MAX_CONCURRENT_NEW_ENTRIES} signals by conviction "
              f"({', '.join(t['ticker'] for t in trades)}) instead of diluting across every "
              f"actionable signal until most fall under Hyperliquid's ${HYPERLIQUID_MIN_NOTIONAL:.0f} minimum.")

    apply_portfolio_risk_adjustments(trades, assets_data)

    if not trades:
        print("\n  No actionable trades this run.")
        return

    log_rows = []
    print(f"\n  {len(trades)} actionable plan(s):")
    for plan in sorted(trades, key=lambda x: -x["conviction"]):
        ticker = plan["ticker"]
        try:
            log_rows.append(_process_ticker(plan, broker, dl, mode, args, open_positions, resting_coins))
        except Exception as e:
            # A single ticker's failure (bad coin mapping, unexpected API
            # shape, etc.) must never take down every other actionable
            # trade in the run -- log it and move on to the next ticker.
            print(f"  {ticker}: unexpected ERROR, skipping this ticker only — {e}")
            log_rows.append({"timestamp": datetime.now().isoformat(), "mode": mode, "ticker": ticker,
                              "coin": None, "direction": plan.get("direction"), "leverage": plan.get("leverage"),
                              "entry_price": plan.get("entry"), "size": 0, "sl1": plan.get("sl1"),
                              "tp1": plan.get("tp1"), "tp2": plan.get("tp2"),
                              "status": "ERROR", "detail": f"unexpected error: {e}"})

    _append_log(log_rows)
    print(f"\n  Logged {len(log_rows)} row(s) to {LOG_PATH}")
    print("\nLive executor run complete.")


def main():
    p = argparse.ArgumentParser(description="Script 30 — Live Order Execution (Hyperliquid)")
    p.add_argument("--live", action="store_true",
                    help="Actually place orders. Without this flag, runs are "
                         "dry-run only — no orders are sent.")
    p.add_argument("--tickers", nargs="*", default=None, metavar="T",
                    help="Restrict to a subset of tickers (e.g. --tickers BTC ETH) "
                         "instead of the full dynamic universe.")
    p.add_argument("--loop", action="store_true",
                    help="Run forever, sleeping --interval minutes between checks, "
                         "instead of checking once and exiting.")
    p.add_argument("--interval", type=int, default=15, metavar="MINUTES",
                    help="Minutes between checks in --loop mode (default 15). "
                         "The underlying daily-bar data/cache doesn't change faster "
                         "than every few hours, so a shorter interval doesn't get "
                         "fresher signals -- it exists to check pending GTC entry "
                         "brackets more often (has one filled? attach TP. has it "
                         "expired? stop tracking it) and to react sooner to a "
                         "signal that wasn't fillable last cycle.")
    args = p.parse_args()
    mode = "LIVE" if args.live else "DRY_RUN"

    from hyperliquid_broker import HyperliquidBroker
    broker = HyperliquidBroker()
    import data_loader as dl

    if not args.loop:
        run_once(broker, dl, args, mode)
        return

    print(f"[30] Continuous mode — checking every {args.interval} minute(s). Ctrl+C to stop.\n")
    while True:
        try:
            run_once(broker, dl, args, mode)
        except Exception:
            print(f"\n[30] Iteration failed, will retry next cycle:\n{traceback.format_exc()}")
        next_run = datetime.now().timestamp() + args.interval * 60
        print(f"\n[30] Sleeping {args.interval} min — next check at "
              f"{datetime.fromtimestamp(next_run):%Y-%m-%d %H:%M}")
        try:
            time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            print("\n[30] Stopped.")
            break


if __name__ == "__main__":
    main()
