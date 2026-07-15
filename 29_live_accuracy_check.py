"""
================================================================
Script 29 — Live Accuracy Check (Paper-Trade + Prediction Log)
================================================================
Every other script in this suite operates on yesterday's close and
stops at "here's the plan" -- there was no way to see whether the
suite's calls actually hold up once real time passes. This script
closes that loop, acting as the paper-trading layer for an AI bot that
would otherwise place these trades for real:

  1. PREDICTION LOG: every run appends today's ML signal call (Script
     25's get_best_ml_signal -- a clean, single-bar-ahead, easy-to-grade
     claim) to predictions_log.csv, keyed to the close it was made from.
     The next time this script runs (once that ticker's next close is
     in the cache), it grades yesterday's logged calls against the
     realized close-to-close return and reports a running hit rate --
     an actual, growing track record instead of a one-off in-sample
     accuracy number.

  2. PAPER TRADE LOG: every actionable (LONG/SHORT) plan from Script 17
     gets logged to trade_outcomes_log.csv as an OPEN paper position
     (entry/SL1/SL2/TP1/TP2, no real capital). Each run walks forward
     through newly-available daily High/Low and resolves any trade
     whose stop or target has been touched (SL checked before TP on a
     day both could plausibly have hit -- daily OHLC can't tell us the
     intraday order, so the conservative assumption is the worse
     outcome), or times out after Script 17's HORIZON_DAYS.

  3. LIVE SNAPSHOT: fetches each ticker's current price (Hyperliquid
     ticker last/close, not a full OHLCV bar -- Hyperliquid's ticker
     endpoint doesn't populate intraday open/high/low the way a
     yfinance-style equities quote does) and checks it against today's
     plan.

No real orders are ever placed -- this is a paper-trading / track-
record tool, not an execution layer. There is no live-order-placement
code anywhere in this suite.
================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # this script saves charts to disk, never shows an
                         # interactive window -- it's meant to be run
                         # repeatedly/unattended, and a blocking plt.show()
                         # would hang that
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from importlib import import_module as _im
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, SYMBOLS, COMMON_SYMBOLS, get_all_tickers, format_price

_r17 = _im("17_trade_planner")
generate_plan = _r17.generate_plan
HORIZON_DAYS = _r17.HORIZON_DAYS

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH  = os.path.join(SUITE_DIR, "predictions_log.csv")
LOG_COLS  = ["date", "ticker", "ml_signal", "ml_confidence", "close_at_call",
             "plan_direction", "next_close", "realized_return", "ml_correct",
             "bayes_signal", "bayes_confidence", "bayes_correct"]

TRADE_LOG_PATH = os.path.join(SUITE_DIR, "trade_outcomes_log.csv")
TRADE_LOG_COLS = ["ticker", "entry_date", "direction", "entry_price", "sl1", "sl2", "tp1", "tp2",
                   "status", "exit_date", "exit_price", "return_pct", "days_held"]

LIVE_LOG_PATH       = os.path.join(SUITE_DIR, "live_orders_log.csv")   # written by Script 30
LIVE_PERF_PLOT_PATH = os.path.join(SUITE_DIR, "reports", "live_performance.png")


# ============================================================
# Live Price Fetch (Hyperliquid ticker snapshot, not a full OHLCV bar)
# ============================================================
def fetch_live_price(symbol):
    """Best-effort current price via Hyperliquid's ticker endpoint
    (data_loader.get_all_tickers, ~90s in-memory cache). Returns None if
    the symbol isn't in today's snapshot."""
    try:
        tickers = get_all_tickers()
        t = tickers.get(symbol)
        if t is None:
            return None
        last = t.get("last") or t.get("close")
        return float(last) if last is not None else None
    except Exception:
        return None


# ============================================================
# Prediction Log
# ============================================================
def _load_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLS)


def grade_pending(log_df, assets_data):
    """Fill in realized_return/ml_correct (and bayes_correct, if that call
    had a Bayesian-detection signal) for any logged rows whose next bar's
    close is now available in the cached history."""
    if log_df.empty:
        return log_df
    if "bayes_correct" not in log_df.columns:
        log_df["bayes_correct"] = np.nan
    pending = log_df["realized_return"].isna()
    for idx in log_df[pending].index:
        row = log_df.loc[idx]
        ticker = row["ticker"]
        if ticker not in assets_data:
            continue
        df = assets_data[ticker]
        call_date = pd.Timestamp(row["date"])
        after = df.index[df.index > call_date]
        if len(after) == 0:
            continue   # next close not in cache yet -- grade on a later run
        next_close = float(df.loc[after[0], "close"])
        ret = next_close / row["close_at_call"] - 1
        log_df.loc[idx, "next_close"] = next_close
        log_df.loc[idx, "realized_return"] = ret
        if row["ml_signal"] == "LONG":
            correct = ret > 0
        elif row["ml_signal"] == "SHORT":
            correct = ret < 0
        else:
            correct = np.nan   # FLAT calls aren't directional -- not gradeable
        log_df.loc[idx, "ml_correct"] = correct

        bayes_signal = row.get("bayes_signal")
        if bayes_signal == "LONG":
            log_df.loc[idx, "bayes_correct"] = ret > 0
        elif bayes_signal == "SHORT":
            log_df.loc[idx, "bayes_correct"] = ret < 0
        # else: no Bayesian-detection signal was logged for this call -- leave NaN
    return log_df


def append_todays_calls(log_df, plans_with_date):
    existing_keys = (set(zip(log_df["date"].astype(str), log_df["ticker"]))
                      if not log_df.empty else set())
    new_rows = []
    for plan, bar_date in plans_with_date:
        ml = plan.get("ml")
        if ml is None:
            continue   # NO TRADE tickers never reach Script 17's ML block
        date_str = str(pd.Timestamp(bar_date).date())
        if (date_str, plan["ticker"]) in existing_keys:
            continue
        bd = plan.get("bayes_detect")
        new_rows.append({
            "date": date_str, "ticker": plan["ticker"], "ml_signal": ml["signal"],
            "ml_confidence": ml["confidence"], "close_at_call": plan["price"],
            "plan_direction": plan["direction"], "next_close": np.nan,
            "realized_return": np.nan, "ml_correct": np.nan,
            "bayes_signal": bd["signal"] if bd else np.nan,
            "bayes_confidence": bd["confidence"] if bd else np.nan,
            "bayes_correct": np.nan,
        })
    if new_rows:
        log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)
    return log_df


# ============================================================
# Full SL/TP Paper Trade Log
# ============================================================
def _load_trade_log():
    if os.path.exists(TRADE_LOG_PATH):
        return pd.read_csv(TRADE_LOG_PATH)
    return pd.DataFrame(columns=TRADE_LOG_COLS)


def log_new_trades(trade_log, plans_with_date):
    existing_keys = (set(zip(trade_log["ticker"], trade_log["entry_date"].astype(str)))
                      if not trade_log.empty else set())
    new_rows = []
    for plan, bar_date in plans_with_date:
        if plan["direction"] not in ("LONG", "SHORT"):
            continue
        date_str = str(pd.Timestamp(bar_date).date())
        if (plan["ticker"], date_str) in existing_keys:
            continue
        new_rows.append({
            "ticker": plan["ticker"], "entry_date": date_str, "direction": plan["direction"],
            "entry_price": plan["entry"], "sl1": plan["sl1"], "sl2": plan["sl2"],
            "tp1": plan["tp1"], "tp2": plan["tp2"], "status": "OPEN",
            "exit_date": np.nan, "exit_price": np.nan, "return_pct": np.nan, "days_held": np.nan,
        })
    if new_rows:
        trade_log = pd.concat([trade_log, pd.DataFrame(new_rows)], ignore_index=True)
    return trade_log


def resolve_open_trades(trade_log, assets_data, max_hold_days=HORIZON_DAYS):
    """
    Walks forward day-by-day through cached daily High/Low after each OPEN
    paper trade's entry_date. On any day both a stop and a target could
    plausibly have been touched, checks SL first (daily OHLC can't tell us
    the intraday order, so the conservative assumption is the worse
    outcome). Resolves to SL1_HIT/SL2_HIT/TP1_HIT/TP2_HIT, or TIMEOUT after
    max_hold_days (exit at that day's close), or leaves the row OPEN if
    not enough bars have elapsed yet.
    """
    if trade_log.empty:
        return trade_log
    open_mask = trade_log["status"] == "OPEN"
    for idx in trade_log[open_mask].index:
        row = trade_log.loc[idx]
        ticker = row["ticker"]
        if ticker not in assets_data:
            continue
        df = assets_data[ticker]
        entry_date = pd.Timestamp(row["entry_date"])
        after = df.index[df.index > entry_date]
        if len(after) == 0:
            continue   # no new bars since entry yet

        direction = row["direction"]
        entry_price = float(row["entry_price"])
        sl1, sl2, tp1, tp2 = float(row["sl1"]), float(row["sl2"]), float(row["tp1"]), float(row["tp2"])

        resolved = False
        for days_held, dt in enumerate(after[:max_hold_days], start=1):
            hi, lo = float(df.loc[dt, "high"]), float(df.loc[dt, "low"])
            if direction == "LONG":
                if lo <= sl2:
                    status, exit_price = "SL2_HIT", sl2
                elif lo <= sl1:
                    status, exit_price = "SL1_HIT", sl1
                elif hi >= tp2:
                    status, exit_price = "TP2_HIT", tp2
                elif hi >= tp1:
                    status, exit_price = "TP1_HIT", tp1
                else:
                    continue
            else:  # SHORT
                if hi >= sl2:
                    status, exit_price = "SL2_HIT", sl2
                elif hi >= sl1:
                    status, exit_price = "SL1_HIT", sl1
                elif lo <= tp2:
                    status, exit_price = "TP2_HIT", tp2
                elif lo <= tp1:
                    status, exit_price = "TP1_HIT", tp1
                else:
                    continue

            ret = ((exit_price - entry_price) / entry_price if direction == "LONG"
                   else (entry_price - exit_price) / entry_price)
            trade_log.loc[idx, ["status", "exit_date", "exit_price", "return_pct", "days_held"]] = \
                [status, str(dt.date()), exit_price, ret, days_held]
            resolved = True
            break

        if not resolved and len(after) >= max_hold_days:
            dt = after[max_hold_days - 1]
            exit_price = float(df.loc[dt, "close"])
            ret = ((exit_price - entry_price) / entry_price if direction == "LONG"
                   else (entry_price - exit_price) / entry_price)
            trade_log.loc[idx, ["status", "exit_date", "exit_price", "return_pct", "days_held"]] = \
                ["TIMEOUT", str(dt.date()), exit_price, ret, max_hold_days]

    return trade_log


# ============================================================
# Live Snapshot vs Trade Plan
# ============================================================
def snapshot_status(plan, live_price):
    if plan["direction"] == "NO TRADE" or live_price is None:
        return "-"
    d, px = plan["direction"], live_price
    if d == "LONG":
        if px <= plan["sl2"]: return "SL2 HIT (disaster)"
        if px <= plan["sl1"]: return "SL1 HIT"
        if px >= plan["tp2"]: return "TP2 HIT (full target)"
        if px >= plan["tp1"]: return "TP1 HIT"
        return "ON TRACK" if px >= plan["entry"] else "OPEN (below entry)"
    else:  # SHORT
        if px >= plan["sl2"]: return "SL2 HIT (disaster)"
        if px >= plan["sl1"]: return "SL1 HIT"
        if px <= plan["tp2"]: return "TP2 HIT (full target)"
        if px <= plan["tp1"]: return "TP1 HIT"
        return "ON TRACK" if px <= plan["entry"] else "OPEN (above entry)"


# ============================================================
# Live Trading Performance (Script 30's live_orders_log.csv -- real
# money only. Separate from the paper-trade/prediction tracking above,
# which is simulated). This is the report to check "how are we
# actually doing" and "what needs fixing" -- everything else in this
# suite is signal generation or paper simulation, nothing else reads
# real fills back into a performance summary.
# ============================================================
def _load_live_log():
    if os.path.exists(LIVE_LOG_PATH):
        return pd.read_csv(LIVE_LOG_PATH)
    return pd.DataFrame()


def summarize_live_performance(live_log):
    """Win rate / P&L / per-ticker breakdown from Script 30's real fills,
    plus counts of anything that needs attention (UNPROTECTED positions,
    entry errors, still-open positions). Returns None if there's no live
    trading history yet."""
    if live_log.empty or "mode" not in live_log.columns:
        return None
    live = live_log[live_log["mode"] == "LIVE"].copy()
    if live.empty or "closed" not in live.columns:
        return None

    live["closed"] = live["closed"].fillna(False).astype(bool)
    closed = live[live["closed"]].copy()
    if len(closed):
        closed["realized_pnl"] = closed["realized_pnl"].astype(float)
    open_positions = live[live["status"].isin(["PLACED", "SL_ONLY"]) & (~live["closed"])]
    unprotected = live[live["status"] == "UNPROTECTED"]
    errors = live[live["status"] == "ERROR"]

    summary = {
        "n_closed": len(closed), "n_open": len(open_positions),
        "n_unprotected": len(unprotected), "n_errors": len(errors),
    }
    if len(closed):
        wins = closed["realized_pnl"] > 0
        summary["win_rate"]   = float(wins.mean())
        summary["total_pnl"]  = float(closed["realized_pnl"].sum())
        summary["avg_pnl"]    = float(closed["realized_pnl"].mean())
        summary["best"]       = closed.loc[closed["realized_pnl"].idxmax()]
        summary["worst"]      = closed.loc[closed["realized_pnl"].idxmin()]
        summary["per_ticker"] = (closed.groupby("ticker")["realized_pnl"]
                                  .agg(trades="count", total_pnl="sum", avg_pnl="mean")
                                  .sort_values("total_pnl"))

    return summary, closed, open_positions, unprotected, errors


def plot_live_performance(closed):
    """Equity curve + drawdown + per-ticker P&L, saved to disk (see the
    matplotlib.use('Agg') note at the top of this file -- never a
    blocking plt.show(), this needs to survive an unattended run)."""
    if closed.empty:
        return
    plt.style.use("seaborn-v0_8-darkgrid")
    closed = closed.sort_values("exit_time")
    cum_pnl = closed["realized_pnl"].cumsum()
    drawdown = cum_pnl - cum_pnl.cummax()

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(range(len(cum_pnl)), cum_pnl.values, lw=1.8, color="steelblue")
    ax0.axhline(0, color="gray", lw=1, ls="--")
    ax0.set_title("Live Trading — Cumulative Realized P&L ($)", fontsize=11)
    ax0.set_xlabel("Closed trade #")
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.fill_between(range(len(drawdown)), drawdown.values, 0, color="crimson", alpha=0.4)
    ax1.set_title("Drawdown ($)", fontsize=10)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    per_ticker = closed.groupby("ticker")["realized_pnl"].sum().sort_values()
    colors = ["crimson" if v < 0 else "seagreen" for v in per_ticker.values]
    ax2.barh(per_ticker.index, per_ticker.values, color=colors, alpha=0.85)
    ax2.set_title("P&L by Ticker ($)", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Live Trading Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(LIVE_PERF_PLOT_PATH), exist_ok=True)
    plt.savefig(LIVE_PERF_PLOT_PATH, dpi=120)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  LIVE ACCURACY CHECK -- Paper-Trade + Prediction Log")
    print(f"  {datetime.now():%Y-%m-%d %H:%M}")
    print("  (Paper trades only -- no real orders are placed anywhere in this suite)")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    log_df = _load_log()

    print("\n  Grading prior predictions against realized closes...")
    log_df = grade_pending(log_df, assets_data)
    graded = log_df.dropna(subset=["ml_correct"]) if not log_df.empty else log_df
    if len(graded):
        hit_rate = graded["ml_correct"].astype(bool).mean()
        print(f"  Running ML signal hit rate: {hit_rate:.1%}  "
              f"({int(graded['ml_correct'].astype(bool).sum())}/{len(graded)} graded calls)")
    else:
        print("  No graded predictions yet -- run this again after the next close.")

    bayes_graded = log_df.dropna(subset=["bayes_correct"]) if not log_df.empty else log_df
    if len(bayes_graded):
        bayes_hit_rate = bayes_graded["bayes_correct"].astype(bool).mean()
        print(f"  Running Bayes-detect signal hit rate: {bayes_hit_rate:.1%}  "
              f"({int(bayes_graded['bayes_correct'].astype(bool).sum())}/{len(bayes_graded)} graded calls)")

    trade_log = _load_trade_log()
    print("\n  Resolving open paper trades against realized daily High/Low...")
    trade_log = resolve_open_trades(trade_log, assets_data)
    resolved = trade_log[trade_log["status"] != "OPEN"] if not trade_log.empty else trade_log
    if len(resolved):
        wins = resolved["status"].isin(["TP1_HIT", "TP2_HIT"])
        win_rate = wins.mean()
        avg_return = resolved["return_pct"].astype(float).mean()
        print(f"  Running paper trade win rate: {win_rate:.1%}  "
              f"({int(wins.sum())}/{len(resolved)} resolved trades)  "
              f"avg return={avg_return:+.2%}")
    else:
        print("  No resolved paper trades yet -- run this again after SL/TP has time to hit.")

    print("\n" + "=" * 70)
    print("  LIVE TRADING PERFORMANCE (real orders only -- Script 30)")
    print("=" * 70)
    live_result = summarize_live_performance(_load_live_log())
    if live_result is None:
        print("  No live trades yet -- nothing to report until Script 30 runs with --live.")
    else:
        live_summary, live_closed, live_open, live_unprotected, live_errors = live_result
        if live_summary["n_closed"]:
            print(f"  Closed trades: {live_summary['n_closed']}  win rate={live_summary['win_rate']:.1%}  "
                  f"total P&L=${live_summary['total_pnl']:+,.2f}  avg P&L=${live_summary['avg_pnl']:+,.2f}")
            print(f"  Best:  {live_summary['best']['ticker']}  ${live_summary['best']['realized_pnl']:+,.2f}")
            print(f"  Worst: {live_summary['worst']['ticker']}  ${live_summary['worst']['realized_pnl']:+,.2f}")
            print("\n  Per-ticker:")
            print(live_summary["per_ticker"].to_string())
            plot_live_performance(live_closed)
            print(f"\n  Chart saved to {LIVE_PERF_PLOT_PATH}")
        else:
            print("  No closed live trades yet -- positions are still open or none have been placed.")
        if live_summary["n_open"]:
            print(f"\n  Currently open live positions: {live_summary['n_open']} "
                  f"({', '.join(live_open['ticker'].tolist())})")
        if live_summary["n_unprotected"]:
            print(f"\n  !! {live_summary['n_unprotected']} UNPROTECTED row(s) in live_orders_log.csv "
                  f"-- check Hyperliquid directly, these may have no live stop-loss.")
        if live_summary["n_errors"]:
            print(f"  {live_summary['n_errors']} ERROR row(s) logged -- entries that failed to place.")

    print("\n  Building today's trade plan + fetching live prices...")
    rows, plans_with_date = [], []
    for ticker, df in assets_data.items():
        try:
            plan = generate_plan(ticker, df)
        except Exception as e:
            print(f"  {ticker}: plan error -- {e}")
            continue
        plans_with_date.append((plan, df.index[-1]))

        symbol = SYMBOLS.get(ticker) or COMMON_SYMBOLS.get(ticker)
        live_price = fetch_live_price(symbol) if symbol else None
        if live_price is None:
            rows.append({"Ticker": ticker, "Dir": plan["direction"],
                         "PrevClose": format_price(plan["price"]), "Live": "n/a",
                         "Chg%": "n/a", "Status": "no live price"})
            continue
        chg = (live_price / plan["price"] - 1) * 100
        rows.append({
            "Ticker": ticker, "Dir": plan["direction"],
            "PrevClose": format_price(plan["price"]), "Live": format_price(live_price),
            "Chg%": f"{chg:+.2f}", "Status": snapshot_status(plan, live_price),
        })

    print("\n" + pd.DataFrame(rows).to_string(index=False))

    log_df = append_todays_calls(log_df, plans_with_date)
    log_df.to_csv(LOG_PATH, index=False)
    print(f"\n  Predictions logged to {LOG_PATH} for grading on a future run.")

    trade_log = log_new_trades(trade_log, plans_with_date)
    trade_log.to_csv(TRADE_LOG_PATH, index=False)
    print(f"  Paper trades logged to {TRADE_LOG_PATH} for SL/TP resolution on future runs.")
    print("\nLive accuracy check complete.")


if __name__ == "__main__":
    main()
