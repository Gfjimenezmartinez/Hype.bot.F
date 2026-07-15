"""
================================================================
Script 16 — Backtesting Engine
================================================================
Event-driven backtester that tests multiple strategies on each
asset and compares vs buy-and-hold.

Strategies:
  1. MA Crossover (trend-following)
  2. Mean-Reversion (Bollinger Band bounce)
  3. Quantile Breakout (distribution-based)
  4. Regime-Conditioned (uses Script 15's regime labels)
  5. ML Forecast (walk-forward wrapper around Script 25's confidence-
     gated logistic regression -- puts its signal through the same
     commission/slippage/Sharpe/drawdown gauntlet as the other 4)

Metrics: total return, Sharpe, max drawdown, win rate, profit
factor, trade count.  Equity curve + drawdown plot.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from data_loader import (load_all_assets, LOOKBACK_DAYS, format_price,
                          fetch_intraday, SYMBOLS, COMMON_SYMBOLS)

from importlib import import_module as _im

try:
    _r15 = _im("15_regime_detection")
    detect_regime = _r15.detect_regime
    REGIME_NAMES = _r15.REGIME_NAMES
    REGIME_AVAILABLE = True
except Exception:
    REGIME_AVAILABLE = False
    REGIME_NAMES = {0: "Low-Vol Trend", 1: "Mean-Revert", 2: "Crisis"}

try:
    _r25 = _im("25_ml_forecast_signal")
    get_ml_signal = _r25.get_ml_signal
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

try:
    _r17 = _im("17_trade_planner")
    generate_plan = _r17.generate_plan
    TRADE_PLANNER_AVAILABLE = True
except Exception:
    TRADE_PLANNER_AVAILABLE = False

PLOT_STYLE = "seaborn-v0_8-darkgrid"

COMMISSION  = 0.001
SLIPPAGE    = 0.0005
INITIAL_CAP = 100_000

# The ML strategy calibrates on its OWN bars rather than the daily `df` every
# other strategy in this file trades on -- LOOKBACK_DAYS' ~300 daily rows left
# several tickers stuck at 0-3 trades for the whole backtest (not enough
# history to calibrate a confidence threshold reliably). Which bar interval
# is actually best varies by asset category (majors like BTC/ETH/XRP skewed
# toward weekly in a live sweep, younger/more volatile alts toward hourly) --
# see Script 25's select_best_timeframe/get_cached_best_timeframe, cached
# once/day since it's a slow-moving property, not recomputed per call.
# Execution/measurement still happen on the daily timeline (see
# strat_ml_forecast) so backtest()'s and calc_metrics()' sqrt(365)
# annualization stays valid for every strategy regardless of what interval
# the ML strategy itself calibrated on.
ML_STRATEGY_CONFIG = {
    "1h": {"window": 1500, "refit_every": 120, "min_history": 300},   # ~62-day
                                                                        # window, refit
                                                                        # every 5 days
    "1d": {"window": 400,  "refit_every": 5,   "min_history": 150},   # ~13-month
                                                                        # window, refit
                                                                        # every 5 days
    "1w": {"window": 150,  "refit_every": 2,   "min_history": 130},   # ~2.9-year
                                                                        # window, refit
                                                                        # every 2 weeks
}
# refit_every above is a FLOOR, not the actual cadence -- a fixed "every 5 days"
# on a "1d"-selected major with ~2000 raw rows (BTC/ETH/AAVE/SOL all have that
# much daily history) means ~370 refits/ticker, each running ~7 internal
# logistic fits (get_ml_signal's confidence-threshold CV) -- this was the
# actual cause of a 12.7-minute full-suite run. Scaling refit_every to the
# available history keeps a hard ceiling per ticker regardless of how much
# data it has, so runtime stays well inside the ~10-minute budget an
# automated/scheduled run needs.
ML_MAX_REFITS_PER_TICKER = 20   # was 40 -- halved again once the full suite grew to 34
                                 # scripts (this strategy alone was ~237s of Script 16's
                                 # 277s); still enough refit checkpoints for a meaningful
                                 # walk-forward evaluation, just a coarser cadence


# ============================================================
# Strategy Signals — each returns array of +1 / -1 / 0
# ============================================================
def strat_ma_crossover(df, fast=10, slow=30):
    close = df["close"].values
    ma_f  = pd.Series(close).rolling(fast).mean().values
    ma_s  = pd.Series(close).rolling(slow).mean().values
    sig   = np.zeros(len(close))
    for i in range(slow, len(close)):
        if ma_f[i] > ma_s[i]:
            sig[i] = 1
        elif ma_f[i] < ma_s[i]:
            sig[i] = -1
    return sig


def strat_mean_reversion(df, window=20, n_std=2.0):
    close = df["close"].values
    ma    = pd.Series(close).rolling(window).mean().values
    std   = pd.Series(close).rolling(window).std().values
    sig   = np.zeros(len(close))
    for i in range(window, len(close)):
        upper = ma[i] + n_std * std[i]
        lower = ma[i] - n_std * std[i]
        if close[i] < lower:
            sig[i] = 1
        elif close[i] > upper:
            sig[i] = -1
        else:
            sig[i] = sig[i - 1] * 0.5
    sig = np.sign(sig)
    return sig


def strat_quantile(df, lookback=60):
    returns = df["log_return"].values
    sig     = np.zeros(len(returns))
    for i in range(lookback, len(returns)):
        window = returns[i - lookback:i]
        try:
            params = stats.t.fit(window)
            q10    = stats.t.ppf(0.10, *params)
            q90    = stats.t.ppf(0.90, *params)
            if returns[i] < q10:
                sig[i] = 1
            elif returns[i] > q90:
                sig[i] = -1
        except Exception:
            pass
    return sig


def strat_regime_conditioned(df, fast=10, slow=30, vol_window=20):
    """MA crossover in low-vol, mean-revert in high-vol, flat in crisis."""
    close   = df["close"].values
    returns = df["log_return"].values
    vol     = pd.Series(returns).rolling(vol_window).std().values
    vol_q   = pd.Series(vol).rolling(120).rank(pct=True).values

    ma_sig  = strat_ma_crossover(df, fast, slow)
    mr_sig  = strat_mean_reversion(df, vol_window, 2.0)

    sig = np.zeros(len(close))
    for i in range(max(120, slow), len(close)):
        if np.isnan(vol_q[i]):
            continue
        if vol_q[i] > 0.85:
            sig[i] = 0
        elif vol_q[i] < 0.35:
            sig[i] = ma_sig[i]
        else:
            sig[i] = mr_sig[i]
    return sig


def strat_ml_forecast(df, ticker=None):
    """
    Walk-forward wrapper around Script 25's confidence-gated logistic
    regression. Resolves `ticker`'s empirically-best bar interval (1h/1d/1w
    -- Script 25's get_cached_best_timeframe, cached once/day) rather than
    assuming one interval fits every crypto category, fetches ITS OWN bars
    at that interval (not the daily `df` every other strategy trades on),
    then refits every ML_STRATEGY_CONFIG[interval]["refit_every"] bars on a
    sliding window (data through that bar only -- no lookahead), holding the
    signal between refits. The resulting signal is forward-filled onto
    `df`'s daily index, so execution/PnL/Sharpe/drawdown still run on the
    same daily timeline as the other 4 strategies -- only the signal's
    calibration data/interval changes.
    """
    n   = len(df)
    sig = np.zeros(n)
    if not ML_AVAILABLE or ticker is None:
        return sig

    symbol = SYMBOLS.get(ticker) or COMMON_SYMBOLS.get(ticker)
    if not symbol:
        return sig

    interval = _r25.get_cached_best_timeframe(symbol)
    cfg      = ML_STRATEGY_CONFIG.get(interval, ML_STRATEGY_CONFIG["1d"])
    limit    = _r25.TIMEFRAME_CANDIDATES.get(interval, 2000)

    bars = fetch_intraday(symbol, interval=interval, limit=limit)
    if bars is None or len(bars) < cfg["min_history"]:
        return sig

    n_b = len(bars)
    span = n_b - cfg["min_history"]
    refit_every = max(cfg["refit_every"], span // ML_MAX_REFITS_PER_TICKER) if span > 0 else cfg["refit_every"]

    signal_map = {"LONG": 1, "SHORT": -1, "FLAT": 0}
    bar_sig = np.zeros(n_b)
    current = 0.0
    for i in range(cfg["min_history"], n_b):
        if (i - cfg["min_history"]) % refit_every == 0:
            window = bars.iloc[max(0, i + 1 - cfg["window"]):i + 1]
            try:
                result  = get_ml_signal(window)
                current = signal_map.get(result["signal"], 0)
            except Exception:
                current = 0.0
        bar_sig[i] = current

    bar_series   = pd.Series(bar_sig, index=bars.index)
    daily_series = bar_series.reindex(df.index, method="ffill").fillna(0.0)
    return daily_series.values


# generate_plan() itself runs a GARCH fit + Kalman/ARIMA forecast + regime
# HMM + 10k-path Monte Carlo + LQR per call -- comparable cost to a single
# ML strategy refit above, before even counting the (disabled, see below)
# ML step. Same refit-cadence-with-a-ceiling pattern as ML_STRATEGY_CONFIG,
# tuned separately since the per-call cost profile differs.
TRADE_PLANNER_CONFIG = {"refit_every": 5, "min_history": 120}
TRADE_PLANNER_MAX_REFITS_PER_TICKER = 40   # measured ~8.5s/ticker at this cap (no ML
                                             # network fetch, unlike strat_ml_forecast) --
                                             # ~1-2min added across the full universe


def strat_trade_planner(df, ticker=None):
    """
    Walk-forward wrapper around Script 17's actual generate_plan() --
    the same regime + trend + quantile + forecast + LQR-leverage decision
    Script 30 executes live. Only the resulting DIRECTION (+1/-1/0) feeds
    the shared backtest() engine below, same as every other strategy here
    -- this measures whether the entry/exit LOGIC has historical edge, not
    Script 17's leverage or position sizing (this backtester always trades
    a fixed 95%-of-cash allocation, no strategy here gets to use leverage).

    Script 25's ML confirmation step is deliberately DISABLED for this
    walk-forward test: get_best_ml_signal() fetches CURRENT intraday data
    by symbol regardless of what historical window it's handed (see
    25_ml_forecast_signal.py's get_best_ml_signal -- it only falls back to
    the caller's df if the live fetch comes up short), so calling it at a
    historical bar would leak today's actual price action into that bar's
    signal. strat_ml_forecast() above already solved this the honest way
    for the ML strategy itself (manually fetching once and slicing
    historical windows); reusing generate_plan() wholesale here would
    silently reintroduce the same leak through Script 17's internal ML
    call, so it's switched off via ML_AVAILABLE instead -- this backtests
    the regime/trend/quantile/forecast/LQR core, not the ML-sharpened
    conviction on top of it.
    """
    n   = len(df)
    sig = np.zeros(n)
    if not TRADE_PLANNER_AVAILABLE or ticker is None:
        return sig

    min_history = TRADE_PLANNER_CONFIG["min_history"]
    if n < min_history:
        return sig

    span = n - min_history
    refit_every = (max(TRADE_PLANNER_CONFIG["refit_every"], span // TRADE_PLANNER_MAX_REFITS_PER_TICKER)
                   if span > 0 else TRADE_PLANNER_CONFIG["refit_every"])

    direction_map = {"LONG": 1, "SHORT": -1, "NO TRADE": 0}
    original_ml_available = _r17.ML_AVAILABLE
    original_adaptive_weights = _r17.ADAPTIVE_SIGNAL_WEIGHTS
    original_walkforward_selection = _r17.WALKFORWARD_MODEL_SELECTION
    _r17.ML_AVAILABLE = False
    # Same lookahead concern as ML above, different mechanism: the adaptive
    # modulator weight (signal_modulator_weight in 17_trade_planner.py)
    # reflects TODAY's accumulated predictions_log.csv track record, which
    # wasn't known at any historical bar being walk-forwarded here. Pin it
    # to the fixed fallback weight for the whole backtest instead.
    _r17.ADAPTIVE_SIGNAL_WEIGHTS = False
    # Same reasoning again for forecast model selection: get_cached_best_model
    # (17_trade_planner.py's forecast_return) reflects TODAY's walk-forward
    # validation, not what was known at any historical bar. Pin to the fixed
    # adaptive->ARIMA->historical-mean fallback order for the whole backtest.
    _r17.WALKFORWARD_MODEL_SELECTION = False
    try:
        current = 0.0
        for i in range(min_history, n):
            if (i - min_history) % refit_every == 0:
                window = df.iloc[:i + 1]   # bars up to and including i only -- no lookahead
                try:
                    plan = generate_plan(ticker, window)
                    current = direction_map.get(plan["direction"], 0)
                except Exception:
                    current = 0.0
            sig[i] = current
    finally:
        _r17.ML_AVAILABLE = original_ml_available
        _r17.ADAPTIVE_SIGNAL_WEIGHTS = original_adaptive_weights
        _r17.WALKFORWARD_MODEL_SELECTION = original_walkforward_selection

    return sig


STRATEGIES = {
    "MA_Cross":   strat_ma_crossover,
    "MeanRevert": strat_mean_reversion,
    "Quantile":   strat_quantile,
    "RegimeCond": strat_regime_conditioned,
}
if ML_AVAILABLE:
    STRATEGIES["MLForecast"] = strat_ml_forecast
if TRADE_PLANNER_AVAILABLE:
    STRATEGIES["TradePlanner"] = strat_trade_planner


# ============================================================
# Backtester Core
# ============================================================
def backtest(df, signals, initial_cap=INITIAL_CAP,
             commission=COMMISSION, slippage=SLIPPAGE):
    close    = df["close"].values
    n        = len(close)
    cash     = initial_cap
    position = 0.0
    equity   = np.zeros(n)
    trades   = []
    entry_price_rec = 0.0

    for i in range(n):
        port_val = cash + position * close[i]
        equity[i] = max(port_val, 0.01)

        if i == 0:
            continue

        target = signals[i]
        current_pos = 1 if position > 0 else (-1 if position < 0 else 0)

        if target != current_pos:
            if position != 0:
                sell_price = close[i] * (1 - slippage * np.sign(position))
                proceeds   = position * sell_price
                cost       = abs(proceeds) * commission
                pnl        = proceeds - cost - position * entry_price_rec
                cash      += proceeds - cost
                position   = 0.0
                trades.append({"pnl": pnl})

            if target != 0 and cash > 100:
                alloc       = cash * 0.95
                entry_price = close[i] * (1 + slippage * target)
                shares      = (alloc / abs(entry_price)) * target
                cost        = abs(shares * entry_price) * commission
                cash       -= shares * entry_price + cost
                position    = shares
                entry_price_rec = entry_price

        equity[i] = cash + position * close[i]

    # Close final position
    if position != 0:
        cash += position * close[-1] * (1 - slippage * np.sign(position))
        position = 0
    equity[-1] = cash

    return equity, trades


# ============================================================
# Performance Metrics
# ============================================================
def calc_metrics(equity, df):
    returns  = np.diff(equity) / equity[:-1]
    returns  = returns[np.isfinite(returns)]
    total_r  = (equity[-1] / equity[0] - 1) * 100
    ann_r    = ((equity[-1] / equity[0]) ** (365 / max(len(equity), 1)) - 1) * 100  # crypto trades 24/7
    vol      = float(np.std(returns) * np.sqrt(365) * 100) if len(returns) > 1 else 0
    sharpe   = float(np.mean(returns) / np.std(returns) * np.sqrt(365)) if np.std(returns) > 0 else 0

    peak     = np.maximum.accumulate(equity)
    dd       = (equity - peak) / peak
    max_dd   = float(dd.min() * 100)

    # Buy and hold comparison
    close    = df["close"].values
    bh_ret   = (close[-1] / close[0] - 1) * 100

    return {
        "total_return": total_r,
        "ann_return":   ann_r,
        "ann_vol":      vol,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "buy_hold":     bh_ret,
        "alpha":        total_r - bh_ret,
    }


def trade_stats(trades):
    if not trades:
        return {"n_trades": 0, "win_rate": 0, "profit_factor": 0}
    pnls    = [t.get("pnl", 0) for t in trades if "pnl" in t]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p < 0]
    n       = len(pnls)
    wr      = len(wins) / n * 100 if n > 0 else 0
    pf      = sum(wins) / abs(sum(losses)) if losses else float("inf")
    return {"n_trades": n, "win_rate": wr, "profit_factor": round(pf, 2)}


# ============================================================
# Plotting
# ============================================================
def plot_backtest(ticker, df, results):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

    colors = {"MA_Cross": "royalblue", "MeanRevert": "darkorange",
              "Quantile": "teal", "RegimeCond": "purple", "MLForecast": "crimson",
              "TradePlanner": "forestgreen"}

    # [0,0:1] Equity curves
    ax0 = fig.add_subplot(gs[0, :])
    bh  = df["close"].values / df["close"].values[0] * INITIAL_CAP
    ax0.plot(df.index, bh, color="gray", lw=1, alpha=0.6, label="Buy & Hold")
    for name, res_tuple in results.items():
        equity = res_tuple[0]
        ax0.plot(df.index, equity, color=colors.get(name, "black"),
                 lw=1.5, label=name)
    ax0.set_ylabel("Portfolio Value ($)")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Strategy Equity Curves", fontsize=11)
    ax0.grid(alpha=0.3)

    # [1,0] Drawdown
    ax1 = fig.add_subplot(gs[1, 0])
    for name, res_tuple in results.items():
        equity = res_tuple[0]
        peak = np.maximum.accumulate(equity)
        dd   = (equity - peak) / peak * 100
        ax1.plot(df.index, dd, color=colors.get(name, "black"), lw=1, label=name)
    ax1.set_ylabel("Drawdown (%)")
    ax1.legend(fontsize=7)
    ax1.set_title("Drawdown", fontsize=10)
    ax1.grid(alpha=0.3)

    # [1,1] Returns bar chart
    ax2    = fig.add_subplot(gs[1, 1])
    names  = list(results.keys())
    rets   = [results[n][2]["total_return"] for n in names]
    bh_r   = results[names[0]][2]["buy_hold"]
    all_n  = names + ["Buy&Hold"]
    all_r  = rets + [bh_r]
    bars_c = [colors.get(n, "gray") for n in names] + ["gray"]
    y_pos  = range(len(all_n))
    ax2.barh(y_pos, all_r, color=bars_c, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(all_n, fontsize=8)
    for i, v in enumerate(all_r):
        ax2.text(v + 0.5, i, f"{v:+.1f}%", va="center", fontsize=7)
    ax2.set_xlabel("Total Return (%)")
    ax2.set_title("Strategy Comparison", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    # [2,0:1] Metrics table
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis("off")
    rows = []
    for name in names:
        m  = results[name][2]
        ts = results[name][3] if len(results[name]) > 3 else {}
        rows.append([
            name,
            f"{m['total_return']:+.1f}%",
            f"{m['ann_return']:+.1f}%",
            f"{m['ann_vol']:.1f}%",
            f"{m['sharpe']:.2f}",
            f"{m['max_drawdown']:.1f}%",
            f"{m['alpha']:+.1f}%",
            f"{ts.get('n_trades', '?')}",
            f"{ts.get('win_rate', 0):.0f}%",
        ])
    rows.append([
        "Buy & Hold",
        f"{bh_r:+.1f}%", "-", "-", "-", "-", "-", "-", "-",
    ])
    table = ax3.table(
        cellText=rows,
        colLabels=["Strategy", "Return", "Ann.Ret", "Ann.Vol",
                   "Sharpe", "MaxDD", "Alpha", "Trades", "Win%"],
        loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    ax3.set_title(f"{ticker} — Performance Metrics", fontsize=10, pad=15)

    fig.suptitle(f"{ticker} — Backtesting Dashboard", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("BACKTESTING ENGINE")
    print(f"Strategies: {', '.join(STRATEGIES.keys())}")
    print(f"Commission: {COMMISSION*100:.2f}%  Slippage: {SLIPPAGE*100:.2f}%")
    print(f"Initial Capital: ${INITIAL_CAP:,.0f}")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    all_summary = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        regime_str = ""
        if REGIME_AVAILABLE:
            try:
                rid, rname, _ = detect_regime(df)
                regime_str = f"  [{rname}]"
            except Exception:
                pass
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}{regime_str}")
        print(f"{'─'*50}")

        try:
            results = {}
            for name, strat_fn in STRATEGIES.items():
                signals = strat_fn(df, ticker) if name in ("MLForecast", "TradePlanner") else strat_fn(df)
                equity, trades = backtest(df, signals)
                metrics = calc_metrics(equity, df)
                tstats  = trade_stats(trades)

                results[name] = (equity, trades, metrics, tstats)

                alpha_str = f"{metrics['alpha']:+.1f}%"
                print(f"  {name:<12}: ret={metrics['total_return']:+.1f}%  "
                      f"sharpe={metrics['sharpe']:.2f}  "
                      f"maxDD={metrics['max_drawdown']:.1f}%  "
                      f"alpha={alpha_str}  "
                      f"trades={tstats['n_trades']}")

            bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
            print(f"  {'Buy&Hold':<12}: ret={bh:+.1f}%")

            best = max(results.items(), key=lambda x: x[1][2]["total_return"])
            all_summary.append({
                "Ticker": ticker,
                "Best": best[0],
                "Return%": f"{best[1][2]['total_return']:+.1f}",
                "Sharpe": f"{best[1][2]['sharpe']:.2f}",
                "MaxDD%": f"{best[1][2]['max_drawdown']:.1f}",
                "Alpha%": f"{best[1][2]['alpha']:+.1f}",
                "BuyHold%": f"{bh:+.1f}",
            })

            plot_backtest(ticker, df, results)

        except Exception as e:
            print(f"  Error: {e}")

    if all_summary:
        print("\n" + "=" * 65)
        print("BACKTESTING SUMMARY — BEST STRATEGY PER ASSET")
        print("=" * 65)
        print(pd.DataFrame(all_summary).to_string(index=False))

        strat_wins = {}
        for s in all_summary:
            b = s["Best"]
            strat_wins[b] = strat_wins.get(b, 0) + 1
        print(f"\n  Strategy win count:")
        for s, c in sorted(strat_wins.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c}/{len(all_summary)} assets")

    print("\nBacktesting complete.")


if __name__ == "__main__":
    main()
