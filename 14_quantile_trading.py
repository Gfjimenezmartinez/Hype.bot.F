"""
================================================================
Script 14 — Quantile-Based Trading Signals
================================================================
Fits distributions to returns at multiple lookback windows
(short/medium/long as proxy for multi-timeframe analysis),
computes quantile price levels, detects trend, and generates
mean-reversion trading signals with entry/stop/target.

Adapted from: quantile_trading.py (crypto quant suite)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, DISPLAY_NAMES, format_price

try:
    from importlib import import_module as _im
    _r15 = _im("15_regime_detection")
    detect_regime = _r15.detect_regime
    REGIME_AVAILABLE = True
except Exception:
    REGIME_AVAILABLE = False

PLOT_STYLE    = "seaborn-v0_8-darkgrid"
PROXIMITY_PCT = 0.015

DISTRIBUTIONS = {
    "Normal":    stats.norm,
    "Student-t": stats.t,
    "Laplace":   stats.laplace,
    "Logistic":  stats.logistic,
}

QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

LOOKBACKS = {
    "bias":  250,
    "entry": 60,
    "risk":  20,
}


# ============================================================
# Distribution Fitting
# ============================================================
def fit_best(returns):
    best_aic, best_name, best_params, best_dist = np.inf, None, None, None
    for name, dist in DISTRIBUTIONS.items():
        try:
            params = dist.fit(returns)
            ll     = float(np.sum(dist.logpdf(returns, *params)))
            aic    = 2 * len(params) - 2 * ll
            if aic < best_aic:
                best_aic, best_name, best_params, best_dist = aic, name, params, dist
        except Exception:
            continue
    return best_name, best_params, best_dist


def quantile_prices(dist, params, current_price, returns):
    q_returns, q_prices = {}, {}
    for level in QUANTILE_LEVELS:
        qr = float(dist.ppf(level, *params))
        q_returns[level] = qr
        q_prices[level]  = current_price * np.exp(qr)
    return q_returns, q_prices


# ============================================================
# Trend & Position Detection
# ============================================================
def detect_trend(close, fast=20, slow=50):
    if len(close) < slow:
        return "NEUTRAL"
    c  = float(close.iloc[-1])
    ma = float(close.rolling(fast).mean().iloc[-1])
    mb = float(close.rolling(slow).mean().iloc[-1])
    if c > ma > mb:
        return "BULLISH"
    if c < ma < mb:
        return "BEARISH"
    return "NEUTRAL"


def market_position(price, q_prices):
    if not q_prices:
        return "UNKNOWN"
    if price <= q_prices.get(0.05, -np.inf):
        return "EXTREME_LOW"
    if price <= q_prices.get(0.10, -np.inf):
        return "OVERSOLD"
    if price <= q_prices.get(0.25, -np.inf):
        return "LOW"
    if price >= q_prices.get(0.95, np.inf):
        return "EXTREME_HIGH"
    if price >= q_prices.get(0.90, np.inf):
        return "OVERBOUGHT"
    if price >= q_prices.get(0.75, np.inf):
        return "HIGH"
    return "NEUTRAL"


def signal_strength(price, q_prices, direction):
    score = 1
    if direction == "long":
        if price <= q_prices.get(0.05, -np.inf):
            score = 5
        elif price <= q_prices.get(0.10, -np.inf):
            score = 4
        elif abs(price - q_prices.get(0.10, price)) / price <= PROXIMITY_PCT:
            score = 3
        elif abs(price - q_prices.get(0.25, price)) / price <= PROXIMITY_PCT:
            score = 2
    else:
        if price >= q_prices.get(0.95, np.inf):
            score = 5
        elif price >= q_prices.get(0.90, np.inf):
            score = 4
        elif abs(price - q_prices.get(0.90, price)) / price <= PROXIMITY_PCT:
            score = 3
        elif abs(price - q_prices.get(0.75, price)) / price <= PROXIMITY_PCT:
            score = 2
    bars   = "#" * score + "-" * (5 - score)
    labels = {5: "EXTREME", 4: "STRONG", 3: "MODERATE", 2: "WEAK", 1: "WATCH"}
    return score, f"[{bars}] {labels[score]}"


def calc_rr(entry, stop, target, short=False):
    try:
        if short:
            risk, reward = stop - entry, entry - target
        else:
            risk, reward = entry - stop, target - entry
        return round(reward / risk, 2) if risk > 0 else 0.0
    except Exception:
        return 0.0


# ============================================================
# Signal Generation
# ============================================================
def generate_signals(price, trend, entry_qp, risk_qp, bias_pos, regime=1):
    signals = {}

    # Regime gating: crisis → no new entries
    if regime == 2:
        return {"REDUCE": {"strength": "[#####] CRISIS", "score": 5,
                "type": "REGIME_REDUCE", "bias": trend, "entry": "-",
                "stop": "-", "target_1": "-", "target_2": "-", "r_r": "0R"}}

    near_support = (
        price <= entry_qp.get(0.10, np.inf)
        or abs(price - entry_qp.get(0.10, price)) / price <= PROXIMITY_PCT
    )
    if trend in ("BULLISH", "NEUTRAL") and near_support:
        score, label = signal_strength(price, entry_qp, "long")
        rr = calc_rr(price,
                     stop=risk_qp.get(0.05, price * 0.97),
                     target=risk_qp.get(0.75, price * 1.05))
        signals["LONG"] = {
            "strength":  label,
            "score":     score,
            "type":      "MEAN_REVERSION_LONG",
            "bias":      trend,
            "entry":     f"${entry_qp.get(0.05, 0):,.2f} - ${entry_qp.get(0.10, 0):,.2f}",
            "stop":      f"${risk_qp.get(0.05, 0):,.2f}",
            "target_1":  f"${risk_qp.get(0.50, 0):,.2f}",
            "target_2":  f"${risk_qp.get(0.75, 0):,.2f}",
            "r_r":       f"{rr:.1f}R",
        }

    near_resistance = (
        price >= entry_qp.get(0.90, 0)
        or abs(price - entry_qp.get(0.90, price)) / price <= PROXIMITY_PCT
    )
    if trend in ("BEARISH", "NEUTRAL") and near_resistance:
        score, label = signal_strength(price, entry_qp, "short")
        rr = calc_rr(price,
                     stop=risk_qp.get(0.95, price * 1.03),
                     target=risk_qp.get(0.25, price * 0.95),
                     short=True)
        signals["SHORT"] = {
            "strength":  label,
            "score":     score,
            "type":      "MEAN_REVERSION_SHORT",
            "bias":      trend,
            "entry":     f"${entry_qp.get(0.90, 0):,.2f} - ${entry_qp.get(0.95, 0):,.2f}",
            "stop":      f"${risk_qp.get(0.95, 0):,.2f}",
            "target_1":  f"${risk_qp.get(0.50, 0):,.2f}",
            "target_2":  f"${risk_qp.get(0.25, 0):,.2f}",
            "r_r":       f"{rr:.1f}R",
        }

    return signals


# ============================================================
# Per-Asset Analysis
# ============================================================
def analyse_asset(ticker, df):
    price = float(df["close"].iloc[-1])
    close = df["close"]
    tf_results = {}

    for tf_name, lb in LOOKBACKS.items():
        sub = df.iloc[-lb:] if len(df) >= lb else df
        returns = sub["log_return"].dropna().values.astype(float)
        if len(returns) < 20:
            continue

        dist_name, params, dist_obj = fit_best(returns)
        if dist_name is None:
            continue

        _, q_prices = quantile_prices(dist_obj, params, price, returns)
        trend    = detect_trend(close) if tf_name == "bias" else None
        position = market_position(price, q_prices)

        tf_results[tf_name] = {
            "lookback": lb, "dist": dist_name, "position": position,
            "q_prices": q_prices, "trend": trend,
            "vol": float(returns.std()),
        }

    if len(tf_results) < 3:
        return tf_results, {}

    trend     = tf_results["bias"]["trend"]
    entry_qp  = tf_results["entry"]["q_prices"]
    risk_qp   = tf_results["risk"]["q_prices"]
    bias_pos  = tf_results["bias"]["position"]

    regime_id = 1
    if REGIME_AVAILABLE:
        try:
            regime_id, _, _ = detect_regime(df)
        except Exception:
            pass

    signals = generate_signals(price, trend, entry_qp, risk_qp, bias_pos,
                               regime=regime_id)

    return tf_results, signals


# ============================================================
# Plotting
# ============================================================
def plot_signals_summary(all_results):
    plt.style.use(PLOT_STYLE)
    tickers = [t for t, (tf, sig) in all_results.items() if "entry" in tf]
    if not tickers:
        return
    n = len(tickers)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, n * 0.4)))

    # Quantile position chart
    ax = axes[0]
    positions = []
    for i, ticker in enumerate(tickers):
        tf = all_results[ticker][0]
        if "entry" not in tf:
            continue
        qp    = tf["entry"]["q_prices"]
        price = float(next(iter(all_results[ticker][0].values())).get("q_prices", {}).get(0.50, 0))
        actual = float([d for d in all_results[ticker][0].values()][0].get("q_prices", {}).get(0.50, 0))

    # Simpler: signal strength heatmap
    ax = axes[0]
    signal_data = []
    for ticker in tickers:
        tf, sigs = all_results[ticker]
        if "entry" not in tf:
            continue
        price = float(tf["entry"]["q_prices"].get(0.50, 0))
        pos   = tf["entry"]["position"]
        pos_map = {"EXTREME_LOW": -3, "OVERSOLD": -2, "LOW": -1,
                   "NEUTRAL": 0, "HIGH": 1, "OVERBOUGHT": 2, "EXTREME_HIGH": 3}
        signal_data.append((ticker, pos_map.get(pos, 0), pos))

    if signal_data:
        names = [s[0] for s in signal_data]
        vals  = [s[1] for s in signal_data]
        cols  = ["#d32f2f" if v <= -2 else "#ff9800" if v == -1
                 else "#4caf50" if v >= 2 else "#2196f3" if v == 1
                 else "#9e9e9e" for v in vals]
        y = range(len(names))
        ax.barh(y, vals, color=cols, alpha=0.8, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        for i, (_, v, label) in enumerate(signal_data):
            ax.text(v + (0.1 if v >= 0 else -0.1), i, label,
                    va="center", ha="left" if v >= 0 else "right", fontsize=6)
        ax.set_xlabel("Position Score", fontsize=8)
        ax.set_title("Market Position  |  Entry Timeframe (60d)", fontsize=10)
        ax.axvline(0, color="gray", lw=0.5)
        ax.grid(axis="x", alpha=0.3)

    # Signal summary
    ax2 = axes[1]
    ax2.axis("off")
    rows = []
    for ticker in tickers:
        tf, sigs = all_results[ticker]
        trend = tf.get("bias", {}).get("trend", "?")
        pos   = tf.get("entry", {}).get("position", "?")
        sig_str = ", ".join(f"{d} ({s['strength'].split(']')[1].strip()})"
                           for d, s in sigs.items()) if sigs else "No signal"
        rows.append([ticker, trend, pos, sig_str])
    if rows:
        table = ax2.table(cellText=rows,
                          colLabels=["Ticker", "Trend", "Position", "Signal"],
                          loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.4)
        ax2.set_title("Signal Summary", fontsize=10, pad=20)

    fig.suptitle("Quantile Trading Signals Dashboard", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("QUANTILE-BASED TRADING SIGNALS")
    print("Multi-Lookback Distribution Analysis")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    all_results = {}
    actionable  = []

    for ticker, df in assets_data.items():
        price = float(df["close"].iloc[-1])
        print(f"\n{'─'*50}")
        print(f"  {ticker}   {format_price(price)}")
        print(f"{'─'*50}")

        try:
            tf_results, signals = analyse_asset(ticker, df)
            all_results[ticker] = (tf_results, signals)

            for tf_name, info in tf_results.items():
                pos_icon = {"EXTREME_LOW": "!!", "OVERSOLD": "! ",
                            "LOW": "v ", "NEUTRAL": "  ",
                            "HIGH": "^ ", "OVERBOUGHT": "! ",
                            "EXTREME_HIGH": "!!"}.get(info["position"], "  ")
                trend_str = f"  trend={info['trend']}" if info.get("trend") else ""
                print(f"  [{tf_name.upper():<5} {info['lookback']:>3}d]  "
                      f"dist={info['dist']:<10} {pos_icon}{info['position']:<14}{trend_str}")
                qp = info["q_prices"]
                for level, label in [(0.05, "Strong Sup"), (0.10, "Support"),
                                     (0.50, "Fair Value"), (0.90, "Resistance"),
                                     (0.95, "Strong Res")]:
                    marker = " << PRICE" if abs(price - qp.get(level, -1)) / max(price, 1) < 0.005 else ""
                    print(f"    {label:<11}: ${qp.get(level, 0):>12,.2f}{marker}")

            if signals:
                print(f"\n  SIGNALS:")
                for direction, sig in signals.items():
                    print(f"    {direction}  {sig['strength']}")
                    print(f"      Type:    {sig['type']}")
                    print(f"      Bias:    {sig['bias']}")
                    print(f"      Entry:   {sig['entry']}")
                    print(f"      Stop:    {sig['stop']}")
                    print(f"      Target1: {sig['target_1']}")
                    print(f"      Target2: {sig['target_2']}")
                    print(f"      R:R:     {sig['r_r']}")
                    actionable.append({
                        "Ticker": ticker, "Dir": direction,
                        "Score": sig["score"], "Strength": sig["strength"],
                        "Bias": sig["bias"], "R:R": sig["r_r"],
                    })
            else:
                eq = tf_results.get("entry", {}).get("q_prices", {})
                if eq:
                    d_lo = abs(price - eq.get(0.10, price)) / price * 100
                    d_hi = abs(price - eq.get(0.90, price)) / price * 100
                    print(f"\n  No signal — {d_lo:.1f}% from support | {d_hi:.1f}% from resistance")

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    print("\n" + "=" * 65)
    print("ACTIONABLE SIGNALS")
    print("=" * 65)
    if actionable:
        df_sig = pd.DataFrame(actionable).sort_values("Score", ascending=False)
        print(df_sig.to_string(index=False))
    else:
        print("  No actionable signals at current levels.")

    if all_results:
        plot_signals_summary(all_results)

    print("\nQuantile trading analysis complete.")


if __name__ == "__main__":
    main()
