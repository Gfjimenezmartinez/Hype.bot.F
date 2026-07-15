"""
================================================================
Script 28 — Statistical Arbitrage: Cointegration & Pairs Trading
================================================================
Scripts 15/18 already build correlation and regime infrastructure across
the whole universe, but nothing tests for COINTEGRATION — a much
stronger, more tradeable relationship than correlation. Two assets can
be cointegrated (their price spread is stationary / mean-reverting)
even with modest correlation, and that spread is exactly what a
market-neutral pairs trade exploits.

Pipeline:
  1. Engle-Granger two-step test on every pair's log-price series.
  2. For the most cointegrated pairs, estimate the hedge ratio via OLS
     and build the (stationary, if cointegrated) spread.
  3. Z-score the spread on a rolling window; trade mean-reversion of the
     z-score (entry at |z|>2, exit at |z|<0.5) with a transaction cost.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import combinations
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS

PLOT_STYLE  = "seaborn-v0_8-darkgrid"
ANN         = 365   # crypto trades 24/7
ZWINDOW     = 40
ENTRY_Z     = 2.0
EXIT_Z      = 0.5
COST_RATE   = 0.001
PVAL_CUTOFF = 0.05
TOP_N_TRADE = 4


# ============================================================
# Cointegration Screen
# ============================================================
def test_all_pairs(log_prices, names):
    results = []
    for i, j in combinations(range(len(names)), 2):
        a, b = log_prices[:, i], log_prices[:, j]
        try:
            score, pvalue, _ = coint(a, b)
        except Exception:
            continue
        results.append({"A": names[i], "B": names[j], "score": float(score), "pvalue": float(pvalue)})
    return sorted(results, key=lambda r: r["pvalue"])


# ============================================================
# Spread + Signal Construction
# ============================================================
def compute_spread(price_a, price_b, method="tls"):
    """
    OLS regresses A on B, i.e. assumes all the noise is in price_a and
    price_b is known exactly — there's no reason to trust one leg's price
    over the other's, so that asymmetry biases the hedge ratio toward
    zero (attenuation bias) whenever both legs carry noise, which they
    always do. Total least squares (orthogonal regression) fits the line
    minimizing PERPENDICULAR distance to both series instead of vertical
    distance to price_a alone.

    TLS via SVD: center both series and stack them as columns of M =
    [price_b_c, price_a_c]. The right singular vector for the SMALLEST
    singular value, [n1, n2], is the direction in which the (centered)
    data has least variance — i.e. the normal to the best-fit line — so
    points on the line satisfy n1*price_b_c + n2*price_a_c = 0, giving
    beta = -n1/n2.
    """
    a_mean, b_mean = np.mean(price_a), np.mean(price_b)
    if method == "tls":
        M = np.column_stack([price_b - b_mean, price_a - a_mean])
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        n1, n2 = Vt[-1]
        beta = -n1 / n2 if abs(n2) > 1e-12 else np.inf
        alpha = a_mean - beta * b_mean
    else:  # "ols" — kept for comparison
        X = np.column_stack([np.ones(len(price_b)), price_b])
        beta_hat, *_ = np.linalg.lstsq(X, price_a, rcond=None)
        alpha, beta = beta_hat
    spread = price_a - (alpha + beta * price_b)
    return spread, float(alpha), float(beta)


def zscore_signals(spread, window=ZWINDOW, entry=ENTRY_Z, exit_z=EXIT_Z):
    s = pd.Series(spread)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std().replace(0, np.nan)
    z = (s - roll_mean) / roll_std

    position = np.zeros(len(z))
    pos = 0
    for t in range(len(z)):
        zt = z.iloc[t]
        if np.isnan(zt):
            position[t] = pos
            continue
        if pos == 0:
            if zt > entry:
                pos = -1   # spread too rich -> short A, long B
            elif zt < -entry:
                pos = 1    # spread too cheap -> long A, short B
        elif abs(zt) < exit_z:
            pos = 0
        position[t] = pos
    return z.values, position


def backtest_pair(spread, position, cost_rate=COST_RATE):
    d_spread = np.diff(spread, prepend=spread[0])
    gross_pnl = position * d_spread
    trades = np.abs(np.diff(position, prepend=0.0))
    spread_scale = max(np.std(spread), 1e-8)
    costs = trades * cost_rate * spread_scale
    net_pnl = gross_pnl - costs
    equity = np.cumsum(net_pnl)

    n_trades = int(np.sum(trades > 0) / 2)  # entries+exits -> round trips
    sharpe = float(np.mean(net_pnl) / max(np.std(net_pnl), 1e-10) * np.sqrt(ANN))
    return {"equity": equity, "net_pnl": net_pnl, "n_trades": n_trades,
            "sharpe": sharpe, "total_pnl": float(equity[-1])}


# ============================================================
# Plotting
# ============================================================
def plot_pairs_screen(results, names):
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, max(4, len(results[:20]) * 0.3)))
    top = results[:20]
    labels = [f"{r['A']}/{r['B']}" for r in top]
    pvals = [r["pvalue"] for r in top]
    colors = ["forestgreen" if p < PVAL_CUTOFF else "steelblue" for p in pvals]
    ax.barh(labels[::-1], pvals[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(PVAL_CUTOFF, color="red", ls="--", lw=1, label=f"p={PVAL_CUTOFF}")
    ax.set_xlabel("Engle-Granger p-value (lower = stronger cointegration)")
    ax.set_title(f"Cointegration Screen — {len(results)} pairs tested, top 20 shown", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def plot_pair_trade(pair_name, dates, spread, z, position, bt):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(3, 1, figure=fig, hspace=0.4)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(dates, spread, color="steelblue", lw=1)
    ax0.set_title(f"{pair_name} — Spread (log-price, TLS hedge ratio)", fontsize=10)
    ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(dates, z, color="darkorange", lw=1)
    ax1.axhline(ENTRY_Z, color="red", ls="--", lw=0.8)
    ax1.axhline(-ENTRY_Z, color="red", ls="--", lw=0.8)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.fill_between(dates, z, 0, where=(position != 0), alpha=0.15, color="green")
    ax1.set_title(f"Z-Score (entry=±{ENTRY_Z}, exit=±{EXIT_Z}) — shaded = in position", fontsize=10)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(dates, bt["equity"], color="forestgreen", lw=1.3)
    ax2.set_title(f"Cumulative P&L (spread units)  |  Sharpe={bt['sharpe']:.2f}  "
                  f"trades={bt['n_trades']}  net={bt['total_pnl']:+.3f}", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Pairs Trade — {pair_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("STATISTICAL ARBITRAGE — COINTEGRATION & PAIRS TRADING")
    print("Engle-Granger two-step test + z-score mean-reversion")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    names = list(assets_data.keys())
    if len(names) < 3:
        print("\n  Need >= 3 assets for a meaningful pairs screen — skipping.")
        return

    min_len = min(len(df) for df in assets_data.values())
    log_prices = np.column_stack([
        np.log(assets_data[n]["close"].values[-min_len:]) for n in names
    ])
    dates = assets_data[names[0]].index[-min_len:]
    print(f"\n  Aligned prices: {min_len} days x {len(names)} assets  "
          f"({len(names)*(len(names)-1)//2} pairs to test)\n")

    results = test_all_pairs(log_prices, names)
    n_sig = sum(1 for r in results if r["pvalue"] < PVAL_CUTOFF)
    print(f"  {n_sig}/{len(results)} pairs cointegrated at p<{PVAL_CUTOFF}\n")
    print(f"  {'Pair':<20}{'p-value':>10}{'EG stat':>10}")
    for r in results[:10]:
        flag = "  <<" if r["pvalue"] < PVAL_CUTOFF else ""
        print(f"  {r['A']+'/'+r['B']:<20}{r['pvalue']:>10.4f}{r['score']:>10.3f}{flag}")

    plot_pairs_screen(results, names)

    tradeable = [r for r in results if r["pvalue"] < PVAL_CUTOFF][:TOP_N_TRADE]
    if not tradeable:
        print(f"\n  No pairs cointegrated at p<{PVAL_CUTOFF} — nothing to backtest.")
        print("\nPairs trading analysis complete.")
        return

    print(f"\n{'='*65}")
    print(f"BACKTESTING TOP {len(tradeable)} COINTEGRATED PAIRS")
    print(f"{'='*65}")
    summary = []
    for r in tradeable:
        i, j = names.index(r["A"]), names.index(r["B"])
        pa, pb = log_prices[:, i], log_prices[:, j]
        spread, alpha, beta = compute_spread(pa, pb, method="tls")
        _, _, beta_ols = compute_spread(pa, pb, method="ols")
        z, position = zscore_signals(spread)
        bt = backtest_pair(spread, position)

        pair_name = f"{r['A']}/{r['B']}"
        print(f"  {pair_name:<15} hedge_ratio(beta)={beta:+.3f} (OLS={beta_ols:+.3f})  "
              f"p={r['pvalue']:.4f}  "
              f"trades={bt['n_trades']:>3}  sharpe={bt['sharpe']:>6.2f}  "
              f"net_pnl={bt['total_pnl']:+.4f}")

        summary.append({"Pair": pair_name, "PValue": f"{r['pvalue']:.4f}",
                         "Beta(TLS)": f"{beta:.3f}", "Beta(OLS)": f"{beta_ols:.3f}",
                         "Trades": bt["n_trades"],
                         "Sharpe": f"{bt['sharpe']:.2f}", "NetPnL": f"{bt['total_pnl']:+.4f}"})

        plot_pair_trade(pair_name, dates, spread, z, position, bt)

    print("\n" + "=" * 65)
    print("PAIRS TRADING SUMMARY")
    print("=" * 65)
    print(pd.DataFrame(summary).to_string(index=False))

    print("\nPairs trading analysis complete.")


if __name__ == "__main__":
    main()
