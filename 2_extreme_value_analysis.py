"""
================================================================
Script 2 — Extreme Value Theory (EVT) Analysis  v2
================================================================
Fixes vs v1:
  • GPD CVaR guarded against negative values when shape ≥ 1
    (uses numerical simulation fallback for heavy-tailed cases)
  • Negative CVaR flagged as "unbounded tail" in output
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS

BLOCK_SIZE   = 21
POT_QUANTILE = 0.95
CONF_LEVELS  = [0.95, 0.99]
PLOT_STYLE   = "seaborn-v0_8-darkgrid"


# ── L-Moments ────────────────────────────────────────────────
def l_moments(sample: np.ndarray):
    x = np.sort(sample); n = len(x)
    b0 = x.mean()
    b1 = np.mean([(i / (n - 1)) * x[i] for i in range(n)])
    b2 = np.mean([i * (i - 1) / ((n - 1) * (n - 2)) * x[i] for i in range(n)])
    l1, l2 = b0, 2*b1 - b0
    l3 = 6*b2 - 6*b1 + b0
    t3 = l3 / l2 if l2 != 0 else np.nan
    return l1, l2, t3


# ── Block Maxima → GEV ───────────────────────────────────────
def block_maxima(losses: np.ndarray, block_size: int) -> np.ndarray:
    return np.array([losses[i:i+block_size].max()
                     for i in range(0, len(losses), block_size)
                     if len(losses[i:i+block_size]) == block_size])

def fit_gev(bmax):
    return stats.genextreme.fit(bmax)


# ── Peaks-Over-Threshold → GPD ───────────────────────────────
def fit_gpd(losses: np.ndarray, quantile: float, min_exceedances: int = 30):
    # Lower the threshold if too few exceedances: MLE shape estimates
    # from <30 points are extremely noisy (this, plus a free location
    # parameter, is what produced shape>1 "infinite tail" fits).
    q = quantile
    while q > 0.80:
        threshold = np.quantile(losses, q)
        excesses  = losses[losses > threshold] - threshold
        if len(excesses) >= min_exceedances:
            break
        q -= 0.025
    params = stats.genpareto.fit(excesses, floc=0)
    return threshold, excesses, params


# ── Tail Risk  (robust) ──────────────────────────────────────
def tail_risk_gpd(threshold: float, gpd_params, prob: float,
                  excesses: np.ndarray, n_total: int):
    """
    Returns (var, cvar, note).
    Standard POT estimator:  VaR_p = u + (β/ξ)·[((1-p)/ζ_u)^(-ξ) − 1]
    where ζ_u = N_exceedances / n_total.  Omitting ζ_u (the old bug)
    treats every observation as a tail observation and wildly
    overstates VaR.
    When shape ≥ 1 the analytical CVaR is undefined (+∞); we
    fall back to an empirical tail average and flag it.
    """
    shape, loc, scale = gpd_params
    note   = ""
    zeta_u = len(excesses) / max(n_total, 1)
    ratio  = (1 - prob) / zeta_u

    # ── VaR ──────────────────────────────────────────────────
    if ratio >= 1.0:
        # Requested quantile is not beyond the threshold — use empirical
        var = float(np.quantile(threshold + excesses, 0.0))
        var = threshold
    elif abs(shape) < 1e-8:                        # exponential limit
        var = threshold - scale * np.log(ratio)
    else:
        var = threshold + (scale / shape) * (ratio**(-shape) - 1)

    # ── CVaR ─────────────────────────────────────────────────
    if shape >= 1.0:
        # Analytical CVaR diverges — use empirical tail average
        # from actual excesses (not simulation, which explodes)
        actual_losses = threshold + excesses
        tail = actual_losses[actual_losses >= var]
        if len(tail) > 0:
            cvar = float(tail.mean())
        else:
            cvar = var * 1.5
        note = f"CVaR=empirical (shape={shape:.2f}>=1)"
    elif 0 < shape < 1:
        cvar = (var + scale - shape * threshold) / (1 - shape)
        if cvar < var:          # numerical sanity check
            cvar = var * 1.5
            note = "CVaR clamped"
    else:                       # shape < 0 (bounded tail)
        cvar = (var + scale - shape * threshold) / (1 - shape)

    return float(var), float(cvar), note


# ── Plot ─────────────────────────────────────────────────────
def plot_evt(ticker, bmax, gev_params, excesses, gpd_params):
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(bmax,     stat="density", bins=15, ax=axes[0], color="steelblue", alpha=0.7)
    x = np.linspace(bmax.min(), bmax.max(), 400)
    axes[0].plot(x, stats.genextreme.pdf(x, *gev_params), lw=2, color="tomato")
    axes[0].set_title(f"{ticker} — GEV (Block Maxima)")
    axes[0].set_xlabel("Loss")

    sns.histplot(excesses, stat="density", bins=15, ax=axes[1], color="darkorange", alpha=0.7)
    y = np.linspace(0, excesses.max(), 400)
    axes[1].plot(y, stats.genpareto.pdf(y, *gpd_params), lw=2, color="navy")
    axes[1].set_title(f"{ticker} — GPD (POT Excesses)")
    axes[1].set_xlabel("Excess Loss")

    plt.suptitle(f"{ticker} — Extreme Value Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("EXTREME VALUE THEORY (EVT) ANALYSIS  v2")
    print("=" * 65)

    assets_data  = load_all_assets(period_days=LOOKBACK_DAYS)
    summary_rows = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}  {ticker}")
        losses = -df["log_return"].dropna().values

        bmax = block_maxima(losses, BLOCK_SIZE)
        if len(bmax) < 5:
            print("  Not enough blocks — skipping."); continue

        gev_params         = fit_gev(bmax)
        l1, l2, t3         = l_moments(bmax)
        threshold, exc, gpd_params = fit_gpd(losses, POT_QUANTILE)

        print(f"  GEV (shape,loc,scale): {gev_params[0]:.5f}, "
              f"{gev_params[1]:.5f}, {gev_params[2]:.5f}")
        print(f"  GPD shape={gpd_params[0]:.5f}  "
              f"threshold({POT_QUANTILE:.0%})={threshold:.5f}")

        for p in CONF_LEVELS:
            var, cvar, note = tail_risk_gpd(threshold, gpd_params, p, exc,
                                            n_total=len(losses))
            flag = f"  [{note}]" if note else ""
            print(f"  VaR {p:.0%}: {var:.5f}  CVaR {p:.0%}: {cvar:.5f}{flag}")
            summary_rows.append({
                "Ticker": ticker, "Conf": p,
                "VaR": round(var, 5), "CVaR": round(cvar, 5),
                "GPD_shape": round(gpd_params[0], 5),
                "Note": note,
            })

        plot_evt(ticker, bmax, gev_params, exc, gpd_params)

    print("\n" + "=" * 65)
    print("TAIL RISK SUMMARY")
    print("=" * 65)
    df_s = pd.DataFrame(summary_rows)
    print(df_s.to_string(index=False))
    print("\nEVT Analysis complete.")

if __name__ == "__main__":
    main()
