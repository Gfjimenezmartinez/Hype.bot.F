"""
================================================================
Script 36 — Parametric Models: Mixture Models
================================================================
Covers: mixture models. (Bayesian linear/logistic regression and basis
expansions, this cluster's other topics, were Script 32; LDA is skipped
as a topic here -- there is no natural text corpus anywhere in this
price-only suite.)

Script 33 fit a Gaussian mixture the Bayesian/variational way (automatic
component pruning via ARD, no need to specify K). This script is the
classical counterpart: plain EM-fit GaussianMixture, with K chosen by
BIC over a grid -- the "traditional" way of answering the same "how many
components" question, useful as a direct contrast with Script 33's
automatic approach on the same data.

Discovered components are interpreted as return regimes and compared
against Script 15's HMM regime call for the same ticker -- a mixture
model treats each bar as i.i.d. (no time-ordering), so this comparison
also surfaces something real: does ignoring persistence still recover
similar regimes, or does the HMM's temporal structure matter?

Also computes VaR under the fitted mixture density vs. a single-Gaussian
assumption -- a return distribution with real regime structure (calm
bars + crisis bars pooled together) has fatter/different tails than a
single Normal fit to the same data, and the mixture-based VaR should
show it.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from importlib import import_module as _im
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

_r15 = _im("15_regime_detection")
detect_regime = _r15.detect_regime

PLOT_STYLE = "seaborn-v0_8-darkgrid"
K_RANGE = range(1, 5)
VAR_ALPHA = 0.05


# ============================================================
# Model Selection — BIC over K
# ============================================================
def fit_best_gmm(returns, k_range=K_RANGE, random_state=42):
    x = np.asarray(returns).reshape(-1, 1)
    bics, models = [], []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=3).fit(x)
        bics.append(gmm.bic(x))
        models.append(gmm)
    best_idx = int(np.argmin(bics))
    return {"best_k": list(k_range)[best_idx], "best_model": models[best_idx],
            "bics": np.array(bics), "k_range": list(k_range)}


# ============================================================
# Regime Persistence — bar-to-bar stickiness of hard-assigned labels
# ============================================================
def regime_persistence(labels, n_components):
    persistence = {}
    for k in range(n_components):
        in_k = labels == k
        if in_k[:-1].sum() == 0:
            persistence[k] = np.nan
            continue
        stayed = np.logical_and(in_k[:-1], in_k[1:]).sum()
        persistence[k] = stayed / in_k[:-1].sum()
    return persistence


# ============================================================
# Mixture VaR vs Single-Gaussian VaR
# ============================================================
def mixture_var(gmm, alpha=VAR_ALPHA, grid_pad=8):
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    lo = means.min() - grid_pad * stds.max()
    hi = means.max() + grid_pad * stds.max()
    grid = np.linspace(lo, hi, 20000)
    cdf = np.zeros_like(grid)
    for w, m, s in zip(weights, means, stds):
        cdf += w * stats.norm.cdf(grid, m, s)
    idx = np.searchsorted(cdf, alpha)
    return float(grid[min(idx, len(grid) - 1)])


def single_gaussian_var(returns, alpha=VAR_ALPHA):
    return float(stats.norm.ppf(alpha, returns.mean(), returns.std()))


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, returns, fit_result, labels, persistence, mix_var, gauss_var, regime_name):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    gmm = fit_result["best_model"]

    # [0,0] BIC curve over K
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(fit_result["k_range"], fit_result["bics"], "o-", color="steelblue", lw=1.6)
    ax0.axvline(fit_result["best_k"], color="crimson", lw=1.0, ls="--", label=f"best K={fit_result['best_k']}")
    ax0.legend(fontsize=8)
    ax0.set_xlabel("K (components)"); ax0.set_ylabel("BIC (lower = better)")
    ax0.set_title(f"{ticker} — Model Selection by BIC", fontsize=10)
    ax0.grid(alpha=0.3)

    # [0,1] Return histogram + mixture components
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(returns, bins=40, density=True, color="lightsteelblue", alpha=0.6, label="Returns")
    r = np.linspace(returns.min(), returns.max(), 300)
    colors = plt.cm.tab10(np.linspace(0, 1, fit_result["best_k"]))
    total = np.zeros_like(r)
    for k in range(fit_result["best_k"]):
        m, s, w = gmm.means_[k, 0], np.sqrt(gmm.covariances_[k, 0, 0]), gmm.weights_[k]
        dens = w * stats.norm.pdf(r, m, s)
        total += dens
        ax1.plot(r, dens, color=colors[k], lw=1.4, label=f"Comp {k} (w={w:.2f})")
    ax1.plot(r, total, color="black", lw=1.8, ls="--", label="Mixture total")
    ax1.legend(fontsize=7)
    ax1.set_title(f"EM Gaussian Mixture (K={fit_result['best_k']})  |  HMM regime: {regime_name}", fontsize=9.5)
    ax1.grid(alpha=0.3)

    # [1,0] Regime persistence bar
    ax2 = fig.add_subplot(gs[1, 0])
    ks = list(persistence.keys())
    vals = [persistence[k] * 100 if not np.isnan(persistence[k]) else 0 for k in ks]
    ax2.bar([f"Comp {k}" for k in ks], vals, color=colors[:len(ks)], alpha=0.85)
    ax2.axhline(50, color="gray", lw=0.8, ls="--", label="coin-flip (no persistence)")
    ax2.legend(fontsize=7)
    ax2.set_ylabel("P(stay in same component next bar) %")
    ax2.set_title("Regime Persistence (i.i.d. clustering, no time-structure)", fontsize=9.5)
    ax2.grid(axis="y", alpha=0.3)

    # [1,1] VaR comparison
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Best K (BIC)", f"{fit_result['best_k']}"],
        ["Mixture VaR (5%)", f"{mix_var:+.5f}"],
        ["Single-Gaussian VaR (5%)", f"{gauss_var:+.5f}"],
        ["Difference", f"{(mix_var - gauss_var):+.5f}"],
        ["HMM regime (current)", regime_name],
    ]
    for k in ks:
        p = persistence[k]
        rows.append([f"Comp {k} persistence", f"{p:.1%}" if not np.isnan(p) else "n/a"])
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    fig.suptitle(f"{ticker} — Parametric Models: Gaussian Mixture (EM)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("PARAMETRIC MODELS — MIXTURE MODELS (EM-fit Gaussian Mixture)")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            print(f"\n  {ticker}: skipped -- need >= 60 return obs")
            continue

        fit_result = fit_best_gmm(returns)
        gmm = fit_result["best_model"]
        labels = gmm.predict(returns.reshape(-1, 1))
        persistence = regime_persistence(labels, fit_result["best_k"])

        mix_var = mixture_var(gmm)
        gauss_var = single_gaussian_var(returns)

        try:
            _, regime_name, _ = detect_regime(df)
        except Exception:
            regime_name = "n/a"

        avg_persist = np.nanmean(list(persistence.values()))

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  Best K (BIC): {fit_result['best_k']}  "
              f"(BICs: {np.round(fit_result['bics'], 1)})")
        print(f"  Regime persistence: " +
              "  ".join(f"comp{k}={v:.1%}" for k, v in persistence.items() if not np.isnan(v)) +
              f"  (avg={avg_persist:.1%})")
        print(f"  VaR(5%): mixture={mix_var:+.5f}  single-Gaussian={gauss_var:+.5f}  "
              f"diff={mix_var-gauss_var:+.5f}")
        print(f"  HMM regime (current): {regime_name}")

        summary.append({
            "Ticker": ticker, "BestK": fit_result["best_k"],
            "AvgPersistence": f"{avg_persist:.1%}",
            "MixtureVaR": f"{mix_var:+.5f}", "GaussianVaR": f"{gauss_var:+.5f}",
            "VaRDiff": f"{mix_var-gauss_var:+.5f}", "HMM_Regime": regime_name,
        })

        plot_dashboard(ticker, returns, fit_result, labels, persistence, mix_var, gauss_var, regime_name)

    if summary:
        print("\n" + "=" * 70)
        print("MIXTURE MODELS SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        k_dist = pd.Series([s["BestK"] for s in summary]).value_counts().sort_index()
        print(f"\n  K distribution across tickers:\n{k_dist.to_string()}")

    print("\nMixture models analysis complete.")


if __name__ == "__main__":
    main()
