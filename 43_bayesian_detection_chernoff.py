"""
================================================================
Script 43 — Detection Theory I: Bayesian Detection, Chernoff Bound
================================================================
Covers: Bayesian detection, the Chernoff bound.

Frames "is the next bar an up-move or a down-move" as a formal binary
hypothesis test -- H0: return ~ N(-delta, sigma^2) (down regime) vs
H1: return ~ N(+delta, sigma^2) (up regime), where delta is the
ticker's own estimated mean-return magnitude and sigma^2 its variance --
rather than an ad hoc confidence threshold like Script 25's. The prior
P(up) is Script 31's Beta-Binomial win-rate posterior mean, not a made-
up number, so this reuses a real Bayesian estimate rather than assuming
pi0=pi1=0.5.

  1. BAYESIAN DETECTION -- the minimum-Bayes-risk decision rule for
     unequal costs (missing an up-move costs more than a false alarm,
     or vice versa -- a real, tunable asymmetry) is a likelihood-ratio
     test against a threshold set by the priors and costs, not 0.5. The
     resulting exact Bayes risk is computed in closed form (both
     hypotheses are Gaussian).
  2. CHERNOFF BOUND -- for equal-variance Gaussians the Chernoff-optimal
     parameter is provably s*=1/2, giving the Bhattacharyya bound
     sqrt(pi0*pi1)*exp(-(mu1-mu0)^2/(8 sigma^2)) as an upper bound on the
     0-1-loss Bayes error probability. Checked as a hard inequality
     (bound >= exact error) for every ticker, not just asserted --
     because the exact error is also closed-form here (Gaussian
     hypotheses), both sides of the inequality are computable exactly.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from importlib import import_module as _im
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

_r31 = _im("31_bayesian_fundamentals")
beta_binomial_winrate = _r31.beta_binomial_winrate

PLOT_STYLE = "seaborn-v0_8-darkgrid"
COST_FALSE_ALARM = 1.0    # C10: cost of declaring "up" when truly "down"
COST_MISS = 2.0           # C01: cost of declaring "down" when truly "up" -- asymmetric on purpose


# ============================================================
# Bayes-Optimal Likelihood-Ratio Threshold
# ============================================================
def bayes_optimal_threshold(mu0, mu1, sigma2, pi0, pi1, c10, c01):
    """
    Minimum-Bayes-risk test: decide H1 iff likelihood ratio exceeds
    eta = (pi0*c10)/(pi1*c01). For Gaussian H0/H1 with common variance,
    this reduces to a simple threshold on x.
    """
    eta = (pi0 * c10) / (pi1 * c01)
    threshold = (2 * sigma2 * np.log(eta) + mu1 ** 2 - mu0 ** 2) / (2 * (mu1 - mu0))
    return float(threshold), float(eta)


def bayes_risk_and_errors(threshold, mu0, mu1, sigma2, pi0, pi1, c10, c01):
    sigma = np.sqrt(sigma2)
    p_fa = 1 - stats.norm.cdf((threshold - mu0) / sigma)     # P(decide H1 | H0)
    p_miss = stats.norm.cdf((threshold - mu1) / sigma)       # P(decide H0 | H1)
    bayes_risk = pi0 * c10 * p_fa + pi1 * c01 * p_miss
    error_01 = pi0 * p_fa + pi1 * p_miss                     # 0-1 loss error probability
    return {"p_fa": float(p_fa), "p_miss": float(p_miss),
            "bayes_risk": float(bayes_risk), "error_01": float(error_01)}


# ============================================================
# Chernoff Bound (Bhattacharyya bound, s*=1/2 for equal-variance Gaussians)
# ============================================================
def chernoff_bound_gaussian(mu0, mu1, sigma2, pi0, pi1):
    return float(np.sqrt(pi0 * pi1) * np.exp(-(mu1 - mu0) ** 2 / (8 * sigma2)))


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, mu0, mu1, sigma2, pi0, pi1, thr_asym, thr_01, res_asym, res_01, chernoff):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    sigma = np.sqrt(sigma2)

    # [0,0] H0/H1 densities with both thresholds
    ax0 = fig.add_subplot(gs[0, 0])
    r = np.linspace(mu0 - 4 * sigma, mu1 + 4 * sigma, 400)
    ax0.plot(r, stats.norm.pdf(r, mu0, sigma), color="steelblue", lw=1.8, label=f"H0: N({mu0:+.4f}, sigma^2)")
    ax0.plot(r, stats.norm.pdf(r, mu1, sigma), color="crimson", lw=1.8, label=f"H1: N({mu1:+.4f}, sigma^2)")
    ax0.axvline(thr_01, color="gray", lw=1.2, ls="--", label="0-1 loss threshold")
    ax0.axvline(thr_asym, color="darkorange", lw=1.2, ls=":", label="asymmetric-cost threshold")
    ax0.legend(fontsize=7.5)
    ax0.set_title(f"{ticker} — Bayesian Detection: H0 vs H1", fontsize=10)
    ax0.grid(alpha=0.3)

    # [0,1] Bayes risk vs Chernoff bound
    ax1 = fig.add_subplot(gs[0, 1])
    names = ["Exact error\n(0-1 loss)", "Chernoff bound\n(Bhattacharyya)"]
    vals = [res_01["error_01"], chernoff]
    colors = ["steelblue", "crimson"]
    ax1.bar(names, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("probability")
    ax1.set_title(f"Chernoff Bound {'>= exact error: OK' if chernoff >= res_01['error_01'] else 'VIOLATED -- BUG'}",
                  fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # [1,0] Cost sensitivity: threshold vs cost ratio
    ax2 = fig.add_subplot(gs[1, 0])
    cost_ratios = np.logspace(-1, 1, 60)
    thresholds = [bayes_optimal_threshold(mu0, mu1, sigma2, pi0, pi1, COST_FALSE_ALARM, COST_FALSE_ALARM / cr)[0]
                  for cr in cost_ratios]
    ax2.plot(cost_ratios, thresholds, color="darkorange", lw=1.8)
    ax2.axvline(COST_MISS / COST_FALSE_ALARM, color="crimson", lw=1.0, ls="--",
                label=f"used C01/C10={COST_MISS/COST_FALSE_ALARM:.1f}")
    ax2.set_xscale("log")
    ax2.legend(fontsize=8)
    ax2.set_xlabel("cost ratio C01/C10"); ax2.set_ylabel("decision threshold")
    ax2.set_title("Threshold Sensitivity to Cost Asymmetry", fontsize=10)
    ax2.grid(alpha=0.3)

    # [1,1] Summary table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Prior P(up) [Script 31 posterior]", f"{pi1:.3f}"],
        ["Asymmetric-cost threshold", f"{thr_asym:+.5f}"],
        ["Asymmetric Bayes risk", f"{res_asym['bayes_risk']:.4f}"],
        ["0-1 loss threshold", f"{thr_01:+.5f}"],
        ["0-1 loss exact error", f"{res_01['error_01']:.4f}"],
        ["Chernoff (Bhattacharyya) bound", f"{chernoff:.4f}"],
        ["Bound >= exact error?", "YES" if chernoff >= res_01["error_01"] else "NO -- BUG"],
    ]
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    fig.suptitle(f"{ticker} — Bayesian Detection + Chernoff Bound", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("DETECTION THEORY I — BAYESIAN DETECTION, CHERNOFF BOUND")
    print(f"Asymmetric costs: C(false alarm)={COST_FALSE_ALARM}  C(miss)={COST_MISS}")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            print(f"\n  {ticker}: skipped -- need >= 60 return obs")
            continue

        delta = float(np.abs(returns.mean()))
        sigma2 = float(returns.var(ddof=1))
        mu0, mu1 = -delta, +delta

        wr = beta_binomial_winrate(returns)
        pi1, pi0 = wr["mean"], 1 - wr["mean"]

        thr_asym, eta_asym = bayes_optimal_threshold(mu0, mu1, sigma2, pi0, pi1, COST_FALSE_ALARM, COST_MISS)
        res_asym = bayes_risk_and_errors(thr_asym, mu0, mu1, sigma2, pi0, pi1, COST_FALSE_ALARM, COST_MISS)

        thr_01, _ = bayes_optimal_threshold(mu0, mu1, sigma2, pi0, pi1, 1.0, 1.0)
        res_01 = bayes_risk_and_errors(thr_01, mu0, mu1, sigma2, pi0, pi1, 1.0, 1.0)

        chernoff = chernoff_bound_gaussian(mu0, mu1, sigma2, pi0, pi1)
        bound_holds = chernoff >= res_01["error_01"]

        latest_return = float(returns[-1])
        decision = "UP (H1)" if latest_return > thr_asym else "DOWN (H0)"

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  Prior P(up)={pi1:.3f}  delta={delta:.5f}  sigma={np.sqrt(sigma2):.5f}")
        print(f"  Asymmetric-cost threshold={thr_asym:+.5f}  Bayes risk={res_asym['bayes_risk']:.4f}  "
              f"-> latest return {latest_return:+.5f} => {decision}")
        print(f"  0-1 loss: threshold={thr_01:+.5f}  exact error={res_01['error_01']:.4f}")
        print(f"  Chernoff (Bhattacharyya) bound={chernoff:.4f}  "
              f"{'>= exact error: OK' if bound_holds else 'VIOLATED -- BUG'}")

        summary.append({
            "Ticker": ticker, "PriorPUp": f"{pi1:.3f}",
            "AsymThreshold": f"{thr_asym:+.5f}", "AsymBayesRisk": f"{res_asym['bayes_risk']:.4f}",
            "ExactError01": f"{res_01['error_01']:.4f}", "ChernoffBound": f"{chernoff:.4f}",
            "BoundHolds": "OK" if bound_holds else "FAILED",
        })

        plot_dashboard(ticker, mu0, mu1, sigma2, pi0, pi1, thr_asym, thr_01, res_asym, res_01, chernoff)

    if summary:
        print("\n" + "=" * 70)
        print("BAYESIAN DETECTION / CHERNOFF BOUND SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        n_ok = sum(1 for s in summary if s["BoundHolds"] == "OK")
        print(f"\n  Chernoff bound holds for {n_ok}/{len(summary)} tickers.")

    print("\nBayesian detection / Chernoff bound analysis complete.")


if __name__ == "__main__":
    main()
