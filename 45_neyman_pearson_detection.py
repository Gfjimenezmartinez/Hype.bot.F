"""
================================================================
Script 45 — Detection Theory III: Frequentist Detection (Neyman-Pearson,
Composite Hypothesis Testing)
================================================================
Covers: frequentist detection, the Neyman-Pearson test for simple
hypotheses, composite hypothesis testing. Closes out this suite's
detection-theory cluster.

  1. NEYMAN-PEARSON -- simple H0 (mu=0, "no drift") vs simple H1
     (mu=mu1, a known positive drift), sigma estimated from data. The NP
     lemma says the likelihood-ratio test (here, a one-sided threshold
     on x) uniquely maximizes detection power at any FIXED false-alarm
     rate -- not just asserted: checked directly against an alternative
     decision region (a two-sided |x|>c region, tuned to have the exact
     same false-alarm probability) that the lemma says must have equal
     or lower power. Frames "is there a real move" with an explicit,
     controllable false-alarm rate, instead of Script 25's ad hoc
     confidence threshold.
  2. COMPOSITE HYPOTHESIS TESTING (GLRT) -- H0: mu=0 vs the composite
     H1: mu != 0 (unknown sign and magnitude), sigma also unknown. The
     generalized likelihood ratio test -- plug in the MLE for every free
     parameter under each hypothesis -- reduces EXACTLY to the ordinary
     one-sample t-statistic here, which is worth deriving explicitly
     rather than just calling `ttest_1samp` as a black box (as Script
     44's second test family already did): it's the same test, now
     understood as a GLRT instead of a "textbook stats" recipe.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

PLOT_STYLE = "seaborn-v0_8-darkgrid"
ALPHA_GRID = np.linspace(0.001, 0.5, 60)
ALPHA_DEFAULT = 0.05


# ============================================================
# 1. Neyman-Pearson Test (simple H0: mu=0 vs simple H1: mu=mu1)
# ============================================================
def np_threshold_and_power(mu1, sigma, alpha):
    """NP-optimal one-sided threshold at false-alarm rate alpha, and the
    resulting detection power (probability of correctly declaring H1)."""
    threshold = sigma * stats.norm.ppf(1 - alpha)
    power = float(stats.norm.sf((threshold - mu1) / sigma))
    return float(threshold), power


def alternative_two_sided_power(mu1, sigma, alpha):
    """
    A DIFFERENT decision region with the SAME false-alarm probability
    alpha (split equally in both tails: |x| > c), which is NOT of the
    likelihood-ratio form for this one-sided detection problem. The
    Neyman-Pearson lemma guarantees this must have power <= the NP
    one-sided test at every alpha -- checked, not assumed.
    """
    c = sigma * stats.norm.ppf(1 - alpha / 2)
    power = float(stats.norm.sf((c - mu1) / sigma) + stats.norm.cdf((-c - mu1) / sigma))
    return float(c), power


def roc_curve(mu1, sigma, alpha_grid=ALPHA_GRID):
    powers_np = np.array([np_threshold_and_power(mu1, sigma, a)[1] for a in alpha_grid])
    powers_alt = np.array([alternative_two_sided_power(mu1, sigma, a)[1] for a in alpha_grid])
    return powers_np, powers_alt


# ============================================================
# 2. Composite Hypothesis Test — GLRT reduces to the one-sample t-test
# ============================================================
def glrt_composite_test(returns):
    """
    GLRT for H0: mu=0 vs composite H1: mu != 0, sigma unknown under
    both: Lambda = sup_mu p(x|mu,sigma_hat_1)/p(x|0,sigma_hat_0), with
    sigma also profiled out via its own MLE under each hypothesis. This
    reduces exactly to T = xbar/(s/sqrt(n)) ~ t_(n-1) under H0 -- the
    ordinary one-sample t-statistic, derived here as a GLRT rather than
    invoked as a canned test.
    """
    n = len(returns)
    xbar, s = returns.mean(), returns.std(ddof=1)
    t_stat = xbar / (s / np.sqrt(n))
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "n": n}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, mu1, sigma, powers_np, powers_alt, glrt):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # [0,0] ROC curve: NP vs alternative
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(ALPHA_GRID, powers_np, color="crimson", lw=1.8, label="NP-optimal (one-sided LRT)")
    ax0.plot(ALPHA_GRID, powers_alt, color="steelblue", lw=1.8, ls="--", label="Alternative (two-sided, same P_FA)")
    ax0.plot([0, 1], [0, 1], color="gray", lw=0.8, ls=":", label="chance line")
    ax0.set_xlabel("P(false alarm)"); ax0.set_ylabel("P(detection)")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — ROC: Neyman-Pearson vs Alternative Region", fontsize=10)
    ax0.grid(alpha=0.3)

    # [0,1] Power gap (NP - alternative) across alpha
    ax1 = fig.add_subplot(gs[0, 1])
    gap = powers_np - powers_alt
    ax1.plot(ALPHA_GRID, gap, color="darkorange", lw=1.8)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.fill_between(ALPHA_GRID, 0, gap, alpha=0.15, color="darkorange")
    ax1.set_xlabel("P(false alarm)"); ax1.set_ylabel("power advantage of NP test")
    ax1.set_title(f"NP Optimality Margin  (min={gap.min():+.4f}, must be >= 0)", fontsize=10)
    ax1.grid(alpha=0.3)

    # [1,0] H0/H1 densities with NP threshold at alpha=0.05
    ax2 = fig.add_subplot(gs[1, 0])
    thr, power = np_threshold_and_power(mu1, sigma, ALPHA_DEFAULT)
    r = np.linspace(-4 * sigma, mu1 + 4 * sigma, 400)
    ax2.plot(r, stats.norm.pdf(r, 0, sigma), color="steelblue", lw=1.8, label="H0: N(0, sigma^2)")
    ax2.plot(r, stats.norm.pdf(r, mu1, sigma), color="crimson", lw=1.8, label=f"H1: N({mu1:+.4f}, sigma^2)")
    ax2.axvline(thr, color="black", lw=1.2, ls="--", label=f"NP threshold (alpha={ALPHA_DEFAULT})")
    ax2.legend(fontsize=8)
    ax2.set_title(f"NP Test at alpha={ALPHA_DEFAULT}: power={power:.3f}", fontsize=10)
    ax2.grid(alpha=0.3)

    # [1,1] Summary
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["NP threshold (alpha=0.05)", f"{thr:+.5f}"],
        ["NP power at alpha=0.05", f"{power:.4f}"],
        ["Min power gap (NP vs alt) over alpha grid", f"{gap.min():+.4f}"],
        ["NP dominates for all alpha?", "YES" if gap.min() >= -1e-9 else "NO -- BUG"],
        ["GLRT t-statistic", f"{glrt['t_stat']:+.4f}"],
        ["GLRT p-value (composite H1: mu!=0)", f"{glrt['p_value']:.4f}"],
        ["Reject H0 at alpha=0.05?", "YES" if glrt["p_value"] < 0.05 else "NO"],
    ]
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    fig.suptitle(f"{ticker} — Neyman-Pearson Detection + Composite Hypothesis Test (GLRT)",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("DETECTION THEORY III — NEYMAN-PEARSON, COMPOSITE HYPOTHESIS TESTING")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []
    worst_gap_overall = np.inf

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            print(f"\n  {ticker}: skipped -- need >= 60 return obs")
            continue

        mu1 = float(np.abs(returns.mean()))
        sigma = float(returns.std(ddof=1))

        powers_np, powers_alt = roc_curve(mu1, sigma)
        gap = powers_np - powers_alt
        worst_gap_overall = min(worst_gap_overall, gap.min())

        thr05, power05 = np_threshold_and_power(mu1, sigma, ALPHA_DEFAULT)
        glrt = glrt_composite_test(returns)

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  NP test (alpha=0.05): threshold={thr05:+.5f}  power={power05:.4f}")
        print(f"  NP optimality margin over alpha grid: min={gap.min():+.5f}  "
              f"{'OK (NP dominates everywhere)' if gap.min() >= -1e-9 else 'VIOLATED -- BUG'}")
        print(f"  GLRT (composite H1: mu!=0): t={glrt['t_stat']:+.4f}  p={glrt['p_value']:.4f}  "
              f"{'reject H0' if glrt['p_value'] < 0.05 else 'fail to reject H0'} at alpha=0.05")

        summary.append({
            "Ticker": ticker, "NP_Power@0.05": f"{power05:.4f}",
            "MinOptimalityMargin": f"{gap.min():+.5f}",
            "GLRT_t": f"{glrt['t_stat']:+.4f}", "GLRT_p": f"{glrt['p_value']:.4f}",
            "RejectH0": "Y" if glrt["p_value"] < 0.05 else "N",
        })

        plot_dashboard(ticker, mu1, sigma, powers_np, powers_alt, glrt)

    if summary:
        print("\n" + "=" * 70)
        print("NEYMAN-PEARSON / GLRT SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        print(f"\n  Worst-case NP optimality margin across ALL tickers and alpha values: "
              f"{worst_gap_overall:+.5f}  "
              f"({'NP lemma verified -- optimal everywhere' if worst_gap_overall >= -1e-9 else 'VIOLATED -- BUG'})")

    print("\nNeyman-Pearson / composite hypothesis testing analysis complete.")


if __name__ == "__main__":
    main()
