"""
================================================================
Script 44 — Detection Theory II: Multiple Hypothesis Testing
================================================================
Covers: multiple hypothesis testing.

Every earlier script that ran a statistical test across the universe
(Script 9's ARCH-LM test, any one-sample mean test, etc.) did it once
per ticker at a raw alpha=0.05 with no correction -- so purely by
chance, roughly one "significant" result would be expected across the
universe even if NOTHING were actually significant anywhere. This
script applies Bonferroni and Benjamini-Hochberg (BH) false-discovery-
rate correction to two test families computed across every ticker:

  1. ARCH-LM test (same test, same statsmodels call, Script 9 already
     uses per-ticker) -- "does this ticker show volatility clustering?"
  2. One-sample t-test for mean return != 0 -- "does this ticker
     have a statistically significant nonzero average return?"

For each family, reports how many tickers are "significant" at raw
alpha=0.05 vs. after Bonferroni vs. after BH correction. The three
counts are guaranteed to satisfy Bonferroni-rejections <= BH-rejections
<= raw-rejections (Bonferroni is the most conservative correction, BH
sits in between, uncorrected is the most permissive) -- checked
directly, not assumed.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS

PLOT_STYLE = "seaborn-v0_8-darkgrid"
ALPHA = 0.05


# ============================================================
# Test Families
# ============================================================
def arch_lm_pvalues(assets_data):
    tickers, pvals = [], []
    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            continue
        stat, pval, _, _ = het_arch(returns)
        tickers.append(ticker)
        pvals.append(pval)
    return tickers, np.array(pvals)


def mean_return_ttest_pvalues(assets_data):
    tickers, pvals = [], []
    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            continue
        _, pval = stats.ttest_1samp(returns, popmean=0.0)
        tickers.append(ticker)
        pvals.append(pval)
    return tickers, np.array(pvals)


# ============================================================
# Multiple-Testing Correction (reuses statsmodels' multipletests --
# a correctly implemented, standard implementation of both procedures)
# ============================================================
def correct_pvalues(pvals, alpha=ALPHA):
    raw_reject = pvals < alpha
    bonf_reject, bonf_p, _, _ = multipletests(pvals, alpha=alpha, method="bonferroni")
    bh_reject, bh_p, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    return {"raw_reject": raw_reject, "bonf_reject": bonf_reject, "bh_reject": bh_reject,
            "bonf_p": bonf_p, "bh_p": bh_p}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(family_name, tickers, pvals, corr):
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    ax0 = axes[0]
    order = np.argsort(pvals)
    sorted_tickers = [tickers[i] for i in order]
    ax0.scatter(range(len(pvals)), pvals[order], color="steelblue", s=30, label="raw p-value")
    ax0.axhline(ALPHA, color="gray", lw=1.0, ls="--", label=f"alpha={ALPHA}")
    m = len(pvals)
    bh_line = np.arange(1, m + 1) / m * ALPHA
    ax0.plot(range(m), bh_line, color="darkorange", lw=1.2, label="BH threshold line (k/m * alpha)")
    ax0.axhline(ALPHA / m, color="crimson", lw=1.0, ls=":", label=f"Bonferroni threshold (alpha/m={ALPHA/m:.4f})")
    ax0.set_xticks(range(m))
    ax0.set_xticklabels(sorted_tickers, rotation=90, fontsize=6)
    ax0.set_ylabel("p-value")
    ax0.legend(fontsize=7)
    ax0.set_title(f"{family_name} — Sorted p-values vs Correction Thresholds", fontsize=10)
    ax0.grid(alpha=0.3)

    ax1 = axes[1]
    counts = [corr["raw_reject"].sum(), corr["bh_reject"].sum(), corr["bonf_reject"].sum()]
    names = ["Raw (alpha=0.05)", "Benjamini-Hochberg", "Bonferroni"]
    colors = ["gray", "darkorange", "crimson"]
    ax1.bar(names, counts, color=colors, alpha=0.85)
    for i, c in enumerate(counts):
        ax1.text(i, c, str(c), ha="center", va="bottom", fontsize=10)
    ax1.set_ylabel(f"# significant tickers (of {m})")
    ax1.set_title(f"{family_name} — Significant Findings by Correction Method", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Multiple Hypothesis Testing: {family_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("DETECTION THEORY II — MULTIPLE HYPOTHESIS TESTING")
    print(f"Bonferroni + Benjamini-Hochberg FDR correction (alpha={ALPHA}) across the universe")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    all_ok = True

    for family_name, test_fn in [("ARCH-LM Test (volatility clustering)", arch_lm_pvalues),
                                  ("One-Sample t-test (mean return != 0)", mean_return_ttest_pvalues)]:
        print(f"\n{'='*70}\n{family_name}\n{'='*70}")
        tickers, pvals = test_fn(assets_data)
        corr = correct_pvalues(pvals)

        n_raw = int(corr["raw_reject"].sum())
        n_bh = int(corr["bh_reject"].sum())
        n_bonf = int(corr["bonf_reject"].sum())
        ordering_ok = n_bonf <= n_bh <= n_raw

        print(f"  m={len(pvals)} tickers tested.")
        print(f"  Significant at raw alpha={ALPHA}:        {n_raw}/{len(pvals)}")
        print(f"  Significant after Benjamini-Hochberg:    {n_bh}/{len(pvals)}")
        print(f"  Significant after Bonferroni:            {n_bonf}/{len(pvals)}")
        print(f"  Ordering (Bonferroni <= BH <= raw): {'OK' if ordering_ok else 'VIOLATED -- BUG'}")
        all_ok = all_ok and ordering_ok

        table = pd.DataFrame({
            "Ticker": tickers, "RawP": np.round(pvals, 4),
            "RawSig": corr["raw_reject"], "BH_Sig": corr["bh_reject"], "Bonf_Sig": corr["bonf_reject"],
        }).sort_values("RawP")
        print("\n" + table.to_string(index=False))

        plot_dashboard(family_name, tickers, pvals, corr)

    print(f"\n{'='*70}")
    print(f"Ordering guarantee (Bonferroni <= BH <= raw) held for both test families: "
          f"{'YES' if all_ok else 'NO -- BUG'}")
    print("\nMultiple hypothesis testing analysis complete.")


if __name__ == "__main__":
    main()
