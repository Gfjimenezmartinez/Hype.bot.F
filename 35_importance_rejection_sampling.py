"""
================================================================
Script 35 — Sampling Methods II: Importance Sampling + Rejection
Sampling
================================================================
Covers: rejection sampling, importance sampling. (MCMC, this cluster's
third method, was Script 34.)

Applies both to the concrete problem Scripts 2/3 care about: how likely
is a genuinely rare tail event (e.g. a 1-in-100-bar loss)? Target
distribution is each ticker's Bayesian posterior predictive Student-t
from Script 31 (reused directly, not refit) -- so this cluster builds on
Script 31 rather than starting from scratch.

  1. REJECTION SAMPLING -- generates exact samples from the target using
     only density evaluations and an easy-to-sample envelope proposal
     (Cauchy, heavier-tailed than the near-Gaussian target for large-n
     posteriors). Framed honestly: we *could* just call scipy's t.rvs
     directly here, but rejection sampling is taught for the general
     case where only the target's density (often unnormalized) is
     available, not a sampler -- the Student-t is used because it's
     convenient and, crucially, checkable: the accepted samples are
     validated against the true distribution with a KS test.
  2. IMPORTANCE SAMPLING -- the actual efficiency payoff: naive Monte
     Carlo estimates a rare tail probability by counting how many of N
     direct draws fall below the threshold, which has huge relative
     variance when the true probability is small (few or zero draws
     land there). Importance sampling deliberately oversamples the tail
     region with a shifted proposal and reweights by the likelihood
     ratio -- unbiased, and dramatically lower variance for the same
     sample budget. Demonstrated empirically: both estimators are run
     many times at the same N, and the variance-reduction factor is
     reported directly, not just asserted.
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
bayesian_return_posterior = _r31.bayesian_return_posterior

PLOT_STYLE = "seaborn-v0_8-darkgrid"
TAIL_QUANTILE = 0.01     # genuinely rare: 1st percentile
N_SAMPLES = 2000
N_REPEATS = 200
REJ_N_SAMPLES = 5000


# ============================================================
# 1. Rejection Sampling
# ============================================================
def find_envelope_M(target_pdf, proposal_pdf, grid):
    ratios = target_pdf(grid) / np.maximum(proposal_pdf(grid), 1e-300)
    return float(np.max(ratios)) * 1.001   # tiny safety margin


def rejection_sample(target_pdf, proposal, M, n_samples, rng):
    """proposal: a frozen scipy.stats distribution (has .rvs, .pdf)."""
    accepted = []
    n_trials = 0
    batch = max(n_samples, 1000)
    while len(accepted) < n_samples:
        x = proposal.rvs(size=batch, random_state=rng)
        u = rng.uniform(size=batch)
        accept_prob = target_pdf(x) / (M * proposal.pdf(x))
        mask = u <= accept_prob
        accepted.extend(x[mask].tolist())
        n_trials += batch
    accepted = np.array(accepted[:n_samples])
    return accepted, n_trials, n_samples / n_trials


# ============================================================
# 2. Importance Sampling vs Naive Monte Carlo (tail probability)
# ============================================================
def naive_mc_tail_prob(target_frozen, threshold, n_samples, rng):
    x = target_frozen.rvs(size=n_samples, random_state=rng)
    hits = (x < threshold).astype(float)
    p_hat = hits.mean()
    se = hits.std(ddof=1) / np.sqrt(n_samples)
    return p_hat, se


def importance_sample_tail_prob(target_pdf, threshold, proposal, n_samples, rng):
    x = proposal.rvs(size=n_samples, random_state=rng)
    w = target_pdf(x) / np.maximum(proposal.pdf(x), 1e-300)
    contrib = w * (x < threshold)
    p_hat = contrib.mean()
    se = contrib.std(ddof=1) / np.sqrt(n_samples)
    return p_hat, se


def compare_estimators(target_frozen, target_pdf, threshold, proposal,
                        n_samples=N_SAMPLES, n_repeats=N_REPEATS, seed=42):
    rng = np.random.default_rng(seed)
    naive_estimates, is_estimates = [], []
    for _ in range(n_repeats):
        p_naive, _ = naive_mc_tail_prob(target_frozen, threshold, n_samples, rng)
        p_is, _ = importance_sample_tail_prob(target_pdf, threshold, proposal, n_samples, rng)
        naive_estimates.append(p_naive)
        is_estimates.append(p_is)
    naive_estimates, is_estimates = np.array(naive_estimates), np.array(is_estimates)
    true_p = target_frozen.cdf(threshold)
    var_reduction = (naive_estimates.std() / max(is_estimates.std(), 1e-12)) ** 2
    return {"naive": naive_estimates, "is": is_estimates, "true_p": true_p,
            "var_reduction": var_reduction}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, target_frozen, rej_samples, ks_pvalue, cmp_result, threshold):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # [0,0] Rejection samples vs true density
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(rej_samples, bins=40, density=True, color="lightsteelblue", alpha=0.7, label="Rejection samples")
    r = np.linspace(rej_samples.min(), rej_samples.max(), 300)
    ax0.plot(r, target_frozen.pdf(r), color="crimson", lw=1.8, label="True target density")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Rejection Sampling  (KS test p={ks_pvalue:.3f})", fontsize=10)
    ax0.grid(alpha=0.3)

    # [0,1] Estimator distributions: naive MC vs IS
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(cmp_result["naive"] * 100, bins=25, alpha=0.6, color="gray", label="Naive MC", density=True)
    ax1.hist(cmp_result["is"] * 100, bins=25, alpha=0.6, color="steelblue", label="Importance Sampling", density=True)
    ax1.axvline(cmp_result["true_p"] * 100, color="crimson", lw=1.5, ls="--", label="True P(tail)")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("estimated tail probability (%)")
    ax1.set_title(f"Estimator Spread over {N_REPEATS} repeats  "
                  f"(var reduction={cmp_result['var_reduction']:.1f}x)", fontsize=9.5)
    ax1.grid(alpha=0.3)

    # [1,0] Estimator bias/variance bar
    ax2 = fig.add_subplot(gs[1, 0])
    names = ["Naive MC", "Importance Sampling"]
    means = [cmp_result["naive"].mean() * 100, cmp_result["is"].mean() * 100]
    stds = [cmp_result["naive"].std() * 100, cmp_result["is"].std() * 100]
    ax2.bar(names, means, yerr=stds, color=["gray", "steelblue"], alpha=0.85, capsize=6)
    ax2.axhline(cmp_result["true_p"] * 100, color="crimson", lw=1.2, ls="--", label="True value")
    ax2.legend(fontsize=8)
    ax2.set_ylabel("estimated tail probability (%) +/- 1 std")
    ax2.set_title("Estimator Mean +/- Std", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # [1,1] Summary table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Tail threshold", f"{threshold:+.5f}"],
        ["True P(X < threshold)", f"{cmp_result['true_p']:.4%}"],
        ["Naive MC: mean (std)", f"{cmp_result['naive'].mean():.4%} ({cmp_result['naive'].std():.4%})"],
        ["Importance Sampling: mean (std)", f"{cmp_result['is'].mean():.4%} ({cmp_result['is'].std():.4%})"],
        ["Variance reduction factor", f"{cmp_result['var_reduction']:.1f}x"],
        ["Rejection sampling KS p-value", f"{ks_pvalue:.3f}"],
    ]
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    fig.suptitle(f"{ticker} — Sampling Methods (Rejection Sampling + Importance Sampling)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("SAMPLING METHODS II — IMPORTANCE SAMPLING + REJECTION SAMPLING")
    print(f"Tail-risk estimation at the {TAIL_QUANTILE:.0%} quantile")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []
    rng_global = np.random.default_rng(1)

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            print(f"\n  {ticker}: skipped -- need >= 60 return obs")
            continue

        post = bayesian_return_posterior(returns)
        target_frozen = stats.t(post["pred_df"], loc=post["pred_loc"], scale=post["pred_scale"])
        target_pdf = target_frozen.pdf
        threshold = float(target_frozen.ppf(TAIL_QUANTILE))

        # ── Rejection sampling: Cauchy envelope, verified against true CDF ──
        proposal_rej = stats.cauchy(loc=post["pred_loc"], scale=post["pred_scale"] * 2.5)
        grid = np.linspace(post["pred_loc"] - 15 * post["pred_scale"],
                            post["pred_loc"] + 15 * post["pred_scale"], 5000)
        M = find_envelope_M(target_pdf, proposal_rej.pdf, grid)
        rej_samples, n_trials, accept_rate = rejection_sample(
            target_pdf, proposal_rej, M, REJ_N_SAMPLES, rng_global)
        ks_stat, ks_pvalue = stats.kstest(rej_samples, target_frozen.cdf)

        # ── Importance sampling vs naive MC ──
        proposal_is = stats.norm(loc=threshold, scale=post["pred_scale"])
        cmp_result = compare_estimators(target_frozen, target_pdf, threshold, proposal_is)

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  Rejection sampling: M={M:.2f}  accept_rate={accept_rate:.1%}  "
              f"KS test vs true dist: p={ks_pvalue:.3f} "
              f"{'(OK -- matches)' if ks_pvalue > 0.05 else '(WARNING -- mismatch)'}")
        print(f"  Tail threshold ({TAIL_QUANTILE:.0%}ile): {threshold:+.5f}  "
              f"true P={cmp_result['true_p']:.4%}")
        print(f"  Naive MC:   mean={cmp_result['naive'].mean():.4%}  std={cmp_result['naive'].std():.4%}")
        print(f"  Importance: mean={cmp_result['is'].mean():.4%}  std={cmp_result['is'].std():.4%}")
        print(f"  Variance reduction: {cmp_result['var_reduction']:.1f}x")

        summary.append({
            "Ticker": ticker, "RejAcceptRate": f"{accept_rate:.1%}", "KS_p": f"{ks_pvalue:.3f}",
            "TrueTailP": f"{cmp_result['true_p']:.4%}",
            "NaiveMC_Std": f"{cmp_result['naive'].std():.4%}",
            "IS_Std": f"{cmp_result['is'].std():.4%}",
            "VarReduction": f"{cmp_result['var_reduction']:.1f}x",
        })

        plot_dashboard(ticker, target_frozen, rej_samples, ks_pvalue, cmp_result, threshold)

    if summary:
        print("\n" + "=" * 70)
        print("IMPORTANCE / REJECTION SAMPLING SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        avg_reduction = np.mean([float(s["VarReduction"].rstrip("x")) for s in summary])
        print(f"\n  Average variance-reduction factor across all tickers: {avg_reduction:.1f}x")

    print("\nSampling methods analysis complete.")


if __name__ == "__main__":
    main()
