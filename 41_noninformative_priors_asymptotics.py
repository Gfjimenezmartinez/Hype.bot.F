"""
================================================================
Script 41 — Estimation Theory III: Noninformative Priors, Multiparameter
Bayesian Inference, Bayesian Asymptotics
================================================================
Covers: noninformative priors, Bayesian inference for multiparameter
models, Bayesian asymptotics. (Ordinary Bayesian inference and Gaussian
linear models overlap with Scripts 31/32, reused/contrasted here rather
than rebuilt.)

  1. NONINFORMATIVE (JEFFREYS) PRIOR -- for the Gaussian location-scale
     model, the Jeffreys prior is p(mu, sigma^2) ~ 1/sigma^2 (improper).
     Combined with the Gaussian likelihood it gives a textbook-known,
     exactly checkable result: the marginal posterior of mu is a
     Student-t(df=n-1, loc=xbar, scale=s/sqrt(n)) -- which makes the
     Jeffreys-prior 95% credible interval for mu NUMERICALLY IDENTICAL
     to the ordinary frequentist t-confidence interval. Not "close" --
     exactly equal, a hard equality this script verifies rather than
     asserts. Contrasted against Script 31's weakly-informative-but-
     proper NIG prior: the two posteriors should converge as n grows.
  2. MULTIPARAMETER BAYESIAN INFERENCE -- the joint (mu, sigma^2)
     posterior under the Jeffreys prior, visualized directly (not just
     each parameter's marginal in isolation).
  3. BAYESIAN ASYMPTOTICS (Bernstein-von Mises) -- the theorem says the
     posterior converges to a Normal centered at the MLE with the
     Fisher-information covariance, REGARDLESS of the prior, as n grows.
     For this model that convergence is exactly "Student-t(df=n-1) ->
     Normal(0,1) as n -> infinity" -- a guaranteed mathematical fact,
     verified numerically via the KS distance between the two shrinking
     monotonically as n increases, tying directly to Script 40's Fisher
     information for this same model.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, special
from importlib import import_module as _im
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

_r31 = _im("31_bayesian_fundamentals")
bayesian_return_posterior = _r31.bayesian_return_posterior

PLOT_STYLE = "seaborn-v0_8-darkgrid"
BVM_N_GRID = [10, 20, 40, 80, 160, 320, 640]


# ============================================================
# 1. Jeffreys (Noninformative) Prior Posterior
# ============================================================
def jeffreys_posterior(returns):
    x = np.asarray(returns)
    n = len(x)
    xbar = x.mean()
    s2 = x.var(ddof=1)
    s = np.sqrt(s2)
    return {"n": n, "xbar": xbar, "s2": s2, "s": s,
            "mu_df": n - 1, "mu_loc": xbar, "mu_scale": s / np.sqrt(n)}


def jeffreys_credible_interval(post, ci=0.95):
    return stats.t.interval(ci, post["mu_df"], loc=post["mu_loc"], scale=post["mu_scale"])


def frequentist_t_interval(returns, ci=0.95):
    """The ordinary undergraduate-stats confidence interval for the mean
    -- included only to check it against jeffreys_credible_interval,
    which the Bayesian/frequentist correspondence says must match exactly."""
    x = np.asarray(returns)
    n = len(x)
    xbar, s = x.mean(), x.std(ddof=1)
    return stats.t.interval(ci, n - 1, loc=xbar, scale=s / np.sqrt(n))


# ============================================================
# 2. Bayesian Asymptotics — Bernstein-von Mises convergence
# ============================================================
def bvm_ks_distance(n):
    """KS distance between standardized t(df=n-1) and N(0,1) -- the
    exact quantity that must shrink to 0 as n grows for Bernstein-von
    Mises to hold in this model (Jeffreys-prior posterior for mu IS a
    (rescaled) Student-t, and BvM says it converges to the Fisher-
    information Normal as n -> infinity)."""
    grid = np.linspace(-6, 6, 4000)
    t_cdf = stats.t.cdf(grid, df=n - 1)
    norm_cdf = stats.norm.cdf(grid)
    return float(np.max(np.abs(t_cdf - norm_cdf)))


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, returns, jeff_post, nig_post, jeff_ci, freq_ci, ks_distances):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # [0,0] Jeffreys posterior for mu vs Script 31's NIG posterior
    ax0 = fig.add_subplot(gs[0, 0])
    r = np.linspace(jeff_post["mu_loc"] - 5 * jeff_post["mu_scale"],
                     jeff_post["mu_loc"] + 5 * jeff_post["mu_scale"], 400)
    ax0.plot(r, stats.t.pdf(r, jeff_post["mu_df"], jeff_post["mu_loc"], jeff_post["mu_scale"]),
             color="crimson", lw=1.8, label="Jeffreys (noninformative) posterior")
    ax0.plot(r, stats.t.pdf(r, nig_post["pred_df"], nig_post["mu_n"],
                             np.sqrt(nig_post["beta_n"] / (nig_post["alpha_n"] * nig_post["kappa_n"]))),
             color="steelblue", lw=1.8, ls="--", label="Script 31's NIG (weak proper prior)")
    ax0.axvspan(*jeff_ci, color="crimson", alpha=0.12)
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Posterior for mu: Noninformative vs Weak-Informative Prior", fontsize=9.5)
    ax0.grid(alpha=0.3)

    # [0,1] Joint (mu, sigma^2) posterior contour -- multiparameter Bayesian inference
    ax1 = fig.add_subplot(gs[0, 1])
    n, xbar, s2 = jeff_post["n"], jeff_post["xbar"], jeff_post["s2"]
    mu_grid = np.linspace(xbar - 4 * jeff_post["mu_scale"], xbar + 4 * jeff_post["mu_scale"], 150)
    sigma2_grid = np.linspace(max(s2 * 0.4, 1e-8), s2 * 1.8, 150)
    MU, S2 = np.meshgrid(mu_grid, sigma2_grid)
    # p(mu, sigma2 | data) = N(mu; xbar, sigma2/n) * InvGamma(sigma2; (n-1)/2, (n-1)s2/2)
    log_dens = (-0.5 * np.log(2 * np.pi * S2 / n) - (MU - xbar) ** 2 * n / (2 * S2)
                + ((n - 1) / 2) * np.log((n - 1) * s2 / 2) - special.gammaln((n - 1) / 2)
                - ((n - 1) / 2 + 1) * np.log(S2) - (n - 1) * s2 / (2 * S2))
    ax1.contourf(MU, S2, np.exp(log_dens), levels=20, cmap="viridis")
    ax1.scatter([xbar], [s2], color="red", marker="x", s=60, label="MLE (xbar, s^2)")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("mu"); ax1.set_ylabel("sigma^2")
    ax1.set_title("Joint Posterior p(mu, sigma^2 | data)  [Jeffreys prior]", fontsize=9.5)

    # [1,0] Bernstein-von Mises convergence: KS distance vs n
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(BVM_N_GRID, ks_distances, "o-", color="darkorange", lw=1.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("n (log scale)"); ax2.set_ylabel("KS distance: t(df=n-1) vs N(0,1)")
    ax2.set_title("Bernstein-von Mises: Posterior -> Fisher-Info Normal as n grows", fontsize=9.5)
    ax2.grid(alpha=0.3)

    # [1,1] Summary table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Jeffreys 95% CI for mu", f"[{jeff_ci[0]:+.6f}, {jeff_ci[1]:+.6f}]"],
        ["Frequentist t 95% CI for mu", f"[{freq_ci[0]:+.6f}, {freq_ci[1]:+.6f}]"],
        ["Intervals match exactly?", "YES" if np.allclose(jeff_ci, freq_ci) else "NO -- BUG"],
        ["n", f"{jeff_post['n']}"],
        ["KS distance at this n", f"{bvm_ks_distance(jeff_post['n']):.4f}"],
    ]
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)

    fig.suptitle(f"{ticker} — Noninformative Priors, Multiparameter Inference, Bayesian Asymptotics",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("ESTIMATION THEORY III — NONINFORMATIVE PRIORS, MULTIPARAMETER")
    print("BAYESIAN INFERENCE, BAYESIAN ASYMPTOTICS")
    print("=" * 70)

    ks_distances = [bvm_ks_distance(n) for n in BVM_N_GRID]
    print("\n  Bernstein-von Mises check (should shrink monotonically):")
    for n, ks in zip(BVM_N_GRID, ks_distances):
        print(f"    n={n:4d}: KS(t_(n-1), N(0,1)) = {ks:.4f}")
    monotonic = all(ks_distances[i] >= ks_distances[i + 1] for i in range(len(ks_distances) - 1))
    print(f"  Monotonically shrinking: {'YES' if monotonic else 'NO -- unexpected'}")

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            print(f"\n  {ticker}: skipped -- need >= 60 return obs")
            continue

        jeff_post = jeffreys_posterior(returns)
        nig_post = bayesian_return_posterior(returns)
        jeff_ci = jeffreys_credible_interval(jeff_post)
        freq_ci = frequentist_t_interval(returns)
        exact_match = bool(np.allclose(jeff_ci, freq_ci))

        nig_mean = nig_post["mu_n"]
        prior_posterior_gap = abs(jeff_post["xbar"] - nig_mean)

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}   (n={jeff_post['n']})")
        print(f"{'─'*55}")
        print(f"  Jeffreys 95% CI:      [{jeff_ci[0]:+.6f}, {jeff_ci[1]:+.6f}]")
        print(f"  Frequentist t 95% CI: [{freq_ci[0]:+.6f}, {freq_ci[1]:+.6f}]")
        print(f"  Exact match: {exact_match}  {'OK' if exact_match else 'FAILED -- BUG'}")
        print(f"  Jeffreys posterior mean vs Script 31's NIG posterior mean: "
              f"{jeff_post['xbar']:+.6f} vs {nig_mean:+.6f}  (gap={prior_posterior_gap:.2e})")

        summary.append({
            "Ticker": ticker, "N": jeff_post["n"],
            "JeffreysVsFreqMatch": "OK" if exact_match else "FAILED",
            "Jeffreys_vs_NIG_Gap": f"{prior_posterior_gap:.2e}",
        })

        plot_dashboard(ticker, returns, jeff_post, nig_post, jeff_ci, freq_ci, ks_distances)

    if summary:
        print("\n" + "=" * 70)
        print("NONINFORMATIVE PRIORS / BAYESIAN ASYMPTOTICS SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        n_ok = sum(1 for s in summary if s["JeffreysVsFreqMatch"] == "OK")
        print(f"\n  Jeffreys/frequentist exact match: {n_ok}/{len(summary)} tickers.")

    print("\nNoninformative priors / Bayesian asymptotics analysis complete.")


if __name__ == "__main__":
    main()
