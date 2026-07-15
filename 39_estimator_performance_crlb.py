"""
================================================================
Script 39 — Estimation Theory I: Estimator Performance, Sufficient
Statistics, Cramér-Rao Lower Bound
================================================================
Covers: estimator performance (bias/variance/consistency), sufficient
statistics, the Cramér-Rao lower bound.

Every earlier script treats the sample mean and sample variance of
returns as obviously-good estimators without ever checking that
formally. This script does:

  1. ESTIMATOR PERFORMANCE -- parametric bootstrap comparing the sample
     mean (mu_hat) and both the MLE (biased, /n) and unbiased (/(n-1))
     sample-variance estimators against their known textbook bias/
     variance formulas for a Gaussian model. Not a synthetic toy: the
     "true" (mu, sigma) is fit from each ticker's actual returns first,
     then bootstrapped from there.
  2. SUFFICIENT STATISTICS -- the factorization theorem says the
     Gaussian likelihood depends on the data only through (n, sum(x),
     sum(x^2)). Demonstrated concretely, not just stated: a second,
     completely different-looking dataset is constructed via a random
     orthogonal rotation of the centered data (which preserves the
     Euclidean norm exactly, hence sum(x^2), and trivially preserves the
     mean since the same mean is added back to every coordinate) --
     giving two datasets with identical sufficient statistics and
     verifiably identical likelihood/MLE, despite different raw values.
  3. CRAMÉR-RAO LOWER BOUND -- computed from the Gaussian Fisher
     information matrix (diagonal: mu and sigma^2 are orthogonal
     parameters for a Gaussian), and checked against the bootstrapped
     variances above: the sample mean achieves the CRLB exactly (it's
     efficient), while the unbiased sample variance sits slightly above
     its CRLB (efficiency ratio n/(n-1) -- unbiased but only
     asymptotically efficient, not finite-sample efficient). Both are
     textbook-known, checkable outcomes, not assertions.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

PLOT_STYLE = "seaborn-v0_8-darkgrid"
N_BOOT = 3000


# ============================================================
# Cramér-Rao Lower Bound (Gaussian Fisher information)
# ============================================================
def crlb_mu_sigma2(sigma2, n):
    """
    Fisher information per observation for N(mu, sigma^2), both unknown,
    is diag(1/sigma^2, 1/(2*sigma^4)) -- mu and sigma^2 are orthogonal
    parameters for a Gaussian (the off-diagonal vanishes because the
    third central moment of a Gaussian is zero). CRLB = inverse Fisher
    information / n.
    """
    crlb_mu = sigma2 / n
    crlb_sigma2 = 2 * sigma2 ** 2 / n
    return crlb_mu, crlb_sigma2


# ============================================================
# Estimator Performance — parametric bootstrap
# ============================================================
def bootstrap_estimator_performance(mu_true, sigma_true, n, n_boot=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    samples = rng.normal(mu_true, sigma_true, size=(n_boot, n))

    mu_hats = samples.mean(axis=1)
    sigma2_mle = samples.var(axis=1, ddof=0)        # MLE: divide by n
    sigma2_unbiased = samples.var(axis=1, ddof=1)    # divide by (n-1)

    sigma2_true = sigma_true ** 2
    return {
        "mu_hat_bias": float(mu_hats.mean() - mu_true), "mu_hat_var": float(mu_hats.var(ddof=1)),
        "sigma2_mle_bias": float(sigma2_mle.mean() - sigma2_true), "sigma2_mle_var": float(sigma2_mle.var(ddof=1)),
        "sigma2_unb_bias": float(sigma2_unbiased.mean() - sigma2_true), "sigma2_unb_var": float(sigma2_unbiased.var(ddof=1)),
        "mu_hats": mu_hats, "sigma2_unbiased_samples": sigma2_unbiased,
    }


# ============================================================
# Sufficient Statistics — factorization theorem, demonstrated concretely
# ============================================================
def _random_rotation_fixing_ones(n, rng):
    """
    A random orthogonal n x n matrix that fixes the all-ones direction
    (eigenvalue 1) and rotates arbitrarily within its orthogonal
    complement -- i.e. it maps the "sum-zero" subspace to itself. A
    generic random orthogonal matrix does NOT have this property (it
    mixes the sum-zero subspace with the all-ones direction), which
    would silently break sum-preservation below.
    """
    u1 = np.ones(n) / np.sqrt(n)
    U, _ = np.linalg.qr(np.column_stack([u1, rng.standard_normal((n, n - 1))]))
    R_sub, _ = np.linalg.qr(rng.standard_normal((n - 1, n - 1)))
    block = np.eye(n)
    block[1:, 1:] = R_sub
    return U @ block @ U.T


def demonstrate_sufficiency(returns, seed=0):
    """
    Constructs a second dataset B with IDENTICAL (n, sum, sum-of-squares)
    to the real data A, via B = xbar + Q @ (x - xbar) for Q a random
    orthogonal matrix that fixes the all-ones direction (see
    _random_rotation_fixing_ones): such a Q preserves the Euclidean norm
    of the centered data exactly (hence sum((x-xbar)^2), hence sum(x^2)
    given the mean is unchanged) AND maps the sum-zero subspace the
    centered data lives in back to itself (hence sum(Q@centered)=0
    exactly, so adding xbar back to every coordinate reproduces the
    original sum exactly, not just approximately). Two very different-
    looking raw samples, identical sufficient statistics -- and, as the
    factorization theorem requires, identical likelihood/MLE.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    xbar = returns.mean()
    centered = returns - xbar
    Q = _random_rotation_fixing_ones(n, rng)
    dataset_b = xbar + Q @ centered

    stats_a = (len(returns), float(returns.sum()), float((returns ** 2).sum()))
    stats_b = (len(dataset_b), float(dataset_b.sum()), float((dataset_b ** 2).sum()))

    def gaussian_loglik(x, mu, sigma2):
        return float(np.sum(-0.5 * np.log(2 * np.pi * sigma2) - (x - mu) ** 2 / (2 * sigma2)))

    mu_a, sigma2_a = returns.mean(), returns.var(ddof=0)
    mu_b, sigma2_b = dataset_b.mean(), dataset_b.var(ddof=0)
    ll_a = gaussian_loglik(returns, mu_a, sigma2_a)
    ll_b = gaussian_loglik(dataset_b, mu_b, sigma2_b)

    return {"dataset_b": dataset_b, "stats_a": stats_a, "stats_b": stats_b,
            "mle_a": (mu_a, sigma2_a), "mle_b": (mu_b, sigma2_b),
            "loglik_a": ll_a, "loglik_b": ll_b,
            "stats_match": bool(np.allclose(stats_a[1:], stats_b[1:], rtol=1e-8)),
            "loglik_match": bool(np.isclose(ll_a, ll_b, rtol=1e-8))}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, mu_true, sigma_true, n, boot, suff, crlb_mu, crlb_sigma2):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # [0,0] Variance vs CRLB
    ax0 = fig.add_subplot(gs[0, 0])
    names = ["mu_hat", "sigma2 (MLE, /n)", "sigma2 (unbiased, /n-1)"]
    variances = [boot["mu_hat_var"], boot["sigma2_mle_var"], boot["sigma2_unb_var"]]
    crlbs = [crlb_mu, crlb_sigma2, crlb_sigma2]
    x = np.arange(3)
    ax0.bar(x - 0.15, variances, width=0.3, color="steelblue", alpha=0.85, label="Bootstrap empirical variance")
    ax0.bar(x + 0.15, crlbs, width=0.3, color="crimson", alpha=0.85, label="CRLB")
    ax0.set_xticks(x); ax0.set_xticklabels(names, fontsize=8)
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Estimator Variance vs Cramér-Rao Bound (n={n})", fontsize=9.5)
    ax0.grid(axis="y", alpha=0.3)

    # [0,1] Sample mean distribution vs CRLB-implied Normal
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(boot["mu_hats"], bins=40, density=True, color="lightsteelblue", alpha=0.7, label="Bootstrap mu_hat")
    from scipy.stats import norm
    r = np.linspace(boot["mu_hats"].min(), boot["mu_hats"].max(), 200)
    ax1.plot(r, norm.pdf(r, mu_true, np.sqrt(crlb_mu)), color="crimson", lw=1.8,
             label="N(mu, CRLB) -- efficient estimator")
    ax1.legend(fontsize=8)
    ax1.set_title("Sample Mean: Achieves the CRLB Exactly", fontsize=10)
    ax1.grid(alpha=0.3)

    # [1,0] Efficiency ratio vs n (unbiased sigma2 estimator)
    ax2 = fig.add_subplot(gs[1, 0])
    n_grid = np.arange(10, 400, 10)
    efficiency_ratio = n_grid / (n_grid - 1)   # Var(s^2)/CRLB = n/(n-1)
    ax2.plot(n_grid, efficiency_ratio, color="darkorange", lw=1.8)
    ax2.axhline(1.0, color="gray", lw=0.8, ls="--", label="fully efficient")
    ax2.axvline(n, color="crimson", lw=1.0, ls=":", label=f"this ticker (n={n})")
    ax2.legend(fontsize=8)
    ax2.set_xlabel("n"); ax2.set_ylabel("Var(unbiased sigma^2) / CRLB")
    ax2.set_title("Unbiased Variance Estimator: Asymptotically (not finite-sample) Efficient", fontsize=9)
    ax2.grid(alpha=0.3)

    # [1,1] Sufficiency demonstration summary
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Dataset A: (n, sum, sum-sq)", f"{suff['stats_a'][0]}, {suff['stats_a'][1]:.6f}, {suff['stats_a'][2]:.6f}"],
        ["Dataset B: (n, sum, sum-sq)", f"{suff['stats_b'][0]}, {suff['stats_b'][1]:.6f}, {suff['stats_b'][2]:.6f}"],
        ["Sufficient stats match?", "YES" if suff["stats_match"] else "NO -- BUG"],
        ["MLE A (mu, sigma2)", f"({suff['mle_a'][0]:.6f}, {suff['mle_a'][1]:.6f})"],
        ["MLE B (mu, sigma2)", f"({suff['mle_b'][0]:.6f}, {suff['mle_b'][1]:.6f})"],
        ["Log-likelihood A vs B", f"{suff['loglik_a']:.4f}  vs  {suff['loglik_b']:.4f}"],
        ["Likelihoods match?", "YES (factorization theorem holds)" if suff["loglik_match"] else "NO -- BUG"],
    ]
    table = ax3.table(cellText=rows, colLabels=["Check", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.5)
    ax3.set_title("Sufficiency Demonstration (orthogonal-rotation construction)", fontsize=9.5, pad=12)

    fig.suptitle(f"{ticker} — Estimation Theory: Performance, Sufficiency, CRLB", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("ESTIMATION THEORY I — ESTIMATOR PERFORMANCE, SUFFICIENCY, CRLB")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            print(f"\n  {ticker}: skipped -- need >= 60 return obs")
            continue

        n = len(returns)
        mu_true, sigma_true = returns.mean(), returns.std(ddof=0)
        crlb_mu, crlb_sigma2 = crlb_mu_sigma2(sigma_true ** 2, n)
        boot = bootstrap_estimator_performance(mu_true, sigma_true, n)
        suff = demonstrate_sufficiency(returns)

        mu_efficiency = boot["mu_hat_var"] / crlb_mu
        sigma2_unb_efficiency = boot["sigma2_unb_var"] / crlb_sigma2

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}   (n={n})")
        print(f"{'─'*55}")
        print(f"  mu_hat:        bias={boot['mu_hat_bias']:+.2e}  var={boot['mu_hat_var']:.3e}  "
              f"CRLB={crlb_mu:.3e}  efficiency={mu_efficiency:.3f} (1.0=efficient)")
        print(f"  sigma2 (MLE):  bias={boot['sigma2_mle_bias']:+.3e}  var={boot['sigma2_mle_var']:.3e}  "
              f"(biased by -sigma^2/n = {-sigma_true**2/n:.3e}, as expected)")
        print(f"  sigma2 (unb.): bias={boot['sigma2_unb_bias']:+.3e}  var={boot['sigma2_unb_var']:.3e}  "
              f"CRLB={crlb_sigma2:.3e}  efficiency={sigma2_unb_efficiency:.3f} "
              f"(expected ~{n/(n-1):.3f} = n/(n-1))")
        print(f"  Sufficiency: stats_match={suff['stats_match']}  loglik_match={suff['loglik_match']}  "
              f"{'OK' if suff['stats_match'] and suff['loglik_match'] else 'FAILED -- BUG'}")

        summary.append({
            "Ticker": ticker, "N": n,
            "MuHat_Efficiency": f"{mu_efficiency:.3f}",
            "Sigma2Unb_Efficiency": f"{sigma2_unb_efficiency:.3f}",
            "ExpectedEfficiency": f"{n/(n-1):.3f}",
            "SufficiencyCheck": "OK" if suff["stats_match"] and suff["loglik_match"] else "FAILED",
        })

        plot_dashboard(ticker, mu_true, sigma_true, n, boot, suff, crlb_mu, crlb_sigma2)

    if summary:
        print("\n" + "=" * 70)
        print("ESTIMATOR PERFORMANCE / CRLB SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        n_ok = sum(1 for s in summary if s["SufficiencyCheck"] == "OK")
        print(f"\n  Sufficiency check passed for {n_ok}/{len(summary)} tickers.")

    print("\nEstimator performance / CRLB analysis complete.")


if __name__ == "__main__":
    main()
