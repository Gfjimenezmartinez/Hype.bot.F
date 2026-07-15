"""
================================================================
Script 40 — Estimation Theory II: Linear Models, Maximum-Likelihood
Estimation
================================================================
Covers: linear models, maximum-likelihood estimation.

Script 23 fits OLS/Ridge/Lasso via sklearn without ever connecting it to
estimation theory: for y = X*beta + eps, eps ~ N(0, sigma^2 I), the OLS
estimator beta_hat = (X^T X)^-1 X^T y IS the MLE, and its asymptotic
covariance sigma^2 (X^T X)^-1 falls straight out of the Fisher
information matrix (the same orthogonality between the mean and
variance parameters as Script 39's Gaussian case, just in regression
form). This script makes that explicit and checks the asymptotic
covariance formula against a nonparametric residual bootstrap on the
SAME causal features Script 23 already builds -- if the two disagree
substantially, either the Gaussian-errors assumption is bad for this
data or there's an implementation bug; if they agree, the classical
OLS standard errors are trustworthy here.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from importlib import import_module as _im
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

_r23 = _im("23_ml_alpha_model")
build_dataset = _r23.build_dataset

PLOT_STYLE = "seaborn-v0_8-darkgrid"
N_BOOT = 1000


# ============================================================
# OLS as MLE
# ============================================================
def fit_ols_mle(X, y):
    """beta_hat = (X^T X)^-1 X^T y is simultaneously the least-squares
    and (under Gaussian errors) the maximum-likelihood estimator."""
    Xd = np.c_[np.ones(len(X)), X.values if hasattr(X, "values") else X]
    yv = y.values if hasattr(y, "values") else np.asarray(y)
    beta_hat = np.linalg.lstsq(Xd, yv, rcond=None)[0]
    resid = yv - Xd @ beta_hat
    sigma2_mle = float(np.mean(resid ** 2))   # MLE variance: /n, not /(n-p)
    return {"beta_hat": beta_hat, "sigma2_mle": sigma2_mle, "resid": resid, "Xd": Xd}


def fisher_info_asymptotic_cov(Xd, sigma2):
    """Asymptotic covariance of the MLE/OLS beta_hat: sigma^2 (X^T X)^-1,
    straight from inverting the Fisher information matrix for a linear-
    Gaussian model."""
    return sigma2 * np.linalg.inv(Xd.T @ Xd)


# ============================================================
# Residual Bootstrap — nonparametric check of the asymptotic covariance
# ============================================================
def bootstrap_ols_covariance(Xd, beta_hat, resid, n_boot=N_BOOT, seed=42):
    """
    Holds X fixed (the classical fixed-design assumption behind the
    Fisher-information formula) and resamples residuals with replacement
    to build y* = X beta_hat + resid*, refitting OLS each time -- a
    standard nonparametric residual bootstrap, independent of the
    Gaussian-errors assumption used to derive the Fisher information.
    """
    rng = np.random.default_rng(seed)
    n = Xd.shape[0]
    betas = np.zeros((n_boot, len(beta_hat)))
    XtX_inv_Xt = np.linalg.inv(Xd.T @ Xd) @ Xd.T
    for b in range(n_boot):
        resid_star = rng.choice(resid, size=n, replace=True)
        y_star = Xd @ beta_hat + resid_star
        betas[b] = XtX_inv_Xt @ y_star
    return np.cov(betas, rowvar=False), betas


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, feat_cols, beta_hat, asym_cov, boot_cov, betas):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    names = ["intercept"] + list(feat_cols)

    asym_sd = np.sqrt(np.diag(asym_cov))
    boot_sd = np.sqrt(np.diag(boot_cov))

    # [0,0] Asymptotic vs bootstrap std error per coefficient
    ax0 = fig.add_subplot(gs[0, 0])
    order = np.argsort(-np.abs(beta_hat))[:10]
    x = np.arange(len(order))
    ax0.bar(x - 0.15, asym_sd[order], width=0.3, color="crimson", alpha=0.85, label="Asymptotic (Fisher info)")
    ax0.bar(x + 0.15, boot_sd[order], width=0.3, color="steelblue", alpha=0.85, label="Residual bootstrap")
    ax0.set_xticks(x); ax0.set_xticklabels([names[i] for i in order], fontsize=7, rotation=30, ha="right")
    ax0.legend(fontsize=8)
    ax0.set_ylabel("std error")
    ax0.set_title(f"{ticker} — Coefficient Std Errors: Theory vs Bootstrap", fontsize=9.5)
    ax0.grid(axis="y", alpha=0.3)

    # [0,1] Bootstrap distribution of the largest coefficient vs asymptotic Normal
    ax1 = fig.add_subplot(gs[0, 1])
    top = order[0]
    from scipy.stats import norm
    ax1.hist(betas[:, top], bins=40, density=True, color="lightsteelblue", alpha=0.7, label="Bootstrap beta*")
    r = np.linspace(betas[:, top].min(), betas[:, top].max(), 200)
    ax1.plot(r, norm.pdf(r, beta_hat[top], asym_sd[top]), color="crimson", lw=1.8,
             label="Asymptotic N(beta_hat, Fisher^-1)")
    ax1.legend(fontsize=8)
    ax1.set_title(f"Coefficient: {names[top]}", fontsize=10)
    ax1.grid(alpha=0.3)

    # [1,0] Ratio of bootstrap to asymptotic std error (should cluster near 1.0)
    ax2 = fig.add_subplot(gs[1, 0])
    ratio = boot_sd / np.maximum(asym_sd, 1e-12)
    ax2.bar(range(len(ratio)), ratio, color="darkorange", alpha=0.85)
    ax2.axhline(1.0, color="gray", lw=1.0, ls="--", label="perfect agreement")
    ax2.legend(fontsize=8)
    ax2.set_xlabel("coefficient index"); ax2.set_ylabel("bootstrap SE / asymptotic SE")
    ax2.set_title("Agreement Across All Coefficients", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    # [1,1] Summary table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Mean SE ratio (boot/asymptotic)", f"{ratio.mean():.3f}"],
        ["Median SE ratio", f"{np.median(ratio):.3f}"],
        ["Max |ratio - 1|", f"{np.max(np.abs(ratio - 1)):.3f}"],
        ["N coefficients (incl. intercept)", f"{len(beta_hat)}"],
    ]
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    fig.suptitle(f"{ticker} — Linear Models: OLS as MLE, Fisher Information vs Bootstrap",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("ESTIMATION THEORY II — LINEAR MODELS, MAXIMUM-LIKELIHOOD ESTIMATION")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        X, y, feat_cols = build_dataset(df)
        if len(X) < 120:
            print(f"\n  {ticker}: skipped -- need >= 120 usable rows")
            continue

        fit = fit_ols_mle(X, y)
        asym_cov = fisher_info_asymptotic_cov(fit["Xd"], fit["sigma2_mle"])
        boot_cov, betas = bootstrap_ols_covariance(fit["Xd"], fit["beta_hat"], fit["resid"])

        asym_sd = np.sqrt(np.diag(asym_cov))
        boot_sd = np.sqrt(np.diag(boot_cov))
        ratio = boot_sd / np.maximum(asym_sd, 1e-12)

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}   (n={len(X)}, p={len(feat_cols)+1})")
        print(f"{'─'*55}")
        print(f"  sigma2_MLE={fit['sigma2_mle']:.6f}")
        print(f"  SE ratio (bootstrap/asymptotic): mean={ratio.mean():.3f}  "
              f"median={np.median(ratio):.3f}  max|ratio-1|={np.max(np.abs(ratio-1)):.3f}")

        summary.append({
            "Ticker": ticker, "N": len(X), "P": len(feat_cols) + 1,
            "Sigma2_MLE": f"{fit['sigma2_mle']:.6f}",
            "MeanSERatio": f"{ratio.mean():.3f}", "MaxDevFrom1": f"{np.max(np.abs(ratio-1)):.3f}",
        })

        plot_dashboard(ticker, feat_cols, fit["beta_hat"], asym_cov, boot_cov, betas)

    if summary:
        print("\n" + "=" * 70)
        print("LINEAR MODELS / MLE SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        avg_ratio = np.mean([float(s["MeanSERatio"]) for s in summary])
        print(f"\n  Average SE ratio (bootstrap/asymptotic) across tickers: {avg_ratio:.3f} (target: 1.0)")

    print("\nLinear models / MLE analysis complete.")


if __name__ == "__main__":
    main()
