"""
================================================================
Script 13 — Portfolio Optimisation (Markowitz + Risk Parity)
================================================================
Ledoit-Wolf shrinkage covariance, efficient frontier, min-variance,
max-Sharpe, equal-weight, and equal-risk-contribution portfolios.
Includes portfolio-level VaR/CVaR (historical + parametric + MC).

Adapted from: portfolio_optimizer.py + risk_analysis.py
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS

PLOT_STYLE = "seaborn-v0_8-darkgrid"
ANN        = 365   # crypto trades 24/7
RF         = 0.045


# ============================================================
# Covariance Estimation
# ============================================================
def ledoit_wolf_cov(returns):
    T, N = returns.shape
    S    = np.cov(returns.T, ddof=1)
    mu   = np.trace(S) / N
    F    = mu * np.eye(N)

    delta2 = 0.0
    for t in range(T):
        x = returns[t].reshape(-1, 1)
        delta2 += np.linalg.norm(x @ x.T - S, "fro") ** 2
    delta2 /= T ** 2

    denom = np.linalg.norm(S - F, "fro") ** 2
    rho   = min(delta2 / denom, 1.0) if denom > 0 else 0.0
    return (1 - rho) * S + rho * F


# ============================================================
# Portfolio Statistics
# ============================================================
def portfolio_stats(w, mu, cov):
    ret = float(w @ mu) * ANN
    vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(ANN)
    shp = (ret - RF) / vol if vol > 1e-8 else 0.0
    return ret, vol, shp


# ============================================================
# Optimisation
# ============================================================
def _min_variance(mu, cov, bounds):
    N  = len(mu)
    w0 = np.ones(N) / N
    res = optimize.minimize(
        lambda w: float(w @ cov @ w), w0, method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-10, "maxiter": 1000})
    return res.x if res.success else w0


def _max_sharpe(mu, cov, bounds):
    N  = len(mu)
    w0 = np.ones(N) / N
    def neg_sharpe(w):
        r, v, _ = portfolio_stats(w, mu, cov)
        return -(r - RF) / max(v, 1e-8)
    res = optimize.minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-10, "maxiter": 1000})
    return res.x if res.success else w0


def risk_parity_weights(cov, tol=1e-10):
    N  = cov.shape[0]
    w0 = np.ones(N) / N
    def objective(w):
        sigma = float(np.sqrt(w @ cov @ w))
        if sigma < 1e-12:
            return 0.0
        rc   = w * (cov @ w) / sigma
        diff = rc[:, None] - rc[None, :]
        return float((diff ** 2).sum())
    bounds      = [(1e-6, 1.0)] * N
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = optimize.minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": tol, "maxiter": 2000})
    w = res.x if res.success else w0
    return np.maximum(w, 0) / np.maximum(w, 0).sum()


def risk_contributions(w, cov):
    sigma = float(np.sqrt(w @ cov @ w))
    if sigma < 1e-12:
        return np.ones(len(w)) / len(w)
    rc = w * (cov @ w) / sigma
    return rc / rc.sum()


def efficient_frontier(returns, asset_names, n_points=80):
    T, N  = returns.shape
    mu    = returns.mean(axis=0)
    cov   = ledoit_wolf_cov(returns)
    bounds = [(0.0, 1.0)] * N

    targets = np.linspace(float(mu.min()) * 1.05, float(mu.max()) * 0.95, n_points)
    frontier_r, frontier_v = [], []
    for tgt in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t},
        ]
        res = optimize.minimize(
            lambda w: float(w @ cov @ w),
            np.ones(N) / N, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000})
        if res.success:
            r, v, _ = portfolio_stats(res.x, mu, cov)
            frontier_r.append(r)
            frontier_v.append(v)

    gmv = _min_variance(mu, cov, bounds)
    msr = _max_sharpe(mu, cov, bounds)
    ew  = np.ones(N) / N
    rp  = risk_parity_weights(cov)

    return {
        "frontier_vols": np.array(frontier_v),
        "frontier_rets": np.array(frontier_r),
        "gmv":  {"w": gmv, "stats": portfolio_stats(gmv, mu, cov)},
        "msr":  {"w": msr, "stats": portfolio_stats(msr, mu, cov)},
        "ew":   {"w": ew,  "stats": portfolio_stats(ew,  mu, cov)},
        "rp":   {"w": rp,  "stats": portfolio_stats(rp,  mu, cov)},
        "mu": mu, "cov": cov, "names": asset_names,
    }


# ============================================================
# Risk Metrics
# ============================================================
def portfolio_var(returns, w, confidence=0.95):
    port_r = returns @ w
    var    = float(-np.percentile(port_r, (1 - confidence) * 100))
    tail   = port_r[port_r <= -var]
    cvar   = float(-tail.mean()) if len(tail) > 0 else var
    return var, cvar


def parametric_var(returns, w, confidence=0.95):
    port_r = returns @ w
    mu     = float(port_r.mean())
    sig    = float(port_r.std())
    z      = stats.norm.ppf(1 - confidence)
    var    = float(-(mu + z * sig))
    cvar   = float(-(mu - sig * stats.norm.pdf(z) / (1 - confidence)))
    return var, cvar


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(result, returns):
    plt.style.use(PLOT_STYLE)
    fig   = plt.figure(figsize=(17, 11))
    gs    = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)
    names = result["names"]
    N     = len(names)
    mu    = result["mu"]
    cov   = result["cov"]

    port_keys   = ["gmv", "msr", "ew", "rp"]
    port_labels = {"gmv": "Min Var", "msr": "Max Sharpe",
                   "ew": "Equal Wt", "rp": "Risk Parity"}
    port_colors = {"gmv": "teal", "msr": "gold",
                   "ew": "royalblue", "rp": "purple"}

    # [0,0] Efficient Frontier
    ax0 = fig.add_subplot(gs[0, 0])
    vf  = result["frontier_vols"] * 100
    rf  = result["frontier_rets"] * 100
    sharpes = (rf - RF * 100) / np.maximum(vf, 1e-6)
    sc = ax0.scatter(vf, rf, c=sharpes, cmap="RdYlGn", s=8, alpha=0.85)
    plt.colorbar(sc, ax=ax0, pad=0.02, label="Sharpe")
    for key in port_keys:
        s = result[key]["stats"]
        ax0.scatter(s[1] * 100, s[0] * 100, s=100, color=port_colors[key],
                    marker="D", edgecolors="white", linewidths=0.7, zorder=6)
        ax0.annotate(f" {port_labels[key]}", (s[1] * 100, s[0] * 100),
                     fontsize=7, fontweight="bold")
    for i, name in enumerate(names):
        av = float(cov[i, i]**0.5) * np.sqrt(ANN) * 100
        ar = float(mu[i]) * ANN * 100
        ax0.scatter(av, ar, s=30, color="darkorange", marker="o",
                    edgecolors="white", linewidths=0.4, zorder=5)
        ax0.annotate(f" {name}", (av, ar), fontsize=5.5)
    ax0.set_xlabel("Annualised Vol (%)", fontsize=8)
    ax0.set_ylabel("Annualised Return (%)", fontsize=8)
    ax0.set_title("Efficient Frontier  |  Ledoit-Wolf Cov", fontsize=10)
    ax0.grid(alpha=0.3)

    # [0,1] Weight Comparison
    ax1 = fig.add_subplot(gs[0, 1])
    x_pos  = np.arange(len(port_keys))
    bottom = np.zeros(len(port_keys))
    colors = plt.cm.tab20(np.linspace(0, 1, N))
    for i, name in enumerate(names):
        vals = np.array([result[k]["w"][i] * 100 for k in port_keys])
        ax1.bar(x_pos, vals, bottom=bottom, color=colors[i],
                alpha=0.85, label=name, edgecolor="white", lw=0.3)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 4:
                ax1.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                         fontsize=5.5, fontweight="bold")
        bottom += vals
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([port_labels[k] for k in port_keys], fontsize=8)
    ax1.set_ylabel("Weight (%)", fontsize=8)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=5, ncol=3, loc="upper right")
    ax1.set_title("Portfolio Weights by Strategy", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # [1,0] Risk Contribution
    ax2    = fig.add_subplot(gs[1, 0])
    y_base = 0
    bar_h  = 0.20
    yticks, ylabels = [], []
    for ki, key in enumerate(port_keys):
        rc   = risk_contributions(result[key]["w"], cov) * 100
        left = 0.0
        for i, name in enumerate(names):
            ax2.barh(y_base, rc[i], left=left, height=bar_h,
                     color=colors[i], alpha=0.85, edgecolor="white", lw=0.3)
            if rc[i] > 6:
                ax2.text(left + rc[i] / 2, y_base, f"{rc[i]:.0f}%",
                         ha="center", va="center", fontsize=5.5, fontweight="bold")
            left += rc[i]
        yticks.append(y_base)
        ylabels.append(port_labels[key])
        y_base += bar_h + 0.06
    ax2.axvline(100 / N, color="gray", lw=0.8, ls="--", alpha=0.6, label="Equal RC")
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels, fontsize=8)
    ax2.set_xlabel("Risk Contribution (%)", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.set_title("Risk Contribution  |  ERC vs Others", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    # [1,1] Correlation Heatmap
    ax3  = fig.add_subplot(gs[1, 1])
    d    = np.sqrt(np.diag(cov))
    d    = np.where(d > 1e-10, d, 1.0)
    corr = cov / np.outer(d, d)
    im   = ax3.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax3, pad=0.02)
    ax3.set_xticks(range(N))
    ax3.set_yticks(range(N))
    ax3.set_xticklabels(names, rotation=35, ha="right", fontsize=6)
    ax3.set_yticklabels(names, fontsize=6)
    for i in range(N):
        for j in range(N):
            ax3.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                     fontsize=5, color="white" if abs(corr[i,j]) > 0.5 else "black")
    ax3.set_title("Correlation Matrix  |  Ledoit-Wolf", fontsize=10)

    fig.suptitle("Portfolio Optimisation Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("PORTFOLIO OPTIMISATION — MARKOWITZ + RISK PARITY")
    print("Ledoit-Wolf Shrinkage Covariance")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    names   = list(assets_data.keys())
    min_len = min(len(df) for df in assets_data.values())
    mat     = np.column_stack([
        assets_data[n]["log_return"].dropna().values[-min_len:]
        for n in names
    ])
    mat = mat[np.isfinite(mat).all(axis=1)]
    T, N = mat.shape
    print(f"\n  Aligned returns: {T} days x {N} assets\n")

    result = efficient_frontier(mat, names)

    port_labels = {"gmv": "Min Variance", "msr": "Max Sharpe",
                   "ew": "Equal Weight", "rp": "Risk Parity"}

    for key in ["gmv", "msr", "ew", "rp"]:
        w   = result[key]["w"]
        r, v, s = result[key]["stats"]
        h_var, h_cvar = portfolio_var(mat, w)
        p_var, p_cvar = parametric_var(mat, w)

        print(f"  [{port_labels[key]}]")
        print(f"    Return={r*100:+.2f}%  Vol={v*100:.2f}%  Sharpe={s:.3f}")
        print(f"    Hist VaR95={h_var*100:.3f}%  CVaR95={h_cvar*100:.3f}%")
        print(f"    Para VaR95={p_var*100:.3f}%  CVaR95={p_cvar*100:.3f}%")

        top = sorted(zip(names, w), key=lambda x: -x[1])
        top_str = "  ".join(f"{n}:{wt:.1%}" for n, wt in top if wt > 0.01)
        print(f"    Top weights: {top_str}")
        print()

    # Risk parity check
    rp_rc = risk_contributions(result["rp"]["w"], result["cov"]) * 100
    print("  Risk Parity — Risk Contributions:")
    for name, rc in zip(names, rp_rc):
        bar = "#" * int(rc / 2)
        print(f"    {name:<10}: {rc:>5.1f}%  {bar}")

    plot_dashboard(result, mat)

    print("\nPortfolio optimisation complete.")


if __name__ == "__main__":
    main()
