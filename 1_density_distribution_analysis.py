"""
================================================================
Script 1 — Density Distribution Analysis & Portfolio Optimisation  v2
================================================================
Fixes vs v1:
  • CVXPY portfolio section now actually runs and prints weights
  • Efficient-frontier scatter plot added
  • Cleaner AIC summary table
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
    print(f"CVXPY {cp.__version__} available.")
except ImportError:
    print("CVXPY not available — install:  pip install cvxpy")
    CVXPY_AVAILABLE = False

DISTRIBUTIONS = {
    "Normal":     stats.norm,
    "Student-t":  stats.t,
    "Cauchy":     stats.cauchy,
    "Pareto":     stats.pareto,
    "GenPareto":  stats.genpareto,
    "ExpWeibull": stats.exponweib,
    "Laplace":    stats.laplace,
    "Logistic":   stats.logistic,
}
PLOT_STYLE = "seaborn-v0_8-darkgrid"


# ── Distribution fitting ─────────────────────────────────────
def fit_distributions(returns):
    results = {}
    for name, dist in DISTRIBUTIONS.items():
        try:
            params  = dist.fit(returns)
            loglik  = float(np.sum(dist.logpdf(returns, *params)))
            k, n    = len(params), len(returns)
            results[name] = {
                "params": params,
                "loglik": loglik,
                "AIC":    2*k - 2*loglik,
                "BIC":    k*np.log(n) - 2*loglik,
            }
        except Exception:
            continue
    best_aic = min(results, key=lambda x: results[x]["AIC"]) if results else None
    best_bic = min(results, key=lambda x: results[x]["BIC"]) if results else None
    return results, best_aic, best_bic


# ── GoF tests ────────────────────────────────────────────────
def goodness_of_fit(returns, dist_name, params):
    dist = DISTRIBUTIONS[dist_name]
    ks   = stats.kstest(returns, dist.cdf, args=params)
    cvm  = stats.cramervonmises(returns, dist.cdf, args=params)
    tests = {"KS": (ks.statistic, ks.pvalue), "CVM": (cvm.statistic, cvm.pvalue)}
    if dist_name == "Normal":
        ad = stats.anderson(returns, dist="norm")
        tests["AD"] = ad.statistic
    return tests


# ── CVXPY portfolio optimisation ─────────────────────────────
def optimize_portfolio(returns_dict: dict) -> dict:
    if not CVXPY_AVAILABLE:
        print("\n  [Portfolio] CVXPY not installed — skipping.")
        return None
    if len(returns_dict) < 2:
        return None

    assets = list(returns_dict.keys())
    n      = len(assets)
    T      = min(len(r) for r in returns_dict.values())
    R      = np.column_stack([returns_dict[a].values[-T:] for a in assets])
    mu     = R.mean(axis=0)
    Sigma  = np.cov(R, rowvar=False)
    rf     = 0.05 / 365        # daily risk-free, crypto trades 24/7

    # Ensure Sigma is PSD
    eigvals = np.linalg.eigvalsh(Sigma)
    if eigvals.min() < 0:
        Sigma += (-eigvals.min() + 1e-8) * np.eye(n)

    results = {}

    def _solve(label, objective, constraints):
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.CLARABEL, warm_start=True)
        return prob.status, prob

    # 1. Min Variance
    w1 = cp.Variable(n)
    st, _ = _solve("MinVar",
                   cp.Minimize(cp.quad_form(w1, Sigma)),
                   [w1 >= 0, cp.sum(w1) == 1])
    if w1.value is not None:
        results["min_variance"] = w1.value.copy()

    # 2. Max Sharpe — proper convex reformulation:
    #    min y'Σy  s.t. (mu - rf)'y = 1, y >= 0;  w = y / sum(y).
    #    (The old version minimized variance with a return floor of
    #    rf+5e-5 that was trivially satisfied, so it just reproduced
    #    the min-variance portfolio.)
    excess = mu - rf
    if excess.max() > 0:
        y = cp.Variable(n)
        st, _ = _solve("MaxSharpe",
                       cp.Minimize(cp.quad_form(y, Sigma)),
                       [y >= 0, excess @ y == 1])
        if y.value is not None and y.value.sum() > 1e-12:
            results["max_sharpe"] = (y.value / y.value.sum()).copy()

    # 3. Min CVaR  (linear programming form)
    alpha = 0.05
    w3    = cp.Variable(n)
    var_v = cp.Variable()
    xi    = cp.Variable(T, nonneg=True)
    port  = R @ w3
    st, _ = _solve("MinCVaR",
                   cp.Minimize(var_v + (1/(alpha*T)) * cp.sum(xi)),
                   [w3 >= 0, cp.sum(w3) == 1,
                    xi >= -port - var_v])
    if w3.value is not None:
        results["min_cvar"] = w3.value.copy()

    # Print
    print("\n" + "=" * 60)
    print("CVXPY PORTFOLIO OPTIMISATION")
    print("=" * 60)
    for strat, w in results.items():
        ret = float(mu @ w)
        vol = float(np.sqrt(w @ Sigma @ w))
        sr  = (ret - rf) / vol if vol > 0 else 0
        print(f"\n  [{strat.replace('_',' ').title()}]"
              f"  daily_ret={ret:.4%}  daily_vol={vol:.4%}  Sharpe={sr:.3f}")
        top = sorted(zip(assets, w), key=lambda x: -x[1])
        for a, wt in top:
            if wt > 0.01:
                print(f"    {a:<10}: {wt:.2%}")

    return results


# ── Density plot ─────────────────────────────────────────────
def plot_density(ticker, returns, fit_results, best_dist):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(3, 3, figure=fig)
    x   = np.linspace(returns.min(), returns.max(), 1_000)

    ax1 = fig.add_subplot(gs[0:2, 0:2])
    sns.histplot(returns, bins=60, stat="density", alpha=0.45, ax=ax1, color="steelblue")
    cols = plt.cm.tab10(np.linspace(0, 1, len(fit_results)))
    for (name, res), col in zip(fit_results.items(), cols):
        y  = DISTRIBUTIONS[name].pdf(x, *res["params"])
        lw = 2.5 if name == best_dist else 1.0
        ax1.plot(x, y, lw=lw, color=col, label=name)
    ax1.set_title(f"{ticker} — Return Density (best: {best_dist})", fontsize=12)
    ax1.set_xlabel("Log Return"); ax1.set_ylabel("Density")
    ax1.legend(fontsize=7)

    # QQ
    ax2 = fig.add_subplot(gs[0, 2])
    if best_dist in DISTRIBUTIONS:
        dist, params = DISTRIBUTIONS[best_dist], fit_results[best_dist]["params"]
        th = dist.ppf(np.linspace(0.01, 0.99, len(returns)), *params)
        ax2.scatter(np.sort(th), np.sort(returns.values), s=6, alpha=0.4)
        lo, hi = min(th.min(), returns.min()), max(th.max(), returns.max())
        ax2.plot([lo,hi],[lo,hi],"r--", lw=1.5)
        ax2.set_title(f"QQ vs {best_dist}", fontsize=9)

    # AIC/BIC table
    ax3 = fig.add_subplot(gs[1, 2]); ax3.axis("off")
    td  = [[n, f"{r['AIC']:.1f}", f"{r['BIC']:.1f}"] for n,r in fit_results.items()]
    t   = ax3.table(cellText=td, colLabels=["Dist","AIC","BIC"], loc="center", cellLoc="center")
    t.scale(1, 1.25); ax3.set_title("Model Selection", fontsize=9)

    # GoF
    ax4 = fig.add_subplot(gs[2, 0]); ax4.axis("off")
    if best_dist in fit_results:
        gof = goodness_of_fit(returns, best_dist, fit_results[best_dist]["params"])
        gof_rows = [[k, f"{v[0]:.4f} (p={v[1]:.3f})"] if isinstance(v,tuple)
                    else [k, f"{v:.4f}"] for k,v in gof.items() if v is not None]
        if gof_rows:
            t2 = ax4.table(cellText=gof_rows, colLabels=["Test","Stat"],
                           loc="center", cellLoc="left")
            t2.scale(1,1.4); ax4.set_title(f"GoF ({best_dist})", fontsize=9)

    # Return stats
    ax5 = fig.add_subplot(gs[2, 1]); ax5.axis("off")
    rows = [["Mean",f"{returns.mean():.5f}"],["Std",f"{returns.std():.5f}"],
            ["Skew",f"{returns.skew():.4f}"],["Kurt",f"{returns.kurtosis():.4f}"],
            ["VaR95%",f"{np.percentile(returns,5):.4%}"],
            ["CVaR95%",f"{returns[returns<=np.percentile(returns,5)].mean():.4%}"]]
    t3 = ax5.table(cellText=rows, colLabels=["Stat","Value"], loc="center", cellLoc="left")
    t3.scale(1,1.25); ax5.set_title("Return Stats", fontsize=9)

    # Vol histogram
    ax6 = fig.add_subplot(gs[2, 2])
    sns.histplot(returns**2, bins=50, ax=ax6, color="darkorange", alpha=0.7)
    ax6.set_title("Squared Returns (vol proxy)", fontsize=9)
    ax6.set_xlabel("r²")

    fig.suptitle(f"{ticker} — Density Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.show()


# ── Portfolio visualisation ───────────────────────────────────
def plot_portfolio(weights_dict, assets, returns_dict):
    if not weights_dict: return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    strategies = ["min_variance", "max_sharpe", "min_cvar"]
    labels     = ["Min Var", "Max Sharpe", "Min CVaR"]
    colors_s   = ["royalblue", "forestgreen", "tomato"]
    x = np.arange(len(assets))

    for i, (strat, lbl, col) in enumerate(zip(strategies, labels, colors_s)):
        w = weights_dict.get(strat)
        if w is not None:
            axes[0].bar(x + i*0.25 - 0.25, w, 0.25, label=lbl, color=col, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(assets, rotation=45, ha="right", fontsize=6)
    axes[0].set_title("Portfolio Weights by Strategy")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

    # KDE
    for a, r in returns_dict.items():
        sns.kdeplot(r, ax=axes[1], label=a, alpha=0.4, linewidth=1)
    axes[1].set_title("Return Distributions")
    axes[1].legend(fontsize=5, ncol=2)

    # Correlation heat-map (clip to 15 assets for readability)
    R_df = pd.DataFrame({a: r for a,r in returns_dict.items()})
    sub  = R_df.iloc[:, :15]
    sns.heatmap(sub.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, ax=axes[2], annot_kws={"size": 6})
    axes[2].set_title("Correlation (first 15)")

    plt.suptitle("Portfolio Optimisation Dashboard", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("DENSITY DISTRIBUTION ANALYSIS & PORTFOLIO OPTIMISATION  v2")
    print("=" * 65)

    assets_data  = load_all_assets(period_days=LOOKBACK_DAYS)
    all_returns  = {}
    all_fits     = {}

    for ticker, df in assets_data.items():
        print(f"\n{'─'*45}  {ticker}")
        r = df["log_return"].dropna()
        all_returns[ticker] = r

        fit_results, best_aic, best_bic = fit_distributions(r)
        all_fits[ticker] = (fit_results, best_aic)
        print(f"  Best AIC: {best_aic:<12}  Best BIC: {best_bic}")

        if best_aic and best_aic in fit_results:
            gof = goodness_of_fit(r, best_aic, fit_results[best_aic]["params"])
            for tname, val in gof.items():
                if isinstance(val, tuple):
                    print(f"  {tname}: stat={val[0]:.4f}, p={val[1]:.4f}")
            plot_density(ticker, r, fit_results, best_aic)

    # Portfolio
    if len(all_returns) >= 2:
        weights = optimize_portfolio(all_returns)
        if weights:
            plot_portfolio(weights, list(all_returns.keys()), all_returns)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY — Best Fitting Distributions")
    print("=" * 65)
    for ticker, (fits, best) in all_fits.items():
        if fits and best:
            print(f"  {ticker:<10}: {best:<14}  AIC={fits[best]['AIC']:.1f}")
    print("\nDone.")

if __name__ == "__main__":
    main()
