"""
================================================================
Script 18 — Copula-Based Correlation Risk Monitor
================================================================
Maps to: Stats L7-L9 (multivariate dependence, copulas)

Standard Pearson correlation underestimates joint tail risk.
This script fits copulas to capture how assets co-move in
crashes vs normal times, then adjusts position sizes.

Methods:
  • Empirical copula from rank-transformed returns
  • Gaussian copula (baseline)
  • Student-t copula (captures tail dependence)
  • Kendall tau & Spearman rho (rank correlations)
  • Tail dependence coefficient estimation
  • Correlation breakdown detection (rolling vs crisis)
  • Concentration risk scoring per trade plan
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, optimize
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS

PLOT_STYLE = "seaborn-v0_8-darkgrid"
TAIL_THRESHOLD = 0.10


# ============================================================
# Rank Correlation Measures
# ============================================================
def rank_correlations(returns_matrix, names):
    n = returns_matrix.shape[1]
    kendall = np.zeros((n, n))
    spearman = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            kendall[i, j] = stats.kendalltau(returns_matrix[:, i],
                                              returns_matrix[:, j])[0]
            spearman[i, j] = stats.spearmanr(returns_matrix[:, i],
                                              returns_matrix[:, j])[0]
    return kendall, spearman


# ============================================================
# Tail Dependence
# ============================================================
def empirical_tail_dependence(u, v, q=0.10):
    """
    Lower tail dependence: P(V <= q | U <= q)
    Upper tail dependence: P(V >= 1-q | U >= 1-q)
    """
    n = len(u)
    lower_mask = (u <= q) & (v <= q)
    upper_mask = (u >= 1 - q) & (v >= 1 - q)
    denom_lo = max((u <= q).sum(), 1)
    denom_up = max((u >= 1 - q).sum(), 1)
    lambda_L = lower_mask.sum() / denom_lo
    lambda_U = upper_mask.sum() / denom_up
    return float(lambda_L), float(lambda_U)


def tail_dependence_matrix(returns_matrix, names, q=TAIL_THRESHOLD):
    T, N = returns_matrix.shape
    # Convert to pseudo-observations (uniform marginals)
    U = np.zeros_like(returns_matrix)
    for j in range(N):
        U[:, j] = stats.rankdata(returns_matrix[:, j]) / (T + 1)

    lower_td = np.zeros((N, N))
    upper_td = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            lower_td[i, j], upper_td[i, j] = empirical_tail_dependence(
                U[:, i], U[:, j], q)
    return lower_td, upper_td, U


# ============================================================
# Gaussian Copula
# ============================================================
def fit_gaussian_copula(U):
    """Fit Gaussian copula via MLE on rank-transformed data."""
    N = U.shape[1]
    Z = stats.norm.ppf(np.clip(U, 1e-6, 1 - 1e-6))
    corr = np.corrcoef(Z.T)
    corr = np.clip(corr, -0.999, 0.999)
    np.fill_diagonal(corr, 1.0)
    return corr


# ============================================================
# Student-t Copula (simplified)
# ============================================================
def fit_t_copula(U, max_nu=30):
    """
    Fit t-copula: estimate correlation + degrees of freedom.
    Lower nu = heavier tails = more tail dependence.
    """
    N = U.shape[1]
    best_nu, best_ll = 5, -np.inf

    for nu in range(3, max_nu + 1):
        Z = stats.t.ppf(np.clip(U, 1e-6, 1 - 1e-6), df=nu)
        corr = np.corrcoef(Z.T)
        try:
            ll = 0
            for t in range(len(U)):
                ll += stats.multivariate_t.logpdf(Z[t], loc=np.zeros(N),
                                                   shape=corr, df=nu)
            if ll > best_ll:
                best_ll, best_nu = ll, nu
        except Exception:
            continue

    Z = stats.t.ppf(np.clip(U, 1e-6, 1 - 1e-6), df=best_nu)
    corr = np.corrcoef(Z.T)
    np.fill_diagonal(corr, 1.0)

    # Analytical tail dependence for t-copula
    lambda_tail = 2 * stats.t.cdf(
        -np.sqrt((best_nu + 1) * (1 - corr) / (1 + corr)),
        df=best_nu + 1)

    return corr, best_nu, lambda_tail


# ============================================================
# Correlation Breakdown Detection
# ============================================================
def correlation_breakdown(returns_matrix, names, window=60, crisis_q=0.10):
    """
    Compare correlation in normal times vs crisis times.
    Crisis = days where portfolio return is in the bottom quantile.
    """
    T, N = returns_matrix.shape
    port_r = returns_matrix.mean(axis=1)
    threshold = np.percentile(port_r, crisis_q * 100)

    normal_mask = port_r > threshold
    crisis_mask = port_r <= threshold

    corr_normal = np.corrcoef(returns_matrix[normal_mask].T) if normal_mask.sum() > 10 else np.eye(N)
    corr_crisis = np.corrcoef(returns_matrix[crisis_mask].T) if crisis_mask.sum() > 5 else np.eye(N)

    breakdown = corr_crisis - corr_normal
    return corr_normal, corr_crisis, breakdown


# ============================================================
# Concentration Risk Score
# ============================================================
def concentration_score(positions, lower_td, names):
    """
    Given a list of (ticker, direction, weight) positions,
    compute how much hidden tail risk exists from correlated bets.
    Returns a risk multiplier: 1.0 = no concentration, >1 = danger.
    """
    if len(positions) <= 1:
        return 1.0, {}

    idx_map = {n: i for i, n in enumerate(names)}
    active  = [(t, d, w) for t, d, w in positions if t in idx_map]

    # Normalize weights to fractions (they come in as percentages)
    total_w = sum(abs(w) for _, _, w in active)
    if total_w < 1e-6:
        return 1.0, {}

    total_tail_overlap = 0.0
    pair_risks = {}
    n_pairs = 0
    for i, (t1, d1, w1) in enumerate(active):
        for j, (t2, d2, w2) in enumerate(active):
            if i >= j:
                continue
            ii, jj = idx_map[t1], idx_map[t2]
            td = lower_td[ii, jj]
            n_pairs += 1

            # Weights as fractions of total allocation
            wf1 = abs(w1) / total_w
            wf2 = abs(w2) / total_w

            if d1 == d2:
                overlap = td * (wf1 + wf2) / 2
            else:
                overlap = -td * (wf1 + wf2) / 4

            total_tail_overlap += max(overlap, 0)
            if td > 0.15:
                pair_risks[f"{t1}/{t2}"] = round(td, 3)

    risk_mult = 1.0 + total_tail_overlap
    return round(min(risk_mult, 3.0), 2), pair_risks


# ============================================================
# Plotting
# ============================================================
def plot_copula_dashboard(names, pearson, kendall, lower_td, upper_td,
                          corr_normal, corr_crisis, t_nu):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)
    N   = len(names)

    def _heatmap(ax, mat, title, cmap="RdBu_r", vmin=-1, vmax=1):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
        ax.set_yticklabels(names, fontsize=5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=4, color="white" if abs(mat[i,j]) > 0.5 else "black")

    _heatmap(fig.add_subplot(gs[0, 0]), pearson,
             "Pearson Correlation")
    _heatmap(fig.add_subplot(gs[0, 1]), kendall,
             "Kendall Tau (Rank)")
    _heatmap(fig.add_subplot(gs[0, 2]), lower_td,
             f"Lower Tail Dependence (q={TAIL_THRESHOLD:.0%})",
             cmap="YlOrRd", vmin=0, vmax=1)

    _heatmap(fig.add_subplot(gs[1, 0]), corr_normal,
             "Correlation — Normal Times")
    _heatmap(fig.add_subplot(gs[1, 1]), corr_crisis,
             "Correlation — Crisis Times")

    ax5 = fig.add_subplot(gs[1, 2])
    breakdown = corr_crisis - corr_normal
    _heatmap(ax5, breakdown,
             f"Breakdown (Crisis - Normal)  |  t-copula nu={t_nu}",
             cmap="RdYlGn_r", vmin=-0.5, vmax=0.5)

    fig.suptitle("Copula-Based Correlation Risk Monitor", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Public API for Script 17
# ============================================================
def get_concentration_adjustment(assets_data, positions):
    """
    Called by trade planner to adjust position sizes.
    positions: list of (ticker, direction, weight_pct)
    Returns: risk_multiplier, pair_risks_dict
    """
    names   = list(assets_data.keys())
    min_len = min(len(df) for df in assets_data.values())
    mat     = np.column_stack([
        assets_data[n]["log_return"].dropna().values[-min_len:]
        for n in names
    ])
    mat = mat[np.isfinite(mat).all(axis=1)]

    lower_td, _, _ = tail_dependence_matrix(mat, names)
    return concentration_score(positions, lower_td, names)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("COPULA-BASED CORRELATION RISK MONITOR")
    print("Tail Dependence + Correlation Breakdown + Concentration Risk")
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
    print(f"\n  Returns matrix: {T} days x {N} assets")

    # Pearson
    pearson = np.corrcoef(mat.T)
    print(f"  Pearson correlation computed.")

    # Rank correlations
    print(f"  Computing Kendall tau + Spearman rho ...")
    kendall, spearman = rank_correlations(mat, names)

    # Tail dependence
    print(f"  Computing empirical tail dependence (q={TAIL_THRESHOLD:.0%}) ...")
    lower_td, upper_td, U = tail_dependence_matrix(mat, names)

    # Gaussian copula
    print(f"  Fitting Gaussian copula ...")
    gauss_corr = fit_gaussian_copula(U)

    # t-copula
    print(f"  Fitting Student-t copula (this may take a moment) ...")
    t_corr, t_nu, t_lambda = fit_t_copula(U)
    print(f"  t-copula: nu={t_nu} (lower nu = heavier tails)")

    # Correlation breakdown
    corr_normal, corr_crisis, breakdown = correlation_breakdown(mat, names)
    avg_breakdown = float(np.mean(breakdown[np.triu_indices(N, k=1)]))
    print(f"  Avg correlation increase in crisis: {avg_breakdown:+.3f}")

    # Print worst tail-dependent pairs
    print(f"\n  HIGHEST TAIL DEPENDENCE PAIRS (lower tail):")
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((names[i], names[j], lower_td[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for a, b, td in pairs[:10]:
        pearson_ij = pearson[names.index(a), names.index(b)]
        print(f"    {a:<8} / {b:<8}: tail_dep={td:.3f}  "
              f"pearson={pearson_ij:.3f}  "
              f"{'!! HIDDEN RISK' if td > pearson_ij + 0.1 else ''}")

    # Print assets with highest avg tail dependence
    print(f"\n  SYSTEMIC RISK RANKING (avg lower tail dependence):")
    avg_td = lower_td.mean(axis=1)
    ranked = sorted(zip(names, avg_td), key=lambda x: -x[1])
    for name, atd in ranked:
        bar = "#" * int(atd * 40)
        print(f"    {name:<10}: {atd:.3f}  {bar}")

    plot_copula_dashboard(names, pearson, kendall, lower_td, upper_td,
                          corr_normal, corr_crisis, t_nu)

    print("\nCopula risk analysis complete.")


if __name__ == "__main__":
    main()
