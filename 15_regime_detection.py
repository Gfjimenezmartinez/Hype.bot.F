"""
================================================================
Script 15 — Market Regime Detection
================================================================
Classifies each trading day into one of 3 regimes:
  0: LOW-VOL TRENDING   (smooth trends, small moves)
  1: MEAN-REVERTING     (choppy, range-bound)
  2: CRISIS / HIGH-VOL  (large moves, tail events)

Methods:
  • Gaussian HMM (hmmlearn if installed, else manual EM)
  • Rolling-statistics fallback (vol percentile + return autocorr)
  • Regime-conditioned signal gating

Each regime has different optimal strategies:
  - Regime 0 → trend-following works
  - Regime 1 → mean-reversion works
  - Regime 2 → reduce exposure / hedge
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
N_REGIMES  = 3
VOL_WINDOW = 20
RET_WINDOW = 10

REGIME_NAMES = {0: "Low-Vol Trend", 1: "Mean-Revert", 2: "Crisis/High-Vol"}
REGIME_STRAT = {0: "TREND-FOLLOW", 1: "MEAN-REVERT", 2: "REDUCE/HEDGE"}


# ============================================================
# Gaussian HMM (try hmmlearn, fallback to manual)
# ============================================================
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


def fit_hmm(features, n_states=N_REGIMES):
    if HMM_AVAILABLE:
        return _fit_hmmlearn(features, n_states)
    return _fit_manual(features, n_states)


def _fit_hmmlearn(features, n_states):
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=200, random_state=42, verbose=False)
    model.fit(features)
    states = model.predict(features)
    states = _relabel_by_volatility(states, features, n_states)
    return states, {
        "means": model.means_,
        "transmat": model.transmat_,
        "method": "hmmlearn",
    }


def _fit_manual(features, n_states):
    """K-means + Viterbi-style classification when hmmlearn unavailable."""
    from sklearn.cluster import KMeans
    km     = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    states = km.fit_predict(features)
    states = _relabel_by_volatility(states, features, n_states)

    transmat = np.zeros((n_states, n_states))
    for i in range(1, len(states)):
        transmat[states[i-1], states[i]] += 1
    row_sums = transmat.sum(axis=1, keepdims=True)
    transmat = np.where(row_sums > 0, transmat / row_sums, 1.0 / n_states)

    means = np.array([features[states == s].mean(axis=0) for s in range(n_states)])
    return states, {
        "means": means,
        "transmat": transmat,
        "method": "kmeans-fallback",
    }


def _relabel_by_volatility(states, features, n_states):
    """Ensure regime 0=low vol, 1=mid, 2=high vol."""
    vol_col = 1 if features.shape[1] > 1 else 0
    regime_vols = [features[states == s, vol_col].mean() for s in range(n_states)]
    order = np.argsort(regime_vols)
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[s] for s in states])


# ============================================================
# Feature Engineering
# ============================================================
def build_features(df):
    r = df["log_return"].values
    n = len(r)

    vol_20  = pd.Series(r).rolling(VOL_WINDOW).std().values
    ret_10  = pd.Series(r).rolling(RET_WINDOW).mean().values
    autocorr = pd.Series(r).rolling(30).apply(
        lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 else 0,
        raw=True).values

    features = np.column_stack([ret_10, vol_20, autocorr])
    valid    = ~np.isnan(features).any(axis=1)
    return features, valid


# ============================================================
# Quick API for other scripts
# ============================================================
def detect_regime(df):
    """
    Returns (current_regime_int, regime_name, recommended_strategy)
    for use by signal scripts (11, 14, 16).

    Note: KMeans clusters are labeled by *relative* vol, so the
    highest-vol cluster is always called "Crisis" even in a calm
    tape (e.g. SPY at 16% ann vol). Guard: only emit Crisis if the
    current 20d vol is genuinely elevated vs the asset's own history
    (top quartile); otherwise downgrade to Mean-Revert.
    """
    features, valid = build_features(df)
    feat_clean = features[valid]
    if len(feat_clean) < 60:
        return 1, REGIME_NAMES[1], REGIME_STRAT[1]
    states, _ = fit_hmm(feat_clean)
    current = int(states[-1])

    if current == 2:
        vol_series = pd.Series(df["log_return"].values).rolling(VOL_WINDOW).std()
        vol_series = vol_series.dropna()
        if len(vol_series) > 40:
            cur_pct = float((vol_series <= vol_series.iloc[-1]).mean())
            if cur_pct < 0.75:
                current = 1
    return current, REGIME_NAMES[current], REGIME_STRAT[current]


# ============================================================
# Regime Statistics
# ============================================================
def regime_stats(returns, states):
    out = {}
    for s in range(N_REGIMES):
        mask = states == s
        if mask.sum() == 0:
            continue
        r = returns[mask]
        out[s] = {
            "count":     int(mask.sum()),
            "pct":       float(mask.mean() * 100),
            "mean_ret":  float(r.mean() * 100),
            "vol":       float(r.std() * np.sqrt(365) * 100),   # crypto trades 24/7
            "sharpe":    float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0,
            "min":       float(r.min() * 100),
            "max":       float(r.max() * 100),
            "skew":      float(pd.Series(r).skew()),
        }
    return out


# ============================================================
# Plotting
# ============================================================
def plot_regime(ticker, df, states, valid_mask, info, rstats):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)

    dates  = df.index[valid_mask]
    close  = df["close"].values[valid_mask]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    # [0,0:1] Price + regime overlay
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates, close, color="gray", lw=1, alpha=0.6)
    for s in range(N_REGIMES):
        mask = states == s
        ax0.scatter(dates[mask], close[mask], c=colors[s], s=4,
                    alpha=0.7, label=f"{REGIME_NAMES[s]} ({rstats.get(s, {}).get('pct', 0):.0f}%)")
    ax0.set_ylabel("Price")
    ax0.legend(fontsize=8, loc="upper left")
    ax0.set_title(f"{ticker} — Price Colored by Regime", fontsize=11)
    ax0.grid(alpha=0.3)

    # [1,0] Regime timeline
    ax1 = fig.add_subplot(gs[1, 0])
    for s in range(N_REGIMES):
        mask = states == s
        ax1.fill_between(range(len(states)), 0, 1, where=mask,
                         color=colors[s], alpha=0.6, label=REGIME_NAMES[s])
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_xlabel("Trading Days")
    ax1.set_title("Regime Timeline", fontsize=10)
    ax1.legend(fontsize=7, loc="upper right")

    # [1,1] Transition matrix
    ax2 = fig.add_subplot(gs[1, 1])
    tm  = info["transmat"]
    im  = ax2.imshow(tm * 100, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax2, pad=0.02, label="%")
    for i in range(N_REGIMES):
        for j in range(N_REGIMES):
            ax2.text(j, i, f"{tm[i,j]*100:.1f}%", ha="center", va="center", fontsize=9)
    labels = [REGIME_NAMES[i][:8] for i in range(N_REGIMES)]
    ax2.set_xticks(range(N_REGIMES))
    ax2.set_yticks(range(N_REGIMES))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_title(f"Transition Matrix ({info['method']})", fontsize=10)

    # [2,0] Return distribution by regime
    ax3 = fig.add_subplot(gs[2, 0])
    returns = df["log_return"].values[valid_mask]
    for s in range(N_REGIMES):
        mask = states == s
        if mask.sum() > 5:
            ax3.hist(returns[mask] * 100, bins=40, alpha=0.5,
                     color=colors[s], label=REGIME_NAMES[s], density=True)
    ax3.set_xlabel("Daily Return (%)")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=7)
    ax3.set_title("Return Distribution by Regime", fontsize=10)
    ax3.grid(alpha=0.3)

    # [2,1] Stats table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    rows = []
    for s in range(N_REGIMES):
        rs = rstats.get(s, {})
        rows.append([
            REGIME_NAMES[s],
            f"{rs.get('pct', 0):.0f}%",
            f"{rs.get('vol', 0):.1f}%",
            f"{rs.get('sharpe', 0):.2f}",
            REGIME_STRAT[s],
        ])
    table = ax4.table(cellText=rows,
                      colLabels=["Regime", "Time%", "AnnVol", "Sharpe", "Strategy"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax4.set_title("Regime Statistics & Recommended Strategy", fontsize=10, pad=15)

    fig.suptitle(f"{ticker} — Regime Detection Dashboard", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("MARKET REGIME DETECTION")
    print(f"Method: {'hmmlearn GaussianHMM' if HMM_AVAILABLE else 'KMeans fallback'}")
    print(f"Regimes: {N_REGIMES} (Low-Vol Trend / Mean-Revert / Crisis)")
    print("=" * 65)

    if not HMM_AVAILABLE:
        print("  [info] pip install hmmlearn for full HMM — using KMeans fallback\n")

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*50}")

        try:
            features, valid = build_features(df)
            feat_clean = features[valid]
            if len(feat_clean) < 60:
                print("  Insufficient data for regime detection.")
                continue

            states, info = fit_hmm(feat_clean)
            returns = df["log_return"].values[valid]
            rstats  = regime_stats(returns, states)

            current = states[-1]
            print(f"  Current regime: {REGIME_NAMES[current]} → {REGIME_STRAT[current]}")
            print(f"  Method: {info['method']}")

            for s in range(N_REGIMES):
                rs = rstats.get(s, {})
                print(f"    Regime {s} ({REGIME_NAMES[s]}): "
                      f"{rs.get('pct', 0):.0f}% of days  "
                      f"vol={rs.get('vol', 0):.1f}%  "
                      f"sharpe={rs.get('sharpe', 0):.2f}")

            print(f"  Transition probabilities from current ({REGIME_NAMES[current]}):")
            tm = info["transmat"]
            for s in range(N_REGIMES):
                print(f"    → {REGIME_NAMES[s]}: {tm[current, s]*100:.1f}%")

            summary.append({
                "Ticker": ticker,
                "Current": REGIME_NAMES[current],
                "Strategy": REGIME_STRAT[current],
                "Stay%": f"{tm[current, current]*100:.0f}%",
                "Vol%": f"{rstats.get(current, {}).get('vol', 0):.0f}%",
            })

            plot_regime(ticker, df, states, valid, info, rstats)

        except Exception as e:
            print(f"  Error: {e}")

    if summary:
        print("\n" + "=" * 65)
        print("REGIME SUMMARY — CURRENT STATE")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))

        regime_counts = {}
        for s in summary:
            r = s["Current"]
            regime_counts[r] = regime_counts.get(r, 0) + 1
        print(f"\n  Market regime distribution:")
        for r, c in sorted(regime_counts.items(), key=lambda x: -x[1]):
            print(f"    {r}: {c}/{len(summary)} assets")

    print("\nRegime detection complete.")


if __name__ == "__main__":
    main()
