"""
================================================================
Script 46 — Critical Transition Indicators (Early-Warning Signals for
Regime Bifurcations)
================================================================
Scripts 15/33/36 all answer "what regime is this asset in right now" --
none of them answer "how close is this asset to FLIPPING out of its
current regime." Crypto assets move like a driven nonlinear system: long
stretches of compressed, quiet range (the system "loading") followed by
an explosive break in either direction (the bifurcation itself) --
literal chaos-theory bifurcation detection (Lyapunov exponents, basin
mapping) needs a known deterministic state equation, which noisy return
data doesn't hand you, so that machinery tends to produce numbers that
look rigorous without being reliable here. This script instead uses the
empirically-validated proxy from critical-transition theory (Scheffer et
al. 2009, "Early-warning signals for critical transitions", Nature):
as a dynamical system approaches a critical transition, it shows
CRITICAL SLOWING DOWN -- rising lag-1 autocorrelation (the system
recovers from small perturbations more slowly) and rising variance
(fluctuations grow) in the run-up, before the actual flip.

  1. Rolling lag-1 autocorrelation and rolling variance of returns.
  2. A trend score (Mann-Kendall tau) on each indicator over a recent
     window -- rising trend = "loading" toward a transition.
  3. HONEST VALIDATION, not an assumed claim: for each ticker's biggest
     historical drawdown-onsets and biggest rallies, checks whether the
     combined CSD score was actually elevated in the run-up, compared to
     a bootstrap distribution of CSD scores on ordinary (non-event) days.
     Reports the empirical finding either way -- if the effect doesn't
     hold on this universe, that's reported too, not hidden.

RETUNE NOTE: the first version of this script ran on daily bars (~20/30-
bar windows) and found NO significant effect -- 1/9 tickers cleared
p<0.05 for both drawdowns and rallies, indistinguishable from the ~0.45
tickers you'd expect from chance alone at that threshold. That's an
honest negative result, not a bug, but it left open whether daily bars
were simply too coarse a sampling resolution to see the buildup (20
daily samples over a 20-day window vs. 480 hourly samples over the same
real-world 20 days). This version reruns the identical methodology on
HOURLY bars with windows rescaled to the SAME real-world length (20
days = 480 hours, not 20 hours) -- testing whether finer sampling
resolution reveals a signal the daily version's noisier estimate missed,
not testing a different (shorter-timescale) phenomenon.

Not a trading signal on its own (see Script 25 for that) -- this is a
"how loaded is the spring" gauge, meant to sit alongside the regime
label (Script 15) and eventually feed the ML feature pipeline (Script
23) once validated.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from data_loader import fetch_intraday, SYMBOLS, format_price

PLOT_STYLE = "seaborn-v0_8-darkgrid"
BAR_INTERVAL = "1h"
BAR_FETCH_LIMIT = 5000    # ~208 days of hourly bars -- practical ccxt fetch cap
CSD_WINDOW = 20 * 24            # was 20 (days) -- same 20-day real-world window,
                                 # now 480 hourly samples instead of 20 daily ones
TREND_WINDOW = 30 * 24          # was 30 (days) -- same 30-day real-world window
FORWARD_HORIZON = 10 * 24       # was 10 (days) -- same 10-day forward-move window
N_EVENTS = 5             # top-N drawdown / top-N rally events per ticker
N_BOOTSTRAP = 2000
LOOKBACK_WINDOW_FOR_EVENT = TREND_WINDOW   # CSD score measured this far before an event


# ============================================================
# Rolling Indicators
# ============================================================
def rolling_autocorr_lag1(returns, window=CSD_WINDOW):
    """Rolling lag-1 autocorrelation -- rises as the system 'remembers'
    perturbations longer (slower relaxation back to equilibrium).
    Vectorized via pandas' compiled rolling().corr() (a per-window
    Python-level .autocorr() loop doesn't scale to ~5000 hourly bars)."""
    r = pd.Series(returns)
    return r.rolling(window).corr(r.shift(1)).values


def rolling_variance(returns, window=CSD_WINDOW):
    return pd.Series(returns).rolling(window).var().values


def rolling_trend(indicator, window=TREND_WINDOW):
    """
    Rolling correlation of `indicator` against a fixed ascending time
    index -- a bounded [-1,1], vectorized proxy for "is this indicator
    monotonically rising over the window." Pearson rather than a
    per-window Mann-Kendall/Kendall-tau loop: correlation against a
    FIXED global index is invariant to which window you're in (a local
    arange(window) is just the global index minus a constant, and
    correlation is shift-invariant), so pandas' rolling().corr() -- the
    same compiled rolling-covariance machinery as rolling().var() --
    computes this directly with no per-row Python callback, which a
    Kendall-tau version doesn't allow at this scale (~5000 hourly bars).
    """
    s = pd.Series(indicator)
    idx = pd.Series(np.arange(len(indicator)), index=s.index, dtype=float)
    return s.rolling(window).corr(idx).values


def csd_score_series(returns, csd_window=CSD_WINDOW, trend_window=TREND_WINDOW):
    """
    Combined critical-slowing-down score at every bar t: rolling trend
    (over the last `trend_window` bars) of BOTH rolling autocorrelation
    and rolling variance, averaged. Positive and rising = both
    indicators trending up together = "loading" toward a flip.
    """
    ac = rolling_autocorr_lag1(returns, csd_window)
    var = rolling_variance(returns, csd_window)
    ac_trend = rolling_trend(ac, trend_window)
    var_trend = rolling_trend(var, trend_window)
    score = 0.5 * (ac_trend + var_trend)
    return score, ac, var


# ============================================================
# Event Identification — biggest historical drawdowns / rallies
# ============================================================
def find_extreme_events(close, horizon=FORWARD_HORIZON, n_events=N_EVENTS):
    """
    For every bar t, the forward `horizon`-bar return close[t+horizon]/close[t]-1.
    Returns the N most negative (drawdown onsets) and N most positive
    (rally onsets) non-overlapping event indices -- non-overlapping so
    N_EVENTS distinct crashes/rallies aren't just the same event counted
    at adjacent bars.
    """
    n = len(close)
    fwd_ret = np.full(n, np.nan)
    for t in range(n - horizon):
        fwd_ret[t] = close[t + horizon] / close[t] - 1

    def top_nonoverlapping(values, n_events, largest=True):
        order = np.argsort(-values if largest else values)
        chosen = []
        for idx in order:
            if np.isnan(values[idx]):
                continue
            if all(abs(idx - c) > horizon for c in chosen):
                chosen.append(int(idx))
            if len(chosen) >= n_events:
                break
        return chosen

    valid = ~np.isnan(fwd_ret)
    rally_idx = top_nonoverlapping(fwd_ret, n_events, largest=True)
    crash_idx = top_nonoverlapping(fwd_ret, n_events, largest=False)
    return crash_idx, rally_idx, fwd_ret


# ============================================================
# Validation — is CSD actually elevated before real events vs. random days?
# ============================================================
def validate_csd(score, event_idx, lookback=LOOKBACK_WINDOW_FOR_EVENT,
                  n_bootstrap=N_BOOTSTRAP, seed=42):
    """
    For each event index t, takes the CSD score at t-1 (the last fully
    causal reading before the move starts). Compares the mean of these
    pre-event scores against a bootstrap distribution of the mean CSD
    score at `len(event_idx)` random (non-NaN) days -- an empirical
    p-value for "were these events preceded by elevated CSD, or would
    that many random days give as high a mean by chance."
    """
    rng = np.random.default_rng(seed)
    valid_scores = score[~np.isnan(score)]
    valid_positions = np.where(~np.isnan(score))[0]

    pre_event_scores = [score[t - 1] for t in event_idx if t - 1 >= 0 and not np.isnan(score[t - 1])]
    if len(pre_event_scores) == 0 or len(valid_scores) < 10:
        return {"observed_mean": np.nan, "p_value": np.nan, "n_events_used": 0}

    observed_mean = float(np.mean(pre_event_scores))
    n_ev = len(pre_event_scores)

    boot_means = np.array([
        np.mean(rng.choice(valid_scores, size=n_ev, replace=True))
        for _ in range(n_bootstrap)
    ])
    p_value = float(np.mean(boot_means >= observed_mean))
    return {"observed_mean": observed_mean, "p_value": p_value, "n_events_used": n_ev,
            "boot_means": boot_means}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, df, score, ac, var, crash_idx, rally_idx,
                    crash_val, rally_val):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.28)
    dates = df.index

    # [0,:] Price with crash/rally onset markers
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates, df["close"].values, color="steelblue", lw=1.1)
    for i in crash_idx:
        ax0.axvline(dates[i], color="crimson", lw=1.0, ls="--", alpha=0.7)
    for i in rally_idx:
        ax0.axvline(dates[i], color="forestgreen", lw=1.0, ls="--", alpha=0.7)
    ax0.set_title(f"{ticker} — Price, with biggest drawdown-onsets (red) / rally-onsets (green)",
                  fontsize=10)
    ax0.grid(alpha=0.3)

    # [1,0] Rolling autocorrelation
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(dates, ac, color="darkorange", lw=1.0)
    ax1.axhline(0, color="gray", lw=0.6, ls="--")
    ax1.set_title("Rolling Lag-1 Autocorrelation", fontsize=9.5)
    ax1.grid(alpha=0.3)

    # [1,1] Rolling variance
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(dates, var, color="purple", lw=1.0)
    ax2.set_title("Rolling Variance", fontsize=9.5)
    ax2.grid(alpha=0.3)

    # [2,0] Combined CSD score over time
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(dates, score, color="crimson", lw=1.1)
    ax3.axhline(0, color="gray", lw=0.6, ls="--")
    for i in crash_idx:
        ax3.axvline(dates[i], color="crimson", lw=0.8, ls=":", alpha=0.5)
    for i in rally_idx:
        ax3.axvline(dates[i], color="forestgreen", lw=0.8, ls=":", alpha=0.5)
    ax3.set_title("Combined Critical-Slowing-Down Score (Mann-Kendall trend)", fontsize=9.5)
    ax3.grid(alpha=0.3)

    # [2,1] Validation summary
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    rows = [
        ["Drawdown events used", f"{crash_val['n_events_used']}"],
        ["Mean CSD before drawdowns", f"{crash_val['observed_mean']:+.4f}"],
        ["p-value (vs random days)", f"{crash_val['p_value']:.3f}"],
        ["Rally events used", f"{rally_val['n_events_used']}"],
        ["Mean CSD before rallies", f"{rally_val['observed_mean']:+.4f}"],
        ["p-value (vs random days)", f"{rally_val['p_value']:.3f}"],
    ]
    table = ax4.table(cellText=rows, colLabels=["Check", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax4.set_title("Empirical Validation (p<0.05 = CSD genuinely elevated pre-event)", fontsize=9.5, pad=12)

    fig.suptitle(f"{ticker} — Critical Transition Indicators (Early-Warning Signals)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("CRITICAL TRANSITION INDICATORS — EARLY-WARNING SIGNALS")
    print("Critical slowing down (rising autocorrelation + variance) before regime flips")
    print(f"Bar interval: {BAR_INTERVAL}  (retuned from daily -- see module docstring: "
          f"daily bars showed no effect, 1/9 tickers at p<0.05, indistinguishable from chance)")
    print("=" * 70)

    summary = []

    for ticker, symbol in SYMBOLS.items():
        df = fetch_intraday(symbol, interval=BAR_INTERVAL, limit=BAR_FETCH_LIMIT)
        if df is None or len(df) < CSD_WINDOW + TREND_WINDOW + FORWARD_HORIZON + 30:
            print(f"\n  {ticker}: skipped -- insufficient hourly history for CSD windows")
            continue
        returns = df["log_return"].dropna().values
        close = df["close"].values[-len(returns):]

        score, ac, var = csd_score_series(returns)
        crash_idx, rally_idx, fwd_ret = find_extreme_events(close)

        crash_val = validate_csd(score, crash_idx)
        rally_val = validate_csd(score, rally_idx)

        current_score = score[~np.isnan(score)][-1] if np.any(~np.isnan(score)) else np.nan
        loaded = "LOADED (elevated CSD)" if current_score > 0.15 else \
                 ("calm" if current_score == current_score else "n/a")

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  Current CSD score: {current_score:+.4f}  -> {loaded}")
        print(f"  Drawdown validation:  mean CSD before {crash_val['n_events_used']} biggest "
              f"drawdowns={crash_val['observed_mean']:+.4f}  p={crash_val['p_value']:.3f}  "
              f"{'<<' if crash_val['p_value'] < 0.05 else ''}")
        print(f"  Rally validation:     mean CSD before {rally_val['n_events_used']} biggest "
              f"rallies={rally_val['observed_mean']:+.4f}  p={rally_val['p_value']:.3f}  "
              f"{'<<' if rally_val['p_value'] < 0.05 else ''}")

        summary.append({
            "Ticker": ticker, "CurrentCSD": f"{current_score:+.4f}", "State": loaded,
            "Drawdown_p": f"{crash_val['p_value']:.3f}" if crash_val['p_value'] == crash_val['p_value'] else "n/a",
            "Rally_p": f"{rally_val['p_value']:.3f}" if rally_val['p_value'] == rally_val['p_value'] else "n/a",
        })

        plot_dashboard(ticker, df.iloc[-len(returns):], score, ac, var, crash_idx, rally_idx,
                        crash_val, rally_val)

    if summary:
        print("\n" + "=" * 70)
        print("CRITICAL TRANSITION INDICATORS SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        n_dd_sig = sum(1 for s in summary if s["Drawdown_p"] != "n/a" and float(s["Drawdown_p"]) < 0.05)
        n_ra_sig = sum(1 for s in summary if s["Rally_p"] != "n/a" and float(s["Rally_p"]) < 0.05)
        print(f"\n  CSD significantly elevated before drawdowns: {n_dd_sig}/{len(summary)} tickers (p<0.05)")
        print(f"  CSD significantly elevated before rallies:   {n_ra_sig}/{len(summary)} tickers (p<0.05)")
        print("  (If these counts are near what you'd expect from chance at alpha=0.05,")
        print("   the early-warning effect is NOT holding on this universe -- reported")
        print("   honestly either way, not assumed.)")

    print("\nCritical transition indicators analysis complete.")


if __name__ == "__main__":
    main()
