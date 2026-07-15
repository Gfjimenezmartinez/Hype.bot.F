"""
================================================================
Script 42 — Estimation Theory V: Kalman Filter (RTS Smoother)
================================================================
Covers: the Kalman filter -- specifically the fixed-interval smoother,
which nothing in Scripts 20/21 implements. Those scripts all run the
forward Kalman filter only: x_hat(k) uses data through time k, which is
exactly right for a LIVE signal (you can't see the future when trading),
but it's not the best possible estimate of what state the system was
actually in at time k using the FULL dataset. The Rauch-Tung-Striebel
(RTS) smoother is the backward pass that produces that best-possible
in-sample estimate.

This is explicitly NOT a live trading tool -- smoothing uses future
data, so a smoothed "trend at time k" could never have been known at
time k. Its value is retrospective: cleaner in-sample trend estimates
for research/diagnostics (e.g. checking whether Script 20's forward
observer is lagging badly, or getting a better denoised trend series to
validate other scripts against), not a signal Script 17 could ever use.

Verified with a hard, guaranteed mathematical property: the smoothed
covariance can never exceed the filtered covariance at any timestep
(smoothing only adds information, never removes it) -- checked as a PSD
matrix inequality (P_filt - P_smooth must have no negative eigenvalues),
not just "smoothed variance looks smaller on average."

Reuses Script 20's build_state_space (same A, C, Q, R) rather than
redefining the state-space model.
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

_r20 = _im("20_adaptive_forecast")
build_state_space = _r20.build_state_space

PLOT_STYLE = "seaborn-v0_8-darkgrid"


# ============================================================
# Forward Kalman Filter — stores the FULL per-timestep history
# (filtered + predicted mean/covariance) that the RTS backward pass needs.
# Script 20's kalman_observer only keeps the final P, not the full history.
# ============================================================
def kalman_filter_full(y, A, C, Q, R, init_vol=None):
    n = A.shape[0]
    T = len(y)
    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))
    x_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))

    log_vol_init = np.log(max(init_vol, 1e-4)) if init_vol else np.log(0.02)
    x_filt[0] = [y[0], 0.0, log_vol_init]
    P_filt[0] = np.eye(n) * 0.1
    x_pred[0], P_pred[0] = x_filt[0], P_filt[0]

    for k in range(T - 1):
        x_p = A @ x_filt[k]
        P_p = A @ P_filt[k] @ A.T + Q
        x_pred[k + 1], P_pred[k + 1] = x_p, P_p

        S = C @ P_p @ C.T + R
        K = P_p @ C.T @ np.linalg.inv(S)
        innov = y[k + 1] - C @ x_p
        x_filt[k + 1] = x_p + (K @ innov.reshape(-1)).flatten()
        P_filt[k + 1] = (np.eye(n) - K @ C) @ P_p

    return x_filt, P_filt, x_pred, P_pred


# ============================================================
# RTS (Rauch-Tung-Striebel) Backward Smoother
# ============================================================
def rts_smoother(x_filt, P_filt, x_pred, P_pred, A):
    T, n = x_filt.shape
    x_smooth = np.zeros((T, n))
    P_smooth = np.zeros((T, n, n))
    x_smooth[-1], P_smooth[-1] = x_filt[-1], P_filt[-1]

    for k in range(T - 2, -1, -1):
        C_gain = P_filt[k] @ A.T @ np.linalg.inv(P_pred[k + 1])
        x_smooth[k] = x_filt[k] + C_gain @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + C_gain @ (P_smooth[k + 1] - P_pred[k + 1]) @ C_gain.T

    return x_smooth, P_smooth


def verify_smoother_covariance_reduction(P_filt, P_smooth, tol=1e-8):
    """
    Hard mathematical guarantee: smoothing can only add information, so
    P_filt[k] - P_smooth[k] must be positive semi-definite (no negative
    eigenvalues, up to numerical tolerance) at EVERY timestep.
    """
    T = P_filt.shape[0]
    min_eig_each_t = np.zeros(T)
    for k in range(T):
        diff = P_filt[k] - P_smooth[k]
        diff = 0.5 * (diff + diff.T)   # symmetrize away float asymmetry
        min_eig_each_t[k] = np.linalg.eigvalsh(diff).min()
    violations = int(np.sum(min_eig_each_t < -tol))
    return {"min_eig_each_t": min_eig_each_t, "violations": violations, "worst": float(min_eig_each_t.min())}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, df, x_filt, x_smooth, P_filt, P_smooth, check):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    dates = df.index

    # [0,0] Filtered vs smoothed price estimate
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(dates, df["close"].values, color="gray", lw=0.9, alpha=0.5, label="Actual")
    ax0.plot(dates, np.exp(x_filt[:, 0]), color="steelblue", lw=1.3, label="Filtered (forward only)")
    ax0.plot(dates, np.exp(x_smooth[:, 0]), color="crimson", lw=1.3, label="Smoothed (RTS, full data)")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Filtered vs Smoothed Price Estimate", fontsize=10)
    ax0.grid(alpha=0.3)

    # [0,1] Filtered vs smoothed trend
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(dates, x_filt[:, 1], color="steelblue", lw=1.1, label="Filtered trend")
    ax1.plot(dates, x_smooth[:, 1], color="crimson", lw=1.1, label="Smoothed trend")
    ax1.axhline(0, color="gray", lw=0.6, ls="--")
    ax1.legend(fontsize=8)
    ax1.set_title("Filtered vs Smoothed Trend (hidden state)", fontsize=10)
    ax1.grid(alpha=0.3)

    # [1,0] Variance reduction from smoothing (price state)
    ax2 = fig.add_subplot(gs[1, 0])
    var_filt_price = P_filt[:, 0, 0]
    var_smooth_price = P_smooth[:, 0, 0]
    ax2.plot(dates, var_filt_price, color="steelblue", lw=1.1, label="Var(filtered price state)")
    ax2.plot(dates, var_smooth_price, color="crimson", lw=1.1, label="Var(smoothed price state)")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.set_title("Estimation Variance: Smoothing Never Increases It", fontsize=10)
    ax2.grid(alpha=0.3)

    # [1,1] Verification summary
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["Timesteps checked", f"{len(check['min_eig_each_t'])}"],
        ["PSD violations (P_filt - P_smooth)", f"{check['violations']}"],
        ["Worst-case min eigenvalue", f"{check['worst']:.2e}  (must be >= ~0)"],
        ["Verification", "PASSED" if check["violations"] == 0 else "FAILED -- BUG"],
    ]
    table = ax3.table(cellText=rows, colLabels=["Check", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)
    ax3.set_title("Smoother Covariance-Reduction Guarantee", fontsize=10, pad=15)

    fig.suptitle(f"{ticker} — Kalman (RTS) Smoother  [in-sample retrospective, not a live signal]",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("ESTIMATION THEORY V — KALMAN FILTER: RTS SMOOTHER")
    print("(In-sample retrospective estimation, NOT a live trading signal)")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        log_prices = np.log(df["close"].values)
        returns = df["log_return"].dropna().values
        if len(log_prices) < 60:
            print(f"\n  {ticker}: skipped -- need >= 60 obs")
            continue

        A, C, _, Q, R = build_state_space()
        init_vol = float(np.std(returns[-60:])) if len(returns) > 60 else float(np.std(returns))
        x_filt, P_filt, x_pred, P_pred = kalman_filter_full(log_prices, A, C, Q, R, init_vol=init_vol)
        x_smooth, P_smooth = rts_smoother(x_filt, P_filt, x_pred, P_pred, A)
        check = verify_smoother_covariance_reduction(P_filt, P_smooth)

        avg_var_reduction = float(np.mean(1 - P_smooth[:, 0, 0] / P_filt[:, 0, 0]))

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  PSD violations: {check['violations']}/{len(check['min_eig_each_t'])} timesteps  "
              f"(worst eigenvalue={check['worst']:.2e})  "
              f"{'OK' if check['violations']==0 else 'FAILED -- BUG'}")
        print(f"  Avg. price-state variance reduction from smoothing: {avg_var_reduction:.1%}")

        summary.append({
            "Ticker": ticker, "PSD_Violations": check["violations"],
            "WorstEigenvalue": f"{check['worst']:.2e}",
            "AvgVarReduction%": f"{avg_var_reduction*100:.1f}",
        })

        plot_dashboard(ticker, df, x_filt, x_smooth, P_filt, P_smooth, check)

    if summary:
        print("\n" + "=" * 70)
        print("KALMAN SMOOTHER SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        total_violations = sum(s["PSD_Violations"] for s in summary)
        print(f"\n  Total PSD violations across all tickers/timesteps: {total_violations} "
              f"(must be 0 for the smoother to be correctly implemented)")

    print("\nKalman smoother analysis complete.")


if __name__ == "__main__":
    main()
