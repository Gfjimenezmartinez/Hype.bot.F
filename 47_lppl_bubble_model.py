"""
================================================================
Script 47 — Log-Periodic Power Law (LPPL) Bubble/Crash Model
================================================================
Script 46's critical-slowing-down approach (rising autocorrelation +
variance before a regime flip, borrowed from ecology/climate science)
tested null on this universe, at both daily and hourly resolution: 1/9
tickers significant, indistinguishable from chance. This script tries
the purpose-built econophysics alternative (Sornette, "Why Stock
Markets Crash", 2003), which directly targets the phenomenon described
-- price growing FASTER than exponential (a genuine finite-time
singularity, not just "a lot of growth") with accelerating log-periodic
oscillations, culminating in a critical time Tc. Tc is a literal
bifurcation point in the mathematical sense: the fitted equation
actually diverges there, not a metaphor.

    ln(p(t)) = A + B*(Tc-t)^m + C*(Tc-t)^m * cos(omega*ln(Tc-t) - phi)

  - m in (0,1): super-exponential growth as t -> Tc (ordinary
    exponential growth is the m=1 boundary case, which grows fast but
    never diverges in finite time; m<1 does).
  - omega: log-periodic oscillation frequency (4-25 in the empirical
    literature) -- accelerating "waves" superimposed on the power-law
    trend as Tc approaches.
  - B<0 required for a genuine bubble (price accelerates upward toward
    Tc); B>0 fits the mirror-image "anti-bubble" (a decelerating crash
    into a bottom) -- this script fits bubbles (B<0) only, the crash-
    after-blow-off-top case the CSD test also targeted.

KNOWN, STATED LIMITATION: LPPL fitting is a hard, multi-modal nonlinear
optimization (many (Tc,m,omega) combinations fit comparably well), and
its real-world crash-prediction record in the literature is genuinely
mixed. This script does NOT assume it works -- same honesty standard as
Script 46:
  1. Reports a GOODNESS-OF-FIT comparison every fit must clear before
     being taken seriously: full LPPL R^2 vs. the SAME (Tc,m) power-law
     backbone WITHOUT the oscillation term. A big R^2 jump from adding
     the log-periodic wiggle is the theory's actual falsifiable claim --
     a monotonic accelerating trend alone fits the power-law-only
     version fine and isn't evidence for LPPL specifically.
  2. Walks backward through history at multiple checkpoints, fits on
     each trailing window, and checks whether the predicted Tc actually
     landed near a real subsequent drawdown -- compared against a
     bootstrap baseline of random guessed Tc dates over the same
     horizon. Reports the empirical result either way.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings("ignore")

from data_loader import fetch_intraday, SYMBOLS, format_price

PLOT_STYLE = "seaborn-v0_8-darkgrid"
BAR_INTERVAL = "1h"
BAR_FETCH_LIMIT = 5000        # ~208 days of hourly bars -- practical ccxt fetch cap

FIT_WINDOW_HOURS      = 90 * 24    # trailing window each LPPL fit uses (90 real days)
TC_MAX_HORIZON_HOURS  = 30 * 24    # Tc must fall within this far into the future to count
CHECKPOINT_STRIDE_HOURS = 10 * 24  # walk-back validation checkpoint spacing (10 real days)
MIN_R2_CONFIDENT = 0.85            # LPPL fit quality bar before a prediction counts

M_BOUNDS     = (0.1, 0.9)
OMEGA_BOUNDS = (4.0, 25.0)
DE_POPSIZE   = 10
DE_MAXITER   = 40
DE_SEED      = 42
N_BOOTSTRAP  = 2000


# ============================================================
# Core LPPL fit -- nonlinear params (Tc, m, omega) via global
# optimization, linear params (A, B, C1, C2) via closed-form OLS at
# each candidate (the equation is linear in A/B/C1/C2 for FIXED
# Tc/m/omega, so there's no need to search all 7 parameters jointly).
# ============================================================
def _design_matrix(t, Tc, m, omega):
    dt = np.clip(Tc - t, 1e-6, None)   # guard t >= Tc (undefined/complex otherwise)
    power = dt ** m
    log_dt = np.log(dt)
    return np.column_stack([
        np.ones_like(t), power,
        power * np.cos(omega * log_dt),
        power * np.sin(omega * log_dt),
    ])


def _lppl_rss(nonlinear_params, t, log_p):
    Tc, m, omega = nonlinear_params
    if Tc <= t[-1] + 1e-3:
        return 1e10   # Tc must be strictly after the fitted window
    X = _design_matrix(t, Tc, m, omega)
    try:
        coef, *_ = np.linalg.lstsq(X, log_p, rcond=None)
    except Exception:
        return 1e10
    resid = log_p - X @ coef
    return float(np.sum(resid ** 2))


def fit_lppl(t, log_p, tc_bounds, m_bounds=M_BOUNDS, omega_bounds=OMEGA_BOUNDS,
             popsize=DE_POPSIZE, maxiter=DE_MAXITER, seed=DE_SEED):
    bounds = [tc_bounds, m_bounds, omega_bounds]
    result = differential_evolution(_lppl_rss, bounds, args=(t, log_p),
                                     popsize=popsize, maxiter=maxiter,
                                     seed=seed, polish=True, tol=1e-9)
    Tc, m, omega = result.x
    X = _design_matrix(t, Tc, m, omega)
    coef, *_ = np.linalg.lstsq(X, log_p, rcond=None)
    A, B, C1, C2 = coef
    fitted = X @ coef
    ss_res = float(np.sum((log_p - fitted) ** 2))
    ss_tot = float(np.sum((log_p - log_p.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Power-law-only backbone at the SAME (Tc, m) -- isolates what the
    # oscillation term specifically contributes.
    dt = np.clip(Tc - t, 1e-6, None)
    Xp = np.column_stack([np.ones_like(t), dt ** m])
    coef_p, *_ = np.linalg.lstsq(Xp, log_p, rcond=None)
    fitted_p = Xp @ coef_p
    ss_res_p = float(np.sum((log_p - fitted_p) ** 2))
    r2_power_only = 1 - ss_res_p / ss_tot if ss_tot > 0 else 0.0

    return {"Tc": float(Tc), "m": float(m), "omega": float(omega),
            "A": float(A), "B": float(B), "C": float(np.hypot(C1, C2)),
            "r2": r2, "r2_power_only": r2_power_only,
            "oscillation_gain": r2 - r2_power_only,
            "is_bubble": bool(B < 0), "fitted": fitted}


# ============================================================
# Walk-Forward Validation — does predicted Tc land near real drawdowns?
# ============================================================
def actual_crash_offset(close, start_idx, horizon):
    """Within close[start_idx : start_idx+horizon], the hour-offset of
    the trough of the largest peak-to-trough drawdown -- the 'actual'
    crash timing to compare a predicted Tc against."""
    window = close[start_idx:start_idx + horizon + 1]
    if len(window) < 10:
        return None
    peak = np.maximum.accumulate(window)
    dd = (window - peak) / peak
    return int(np.argmin(dd))


def validate_lppl(close, n_bars, fit_window=FIT_WINDOW_HOURS,
                   tc_horizon=TC_MAX_HORIZON_HOURS, stride=CHECKPOINT_STRIDE_HOURS,
                   min_r2=MIN_R2_CONFIDENT, n_bootstrap=N_BOOTSTRAP, seed=0):
    """
    Walks backward through history at `stride`-spaced checkpoints, fits
    LPPL on the trailing `fit_window` bars ending at each, and -- for
    fits that are (a) genuine bubbles (B<0), (b) confident (R^2 >=
    min_r2), and (c) predict Tc within `tc_horizon` of the checkpoint --
    records the prediction error: |predicted hours-to-Tc - actual hours
    to the checkpoint's biggest subsequent drawdown|. Compared against a
    bootstrap baseline of RANDOM guessed Tc offsets over the same
    horizon, so a wide tolerance on a short horizon isn't mistaken for
    real skill.
    """
    log_close = np.log(close)
    checkpoints = list(range(fit_window, n_bars - tc_horizon, stride))
    errors, confident_checks = [], 0

    for cp in checkpoints:
        t = np.arange(fit_window, dtype=float)
        log_p = log_close[cp - fit_window:cp]
        tc_bounds = (0.5, float(tc_horizon))   # Tc measured in hours-ahead of t[-1]=fit_window-1
        try:
            fit = fit_lppl(t, log_p, tc_bounds)
        except Exception:
            continue
        if not fit["is_bubble"] or fit["r2"] < min_r2:
            continue

        confident_checks += 1
        predicted_hours = fit["Tc"] - t[-1]
        actual_offset = actual_crash_offset(close, cp, tc_horizon)
        if actual_offset is None:
            continue
        errors.append(abs(predicted_hours - actual_offset))

    if not errors:
        return {"n_checkpoints": len(checkpoints), "n_confident": confident_checks,
                "n_scored": 0, "mean_error_hours": np.nan, "p_value": np.nan}

    rng = np.random.default_rng(seed)
    observed_mean_err = float(np.mean(errors))
    boot_means = np.array([
        np.mean(np.abs(rng.uniform(0, tc_horizon, size=len(errors))
                        - rng.uniform(0, tc_horizon, size=len(errors))))
        for _ in range(n_bootstrap)
    ])
    p_value = float(np.mean(boot_means <= observed_mean_err))  # P(random baseline this good or better)

    return {"n_checkpoints": len(checkpoints), "n_confident": confident_checks,
            "n_scored": len(errors), "mean_error_hours": observed_mean_err,
            "p_value": p_value, "boot_means": boot_means}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, dates, close, live_fit, val_result):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # [0,:] Price with LPPL fit overlay (if a confident current bubble fit exists)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(dates, close, color="steelblue", lw=1.0, label="Price")
    if live_fit is not None:
        n = len(close)
        t = np.arange(FIT_WINDOW_HOURS, dtype=float)
        fitted_prices = np.exp(live_fit["fitted"])
        ax0.plot(dates[-FIT_WINDOW_HOURS:], fitted_prices, color="crimson", lw=1.6,
                  label=f"LPPL fit (R^2={live_fit['r2']:.3f}, Tc in {live_fit['Tc']-t[-1]:.0f}h)")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Price vs. LPPL Fit (trailing {FIT_WINDOW_HOURS//24}d)", fontsize=10)
    ax0.grid(alpha=0.3)

    # [1,0] Live fit summary
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("off")
    if live_fit is not None:
        rows = [
            ["Is bubble (B<0)?", "YES" if live_fit["is_bubble"] else "no"],
            ["m (growth exponent)", f"{live_fit['m']:.3f}"],
            ["omega (log-periodic freq)", f"{live_fit['omega']:.2f}"],
            ["R^2 (full LPPL)", f"{live_fit['r2']:.4f}"],
            ["R^2 (power-law only, no oscillation)", f"{live_fit['r2_power_only']:.4f}"],
            ["Oscillation R^2 gain", f"{live_fit['oscillation_gain']:+.4f}"],
            ["Predicted Tc", f"{live_fit['Tc']-FIT_WINDOW_HOURS+1:.0f}h ahead"],
        ]
    else:
        rows = [["Current fit", "no confident bubble signature"]]
    table = ax1.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)
    ax1.set_title("Live LPPL Fit", fontsize=10, pad=12)

    # [1,1] Validation summary
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis("off")
    rows2 = [
        ["Checkpoints walked", f"{val_result['n_checkpoints']}"],
        ["Confident bubble fits (R^2>={:.2f})".format(MIN_R2_CONFIDENT), f"{val_result['n_confident']}"],
        ["Scored (had a forward window)", f"{val_result['n_scored']}"],
        ["Mean |predicted Tc - actual crash| (hrs)",
         f"{val_result['mean_error_hours']:.1f}" if val_result['mean_error_hours'] == val_result['mean_error_hours'] else "n/a"],
        ["p-value (vs. random-guess baseline)",
         f"{val_result['p_value']:.3f}" if val_result['p_value'] == val_result['p_value'] else "n/a"],
    ]
    table2 = ax2.table(cellText=rows2, colLabels=["Check", "Value"], loc="center", cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(8.5)
    table2.scale(1.0, 1.6)
    ax2.set_title("Walk-Forward Validation (p<0.05 = beats random-guess baseline)", fontsize=9.5, pad=12)

    fig.suptitle(f"{ticker} — Log-Periodic Power Law Bubble/Crash Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("LOG-PERIODIC POWER LAW (LPPL) BUBBLE/CRASH MODEL")
    print(f"Bar interval: {BAR_INTERVAL}  |  fit window: {FIT_WINDOW_HOURS//24}d  "
          f"|  Tc horizon: {TC_MAX_HORIZON_HOURS//24}d")
    print("=" * 70)

    summary = []
    for ticker, symbol in SYMBOLS.items():
        df = fetch_intraday(symbol, interval=BAR_INTERVAL, limit=BAR_FETCH_LIMIT)
        min_needed = FIT_WINDOW_HOURS + TC_MAX_HORIZON_HOURS + CHECKPOINT_STRIDE_HOURS
        if df is None or len(df) < min_needed:
            print(f"\n  {ticker}: skipped -- insufficient hourly history "
                  f"(need >= {min_needed} bars)")
            continue

        close = df["close"].values
        dates = df.index
        n = len(close)

        # Live fit: trailing FIT_WINDOW_HOURS ending at the most recent bar
        t_live = np.arange(FIT_WINDOW_HOURS, dtype=float)
        log_p_live = np.log(close[-FIT_WINDOW_HOURS:])
        tc_bounds_live = (0.5, float(TC_MAX_HORIZON_HOURS))
        try:
            live_fit = fit_lppl(t_live, log_p_live, tc_bounds_live)
            if not (live_fit["is_bubble"] and live_fit["r2"] >= MIN_R2_CONFIDENT):
                live_fit = None
        except Exception:
            live_fit = None

        val_result = validate_lppl(close, n)

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        if live_fit is not None:
            print(f"  LIVE: confident bubble signature -- m={live_fit['m']:.3f}  "
                  f"omega={live_fit['omega']:.2f}  R^2={live_fit['r2']:.4f}  "
                  f"(power-law-only R^2={live_fit['r2_power_only']:.4f}, "
                  f"oscillation gain={live_fit['oscillation_gain']:+.4f})  "
                  f"Tc in ~{live_fit['Tc']-t_live[-1]:.0f}h")
        else:
            print(f"  LIVE: no confident bubble signature right now")
        print(f"  Validation: {val_result['n_confident']}/{val_result['n_checkpoints']} checkpoints "
              f"were confident bubble fits, {val_result['n_scored']} scorable  "
              f"mean_err={val_result['mean_error_hours']:.1f}h  "
              f"p={val_result['p_value']:.3f}" if val_result['n_scored'] > 0 else
              f"  Validation: {val_result['n_confident']}/{val_result['n_checkpoints']} confident, "
              f"none scorable")

        summary.append({
            "Ticker": ticker,
            "LiveBubble": "YES" if live_fit is not None else "no",
            "N_Checkpoints": val_result["n_checkpoints"],
            "N_Confident": val_result["n_confident"],
            "N_Scored": val_result["n_scored"],
            "MeanErrHrs": f"{val_result['mean_error_hours']:.1f}" if val_result['mean_error_hours'] == val_result['mean_error_hours'] else "n/a",
            "P_value": f"{val_result['p_value']:.3f}" if val_result['p_value'] == val_result['p_value'] else "n/a",
        })

        plot_dashboard(ticker, dates, close, live_fit, val_result)

    if summary:
        print("\n" + "=" * 70)
        print("LPPL BUBBLE MODEL SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))
        n_sig = sum(1 for s in summary if s["P_value"] != "n/a" and float(s["P_value"]) < 0.05)
        n_scored_total = sum(1 for s in summary if s["N_Scored"] != "0" and s["P_value"] != "n/a")
        print(f"\n  Tickers where LPPL's Tc prediction significantly beat the random-guess "
              f"baseline: {n_sig}/{n_scored_total} scorable tickers")
        print("  (If this is near what chance alone predicts at alpha=0.05, LPPL isn't")
        print("   earning its keep here either -- reported honestly, not assumed.)")

    print("\nLPPL bubble model analysis complete.")


if __name__ == "__main__":
    main()
