"""
================================================================
Script 20 — Adaptive State Estimation & Forecasting
================================================================
Replaces Script 4's basic Kalman with control-theory methods:

  Ch 1-2: State-space formulation of price dynamics
  Ch 6:   Output feedback observer (Luenberger + Kalman)
  Ch 6:   Loop Transfer Recovery (LTR) for robust estimation
  Ch 7-8: Lyapunov stability analysis of the estimator
  Ch 9-12: MRAC — Model Reference Adaptive Control for online
           parameter adjustment when market dynamics change

State vector: x = [log_price, trend, log_volatility]
Observation:  y = log_price (we only see price)
The observer reconstructs trend from price alone.
MRAC adjusts the model parameters when forecast error grows.

Note on the log_volatility state: with C=[1,0,0] and a block-diagonal
A/Q that never couples it to log_price or trend, that state is
unobservable and unforced — the Kalman gain's third row is always
exactly zero, so it just decays deterministically toward 0 regardless
of the asset. current_vol / fc_vol are therefore computed from an
EWMA of realized returns instead of read off that state.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import linalg
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

PLOT_STYLE = "seaborn-v0_8-darkgrid"


# ============================================================
# State-Space Model (Ch 1-2)
# ============================================================
# x(k+1) = A*x(k) + B*u(k) + G*w(k)
# y(k)   = C*x(k) + v(k)
#
# State: x = [log_price, trend, log_vol]
# A captures: price follows trend, trend is persistent, vol is persistent
# We don't have control input u for estimation, so B=0 here.

def build_state_space(alpha_trend=0.98, alpha_vol=0.95):
    A = np.array([
        [1.0, 1.0, 0.0],     # log_price += trend
        [0.0, alpha_trend, 0.0],  # trend is AR(1)
        [0.0, 0.0, alpha_vol],    # log_vol is AR(1)
    ])
    C = np.array([[1.0, 0.0, 0.0]])  # observe log_price only
    G = np.eye(3) * 0.01  # process noise input
    Q = np.diag([1e-5, 5e-8, 5e-3])  # vol state needs room to adapt
    R = np.array([[2e-3]])
    return A, C, G, Q, R


def fit_state_space(returns, alpha_trend=0.98, alpha_vol=0.95):
    """
    Same AR(1) structure as build_state_space, but R (observation noise)
    is THIS series' own realized daily return variance instead of the
    fixed equity-tuned constant (R=2e-3, ~85% annualized vol). Confirmed
    empirically that a single shared R is wrong for a crypto universe:
    measured R_ratio (actual/assumed variance) ranged 0.11x-3.14x across
    an 11-ticker set spanning 29%-151% annualized vol -- the filter was
    either far too trusting or far too skeptical of new price data
    depending which ticker it ran on.

    Q's proportions relative to R are preserved from the original tuning
    (Q=[1e-5,5e-8,5e-3] against R=2e-3 -> ratios 0.005 / 0.000025 / 2.5x),
    just rescaled to this ticker's actual noise magnitude rather than
    re-guessed -- this keeps the same relative "trust new data vs prior
    estimate" balance that was already reasonably tuned, without carrying
    over the wrong absolute scale.
    """
    daily_var = max(float(np.var(returns)), 1e-10)
    A = np.array([
        [1.0, 1.0, 0.0],
        [0.0, alpha_trend, 0.0],
        [0.0, 0.0, alpha_vol],
    ])
    C = np.array([[1.0, 0.0, 0.0]])
    G = np.eye(3) * 0.01
    R = np.array([[daily_var]])
    Q = np.diag([daily_var * 0.005, daily_var * 0.000025, daily_var * 2.5])
    return A, C, G, Q, R


# ============================================================
# Luenberger Observer (Ch 6)
# ============================================================
def luenberger_observer(y, A, C, L, x0=None):
    """
    x_hat(k+1) = A*x_hat(k) + L*(y(k) - C*x_hat(k))
    L = observer gain (designed via pole placement or Kalman)
    """
    n_states = A.shape[0]
    T = len(y)
    x_hat = np.zeros((T, n_states))
    x_hat[0] = x0 if x0 is not None else np.array([y[0], 0.0, -4.0])
    innovations = np.zeros(T)

    for k in range(T - 1):
        innov = y[k] - C @ x_hat[k]
        innovations[k] = float(innov)
        x_hat[k + 1] = A @ x_hat[k] + (L @ innov.reshape(-1)).flatten()

    innovations[-1] = float(y[-1] - C @ x_hat[-1])
    return x_hat, innovations


# ============================================================
# Kalman Filter as Optimal Observer (Ch 6)
# ============================================================
def kalman_observer(y, A, C, Q, R, init_vol=None):
    n = A.shape[0]
    T = len(y)
    x_hat = np.zeros((T, n))
    P = np.eye(n) * 0.1
    log_vol_init = np.log(max(init_vol, 1e-4)) if init_vol else np.log(0.02)
    x_hat[0] = [y[0], 0.0, log_vol_init]
    innovations = np.zeros(T)
    kalman_gains = np.zeros((T, n))

    for k in range(T - 1):
        # Predict
        x_pred = A @ x_hat[k]
        P_pred = A @ P @ A.T + Q

        # Update
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        innov = y[k + 1] - C @ x_pred
        innovations[k + 1] = float(innov)

        x_hat[k + 1] = x_pred + (K @ innov.reshape(-1)).flatten()
        P = (np.eye(n) - K @ C) @ P_pred
        kalman_gains[k + 1] = K.flatten()

    return x_hat, innovations, kalman_gains, P


# ============================================================
# LTR — Loop Transfer Recovery (Ch 6)
# ============================================================
def ltr_observer_gain(A, C, Q, R, q_ltr=1.0):
    """
    LTR: increase process noise artificially by q_ltr to recover
    full-state-feedback robustness margins at the plant output.
    As q_ltr → ∞, observer loop transfer → full-state loop transfer.
    """
    Q_ltr = Q + q_ltr * (C.T @ C)
    P = linalg.solve_discrete_are(A.T, C.T, Q_ltr, R)
    L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    return L, P


# ============================================================
# Lyapunov Stability Analysis (Ch 7-8)
# ============================================================
def lyapunov_stability(A, L, C):
    """
    Check if the observer error dynamics (A - L*C) are stable.
    Solve discrete Lyapunov equation: A_cl' P A_cl - P + Q = 0
    If P is positive definite, the observer is stable.
    Returns: eigenvalues of A_cl, stability boolean, P matrix.
    """
    A_cl = A - L @ C
    eigs = np.linalg.eigvals(A_cl)
    stable = all(abs(e) < 1.0 for e in eigs)

    P = None
    try:
        P = linalg.solve_discrete_lyapunov(A_cl.T, np.eye(A.shape[0]))
        pd = np.all(np.linalg.eigvalsh(P) > 0)
    except Exception:
        pd = False

    return {
        "eigenvalues": eigs,
        "spectral_radius": float(max(abs(e) for e in eigs)),
        "stable": stable,
        "lyapunov_pd": pd,
        "P": P,
    }


# ============================================================
# Realized Volatility (EWMA) — replaces the unobservable log_vol state
# ============================================================
def ewma_vol(returns, lam=0.94):
    """RiskMetrics-style EWMA volatility, aligned 1:1 with *returns*."""
    sq = pd.Series(returns) ** 2
    var = sq.ewm(alpha=1 - lam, adjust=False).mean().values
    return np.sqrt(var)


# ============================================================
# MRAC — Model Reference Adaptive Control (Ch 9-12)
# ============================================================
class MRACForecaster:
    """
    Scalar MRAC for adaptive forecasting.

    Reference model: y_m(k+1) = a_m * y_m(k)  (desired tracking behavior)
    Plant:           y(k+1)   = theta(k) * y(k) + noise
    Adaptive law:    theta(k+1) = theta(k) + gamma * e(k) * y(k)

    where e(k) = y(k) - y_m(k) is the tracking error.
    gamma is the adaptation gain (learning rate).
    """
    def __init__(self, a_m=0.0, gamma=0.005, theta0=0.0):
        self.a_m = a_m          # reference model parameter
        self.gamma = gamma      # adaptation gain
        self.theta = theta0     # adaptive parameter estimate
        self.y_m = 0.0          # reference model state

    def update(self, y_actual, y_prev):
        # Reference model output
        self.y_m = self.a_m * y_prev

        # Tracking error
        e = y_actual - self.y_m

        # MIT rule adaptation (gradient-based)
        self.theta += self.gamma * e * y_prev

        # Clip for stability
        self.theta = np.clip(self.theta, -0.5, 0.5)
        return self.theta, e, self.y_m

    def forecast(self, y_current):
        return self.theta * y_current


def run_mrac(returns, gamma=0.005):
    n = len(returns)
    mrac = MRACForecaster(a_m=0.0, gamma=gamma)
    thetas = np.zeros(n)
    errors = np.zeros(n)
    forecasts = np.zeros(n)

    for k in range(1, n):
        theta, e, _ = mrac.update(returns[k], returns[k - 1])
        thetas[k] = theta
        errors[k] = e
        if k < n - 1:
            forecasts[k + 1] = mrac.forecast(returns[k])

    return thetas, errors, forecasts


# ============================================================
# Combined Adaptive Forecast
# ============================================================
def adaptive_forecast(df, horizon=5):
    """
    Combines Kalman observer + MRAC for adaptive forecasting.
    Returns forecast, confidence interval, and diagnostics.
    """
    log_prices = np.log(df["close"].values)
    returns = df["log_return"].dropna().values
    T = len(log_prices)

    # Build state-space, calibrated to this ticker's own realized noise
    # level (see fit_state_space docstring) rather than a fixed constant
    A, C, _, Q, R = fit_state_space(returns)

    # Kalman observer — initialize vol from actual data
    actual_daily_vol = float(np.std(returns[-60:])) if len(returns) > 60 else float(np.std(returns))
    x_hat, innov, k_gains, P_final = kalman_observer(
        log_prices, A, C, Q, R, init_vol=actual_daily_vol)

    # LTR gain for robustness check
    L_ltr, P_ltr = ltr_observer_gain(A, C, Q, R, q_ltr=10.0)

    # Lyapunov stability
    L_kalman = k_gains[-1].reshape(-1, 1)
    stability = lyapunov_stability(A, L_kalman, C)

    # MRAC adaptive parameter
    thetas, mrac_errors, mrac_fc = run_mrac(returns)

    # Realized vol (EWMA) — the log_vol Kalman state is unobservable
    # (see module docstring), so vol is estimated from returns directly.
    vol_series = ewma_vol(returns)
    current_vol = float(vol_series[-1])
    # Pad to align with log_prices/dates in case returns dropped leading NaNs.
    pad = T - len(vol_series)
    vol_series_full = np.concatenate([np.full(pad, vol_series[0]), vol_series]) if pad > 0 else vol_series

    # Forecast: propagate state forward
    x_last = x_hat[-1]
    fc_states = np.zeros((horizon, 3))
    for h in range(horizon):
        x_last = A @ x_last
        # Add MRAC correction to trend
        mrac_adj = thetas[-1] * returns[-1] if len(returns) > 0 else 0
        x_last[1] += mrac_adj * 0.3  # blend MRAC into trend
        fc_states[h] = x_last

    fc_prices = np.exp(fc_states[:, 0])
    last_price = float(df["close"].iloc[-1])

    # Confidence from innovation variance + Lyapunov bound
    innov_std = np.std(innov[~np.isnan(innov)])
    fc_std = innov_std * np.sqrt(np.arange(1, horizon + 1))
    upper = last_price * np.exp(fc_states[:, 0] - log_prices[-1] + 1.96 * fc_std)
    lower = last_price * np.exp(fc_states[:, 0] - log_prices[-1] - 1.96 * fc_std)

    return {
        "x_hat": x_hat,
        "innovations": innov,
        "kalman_gains": k_gains,
        "stability": stability,
        "thetas": thetas,
        "mrac_errors": mrac_errors,
        "mrac_fc": mrac_fc,
        "fc_prices": fc_prices,
        "fc_upper": upper,
        "fc_lower": lower,
        "fc_trend": fc_states[:, 1],
        "fc_vol": np.full(horizon, current_vol),
        "vol_series": vol_series_full,
        "current_trend": float(x_hat[-1, 1]),
        "current_vol": current_vol,
        "P_final": P_final,
    }


# ============================================================
# Plotting
# ============================================================
def plot_adaptive(ticker, df, result):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(17, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.40, wspace=0.30)

    dates = df.index
    close = df["close"].values
    x_hat = result["x_hat"]
    stab = result["stability"]

    # [0,0:1] Price + Kalman estimate + forecast
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(dates, close, color="gray", lw=1, alpha=0.6, label="Actual")
    ax0.plot(dates, np.exp(x_hat[:, 0]), color="royalblue", lw=1.5, label="Kalman Estimate")
    fc_dates = pd.date_range(dates[-1], periods=len(result["fc_prices"]) + 1, freq="B")[1:]
    ax0.plot(fc_dates, result["fc_prices"], "r--", lw=2, label="Adaptive Forecast")
    ax0.fill_between(fc_dates, result["fc_lower"], result["fc_upper"],
                     alpha=0.15, color="red", label="95% CI")
    ax0.legend(fontsize=7)
    stab_str = "STABLE" if stab["stable"] else "UNSTABLE"
    ax0.set_title(f"{ticker} — Adaptive Forecast  |  Observer: {stab_str}  "
                  f"|  rho={stab['spectral_radius']:.4f}", fontsize=10)
    ax0.grid(alpha=0.3)

    # [0,2] Observer eigenvalues (stability)
    ax1 = fig.add_subplot(gs[0, 2])
    eigs = stab["eigenvalues"]
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta_circle), np.sin(theta_circle), "k--", lw=0.8, alpha=0.5)
    ax1.scatter(eigs.real, eigs.imag, s=100, c="tomato", zorder=5, edgecolors="white")
    for i, e in enumerate(eigs):
        ax1.annotate(f" λ{i+1}={abs(e):.3f}", (e.real, e.imag), fontsize=7)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect("equal")
    ax1.axhline(0, color="gray", lw=0.3)
    ax1.axvline(0, color="gray", lw=0.3)
    ax1.set_title(f"Observer Eigenvalues (|λ|<1 = stable)", fontsize=9)
    ax1.grid(alpha=0.3)

    # [1,0] Extracted trend
    ax2 = fig.add_subplot(gs[1, 0])
    trend = x_hat[:, 1]
    ax2.plot(dates, trend, color="darkorange", lw=1)
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.fill_between(dates, trend, 0, alpha=0.15,
                     where=trend > 0, color="green")
    ax2.fill_between(dates, trend, 0, alpha=0.15,
                     where=trend < 0, color="red")
    ax2.set_title("Extracted Trend (hidden state)", fontsize=9)
    ax2.grid(alpha=0.3)

    # [1,1] MRAC adaptive parameter
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(dates[-len(result["thetas"]):], result["thetas"],
             color="purple", lw=1)
    ax3.axhline(0, color="gray", lw=0.5, ls="--")
    ax3.set_title(f"MRAC Adaptive θ  |  current={result['thetas'][-1]:.4f}", fontsize=9)
    ax3.set_ylabel("θ(t)")
    ax3.grid(alpha=0.3)

    # [1,2] MRAC tracking error
    ax4 = fig.add_subplot(gs[1, 2])
    rolling_mse = pd.Series(result["mrac_errors"] ** 2).rolling(20).mean().values
    ax4.plot(dates[-len(rolling_mse):], rolling_mse, color="teal", lw=1)
    ax4.set_title("MRAC Tracking Error (rolling MSE)", fontsize=9)
    ax4.set_ylabel("MSE")
    ax4.grid(alpha=0.3)

    # [2,0] Realized volatility (EWMA)
    ax5 = fig.add_subplot(gs[2, 0])
    vol = result["vol_series"] * np.sqrt(365) * 100
    ax5.plot(dates, vol, color="tomato", lw=1)
    ax5.set_title("Realized Annualized Vol (EWMA)", fontsize=9)
    ax5.set_ylabel("Vol %")
    ax5.grid(alpha=0.3)

    # [2,1] Innovation sequence (should be white noise if observer is good)
    ax6 = fig.add_subplot(gs[2, 1])
    innov = result["innovations"]
    ax6.plot(dates, innov, color="steelblue", lw=0.5, alpha=0.7)
    ax6.axhline(0, color="gray", lw=0.5)
    ax6.set_title(f"Innovations (should be white noise)  |  std={np.std(innov):.5f}", fontsize=9)
    ax6.grid(alpha=0.3)

    # [2,2] Kalman gain evolution
    ax7 = fig.add_subplot(gs[2, 2])
    kg = result["kalman_gains"]
    ax7.plot(dates, kg[:, 0], lw=1, label="K_price")
    ax7.plot(dates, kg[:, 1], lw=1, label="K_trend")
    ax7.plot(dates, kg[:, 2], lw=1, label="K_vol")
    ax7.legend(fontsize=7)
    ax7.set_title("Kalman Gain Evolution", fontsize=9)
    ax7.grid(alpha=0.3)

    fig.suptitle(f"{ticker} — Adaptive State Estimation & Forecasting", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("ADAPTIVE STATE ESTIMATION & FORECASTING")
    print("Kalman Observer + LTR + Lyapunov + MRAC")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*50}")

        try:
            result = adaptive_forecast(df, horizon=5)
            stab = result["stability"]

            print(f"  Observer: {'STABLE' if stab['stable'] else '!! UNSTABLE'}"
                  f"  spectral_radius={stab['spectral_radius']:.4f}")
            print(f"  Lyapunov P positive definite: {stab['lyapunov_pd']}")
            print(f"  Eigenvalues: {['%.4f' % abs(e) for e in stab['eigenvalues']]}")
            print(f"  Current trend: {result['current_trend']:+.6f}"
                  f"  ({'UP' if result['current_trend'] > 0 else 'DOWN'})")
            print(f"  Current vol:   {result['current_vol']*np.sqrt(365)*100:.1f}% ann")
            print(f"  MRAC theta:    {result['thetas'][-1]:+.4f}")

            fc = result["fc_prices"]
            last = float(df["close"].iloc[-1])
            print(f"  5-day forecast: ${fc[-1]:,.2f} ({(fc[-1]/last-1)*100:+.1f}%)")
            print(f"  95% CI: ${result['fc_lower'][-1]:,.2f} – ${result['fc_upper'][-1]:,.2f}")

            summary.append({
                "Ticker": ticker,
                "Trend": f"{result['current_trend']:+.5f}",
                "Vol%": f"{result['current_vol']*np.sqrt(365)*100:.0f}",
                "MRAC_θ": f"{result['thetas'][-1]:+.4f}",
                "5d_FC%": f"{(fc[-1]/last-1)*100:+.1f}",
                "Stable": "Y" if stab["stable"] else "N",
                "ρ": f"{stab['spectral_radius']:.3f}",
            })

            plot_adaptive(ticker, df, result)

        except Exception as e:
            print(f"  Error: {e}")

    if summary:
        print("\n" + "=" * 65)
        print("ADAPTIVE FORECAST SUMMARY")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nAdaptive forecasting complete.")


if __name__ == "__main__":
    main()
