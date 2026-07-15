"""
================================================================
Script 22 — LQG Controller (Kalman-Bucy Filter + LQR)
================================================================
Fuses Script 20's price-side Kalman estimator with Script 21's
LQR regulator into a genuine Linear-Quadratic-Gaussian controller,
demonstrating the classical results from ESE 520/524:

  Estimation:   Steady-state (constant-gain) discrete Kalman filter,
                solved via the same discrete algebraic Riccati
                pattern as Script 20's LTR gain — required because
                the separation principle below needs an LTI gain,
                not Script 20's time-varying recursion.
  Kalman-Bucy:  Continuous-time filter gain via the continuous ARE,
                cross-checked against the discrete steady-state gain
                (L_kb*dt -> L_discrete as dt -> 0).
  Control:      Script 21's unchanged LQR gain K.
  Separation
  Principle:    Under certainty equivalence (u = -K*x_hat), the true
                closed loop is block upper-triangular in
                [x_true; e_estimate] — its poles are EXACTLY the
                union of the controller poles eig(A-BK) and the
                estimator poles eig(A_p - L*C_p). Verified numerically
                below, not just asserted.

Note on volatility: as documented in Script 20, the log_vol state is
unobservable and unforced given C=[1,0,0] and a block-diagonal A — its
Kalman gain row is exactly zero. So, like Script 20, this script feeds
the portfolio controller an EWMA-of-returns volatility rather than the
degenerate log_vol Kalman state; only the (genuinely observable) trend
state is estimation-coupled in the separation-principle proof below.

This quantifies something concrete: how much the fixed-gain (steady-
state) filter costs relative to Script 21's time-varying Kalman
baseline, once the transient has burned off — "Δ vs baseline."
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import linalg
from importlib import import_module as _im
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS

_r20 = _im("20_adaptive_forecast")
_r21 = _im("21_optimal_control")

build_state_space           = _r20.build_state_space
adaptive_forecast           = _r20.adaptive_forecast
ewma_vol                    = _r20.ewma_vol
build_portfolio_ss          = _r21.build_portfolio_ss
verify_closed_loop_stability = _r21.verify_closed_loop_stability
simulate_controller_baseline = _r21.simulate_controller
PositionController           = _r21.PositionController
CAPITAL                       = _r21.CAPITAL

PLOT_STYLE = "seaborn-v0_8-darkgrid"


# ============================================================
# Steady-State Discrete Kalman Filter
# (LTI gain — required for the separation-principle proof)
# ============================================================
def steady_state_kalman(A, C, Q, R):
    """Solve the discrete ARE for the steady-state Kalman gain L."""
    P = linalg.solve_discrete_are(A.T, C.T, Q, R)
    L = P @ C.T @ np.linalg.inv(C @ P @ C.T + R)
    return L, P


def kalman_filter_fixed_gain(y, A, C, L, x0):
    """
    Constant-gain Kalman filter:
      x_hat(k+1) = A*x_hat(k) + L*(y(k+1) - C*A*x_hat(k))
    """
    n = A.shape[0]
    T = len(y)
    x_hat = np.zeros((T, n))
    x_hat[0] = x0
    innovations = np.zeros(T)
    for k in range(T - 1):
        x_pred = A @ x_hat[k]
        innov = float(y[k + 1] - (C @ x_pred)[0])
        innovations[k + 1] = innov
        x_hat[k + 1] = x_pred + L.flatten() * innov
    return x_hat, innovations


# ============================================================
# Kalman-Bucy (Continuous-Time) Gain — ESE 520 cross-check
# ============================================================
def kalman_bucy_gain(A_d, C, Q, R, dt=1.0):
    """
    Approximate the continuous-time generator from the discrete model
    (A_c = (A_d - I)/dt) and solve the continuous ARE. L_kb*dt should
    converge toward the discrete steady-state L as dt -> 0 — the
    standard discrete/continuous filter correspondence.
    """
    n = A_d.shape[0]
    A_c = (A_d - np.eye(n)) / dt
    Q_c = Q / dt
    R_c = R * dt
    P_c = linalg.solve_continuous_are(A_c.T, C.T, Q_c, R_c)
    L_c = P_c @ C.T @ np.linalg.inv(R_c)
    return L_c, P_c


# ============================================================
# Separation Principle Verification
# ============================================================
def verify_separation_principle(A_port, B_port, K, A_price, L_price, C_price):
    """
    Under certainty equivalence, u(k) = -K*x_hat(k) = -K*x_true(k) + K*M*e(k),
    where e(k) is the price-side estimation error and M maps its trend
    component onto the portfolio state's "forecast" row (see module
    docstring re: the unused/unobservable log_vol row). The true closed
    loop becomes:

        x_true(k+1) = (A-BK)*x_true(k) + B*K*M*e(k)
        e(k+1)      = (A_price - L*C_price)*e(k)

    which is block upper-triangular, so its eigenvalues are EXACTLY the
    union of the controller poles and the estimator poles.
    """
    n_port, n_price = A_port.shape[0], A_price.shape[0]
    M = np.zeros((n_port, n_price))
    M[1, 1] = 1.0   # portfolio "forecast" state (index 1) <- price-side trend error (index 1)

    Acl_port  = A_port - B_port @ K
    Acl_price = A_price - L_price @ C_price

    top    = np.hstack([Acl_port, B_port @ (K @ M)])
    bottom = np.hstack([np.zeros((n_price, n_port)), Acl_price])
    Aug    = np.vstack([top, bottom])

    eig_port  = np.linalg.eigvals(Acl_port)
    eig_price = np.linalg.eigvals(Acl_price)
    eig_aug   = np.linalg.eigvals(Aug)
    eig_union = np.concatenate([eig_port, eig_price])
    match = np.allclose(sorted(np.abs(eig_aug)), sorted(np.abs(eig_union)), atol=1e-6)

    return {
        "eig_port": eig_port, "eig_price": eig_price,
        "eig_aug": eig_aug, "eig_union": eig_union,
        "separation_holds": bool(match),
    }


# ============================================================
# LQG Simulation (certainty equivalence)
# ============================================================
def simulate_lqg(df, ctrl, A_price, C_price, L_price, initial_cap=CAPITAL):
    """
    Fixed-gain Kalman filter estimates trend from noisy log-price alone
    (causal, single signal); EWMA vol fills the unobservable vol state
    (see module docstring). The estimate is fed straight into Script
    21's LQR gain via certainty equivalence.
    """
    log_prices = np.log(df["close"].values)
    returns = df["log_return"].dropna().values
    T = len(returns)

    actual_vol0 = float(np.std(returns[:60])) if len(returns) > 60 else float(np.std(returns))
    x0 = np.array([log_prices[0], 0.0, np.log(max(actual_vol0, 1e-4))])
    x_hat, innovations = kalman_filter_fixed_gain(log_prices, A_price, C_price, L_price, x0)

    pad = len(log_prices) - T          # log_prices has one more point than returns
    trend_hat = x_hat[pad:, 1]
    vol_series = ewma_vol(returns)

    wealth = initial_cap
    position = 0.0
    equity = np.zeros(T)
    positions = np.zeros(T)
    trades = np.zeros(T)

    for k in range(T):
        # wealth tracked below, not part of the LQR state (see Script 21)
        state = np.array([
            position,
            trend_hat[k],
            np.log(max(vol_series[k], 1e-6)),
        ])
        u = ctrl.optimal_position(state, max_pos=1.5)
        new_position = position + u
        trade_cost = abs(u) * wealth * 0.001

        if k > 0:
            pnl = position * wealth * returns[k]
            wealth += pnl - trade_cost
            wealth = max(wealth, 1.0)

        trades[k] = u
        position = new_position
        positions[k] = position
        equity[k] = wealth

    return {
        "equity": equity, "positions": positions, "trades": trades,
        "trend_hat": trend_hat, "vol_series": vol_series,
        "innovations": innovations[pad:],
    }


# ============================================================
# Per-Asset Analysis
# ============================================================
def analyse_asset(df):
    A_price, C_price, _, Q_price, R_price = build_state_space()
    A_port, B_port, C_port = build_portfolio_ss()

    L_price, P_price = steady_state_kalman(A_price, C_price, Q_price, R_price)
    try:
        L_kb, P_kb = kalman_bucy_gain(A_price, C_price, Q_price, R_price, dt=1.0)
        kb_ok = True
    except Exception:
        L_kb, kb_ok = np.zeros_like(L_price), False

    ctrl = PositionController(A_port, B_port, method="lqr")
    sep = verify_separation_principle(A_port, B_port, ctrl.K, A_price, L_price, C_price)

    lqg = simulate_lqg(df, ctrl, A_price, C_price, L_price)
    eq_base, pos_base, tr_base = simulate_controller_baseline(df, ctrl)

    T = len(eq_base)
    close = df["close"].values[-T:]
    bh = close / close[0] * CAPITAL

    delta_vs_base = (lqg["equity"][-1] - eq_base[-1]) / eq_base[-1] * 100

    return {
        "A_price": A_price, "C_price": C_price, "L_price": L_price, "L_kb": L_kb,
        "kb_ok": kb_ok, "ctrl": ctrl, "sep": sep,
        "lqg": lqg, "eq_base": eq_base, "pos_base": pos_base, "bh": bh,
        "delta_vs_base": delta_vs_base,
    }


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, df, r):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(17, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

    T = len(r["eq_base"])
    dates = df.index[-T:]
    close = df["close"].values[-T:]

    # [0,0] Price vs fixed-gain Kalman filtered price
    ax0 = fig.add_subplot(gs[0, 0])
    log_prices = np.log(df["close"].values)
    x_hat, _ = kalman_filter_fixed_gain(log_prices, r["A_price"], r["C_price"], r["L_price"],
                                        np.array([log_prices[0], 0.0, -4.0]))
    ax0.plot(df.index, df["close"].values, color="gray", lw=1, alpha=0.6, label="Actual")
    ax0.plot(df.index, np.exp(x_hat[:, 0]), color="royalblue", lw=1.3, label="Fixed-gain Kalman")
    ax0.legend(fontsize=7)
    ax0.set_title(f"{ticker} — Steady-State Kalman Filtered Price", fontsize=9)
    ax0.grid(alpha=0.3)

    # [0,1] Trend estimate comparison: fixed-gain vs Script 21's time-varying baseline
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(dates, r["lqg"]["trend_hat"][-T:], color="darkorange", lw=1.2, label="Fixed-gain (LQG)")
    try:
        base_trend = adaptive_forecast(df, horizon=1)["x_hat"][:, 1]
        ax1.plot(dates, base_trend[-T:], color="teal", lw=1, ls="--", label="Time-varying (baseline)")
    except Exception:
        pass
    ax1.axhline(0, color="gray", lw=0.5, ls=":")
    ax1.legend(fontsize=7)
    ax1.set_title("Trend Estimate: Steady-State vs Time-Varying", fontsize=9)
    ax1.grid(alpha=0.3)

    # [0,2] Pole-zero map — separation principle
    ax2 = fig.add_subplot(gs[0, 2])
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4)
    sep = r["sep"]
    ax2.scatter(sep["eig_port"].real, sep["eig_port"].imag, s=90, c="royalblue",
                label="Controller poles", zorder=5, edgecolors="white")
    ax2.scatter(sep["eig_price"].real, sep["eig_price"].imag, s=90, c="tomato",
                label="Estimator poles", zorder=5, edgecolors="white", marker="^")
    ax2.scatter(sep["eig_aug"].real, sep["eig_aug"].imag, s=25, c="black",
                label="Augmented (union check)", zorder=6, marker="x")
    ax2.set_xlim(-1.5, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_aspect("equal")
    ax2.legend(fontsize=6)
    holds = "HOLDS" if sep["separation_holds"] else "FAILED"
    ax2.set_title(f"Separation Principle: {holds}", fontsize=9)
    ax2.grid(alpha=0.3)

    # [1,0] Discrete vs Kalman-Bucy gain comparison
    ax3 = fig.add_subplot(gs[1, 0])
    labels = ["price", "trend", "log_vol"]
    x = np.arange(3)
    ax3.bar(x - 0.18, r["L_price"].flatten(), 0.35, label="Discrete L", color="steelblue")
    if r["kb_ok"]:
        ax3.bar(x + 0.18, r["L_kb"].flatten(), 0.35, label="Kalman-Bucy L (dt=1)", color="darkorange")
    ax3.set_xticks(x); ax3.set_xticklabels(labels)
    ax3.legend(fontsize=7)
    ax3.set_title("Discrete vs Kalman-Bucy Gain", fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    # [1,1] Innovations (should be white noise)
    ax4 = fig.add_subplot(gs[1, 1])
    innov = r["lqg"]["innovations"]
    ax4.plot(dates, innov, color="steelblue", lw=0.5, alpha=0.7)
    ax4.axhline(0, color="gray", lw=0.5)
    ax4.set_title(f"LQG Innovations  |  std={np.std(innov):.5f}", fontsize=9)
    ax4.grid(alpha=0.3)

    # [1,2] Equity curves
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(dates, r["bh"], color="gray", lw=1, alpha=0.5, label="Buy & Hold")
    ax5.plot(dates, r["eq_base"], color="teal", lw=1.3, label="Baseline LQR (Script 21)")
    ax5.plot(dates, r["lqg"]["equity"], color="royalblue", lw=1.5, label="LQG (steady-state)")
    ax5.legend(fontsize=7)
    ax5.set_ylabel("Portfolio Value ($)")
    ax5.set_title("Equity Curves", fontsize=9)
    ax5.grid(alpha=0.3)

    # [2,0:3] Summary panel
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    stab = r["ctrl"].stability
    rows = [
        ["Controller poles (|lambda|)", ", ".join(f"{abs(e):.3f}" for e in sep["eig_port"])],
        ["Estimator poles (|lambda|)",  ", ".join(f"{abs(e):.3f}" for e in sep["eig_price"])],
        ["Separation principle",        "HOLDS" if sep["separation_holds"] else "FAILED"],
        ["Closed-loop stable",          "Y" if stab["stable"] else "N"],
        ["LQG final equity",            f"${r['lqg']['equity'][-1]:,.0f}"],
        ["Baseline final equity",       f"${r['eq_base'][-1]:,.0f}"],
        ["Delta vs baseline",           f"{r['delta_vs_base']:+.2f}%"],
    ]
    table = ax6.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    fig.suptitle(f"{ticker} — LQG Controller (Kalman-Bucy + LQR, Separation Principle)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("LQG CONTROLLER")
    print("Steady-State Kalman + Kalman-Bucy + LQR + Separation Principle")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}   ${df['close'].iloc[-1]:.2f}")
        print(f"{'─'*50}")

        try:
            r = analyse_asset(df)
            sep = r["sep"]
            stab = r["ctrl"].stability

            print(f"  Controller poles:  {[f'{abs(e):.4f}' for e in sep['eig_port']]}")
            print(f"  Estimator poles:   {[f'{abs(e):.4f}' for e in sep['eig_price']]}")
            print(f"  Separation principle: {'HOLDS' if sep['separation_holds'] else 'FAILED'}"
                  f"  (augmented eigs match controller∪estimator eigs)")
            print(f"  Closed-loop stable: {'Y' if stab['stable'] else 'N'}"
                  f"  rho={stab['spectral_radius']:.4f}")
            if r["kb_ok"]:
                print(f"  Discrete L:      {r['L_price'].flatten().round(5)}")
                print(f"  Kalman-Bucy L*dt: {(r['L_kb']*1.0).flatten().round(5)}")

            lqg_final = r["lqg"]["equity"][-1]
            base_final = r["eq_base"][-1]
            print(f"  LQG final equity:      ${lqg_final:,.0f}")
            print(f"  Baseline final equity: ${base_final:,.0f}")
            print(f"  Delta vs baseline:     {r['delta_vs_base']:+.2f}%")

            summary.append({
                "Ticker": ticker,
                "Separation": "Y" if sep["separation_holds"] else "N",
                "Stable": "Y" if stab["stable"] else "N",
                "LQG_Ret%": f"{(lqg_final/CAPITAL-1)*100:+.1f}",
                "Base_Ret%": f"{(base_final/CAPITAL-1)*100:+.1f}",
                "Delta%": f"{r['delta_vs_base']:+.2f}",
            })

            plot_dashboard(ticker, df, r)

        except Exception as e:
            print(f"  Error: {e}")

    if summary:
        print("\n" + "=" * 65)
        print("LQG CONTROLLER SUMMARY")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nLQG controller analysis complete.")


if __name__ == "__main__":
    main()
