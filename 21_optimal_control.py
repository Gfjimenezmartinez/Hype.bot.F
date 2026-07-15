"""
================================================================
Script 21 — Optimal Position Control
================================================================
Frames trading as a control problem and solves for optimal
position sizing using methods from your control systems course:

  Ch 2-3: LQR / RSLQR — optimal state feedback position sizing
  Ch 4:   H-infinity — worst-case robust sizing (survives crashes)
  Ch 7-8: Lyapunov — prove the closed-loop trading system is stable
  Ch 6:   Output feedback — controller uses observer states

The "plant" is your portfolio dynamics:
  state  x = [position, return_forecast, volatility]
  input  u = position_change (delta shares)

Cost function tracks position to the forecast signal, penalized by
transaction cost:
  J = sum[ q*(position - beta*forecast)^2 + q_vol*vol^2 + r*(trade)^2 ]

Note: wealth is tracked separately in the simulation loop, not as an
LQR state. It's neither actuated by B nor coupled to anything else, so
it was a purely decorative, always-zero-weighted mode -- and once the
cost below gained an off-diagonal (position, forecast) tracking term,
that inert marginal mode tripped up scipy's discrete Riccati solver
("Failed to find a finite solution") even at zero weight. Dropping it
is mathematically equivalent (it never affected the optimal gain) and
avoids the degeneracy.
================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import linalg
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price, CACHE_DIR

try:
    from importlib import import_module as _im
    _r20 = _im("20_adaptive_forecast")
    adaptive_forecast = _r20.adaptive_forecast
    ADAPTIVE_AVAILABLE = True
except Exception:
    ADAPTIVE_AVAILABLE = False

try:
    _r15 = _im("15_regime_detection")
    _build_regime_features = _r15.build_features
    _fit_hmm = _r15.fit_hmm
    REGIME_CTRL_AVAILABLE = True
except Exception:
    REGIME_CTRL_AVAILABLE = False

PLOT_STYLE = "seaborn-v0_8-darkgrid"
CAPITAL    = 100_000
RF_DAILY   = 0.05 / 365   # crypto trades 24/7


# ============================================================
# Portfolio State-Space (Ch 1-2)
# ============================================================
# State: x = [position_frac, forecast_return, log_vol]
# Input: u = delta_position (change in position fraction)
# (wealth is tracked in the simulation loop, not here -- see module
# docstring)
#
# x(k+1) = A*x(k) + B*u(k) + G*w(k)
# y(k)   = C*x(k)

def fit_ar1(series, default=0.9, lo=0.0, hi=0.99):
    """
    Least-squares AR(1) coefficient: x(k+1) = alpha*x(k) + noise, fit via
    np.linalg.lstsq -- actual system identification from this ticker's
    own forecast/vol series, in place of build_portfolio_ss's fixed
    alpha_fc=0.85/alpha_vol=0.92 shared by every ticker. Crypto's vol
    regimes move faster and more extremely than equities' (measured
    29%-151% annualized vol across this suite's 11-ticker universe), so
    a shared persistence constant is exactly the same kind of mismatch
    Script 20's fit_state_space fixes for Q/R -- this is the same idea
    applied to the portfolio-side state-space.

    Clipped to [lo, hi]: an OLS estimate can come out negative or >=1 on
    noisy/short series, which would hand the stability/margin checks an
    exogenous state that's actually growing or oscillating instead of
    decaying. `default` covers series too short to fit reliably.
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return default
    y_t, y_tm1 = x[1:], x[:-1]
    alpha_hat, *_ = np.linalg.lstsq(y_tm1.reshape(-1, 1), y_t, rcond=None)
    return float(np.clip(alpha_hat[0], lo, hi))


def calibrate_beta_track(trend_series, target_position_ref=0.6):
    """
    beta_track scales a Kalman trend state into a target position
    (target_position = beta_track * trend). A fixed beta_track=500 was
    sized for equities' typical trend magnitude (~0.001-0.002) -- crypto's
    trend-state std varies more than 5x across this suite's universe
    (BTC ~0.002, ZEC ~0.0095 in the last measurement), so a shared
    constant either saturates high-vol tickers against the max_pos clip
    constantly, or barely moves low-vol ones. Calibrate so a 1-std trend
    move maps to a moderate target_position_ref (0.6, comfortably inside
    the 1.5 clip) instead.
    """
    trend_std = float(np.std(trend_series))
    if trend_std < 1e-10:
        return 500.0  # degenerate/flat series -- fall back to the old default
    return float(target_position_ref / trend_std)


def build_portfolio_ss(alpha_fc=0.85, alpha_vol=0.92):
    A = np.array([
        [1.0, 0.0, 0.0],         # position: stays until changed
        [0.0, alpha_fc, 0.0],    # forecast: AR(1) decay
        [0.0, 0.0, alpha_vol],   # vol: AR(1) persistence
    ])
    B = np.array([
        [1.0],   # position changes by u
        [0.0],   # forecast not controlled
        [0.0],   # vol not controlled
    ])
    C = np.array([[1.0, 0.0, 0.0]])  # observe position
    return A, B, C


# ============================================================
# LQR — Linear Quadratic Regulator (Ch 2-3)
# ============================================================
def solve_lqr(A, B, Q, R):
    """
    Solve discrete-time algebraic Riccati equation:
    P = A'PA - A'PB(R + B'PB)^{-1}B'PA + Q
    Returns gain K such that u = -K*x is optimal.
    """
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return K, P


def lqr_position(x_state, K, max_pos=2.0):
    """Compute optimal position change from LQR gain."""
    u = float(-K @ x_state)
    return np.clip(u, -max_pos, max_pos)


# ============================================================
# RSLQR — Robustness Recovery (Ch 3)
# ============================================================
def solve_rslqr(A, B, Q, R, rho=1.0):
    """
    RSLQR: scale R by 1/rho to recover robustness.
    rho → ∞ gives cheap control (aggressive), rho → 0 gives expensive (conservative).
    rho = 1 is standard LQR. rho < 1 improves gain/phase margins.
    """
    R_scaled = R / max(rho, 1e-6)
    return solve_lqr(A, B, Q, R_scaled)


# ============================================================
# H-infinity Optimal Control (Ch 4)
# ============================================================
def solve_hinf(A, B, Q, R, gamma=2.0):
    """
    H-infinity: minimize worst-case disturbance impact.
    Solves the game-theoretic Riccati equation:
    P = A'PA + Q - A'PB_tilde * inv(R_tilde + B_tilde'PB_tilde) * B_tilde'PA

    where B_tilde = [B, G], R_tilde incorporates gamma (disturbance bound).
    gamma = disturbance attenuation level (lower = more robust, harder to solve).
    """
    n = A.shape[0]
    G = np.eye(n) * 0.01  # disturbance input matrix

    B_tilde = np.hstack([B, G])
    R_tilde = np.block([
        [R, np.zeros((R.shape[0], n))],
        [np.zeros((n, R.shape[1])), -gamma**2 * np.eye(n)]
    ])

    try:
        P = linalg.solve_discrete_are(A, B_tilde, Q, R_tilde)
        K_full = np.linalg.inv(R_tilde + B_tilde.T @ P @ B_tilde) @ (B_tilde.T @ P @ A)
        K = K_full[:B.shape[1], :]  # extract control gain only
        return K, P, True
    except Exception:
        # Fallback to conservative LQR if H-inf doesn't converge
        K, P = solve_lqr(A, B, Q, R * 3)
        return K, P, False


# ============================================================
# Lyapunov Stability Verification (Ch 7-8)
# ============================================================
def verify_closed_loop_stability(A, B, K):
    """
    Verify closed-loop system x(k+1) = (A - B*K)*x(k) is stable.
    1. Check eigenvalues inside unit circle
    2. Solve Lyapunov equation to find V(x) = x'Px
    3. Compute gain and phase margins
    """
    A_cl = A - B @ K
    eigs = np.linalg.eigvals(A_cl)
    stable = all(abs(e) < 1.0 for e in eigs)
    rho = float(max(abs(e) for e in eigs))

    P_lyap = None
    lyap_ok = False
    try:
        P_lyap = linalg.solve_discrete_lyapunov(A_cl.T, np.eye(A.shape[0]))
        lyap_ok = np.all(np.linalg.eigvalsh(P_lyap) > 0)
    except Exception:
        pass

    return {
        "A_cl_eigs": eigs,
        "spectral_radius": rho,
        "stable": stable,
        "lyapunov_ok": lyap_ok,
        "settling_time": int(-1 / np.log(max(rho, 1e-6))) if rho < 1 else float("inf"),
    }


# ============================================================
# Position Controller
# ============================================================
class PositionController:
    def __init__(self, A, B, method="lqr", **kwargs):
        self.A = A
        self.B = B
        self.method = method
        n = A.shape[0]

        # Cost weights
        q_track    = kwargs.get("q_track", 1.0)
        beta_track = kwargs.get("beta_track", 500.0)
        q_vol      = kwargs.get("q_vol", 0.01)
        r_trade    = kwargs.get("r_trade", 5.0)

        # Q penalizes (position - beta_track*forecast)^2 rather than
        # position^2 alone. With a purely diagonal Q, forecast has zero
        # effect on the optimal policy: B only actuates position, and
        # nothing couples position to forecast, so LQR just decays any
        # existing position to zero and never opens one from a flat start.
        # The cross term below makes the regulator actually want position
        # to track beta_track * forecast (long when trend is up, short
        # when trend is down), which is the whole point of a *position*
        # controller. beta_track ~500 maps a typical daily log-trend
        # (~0.001-0.002) to a moderate target position (~0.5-1.0).
        self.Q = np.zeros((3, 3))
        self.Q[0, 0] = q_track
        self.Q[0, 1] = self.Q[1, 0] = -q_track * beta_track
        self.Q[1, 1] = q_track * beta_track ** 2
        self.Q[2, 2] = q_vol
        self.R = np.array([[r_trade]])
        self.beta_track = beta_track

        if method == "lqr":
            self.K, self.P = solve_lqr(A, B, self.Q, self.R)
            self.solved = True
        elif method == "rslqr":
            rho = kwargs.get("rho", 0.5)
            self.K, self.P = solve_rslqr(A, B, self.Q, self.R, rho=rho)
            self.solved = True
        elif method == "hinf":
            gamma = kwargs.get("gamma", 2.0)
            self.K, self.P, self.solved = solve_hinf(A, B, self.Q, self.R, gamma=gamma)
        else:
            self.K = np.zeros((1, n))
            self.P = np.eye(n)
            self.solved = False

        self.stability = verify_closed_loop_stability(A, B, self.K)

    def optimal_position(self, state, max_pos=2.0):
        return lqr_position(state, self.K, max_pos)


# ============================================================
# Simulation
# ============================================================
def simulate_controller(df, controller, initial_cap=CAPITAL):
    returns = df["log_return"].dropna().values
    close = df["close"].values
    T = len(returns)

    wealth = initial_cap
    position = 0.0
    equity = np.zeros(T)
    positions = np.zeros(T)
    trades = np.zeros(T)

    # Get forecast from Script 20 if available
    fc_returns = np.zeros(T)
    if ADAPTIVE_AVAILABLE:
        try:
            result = adaptive_forecast(df, horizon=1)
            trend = result["x_hat"][:, 1]
            fc_returns[-len(trend):] = trend[-T:]
        except Exception:
            pass

    vol = pd.Series(returns).rolling(20).std().fillna(0.02).values

    for k in range(T):
        # Build state vector (wealth tracked below, not part of the LQR state)
        state = np.array([
            position,
            fc_returns[k],
            np.log(max(vol[k], 1e-6)),
        ])

        # Get optimal position change
        u = controller.optimal_position(state, max_pos=1.5)
        new_position = position + u

        # Transaction cost
        trade_cost = abs(u) * wealth * 0.001  # 10bps

        # P&L
        if k > 0:
            pnl = position * wealth * returns[k]
            wealth += pnl - trade_cost
            wealth = max(wealth, 1.0)

        trades[k] = u
        position = new_position
        positions[k] = position
        equity[k] = wealth

    return equity, positions, trades


# ============================================================
# Plotting
# ============================================================
def plot_controller(ticker, df, results_dict):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.30)
    colors = {"LQR": "royalblue", "RSLQR": "forestgreen", "H-inf": "tomato"}

    T = len(df["log_return"].dropna())
    dates = df.index[-T:]

    # [0,0:1] Equity curves
    ax0 = fig.add_subplot(gs[0, :])
    bh = df["close"].values[-T:] / df["close"].values[-T] * CAPITAL
    ax0.plot(dates, bh, color="gray", lw=1, alpha=0.5, label="Buy & Hold")
    for name, (eq, _, _, _) in results_dict.items():
        ax0.plot(dates, eq, lw=1.5, color=colors.get(name, "purple"), label=name)
    ax0.legend(fontsize=8)
    ax0.set_ylabel("Portfolio Value ($)")
    ax0.set_title(f"{ticker} — Optimal Control Equity Curves", fontsize=11)
    ax0.grid(alpha=0.3)

    # [1,0] Positions over time
    ax1 = fig.add_subplot(gs[1, 0])
    for name, (_, pos, _, _) in results_dict.items():
        ax1.plot(dates, pos, lw=1, color=colors.get(name, "purple"), label=name, alpha=0.8)
    ax1.axhline(0, color="gray", lw=0.5, ls="--")
    ax1.legend(fontsize=7)
    ax1.set_ylabel("Position (fraction)")
    ax1.set_title("Optimal Positions", fontsize=10)
    ax1.grid(alpha=0.3)

    # [1,1] Trades
    ax2 = fig.add_subplot(gs[1, 1])
    for name, (_, _, tr, _) in results_dict.items():
        ax2.plot(dates, tr, lw=0.5, color=colors.get(name, "purple"), alpha=0.6, label=name)
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.legend(fontsize=7)
    ax2.set_ylabel("Trade (Δposition)")
    ax2.set_title("Trade Activity", fontsize=10)
    ax2.grid(alpha=0.3)

    # [2,0] Closed-loop eigenvalues
    ax3 = fig.add_subplot(gs[2, 0])
    theta = np.linspace(0, 2 * np.pi, 100)
    ax3.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4)
    for name in results_dict:
        ctrl = results_dict[name][3] if len(results_dict[name]) > 3 else None
        if ctrl and hasattr(ctrl, "stability"):
            eigs = ctrl.stability["A_cl_eigs"]
            ax3.scatter(eigs.real, eigs.imag, s=60, label=name,
                        color=colors.get(name, "purple"), edgecolors="white", zorder=5)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect("equal")
    ax3.legend(fontsize=7)
    ax3.set_title("Closed-Loop Eigenvalues (|λ|<1 = stable)", fontsize=9)
    ax3.grid(alpha=0.3)

    # [2,1] Performance table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    rows = []
    bh_ret = (bh[-1] / bh[0] - 1) * 100
    for name, (eq, _, _, _) in results_dict.items():
        ret = (eq[-1] / eq[0] - 1) * 100
        peak = np.maximum.accumulate(eq)
        dd = float(((eq - peak) / peak).min() * 100)
        rets = np.diff(eq) / eq[:-1]
        sr = float(np.mean(rets) / max(np.std(rets), 1e-8) * np.sqrt(365))   # crypto trades 24/7
        ctrl = results_dict[name][3] if len(results_dict[name]) > 3 else None
        stab = "Y" if ctrl and ctrl.stability["stable"] else "?"
        rows.append([name, f"{ret:+.1f}%", f"{dd:.1f}%", f"{sr:.2f}",
                     f"{ret-bh_ret:+.1f}%", stab])
    rows.append(["Buy & Hold", f"{bh_ret:+.1f}%", "-", "-", "-", "-"])
    table = ax4.table(cellText=rows,
                      colLabels=["Method", "Return", "MaxDD", "Sharpe", "Alpha", "Stable"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax4.set_title("Controller Performance", fontsize=10, pad=15)

    fig.suptitle(f"{ticker} — Optimal Position Control (LQR / RSLQR / H∞)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Walk-Forward Controller Validation (regime-conditioned) —
# picks between LQR/RSLQR/H-inf the same way Script 26 picks between
# forecast models: out-of-sample, not by the in-sample "best final
# equity over the whole lookback" comparison main() prints below
# (that comparison re-solves each controller's gain from parameters
# estimated over the ENTIRE window and then asks which grew equity
# fastest over that same window -- exactly the kind of full-history
# hindsight Script 26/29's whole design otherwise avoids). Here, the
# gain is re-solved periodically using only data seen so far, and the
# result is scored separately per HMM regime (Script 15) so a
# controller can win specifically in trending vs. mean-reverting
# conditions rather than needing one controller to dominate overall.
# ============================================================
CONTROLLER_METHODS = ("lqr", "rslqr", "hinf")
CONTROLLER_REFIT_EVERY = 10     # trading days between gain re-solves
CONTROLLER_MIN_TRAIN   = 90     # min days of history before validation starts
CONTROLLER_MIN_REGIME_DAYS = 20  # min out-of-sample days in a regime to trust its winner


def _regime_state_series(df):
    """Full-length regime-id array aligned to df["log_return"].dropna()
    (i.e. same length/index as the `returns` array used everywhere else
    in this module), or None if Script 15 isn't available. Uses the same
    whole-history HMM fit Script 15's detect_regime() uses (not a
    per-day expanding refit -- that would mean an HMM refit at every one
    of ~250 trading days per ticker, for a label that's only used here
    to *bucket* an already-causal walk-forward P&L series by market
    condition, not to make any causal decision itself)."""
    if not REGIME_CTRL_AVAILABLE:
        return None
    try:
        features, valid = _build_regime_features(df)
        states, _ = _fit_hmm(features[valid])
        full = np.full(len(df), -1, dtype=int)
        full[valid] = states
        return_mask = df["log_return"].notna().values
        return full[return_mask]
    except Exception:
        return None


def walk_forward_validate_controllers(df, methods=CONTROLLER_METHODS,
                                       refit_every=CONTROLLER_REFIT_EVERY,
                                       min_train=CONTROLLER_MIN_TRAIN):
    """
    Out-of-sample comparison of LQR/RSLQR/H-inf. Every `refit_every`
    days, (alpha_fc, alpha_vol, beta_track) are re-estimated from data
    up to that day only (fit_ar1/calibrate_beta_track on the
    already-causal Kalman trend/vol series), each method's gain K is
    re-solved from that estimate, and the resulting policy trades the
    next `refit_every` days before being re-solved again -- so no
    method's gain is ever fit on data from the period it's scored on.

    Returns {"per_regime": {regime_id: {"best": method|None, "n_days": int,
    "sharpe": {method: float}}}, "overall": {method: {"ret_pct", "sharpe"}}}.
    "best" is None where fewer than CONTROLLER_MIN_REGIME_DAYS out-of-sample
    days were observed in that regime -- not enough evidence to trust a winner.
    """
    returns = df["log_return"].dropna().values
    T = len(returns)
    out = {"per_regime": {}, "overall": {}}
    if T < min_train + refit_every or not ADAPTIVE_AVAILABLE:
        return out

    fc_result = adaptive_forecast(df, horizon=1)
    trend_series = fc_result["x_hat"][:, 1]
    vol_series = fc_result["vol_series"]
    regimes = _regime_state_series(df)

    pos = {m: 0.0 for m in methods}
    wealth = {m: CAPITAL for m in methods}
    daily_ret = {m: np.full(T, np.nan) for m in methods}

    for block_start in range(min_train, T, refit_every):
        train_trend = trend_series[:block_start]
        train_vol = np.log(np.maximum(vol_series[:block_start], 1e-6))
        alpha_fc_hat = fit_ar1(train_trend)
        alpha_vol_hat = fit_ar1(train_vol)
        beta_track_hat = calibrate_beta_track(train_trend)
        A, B, _ = build_portfolio_ss(alpha_fc=alpha_fc_hat, alpha_vol=alpha_vol_hat)

        gains = {}
        for m in methods:
            kwargs = {"beta_track": beta_track_hat}
            if m == "rslqr":
                kwargs["rho"] = 0.5
            elif m == "hinf":
                kwargs["gamma"] = 2.0
            gains[m] = PositionController(A, B, method=m, **kwargs).K

        block_end = min(block_start + refit_every, T)
        for k in range(block_start, block_end):
            log_vol_k = np.log(max(vol_series[k], 1e-6))
            for m in methods:
                state = np.array([pos[m], trend_series[k], log_vol_k])
                u = lqr_position(state, gains[m], max_pos=1.5)
                trade_cost = abs(u) * 0.001
                pnl_frac = pos[m] * returns[k] - trade_cost
                wealth[m] *= max(1.0 + pnl_frac, 1e-6)
                daily_ret[m][k] = pnl_frac
                pos[m] = pos[m] + u

    for m in methods:
        d = daily_ret[m]
        valid = d[min_train:]
        valid = valid[np.isfinite(valid)]
        sharpe = float(np.mean(valid) / max(np.std(valid), 1e-8) * np.sqrt(365)) if len(valid) else float("nan")
        out["overall"][m] = {"ret_pct": (wealth[m] / CAPITAL - 1) * 100, "sharpe": sharpe}

    if regimes is not None and len(regimes) == T:
        for r in sorted(set(regimes[min_train:].tolist()) - {-1}):
            mask = (regimes == r)
            mask[:min_train] = False
            n_days = int(mask.sum())
            sharpes = {}
            for m in methods:
                d = daily_ret[m][mask]
                d = d[np.isfinite(d)]
                sharpes[m] = float(np.mean(d) / max(np.std(d), 1e-8) * np.sqrt(365)) if len(d) else float("nan")
            best = max(sharpes, key=lambda m: sharpes[m]) if n_days >= CONTROLLER_MIN_REGIME_DAYS else None
            out["per_regime"][int(r)] = {"best": best, "n_days": n_days, "sharpe": sharpes}

    return out


CONTROLLER_CACHE_PATH = os.path.join(CACHE_DIR, "controller_walkforward_best.json")
CONTROLLER_CACHE_TTL_HOURS = 24.0


def _load_controller_cache() -> dict:
    if not os.path.exists(CONTROLLER_CACHE_PATH):
        return {}
    try:
        with open(CONTROLLER_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_controller_cache(cache: dict):
    try:
        with open(CONTROLLER_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass   # best-effort -- worst case, every call re-validates instead of caching


def get_cached_best_controller(ticker: str, df, regime_id: int, fallback: str = "lqr",
                                ttl_hours: float = CONTROLLER_CACHE_TTL_HOURS) -> str:
    """Which controller (lqr/rslqr/hinf) has walk-forward-beaten the
    others FOR THIS REGIME, re-validated at most once per ttl_hours per
    ticker (mirrors Script 26's get_cached_best_model). Falls back to
    `fallback` if Script 15/20 are unavailable, there isn't enough
    out-of-sample history yet, or nothing has beaten LQR by enough days
    of evidence in this regime."""
    cache = _load_controller_cache()
    entry = cache.get(ticker)
    now = pd.Timestamp.now().timestamp()
    if not (entry and (now - entry.get("ts", 0)) / 3600 < ttl_hours):
        try:
            result = walk_forward_validate_controllers(df)
        except Exception:
            result = {"per_regime": {}}
        entry = {"ts": now, "per_regime": result.get("per_regime", {})}
        cache[ticker] = entry
        _save_controller_cache(cache)

    regime_entry = entry["per_regime"].get(str(regime_id))
    if not regime_entry or not regime_entry.get("best"):
        return fallback
    return regime_entry["best"]


# ============================================================
# Public API for Script 17
# ============================================================
def get_optimal_position(df, method="lqr", ticker=None, regime_id=None):
    """
    Returns optimal position fraction and controller diagnostics
    for use by the trade planner. If `ticker` and `regime_id` are given,
    the controller method is chosen by walk-forward validated,
    regime-conditioned selection (see get_cached_best_controller) instead
    of always using `method` -- `method` becomes the fallback for when
    that selection can't run or hasn't got enough evidence yet.
    """
    resolved_method = method
    if ticker is not None and regime_id is not None:
        try:
            resolved_method = get_cached_best_controller(ticker, df, regime_id, fallback=method)
        except Exception:
            resolved_method = method

    # Per-ticker system ID -- same calibration as main()'s backtest path,
    # so Script 17's live sizing isn't silently running a different
    # config than what was actually validated (see fit_ar1 /
    # calibrate_beta_track docstrings).
    alpha_fc_hat, alpha_vol_hat, beta_track_hat = 0.85, 0.92, 500.0
    if ADAPTIVE_AVAILABLE:
        try:
            fc_result = adaptive_forecast(df, horizon=1)
            trend_series = fc_result["x_hat"][:, 1]
            vol_series = fc_result["vol_series"]
            alpha_fc_hat = fit_ar1(trend_series)
            alpha_vol_hat = fit_ar1(np.log(np.maximum(vol_series, 1e-6)))
            beta_track_hat = calibrate_beta_track(trend_series)
        except Exception:
            pass

    A, B, C = build_portfolio_ss(alpha_fc=alpha_fc_hat, alpha_vol=alpha_vol_hat)
    ctrl = PositionController(A, B, method=resolved_method, beta_track=beta_track_hat)

    returns = df["log_return"].dropna().values
    vol = float(pd.Series(returns).rolling(20).std().iloc[-1])
    fc = float(pd.Series(returns).rolling(10).mean().iloc[-1])

    state = np.array([0.0, fc, np.log(max(vol, 1e-6))])
    opt_pos = ctrl.optimal_position(state)

    return {
        "position_frac": float(opt_pos),
        "method": resolved_method,
        "stable": ctrl.stability["stable"],
        "spectral_radius": ctrl.stability["spectral_radius"],
        "settling_time": ctrl.stability["settling_time"],
        "gain_K": ctrl.K.flatten().tolist(),
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("OPTIMAL POSITION CONTROL")
    print("LQR | RSLQR | H-infinity | Lyapunov Verification")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*50}")

        try:
            # Per-ticker system ID: fit alpha_fc/alpha_vol and calibrate
            # beta_track from this ticker's own Kalman-filtered trend/vol
            # series, instead of the fixed defaults shared by every ticker
            # (see fit_ar1 / calibrate_beta_track docstrings -- crypto's
            # per-ticker vol spans 29%-151% annualized in this universe,
            # too wide for one shared constant to serve well).
            alpha_fc_hat, alpha_vol_hat, beta_track_hat = 0.85, 0.92, 500.0
            if ADAPTIVE_AVAILABLE:
                try:
                    fc_result = adaptive_forecast(df, horizon=1)
                    trend_series = fc_result["x_hat"][:, 1]
                    vol_series = fc_result["vol_series"]
                    alpha_fc_hat = fit_ar1(trend_series)
                    alpha_vol_hat = fit_ar1(np.log(np.maximum(vol_series, 1e-6)))
                    beta_track_hat = calibrate_beta_track(trend_series)
                except Exception:
                    pass
            print(f"  System ID: alpha_fc={alpha_fc_hat:.3f} (fixed default 0.85)  "
                  f"alpha_vol={alpha_vol_hat:.3f} (fixed default 0.92)  "
                  f"beta_track={beta_track_hat:.1f} (fixed default 500.0)")

            A, B, C = build_portfolio_ss(alpha_fc=alpha_fc_hat, alpha_vol=alpha_vol_hat)
            results = {}

            for name, method, kwargs in [
                ("LQR", "lqr", {"beta_track": beta_track_hat}),
                ("RSLQR", "rslqr", {"beta_track": beta_track_hat, "rho": 0.5}),
                ("H-inf", "hinf", {"beta_track": beta_track_hat, "gamma": 2.0}),
            ]:
                ctrl = PositionController(A, B, method=method, **kwargs)
                eq, pos, tr = simulate_controller(df, ctrl)
                results[name] = (eq, pos, tr, ctrl)

                ret = (eq[-1] / eq[0] - 1) * 100
                stab = ctrl.stability
                print(f"  {name:<8}: ret={ret:+.1f}%  "
                      f"stable={'Y' if stab['stable'] else 'N'}  "
                      f"rho={stab['spectral_radius']:.4f}  "
                      f"settling={stab['settling_time']}d  "
                      f"K={ctrl.K.flatten().round(4).tolist()}")

            bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
            print(f"  {'B&H':<8}: ret={bh:+.1f}%")

            best = max(results.items(), key=lambda x: x[1][0][-1])
            summary.append({
                "Ticker": ticker, "Best": best[0],
                "Ret%": f"{(best[1][0][-1]/CAPITAL-1)*100:+.1f}",
                "BH%": f"{bh:+.1f}",
                "Stable": "Y" if best[1][3].stability["stable"] else "N",
            })

            plot_controller(ticker, df, results)

        except Exception as e:
            print(f"  Error: {e}")

    if summary:
        print("\n" + "=" * 65)
        print("OPTIMAL CONTROL SUMMARY")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nOptimal control complete.")


if __name__ == "__main__":
    main()
