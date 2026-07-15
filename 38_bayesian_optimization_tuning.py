"""
================================================================
Script 38 — Bayesian Numerical Analysis: Bayesian Optimization +
Bayesian Quadrature
================================================================
Covers: Bayesian optimization, Bayesian quadrature. Closes out the
Bayesian methods series by applying it to the suite's own machinery
rather than another return-distribution model.

  1. BAYESIAN OPTIMIZATION -- tunes Script 25's confidence threshold
     (get_ml_signal's conf_threshold override -- normally auto-
     calibrated per ticker, here swept directly) against Script 16's
     actual backtested Sharpe ratio, using a GP surrogate (Script 37's
     tool) + Expected Improvement acquisition. This is the capstone
     integration: the objective function IS the walk-forward signal
     generation + backtest machinery built earlier in this series, not
     a toy function. Compared directly against a brute-force grid
     search over the same range -- the actual point of Bayesian
     optimization is finding a comparably good optimum with far fewer
     expensive evaluations, and that claim is checked, not assumed.

  2. BAYESIAN QUADRATURE -- estimates an option's expected discounted
     payoff (Script 3's territory) from a handful of function
     evaluations, using a GP fit over log-price and the closed-form
     Gaussian-kernel/Gaussian-measure integral (O'Hagan 1991; Rasmussen &
     Ghahramani 2003, "Bayesian Monte Carlo"). Verified against the exact
     closed-form Black-Scholes price -- unlike most integrals encountered
     in practice, this one has a known right answer, so both Bayesian
     quadrature and naive Monte Carlo can be graded directly, at matched
     (tiny) sample budgets. Uses 365-day annualization throughout (crypto
     trades 24/7 -- see data_loader/Script 16's own convention), not the
     252-trading-day convention this cluster's original equities version
     used.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from importlib import import_module as _im
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

_r16 = _im("16_backtesting")
_r25 = _im("25_ml_forecast_signal")
backtest      = _r16.backtest
calc_metrics  = _r16.calc_metrics
get_ml_signal = _r25.get_ml_signal

PLOT_STYLE = "seaborn-v0_8-darkgrid"
BAYESOPT_N_TICKERS = 2    # was 3 -- crypto's dynamic top-gainers/volume universe has no
                          # fixed SPY/AAPL-equivalent names -- just take the first N of
                          # whatever load_all_assets returns today. This script was ~430s
                          # of a 34-script full-suite run (the single biggest contributor);
                          # cut here, plus GRIDSEARCH_N and WF_RETRAIN_EVERY below.
CONF_BOUNDS = (0.52, 0.85)
BAYESOPT_N_INIT = 4
BAYESOPT_N_ITER = 10
GRIDSEARCH_N = 15        # was 25 -- still clearly > BayesOpt's 14 evals (the comparison
                         # this script exists to make), just less of it
WF_MIN_TRAIN = 150        # walk-forward warmup before the first refit
WF_RETRAIN_EVERY = 10     # was 5 -- roughly halves refit count per objective-function
                          # call, the dominant per-eval cost

ANN_DAYS = 365            # crypto trades 24/7 -- not the 252-trading-day convention
RF = 0.0                  # no natural risk-free analog for a perp; kept at 0 rather
                          # than borrowing an equities-market rate
BQ_HORIZON_DAYS = 21
BQ_N_DESIGN = 20
BQ_MC_SMALL = 20
BQ_MC_LARGE = 100_000


# ============================================================
# 1. Bayesian Optimization — GP surrogate + Expected Improvement
# ============================================================
def build_walk_forward_signal(df, conf_threshold, min_train=WF_MIN_TRAIN,
                               retrain_every=WF_RETRAIN_EVERY):
    """
    Expanding-window LONG/SHORT/FLAT signal series, no lookahead: refits
    via get_ml_signal every `retrain_every` bars using only data through
    that bar (get_ml_signal's own conf_threshold override bypasses its
    usual per-ticker auto-calibration -- exactly the knob being tuned
    here), holding the signal until the next refit.
    """
    n = len(df)
    sig = np.zeros(n)
    signal_map = {"LONG": 1, "SHORT": -1, "FLAT": 0}
    current = 0.0
    for i in range(min_train, n):
        if (i - min_train) % retrain_every == 0:
            try:
                result = get_ml_signal(df.iloc[:i + 1], conf_threshold=conf_threshold)
                current = signal_map.get(result["signal"], 0)
            except Exception:
                current = 0.0
        sig[i] = current
    return sig


def objective_sharpe(conf_threshold, df):
    sig = build_walk_forward_signal(df, conf_threshold)
    equity, _ = backtest(df, sig)
    metrics = calc_metrics(equity, df)
    return metrics["sharpe"]


def expected_improvement(X_grid, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_grid.reshape(-1, 1), return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = mu - y_best - xi
    z = imp / sigma
    ei = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    return ei


def bayesian_optimize(objective_fn, bounds, n_init=BAYESOPT_N_INIT, n_iter=BAYESOPT_N_ITER, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(bounds[0], bounds[1], size=n_init)
    y = np.array([objective_fn(x) for x in X])

    kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(0.1, (1e-2, 1e1)) + WhiteKernel(1e-2, (1e-4, 1e0))
    best_history = [y.max()]

    for _ in range(n_iter):
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       normalize_y=True, random_state=seed).fit(X.reshape(-1, 1), y)
        grid = np.linspace(bounds[0], bounds[1], 400)
        ei = expected_improvement(grid, gp, y.max())
        x_next = float(grid[np.argmax(ei)])
        y_next = objective_fn(x_next)
        X, y = np.append(X, x_next), np.append(y, y_next)
        best_history.append(y.max())

    best_idx = int(np.argmax(y))
    return {"X": X, "y": y, "best_x": float(X[best_idx]), "best_y": float(y[best_idx]),
            "best_history": np.array(best_history), "n_evals": len(X)}


def grid_search(objective_fn, bounds, n=GRIDSEARCH_N):
    grid = np.linspace(bounds[0], bounds[1], n)
    y = np.array([objective_fn(x) for x in grid])
    best_idx = int(np.argmax(y))
    return {"grid": grid, "y": y, "best_x": float(grid[best_idx]), "best_y": float(y[best_idx])}


# ============================================================
# 2. Bayesian Quadrature — GP-based numerical integration
# ============================================================
def bs_call_price(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def bayesian_quadrature_call_price(S0, K, r, sigma, T, n_design=BQ_N_DESIGN, seed=0):
    """
    Prices E[e^{-rT} max(S_T-K,0)] from n_design evaluations by fitting a
    GP over x=log(S_T) (Gaussian under GBM) and using the closed-form
    RBF-kernel-mean-against-a-Gaussian-measure integral (O'Hagan 1991) --
    exact given the fitted kernel hyperparameters, no numerical
    integration of the kernel mean needed.
    """
    rng = np.random.default_rng(seed)
    mu_p = np.log(S0) + (r - 0.5 * sigma ** 2) * T
    sigma_p = sigma * np.sqrt(T)

    x = rng.normal(mu_p, sigma_p, size=n_design)
    payoff = np.maximum(np.exp(x) - K, 0.0)

    kernel = ConstantKernel(1.0, (1e-3, 1e6)) * RBF(sigma_p, (1e-3, 1e3)) + WhiteKernel(1e-6, (1e-8, 1e-1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                   normalize_y=True, random_state=seed).fit(x.reshape(-1, 1), payoff)

    k = gp.kernel_
    amp2 = k.k1.k1.constant_value
    length = k.k1.k2.length_scale
    noise = k.k2.noise_level
    y_mean, y_std = gp._y_train_mean, gp._y_train_std

    K_mat = k(x.reshape(-1, 1)) + noise * np.eye(n_design)
    z = amp2 * np.sqrt(length ** 2 / (length ** 2 + sigma_p ** 2)) * \
        np.exp(-0.5 * (x - mu_p) ** 2 / (length ** 2 + sigma_p ** 2))
    alpha = np.linalg.solve(K_mat, (payoff - y_mean) / y_std)
    integral_std_units = float(z @ alpha)
    expected_payoff = integral_std_units * y_std + y_mean

    return float(np.exp(-r * T) * expected_payoff), x, payoff


def mc_call_price(S0, K, r, sigma, T, n_paths, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(np.log(S0) + (r - 0.5 * sigma ** 2) * T, sigma * np.sqrt(T), size=n_paths)
    payoff = np.maximum(np.exp(x) - K, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


# ============================================================
# Plotting
# ============================================================
def plot_bayesopt(ticker, bo_result, gs_result, bounds):
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax0 = axes[0]
    ax0.plot(gs_result["grid"], gs_result["y"], color="gray", lw=1.3, label="Grid search (ground truth)")
    ax0.scatter(bo_result["X"], bo_result["y"], color="crimson", s=25, zorder=5, label="BayesOpt evaluations")
    ax0.axvline(bo_result["best_x"], color="crimson", lw=1.0, ls="--")
    ax0.axvline(gs_result["best_x"], color="gray", lw=1.0, ls=":")
    ax0.set_xlabel("conf_threshold"); ax0.set_ylabel("backtested Sharpe")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Objective Surface (Sharpe vs conf_threshold)", fontsize=10)
    ax0.grid(alpha=0.3)

    ax1 = axes[1]
    ax1.plot(bo_result["best_history"], "o-", color="steelblue", lw=1.5)
    ax1.axhline(gs_result["best_y"], color="gray", lw=1.0, ls="--", label=f"Grid search best ({GRIDSEARCH_N} evals)")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("BayesOpt iteration"); ax1.set_ylabel("best Sharpe so far")
    ax1.set_title(f"Convergence  ({bo_result['n_evals']} evals vs {GRIDSEARCH_N} for grid)", fontsize=10)
    ax1.grid(alpha=0.3)

    fig.suptitle(f"{ticker} — Bayesian Optimization of ML Signal Confidence Threshold",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_quadrature(ticker, bq_price, mc_small_price, mc_large_price, true_price, x, payoff):
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax0 = axes[0]
    ax0.scatter(x, payoff, color="steelblue", s=20, label=f"{len(x)} design points")
    ax0.set_xlabel("log(S_T)"); ax0.set_ylabel("payoff")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Bayesian Quadrature Design Points", fontsize=10)
    ax0.grid(alpha=0.3)

    ax1 = axes[1]
    names = [f"Bayesian Quad\n(n={BQ_N_DESIGN})", f"Naive MC\n(n={BQ_MC_SMALL})",
             f"Naive MC\n(n={BQ_MC_LARGE:,})", "True (Black-Scholes)"]
    vals = [bq_price, mc_small_price, mc_large_price, true_price]
    colors = ["crimson", "gray", "steelblue", "forestgreen"]
    ax1.bar(names, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("call price ($)")
    ax1.set_title("Price Estimate Comparison", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{ticker} — Bayesian Quadrature vs Monte Carlo (Option Pricing)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("BAYESIAN NUMERICAL ANALYSIS — BAYESIAN OPTIMIZATION + QUADRATURE")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)

    # ── Part 1: Bayesian Optimization ────────────────────────
    bayesopt_tickers = list(assets_data.keys())[:BAYESOPT_N_TICKERS]
    print(f"\n{'='*70}\nPART 1 -- BAYESIAN OPTIMIZATION "
          f"(tuning Script 25's conf_threshold vs Script 16's Sharpe)\n"
          f"Tickers: {bayesopt_tickers}\n{'='*70}")
    bo_summary = []
    for ticker in bayesopt_tickers:
        df = assets_data[ticker]
        if len(df) < WF_MIN_TRAIN + 60:
            print(f"\n  {ticker}: skipped -- need >= {WF_MIN_TRAIN + 60} rows for walk-forward")
            continue
        objective_fn = lambda ct, _df=df: objective_sharpe(ct, _df)

        gs_result = grid_search(objective_fn, CONF_BOUNDS)
        bo_result = bayesian_optimize(objective_fn, CONF_BOUNDS)

        gap = gs_result["best_y"] - bo_result["best_y"]
        print(f"\n  {ticker}: Grid search best Sharpe={gs_result['best_y']:.3f} "
              f"at conf={gs_result['best_x']:.3f}  ({GRIDSEARCH_N} evals)")
        print(f"  {ticker}: BayesOpt best Sharpe={bo_result['best_y']:.3f} "
              f"at conf={bo_result['best_x']:.3f}  ({bo_result['n_evals']} evals)  "
              f"gap={gap:+.3f}")

        bo_summary.append({
            "Ticker": ticker, "GridSearch_BestSharpe": f"{gs_result['best_y']:.3f}",
            "GridSearch_Evals": GRIDSEARCH_N,
            "BayesOpt_BestSharpe": f"{bo_result['best_y']:.3f}",
            "BayesOpt_Evals": bo_result["n_evals"], "Gap": f"{gap:+.3f}",
        })
        plot_bayesopt(ticker, bo_result, gs_result, CONF_BOUNDS)

    if bo_summary:
        print("\n" + pd.DataFrame(bo_summary).to_string(index=False))

    # ── Part 2: Bayesian Quadrature ───────────────────────────
    print(f"\n{'='*70}\nPART 2 -- BAYESIAN QUADRATURE "
          f"(option pricing vs closed-form Black-Scholes)\n{'='*70}")
    bq_summary = []
    for i, (ticker, df) in enumerate(assets_data.items()):
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            continue
        S0 = float(df["close"].iloc[-1])
        sigma = float(returns.std() * np.sqrt(ANN_DAYS))
        T = BQ_HORIZON_DAYS / ANN_DAYS
        K = S0   # ATM, same convention as Script 3

        # Independent draw per ticker -- a fixed seed for every ticker would
        # reuse the identical standardized random draw at every scale,
        # turning what should be N independent trials into 1 trial repeated
        # N times, which would understate how noisy small-N estimators are.
        seed = i
        true_price = bs_call_price(S0, K, RF, sigma, T)
        bq_price, x, payoff = bayesian_quadrature_call_price(S0, K, RF, sigma, T, seed=seed)
        mc_small = mc_call_price(S0, K, RF, sigma, T, BQ_MC_SMALL, seed=seed)
        mc_large = mc_call_price(S0, K, RF, sigma, T, BQ_MC_LARGE, seed=seed)

        bq_err = abs(bq_price - true_price) / true_price
        mc_small_err = abs(mc_small - true_price) / true_price

        print(f"\n  {ticker}: True(BS)={format_price(true_price)}  "
              f"BayesQuad(n={BQ_N_DESIGN})={format_price(bq_price)} (err={bq_err:.1%})  "
              f"NaiveMC(n={BQ_MC_SMALL})={format_price(mc_small)} (err={mc_small_err:.1%})  "
              f"NaiveMC(n={BQ_MC_LARGE:,})={format_price(mc_large)}")

        bq_summary.append({
            "Ticker": ticker, "TruePrice": f"{true_price:.4f}",
            "BayesQuad": f"{bq_price:.4f}", "BQ_Err%": f"{bq_err*100:.1f}",
            "NaiveMC_Small": f"{mc_small:.4f}", "NaiveMC_SmallErr%": f"{mc_small_err*100:.1f}",
            "NaiveMC_Large": f"{mc_large:.4f}",
        })

        if i < 4:   # representative plots -- crypto universe is already small (~10),
                    # no need for a fixed-name subset like the equities original used
            plot_quadrature(ticker, bq_price, mc_small, mc_large, true_price, x, payoff)

    if bq_summary:
        print("\n" + "=" * 70)
        print("BAYESIAN QUADRATURE SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(bq_summary).to_string(index=False))
        bq_errs = [float(s["BQ_Err%"]) for s in bq_summary]
        mc_errs = [float(s["NaiveMC_SmallErr%"]) for s in bq_summary]
        print(f"\n  Mean abs error -- Bayesian Quadrature (n={BQ_N_DESIGN}): {np.mean(bq_errs):.1f}%   "
              f"Naive MC (n={BQ_MC_SMALL}): {np.mean(mc_errs):.1f}%")

    print("\nBayesian numerical analysis complete.")


if __name__ == "__main__":
    main()
