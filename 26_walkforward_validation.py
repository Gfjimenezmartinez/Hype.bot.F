"""
================================================================
Script 26 — Walk-Forward / Expanding-Window Forecast Validation
================================================================
Script 20 fits its Kalman filter and reports IN-SAMPLE fit statistics.
None of the suite's forecasting tools ask the question that actually
matters for trading: "if you had only known the past, how well would
this have predicted tomorrow, day after day, for a year?"

This script answers that with a true expanding-window backtest: at each
day t, fit/update each candidate model using ONLY data up to t-1, forecast
r_t, then compare to the realized r_t. No lookahead. Metrics:

  RMSE / MAE       — forecast error magnitude
  Hit Rate         — % of days the predicted direction was correct
  Theil's U        — RMSE(model) / RMSE(naive no-change forecast)
                      U < 1  → model beats the trivial baseline
                      U >= 1 → model adds no value out-of-sample

This is the tool that tells you whether ARMA/EWMA/AR(1) fitting is
actually earning its keep, or just overfitting in-sample noise.

Also validates multi-day-ahead cumulative forecasts (1d/1w/1m), not
just 1-step-ahead -- the horizons Script 17 and options-pricing scripts
actually consume, walk-forward-validated for the first time rather than
just assumed accurate at longer lead times.
================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price, CACHE_DIR

try:
    from importlib import import_module as _im
    _r20 = _im("20_adaptive_forecast")
    KALMAN_AVAILABLE = True
except Exception:
    KALMAN_AVAILABLE = False

PLOT_STYLE = "seaborn-v0_8-darkgrid"
MIN_TRAIN  = 60     # minimum history before the first forecast is made
EWMA_SPAN  = 20
HORIZONS   = {"1d": 1, "1w": 7, "1m": 30}   # calendar days -- crypto trades 24/7,
                                             # no trading-day/weekend convention


# ============================================================
# Candidate Models (all refit at every step, using data < t only)
# ============================================================
def _ar1_forecast(train):
    x, y = train[:-1], train[1:]
    if len(x) < 5 or x.std() < 1e-12:
        return float(train.mean())
    phi = np.cov(x, y)[0, 1] / np.var(x)
    phi = np.clip(phi, -0.99, 0.99)
    mu = float(train.mean())
    return mu + phi * (train[-1] - mu)


def _ar1_fit(train):
    """Same fit as _ar1_forecast, but returns (mu, phi, last) instead of
    just the 1-step point forecast, so _ar1_decaying_sum below can derive
    any horizon from a single fit."""
    x, y = train[:-1], train[1:]
    if len(x) < 5 or x.std() < 1e-12:
        return float(train.mean()), 0.0, float(train[-1])
    phi = np.cov(x, y)[0, 1] / np.var(x)
    phi = float(np.clip(phi, -0.99, 0.99))
    return float(train.mean()), phi, float(train[-1])


def _ar1_decaying_sum(x0, mu, phi, h):
    """
    Sum_{i=1}^{h} [mu + phi^i*(x0-mu)] -- the h-day CUMULATIVE forecast
    implied by a single fitted AR(1) step, closed-form (no need to refit
    or simulate h times). Reused for both:
      - the return-space AR1 candidate: mu=train.mean(), phi=fitted AR(1)
        coefficient, x0=train[-1]
      - the Kalman trend state: mu=0, phi=alpha_trend (Script 20's
        build_state_space -- the trend row decays toward zero, NOT toward
        the historical mean return, which is the actual difference
        between the two candidates), x0=filtered trend state
    """
    if h <= 0:
        return 0.0
    if abs(phi - 1.0) < 1e-8:
        return h * mu + (x0 - mu) * h
    decay_sum = phi * (1 - phi ** h) / (1 - phi)
    return h * mu + (x0 - mu) * decay_sum


def _kalman_trend_states(returns, min_train=MIN_TRAIN, vol_window=60):
    """
    Expanding-window Kalman walk-forward filter, no lookahead: at each
    step t, refits Script 20's 3-state price/trend/vol filter
    (fit_state_space/kalman_observer) on returns[:t] only. Returns the
    filtered trend state at each t plus alpha_trend (the trend row's own
    AR(1) decay, fixed by fit_state_space's default) -- together these let
    both the 1-step forecast (trend state directly, same "log_price(k+1) =
    log_price(k) + trend(k)" structure Script 20 itself relies on) and
    any h-day cumulative forecast (via _ar1_decaying_sum) be derived
    without re-running the filter per horizon.

    This refits from scratch at each t (same expanding-window pattern
    already used for AR1/HistMean/EWMA below) rather than running one
    single-pass filter, because Script 20's own kalman_observer seeds its
    initial log-vol state from init_vol -- computing that from a trailing
    window at each t (not the last-60-days-of-the-WHOLE-series that
    Script 20's own adaptive_forecast() uses) is what keeps this genuinely
    causal. O(T^2) but T~500 and each fit is cheap numpy, so this is fast
    in practice.

    Uses fit_state_space (R = returns[:t]'s own realized variance) rather
    than the fixed build_state_space -- same per-ticker noise-calibration
    fix as Script 20's production path, still fully causal since variance
    at each t is estimated only from data available up to t.
    """
    T = len(returns)
    log_prices = np.concatenate([[0.0], np.cumsum(returns)])
    trend = np.full(T, np.nan)
    alpha_trend = 0.98

    for t in range(min_train, T):
        A, C, _, Q, R = _r20.fit_state_space(returns[:t])
        alpha_trend = float(A[1, 1])
        init_vol = float(np.std(returns[max(0, t - vol_window):t])) if t > 5 else 0.02
        x_hat, _, _, _ = _r20.kalman_observer(log_prices[:t + 1], A, C, Q, R, init_vol=init_vol)
        trend[t] = x_hat[-1, 1]

    return trend, alpha_trend


def _kalman_walkforward_forecasts(returns, min_train=MIN_TRAIN, vol_window=60):
    """1-step-ahead forecast = the filtered trend state directly."""
    trend, _ = _kalman_trend_states(returns, min_train=min_train, vol_window=vol_window)
    return trend


ENSEMBLE_COMPONENTS = ["HistMean", "EWMA", "AR1"]
ENSEMBLE_MIN_HISTORY = 20   # min past errors observed before trusting adaptive weights
ENSEMBLE_LOOKBACK    = 60   # window of past errors used to weight each component


def walk_forward_forecasts(returns, min_train=MIN_TRAIN, ewma_span=EWMA_SPAN):
    """
    Expanding-window 1-step-ahead forecasts. Returns (actual, {model: fc}),
    both length T with NaN before min_train.

    "Ensemble" combines HistMean/EWMA/AR1 via inverse-recent-MSE weighting:
    each component's weight at time t is based ONLY on its own error history
    strictly before t (no lookahead) — components that have been more
    accurate recently get more say. This is a standard forecast-combination
    technique (Bates & Granger 1969) and often beats every individual
    component, including naive, even when none of the components do alone.

    "Kalman" (Script 20's trend-state estimator) is included here for the
    first time — it was never walk-forward validated before, only shown
    with in-sample fit plots. It's NOT in the Ensemble (kept separate so
    its standalone accuracy is visible before deciding whether it earns
    a place in the blend).
    """
    T = len(returns)
    model_names = ["Naive", "HistMean", "EWMA", "AR1", "Ensemble"]
    if KALMAN_AVAILABLE:
        model_names.append("Kalman")
    fc = {m: np.full(T, np.nan) for m in model_names}
    actual = np.full(T, np.nan)
    err_hist = {m: [] for m in ENSEMBLE_COMPONENTS}

    if KALMAN_AVAILABLE:
        try:
            fc["Kalman"] = _kalman_walkforward_forecasts(returns, min_train=min_train)
        except Exception:
            pass

    for t in range(min_train, T):
        train = returns[:t]
        actual[t] = returns[t]
        fc["Naive"][t] = 0.0
        fc["HistMean"][t] = float(train.mean())
        fc["EWMA"][t] = float(pd.Series(train).ewm(span=ewma_span).mean().iloc[-1])
        fc["AR1"][t] = _ar1_forecast(train)

        if all(len(err_hist[m]) >= ENSEMBLE_MIN_HISTORY for m in ENSEMBLE_COMPONENTS):
            inv_mse = {m: 1.0 / max(np.mean(np.array(err_hist[m][-ENSEMBLE_LOOKBACK:]) ** 2), 1e-12)
                       for m in ENSEMBLE_COMPONENTS}
            wsum = sum(inv_mse.values())
            fc["Ensemble"][t] = sum(inv_mse[m] / wsum * fc[m][t] for m in ENSEMBLE_COMPONENTS)
        else:
            fc["Ensemble"][t] = float(np.mean([fc[m][t] for m in ENSEMBLE_COMPONENTS]))

        for m in ENSEMBLE_COMPONENTS:
            err_hist[m].append(actual[t] - fc[m][t])

    if KALMAN_AVAILABLE:
        fc["Kalman"][:min_train] = np.nan  # mask to the same evaluation window as everything else

    return actual, fc


def compute_metrics(actual, fc_dict, min_train=MIN_TRAIN):
    a = actual[min_train:]
    out = {}
    for name, f in fc_dict.items():
        p = f[min_train:]
        err = a - p
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        same_sign = np.sign(a) == np.sign(p)
        hit = float(np.mean(same_sign[a != 0]))
        out[name] = {"RMSE": rmse, "MAE": mae, "HitRate": hit, "err": err}

    naive_rmse = out["Naive"]["RMSE"]
    for name in out:
        out[name]["TheilU"] = out[name]["RMSE"] / naive_rmse if naive_rmse > 0 else np.nan
    return out


# ============================================================
# Multi-Horizon Validation (1d/1w/1m) -- fills the gap left by
# retiring Script 4, which validated these horizons in-sample only.
# Genuinely walk-forward here: each h-day forecast is derived
# analytically from the SAME 1-step fit at t (_ar1_decaying_sum), so
# the expensive per-t refit (AR1/Kalman) runs once regardless of how
# many horizons are checked -- no O(T^2 * n_horizons) blowup.
# ============================================================
def walk_forward_forecasts_multi(returns, min_train=MIN_TRAIN, ewma_span=EWMA_SPAN,
                                  horizons=HORIZONS):
    """
    Multi-horizon expanding-window forecasts. Ensemble weights are
    computed from each component's 1-day error history (the freshest
    walk-forward feedback available) and reused across all horizons at
    that same t -- a component that's been accurate at 1-day-ahead
    recently gets more say at 1-week/1-month too, rather than fitting
    separate weight histories per horizon (which would need far more
    data to estimate reliably at the 1-month lookback).
    """
    T = len(returns)
    model_names = ["Naive", "HistMean", "EWMA", "AR1", "Ensemble"]
    if KALMAN_AVAILABLE:
        model_names.append("Kalman")

    fc = {m: {h_label: np.full(T, np.nan) for h_label in horizons} for m in model_names}
    actual = {h_label: np.full(T, np.nan) for h_label in horizons}
    err_hist = {m: [] for m in ENSEMBLE_COMPONENTS}   # 1-day errors drive weights at every horizon

    kalman_trend, alpha_trend = None, None
    if KALMAN_AVAILABLE:
        try:
            kalman_trend, alpha_trend = _kalman_trend_states(returns, min_train=min_train)
        except Exception:
            kalman_trend = None

    for t in range(min_train, T):
        train = returns[:t]
        mu_hist = float(train.mean())
        mu_ewma = float(pd.Series(train).ewm(span=ewma_span).mean().iloc[-1])
        mu_ar1, phi_ar1, last_ar1 = _ar1_fit(train)

        one_day = {
            "HistMean": mu_hist,
            "EWMA": mu_ewma,
            "AR1": _ar1_decaying_sum(last_ar1, mu_ar1, phi_ar1, 1),
        }
        if all(len(err_hist[m]) >= ENSEMBLE_MIN_HISTORY for m in ENSEMBLE_COMPONENTS):
            inv_mse = {m: 1.0 / max(np.mean(np.array(err_hist[m][-ENSEMBLE_LOOKBACK:]) ** 2), 1e-12)
                       for m in ENSEMBLE_COMPONENTS}
            wsum = sum(inv_mse.values())
            weights = {m: inv_mse[m] / wsum for m in ENSEMBLE_COMPONENTS}
        else:
            weights = {m: 1.0 / len(ENSEMBLE_COMPONENTS) for m in ENSEMBLE_COMPONENTS}

        for h_label, h in horizons.items():
            if t + h > T:
                continue
            actual[h_label][t] = float(np.sum(returns[t:t + h]))
            fc["Naive"][h_label][t] = 0.0
            fc["HistMean"][h_label][t] = mu_hist * h
            fc["EWMA"][h_label][t] = mu_ewma * h
            fc["AR1"][h_label][t] = _ar1_decaying_sum(last_ar1, mu_ar1, phi_ar1, h)
            if kalman_trend is not None and np.isfinite(kalman_trend[t]):
                fc["Kalman"][h_label][t] = _ar1_decaying_sum(kalman_trend[t], 0.0, alpha_trend, h)
            fc["Ensemble"][h_label][t] = sum(
                weights[m] * fc[m][h_label][t] for m in ENSEMBLE_COMPONENTS)

        for m in ENSEMBLE_COMPONENTS:
            err_hist[m].append(returns[t] - one_day[m])

    return actual, fc


def compute_metrics_multi(actual, fc_dict, horizons=HORIZONS, min_train=MIN_TRAIN):
    out = {}
    for h_label in horizons:
        a_full = actual[h_label]
        valid = ~np.isnan(a_full)
        valid[:min_train] = False
        a = a_full[valid]
        out[h_label] = {}
        for name, f in fc_dict.items():
            p = f[h_label][valid]
            if len(a) == 0:
                out[h_label][name] = {"RMSE": np.nan, "MAE": np.nan, "HitRate": np.nan}
                continue
            err = a - p
            rmse = float(np.sqrt(np.mean(err ** 2)))
            mae = float(np.mean(np.abs(err)))
            same_sign = np.sign(a) == np.sign(p)
            nz = a != 0
            hit = float(np.mean(same_sign[nz])) if nz.any() else np.nan
            out[h_label][name] = {"RMSE": rmse, "MAE": mae, "HitRate": hit}

        naive_rmse = out[h_label]["Naive"]["RMSE"]
        for name in out[h_label]:
            r = out[h_label][name]["RMSE"]
            out[h_label][name]["TheilU"] = (r / naive_rmse
                                             if naive_rmse and naive_rmse > 0 else np.nan)
    return out


# ============================================================
# Cached Best-Model Selection + Fast Single-Shot Forecast — for Script
# 17's live forecast_return(), which previously always tried Script 20's
# Kalman forecast first with a fixed ARIMA fallback, regardless of
# whether Kalman was actually winning walk-forward for that specific
# ticker. Mirrors Script 25's get_cached_best_timeframe pattern: the
# expensive O(T^2) walk-forward validation above (the Kalman refit-at-
# every-t loop) runs at most once per ttl_hours per ticker, cached;
# single_forecast() below produces today's actual point forecast in
# O(T) by reusing the same per-model building blocks without re-running
# the walk-forward loop, which exists only to VALIDATE accuracy, not to
# produce a live number.
# ============================================================
BEST_MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "walkforward_best_model.json")
BEST_MODEL_CACHE_TTL_HOURS = 24.0


def best_model_for_ticker(returns, horizon_days=5, min_train=MIN_TRAIN, ewma_span=EWMA_SPAN):
    """Walk-forward-validates every candidate model at horizon_days and
    returns (best_model_name, metrics_at_that_horizon). 'Naive' winning
    is a real, useful result -- it means nothing tested actually beats a
    no-change forecast for this ticker right now, not a failure."""
    h_label = f"{horizon_days}d"
    actual, fc = walk_forward_forecasts_multi(np.asarray(returns), min_train=min_train,
                                               ewma_span=ewma_span, horizons={h_label: horizon_days})
    metrics = compute_metrics_multi(actual, fc, horizons={h_label: horizon_days},
                                     min_train=min_train)[h_label]
    valid = {n: v for n, v in metrics.items() if np.isfinite(v["RMSE"])}
    if not valid:
        return "Naive", metrics
    best = min(valid, key=lambda n: valid[n]["RMSE"])
    return best, metrics


def _load_best_model_cache() -> dict:
    if not os.path.exists(BEST_MODEL_CACHE_PATH):
        return {}
    try:
        with open(BEST_MODEL_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_best_model_cache(cache: dict):
    try:
        with open(BEST_MODEL_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass   # best-effort -- worst case, every call re-validates instead of caching


def get_cached_best_model(ticker: str, returns, horizon_days: int = 5,
                           ttl_hours: float = BEST_MODEL_CACHE_TTL_HOURS) -> str:
    """Cached wrapper around best_model_for_ticker() -- which model has
    walk-forward-beaten the others for `ticker` at horizon_days,
    re-validated at most once per ttl_hours. Which forecast method wins
    is a slow-moving property, not worth Script 26's O(T^2) Kalman
    refit loop on every call."""
    cache = _load_best_model_cache()
    key = f"{ticker}_{horizon_days}d"
    entry = cache.get(key)
    now = pd.Timestamp.now().timestamp()
    if entry and (now - entry.get("ts", 0)) / 3600 < ttl_hours:
        return entry["best_model"]

    best, _ = best_model_for_ticker(returns, horizon_days=horizon_days)
    cache[key] = {"best_model": best, "ts": now}
    _save_best_model_cache(cache)
    return best


def single_forecast(model_name: str, returns, horizon_days: int = 5, df=None,
                     ewma_span: int = EWMA_SPAN):
    """Fast O(T) point forecast (cumulative horizon_days log-return) from
    a single named model -- the O(T) single-fit versions of the same
    model families walk_forward_forecasts_multi validates, NOT that
    O(T^2) walk-forward loop itself. Returns None (caller should fall
    back) if the model can't be evaluated (e.g. Kalman unavailable)."""
    arr = np.asarray(returns)
    if model_name == "Naive":
        return 0.0
    if model_name == "HistMean":
        return float(arr.mean()) * horizon_days
    if model_name == "EWMA":
        return float(pd.Series(arr).ewm(span=ewma_span).mean().iloc[-1]) * horizon_days
    if model_name == "AR1":
        mu, phi, last = _ar1_fit(arr)
        return _ar1_decaying_sum(last, mu, phi, horizon_days)
    if model_name == "Ensemble":
        mu, phi, last = _ar1_fit(arr)
        ar1_fc  = _ar1_decaying_sum(last, mu, phi, horizon_days)
        hist_fc = float(arr.mean()) * horizon_days
        ewma_fc = float(pd.Series(arr).ewm(span=ewma_span).mean().iloc[-1]) * horizon_days
        # Equal weights -- the same cold-start behavior
        # walk_forward_forecasts_multi itself falls back to before enough
        # rolling error history exists to weight components by accuracy.
        return float(np.mean([hist_fc, ewma_fc, ar1_fc]))
    if model_name == "Kalman":
        if not KALMAN_AVAILABLE or df is None:
            return None
        try:
            result = _r20.adaptive_forecast(df, horizon=horizon_days)
            last_price = float(df["close"].iloc[-1])
            fc_prices = result["fc_prices"]
            fc_rets = np.diff(np.log(np.concatenate([[last_price], fc_prices])))
            return float(np.sum(fc_rets))
        except Exception:
            return None
    return None


# ============================================================
# Plotting
# ============================================================
def plot_validation(ticker, dates, actual, fc_dict, metrics, min_train=MIN_TRAIN):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)
    colors = {"Naive": "gray", "HistMean": "steelblue",
              "EWMA": "darkorange", "AR1": "forestgreen", "Ensemble": "crimson",
              "Kalman": "purple"}
    d = dates[min_train:]
    a = actual[min_train:]

    # [0,0] Rolling 20d RMSE per model — is any model consistently better?
    ax0 = fig.add_subplot(gs[0, 0])
    for name, m in metrics.items():
        roll_rmse = pd.Series(m["err"] ** 2).rolling(20).mean() ** 0.5
        ax0.plot(d, roll_rmse, lw=1, color=colors.get(name), label=name, alpha=0.85)
    ax0.set_title(f"{ticker} — Rolling 20d Out-of-Sample RMSE", fontsize=10)
    ax0.legend(fontsize=7)
    ax0.grid(alpha=0.3)

    # [0,1] Cumulative squared error — separates skill from luck
    ax1 = fig.add_subplot(gs[0, 1])
    for name, m in metrics.items():
        ax1.plot(d, np.cumsum(m["err"] ** 2), lw=1.3,
                  color=colors.get(name), label=name)
    ax1.set_title("Cumulative Squared Error (lower = better)", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # [1,0] RMSE / Hit Rate bar comparison
    ax2 = fig.add_subplot(gs[1, 0])
    names = list(metrics.keys())
    rmses = [metrics[n]["RMSE"] for n in names]
    bars = ax2.bar(names, rmses, color=[colors.get(n) for n in names], alpha=0.85)
    ax2.set_title("Out-of-Sample RMSE by Model (lower = better)", fontsize=10)
    ax2.tick_params(axis="x", labelrotation=20)
    ax2.grid(alpha=0.3, axis="y")
    for b, n in zip(bars, names):
        ax2.annotate(f"U={metrics[n]['TheilU']:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                     ha="center", va="bottom", fontsize=7)

    # [1,1] Hit rate vs coin-flip baseline
    ax3 = fig.add_subplot(gs[1, 1])
    hits = [metrics[n]["HitRate"] * 100 for n in names]
    ax3.bar(names, hits, color=[colors.get(n) for n in names], alpha=0.85)
    ax3.axhline(50, color="black", lw=1, ls="--", label="Coin flip (50%)")
    ax3.set_title("Directional Hit Rate", fontsize=10)
    ax3.set_ylabel("Hit Rate %")
    ax3.tick_params(axis="x", labelrotation=20)
    ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3, axis="y")

    fig.suptitle(f"{ticker} — Walk-Forward Forecast Validation "
                 f"(expanding window, no lookahead)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("WALK-FORWARD / EXPANDING-WINDOW FORECAST VALIDATION")
    print("No lookahead — every forecast uses only data strictly before it")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []
    horizon_summary = []

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        dates = df.index[-len(returns):]
        if len(returns) < MIN_TRAIN + 30:
            print(f"\n  {ticker}: skipped — need >= {MIN_TRAIN + 30} return obs")
            continue

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}   ({len(returns)} obs)")
        print(f"{'─'*55}")

        actual, fc = walk_forward_forecasts(returns)
        metrics = compute_metrics(actual, fc)

        for name, m in metrics.items():
            flag = "  << beats naive" if name != "Naive" and m["TheilU"] < 1.0 else ""
            print(f"  {name:<9}: RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}  "
                  f"Hit={m['HitRate']*100:5.1f}%  TheilU={m['TheilU']:.3f}{flag}")

        best = min(metrics, key=lambda n: metrics[n]["RMSE"])
        summary.append({
            "Ticker": ticker,
            "BestModel": best,
            "BestRMSE": f"{metrics[best]['RMSE']:.5f}",
            "BestTheilU": f"{metrics[best]['TheilU']:.3f}",
            "BestHit%": f"{metrics[best]['HitRate']*100:.1f}",
            "BeatsNaive": "Y" if metrics[best]["TheilU"] < 1.0 and best != "Naive" else "N",
        })

        plot_validation(ticker, dates, actual, fc, metrics)

        # Multi-horizon: does the SAME model family still add value at
        # 1w/1m, the horizons Script 17 and options scripts actually use,
        # or does it only look good 1-day-ahead?
        h_actual, h_fc = walk_forward_forecasts_multi(returns)
        h_metrics = compute_metrics_multi(h_actual, h_fc)
        print("  Multi-horizon (cumulative return forecast):")
        row = {"Ticker": ticker}
        for h_label in HORIZONS:
            hm = h_metrics[h_label]
            valid_models = {n: v for n, v in hm.items() if np.isfinite(v["RMSE"])}
            if not valid_models:
                print(f"    {h_label:<3}: insufficient data")
                continue
            h_best = min(valid_models, key=lambda n: valid_models[n]["RMSE"])
            print(f"    {h_label:<3}: best={h_best:<9} TheilU={hm[h_best]['TheilU']:.3f}  "
                  f"Hit={hm[h_best]['HitRate']*100:5.1f}%")
            row[f"{h_label}_Best"] = h_best
            row[f"{h_label}_TheilU"] = f"{hm[h_best]['TheilU']:.3f}"
        horizon_summary.append(row)

    if summary:
        print("\n" + "=" * 65)
        print("VALIDATION SUMMARY — BEST MODEL PER ASSET (OUT-OF-SAMPLE)")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))
        n_beat = sum(1 for s in summary if s["BeatsNaive"] == "Y")
        print(f"\n  {n_beat}/{len(summary)} assets: best fitted model beat the "
              f"naive no-change forecast out-of-sample.")

    if horizon_summary:
        print("\n" + "=" * 65)
        print("MULTI-HORIZON SUMMARY — BEST MODEL PER ASSET, PER HORIZON")
        print("=" * 65)
        print(pd.DataFrame(horizon_summary).to_string(index=False))
        for h_label in HORIZONS:
            col = f"{h_label}_TheilU"
            beats = sum(1 for r in horizon_summary
                        if col in r and float(r[col]) < 1.0 and r.get(f"{h_label}_Best") != "Naive")
            n = sum(1 for r in horizon_summary if col in r)
            if n:
                print(f"  {h_label}: {beats}/{n} assets beat naive at this horizon")

    print("\nWalk-forward validation complete.")


if __name__ == "__main__":
    main()
