"""
================================================================
Script 25 — ML Forecast Signal (Confidence-Gated Logistic Regression)
================================================================
Every ML script so far (23: Ridge/Lasso regression, 24: Perceptron/
SVM/MLP/KNN/RF/LogReg classification) has been a standalone diagnostic
-- validating whether a signal generalizes out-of-sample, never
actually feeding a trade decision. This script closes that loop with
the one result that actually looked like signal rather than noise:
Script 24's confidence-thresholded Logistic Regression (e.g. MSTR:
70.6% accuracy at 13% coverage vs. a 56.3% baseline). Deliberately not
a bigger ensemble -- just the one validated, simple model.

get_ml_signal(df) is the low-level API: fits/predicts on whatever bars
you hand it. get_best_ml_signal(symbol, ticker_df) is the recommended
entry point -- it resolves the ticker's empirically-best bar interval
first (see "Timeframe selection" below), since crypto assets vary
enough by category (majors vs. young/volatile alts) that no single
interval is best for all of them. Consumed by Script 17's trade
planner as a bounded conviction modulator (same pattern already used
there for Script 21's LQR output) -- not a replacement for the
existing regime/trend/quantile logic, and not a hard veto.

Reuses Script 24's build_classification_dataset/fit_logistic/
confidence_threshold_curve and Script 23's chronological_split rather
than re-deriving any of it.
================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib import import_module as _im
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit

from data_loader import (load_all_assets, LOOKBACK_DAYS, format_price,
                          fetch_intraday, fetch_funding_history, attach_funding, CACHE_DIR)

_r23 = _im("23_ml_alpha_model")
_r24 = _im("24_classification_signals")
chronological_split          = _r23.chronological_split
build_classification_dataset = _r24.build_classification_dataset
fit_logistic                 = _r24.fit_logistic
confidence_threshold_curve   = _r24.confidence_threshold_curve

PLOT_STYLE = "seaborn-v0_8-darkgrid"
DEFAULT_CONF_THRESHOLD = 0.60   # fallback only -- used when calibration can't run
                                 # (too few rows); live signals calibrate per ticker
CONF_THRESHOLD_GRID = [0.55, 0.60, 0.65, 0.70, 0.75]
MIN_CV_COVERAGE = 0.15   # crypto tickers vary widely in how often the model is
                          # confident (BTC/PENGU barely clear 0.50, SUI needs 0.80) --
                          # a threshold that only "wins" on a handful of CV samples is
                          # the same small-n trap the Wilson-CI check caught on the
                          # MSTR/ETH raw accuracy numbers (flashy on n=13-18, not
                          # meaningfully different from chance). Require a threshold to
                          # clear at least 15% of CV validation folds before its
                          # accuracy is trusted enough to select it.
MIN_ROWS = 120


# ============================================================
# Per-ticker confidence threshold calibration
# ============================================================
def calibrate_conf_threshold(X_tr, y_tr, thresholds=CONF_THRESHOLD_GRID,
                              n_splits=5, min_coverage=MIN_CV_COVERAGE,
                              default=DEFAULT_CONF_THRESHOLD):
    """
    Selects a per-ticker confidence threshold via TimeSeriesSplit CV on the
    TRAIN split only -- never touches the held-out test split get_ml_signal
    uses for its honest recent_test_acc number. Among thresholds that clear
    min_coverage of CV validation folds, picks the one with the best CV
    accuracy, so a threshold can't win by being lucky on a handful of samples.
    Falls back to `default` if no threshold clears the coverage bar (e.g. too
    little data for a stable estimate).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    correct = {t: 0 for t in thresholds}
    total = {t: 0 for t in thresholds}
    n_confident = {t: 0 for t in thresholds}
    n_val = {t: 0 for t in thresholds}

    for tr_idx, val_idx in tscv.split(X_tr):
        X_fold_tr, y_fold_tr = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_fold_val, y_fold_val = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
        clf = fit_logistic(X_fold_tr, y_fold_tr)
        up_idx = list(clf.classes_).index(1)
        proba = clf.predict_proba(X_fold_val)[:, up_idx]
        y_val = y_fold_val.values
        for t in thresholds:
            confident = (proba >= t) | (proba <= 1 - t)
            n_val[t] += len(y_val)
            n_confident[t] += int(confident.sum())
            if confident.any():
                pred = np.where(proba[confident] >= 0.5, 1, -1)
                correct[t] += int((pred == y_val[confident]).sum())
                total[t] += int(confident.sum())

    best_t, best_acc = default, -1.0
    for t in thresholds:
        coverage = n_confident[t] / max(n_val[t], 1)
        if coverage < min_coverage or total[t] == 0:
            continue
        acc = correct[t] / total[t]
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t


# ============================================================
# Public API — consumed by Script 17
# ============================================================
def get_ml_signal(df, conf_threshold=None):
    """
    LONG/SHORT/FLAT direction call for tomorrow, gated by confidence:
      - Fits on a chronological train split, reports HONEST recent
        out-of-sample accuracy at conf_threshold on the held-out split
        (recent_test_acc/recent_coverage) -- numbers the live model
        hasn't seen.
      - Refits on the FULL history for the actual live prediction --
        the holdout's job is to produce an honest validation number,
        not to be permanently withheld from the model making the call.

    conf_threshold=None (default) calibrates a per-ticker threshold via
    calibrate_conf_threshold() on the train split -- crypto tickers vary too
    widely in confidence behavior to share one fixed threshold. Pass an
    explicit value to override (e.g. for testing).
    """
    X, y, feat_cols = build_classification_dataset(df)
    if len(X) < MIN_ROWS:
        return {"signal": "FLAT", "proba_up": 0.5, "confidence": 0.0,
                "recent_test_acc": np.nan, "recent_coverage": 0.0, "top_features": [],
                "conf_threshold": DEFAULT_CONF_THRESHOLD}

    X_tr, X_te, y_tr, y_te = chronological_split(X, y)

    if conf_threshold is None:
        conf_threshold = calibrate_conf_threshold(X_tr, y_tr)

    clf_val = fit_logistic(X_tr, y_tr)
    coverage, accuracy = confidence_threshold_curve(clf_val, X_te, y_te, thresholds=[conf_threshold])
    recent_coverage = float(coverage[0])
    recent_test_acc = float(accuracy[0])   # NaN if nothing crossed the threshold

    clf_live = fit_logistic(X, y)
    # X.iloc[[-1]] is NOT today -- build_classification_dataset dropna's off
    # the most recent date (its target, tomorrow's return, doesn't exist yet),
    # so the last surviving row is yesterday, whose "next-day" outcome is
    # already known and was part of clf_live's own training fit. Recompute
    # today's causal features directly (add_features needs no target) so the
    # live call actually predicts an unrealized return, not a memorized one.
    latest_features = _r23.add_features(df)[feat_cols].iloc[[-1]]
    up_idx = list(clf_live.classes_).index(1)
    proba_up = float(clf_live.predict_proba(latest_features)[:, up_idx][0])

    if proba_up >= conf_threshold:
        signal = "LONG"
    elif proba_up <= 1 - conf_threshold:
        signal = "SHORT"
    else:
        signal = "FLAT"
    confidence = abs(proba_up - 0.5) * 2

    coef = clf_live.named_steps["logreg"].coef_.flatten()
    top_idx = np.argsort(np.abs(coef))[::-1][:3]
    top_features = [feat_cols[i] for i in top_idx]

    return {
        "signal": signal, "proba_up": proba_up, "confidence": confidence,
        "recent_test_acc": recent_test_acc, "recent_coverage": recent_coverage,
        "top_features": top_features, "conf_threshold": conf_threshold,
    }


# ============================================================
# Timeframe selection -- crypto assets on Hyperliquid vary wildly in how
# much history actually exists (a token listed 4 months ago has ~20 weekly
# bars and ~5 monthly bars, nowhere near MIN_ROWS=120). Rather than assume
# a bar interval, fetch each candidate and let get_ml_signal's own honest
# held-out accuracy (recent_test_acc/recent_coverage) decide which one is
# both viable AND actually predictive for this specific ticker.
# ============================================================
TIMEFRAME_CANDIDATES = {
    "1h": 5000,   # ~208 days -- Hyperliquid's practical ccxt fetch cap
    "1d": 2000,   # ~5.5 years (most tickers have far less; capped by listing date)
    "1w": 500,    # ~9.5 years
    # "1M" deliberately excluded: an empirical sweep across the live universe
    # (2026-07) showed every single ticker, including BTC/ETH, comes in under
    # MIN_ROWS=120 monthly bars -- Hyperliquid simply hasn't existed long
    # enough. Structurally guaranteed to fail right now, so testing it every
    # run would just be wasted fetches/fits. Revisit once the exchange has
    # ~10 years of history.
}
MIN_TIMEFRAME_COVERAGE = 0.10   # a timeframe whose confident predictions cover
                                 # under 10% of the held-out test window is too
                                 # thin a sample to trust its accuracy number


def evaluate_timeframe(symbol, interval, limit):
    """
    Fetches `symbol` at one candidate bar interval and returns its raw row
    count plus Script 25's own held-out (test-split) accuracy/coverage --
    the same honest numbers get_ml_signal always reports, just surfaced
    per-timeframe instead of for a single assumed interval.

    A NaN test_acc has two distinct causes that matter for interpreting
    the sweep, so both are reported explicitly rather than collapsed into
    one "insufficient" bucket:
      - insufficient_rows: fewer than MIN_ROWS feature rows -- the model
        was never fit at all.
      - zero_test_coverage: plenty of rows, model fit fine, but the
        calibrated confidence threshold happened to produce zero
        confident predictions in this particular held-out test window.
    """
    df = fetch_intraday(symbol, interval=interval, limit=limit)
    if df is None:
        return {"available": False, "rows": 0, "feature_rows": 0}
    X, y, _ = build_classification_dataset(df)
    if len(X) < MIN_ROWS:
        return {"available": True, "rows": len(df), "feature_rows": len(X),
                "test_acc": np.nan, "coverage": 0.0, "reason": "insufficient_rows"}
    sig = get_ml_signal(df)
    reason = None if sig["recent_test_acc"] == sig["recent_test_acc"] else "zero_test_coverage"
    return {
        "available": True, "rows": len(df), "feature_rows": len(X),
        "test_acc": sig["recent_test_acc"], "coverage": sig["recent_coverage"],
        "reason": reason,
    }


def select_best_timeframe(symbol, candidates=TIMEFRAME_CANDIDATES,
                           min_coverage=MIN_TIMEFRAME_COVERAGE):
    """
    Evaluates every candidate interval for `symbol` and returns
    (best_interval_or_None, {interval: evaluate_timeframe(...) result}).
    A candidate is only eligible to win if it cleared MIN_ROWS (so
    test_acc isn't NaN) AND its coverage clears min_coverage -- same
    small-sample guard calibrate_conf_threshold already applies to
    confidence thresholds, applied here to the timeframe choice itself.
    """
    results = {}
    for interval, limit in candidates.items():
        try:
            results[interval] = evaluate_timeframe(symbol, interval, limit)
        except Exception as e:
            results[interval] = {"available": False, "rows": 0, "error": str(e)}

    eligible = {
        tf: r for tf, r in results.items()
        if r.get("available") and r.get("coverage", 0) >= min_coverage
        and r.get("test_acc") == r.get("test_acc")   # excludes NaN
    }
    best = max(eligible, key=lambda tf: eligible[tf]["test_acc"]) if eligible else None
    return best, results


# ============================================================
# Cached timeframe selection -- an automated pipeline (run.py, scheduled
# runs) calls into this from multiple scripts (16's backtester, 17's trade
# planner) many times a day. Which bar interval is most predictive for a
# given ticker is a slow-moving property (driven by listing age / typical
# vol regime, not by today's price action), so the full 3-interval sweep
# only needs to re-run once a day, not on every call.
# ============================================================
TIMEFRAME_CACHE_PATH = os.path.join(CACHE_DIR, "ml_best_timeframe.json")
TIMEFRAME_CACHE_TTL_HOURS = 24.0


def _load_timeframe_cache() -> dict:
    if not os.path.exists(TIMEFRAME_CACHE_PATH):
        return {}
    try:
        with open(TIMEFRAME_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_timeframe_cache(cache: dict):
    try:
        with open(TIMEFRAME_CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass   # best-effort -- worst case, every call re-sweeps instead of caching


def get_cached_best_timeframe(symbol: str, ttl_hours: float = TIMEFRAME_CACHE_TTL_HOURS) -> str:
    """
    Cached wrapper around select_best_timeframe(). Falls back to "1d" if no
    candidate clears the eligibility bar (see select_best_timeframe) --
    daily bars are the suite's original, most-tested default.
    """
    cache = _load_timeframe_cache()
    entry = cache.get(symbol)
    now = pd.Timestamp.now().timestamp()
    if entry and (now - entry.get("ts", 0)) / 3600 < ttl_hours:
        return entry["interval"]

    best, _ = select_best_timeframe(symbol)
    interval = best or "1d"
    cache[symbol] = {"interval": interval, "ts": now}
    _save_timeframe_cache(cache)
    return interval


def get_best_ml_signal(symbol: str, ticker_df=None):
    """
    Recommended entry point for consumers that want an ML signal without
    assuming a bar interval: resolves `symbol`'s empirically-best timeframe
    (cached, see get_cached_best_timeframe), fetches those bars, and returns
    get_ml_signal()'s normal dict plus "interval_used". Falls back to
    `ticker_df` (whatever the caller already has loaded, e.g. daily bars)
    if intraday data can't be fetched or doesn't clear MIN_ROWS.
    """
    interval = get_cached_best_timeframe(symbol)
    limit = TIMEFRAME_CANDIDATES.get(interval, 2000)
    df = fetch_intraday(symbol, interval=interval, limit=limit)

    if df is None or len(df) < MIN_ROWS:
        if ticker_df is None:
            return None
        result = get_ml_signal(ticker_df)
        result["interval_used"] = "1d(fallback)"
        return result

    # Funding rate -- a bonus feature (crowded/extreme positioning is a
    # precondition for a squeeze), never required for a signal. Relies on
    # data_loader.load_all_funding() having pre-warmed the cache; a cold
    # per-ticker fetch here would add real latency to a live per-ticker
    # loop (Script 17), so this call should always be a cache hit in
    # normal use -- if it isn't, use_cache still makes this correct, just
    # slower for this one ticker.
    try:
        funding = fetch_funding_history(symbol)
        if funding is not None:
            df = attach_funding(df, funding)
    except Exception:
        pass

    result = get_ml_signal(df)
    result["interval_used"] = interval
    return result


def print_timeframe_sweep(symbols: dict):
    """Standalone report: for every symbol, which bar interval is actually
    viable on Hyperliquid's real history, and which one's held-out accuracy
    is best. Run directly (see __main__ guard) since this is a diagnostic
    tool, not part of the live get_ml_signal()/Script 17 path."""
    print("\n" + "=" * 78)
    print("TIMEFRAME SWEEP — 1h / 1d / 1w")
    print("=" * 78)
    rows = []
    for name, symbol in symbols.items():
        best, results = select_best_timeframe(symbol)
        cell = {"Ticker": name, "Best": best or "none"}
        for tf in TIMEFRAME_CANDIDATES:
            r = results[tf]
            if not r.get("available") or r.get("rows", 0) == 0:
                cell[tf] = "no data"
            elif r.get("reason") == "insufficient_rows":
                cell[tf] = f"{r['feature_rows']}<{MIN_ROWS}rows"
            elif r.get("reason") == "zero_test_coverage":
                cell[tf] = f"{r['feature_rows']}rows/0%cov"
            else:
                cell[tf] = f"{r['test_acc']:.2f}@{r['coverage']*100:.0f}%cov"
        rows.append(cell)
    print(pd.DataFrame(rows).to_string(index=False))


# ============================================================
# Validation detail (main()/plotting only -- not part of the API
# Script 17 uses, keeps get_ml_signal() cheap and single-purpose)
# ============================================================
def _validation_predictions(df, conf_threshold=DEFAULT_CONF_THRESHOLD):
    X, y, _ = build_classification_dataset(df)
    X_tr, X_te, y_tr, y_te = chronological_split(X, y)
    clf_val = fit_logistic(X_tr, y_tr)
    up_idx = list(clf_val.classes_).index(1)
    proba_up = clf_val.predict_proba(X_te)[:, up_idx]
    return X_te.index, proba_up, y_te.values


# ============================================================
# Plotting — one simple plot per ticker, not another dashboard
# ============================================================
def plot_signal(ticker, dates, proba_up, y_actual, conf_threshold=DEFAULT_CONF_THRESHOLD):
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, proba_up, color="royalblue", lw=1.3, label="P(up)")
    ax.axhline(conf_threshold, color="forestgreen", lw=0.8, ls="--", label=f"LONG >= {conf_threshold}")
    ax.axhline(1 - conf_threshold, color="tomato", lw=0.8, ls="--", label=f"SHORT <= {1-conf_threshold:.2f}")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    up_days = y_actual == 1
    ax.scatter(np.array(dates)[up_days], np.full(up_days.sum(), 1.03), marker="^",
               s=12, color="forestgreen", label="actual up")
    ax.scatter(np.array(dates)[~up_days], np.full((~up_days).sum(), -0.03), marker="v",
               s=12, color="tomato", label="actual down")
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("P(up)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{ticker} — ML Forecast Signal (recent validation window)", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("ML FORECAST SIGNAL")
    print("Confidence-Gated Logistic Regression (validated in Script 24)")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*50}")

        try:
            sig = get_ml_signal(df)
            print(f"  Signal: {sig['signal']}  P(up)={sig['proba_up']:.3f}  "
                  f"confidence={sig['confidence']:.0%}  "
                  f"conf_threshold={sig['conf_threshold']:.2f} (per-ticker calibrated)")
            print(f"  Recent validated accuracy: {sig['recent_test_acc']:.3f}  "
                  f"(coverage {sig['recent_coverage']*100:.0f}% of recent test days)")
            print(f"  Top features: {sig['top_features']}")

            summary.append({
                "Ticker": ticker, "Signal": sig["signal"],
                "P_up": f"{sig['proba_up']:.3f}", "Confidence": f"{sig['confidence']:.0%}",
                "ConfThresh": f"{sig['conf_threshold']:.2f}",
                "Recent_Acc": f"{sig['recent_test_acc']:.3f}",
                "Coverage": f"{sig['recent_coverage']*100:.0f}%",
            })

            dates, proba_up, y_actual = _validation_predictions(df)
            plot_signal(ticker, dates, proba_up, y_actual, conf_threshold=sig["conf_threshold"])

        except Exception as e:
            print(f"  Error: {e}")

    if summary:
        print("\n" + "=" * 65)
        print("ML FORECAST SIGNAL SUMMARY")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nML forecast signal complete.")


if __name__ == "__main__":
    main()
