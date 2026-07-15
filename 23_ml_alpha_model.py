"""
================================================================
Script 23 — ML Alpha Model (Ridge / Lasso + Bias-Variance Validation)
================================================================
Brings the regularized-regression / model-validation toolkit from your
ML coursework (linear regression baseline, Ridge/Lasso, train-vs-test
bias-variance analysis) onto the trading bot's actual data, to answer
a question no other script in the suite asks: does this signal
generalize out-of-sample, or is it just fit to noise?

  Linear regression baseline : sklearn LinearRegression (unregularized)
  Ridge (L2) / Lasso (L1)    : regularized regression, coefficient
                                shrinkage and (for Lasso) sparsity
  Bias-variance validation   : train MSE vs test MSE swept across
                                regularization strength lambda
  Model selection            : TimeSeriesSplit cross-validation
                                (NOT a random k-fold — financial data
                                is autocorrelated, a random split leaks
                                future information into training folds)

Two deliberate departures from a textbook homework treatment of this
material:
  1. The complexity axis is regularization strength (lambda), not
     polynomial degree. Polynomial basis expansion of already-noisy
     daily return features with ~250-500 obs/asset overfits long
     before degree 20 -- the lesson (train error always improves with
     complexity, test error is U-shaped) carries over; the axis that's
     safe to sweep on this data does not.
  2. The train/test split is chronological, not random. Shuffling a
     time series before splitting leaks future values into "training"
     via autocorrelation -- silently invalidating the whole exercise.

Feature engineering (add_features/_rsi below) is causal (shift/rolling-
only) -- RSI, 20d vol, volume ratio, lagged returns/volume, rolling
return-percentile. Originally lived in the now-retired Script 11
(_deprecated/11_full_forecast_system.py); moved here since this is the
only script that ever used it, and Scripts 24/25/26 reuse it
transitively through build_dataset() below rather than importing it
directly.

This is a standalone validation script for now -- it reports whether a
linear signal survives out-of-sample and which features matter, but
its predictions are not yet wired into Scripts 17/20/21's forecast
inputs. That's a natural follow-up once these results are reviewed.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

PLOT_STYLE   = "seaborn-v0_8-darkgrid"
TRAIN_FRAC   = 0.70
N_CV_SPLITS  = 5
LASSO_KW     = {"max_iter": 20_000}

# Ridge and Lasso do NOT share an effective regularization scale in sklearn:
# Ridge minimizes ||y-Xw||^2 + alpha*||w||^2 (unnormalized by n), while Lasso
# minimizes (1/2n)*||y-Xw||^2 + alpha*||w||_1 (normalized). With ~300 training
# rows and ~14 standardized features, Ridge needs alpha in the hundreds-to-
# thousands range to visibly shrink anything, while Lasso is already fully
# sparse (zeroing every coefficient) above alpha~0.003. A single shared grid
# leaves one of the two flat across its entire tested range -- confirmed
# empirically before picking these two grids, each spanning its own model's
# underfit -> optimal -> overfit transition.
RIDGE_LAMBDA_GRID = [1e5, 3e4, 1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0, 1.0]
LASSO_LAMBDA_GRID = [10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4, 3e-5, 1e-5]

# Squeeze/funding features -- same construction Script 27's crypto-perps
# screener uses for its "explosive setup" score: a compressed Bollinger
# Band width (squeeze) and a crowded/extreme funding rate both tend to
# precede a violent break in either direction. bb_pctile is always
# computable from OHLCV already in df; funding_rate/funding_extreme only
# appear when the caller merged funding via data_loader.attach_funding()
# (most callers don't -- this degrades gracefully to skipping the
# feature, not an error).
SQUEEZE_VOL_WINDOW = 20
SQUEEZE_WINDOW     = 60
FUNDING_EXTREME_WINDOW = 60


# ============================================================
# Feature Engineering (causal -- shift/rolling-only, no lookahead)
# ============================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    r = df["log_return"]
    df["rsi"] = _rsi(df["close"])
    df["vol_20"] = r.rolling(20).std()
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    for lag in [1, 2, 3, 5, 10]:
        df[f"r_lag{lag}"]   = r.shift(lag)
        df[f"vol_lag{lag}"] = df["volume"].shift(lag)
    df["roll_q"] = r.rolling(50).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        if len(x.dropna()) > 1 else np.nan, raw=False
    )

    # Squeeze -- compressed Bollinger-Band width often precedes an
    # explosive break in either direction (same construction as Script
    # 27's "explosive setup" score).
    mid   = df["close"].rolling(SQUEEZE_VOL_WINDOW).mean()
    std20 = df["close"].rolling(SQUEEZE_VOL_WINDOW).std()
    bb_width = (4 * std20 / mid).replace([np.inf, -np.inf], np.nan)
    df["bb_pctile"] = bb_width.rolling(SQUEEZE_WINDOW).rank(pct=True)

    # Funding rate -- only present if the caller merged it via
    # data_loader.attach_funding(); an extreme (very positive or very
    # negative) funding rate means crowded, over-leveraged positioning on
    # one side, the exact precondition for a liquidation-cascade squeeze.
    if "funding_rate" in df.columns:
        df["funding_extreme"] = df["funding_rate"].abs().rolling(FUNDING_EXTREME_WINDOW).rank(pct=True)

    return df.dropna()


def _rsi(price: pd.Series, period=14) -> pd.Series:
    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================
# Dataset construction
# ============================================================
def build_dataset(df):
    """
    Features at time t use only data through t (add_features above is
    shift/rolling-only). Target is next-day log-return -- the least
    overlap-autocorrelated horizon, and the standard 'is there a linear
    signal at all' baseline.
    """
    feat = add_features(df)
    feat_cols = [c for c in feat.columns
                 if "lag" in c or c in ("vol_20", "rsi", "vol_ratio", "roll_q",
                                         "bb_pctile", "funding_rate", "funding_extreme")]
    X = feat[feat_cols].copy()
    y = feat["log_return"].shift(-1).rename("target")
    data = pd.concat([X, y], axis=1).dropna()
    return data[feat_cols], data["target"], feat_cols


def chronological_split(X, y, train_frac=TRAIN_FRAC):
    n_tr = int(len(X) * train_frac)
    return X.iloc[:n_tr], X.iloc[n_tr:], y.iloc[:n_tr], y.iloc[n_tr:]


# ============================================================
# Bias-Variance Sweep (train MSE vs test MSE across lambda)
# ============================================================
def bias_variance_sweep(X_tr, y_tr, X_te, y_te, ridge_lambdas, lasso_lambdas):
    """
    Ridge and Lasso each get their own lambda grid (see RIDGE_LAMBDA_GRID /
    LASSO_LAMBDA_GRID comment -- they are not on the same effective scale).
    Returns two per-model tables plus the Lasso coefficient path (the one
    that's actually interpretable, since Lasso drives coefficients to
    exactly zero).
    """
    ridge_rows = []
    for lam in ridge_lambdas:
        ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=lam))])
        ridge.fit(X_tr, y_tr)
        ridge_rows.append({
            "lambda": lam,
            "train_mse": mean_squared_error(y_tr, ridge.predict(X_tr)),
            "test_mse":  mean_squared_error(y_te, ridge.predict(X_te)),
        })

    lasso_rows = []
    coef_path = []
    for lam in lasso_lambdas:
        lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=lam, **LASSO_KW))])
        lasso.fit(X_tr, y_tr)
        coef = lasso.named_steps["model"].coef_
        lasso_rows.append({
            "lambda": lam,
            "train_mse": mean_squared_error(y_tr, lasso.predict(X_tr)),
            "test_mse":  mean_squared_error(y_te, lasso.predict(X_te)),
            "nonzero":   int(np.sum(np.abs(coef) > 1e-8)),
        })
        coef_path.append(coef.copy())

    ridge_bv = pd.DataFrame(ridge_rows)
    lasso_bv = pd.DataFrame(lasso_rows)
    coef_path = np.array(coef_path)   # (n_lambda, n_features)
    return ridge_bv, lasso_bv, coef_path


# ============================================================
# TimeSeriesSplit CV lambda selection (model-selection step)
# ============================================================
def cv_select_lambda(X_tr, y_tr, lambdas, model_cls, n_splits=N_CV_SPLITS, **model_kwargs):
    """
    Cross-validated lambda selection using TimeSeriesSplit -- each fold's
    validation block comes strictly after its training block, unlike a
    random k-fold which would let future data leak into training via
    autocorrelation.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_mse = []
    for lam in lambdas:
        fold_mses = []
        for tr_idx, val_idx in tscv.split(X_tr):
            pipe = Pipeline([("scaler", StandardScaler()),
                              ("model", model_cls(alpha=lam, **model_kwargs))])
            pipe.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            pred = pipe.predict(X_tr.iloc[val_idx])
            fold_mses.append(mean_squared_error(y_tr.iloc[val_idx], pred))
        cv_mse.append(float(np.mean(fold_mses)))
    cv_mse = np.array(cv_mse)
    best_lambda = lambdas[int(np.argmin(cv_mse))]
    return best_lambda, cv_mse


# ============================================================
# Per-Asset Analysis
# ============================================================
def analyse_asset(df):
    X, y, feat_cols = build_dataset(df)
    if len(X) < 120:
        raise ValueError(f"only {len(X)} usable rows after feature engineering (need >=120)")

    X_tr, X_te, y_tr, y_te = chronological_split(X, y)

    # OLS baseline (unregularized)
    ols = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    ols.fit(X_tr, y_tr)
    ols_result = {
        "train_mse": mean_squared_error(y_tr, ols.predict(X_tr)),
        "test_mse":  mean_squared_error(y_te, ols.predict(X_te)),
        "test_r2":   r2_score(y_te, ols.predict(X_te)),
    }

    # Bias-variance sweep on the single chronological split (each model on
    # its own lambda scale -- see RIDGE_LAMBDA_GRID / LASSO_LAMBDA_GRID)
    ridge_bv, lasso_bv, coef_path = bias_variance_sweep(
        X_tr, y_tr, X_te, y_te, RIDGE_LAMBDA_GRID, LASSO_LAMBDA_GRID)

    # Sanity check: less regularization should never fit training data
    # worse (train MSE should be ~monotonically non-increasing as lambda
    # decreases). A meaningful violation means the split/model wiring is
    # backwards. RIDGE_LAMBDA_GRID is already sorted descending.
    monotonic_ok = bool(np.all(np.diff(ridge_bv["train_mse"].values) <= 1e-6))

    # TimeSeriesSplit CV lambda selection (on train split only)
    ridge_lambda, ridge_cv_mse = cv_select_lambda(X_tr, y_tr, RIDGE_LAMBDA_GRID, Ridge)
    lasso_lambda, lasso_cv_mse = cv_select_lambda(X_tr, y_tr, LASSO_LAMBDA_GRID, Lasso, **LASSO_KW)

    ridge_best = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=ridge_lambda))]).fit(X_tr, y_tr)
    lasso_best = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=lasso_lambda, **LASSO_KW))]).fit(X_tr, y_tr)

    lasso_coef = lasso_best.named_steps["model"].coef_
    ridge_result = {
        "lambda_": ridge_lambda, "cv_mse": ridge_cv_mse,
        "test_mse": mean_squared_error(y_te, ridge_best.predict(X_te)),
        "model": ridge_best,
    }
    lasso_result = {
        "lambda_": lasso_lambda, "cv_mse": lasso_cv_mse,
        "test_mse": mean_squared_error(y_te, lasso_best.predict(X_te)),
        "test_r2":  r2_score(y_te, lasso_best.predict(X_te)),
        "nonzero":  int(np.sum(np.abs(lasso_coef) > 1e-8)),
        "coef":     lasso_coef,
        "model":    lasso_best,
    }

    return {
        "feat_cols": feat_cols, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "ols": ols_result, "ridge_bv": ridge_bv, "lasso_bv": lasso_bv, "coef_path": coef_path,
        "ridge": ridge_result, "lasso": lasso_result, "monotonic_ok": monotonic_ok,
    }


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, r):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(17, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)

    ridge_bv = r["ridge_bv"]
    lasso_bv = r["lasso_bv"]
    feat_cols = r["feat_cols"]

    # [0,0] Bias-variance validation curve. Ridge and Lasso are NOT on the
    # same lambda scale (see RIDGE_LAMBDA_GRID / LASSO_LAMBDA_GRID comment),
    # so each gets its own x-axis (twiny) sharing the MSE y-axis.
    ax0 = fig.add_subplot(gs[0, 0])
    ax0b = ax0.twiny()
    ax0.plot(ridge_bv["lambda"], ridge_bv["train_mse"], "o-", color="royalblue", lw=1.3, label="Ridge train")
    ax0.plot(ridge_bv["lambda"], ridge_bv["test_mse"],  "o--", color="royalblue", lw=1.3, alpha=0.6, label="Ridge test")
    ax0b.plot(lasso_bv["lambda"], lasso_bv["train_mse"], "s-", color="darkorange", lw=1.3, label="Lasso train")
    ax0b.plot(lasso_bv["lambda"], lasso_bv["test_mse"],  "s--", color="darkorange", lw=1.3, alpha=0.6, label="Lasso test")
    ax0.set_xscale("log"); ax0.invert_xaxis()
    ax0b.set_xscale("log"); ax0b.invert_xaxis()
    ax0.set_xlabel("Ridge lambda", color="royalblue", fontsize=8)
    ax0b.set_xlabel("Lasso lambda", color="darkorange", fontsize=8)
    ax0.set_ylabel("MSE")
    h0, l0 = ax0.get_legend_handles_labels()
    h0b, l0b = ax0b.get_legend_handles_labels()
    ax0.legend(h0 + h0b, l0 + l0b, fontsize=6)
    ax0.set_title(f"{ticker} — Bias-Variance Validation Curve", fontsize=9)
    ax0.grid(alpha=0.3)

    # [0,1] Lasso coefficient path
    ax1 = fig.add_subplot(gs[0, 1])
    lam_arr = np.array(LASSO_LAMBDA_GRID)
    paths = r["coef_path"]  # (n_lambda, n_features)
    for j, name in enumerate(feat_cols):
        ax1.plot(lam_arr, paths[:, j], lw=1.0, label=name if j < 8 else None)
    ax1.set_xscale("log")
    ax1.invert_xaxis()
    ax1.axhline(0, color="gray", lw=0.5, ls=":")
    ax1.axvline(r["lasso"]["lambda_"], color="black", lw=1.0, ls="--", alpha=0.6)
    ax1.set_xlabel("Lasso lambda (log scale)")
    ax1.set_ylabel("coefficient")
    ax1.legend(fontsize=5, ncol=2)
    ax1.set_title("Lasso Coefficient Path (dashed = CV-selected lambda)", fontsize=9)
    ax1.grid(alpha=0.3)

    # [0,2] TimeSeriesSplit CV MSE vs lambda (each model, its own x-axis)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2b = ax2.twiny()
    ax2.plot(RIDGE_LAMBDA_GRID, r["ridge"]["cv_mse"], "o-", color="royalblue", label="Ridge CV MSE")
    ax2b.plot(LASSO_LAMBDA_GRID, r["lasso"]["cv_mse"], "s-", color="darkorange", label="Lasso CV MSE")
    ax2.axvline(r["ridge"]["lambda_"], color="royalblue", lw=1.0, ls="--", alpha=0.5)
    ax2b.axvline(r["lasso"]["lambda_"], color="darkorange", lw=1.0, ls="--", alpha=0.5)
    ax2.set_xscale("log"); ax2.invert_xaxis()
    ax2b.set_xscale("log"); ax2b.invert_xaxis()
    ax2.set_xlabel("Ridge lambda", color="royalblue", fontsize=8)
    ax2b.set_xlabel("Lasso lambda", color="darkorange", fontsize=8)
    ax2.set_ylabel("CV MSE")
    h2, l2 = ax2.get_legend_handles_labels()
    h2b, l2b = ax2b.get_legend_handles_labels()
    ax2.legend(h2 + h2b, l2 + l2b, fontsize=7)
    ax2.set_title(f"TimeSeriesSplit CV  |  selected: Ridge={r['ridge']['lambda_']:.3g}, "
                  f"Lasso={r['lasso']['lambda_']:.3g}", fontsize=8.5)
    ax2.grid(alpha=0.3)

    # [1,0] Predicted vs actual (test set, Lasso best)
    ax3 = fig.add_subplot(gs[1, 0])
    y_te = r["y_te"].values
    pred = r["lasso"]["model"].predict(r["X_te"])
    ax3.scatter(y_te, pred, s=14, alpha=0.5, color="darkorange")
    lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
    ax3.plot(lims, lims, "k--", lw=0.8, alpha=0.5)
    ax3.set_xlabel("Actual next-day return")
    ax3.set_ylabel("Predicted")
    ax3.set_title(f"Predicted vs Actual (test)  |  R^2={r['lasso']['test_r2']:.4f}", fontsize=9)
    ax3.grid(alpha=0.3)

    # [1,1] Actual vs predicted over the test period
    ax4 = fig.add_subplot(gs[1, 1])
    idx = r["y_te"].index
    ax4.plot(idx, y_te, color="gray", lw=1.0, alpha=0.6, label="Actual")
    ax4.plot(idx, pred, color="darkorange", lw=1.2, label="Lasso predicted")
    ax4.axhline(0, color="black", lw=0.4)
    ax4.legend(fontsize=7)
    ax4.set_title("Test-Period Forecast vs Actual", fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.tick_params(axis="x", rotation=30, labelsize=6)

    # [1,2] Feature importance (Lasso coefficients at selected lambda)
    ax5 = fig.add_subplot(gs[1, 2])
    coef = r["lasso"]["coef"]
    order = np.argsort(np.abs(coef))[::-1]
    names_sorted = [feat_cols[i] for i in order]
    vals_sorted = coef[order]
    colors = ["forestgreen" if v > 0 else "tomato" for v in vals_sorted]
    ax5.barh(range(len(vals_sorted)), vals_sorted, color=colors, alpha=0.85)
    ax5.set_yticks(range(len(vals_sorted)))
    ax5.set_yticklabels(names_sorted, fontsize=7)
    ax5.invert_yaxis()
    ax5.axvline(0, color="gray", lw=0.5)
    ax5.set_title(f"Feature Importance (Lasso, lambda={r['lasso']['lambda_']:.3g}, "
                  f"{r['lasso']['nonzero']}/{len(feat_cols)} nonzero)", fontsize=8.5)
    ax5.grid(axis="x", alpha=0.3)

    fig.suptitle(f"{ticker} — ML Alpha Model (Ridge/Lasso, Bias-Variance Validation)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("ML ALPHA MODEL")
    print("Ridge/Lasso + Bias-Variance Validation (TimeSeriesSplit CV)")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*50}")

        try:
            r = analyse_asset(df)
            ols, ridge, lasso = r["ols"], r["ridge"], r["lasso"]

            print(f"  Train/test rows: {len(r['X_tr'])}/{len(r['X_te'])}  "
                  f"features={len(r['feat_cols'])}")
            print(f"  Bias-variance sanity check (train MSE monotonic): "
                  f"{'OK' if r['monotonic_ok'] else 'VIOLATED -- check wiring'}")
            print(f"  OLS   : train_mse={ols['train_mse']:.6f}  test_mse={ols['test_mse']:.6f}  "
                  f"test_r2={ols['test_r2']:+.4f}")
            print(f"  Ridge : lambda={ridge['lambda_']:.3g}  test_mse={ridge['test_mse']:.6f}")
            print(f"  Lasso : lambda={lasso['lambda_']:.3g}  test_mse={lasso['test_mse']:.6f}  "
                  f"test_r2={lasso['test_r2']:+.4f}  nonzero={lasso['nonzero']}/{len(r['feat_cols'])}")

            top_feats = [r["feat_cols"][i] for i in np.argsort(np.abs(lasso["coef"]))[::-1][:3]]
            print(f"  Top Lasso features: {top_feats}")

            summary.append({
                "Ticker": ticker,
                "OLS_TestMSE":   f"{ols['test_mse']:.6f}",
                "Ridge_Lambda":  f"{ridge['lambda_']:.3g}",
                "Ridge_TestMSE": f"{ridge['test_mse']:.6f}",
                "Lasso_Lambda":  f"{lasso['lambda_']:.3g}",
                "Lasso_TestMSE": f"{lasso['test_mse']:.6f}",
                "Lasso_R2":      f"{lasso['test_r2']:+.4f}",
                "NonZero": f"{lasso['nonzero']}/{len(r['feat_cols'])}",
            })

            plot_dashboard(ticker, r)

        except Exception as e:
            print(f"  Error: {e}")

    if summary:
        print("\n" + "=" * 65)
        print("ML ALPHA MODEL SUMMARY")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nML alpha model analysis complete.")


if __name__ == "__main__":
    main()
