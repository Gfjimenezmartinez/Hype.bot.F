"""
================================================================
Script 8 — ARMA / ARIMA Modeling for Stock Returns
================================================================
Adapted from: arma_models.py  (full enhanced version)
Data source : Yahoo Finance (yfinance) via data_loader.py

Methods:
  • ADF / KPSS stationarity
  • Grid-search ARIMA(p, d, q) by AIC
  • Residual diagnostics: Ljung-Box, Jarque-Bera, Shapiro-Wilk,
    ARCH LM test
  • Out-of-sample forecast vs actual
  • GARCH(1,1) on residuals (if ARCH effects detected)

Single best asset at a time; loops over all symbols.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera, shapiro
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

from data_loader import load_all_assets, SYMBOLS, LOOKBACK_DAYS

plt.style.use("seaborn-v0_8-darkgrid")

MAX_AR   = 2
MAX_MA   = 2
MAX_D    = 1
TEST_PCT = 0.15          # fraction for out-of-sample test
HORIZON  = 5             # forecast steps


# ============================================================
# Stationarity
# ============================================================
def find_d(series: pd.Series, max_d=2) -> int:
    cur = series.copy()
    for d in range(max_d + 1):
        if adfuller(cur.dropna())[1] < 0.05:
            return d
        cur = cur.diff().dropna()
    return max_d


# ============================================================
# ARIMA grid search
# ============================================================
def grid_search_arima(series: pd.Series, max_p, d, max_q):
    best_aic, best_order, best_model = np.inf, None, None
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue
            try:
                res = ARIMA(series.values, order=(p, d, q)).fit()
                rows.append({"order": (p, d, q), "aic": res.aic, "bic": res.bic})
                if res.aic < best_aic:
                    best_aic, best_order, best_model = res.aic, (p, d, q), res
            except Exception:
                continue
    df = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True) if rows else pd.DataFrame()
    return best_order, best_model, df


# ============================================================
# Diagnostics
# ============================================================
def run_diagnostics(residuals: pd.Series) -> dict:
    res = {}
    lb = acorr_ljungbox(residuals, lags=[10], return_df=True)
    res["LB_pval"]  = float(lb["lb_pvalue"].iloc[0])
    jb_s, jb_p     = jarque_bera(residuals)
    res["JB_pval"]  = float(jb_p)
    if len(residuals) < 5000:
        sw_s, sw_p   = shapiro(residuals)
        res["SW_pval"] = float(sw_p)
    arch_s, arch_p  = het_arch(residuals)[:2]
    res["ARCH_pval"] = float(arch_p)
    return res


# ============================================================
# Residual plot
# ============================================================
def plot_residuals(resid, ticker, order):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(np.asarray(resid), linewidth=0.8, color="steelblue")
    axes[0, 0].axhline(0, linestyle="--", color="red", alpha=0.6)
    axes[0, 0].set_title(f"{ticker} ARIMA{order} Residuals")

    from scipy import stats
    axes[0, 1].hist(resid, bins=50, density=True, alpha=0.7, color="steelblue")
    x = np.linspace(resid.min(), resid.max(), 200)
    axes[0, 1].plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), "r-", linewidth=2)
    axes[0, 1].set_title("Residual Distribution")

    stats.probplot(resid, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title("QQ Plot")

    plot_acf(resid, lags=min(30, len(resid)//4), ax=axes[1, 0])
    axes[1, 0].set_title("ACF Residuals")
    plot_pacf(resid, lags=min(30, len(resid)//4), ax=axes[1, 1])
    axes[1, 1].set_title("PACF Residuals")

    axes[1, 2].plot(np.cumsum(resid), linewidth=1.5, color="darkorange")
    axes[1, 2].set_title("Cumulative Residuals")

    plt.suptitle(f"{ticker} — Residual Diagnostics ARIMA{order}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Forecast plot
# ============================================================
def plot_forecast(ticker, train, test, forecast, ci, order):
    fig, ax = plt.subplots(figsize=(12, 5))
    show_train = train.iloc[-60:]
    ax.plot(show_train.index, show_train.values, color="gray", linewidth=1.5, label="Train")
    n = min(len(test), len(forecast))
    ax.plot(test.index[:n], test.values[:n],   "o-", color="steelblue", linewidth=1.5, label="Actual")
    ax.plot(test.index[:n], forecast[:n],      "s--", color="tomato", linewidth=1.5, label="Forecast")
    if ci is not None and len(ci) >= n:
        ci_arr = np.asarray(ci)
        ax.fill_between(test.index[:n], ci_arr[:n, 0], ci_arr[:n, 1], alpha=0.25, color="tomato")
    ax.set_title(f"{ticker} — ARIMA{order} Out-of-Sample Forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("ARMA / ARIMA MODELING FOR STOCK RETURNS")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary_rows = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}")
        print(f"{'─'*50}")

        returns = df["log_return"].dropna()
        split   = int(len(returns) * (1 - TEST_PCT))
        train   = returns.iloc[:split]
        test    = returns.iloc[split:]

        # Stationarity
        d_opt = find_d(train)
        print(f"  Optimal d: {d_opt}")

        # Grid search
        order, model, grid_df = grid_search_arima(train, MAX_AR, d_opt, MAX_MA)
        if model is None:
            print("  No model converged — skipping.")
            continue
        print(f"  Best ARIMA{order}  AIC={model.aic:.2f}  BIC={model.bic:.2f}")
        if not grid_df.empty:
            print("  Top 5 by AIC:")
            print(grid_df.head(5)[["order", "aic", "bic"]].to_string(index=False))

        # Diagnostics
        diag = run_diagnostics(model.resid)
        print(f"  Ljung-Box p={diag['LB_pval']:.4f}  "
              f"JB p={diag['JB_pval']:.4f}  "
              f"ARCH p={diag['ARCH_pval']:.4f}")
        plot_residuals(model.resid, ticker, order)

        # GARCH if ARCH effects
        if ARCH_AVAILABLE and diag["ARCH_pval"] < 0.05:
            print("  ARCH effects detected — fitting GARCH(1,1) on residuals…")
            try:
                gres = arch_model(model.resid * 100, vol="Garch", p=1, q=1).fit(disp="off")
                print(f"  GARCH: ω={gres.params['omega']:.4f}  "
                      f"α={gres.params['alpha[1]']:.4f}  β={gres.params['beta[1]']:.4f}")
            except Exception as e:
                print(f"  GARCH failed: {e}")

        # Out-of-sample forecast
        h      = min(HORIZON, len(test))
        fc_obj = model.get_forecast(steps=h)
        fc     = np.asarray(fc_obj.predicted_mean)
        ci     = fc_obj.conf_int()

        # Accuracy metrics
        act = test.values[:h]
        if len(act) == h:
            rmse  = float(np.sqrt(np.mean((act - fc)**2)))
            da    = float(np.mean(np.sign(act) == np.sign(fc)) * 100)
            print(f"  RMSE: {rmse:.6f}   Directional Accuracy: {da:.1f}%")
            summary_rows.append({"Ticker": ticker, "Order": str(order),
                                  "AIC": round(model.aic, 2),
                                  "RMSE": round(rmse, 6), "DA%": round(da, 1)})
        plot_forecast(ticker, train, test, fc, ci, order)

    if summary_rows:
        print("\n" + "=" * 65)
        print("ARMA SUMMARY TABLE")
        print("=" * 65)
        print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nARMA / ARIMA analysis complete.")


if __name__ == "__main__":
    main()
