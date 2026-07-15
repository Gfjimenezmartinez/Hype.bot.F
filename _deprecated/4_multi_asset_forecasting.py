"""
================================================================
Script 4 — Multi-Asset Time-Series Forecasting
================================================================
Adapted from: 4_multi_asset_time_series_forecast.py  (full version)
Data source : Yahoo Finance (yfinance) via data_loader.py

Methods:
  • Local Linear Trend Kalman Filter (price smoothing + trend)
  • Adaptive Kalman Filter
  • ARIMA(1,0,1) for mean return forecasting
  • GARCH(1,1) for volatility forecasting
  • Combined Kalman + ARIMA-GARCH ensemble
  • 1-day / 1-week / 1-month horizons
  • Multi-asset comparison dashboard

Assets: all symbols in SYMBOLS (data_loader.py)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: arch not installed — GARCH fallback to historical vol.")

from data_loader import load_all_assets, SYMBOLS, LOOKBACK_DAYS, calculate_asset_metrics

plt.style.use("seaborn-v0_8-darkgrid")

# ============================================================
# Configuration
# ============================================================
FORECAST_HORIZONS = {"1 Day": 1, "1 Week": 7, "1 Month": 30}
CONFIDENCE        = 0.95

# Per-asset Kalman tuning (equity vols are lower than crypto)
_DEFAULT_KP = {"q_price": 1e-5, "q_trend": 5e-8, "r_obs": 2e-3}
KALMAN_PARAMS = {
    "AAPL": {"q_price": 5e-6, "q_trend": 2e-8, "r_obs": 5e-4},
    "MSFT": {"q_price": 5e-6, "q_trend": 2e-8, "r_obs": 5e-4},
    "GOOGL":{"q_price": 5e-6, "q_trend": 2e-8, "r_obs": 5e-4},
    "MSTR": {"q_price": 5e-5, "q_trend": 2e-7, "r_obs": 5e-3},
    "BNO":  {"q_price": 2e-5, "q_trend": 5e-8, "r_obs": 1e-3},
}


# ============================================================
# Kalman Filter
# ============================================================
class LocalLinearTrend:
    """
    State: [log_price, trend]
    Observation: log_price
    """
    def __init__(self, q_price=1e-5, q_trend=5e-8, r_obs=2e-3):
        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.diag([q_price, q_trend])
        self.R = np.array([[r_obs]])

    def filter(self, y: np.ndarray):
        n = len(y)
        x = np.zeros((n, 2))
        P = np.zeros((n, 2, 2))
        x[0] = [y[0], 0.0]
        P[0] = np.eye(2)
        for t in range(1, n):
            xp = self.F @ x[t - 1]
            Pp = self.F @ P[t - 1] @ self.F.T + self.Q
            S  = self.H @ Pp @ self.H.T + self.R
            K  = Pp @ self.H.T @ np.linalg.inv(S)
            inn = y[t] - self.H @ xp
            x[t] = xp + (K @ inn).flatten()
            P[t] = (np.eye(2) - K @ self.H) @ Pp
        return x, P

    def forecast(self, x_last, P_last, steps, damp=0.90):
        """
        Damped-trend forecast: the trend component decays by `damp`
        each step. Undamped extrapolation compounds the most recent
        momentum estimate linearly for the whole horizon, which is
        what produced +40–55% 30-day 'targets' on hot names.
        damp=0.90 → cumulative trend multiplier over 30d ≈ 9x the
        daily trend instead of 30x.
        """
        x, P = x_last.copy(), P_last.copy()
        Fd = self.F.copy()
        fcast, fvar = [], []
        for _ in range(steps):
            Fd[0, 1] = Fd[0, 1]            # price picks up current trend
            x = Fd @ x
            x[1] *= damp                    # decay the trend state
            P = Fd @ P @ Fd.T + self.Q
            fcast.append(x[0])
            fvar.append(P[0, 0])
        return np.array(fcast), np.array(fvar)


def adaptive_kalman_filter(prices: np.ndarray, window=20) -> tuple[np.ndarray, np.ndarray]:
    kf = LocalLinearTrend()
    fp, tr = [], []
    x = np.array([prices[0], 0.0])
    P = np.eye(2)
    for i, price in enumerate(prices):
        if i > window:
            rec  = prices[i - window:i]
            rvol = np.std(np.diff(rec) / rec[:-1])
            kf.R = np.array([[max(1e-4, rvol**2)]])
        xp = kf.F @ x
        Pp = kf.F @ P @ kf.F.T + kf.Q
        S  = kf.H @ Pp @ kf.H.T + kf.R
        K  = Pp @ kf.H.T @ np.linalg.inv(S)
        inn = price - kf.H @ xp
        x  = xp + (K @ inn).flatten()
        P  = (np.eye(2) - K @ kf.H) @ Pp
        fp.append(x[0])
        tr.append(x[1])
    return np.array(fp), np.array(tr)


# ============================================================
# ARIMA + GARCH
# ============================================================
def fit_arima(returns: pd.Series):
    try:
        return ARIMA(returns.values, order=(1, 0, 1)).fit()
    except Exception:
        return ARIMA(returns.values, order=(1, 0, 0)).fit()


def fit_garch(returns: pd.Series):
    if not ARCH_AVAILABLE:
        return None
    try:
        res = arch_model(returns * 100, mean="Constant", vol="GARCH",
                         p=1, q=1, dist="normal").fit(disp="off")
        return res
    except Exception:
        return None


def arima_garch_forecast(last_price, arima_res, garch_res, horizon, conf=0.95):
    mu    = np.asarray(arima_res.forecast(horizon))
    if garch_res is not None:
        sigma = np.sqrt(
            garch_res.forecast(horizon=horizon).variance.iloc[-1].values
        ) / 100
    else:
        sigma = np.full(horizon, arima_res.resid.std())

    cum_mu  = np.cumsum(mu)
    cum_sig = np.sqrt(np.cumsum(sigma**2))
    z       = norm.ppf((1 + conf) / 2)
    median  = last_price * np.exp(cum_mu)
    lower   = last_price * np.exp(cum_mu - z * cum_sig)
    upper   = last_price * np.exp(cum_mu + z * cum_sig)
    return median, lower, upper


# ============================================================
# Combined Forecast
# ============================================================
def combined_forecast(df, kp_params, horizon=30, conf=0.95):
    returns    = df["log_return"].dropna()
    log_prices = np.log(df["close"].values)
    last_price = float(df["close"].iloc[-1])

    # Kalman
    kf = LocalLinearTrend(**kp_params)
    states, covs = kf.filter(log_prices)
    kf_fcast_log, kf_fvar = kf.forecast(states[-1], covs[-1], horizon)
    kf_median = np.exp(kf_fcast_log)
    kf_upper  = np.exp(kf_fcast_log + 2 * np.sqrt(kf_fvar))
    kf_lower  = np.exp(kf_fcast_log - 2 * np.sqrt(kf_fvar))

    # ARIMA+GARCH
    arima_res = fit_arima(returns)
    garch_res = fit_garch(returns)
    ag_median, ag_lower, ag_upper = arima_garch_forecast(
        last_price, arima_res, garch_res, horizon, conf
    )

    # Blend: high-vol → favour ARIMA-GARCH; low-vol → favour Kalman
    recent_vol = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
    kw = 0.35 if recent_vol > 0.015 else 0.65

    comb_med  = kw * kf_median + (1 - kw) * ag_median
    comb_low  = kw * kf_lower  + (1 - kw) * ag_lower
    comb_high = kw * kf_upper  + (1 - kw) * ag_upper

    return {
        "kalman":      {"median": kf_median, "lower": kf_lower, "upper": kf_upper},
        "arima_garch": {"median": ag_median, "lower": ag_lower, "upper": ag_upper},
        "combined":    {"median": comb_med,  "lower": comb_low,  "upper": comb_high,
                        "kw": kw},
        "states": states, "covs": covs,
    }


# ============================================================
# Visualisation
# ============================================================
def plot_asset(ticker, df, fc_res, last_price):
    hist  = df["close"].iloc[-80:]
    n_fc  = len(fc_res["combined"]["median"])
    dates = [df.index[-1] + timedelta(days=i + 1) for i in range(n_fc)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    colors = {"kalman": "royalblue", "arima_garch": "forestgreen", "combined": "tomato"}
    labels = {"kalman": "Kalman", "arima_garch": "ARIMA+GARCH", "combined": "Combined"}

    # Price + forecast
    ax = axes[0, 0]
    ax.plot(hist.index, hist.values, color="gray", linewidth=1.5, label="Historical")
    for key, col in colors.items():
        fc = fc_res[key]
        ax.plot(dates, fc["median"], color=col, linewidth=2,
                linestyle="--" if key != "combined" else "-", label=labels[key])
        if "lower" in fc:
            ax.fill_between(dates, fc["lower"], fc["upper"], alpha=0.12, color=col)
    ax.axhline(last_price, linestyle=":", color="black", alpha=0.5, label="Current")
    ax.set_title(f"{ticker} — Price Forecast")
    ax.legend(fontsize=8)
    ax.set_ylabel("Price ($)")

    # Kalman states
    ax2 = axes[0, 1]
    ax2.plot(df.index, np.exp(fc_res["states"][:, 0]), color="royalblue",
             linewidth=2, label="Filtered price")
    ax2.plot(df.index, df["close"].values, color="gray", alpha=0.5, linewidth=1, label="Actual")
    ax2.set_title(f"{ticker} — Kalman Filtered Price")
    ax2.legend(fontsize=8)

    ax3 = axes[1, 0]
    trend = fc_res["states"][:, 1]
    ax3.plot(df.index, trend, color="darkorange", linewidth=1.5)
    ax3.axhline(0, color="black", linestyle="--", alpha=0.4)
    col_now = "forestgreen" if trend[-1] > 0 else "tomato"
    ax3.axhline(trend[-1], color=col_now, linestyle=":", alpha=0.6)
    ax3.set_title("Latent Trend (Kalman)")
    ax3.set_ylabel("Log-price trend per day")

    ax4 = axes[1, 1]
    uncertainty = np.sqrt(fc_res["covs"][:, 0, 0])
    ax4.plot(df.index, uncertainty, color="purple", linewidth=1.5)
    ax4.set_title("Kalman State Uncertainty")
    ax4.set_ylabel("σ (state covariance)")

    plt.suptitle(f"{ticker} — Forecasting Dashboard  (Kalman wt={fc_res['combined']['kw']:.0%})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def print_comparison_table(all_metrics, all_fc):
    print("\n" + "=" * 100)
    print("MULTI-ASSET FORECAST DASHBOARD")
    print("=" * 100)
    hdr = (f"{'Ticker':<10} {'Price':>10} {'AnnVol%':>8} {'Trend':>9} "
           f"{'1D%':>7} {'1W%':>7} {'1M%':>8}")
    print(hdr)
    print("-" * 100)
    for ticker in all_metrics:
        if ticker not in all_fc:
            continue
        m   = all_metrics[ticker]
        fc  = all_fc[ticker]["combined"]["median"]
        lp  = m["current_price"]
        d1  = (fc[0] / lp - 1) * 100
        d7  = (fc[min(6,  len(fc)-1)] / lp - 1) * 100
        d30 = (fc[min(29, len(fc)-1)] / lp - 1) * 100
        tr  = all_fc[ticker]["states"][-1, 1]
        arrow = "↗" if tr > 0 else "↘"
        print(f"{ticker:<10} ${lp:>9.2f} {m['ann_vol_pct']:>7.1f}%"
              f"  {arrow} {tr:>7.4f}  {d1:>+6.1f}%  {d7:>+6.1f}%  {d30:>+7.1f}%")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("MULTI-ASSET TIME-SERIES FORECASTING")
    print("Kalman Filter + ARIMA + GARCH")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    metrics     = calculate_asset_metrics(assets_data)
    all_fc      = {}

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}   price=${df['close'].iloc[-1]:.2f}")
        print(f"{'─'*50}")
        kp = KALMAN_PARAMS.get(ticker, _DEFAULT_KP)
        try:
            fc_res = combined_forecast(df, kp, horizon=30, conf=CONFIDENCE)
            all_fc[ticker] = fc_res
            lp = float(df["close"].iloc[-1])
            comb = fc_res["combined"]["median"]
            print(f"  30-day target: ${comb[-1]:.2f}  "
                  f"({(comb[-1]/lp - 1)*100:+.1f}%)")
            plot_asset(ticker, df, fc_res, lp)
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if all_fc:
        print_comparison_table(metrics, all_fc)

        # Correlation heat-map
        r_df = pd.DataFrame({t: assets_data[t]["log_return"]
                             for t in all_fc}).dropna()
        if len(r_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(r_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                        center=0, square=True, ax=ax, annot_kws={"size": 7})
            ax.set_title("Return Correlation Matrix")
            plt.tight_layout()
            plt.show()

    print("\nForecasting complete.")


if __name__ == "__main__":
    main()
