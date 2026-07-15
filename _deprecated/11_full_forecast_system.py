"""
================================================================
Script 11 — Full Forecast System
================================================================
Adapted from: 12_forecast.py  (density + forecast + trading signals)
Data source : Yahoo Finance (yfinance) via data_loader.py

Methods:
  • Ensemble forecasting: ARIMA(1,0,1) + SARIMAX + Random Forest
  • VaR / CVaR risk metrics (historical & parametric)
  • Quantile-based trading signals
    – BUY  when rolling quantile ∈ [5%, 15%]  (oversold)
    – SELL when rolling quantile ∈ [85%, 95%] (overbought)
  • Optional CVXPY portfolio optimisation
  • Full dashboard: price forecast, distribution, quantile chart,
    RSI, risk table, signal summary

Assets: all symbols in SYMBOLS (data_loader.py)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

from data_loader import load_all_assets, SYMBOLS, LOOKBACK_DAYS

try:
    from importlib import import_module as _im
    _r15 = _im("15_regime_detection")
    detect_regime = _r15.detect_regime
    REGIME_AVAILABLE = True
except Exception:
    REGIME_AVAILABLE = False

PLOT_STYLE   = "seaborn-v0_8-darkgrid"
HORIZON      = 5          # forecast days
CONF_LEVELS  = [0.95, 0.99]

TRADING_Q = {
    "long_lo":  0.05, "long_hi":  0.15,
    "short_lo": 0.85, "short_hi": 0.95,
    "exit_long": 0.60, "exit_short": 0.40,
}


# ============================================================
# Feature Engineering
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
    return df.dropna()


def _rsi(price: pd.Series, period=14) -> pd.Series:
    delta = price.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ============================================================
# Forecasting
# ============================================================
def arima_fc(series: pd.Series, h: int):
    try:
        res = ARIMA(series.values, order=(1, 0, 1)).fit()
        return np.asarray(res.forecast(h))
    except Exception:
        return np.full(h, series.mean())


def rf_fc(df_feat: pd.DataFrame, h: int):
    feat_cols = [c for c in df_feat.columns if "lag" in c or c in ["vol_20", "rsi"]]
    X = df_feat[feat_cols].dropna()
    y = df_feat["log_return"].loc[X.index]
    if len(X) < 80:
        return np.full(h, y.mean())
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    preds = []
    row   = X.iloc[[-1]].copy()
    for _ in range(h):
        p = float(model.predict(row)[0])
        preds.append(p)
        new_row = row.copy()
        for lag in range(4, 0, -1):
            if f"r_lag{lag}" in new_row:
                if lag > 1:
                    new_row[f"r_lag{lag}"] = new_row[f"r_lag{lag-1}"]
                else:
                    new_row["r_lag1"] = p
        row = new_row
    return np.array(preds)


def ensemble_fc(df_feat: pd.DataFrame, h: int):
    r = df_feat["log_return"]
    a = arima_fc(r, h)
    f = rf_fc(df_feat, h)
    return 0.45 * a + 0.55 * f


# ============================================================
# Risk Metrics
# ============================================================
def risk_metrics(returns: pd.Series) -> dict:
    out = {}
    for conf in CONF_LEVELS:
        alpha  = 1 - conf
        h_var  = float(np.percentile(returns, alpha * 100))
        h_cvar = float(returns[returns <= h_var].mean())
        mu, sig = float(returns.mean()), float(returns.std())
        p_var  = float(stats.norm.ppf(alpha, mu, sig))
        p_cvar = float(mu - sig * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)
        out[conf] = {
            "hist_var":  h_var,  "hist_cvar":  h_cvar,
            "param_var": p_var,  "param_cvar": p_cvar,
        }
    return out


# ============================================================
# Trading Signals
# ============================================================
def trading_signal(df_feat: pd.DataFrame, fc: np.ndarray,
                   regime: int = 1) -> dict:
    q = float(df_feat["roll_q"].iloc[-1]) if "roll_q" in df_feat.columns else 0.5
    fc_1d = float(fc[0])
    fc_mag = min(abs(fc_1d) / 0.02, 1.0)

    # Regime gating: regime 2 (crisis) → force HOLD/reduce
    if regime == 2:
        return {"signal": "REDUCE", "confidence": 0.80,
                "quantile": q, "fc_1d": fc_1d, "regime": regime}

    sig = "HOLD"
    conf = 0.5

    # Regime 0 (trending): widen thresholds, favour trend signals
    # Regime 1 (mean-revert): use standard thresholds
    long_lo  = TRADING_Q["long_lo"]  * (1.5 if regime == 0 else 1.0)
    long_hi  = TRADING_Q["long_hi"]  * (1.5 if regime == 0 else 1.0)
    short_lo = TRADING_Q["short_lo"] - (0.05 if regime == 0 else 0.0)
    short_hi = TRADING_Q["short_hi"] - (0.05 if regime == 0 else 0.0)

    if long_lo <= q <= long_hi and fc_1d > 0:
        sig = "BUY"
        conf = (1 - q) * 0.6 + fc_mag * 0.4
    elif short_lo <= q <= short_hi and fc_1d < 0:
        sig = "SELL"
        conf = q * 0.6 + fc_mag * 0.4
    elif q > TRADING_Q["exit_long"]:
        sig = "EXIT_LONG"
        conf = min(0.5 + (q - TRADING_Q["exit_long"]) / (1 - TRADING_Q["exit_long"]), 0.95)
    elif q < TRADING_Q["exit_short"]:
        sig = "EXIT_SHORT"
        conf = min(0.5 + (TRADING_Q["exit_short"] - q) / TRADING_Q["exit_short"], 0.95)
    else:
        conf = 0.3 + 0.4 * (1 - abs(q - 0.5) / 0.5)
    return {"signal": sig, "confidence": conf, "quantile": q,
            "fc_1d": fc_1d, "regime": regime}


# ============================================================
# Visualisation
# ============================================================
def plot_dashboard(ticker, df_feat, fc, rm, signal_info):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(3, 3, figure=fig)

    returns = df_feat["log_return"]
    last_p  = float(df_feat["close"].iloc[-1])

    # 1. Price + forecast
    ax1 = fig.add_subplot(gs[0, :2])
    hist = df_feat["close"].iloc[-60:]
    ax1.plot(hist.index, hist.values, color="gray", linewidth=1.5, label="Historical")
    fc_dates = pd.date_range(hist.index[-1], periods=len(fc) + 1, freq="B")[1:]
    fc_prices = last_p * np.exp(np.cumsum(fc))
    ax1.plot(fc_dates, fc_prices, "r--", linewidth=2, label="Forecast")
    ax1.fill_between(fc_dates, fc_prices * 0.97, fc_prices * 1.03, alpha=0.15, color="red")
    sig_col = "green" if signal_info["signal"] == "BUY" else \
              "red"   if signal_info["signal"] == "SELL" else "gray"
    ax1.scatter(fc_dates[0], fc_prices[0], s=200, color=sig_col,
                marker="^" if signal_info["signal"] == "BUY" else
                       "v" if signal_info["signal"] == "SELL" else "o",
                zorder=5, label=f"Signal: {signal_info['signal']}")
    ax1.set_title(f"{ticker} — Price Forecast")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 2. Return distribution + VaR
    ax2 = fig.add_subplot(gs[0, 2])
    sns.histplot(returns, bins=50, stat="density", alpha=0.5, ax=ax2, color="steelblue")
    for conf_lvl, m in rm.items():
        ax2.axvline(m["hist_var"],  linestyle="--", color="red",    linewidth=1.5,
                    label=f"VaR {int(conf_lvl*100)}% {m['hist_var']:.3%}")
        ax2.axvline(m["hist_cvar"], linestyle=":",  color="darkred", linewidth=1.5,
                    label=f"CVaR {int(conf_lvl*100)}%")
    ax2.set_title("Return Dist + Risk Metrics")
    ax2.legend(fontsize=7)

    # 3. Rolling quantile
    ax3 = fig.add_subplot(gs[1, 0])
    if "roll_q" in df_feat.columns:
        q_ser = df_feat["roll_q"].iloc[-80:]
        ax3.plot(q_ser.index, q_ser.values, linewidth=1.5, color="purple")
        ax3.axhspan(TRADING_Q["long_lo"], TRADING_Q["long_hi"],   alpha=0.15, color="green")
        ax3.axhspan(TRADING_Q["short_lo"], TRADING_Q["short_hi"], alpha=0.15, color="red")
        ax3.scatter(q_ser.index[-1], q_ser.iloc[-1], s=100, color="black", zorder=5)
    ax3.set_title("Rolling Quantile (50-day window)")
    ax3.set_ylabel("Quantile")
    ax3.grid(alpha=0.3)

    # 4. Signal info table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    rows = [
        ["Signal",       signal_info["signal"]],
        ["Confidence",   f"{signal_info['confidence']:.2%}"],
        ["Quantile",     f"{signal_info['quantile']:.2%}"],
        ["Forecast 1D",  f"{signal_info['fc_1d']:.4%}"],
        ["Current Price", f"${last_p:.2f}"],
    ]
    t = ax4.table(cellText=rows, loc="center", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.2, 1.6)
    ax4.set_title("Signal Details")

    # 5. Risk table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    r_rows = []
    for conf_lvl, m in rm.items():
        r_rows += [
            [f"VaR {int(conf_lvl*100)}%",  f"{m['hist_var']:.4%}"],
            [f"CVaR {int(conf_lvl*100)}%", f"{m['hist_cvar']:.4%}"],
            ["", ""],
        ]
    t2 = ax5.table(cellText=r_rows, loc="center", cellLoc="left")
    t2.auto_set_font_size(False)
    t2.set_fontsize(10)
    t2.scale(1.2, 1.4)
    ax5.set_title("Risk Metrics")

    # 6. Forecast bar
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.bar(range(1, len(fc) + 1), fc * 100, color="steelblue", alpha=0.8)
    ax6.axhline(0, color="black", linewidth=0.8)
    ax6.set_xlabel("Forecast Day")
    ax6.set_ylabel("Return (%)")
    ax6.set_title("Daily Forecast Returns")
    ax6.grid(axis="y", alpha=0.3)

    # 7. RSI
    ax7 = fig.add_subplot(gs[2, 1])
    if "rsi" in df_feat.columns:
        rsi_ser = df_feat["rsi"].iloc[-60:]
        ax7.plot(rsi_ser.index, rsi_ser.values, linewidth=1.5, color="darkorange")
        ax7.axhline(70, linestyle="--", color="red", alpha=0.6)
        ax7.axhline(30, linestyle="--", color="green", alpha=0.6)
        ax7.fill_between(rsi_ser.index, 30, 70, alpha=0.05, color="gray")
        ax7.set_ylim(0, 100)
    ax7.set_title("RSI (14-day)")
    ax7.grid(alpha=0.3)

    # 8. Vol 20-day
    ax8 = fig.add_subplot(gs[2, 2])
    if "vol_20" in df_feat.columns:
        v = df_feat["vol_20"].iloc[-80:] * np.sqrt(252) * 100
        ax8.plot(v.index, v.values, color="tomato", linewidth=1.5)
        ax8.axhline(float(v.mean()), linestyle="--", color="black", alpha=0.5)
    ax8.set_title("20-day Annualised Vol (%)")
    ax8.grid(alpha=0.3)

    fig.suptitle(f"{ticker} — Full Forecast & Trading Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Portfolio Optimisation (CVXPY)
# ============================================================
def optimise_portfolio(all_returns: dict, all_fc: dict, all_signals: dict,
                       max_weight: float = 0.20, risk_aversion: float = 8.0):
    if not CVXPY_AVAILABLE or len(all_returns) < 2:
        return
    assets = list(all_returns.keys())

    # Gate out assets whose own signal says reduce/exit-long: the old
    # version let the highest 1-day forecast (a REDUCE-flagged name)
    # absorb 100% of the portfolio.
    investable = [a for a in assets
                  if all_signals.get(a, {}).get("signal", "HOLD")
                  not in ("REDUCE", "SELL", "EXIT_LONG")]
    if len(investable) < 2:
        print("\n  [Portfolio] <2 investable assets after signal gating — "
              "defaulting to defensive equal weight over lowest-vol names.")
        vols = {a: float(all_returns[a].std()) for a in assets}
        investable = sorted(vols, key=vols.get)[:5]

    n      = len(investable)
    T      = min(len(all_returns[a]) for a in investable)
    R      = np.column_stack([all_returns[a].values[-T:] for a in investable])
    mu_vec = R.mean(axis=0)
    Sigma  = np.cov(R, rowvar=False)

    # Forecast tilt: shrink hard. A 1-day forecast of ±3% is ~30x the
    # daily mean return; unshrunk it completely dominates the objective.
    fc_adj = np.array([all_fc[a][0] if a in all_fc else 0.0 for a in investable])
    sig_w  = np.array([all_signals.get(a, {}).get("confidence", 0.5)
                       for a in investable])
    adj_mu = mu_vec + 0.05 * fc_adj * sig_w

    w    = cp.Variable(n)
    obj  = cp.Maximize(adj_mu @ w - 0.5 * risk_aversion * cp.quad_form(w, Sigma))
    cons = [w >= 0, cp.sum(w) == 1, w <= max_weight]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.CLARABEL, warm_start=True)
    except Exception:
        prob.solve(warm_start=True)
    if w.value is None:
        return
    print("\n" + "=" * 55)
    print("CVXPY PORTFOLIO WEIGHTS")
    print(f"(cap={max_weight:.0%}, risk_aversion={risk_aversion}, "
          f"{len(assets)-n} assets excluded by signal gating)")
    print("=" * 55)
    for a, wt in sorted(zip(investable, w.value), key=lambda x: -x[1]):
        if abs(wt) > 0.005:
            print(f"  {a:<10}: {wt:.2%}  (signal={all_signals.get(a, {}).get('signal', 'N/A')})")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("FULL FORECAST SYSTEM — DENSITY, SIGNALS & TRADING")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    all_returns, all_fc_map, all_sigs = {}, {}, {}

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}")
        print(f"{'─'*50}")

        df_feat = add_features(df)
        if len(df_feat) < 80:
            print("  Insufficient rows after feature engineering — skipping.")
            continue

        returns = df_feat["log_return"]
        all_returns[ticker] = returns

        # Regime detection
        regime_id, regime_name, regime_strat = 1, "Mean-Revert", "MEAN-REVERT"
        if REGIME_AVAILABLE:
            try:
                regime_id, regime_name, regime_strat = detect_regime(df)
            except Exception:
                pass
        print(f"  Regime: {regime_name} -> {regime_strat}")

        fc      = ensemble_fc(df_feat, HORIZON)
        rm      = risk_metrics(returns)
        signal  = trading_signal(df_feat, fc, regime=regime_id)

        all_fc_map[ticker] = fc
        all_sigs[ticker]   = signal

        print(f"  Signal: {signal['signal']}  "
              f"conf={signal['confidence']:.2%}  "
              f"q={signal['quantile']:.2%}  "
              f"fc_1d={signal['fc_1d']:.4%}")
        print(f"  VaR 95%: {rm[0.95]['hist_var']:.4%}  "
              f"CVaR 95%: {rm[0.95]['hist_cvar']:.4%}")

        plot_dashboard(ticker, df_feat, fc, rm, signal)

    # Portfolio
    optimise_portfolio(all_returns, all_fc_map, all_sigs)

    # Signal summary
    print("\n" + "=" * 65)
    print("ACTIONABLE SIGNALS SUMMARY")
    print("=" * 65)
    for ticker, sig in all_sigs.items():
        if sig["signal"] not in ("HOLD",):
            print(f"  {ticker:<10}: {sig['signal']:<12} "
                  f"conf={sig['confidence']:.2%}  "
                  f"fc_1d={sig['fc_1d']:.4%}")

    print("\nFull forecast system complete.")


if __name__ == "__main__":
    main()
