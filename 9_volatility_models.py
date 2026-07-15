"""
================================================================
Script 9 — Volatility Models
================================================================
Adapted from: 10_volatility_models.py  (enhanced version)
Data source : Yahoo Finance (yfinance) via data_loader.py

Methods:
  • ARCH(1), GARCH(1,1)-Normal, GARCH(1,1)-t,
    EGARCH(1,1)-t, GJR-GARCH(1,1)-t
  • AIC / BIC model comparison + persistence
  • 10-day vol forecast with approx. confidence bands
  • ACF diagnostics on returns and squared returns
  • Leverage-effect test (gamma in GJR)

Assets: all symbols in SYMBOLS (data_loader.py)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    print("arch package not installed — run: pip install arch")
    ARCH_AVAILABLE = False

from data_loader import load_all_assets, SYMBOLS, LOOKBACK_DAYS

SCALE            = 100        # multiply returns for numerical stability
FORECAST_HORIZON = 10         # days
PLOT_STYLE       = "seaborn-v0_8-darkgrid"


# ============================================================
# Model fitting
# ============================================================
MODEL_CONFIGS = {
    "ARCH(1)":          dict(mean="Zero",     vol="ARCH",  p=1,      dist="normal"),
    "GARCH(1,1)-N":     dict(mean="Constant", vol="GARCH", p=1, q=1, dist="normal"),
    "GARCH(1,1)-t":     dict(mean="Constant", vol="GARCH", p=1, q=1, dist="t"),
    "EGARCH(1,1)-t":    dict(mean="Constant", vol="EGARCH",p=1, q=1, dist="t"),
    "GJR-GARCH(1,1)-t": dict(mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t"),
}


def fit_all_models(scaled_returns: pd.Series) -> dict:
    fitted = {}
    for name, cfg in MODEL_CONFIGS.items():
        try:
            res = arch_model(scaled_returns, **cfg).fit(disp="off", show_warning=False)
            fitted[name] = res
            print(f"  ✓ {name:<22} AIC={res.aic:.2f}  BIC={res.bic:.2f}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return fitted


def compare_models(fitted: dict, scale: float) -> pd.DataFrame:
    rows = []
    for name, res in fitted.items():
        persist = None
        p = res.params
        if "alpha[1]" in p and "beta[1]" in p:
            persist = float(p["alpha[1]"] + p["beta[1]"])
            if "gamma[1]" in p:
                persist += 0.5 * float(p["gamma[1]"])
        rows.append({
            "Model":       name,
            "AIC":         round(res.aic, 2),
            "BIC":         round(res.bic, 2),
            "LogLik":      round(res.loglikelihood, 2),
            "Persistence": round(persist, 4) if persist else None,
            "MeanVol%":    round(float(res.conditional_volatility.mean()) / scale, 4),
        })
    df = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    delta = df["AIC"] - df["AIC"].min()
    df["AIC_weight"] = (np.exp(-delta / 2) / np.exp(-delta / 2).sum()).round(3)
    return df


# ============================================================
# Diagnostics plot
# ============================================================
def plot_vol_diagnostics(returns: pd.Series, scaled: pd.Series, ticker: str):
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))

    axes[0, 0].plot(returns.values, linewidth=0.7, color="steelblue")
    axes[0, 0].set_title(f"{ticker} — Log Returns")
    axes[0, 0].axhline(0, linestyle="--", color="red", alpha=0.4)

    axes[0, 1].plot(returns.rolling(20).std().values * np.sqrt(365) * 100,   # crypto trades 24/7
                    color="darkorange", linewidth=1.2)
    axes[0, 1].set_title("20-day Rolling Vol (annualised %)")

    # PACF requires nlags < 50% of the sample size — cap for short histories.
    max_lags = max(1, min(40, len(returns) // 2 - 1))

    plot_acf(returns,   lags=max_lags, ax=axes[1, 0])
    axes[1, 0].set_title("ACF Returns")
    plot_acf(returns**2, lags=max_lags, ax=axes[1, 1])
    axes[1, 1].set_title("ACF Squared Returns (vol clustering)")

    plot_pacf(returns**2, lags=max_lags, ax=axes[2, 0])
    axes[2, 0].set_title("PACF Squared Returns")
    stats.probplot(scaled, dist="norm", plot=axes[2, 1])
    axes[2, 1].set_title("QQ vs Normal")

    plt.suptitle(f"{ticker} — Volatility Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Forecast plot
# ============================================================
def plot_forecasts(ticker: str, fitted: dict, horizon: int, scale: float):
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, res in fitted.items():
        try:
            fc = res.forecast(horizon=horizon)
            vol = np.sqrt(fc.variance.iloc[-1].values) / scale
            ax.plot(range(1, horizon + 1), vol,
                    marker="o", linewidth=2, label=name)
        except Exception:
            continue
    ax.set_xlabel("Forecast Horizon (days)")
    ax.set_ylabel("Volatility")
    ax.set_title(f"{ticker} — {horizon}-day Volatility Forecast")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    if not ARCH_AVAILABLE:
        print("arch package required. Install with: pip install arch")
        return

    print("=" * 65)
    print("VOLATILITY MODELS — GARCH FAMILY")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    all_compare = {}

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}")
        print(f"{'─'*50}")

        returns = df["log_return"].dropna()
        scaled  = returns * SCALE

        # ARCH-LM test
        lm, lm_p = het_arch(returns)[:2]
        print(f"  ARCH-LM test: stat={lm:.4f}  p={lm_p:.4f}  "
              f"{'→ ARCH effects present' if lm_p < 0.05 else '→ no ARCH effects'}")

        plot_vol_diagnostics(returns, scaled, ticker)

        fitted  = fit_all_models(scaled)
        cmp_df  = compare_models(fitted, SCALE)
        print(f"\n  Model comparison:")
        print(cmp_df.to_string(index=False))
        all_compare[ticker] = cmp_df

        # Leverage effect
        if "GJR-GARCH(1,1)-t" in fitted:
            gjr = fitted["GJR-GARCH(1,1)-t"]
            gamma = gjr.params.get("gamma[1]", np.nan)
            if not np.isnan(gamma):
                print(f"  GJR gamma (leverage): {float(gamma):.4f} "
                      f"{'(negative asymmetry present)' if gamma > 0 else ''}")

        plot_forecasts(ticker, fitted, FORECAST_HORIZON, SCALE)

    print("\n" + "=" * 65)
    print("BEST MODELS BY AIC")
    print("=" * 65)
    for ticker, cmp in all_compare.items():
        if not cmp.empty:
            best = cmp.iloc[0]
            print(f"  {ticker:<10}: {best['Model']:<25} AIC={best['AIC']:.2f}")

    print("\nVolatility modeling complete.")


if __name__ == "__main__":
    main()
