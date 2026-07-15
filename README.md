# Hype.bot — Crypto Quant & Live Trading Suite

A 47-script quantitative research suite for **Hyperliquid perpetuals**, spanning distribution/risk analysis, volatility and forecasting models, ML and Bayesian signal generation, walk-forward backtesting, and a live order-execution layer (`hyperliquid-python-sdk`) that can turn Script 17's trade plans into real orders.

All market data comes from **Hyperliquid perpetual markets via `ccxt`** — this is a crypto-native project; it does not use Yahoo Finance or trade equities.

---

## Asset Universe

`data_loader.py`'s `SYMBOLS` dict is **dynamic by default**: at import time it fetches every native Hyperliquid perp, ranks by 24h % change and 24h notional volume, and takes the top gainers + top-volume names (deduped, backfilled from the volume ranking). This means the active universe changes from run to run — majors like BTC/ETH show up naturally via the volume ranking even when they aren't today's biggest movers.

If the live fetch fails (or you pass `run.py --tickers ...` / `30_live_executor.py --tickers ...`), it falls back to `COMMON_SYMBOLS`, a fixed list of live-verified Hyperliquid perp markets:

| Key | Hyperliquid Symbol |
|---|---|
| BTC | BTC/USDC:USDC |
| ETH | ETH/USDC:USDC |
| HYPE | HYPE/USDC:USDC |
| PAXG | PAXG/USDC:USDC |
| ZEC | ZEC/USDC:USDC |
| PENGU | PENGU/USDC:USDC |
| XRP | XRP/USDC:USDC |
| LIT | LIT/USDC:USDC |
| SOL | SOL/USDC:USDC |
| CRV | CRV/USDC:USDC |
| SUI | SUI/USDC:USDC |

Any valid Hyperliquid perp works — just add it to `COMMON_SYMBOLS` in `data_loader.py`.

---

## Installation

```bash
pip install ccxt pandas numpy scipy matplotlib seaborn \
            statsmodels scikit-learn arch cvxpy \
            hyperliquid-python-sdk eth-account python-dotenv
```

- `ccxt` — Hyperliquid market data (OHLCV, tickers, funding)
- `arch` — GARCH family volatility models (Script 9 and others)
- `cvxpy` — portfolio optimisation (Scripts 1, 13) — optional but recommended
- `hyperliquid-python-sdk` + `eth-account` + `python-dotenv` — **only needed for live order execution** (Script 30 / `hyperliquid_broker.py`); every analysis/signal/backtest script works without them

---

## File Structure

```
Stocks.Quant.Main.2/
├── data_loader.py                        ← shared Hyperliquid/ccxt data fetcher
├── run.py                                ← master runner / CLI, combined PDF report
├── hyperliquid_broker.py                 ← live order execution wrapper (Hyperliquid SDK)
├── config.json.example                   ← copy to config.json, fill in live-trading credentials
│
├── 1_density_distribution_analysis.py    ┐
├── 2_extreme_value_analysis.py           │  Analysis
├── 3_montecarlo_options_pricing.py       ┘
├── 9_volatility_models.py                   Models
├── 12_advanced_options.py                ┐
├── 13_portfolio_optimizer.py             │
├── 14_quantile_trading.py                │
├── 15_regime_detection.py                │
├── 16_backtesting.py                     │
├── 17_trade_planner.py                   │  Signals & Trading
├── 18_copula_risk.py                     │
├── 20_adaptive_forecast.py               │
├── 21_optimal_control.py                 │
├── 23_ml_alpha_model.py                  │
├── 24_classification_signals.py          │
├── 25_ml_forecast_signal.py              │
├── 26_walkforward_validation.py          │
├── 27_momentum_screener.py               │
├── 28_pairs_trading.py                   │
├── 29_live_accuracy_check.py             │  Paper-trade + prediction tracking
├── 30_live_executor.py                   ┘  Live order execution (real money)
│
├── 31_bayesian_fundamentals.py           ┐
├── 32_bayesian_regression_classification │
├── 33_variational_and_ep.py              │
├── 34_mcmc_bayesian_garch.py             │
├── 35_importance_rejection_sampling.py   │
├── 36_mixture_models_regime.py           │  Bayesian / Statistical Methods
├── 37_gaussian_processes.py              │  (Estimation Theory, Detection Theory,
├── 38_bayesian_optimization_tuning.py    │   Approximate Inference, Sampling Methods)
├── 39_estimator_performance_crlb.py      │
├── 40_mle_linear_models.py               │
├── 41_noninformative_priors_asymptotics  │
├── 42_kalman_smoother.py                 │
├── 43_bayesian_detection_chernoff.py     │
├── 44_multiple_hypothesis_testing.py     │
├── 45_neyman_pearson_detection.py        │
├── 46_critical_transition_indicators.py  │
└── 47_lppl_bubble_model.py               ┘
```

Scripts 4, 8, 11, 19, 22 were retired — superseded by later scripts (see the comment above `SCRIPTS` in `run.py`) and moved to `_deprecated/`, not deleted.

---

## Usage

### Run everything (combined PDF report)
```bash
python run.py
```
Trade Planner (17) and Live Accuracy Check (29) always run **last** in the default full run, since 17 synthesizes every other script's output and 29 consumes 17's plan.

### Run specific scripts
```bash
python run.py --scripts 1 3 9
```

### Override tickers / lookback
```bash
python run.py --tickers BTC ETH SOL
python run.py --days 500
```

### List scripts
```bash
python run.py --list
```

### Run a single script directly
```bash
python 9_volatility_models.py
```

---

## Live Trading (Script 30)

`30_live_executor.py` turns Script 17's trade plans into real Hyperliquid orders via `hyperliquid_broker.py`. It is **not** included in `run.py`'s script list — it never runs as a side effect of the routine report.

**Setup:**
1. Generate a Hyperliquid API/agent wallet at `app.hyperliquid.xyz/API` (a separate, revocable signing key — not your main wallet's key).
2. `copy config.json.example config.json` and fill in `secret_key` (the agent wallet's private key) and `account_address` (your main account — required whenever `secret_key` belongs to an agent wallet, since its address differs from your account's). Same format as hyperliquid-python-sdk's own official examples. `config.json` is gitignored — never commit it.

**Modes:**
```bash
python 30_live_executor.py                        # dry run, full universe — no orders sent
python 30_live_executor.py --tickers BTC           # dry run, one ticker
python 30_live_executor.py --live                  # PLACES REAL ORDERS, once
python 30_live_executor.py --live --loop           # PLACES REAL ORDERS, forever (default: every 15 min)
python 30_live_executor.py --live --loop --interval 30
```

Defaults to dry-run (prints/logs intent, sends nothing) — `--live` is required to actually trade. Per run: an IOC limit entry within 0.5% of Script 17's planned price (fills within that band, otherwise cancels — never chases an unbounded price or forces a fill), a reduce-only stop-market at SL1, and two reduce-only take-profits (half size each) at TP1/TP2, with a slippage buffer on trigger orders and a retry-then-`UNPROTECTED`-warning path if SL placement fails after entry. Every run reconciles prior live orders against actual Hyperliquid fills into `live_orders_log.csv` (realized P&L, not just entry intent).

---

## Script Summary

| # | Script | Key Output |
|---|---|---|
| 1 | Density Distribution | Best-fit PDF, AIC/BIC, QQ-plot, CVaR, CVXPY portfolio weights |
| 2 | Extreme Value Theory | GEV block-maxima, GPD POT excesses, VaR/CVaR at 95%/99% |
| 3 | Monte Carlo Options | GBM / Merton / Heston paths, Call price, Greeks, IV smile |
| 9 | Volatility Models | GARCH family comparison, persistence, leverage (GJR) |
| 12 | Advanced Options | CRR binomial, barrier/Asian exotics, IV surface |
| 13 | Portfolio Optimizer | Markowitz + risk parity |
| 14 | Quantile Trading | Regime-gated quantile breakout signals |
| 15 | Regime Detection | HMM market regime classification |
| 16 | Backtesting Engine | MA cross, mean-reversion, quantile, regime-conditioned, ML forecast, and TradePlanner (Script 17's actual logic, walk-forward) strategies vs buy-and-hold |
| 17 | Trade Planner | Direction, leverage, SL1/SL2/TP1/TP2, position size, R:R — synthesizes regime/vol/forecast/ML/Bayesian-detection/LQR |
| 18 | Copula Risk | Tail-dependence concentration risk across open positions |
| 20 | Adaptive Forecast | Kalman observer + MRAC forecast |
| 21 | Optimal Control | LQR/H-infinity optimal position sizing |
| 23 | ML Alpha Model | Ridge/Lasso with bias-variance validation |
| 24 | Classification Signals | Perceptron/SVM/MLP directional signals |
| 25 | ML Forecast Signal | Confidence-gated logistic regression, auto-selected bar interval |
| 26 | Walk-Forward Validation | Naive/HistMean/EWMA/AR1/Ensemble/Kalman forecast comparison |
| 27 | Momentum Screener | Breakout/exhaustion screener across the wide Hyperliquid universe |
| 28 | Pairs Trading | Cointegration-based statistical arbitrage |
| 29 | Live Accuracy Check | Paper-trade log + ML/Bayesian-detection prediction grading, running hit rates |
| 30 | Live Executor | **Real order placement** on Hyperliquid — dry-run by default |
| 31–47 | Bayesian / Statistical Methods | Estimation theory (CRLB, MLE, Jeffreys priors), detection theory (Bayesian, Neyman-Pearson, multiple hypothesis testing), approximate inference, sampling methods, Gaussian processes, Kalman smoothing, LPPL bubble model |

---

## Customising Tickers

Add to `COMMON_SYMBOLS` in `data_loader.py`:

```python
COMMON_SYMBOLS = {
    "BTC":  "BTC/USDC:USDC",
    "DOGE": "DOGE/USDC:USDC",   # add any live Hyperliquid perp
    # ...
}
```

Any Hyperliquid perpetual market works. The default dynamic universe (`SYMBOLS`, top gainers + top volume) doesn't need editing — it rebuilds itself every run.

---

## Notes

- All scripts import `data_loader.py` from the same directory — run them from inside this folder, or use `run.py`.
- Crypto trades 24/7, so annualization throughout the suite uses √365, not √252.
- Parquet OHLCV cache (`.cache/`) refreshes every 12 hours; `run.py --clear-cache` forces a re-fetch.
- Never commit `config.json` — it holds your Hyperliquid signing key. `config.json.example` documents the required fields.
#   H y p e . b o t . F  
 