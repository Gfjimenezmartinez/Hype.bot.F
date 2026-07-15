"""
================================================================
Script 17 — Trade Planner (Manual Execution)
================================================================
Synthesizes data from across the suite to produce exact,
actionable trade parameters for each asset:

  • Direction (LONG / SHORT / NO TRADE)
  • Leverage (1x–5x based on regime + vol + conviction)
  • Entry zone
  • Stop Loss 1 (tight) & Stop Loss 2 (disaster)
  • Take Profit 1 (conservative) & Take Profit 2 (full target)
  • Position size (% of capital)
  • Risk:Reward ratio
  • Confidence score

Data sources used:
  - Regime detection (Script 15) → trade/don't-trade gate
  - GARCH volatility (Script 9 approach) → adaptive SL/TP widths
  - Quantile levels (Script 14 approach) → support/resistance
  - ARIMA forecast (Script 4 approach) → directional bias
  - Monte Carlo VaR (Script 3 approach) → disaster stop
  - Trend detection → confirmation filter
================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

from data_loader import (load_all_assets, LOOKBACK_DAYS, DISPLAY_NAMES,
                          format_price, SYMBOLS, COMMON_SYMBOLS, load_all_funding)

try:
    from importlib import import_module as _im
    _r15 = _im("15_regime_detection")
    detect_regime = _r15.detect_regime
    REGIME_NAMES = _r15.REGIME_NAMES
    REGIME_AVAILABLE = True
except Exception:
    REGIME_AVAILABLE = False
    REGIME_NAMES = {0: "Low-Vol Trend", 1: "Mean-Revert", 2: "Crisis"}
    def detect_regime(df):
        return 1, "Mean-Revert", "MEAN-REVERT"

try:
    _r18 = _im("18_copula_risk")
    get_concentration_adjustment = _r18.get_concentration_adjustment
    COPULA_AVAILABLE = True
except Exception:
    COPULA_AVAILABLE = False

try:
    _r20 = _im("20_adaptive_forecast")
    adaptive_forecast_fn = _r20.adaptive_forecast
    ADAPTIVE_AVAILABLE = True
except Exception:
    ADAPTIVE_AVAILABLE = False

try:
    _r21 = _im("21_optimal_control")
    get_optimal_position = _r21.get_optimal_position
    CONTROL_AVAILABLE = True
except Exception:
    CONTROL_AVAILABLE = False

try:
    _r25 = _im("25_ml_forecast_signal")
    get_ml_signal      = _r25.get_ml_signal
    get_best_ml_signal = _r25.get_best_ml_signal
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

try:
    _r43 = _im("43_bayesian_detection_chernoff")
    beta_binomial_winrate  = _r43.beta_binomial_winrate
    BAYES_DETECT_AVAILABLE = True
except Exception:
    BAYES_DETECT_AVAILABLE = False

try:
    _r26 = _im("26_walkforward_validation")
    get_cached_best_model = _r26.get_cached_best_model
    wf_single_forecast    = _r26.single_forecast
    WALKFORWARD_AVAILABLE = True
except Exception:
    WALKFORWARD_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA

PLOT_STYLE    = "seaborn-v0_8-darkgrid"
CAPITAL       = 100_000
RF            = 0.05
MAX_RISK_PCT  = 0.05       # max 5% of capital per trade (raised from 2%)
MAX_LEVERAGE  = 15.0       # hard leverage ceiling (raised from 5x) -- vol_scale in
                            # calc_leverage() still de-leverages high-vol/wide-stop
                            # tickers below this automatically
MAX_POSITION_PCT     = 50.0   # hard ceiling on any single position, % of capital
                                # (raised from 20.0 -- this, not leverage, is what
                                # actually caps position notional in practice; every
                                # trade so far has hit this cap, not MAX_RISK_PCT)
MAX_GROSS_EXPOSURE_PCT = 100.0  # hard ceiling on sum of all position_pct across the book
MC_PATHS      = 10_000
MC_STEPS      = 21         # ~1 month of trading days
HORIZON_DAYS  = 21


# ============================================================
# Adaptive Signal Reliability — Beta-Bernoulli bandit over Script 29's
# graded track record (predictions_log.csv). Replaces each confirmation
# signal's fixed conviction-modulator weight with one that reflects how
# well THAT signal has actually performed, instead of an assumed
# constant. Beta(1,1) uninformative prior over "was this signal's call
# correct" means the posterior naturally starts at the FIXED_MODULATOR_
# WEIGHT fallback below and only moves once enough graded calls exist
# to say something real -- a brand-new or thin-data signal can't swing
# conviction on noise, and today's live behavior doesn't regress to
# zero just because the track record is still empty.
#
# ADAPTIVE_SIGNAL_WEIGHTS is a module-level toggle (same pattern as
# ML_AVAILABLE) so Script 16's walk-forward TradePlanner backtest can
# pin every call to the fixed fallback -- using TODAY's accumulated hit
# rate to weight a HISTORICAL bar would itself be lookahead, since that
# hit rate wasn't known back then.
# ============================================================
SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_LOG_PATH = os.path.join(SUITE_DIR, "predictions_log.csv")

FIXED_MODULATOR_WEIGHT  = 0.3   # today's status-quo weight, used as fallback
MAX_MODULATOR_WEIGHT    = 0.6   # ceiling once a signal has a genuinely proven track record
MIN_GRADED_FOR_ADAPTIVE = 30    # below this, the empirical rate is too noisy to trust
ADAPTIVE_SIGNAL_WEIGHTS = True  # Script 16 sets this False for its backtest

# Same lookahead concern as ADAPTIVE_SIGNAL_WEIGHTS above, different
# mechanism: get_cached_best_model() reflects TODAY's walk-forward
# validation, which wasn't known at any historical bar. Script 16 pins
# this False for its backtest too.
WALKFORWARD_MODEL_SELECTION = True

_signal_hit_rate_cache = None


def _compute_signal_hit_rates():
    rates = {}
    if not os.path.exists(PRED_LOG_PATH):
        return rates
    try:
        log_df = pd.read_csv(PRED_LOG_PATH)
    except Exception:
        return rates
    for col in ("ml_correct", "bayes_correct"):
        if col not in log_df.columns:
            continue
        graded = log_df[col].dropna()
        if graded.empty:
            continue
        n_correct = int(graded.astype(bool).sum())
        n_total = len(graded)
        a_n, b_n = 1 + n_correct, 1 + (n_total - n_correct)   # Beta(1,1) prior
        rates[col] = {"hit_rate": a_n / (a_n + b_n), "n": n_total}
    return rates


def refresh_signal_weights():
    """Force a re-read of predictions_log.csv on the next lookup -- call
    at the top of a long --loop iteration to pick up calls Script 29
    graded since this process started."""
    global _signal_hit_rate_cache
    _signal_hit_rate_cache = None


def signal_modulator_weight(signal_col):
    """Beta-Bernoulli posterior-derived modulator weight for `signal_col`
    ('ml_correct' or 'bayes_correct'), falling back to
    FIXED_MODULATOR_WEIGHT until enough graded history exists or when
    ADAPTIVE_SIGNAL_WEIGHTS is off (Script 16's backtest)."""
    if not ADAPTIVE_SIGNAL_WEIGHTS:
        return FIXED_MODULATOR_WEIGHT
    global _signal_hit_rate_cache
    if _signal_hit_rate_cache is None:
        _signal_hit_rate_cache = _compute_signal_hit_rates()
    info = _signal_hit_rate_cache.get(signal_col)
    if not info or info["n"] < MIN_GRADED_FOR_ADAPTIVE:
        return FIXED_MODULATOR_WEIGHT
    # hit_rate <= 0.5 (no better than chance) -> 0, not a fixed floor --
    # a disproven signal should lose its influence entirely, not just
    # shrink toward the old constant.
    return max(0.0, (info["hit_rate"] - 0.5) * 2) * MAX_MODULATOR_WEIGHT


# ============================================================
# Volatility Estimation (GARCH or rolling)
# ============================================================
def estimate_volatility(returns):
    ann_vol = float(returns.std() * np.sqrt(365))   # crypto trades 24/7
    daily_vol = float(returns.std())
    garch_vol = None

    if ARCH_AVAILABLE and len(returns) > 60:
        try:
            res = arch_model(returns * 100, mean="Constant", vol="GARCH",
                             p=1, q=1, dist="normal").fit(disp="off")
            garch_vol = float(np.sqrt(res.forecast(horizon=1)
                              .variance.iloc[-1].values[0])) / 100
        except Exception:
            pass

    current_vol = garch_vol if garch_vol else daily_vol
    return {
        "ann_vol": ann_vol,
        "daily_vol": daily_vol,
        "garch_vol": garch_vol,
        "current_vol": current_vol,
    }


# ============================================================
# Forecast — Script 26's walk-forward-validated best model, falling
# back to the fixed adaptive (Script 20) -> ARIMA -> historical-mean
# order if walk-forward selection is unavailable/off or its chosen
# model can't produce a number this call.
# ============================================================
def forecast_return(returns, horizon=5, df=None, ticker=None):
    # Walk-forward-validated model selection (Script 26) — replaces the
    # fixed "always try Kalman first" assumption below with whichever
    # model has actually been winning for THIS ticker recently. "Naive"
    # winning is meaningful too: it means nothing tested beats a
    # no-change forecast right now, so fc_total correctly comes back ~0
    # instead of leaning on a forecast that has no real edge.
    if (WALKFORWARD_AVAILABLE and WALKFORWARD_MODEL_SELECTION
            and ticker is not None and len(returns) >= 90):
        try:
            best_model = get_cached_best_model(ticker, returns, horizon_days=horizon)
            fc_total = wf_single_forecast(best_model, returns, horizon_days=horizon, df=df)
            if fc_total is not None:
                return float(fc_total), np.full(horizon, fc_total / horizon), f"WalkForward:{best_model}"
        except Exception:
            pass

    # Try adaptive observer + MRAC first
    if ADAPTIVE_AVAILABLE and df is not None:
        try:
            result = adaptive_forecast_fn(df, horizon=horizon)
            last_price = float(df["close"].iloc[-1])
            fc_prices = result["fc_prices"]
            fc_rets = np.diff(np.log(np.concatenate([[last_price], fc_prices])))
            return float(np.sum(fc_rets)), fc_rets, "Adaptive (Kalman+MRAC)"
        except Exception:
            pass
    # ARIMA fallback
    try:
        res = ARIMA(returns.values, order=(1, 0, 1)).fit()
        fc  = np.asarray(res.forecast(horizon))
        return float(np.sum(fc)), fc, "ARIMA fallback"
    except Exception:
        return float(returns.mean() * horizon), np.full(horizon, returns.mean()), "HistMean fallback"


# ============================================================
# Quantile Levels
# ============================================================
def quantile_levels(returns, price, lookback=60):
    r = returns.iloc[-lookback:].values
    best_dist, best_params = None, None
    best_aic = np.inf
    for name, dist in [("t", stats.t), ("laplace", stats.laplace),
                       ("logistic", stats.logistic), ("norm", stats.norm)]:
        try:
            params = dist.fit(r)
            ll = float(np.sum(dist.logpdf(r, *params)))
            aic = 2 * len(params) - 2 * ll
            if aic < best_aic:
                best_aic, best_dist, best_params = aic, dist, params
        except Exception:
            continue

    levels = {}
    if best_dist:
        for q in [0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98]:
            qr = float(best_dist.ppf(q, *best_params))
            levels[q] = price * np.exp(qr)
    return levels


# ============================================================
# Monte Carlo Disaster Levels
# ============================================================
def mc_var_levels(price, daily_vol, horizon=MC_STEPS, n_paths=MC_PATHS):
    dt  = 1.0
    Z   = np.random.default_rng(42).standard_normal((n_paths, horizon))
    log_ret = (RF / 365 - 0.5 * daily_vol**2) * dt + daily_vol * np.sqrt(dt) * Z
    final   = price * np.exp(np.cumsum(log_ret, axis=1)[:, -1])

    var_95  = float(np.percentile(final, 5))
    var_99  = float(np.percentile(final, 1))
    upside_95 = float(np.percentile(final, 95))
    upside_99 = float(np.percentile(final, 99))
    return {
        "mc_var_95": var_95, "mc_var_99": var_99,
        "mc_up_95": upside_95, "mc_up_99": upside_99,
        "mc_median": float(np.median(final)),
    }


# ============================================================
# Trend Detection
# ============================================================
def detect_trend(close, fast=10, slow=30):
    if len(close) < slow:
        return "NEUTRAL", 0.0
    ma_f = float(close.rolling(fast).mean().iloc[-1])
    ma_s = float(close.rolling(slow).mean().iloc[-1])
    c    = float(close.iloc[-1])
    if c > ma_f > ma_s:
        strength = (c - ma_s) / ma_s
        return "BULLISH", strength
    if c < ma_f < ma_s:
        strength = (ma_s - c) / ma_s
        return "BEARISH", strength
    return "NEUTRAL", 0.0


# ============================================================
# Leverage — LQR optimal + regime scaling
# ============================================================
def calc_leverage(regime_id, ann_vol, conviction, df=None, ticker=None):
    """
    Base leverage from regime + vol + conviction (always meaningful).
    Optimal-control signal (Script 21) acts as a secondary modulator
    reflecting the controller's directional conviction, scaled via tanh
    so it never collapses to a constant floor. Which controller
    (LQR/RSLQR/H-inf) is used is walk-forward validated per regime by
    Script 21 itself when `ticker` is given -- see
    get_cached_best_controller there; falls back to plain LQR otherwise.
    """
    regime_cap = {0: MAX_LEVERAGE, 1: MAX_LEVERAGE * 0.6, 2: 1.0}.get(regime_id, 1.0)
    # Floor lowered from 0.3 to 0.15: crypto's per-ticker vol spans
    # ~29%-151% annualized in this universe (vs equities' typical 20-40%),
    # and a 0.3 floor was pinning every ticker above ~100% vol (ZEC,
    # PENGU, LIT all measured at the same floored multiplier despite ZEC
    # running at 151% vol vs PENGU's 115%) -- the formula wants to
    # de-leverage them further but the floor wouldn't let it. Given
    # Hyperliquid's leverage ceiling is the whole reason for extra
    # caution here, let genuinely extreme-vol names keep scaling down.
    vol_scale  = max(0.15, min(1.0, 0.30 / max(ann_vol, 0.05)))
    conv_scale = 0.5 + conviction * 0.5
    leverage   = regime_cap * vol_scale * conv_scale

    if CONTROL_AVAILABLE and df is not None:
        try:
            ctrl = get_optimal_position(df, method="lqr", ticker=ticker, regime_id=regime_id)
            # Raw LQR pull is tiny by construction (heavily cost-penalized);
            # tanh-squash it into a [0.7, 1.0] modulator instead of letting
            # it dominate or collapse to a useless floor.
            lqr_pull = abs(ctrl["position_frac"])
            ctrl_mod = 0.7 + 0.3 * np.tanh(lqr_pull * 200)
            leverage *= ctrl_mod
        except Exception:
            pass

    return round(max(1.0, min(leverage, MAX_LEVERAGE)), 1)


# ============================================================
# Position Sizing
# ============================================================
def calc_position_size(capital, price, stop_pct, leverage, max_risk_pct=MAX_RISK_PCT,
                        max_position_pct=MAX_POSITION_PCT):
    if capital <= 0:
        return 0.0, 0.0   # nothing to size a position with -- e.g. live equity is genuinely $0
    risk_amount = capital * max_risk_pct
    risk_per_share = price * abs(stop_pct)
    if risk_per_share < 0.001:
        return 0.0, 0.0
    shares = risk_amount / risk_per_share
    notional = shares * price
    position_pct = notional / capital * 100

    capped_pct = min(position_pct, leverage * 100, max_position_pct)
    if position_pct > 0:
        shares *= capped_pct / position_pct

    return round(shares, 4), round(capped_pct, 1)


# ============================================================
# Core: Generate Trade Plan
# ============================================================
def generate_plan(ticker, df, capital=CAPITAL):
    price   = float(df["close"].iloc[-1])
    returns = df["log_return"].dropna()

    # 1. Regime
    regime_id, regime_name, regime_strat = detect_regime(df)

    # 2. Volatility
    vol = estimate_volatility(returns)

    # 3. Forecast (walk-forward-validated best model → adaptive → ARIMA fallback)
    fc_total, fc_daily, fc_source = forecast_return(returns, horizon=5, df=df, ticker=ticker)

    # 4. Quantile levels (60-day and 20-day)
    q60 = quantile_levels(returns, price, lookback=60)
    q20 = quantile_levels(returns, price, lookback=20)

    # 5. Monte Carlo
    mc = mc_var_levels(price, vol["current_vol"])

    # 6. Trend
    trend, trend_strength = detect_trend(df["close"])

    # ── Direction Decision ──────────────────────────────────
    direction = "NO TRADE"
    conviction = 0.0

    if regime_id == 2:
        direction = "NO TRADE"
        conviction = 0.0
    elif regime_id == 0:  # trending
        if trend == "BULLISH" and fc_total > 0:
            direction = "LONG"
            conviction = min(0.3 + trend_strength * 5 + abs(fc_total) * 4, 1.0)
        elif trend == "BEARISH" and fc_total < 0:
            direction = "SHORT"
            conviction = min(0.3 + trend_strength * 5 + abs(fc_total) * 4, 1.0)
    elif regime_id == 1:  # mean-reverting
        if q60 and price <= q60.get(0.10, price * 0.95):
            direction = "LONG"
            conviction = min(0.4 + abs(price - q60[0.10]) / price * 10, 1.0)
        elif q60 and price >= q60.get(0.90, price * 1.05):
            direction = "SHORT"
            conviction = min(0.4 + abs(price - q60[0.90]) / price * 10, 1.0)
        elif fc_total > 0.005 and trend in ("BULLISH", "NEUTRAL"):
            direction = "LONG"
            conviction = min(0.2 + abs(fc_total) * 3, 0.85)
        elif fc_total < -0.005 and trend in ("BEARISH", "NEUTRAL"):
            direction = "SHORT"
            conviction = min(0.2 + abs(fc_total) * 3, 0.85)

    # ── No trade → return early ─────────────────────────────
    if direction == "NO TRADE":
        return {
            "ticker": ticker, "price": price, "direction": direction,
            "regime": regime_name, "trend": trend, "conviction": 0,
            "leverage": 1.0, "entry": "-", "sl1": "-", "sl2": "-",
            "tp1": "-", "tp2": "-", "position_pct": 0, "shares": 0,
            "rr": 0, "vol": vol, "mc": mc, "q60": q60, "q20": q20,
            "fc_total": fc_total, "fc_source": fc_source, "ml": None, "bayes_detect": None,
        }

    # ── ML confirmation (Script 25) — bounded conviction modulator ──
    # Same pattern as the LQR leverage modulator below: a validated
    # secondary signal that sharpens or softens conviction, never a
    # replacement for the regime/trend/quantile decision above and
    # never enough on its own to flip a trade. Modulator weight is
    # adaptive (see signal_modulator_weight above) -- starts at the old
    # fixed 30% and moves toward this signal's own proven track record
    # once Script 29 has graded enough of its calls; conviction stays
    # capped at 1.0.
    ml_result = None
    if ML_AVAILABLE:
        try:
            symbol = SYMBOLS.get(ticker) or COMMON_SYMBOLS.get(ticker)
            ml_result = (get_best_ml_signal(symbol, ticker_df=df) if symbol
                         else get_ml_signal(df))
            agrees    = (ml_result["signal"] == "LONG"  and direction == "LONG") or \
                        (ml_result["signal"] == "SHORT" and direction == "SHORT")
            disagrees = (ml_result["signal"] == "LONG"  and direction == "SHORT") or \
                        (ml_result["signal"] == "SHORT" and direction == "LONG")
            ml_weight = signal_modulator_weight("ml_correct")
            if agrees:
                conviction = min(conviction * (1.0 + ml_weight * ml_result["confidence"]), 1.0)
            elif disagrees:
                conviction = conviction * (1.0 - ml_weight * ml_result["confidence"])
            # ml_result["signal"] == "FLAT" -> model itself isn't confident, no adjustment
        except Exception:
            pass

    # ── Bayesian detection confirmation (Script 43) — same bounded-
    # conviction-modulator pattern as ML above, a second and differently-
    # derived opinion. H0: return ~ N(-delta, sigma^2) vs H1 ~
    # N(+delta, sigma^2), prior P(up) from Script 31's real Beta-Binomial
    # win-rate posterior (not an assumed 50/50). Purely a function of
    # this ticker's own return history -- no live/network fetch, so
    # unlike the ML block above it's already lookahead-safe for Script
    # 16's TradePlanner backtest.
    #
    # Deliberately uses SYMMETRIC (0-1 loss) costs here, not Script 43's
    # own 2:1 asymmetric costs: that asymmetry was designed for a
    # risk-averse standalone diagnostic, and checked against real data
    # it makes the decision "LONG" for nearly every input (verified: on
    # ETH, the asymmetric threshold came out at -0.26, far outside any
    # realistic daily return, because delta -- the mean-return magnitude
    # in the denominator of Script 43's closed-form threshold -- is
    # intrinsically tiny). A confirmation signal feeding conviction
    # should be a neutral second opinion, not skewed by a cost ratio
    # that isn't about direction at all. Evaluates the posterior
    # P(H1|x) directly from the two Gaussian likelihoods instead of
    # Script 43's threshold-on-x formula, for the same reason -- no
    # division by delta, so no blowup regardless of how small it is.
    bayes_result = None
    if BAYES_DETECT_AVAILABLE and len(returns) >= 60:
        try:
            delta  = float(np.abs(returns.mean()))
            sigma2 = float(returns.var(ddof=1))
            sigma  = np.sqrt(sigma2)
            wr = beta_binomial_winrate(returns)
            pi1, pi0 = wr["mean"], 1 - wr["mean"]
            latest_return = float(returns.iloc[-1])

            f0 = stats.norm.pdf(latest_return, -delta, sigma)
            f1 = stats.norm.pdf(latest_return, delta, sigma)
            p_up = (pi1 * f1) / (pi1 * f1 + pi0 * f0) if (f1 + f0) > 0 else pi1
            bayes_signal = "LONG" if p_up > 0.5 else "SHORT"
            bayes_confidence = float(abs(p_up - 0.5) * 2)   # 0 at the decision boundary, 1 at the extreme
            bayes_result = {"signal": bayes_signal, "confidence": bayes_confidence,
                             "p_up": p_up, "prior_p_up": pi1}

            agrees    = (bayes_signal == "LONG"  and direction == "LONG") or \
                        (bayes_signal == "SHORT" and direction == "SHORT")
            disagrees = (bayes_signal == "LONG"  and direction == "SHORT") or \
                        (bayes_signal == "SHORT" and direction == "LONG")
            bayes_weight = signal_modulator_weight("bayes_correct")
            if agrees:
                conviction = min(conviction * (1.0 + bayes_weight * bayes_confidence), 1.0)
            elif disagrees:
                conviction = conviction * (1.0 - bayes_weight * bayes_confidence)
        except Exception:
            pass

    # ── Leverage ────────────────────────────────────────────
    leverage = calc_leverage(regime_id, vol["ann_vol"], conviction, df=df, ticker=ticker)

    # ── Stop Loss & Take Profit ─────────────────────────────
    cv = vol["current_vol"]
    # TP multipliers scale with conviction: stronger setups get wider
    # profit targets, so R:R reflects actual signal strength instead of
    # being a fixed constant.
    atr_mult_sl1 = 1.5
    atr_mult_sl2 = 3.0
    atr_mult_tp1 = 1.5 + conviction * 1.5   # R:R ranges ~1.0R to ~2.0R
    atr_mult_tp2 = 3.0 + conviction * 3.0   # R:R2 ranges ~2.0R to ~4.0R

    if direction == "LONG":
        entry = price
        sl1   = price * (1 - atr_mult_sl1 * cv)
        sl2   = max(mc["mc_var_95"], price * (1 - atr_mult_sl2 * cv))
        tp1   = price * (1 + atr_mult_tp1 * cv)
        tp2   = min(q60.get(0.90, price * 1.10),
                    price * (1 + atr_mult_tp2 * cv))

        # Clamp TP to quantile resistance if lower
        if q60.get(0.75):
            tp1 = max(tp1, q60[0.75])
        if q60.get(0.95):
            tp2 = min(tp2, q60[0.95]) if q60[0.95] > tp1 else tp2

        # Clamp SL to quantile support if higher
        if q20.get(0.05):
            sl2 = max(sl2, q20[0.05])

    else:  # SHORT
        entry = price
        sl1   = price * (1 + atr_mult_sl1 * cv)
        sl2   = min(mc["mc_up_95"], price * (1 + atr_mult_sl2 * cv))
        tp1   = price * (1 - atr_mult_tp1 * cv)
        tp2   = max(q60.get(0.10, price * 0.90),
                    price * (1 - atr_mult_tp2 * cv))

        if q60.get(0.25):
            tp1 = min(tp1, q60[0.25])
        if q60.get(0.05):
            tp2 = max(tp2, q60[0.05]) if q60[0.05] < tp1 else tp2

        if q20.get(0.95):
            sl2 = min(sl2, q20[0.95])

    # ── Position Sizing ─────────────────────────────────────
    # Risk-per-trade scaled by this trade's own conviction instead of a
    # flat MAX_RISK_PCT for every ticker -- a high-conviction setup (e.g.
    # ZEC on a strong breakout) gets up to 1.5x the base risk budget, a
    # marginal one (e.g. a weak ETH signal) as little as 0.5x. Reuses the
    # conviction score already computed above (regime/trend/quantile,
    # sharpened by Script 25's ML confirmation) rather than a new metric.
    stop_pct = abs(entry - sl1) / entry
    risk_scale = 0.5 + conviction
    shares, position_pct = calc_position_size(capital, price, stop_pct, leverage,
                                               max_risk_pct=MAX_RISK_PCT * risk_scale)

    # ── Risk:Reward ─────────────────────────────────────────
    risk   = abs(entry - sl1)
    reward = abs(tp1 - entry)
    rr     = round(reward / risk, 2) if risk > 0.001 else 0.0

    return {
        "ticker": ticker, "price": price, "direction": direction,
        "regime": regime_name, "trend": trend,
        "conviction": round(conviction, 2),
        "leverage": leverage,
        "entry": round(entry, 2),
        "sl1": round(sl1, 2), "sl2": round(sl2, 2),
        "tp1": round(tp1, 2), "tp2": round(tp2, 2),
        "position_pct": position_pct, "shares": shares,
        "rr": rr,
        "vol": vol, "mc": mc, "q60": q60, "q20": q20,
        "fc_total": fc_total, "fc_source": fc_source, "ml": ml_result, "bayes_detect": bayes_result,
    }


# ============================================================
# Portfolio-Level Risk Adjustment
# ============================================================
def apply_portfolio_risk_adjustments(trades, assets_data=None):
    """Gross exposure cap + copula concentration adjustment, applied
    in-place to a list of actionable (LONG/SHORT) plans. Shared by the
    manual sheet (main()) and Script 30's live executor so real orders
    get the same book-wide risk treatment as the printed plan."""
    if not trades:
        return trades

    raw_total_alloc = sum(p["position_pct"] for p in trades)
    if raw_total_alloc > MAX_GROSS_EXPOSURE_PCT:
        weights = [p["position_pct"] * max(p["conviction"], 0.05) for p in trades]
        total_weight = sum(weights)
        for p, w in zip(trades, weights):
            new_pct = min(MAX_GROSS_EXPOSURE_PCT * w / total_weight, MAX_POSITION_PCT)
            scale = new_pct / p["position_pct"] if p["position_pct"] > 0 else 0.0
            p["position_pct"] = new_pct
            p["shares"] *= scale
        print(f"\n  !! GROSS EXPOSURE {raw_total_alloc:.1f}% exceeds "
              f"{MAX_GROSS_EXPOSURE_PCT:.0f}% cap — redistributing by "
              f"conviction (higher-conviction trades keep more, weaker "
              f"ones cut more)")

    if COPULA_AVAILABLE and assets_data is not None:
        print(f"\n  {'─'*56}")
        print(f"  CONCENTRATION RISK CHECK (Copula Tail Dependence)")
        print(f"  {'─'*56}")
        try:
            positions = [(p["ticker"], p["direction"], p["position_pct"])
                         for p in trades]
            risk_mult, pair_risks = get_concentration_adjustment(
                assets_data, positions)
            print(f"  Risk multiplier: {risk_mult:.2f}x")
            if risk_mult > 1.3:
                adj_factor = 1.0 / risk_mult
                print(f"  !! CONCENTRATED — reduce all positions by {(1-adj_factor)*100:.0f}%")
                for p in trades:
                    p["position_pct"] *= adj_factor
                    p["shares"] *= adj_factor
            else:
                print(f"  Concentration OK — no adjustment needed.")
            if pair_risks:
                print(f"  Correlated pairs (tail dep > 15%):")
                for pair, td in sorted(pair_risks.items(), key=lambda x: -x[1]):
                    print(f"    {pair}: {td:.3f}")
        except Exception as e:
            print(f"  Copula check failed: {e}")

    return trades


# ============================================================
# Display
# ============================================================
def print_plan(plan):
    t = plan["ticker"]
    d = plan["direction"]

    icon = {"LONG": ">>", "SHORT": "<<", "NO TRADE": "--"}.get(d, "--")
    print(f"\n{'='*60}")
    print(f"  {icon}  {t}  |  {d}  |  {format_price(plan['price'])}")
    print(f"{'='*60}")
    print(f"  Regime:     {plan['regime']}")
    print(f"  Trend:      {plan['trend']}")
    print(f"  Conviction: {plan['conviction']:.0%}")
    fc_target = plan['price'] * np.exp(plan['fc_total'])
    print(f"  5d Forecast:{plan['fc_total']:+.4f} ({plan['fc_total']*100:+.2f}%)  "
          f"-> {format_price(fc_target)}")
    fc_src = plan.get("fc_source") or ("Adaptive (Kalman+MRAC)" if ADAPTIVE_AVAILABLE else "ARIMA fallback")
    lev_src = "LQR optimal" if CONTROL_AVAILABLE else "heuristic"
    print(f"  Sources:    forecast={fc_src}  leverage={lev_src}")
    ml = plan.get("ml")
    if ml:
        print(f"  ML confirm: {ml['signal']}  (conf={ml['confidence']:.0%}, "
              f"recent_acc={ml['recent_test_acc']:.0%}, "
              f"interval={ml.get('interval_used', '1d')})")
    bd = plan.get("bayes_detect")
    if bd:
        print(f"  Bayes confirm: {bd['signal']}  (conf={bd['confidence']:.0%}, "
              f"P(up|x)={bd['p_up']:.0%}, prior_P(up)={bd['prior_p_up']:.0%})")

    if d == "NO TRADE":
        print(f"  Action:     STAY OUT — regime/conditions unfavorable")
        v = plan["vol"]
        print(f"  Vol:        daily={v['daily_vol']*100:.2f}%  ann={v['ann_vol']*100:.1f}%"
              f"  garch={'%.2f%%' % (v['garch_vol']*100) if v['garch_vol'] else 'N/A'}")
        return

    print(f"  {'─'*56}")
    print(f"  Leverage:   {plan['leverage']}x")
    print(f"  Entry:      {format_price(plan['entry']):>12}")
    print(f"  Stop Loss 1:{format_price(plan['sl1']):>12}  ({abs(plan['entry']-plan['sl1'])/plan['entry']*100:>+.2f}%)")
    print(f"  Stop Loss 2:{format_price(plan['sl2']):>12}  ({abs(plan['entry']-plan['sl2'])/plan['entry']*100:>+.2f}%)")
    print(f"  Take Profit1:{format_price(plan['tp1']):>11}  ({abs(plan['tp1']-plan['entry'])/plan['entry']*100:>+.2f}%)")
    print(f"  Take Profit2:{format_price(plan['tp2']):>11}  ({abs(plan['tp2']-plan['entry'])/plan['entry']*100:>+.2f}%)")
    print(f"  {'─'*56}")
    print(f"  Risk:Reward: {plan['rr']}R")
    print(f"  Position:    {plan['position_pct']:.1f}% of capital  ({plan['shares']:.4f} shares)")
    print(f"  Max Risk:    ${CAPITAL * MAX_RISK_PCT:,.0f} ({MAX_RISK_PCT*100:.0f}% of ${CAPITAL:,.0f})")

    v = plan["vol"]
    print(f"  Vol:         daily={v['daily_vol']*100:.2f}%  ann={v['ann_vol']*100:.1f}%"
          f"  garch={'%.2f%%' % (v['garch_vol']*100) if v['garch_vol'] else 'N/A'}")

    mc = plan["mc"]
    print(f"  MC 21d:      95%ile={format_price(mc['mc_var_95'])}–{format_price(mc['mc_up_95'])}"
          f"  median={format_price(mc['mc_median'])}")


def plot_plan(plan):
    if plan["direction"] == "NO TRADE":
        return

    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 4))
    t  = plan["ticker"]
    p  = plan["price"]
    d  = plan["direction"]

    levels = [
        ("SL2",  plan["sl2"], "#B71C1C", "--"),
        ("SL1",  plan["sl1"], "#F44336", "-"),
        ("Entry", plan["entry"], "white", "-"),
        ("TP1",  plan["tp1"], "#4CAF50", "-"),
        ("TP2",  plan["tp2"], "#1B5E20", "--"),
    ]
    if d == "SHORT":
        levels = levels[::-1]

    y_vals = [l[1] for l in levels]
    y_min, y_max = min(y_vals) * 0.995, max(y_vals) * 1.005

    for label, val, color, ls in levels:
        ax.axhline(val, color=color, linestyle=ls, lw=2, alpha=0.8)
        side = y_max if val > p else y_min
        ax.text(0.02, val, f"  {label}: {format_price(val)}", transform=ax.get_yaxis_transform(),
                va="bottom" if val < p else "top", fontsize=9, fontweight="bold", color=color)

    # Fill zones
    ax.axhspan(plan["sl1"], plan["entry"], alpha=0.08, color="red")
    ax.axhspan(plan["entry"], plan["tp1"], alpha=0.08, color="green")

    mc = plan["mc"]
    ax.axhspan(mc["mc_var_95"], mc["mc_up_95"], alpha=0.04, color="blue",
               label=f"MC 95% range")

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Price ($)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"{t}  |  {d} {plan['leverage']}x  |  "
                 f"R:R={plan['rr']}  |  Conv={plan['conviction']:.0%}  |  "
                 f"{plan['regime']}", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  TRADE PLANNER — MANUAL EXECUTION SHEET")
    print(f"  Capital: ${CAPITAL:,.0f}  |  Max Risk/Trade: {MAX_RISK_PCT*100:.0f}%")
    print(f"  Horizon: {HORIZON_DAYS}d  |  MC Paths: {MC_PATHS:,}")
    print("=" * 60)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    if ML_AVAILABLE:
        # Pre-warm funding-rate cache once, in parallel, before the per-ticker
        # loop below -- get_best_ml_signal fetches funding per-ticker, and a
        # cold sequential fetch there (~18-80s/ticker) would multiply badly
        # across 9-10 tickers in a live run.
        load_all_funding(symbols={t: (SYMBOLS.get(t) or COMMON_SYMBOLS.get(t))
                                   for t in assets_data if (SYMBOLS.get(t) or COMMON_SYMBOLS.get(t))})
    plans = []

    for ticker, df in assets_data.items():
        try:
            plan = generate_plan(ticker, df)
            plans.append(plan)
            print_plan(plan)
            plot_plan(plan)
        except Exception as e:
            print(f"\n  {ticker}: Error — {e}")

    # ── Summary Table ───────────────────────────────────────
    trades   = [p for p in plans if p["direction"] != "NO TRADE"]
    no_trade = [p for p in plans if p["direction"] == "NO TRADE"]

    apply_portfolio_risk_adjustments(trades, assets_data)

    print(f"\n\n{'='*60}")
    print("  TRADE SHEET SUMMARY")
    print(f"{'='*60}")

    if trades:
        print(f"\n  ACTIONABLE ({len(trades)} assets):")
        print(f"  {'Ticker':<10} {'Dir':<6} {'Lev':>4} {'Entry':>12} "
              f"{'SL1':>12} {'TP1':>12} {'R:R':>5} {'Conv':>5} {'Pos%':>6}")
        print(f"  {'─'*78}")
        for p in sorted(trades, key=lambda x: -x["conviction"]):
            print(f"  {p['ticker']:<10} {p['direction']:<6} {p['leverage']:>3.0f}x "
                  f"${p['entry']:>10,.2f} ${p['sl1']:>10,.2f} "
                  f"${p['tp1']:>10,.2f} {p['rr']:>4.1f}R "
                  f"{p['conviction']:>4.0%} {p['position_pct']:>5.1f}%")

    if no_trade:
        print(f"\n  NO TRADE ({len(no_trade)} assets): "
              f"{', '.join(p['ticker'] for p in no_trade)}")

    total_alloc = sum(p["position_pct"] for p in trades)
    print(f"\n  Total allocation: {total_alloc:.1f}% of capital")
    print(f"  Remaining cash:  {max(0, 100 - total_alloc):.1f}%")
    print(f"  Gross cap:       {MAX_GROSS_EXPOSURE_PCT:.0f}%  |  "
          f"Per-position cap: {MAX_POSITION_PCT:.0f}%")

    print("\nTrade planner complete.")


if __name__ == "__main__":
    main()
