"""
================================================================
run.py — Master Orchestrator  v2
Stock Quant Suite
================================================================
Improvements vs v1:
  • Data is fetched ONCE at startup and cached to parquet
    (data_loader handles 12-hour TTL automatically).
  • Always headless (Agg backend) — no GUI windows ever pop up,
    so nothing blocks waiting for you to close a chart.
  • Every figure from every script is collected into ONE combined
    PDF report at the end (reports/report_<timestamp>.pdf).
  • --clear-cache flag: wipe parquet cache before starting
  • Better timing table at the end

Usage:
    python run.py                           # run all, write PDF report
    python run.py --scripts 1 3 9           # run specific
    python run.py --list                    # list all scripts
    python run.py --tickers AAPL MSFT SPY   # override subset
    python run.py --days 500                # override lookback
    python run.py --pdf out.pdf             # custom PDF path
    python run.py --clear-cache             # force re-download
================================================================
"""

import argparse
import importlib
import sys
import os
import time
import traceback
import warnings
warnings.filterwarnings("ignore")

# Headless backend BEFORE any script imports pyplot — no GUI windows ever
# pop up, so nothing blocks waiting for a window to be closed.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
if SUITE_DIR not in sys.path:
    sys.path.insert(0, SUITE_DIR)

REPORTS_DIR = os.path.join(SUITE_DIR, "reports")

SCRIPTS = {
    # ── Analysis ────────────────────────────────────────────
    1:  ("1_density_distribution_analysis", "Density Distribution & Portfolio Optimisation"),
    2:  ("2_extreme_value_analysis",         "Extreme Value Theory (Block Maxima + POT)"),
    3:  ("3_montecarlo_options_pricing",     "Monte Carlo Options (top 8 by vol)"),
    # ── Models ──────────────────────────────────────────────
    9:  ("9_volatility_models",              "GARCH Family Volatility Models"),
    # ── Signals & Trading ───────────────────────────────────
    12: ("12_advanced_options",              "Advanced Options (CRR + Exotics + IV Surface)"),
    13: ("13_portfolio_optimizer",           "Portfolio Optimisation (Markowitz + Risk Parity)"),
    14: ("14_quantile_trading",             "Quantile Trading (Regime-Gated)"),
    15: ("15_regime_detection",             "Market Regime Detection (HMM)"),
    16: ("16_backtesting",                  "Backtesting Engine (4 Strategies)"),
    17: ("17_trade_planner",                "Trade Planner (Leverage + SL/TP)"),
    18: ("18_copula_risk",                 "Copula Correlation Risk Monitor"),
    29: ("29_live_accuracy_check",         "Live Accuracy Check (Paper-Trade + Prediction Log)"),
    20: ("20_adaptive_forecast",           "Adaptive Forecast (Observer + MRAC)"),
    21: ("21_optimal_control",             "Optimal Position Control (LQR/H-inf)"),
    23: ("23_ml_alpha_model",              "ML Alpha Model (Ridge/Lasso, Bias-Variance Validation)"),
    24: ("24_classification_signals",      "Classification Trading Signals (Perceptron/SVM/MLP)"),
    25: ("25_ml_forecast_signal",          "ML Forecast Signal (Confidence-Gated Logistic Regression)"),
    26: ("26_walkforward_validation",      "Walk-Forward Forecast Validation (Naive/HistMean/EWMA/AR1/Ensemble/Kalman)"),
    27: ("27_momentum_screener",           "Momentum Breakout / Exhaustion Screener (Hyperliquid, wide universe)"),
    28: ("28_pairs_trading",               "Statistical Arbitrage (Cointegration + Pairs Trading)"),
    # ── Bayesian / Statistical Methods ───────────────────────
    31: ("31_bayesian_fundamentals",       "Bayesian Fundamentals (NIG Conjugate, Bayes Factors, Decision Theory)"),
    32: ("32_bayesian_regression_classification", "Approximate Inference I (Laplace: Bayesian Linear/Logistic Regression)"),
    33: ("33_variational_and_ep",          "Approximate Inference II (Variational Bayes GMM + Expectation Propagation)"),
    34: ("34_mcmc_bayesian_garch",         "Sampling Methods I (Bayesian GARCH via Metropolis-Hastings)"),
    35: ("35_importance_rejection_sampling", "Sampling Methods II (Importance Sampling + Rejection Sampling)"),
    36: ("36_mixture_models_regime",       "Parametric Models (EM Gaussian Mixture vs HMM Regimes)"),
    37: ("37_gaussian_processes",          "Nonparametric Models (Gaussian Process Regression + Classification)"),
    38: ("38_bayesian_optimization_tuning", "Bayesian Numerical Analysis (BayesOpt Conf-Threshold Tuning + Quadrature)"),
    39: ("39_estimator_performance_crlb",  "Estimation Theory I (Estimator Performance, Sufficiency, CRLB)"),
    40: ("40_mle_linear_models",           "Estimation Theory II (OLS as MLE, Fisher Info vs Bootstrap)"),
    41: ("41_noninformative_priors_asymptotics", "Estimation Theory III (Jeffreys Prior, Bernstein-von Mises)"),
    42: ("42_kalman_smoother",             "Estimation Theory V (Kalman RTS Smoother, in-sample only)"),
    43: ("43_bayesian_detection_chernoff", "Detection Theory I (Bayesian Detection + Chernoff Bound)"),
    44: ("44_multiple_hypothesis_testing", "Detection Theory II (Bonferroni + Benjamini-Hochberg FDR)"),
    45: ("45_neyman_pearson_detection",    "Detection Theory III (Neyman-Pearson + GLRT)"),
    46: ("46_critical_transition_indicators", "Critical Transition Indicators (Early-Warning Signals, hourly)"),
    47: ("47_lppl_bubble_model",           "Log-Periodic Power Law Bubble/Crash Model"),
}
# 4, 8, 11, 19, 22 retired -- superseded by 20/26 (forecasting), 14 (quantile
# signals), 2/3 (risk), 13/20/21/23/24 (the optimizers 19 generalized), and 21
# (which already runs Kalman-forecast-into-LQR in production, what 22 proved).
# Moved to _deprecated/, not deleted -- see that folder if anything is missing.

# These need >=2 assets to build a covariance/correlation matrix — with a
# single ticker they fall back to the full universe instead of erroring.
MULTI_ASSET_SCRIPTS = {13, 18, 28}

# Trade Planner (17) synthesizes regime/vol/forecast/ML/LQR/copula outputs
# from across the rest of the suite into one actionable sheet, and Live
# Accuracy Check (29) in turn consumes Trade Planner's plans — both read
# more naturally as the report's closing synthesis than sorted-by-id would
# place them. Doesn't change correctness (each script calls the others'
# functions directly rather than depending on run.py's execution order),
# just where they land in the default full-run PDF.
REPORT_TAIL = [17, 29]


def default_report_order():
    ids = sorted(SCRIPTS.keys())
    tail = [sid for sid in REPORT_TAIL if sid in ids]
    body = [sid for sid in ids if sid not in tail]
    return body + tail


def apply_overrides(tickers=None, days=None):
    import data_loader as dl
    if tickers:
        # SYMBOLS is a DYNAMIC daily snapshot (today's top gainers/volume) --
        # a requested ticker not in today's snapshot isn't necessarily wrong,
        # it just didn't make today's cut. Fall back to COMMON_SYMBOLS (the
        # fixed majors list) before giving up, so e.g. --tickers ZEC still
        # resolves on a day ZEC isn't in the top 10 by gainers/volume.
        lookup = {**dl.COMMON_SYMBOLS, **dl.SYMBOLS}
        ov = {t: lookup[t] for t in tickers if t in lookup}
        if not ov:
            print(f"[run.py] Warning: none of {tickers} in SYMBOLS or COMMON_SYMBOLS — using full universe.")
        else:
            dl.SYMBOLS = ov
            print(f"[run.py] Tickers overridden: {list(dl.SYMBOLS.keys())}")
    if days:
        dl.LOOKBACK_DAYS = int(days)
        print(f"[run.py] Lookback overridden: {dl.LOOKBACK_DAYS} days")


def warm_cache(days, symbols=None):
    """Pre-fetch assets once so scripts reuse cached parquet files."""
    import data_loader as dl
    days = days or dl.LOOKBACK_DAYS
    label = f"{len(symbols)} assets" if symbols is not None else "all assets"
    print(f"\n[run.py] Pre-fetching {label} (lookback={days}d) …")
    t0 = time.time()
    assets = dl.load_all_assets(symbols=symbols, period_days=days, verbose=True)
    print(f"[run.py] Cache warm in {time.time()-t0:.1f}s  "
          f"({len(assets)} assets)\n")


def patch_no_plots():
    """Replace plt.show() with a no-op (Agg backend already makes it inert)."""
    plt.show = lambda *a, **kw: None


def _save_open_figures(pdf: PdfPages, sid: int, description: str):
    """Add a title page + every open figure for this script to the PDF, then close them."""
    fignums = plt.get_fignums()
    if not fignums:
        return 0
    title_fig = plt.figure(figsize=(11, 1.2))
    title_fig.text(0.5, 0.5, f"Script {sid}: {description}",
                    ha="center", va="center", fontsize=16, weight="bold")
    pdf.savefig(title_fig)
    plt.close(title_fig)
    for num in fignums:
        fig = plt.figure(num)
        pdf.savefig(fig)
    plt.close("all")
    return len(fignums)


def run_scripts(script_ids: list, pdf_path: str, full_symbols: dict = None):
    import data_loader as dl

    times  = {}
    errors = {}
    fig_counts = {}

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for i, sid in enumerate(script_ids, 1):
            if sid not in SCRIPTS:
                print(f"[run.py] Unknown id {sid} — skipping.")
                continue
            module_name, description = SCRIPTS[sid]
            print(f"\n{'='*70}")
            print(f"  [{i}/{len(script_ids)}]  Script {sid}: {description}")
            print(f"{'='*70}")

            # Multi-asset scripts need >=2 tickers — fall back to the full
            # universe just for this script if we're running a single ticker.
            narrow_symbols = None
            if full_symbols and sid in MULTI_ASSET_SCRIPTS and len(dl.SYMBOLS) < 2:
                narrow_symbols = dl.SYMBOLS
                dl.SYMBOLS = full_symbols
                print(f"[run.py] Only {len(narrow_symbols)} ticker active — "
                      f"running Script {sid} against the full universe "
                      f"({len(full_symbols)} tickers) instead.")

            t0 = time.time()
            try:
                # Reload so overrides are respected on each call
                if module_name in sys.modules:
                    del sys.modules[module_name]
                mod = importlib.import_module(module_name)
                mod.main()
            except Exception:
                tb = traceback.format_exc()
                errors[sid] = tb
                print(f"\n[run.py] [ERROR] Script {sid} raised an exception:\n{tb}")
            finally:
                fig_counts[sid] = _save_open_figures(pdf, sid, description)
                if narrow_symbols is not None:
                    dl.SYMBOLS = narrow_symbols
            elapsed = time.time() - t0
            times[sid] = elapsed
            print(f"\n[run.py] Script {sid} finished in {elapsed:.1f}s "
                  f"({fig_counts[sid]} figure(s) saved to PDF)")

    # ── Final table ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RUN SUMMARY")
    print("=" * 70)
    for sid, t in times.items():
        status = "[ERROR]" if sid in errors else "[OK]"
        name   = SCRIPTS[sid][1]
        print(f"  {sid:>2}. {status}  {name:<52}  {t:>6.1f}s")
    print(f"\n  Total elapsed: {sum(times.values()):.1f}s")
    if errors:
        print(f"\n  {len(errors)} script(s) had errors — see output above.")
    print(f"\n  PDF report saved to: {pdf_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Stock Quant Suite — master runner v2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--scripts",      nargs="*", type=int, default=None, metavar="N")
    p.add_argument("--list",         action="store_true")
    p.add_argument("--tickers",      nargs="*", default=None, metavar="T")
    p.add_argument("--days",         type=int,  default=None, metavar="N")
    p.add_argument("--no-plots",     action="store_true",
                   help="No-op (kept for compatibility) — runs are always headless now.")
    p.add_argument("--clear-cache",  action="store_true",
                   help="Delete cached parquet files before running.")
    p.add_argument("--pdf",          type=str, default=None, metavar="PATH",
                   help="Output path for the combined PDF report "
                        "(default: reports/report_<timestamp>.pdf)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("\nAvailable scripts:")
        for sid, (_, desc) in SCRIPTS.items():
            print(f"  {sid:>2}. {desc}")
        return

    import data_loader as dl

    full_symbols = dict(dl.SYMBOLS)  # captured before any --tickers override

    if args.clear_cache:
        dl.clear_cache()

    apply_overrides(args.tickers, args.days)
    patch_no_plots()

    # Pre-warm cache once (all scripts will hit parquet, not Yahoo)
    warm_cache(args.days or dl.LOOKBACK_DAYS)

    single_ticker_mode = len(dl.SYMBOLS) < 2 and len(dl.SYMBOLS) < len(full_symbols)
    if single_ticker_mode:
        print(f"[run.py] Only 1 ticker active — also pre-fetching the full "
              f"universe for multi-asset scripts {sorted(MULTI_ASSET_SCRIPTS)}.")
        warm_cache(args.days or dl.LOOKBACK_DAYS, symbols=full_symbols)

    dl._QUIET = True

    ids = sorted(args.scripts) if args.scripts else default_report_order()

    pdf_path = args.pdf or os.path.join(
        REPORTS_DIR, f"report_{time.strftime('%Y%m%d_%H%M%S')}.pdf")
    run_scripts(ids, pdf_path, full_symbols=full_symbols if single_ticker_mode else None)


if __name__ == "__main__":
    main()
