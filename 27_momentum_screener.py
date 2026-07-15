"""
================================================================
Script 27 — Momentum Breakout / Exhaustion Screener
================================================================
Every other script in this suite does deep analysis (GARCH, Monte
Carlo, CVXPY, ...) on a small, already-known shortlist — good for
"how risky is this position", bad for "what's about to move".

This script is the opposite trade-off: no heavy stats, the full
Hyperliquid universe filtered to liquid perps (typically 40-80 names
clearing the 24h volume floor, not just today's top 10), everything
fetched in parallel. It runs in single-digit seconds once warm and
ranks two lists:

  • EXPLOSIVE — early breakout setups: volume surge, accelerating
    momentum, a volatility squeeze breaking out, funding tailwind
    (very negative funding = shorts paying longs = squeeze fuel).
    The setup BEFORE a 50-100% move, not the move itself.
  • EXHAUSTION — parabolic-extension setups ripe for a reversal:
    extreme distance from its own mean, overbought RSI, momentum
    decelerating even as price holds near highs, fading volume,
    crowded-long funding. Candidates to short the top.

Every feature is rank-normalized (percentile within the current
scan, not a hand-tuned absolute threshold) so the score adapts to
whatever regime the whole market is in right now.

Two timeframe profiles, since a setup that plays out in 2 hours and
one that plays out over a week look nothing alike on the same bars:
  • "day"   — 15m bars, ~2 days of history. Catches a move that will
    fully round-trip within a single session.
  • "swing" — 4h bars, ~33 days of history. Catches a base/breakout
    that takes several days to play out — 15m noise would drown out
    a slow multi-day squeeze, and 4h bars still resolve one inside a
    reasonable lookback window.
The rolling-window lengths (VOL_WINDOW, RSI_WINDOW, SQUEEZE_WINDOW)
are expressed in bar counts, not time, so they automatically mean
"5h of context" on day bars and "80h of context" on swing bars
without needing separate tuning.

This is a screener, not a signal — it tells you where to point the
deeper scripts (11, 14, 16, 17), not what size to put on.
================================================================
"""

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from data_loader import get_all_tickers, fetch_intraday_parallel

# ============================================================
# Configuration
# ============================================================
PROFILES = {
    "day": {
        "label":             "DAY TRADE",
        "interval":          "15m",
        "lookback_bars":     200,          # ~50h of 15m bars
        "min_quote_vol_24h": 2_000_000,
    },
    "swing": {
        "label":             "SWING TRADE",
        "interval":          "4h",
        "lookback_bars":     200,          # ~33 days of 4h bars
        "min_quote_vol_24h": 2_000_000,
    },
}

VOL_WINDOW     = 20
RSI_WINDOW     = 14
SQUEEZE_WINDOW = 60    # trailing window the Bollinger-width percentile is measured against
TOP_N          = 15
MAX_WORKERS    = 20
PLOT_STYLE     = "seaborn-v0_8-darkgrid"


# ============================================================
# Universe
# ============================================================
def build_scan_universe(min_quote_vol: float) -> tuple[dict, dict]:
    """
    Every liquid Hyperliquid crypto perp, not just the dynamic top-10 the
    rest of the suite uses — a token that's about to explode is by
    definition not yet in a top-gainers list. Returns
    (symbols {name: exchange_symbol}, ticker_info {name: raw ticker dict}).
    """
    tickers = get_all_tickers()
    symbols, info = {}, {}
    for sym, t in tickers.items():
        qvol = t.get("quoteVolume") or 0.0
        if qvol < min_quote_vol:
            continue
        name = sym.split("/")[0]
        symbols[name] = sym
        info[name] = t
    return symbols, info


# ============================================================
# Feature Engineering (per-symbol, vectorized)
# ============================================================
def compute_rsi(close: pd.Series, period: int = RSI_WINDOW) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(name: str, df: pd.DataFrame, ticker: dict) -> dict:
    close = df["close"]
    vol   = df["volume"]
    if len(close) < SQUEEZE_WINDOW + 5:
        return None

    vol_mean = vol.rolling(VOL_WINDOW).mean()
    vol_std  = vol.rolling(VOL_WINDOW).std()
    vol_z    = float((vol.iloc[-1] - vol_mean.iloc[-1]) / vol_std.iloc[-1]) if vol_std.iloc[-1] > 0 else 0.0

    roc_fast = float(close.pct_change(3).iloc[-1])
    roc_slow = float(close.pct_change(12).iloc[-1])
    # Per-3-bar pace right now vs the per-3-bar-equivalent pace over the
    # last 12 bars — positive means the move is accelerating, negative
    # means it's losing steam even if price is still near its highs.
    accel = roc_fast - roc_slow / 4

    mid   = close.rolling(VOL_WINDOW).mean()
    std20 = close.rolling(VOL_WINDOW).std()
    bb_width  = (4 * std20 / mid).replace([np.inf, -np.inf], np.nan)
    bb_pctile = float(bb_width.rolling(SQUEEZE_WINDOW).rank(pct=True).iloc[-1])
    was_squeezed = bool((bb_width.rolling(SQUEEZE_WINDOW).rank(pct=True).iloc[-4:-1] < 0.2).any())
    upper = mid + 2 * std20
    breakout = bool(close.iloc[-1] > upper.iloc[-1])

    z_ext = float((close.iloc[-1] - mid.iloc[-1]) / std20.iloc[-1]) if std20.iloc[-1] > 0 else 0.0
    rsi   = float(compute_rsi(close).iloc[-1])
    if np.isnan(rsi):
        rsi = 50.0

    funding = float(ticker.get("info", {}).get("funding") or 0.0)
    mark_px = float(ticker.get("info", {}).get("markPx") or close.iloc[-1])
    oi_usd  = float(ticker.get("info", {}).get("openInterest") or 0.0) * mark_px

    prev_close = ticker.get("previousClose")
    last_px    = ticker.get("last") or float(close.iloc[-1])
    chg_24h = (last_px - prev_close) / prev_close if prev_close else float("nan")

    # Fractional bar-to-bar volatility — same quantity as Script 17's
    # vol["daily_vol"] (returns.std(), not the price-band std used for
    # z_ext above), so the SL/TP multiples below are on the same footing
    # as the trade planner's, just computed on whatever bar interval this
    # profile scans (15m for day, 4h for swing).
    cv = float(close.pct_change().rolling(VOL_WINDOW).std().iloc[-1])
    if not np.isfinite(cv):
        cv = 0.0

    return {
        "name": name, "price": float(close.iloc[-1]), "chg_24h": chg_24h,
        "vol_z": vol_z, "accel": accel, "z_ext": z_ext, "rsi": rsi,
        "bb_pctile": bb_pctile, "squeeze_breakout": was_squeezed and breakout,
        "funding": funding, "oi_usd": oi_usd, "cv": cv,
    }


# ============================================================
# Cross-sectional scoring — percentile-ranked, not hand-tuned
# thresholds, so the score adapts to whatever regime the whole
# market is in right now rather than a fixed magic number.
# ============================================================
def score_universe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).set_index("name")

    explosion = pd.DataFrame(index=df.index)
    explosion["vol_surge"]   = df["vol_z"].rank(pct=True)
    explosion["accel"]       = df["accel"].rank(pct=True)
    explosion["squeeze"]     = df["squeeze_breakout"].astype(float)
    explosion["funding_fuel"] = (-df["funding"]).rank(pct=True)  # very negative funding = bullish squeeze fuel
    df["explosion_score"] = (
        0.35 * explosion["vol_surge"] + 0.30 * explosion["accel"] +
        0.20 * explosion["squeeze"] + 0.15 * explosion["funding_fuel"]
    ) * 100

    exhaustion = pd.DataFrame(index=df.index)
    exhaustion["extension"] = df["z_ext"].rank(pct=True)
    exhaustion["overbought"] = df["rsi"].rank(pct=True)
    exhaustion["decel"]     = (-df["accel"]).rank(pct=True)      # losing steam while still extended
    exhaustion["crowded_long"] = df["funding"].rank(pct=True)     # very positive funding = crowded longs
    exhaustion["fading_vol"] = (-df["vol_z"]).rank(pct=True)
    df["exhaustion_score"] = (
        0.30 * exhaustion["extension"] + 0.20 * exhaustion["overbought"] +
        0.20 * exhaustion["decel"] + 0.15 * exhaustion["crowded_long"] +
        0.15 * exhaustion["fading_vol"]
    ) * 100

    return df


# ============================================================
# Display
# ============================================================
def compute_trade_levels(price: float, cv: float, direction: str, conviction: float):
    """
    Same ATR/vol-multiple SL/TP convention as Script 17's plan_trade():
    stop and target set as a multiple of current fractional volatility,
    TP multiple scaling with conviction so a stronger setup gets a wider
    target. This screener has no GARCH/Monte Carlo/quantile data to
    clamp against (that's Script 17's job) — these are first-pass levels
    to sanity-check a hit against, not a replacement for running the
    full planner before sizing a real position.
    """
    cv = max(cv, 1e-4)
    atr_mult_sl = 1.5
    atr_mult_tp = 1.5 + conviction * 1.5   # R:R ~1.0R to ~2.0R, matches Script 17
    if direction == "LONG":
        sl = price * (1 - atr_mult_sl * cv)
        tp = price * (1 + atr_mult_tp * cv)
    else:  # SHORT
        sl = price * (1 + atr_mult_sl * cv)
        tp = price * (1 - atr_mult_tp * cv)
    return sl, tp


def print_table(df: pd.DataFrame, score_col: str, title: str, direction: str):
    top = df.sort_values(score_col, ascending=False).head(TOP_N)
    print(f"\n{'='*127}\n{title}\n{'='*127}")
    print(f"{'Name':<8}{'Score':>7}{'Price':>14}{'24h%':>8}{'VolZ':>7}"
          f"{'Accel%':>9}{'RSI':>6}{'Funding%':>10}"
          f"{'Entry':>14}{'SL':>14}{'TP':>14}{'R:R':>6}")
    for name, r in top.iterrows():
        chg = f"{r['chg_24h']*100:+.1f}" if pd.notna(r["chg_24h"]) else "  n/a"
        conviction = float(r[score_col]) / 100.0
        sl, tp = compute_trade_levels(r["price"], r["cv"], direction, conviction)
        rr = abs(tp - r["price"]) / abs(r["price"] - sl) if abs(r["price"] - sl) > 1e-9 else 0.0
        print(f"{name:<8}{r[score_col]:>7.1f}{r['price']:>14,.4f}{chg:>8}"
              f"{r['vol_z']:>7.2f}{r['accel']*100:>+9.2f}{r['rsi']:>6.1f}"
              f"{r['funding']*100:>+10.4f}"
              f"{r['price']:>14,.4f}{sl:>14,.4f}{tp:>14,.4f}{rr:>6.2f}")
    return top


def plot_screener(exp_top: pd.DataFrame, exh_top: pd.DataFrame, interval: str,
                   label: str, save_path: str = None):
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax0 = axes[0]
    ax0.barh(exp_top.index[::-1], exp_top["explosion_score"][::-1], color="forestgreen", alpha=0.85)
    ax0.set_title("EXPLOSIVE — Early Breakout Candidates", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Explosion Score (0-100)")
    ax0.grid(alpha=0.3, axis="x")

    ax1 = axes[1]
    ax1.barh(exh_top.index[::-1], exh_top["exhaustion_score"][::-1], color="firebrick", alpha=0.85)
    ax1.set_title("EXHAUSTION — Short-the-Top Candidates", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Exhaustion Score (0-100)")
    ax1.grid(alpha=0.3, axis="x")

    fig.suptitle(f"Momentum Screener — {label} ({interval} bars)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# Core scan — one full pass over the universe for one profile
# ============================================================
def scan_once(profile_name: str, plot: bool = True, save_path: str = None):
    profile = PROFILES[profile_name]
    interval, lookback, min_vol, label = (
        profile["interval"], profile["lookback_bars"],
        profile["min_quote_vol_24h"], profile["label"],
    )

    t_start = time.time()
    print("=" * 65)
    print(f"MOMENTUM BREAKOUT / EXHAUSTION SCREENER — {label}")
    print(f"Interval: {interval}   Liquidity floor: ${min_vol:,.0f} 24h volume")
    print("=" * 65)

    symbols, ticker_info = build_scan_universe(min_vol)
    print(f"\n  Scan universe: {len(symbols)} liquid perpetuals "
          f"(of the full Hyperliquid crypto book)")

    intraday = fetch_intraday_parallel(symbols, interval=interval,
                                        limit=lookback, max_workers=MAX_WORKERS)

    rows = []
    for name, df in intraday.items():
        feat = compute_features(name, df, ticker_info[name])
        if feat is not None:
            rows.append(feat)

    if len(rows) < 5:
        print("\n  Not enough symbols returned usable data — aborting.")
        return None

    scored = score_universe(rows)
    print(f"\n  Scored {len(scored)} symbols in {time.time()-t_start:.1f}s total "
          f"(fetch + compute).")

    exp_top = print_table(scored, "explosion_score",
                           f"TOP EXPLOSIVE — Early Breakout Candidates ({label})",
                           direction="LONG")
    exh_top = print_table(scored, "exhaustion_score",
                           f"TOP EXHAUSTION — Short-the-Top Candidates ({label})",
                           direction="SHORT")

    if plot:
        plot_screener(exp_top, exh_top, interval, label, save_path=save_path)

    print(f"\n{label} screener complete in {time.time()-t_start:.1f}s.")
    return scored


def scan_all(profiles=None, plot: bool = True):
    """Run every requested profile back-to-back (default: all of them)."""
    profiles = profiles or list(PROFILES.keys())
    results = {name: scan_once(name, plot=plot) for name in profiles}
    print("\nReminder: this ranks SETUPS, not confirmed trades — cross-check")
    print("a name here against scripts 11/14 (signals) and 16/17 (sizing)")
    print("before acting on it.")
    return results


# ============================================================
# Watch mode — the exchange/market-list init (~20-25s) only happens
# once per process; each rescan after that is ~2-4s, so leaving this
# running is far faster than re-invoking the script from scratch.
# Day-trade setups move fast (--watch 60 is reasonable); swing setups
# don't change minute to minute, so a much longer interval (900s+)
# makes more sense there — pick per-profile, don't share one cadence.
# ============================================================
def watch(profile_name: str, interval_sec: int, iterations: int = None):
    label = PROFILES[profile_name]["label"]
    chart_path = f"27_screener_{profile_name}_latest.png"
    print(f"[watch] {label} — rescanning every {interval_sec}s — Ctrl+C to stop.")
    print(f"[watch] Chart refreshed at: {chart_path}\n")
    n = 0
    try:
        while True:
            n += 1
            print(f"\n{'#'*65}\n# {label} scan {n} — {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'#'*65}")
            t0 = time.time()
            scan_once(profile_name, plot=True, save_path=chart_path)
            if iterations and n >= iterations:
                break
            sleep_for = max(1.0, interval_sec - (time.time() - t0))
            print(f"\n[watch] Next scan in {sleep_for:.0f}s ...")
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[watch] Stopped.")


# ============================================================
# Main — plain main() (no args) is what run.py's orchestrator calls,
# and runs BOTH profiles so a single suite run covers day and swing.
# `python 27_momentum_screener.py --profile day --watch 60` runs one
# profile standalone as a live monitor instead.
# ============================================================
def main():
    scan_all(plot=True)


def parse_args():
    p = argparse.ArgumentParser(description="Momentum breakout / exhaustion screener")
    p.add_argument("--profile", choices=["day", "swing", "both"], default="both",
                   help="Timeframe profile: 15m/~2d for day trades, "
                        "4h/~33d for swing trades, or both (default).")
    p.add_argument("--watch", type=int, default=None, metavar="SECONDS",
                   help="Run continuously, rescanning every N seconds instead of once. "
                        "Requires a single --profile (day or swing), not both.")
    p.add_argument("--iterations", type=int, default=None, metavar="N",
                   help="With --watch, stop after N scans (default: run until Ctrl+C).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.watch:
        if args.profile == "both":
            raise SystemExit("--watch needs a single --profile (day or swing), not both.")
        watch(args.profile, args.watch, iterations=args.iterations)
    elif args.profile == "both":
        scan_all(plot=True)
    else:
        scan_once(args.profile, plot=True)
