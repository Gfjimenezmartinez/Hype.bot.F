"""
================================================================
Crypto Quant Suite — Shared Data Loader (Hyperliquid / ccxt)
================================================================
Fetches OHLCV for Hyperliquid perpetuals via ccxt, with parquet
caching so each script re-uses the same files instead of hitting
the exchange repeatedly per run.

Universe is dynamic by default: top N by 24h % gain + top N by
24h notional volume on Hyperliquid perpetuals (majors like BTC/ETH
show up naturally via the volume ranking). Pass an explicit
`symbols` dict, or override via run.py --tickers, to pin a fixed
set instead.
================================================================
"""

import os
import sys
import time
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import ccxt

# Windows' default console codepage (cp1252) can't encode the Unicode
# symbols (checkmarks, em-dashes, box-drawing) used in print statements
# throughout this suite -- without this, an error message trying to
# print e.g. a checkmark crashes with UnicodeEncodeError and masks
# whatever the ACTUAL error was. Every script imports this module first,
# so fixing stdout/stderr encoding here covers the whole suite. Wrapped
# in try/except since reconfigure() can fail on some redirected streams
# (e.g. certain CI/pipe setups) -- not worth hard-failing over.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Cache directory (same folder as this file) ─────────────────
_DIR      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
TIMEFRAME     = "1d"
LOOKBACK_DAYS = 365
N_GAINERS     = 5
N_VOLUME      = 5
MIN_QUOTE_VOL = 1_000_000   # filter out illiquid/dead perp markets
ALL_LIQUID_UNIVERSE = True  # scan every perp clearing MIN_QUOTE_VOL (~58 of
                             # 232 markets), not just today's top gainers/volume.
                             # Set False to fall back to the narrower N_GAINERS +
                             # N_VOLUME selection.
_QUIET        = False

# Well-known majors — used to resolve --tickers overrides even when a
# name isn't in today's dynamic top-gainers/top-volume snapshot. Every
# symbol below was live-verified against Hyperliquid's actual market
# list before being added here (ex.load_markets(), swap markets only).
COMMON_SYMBOLS = {
    "BTC":   "BTC/USDC:USDC",
    "ETH":   "ETH/USDC:USDC",
    "HYPE":  "HYPE/USDC:USDC",
    "PAXG":  "PAXG/USDC:USDC",
    "ZEC":   "ZEC/USDC:USDC",
    "PENGU": "PENGU/USDC:USDC",
    "XRP":   "XRP/USDC:USDC",
    "LIT":   "LIT/USDC:USDC",
    "SOL":   "SOL/USDC:USDC",
    "CRV":   "CRV/USDC:USDC",
    "SUI":   "SUI/USDC:USDC",
}

DISPLAY_NAMES = {
    "BTC":   "Bitcoin",
    "ETH":   "Ethereum",
    "HYPE":  "Hyperliquid",
    "PAXG":  "PAX Gold",
    "ZEC":   "Zcash",
    "PENGU": "Pudgy Penguins",
    "XRP":   "XRP",
    "LIT":   "Lighter",
    "SOL":   "Solana",
    "CRV":   "Curve DAO",
    "SUI":   "Sui",
}

def format_price(price: float) -> str:
    """
    Format a price with enough decimals to stay meaningful for sub-$1
    crypto assets — e.g. $0.02166 (PROMPT) rather than rounding to $0.02.
    Majors like BTC/ETH stay at the usual 2 decimals; sub-$1 tokens get
    enough decimals for ~4 significant figures (capped at 8).
    """
    ap = abs(price)
    if ap == 0:
        decimals = 2
    elif ap >= 1:
        decimals = 2
    else:
        decimals = min(int(-math.floor(math.log10(ap) + 1e-12)) + 3, 8)
    return f"${price:,.{decimals}f}"


_exchange = None


def _get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.hyperliquid()
        _exchange.load_markets()
    return _exchange


def _fetch_with_retry(fn, *args, max_retries=4, base_delay=3.0, **kwargs):
    """Retry a ccxt call with exponential backoff on ccxt.NetworkError
    (covers RateLimitExceeded/DDoSProtection/RequestTimeout/
    ExchangeNotAvailable -- all subclasses in ccxt's hierarchy). Without
    this, a single transient 429 permanently drops that ticker's data for
    the rest of the 12h cache window and every script that calls
    load_all_assets() in the meantime silently runs one ticker short --
    observed happening in practice during a full run.py pass across ~37
    scripts' worth of sequential fetches."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ccxt.NetworkError as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(base_delay * attempt)
    raise last_exc


# ============================================================
# CACHE HELPERS
# ============================================================

def _cache_path(symbol: str, period_days: int) -> str:
    key = hashlib.md5(f"{symbol}_{period_days}".encode()).hexdigest()[:10]
    safe = symbol.replace("/", "_").replace(":", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{key}.parquet")


def _cache_valid(path: str, max_age_hours: float = 12.0) -> bool:
    if not os.path.exists(path):
        return False
    age_h = (pd.Timestamp.now().timestamp() - os.path.getmtime(path)) / 3600
    return age_h < max_age_hours


def clear_cache():
    """Delete all cached parquet files."""
    for f in os.listdir(CACHE_DIR):
        if f.endswith(".parquet"):
            os.remove(os.path.join(CACHE_DIR, f))
    print("Cache cleared.")


_cache_write_warned = False

def _try_cache_write(df: "pd.DataFrame", cpath: str):
    """
    Best-effort cache write. Failures are non-fatal (fresh data still
    gets returned), but silently swallowing every failure forever hid a
    real bug for a long time: without pyarrow/fastparquet installed,
    every write fails and every run refetches from Hyperliquid with zero
    caching benefit, with no visible sign anything was wrong. Warn once
    per process instead of staying silent forever.
    """
    global _cache_write_warned
    try:
        df.to_parquet(cpath)
    except Exception as e:
        if not _cache_write_warned:
            print(f"[data_loader] WARNING: parquet cache write failed ({e}) — "
                  f"every run will refetch from Hyperliquid with no caching. "
                  f"Run: pip install pyarrow")
            _cache_write_warned = True


# ============================================================
# DYNAMIC UNIVERSE — top gainers + top volume
# ============================================================

_ticker_cache = {"data": None, "ts": 0.0}


def get_all_tickers(ttl: float = 90.0) -> dict:
    """
    Bulk fetch_tickers() for every native Hyperliquid crypto perp (swap
    markets), cached in-memory for `ttl` seconds. This single call is the
    slow part of any universe-wide operation (~10-15s for the full
    ~450-market book) — callers that need a full-universe snapshot
    (get_top_universe, the momentum screener) should go through this
    instead of calling ex.fetch_tickers() directly, so a burst of calls
    within the TTL window is nearly free.

    Each ticker's "info" dict also carries "funding" (current funding
    rate) and "openInterest" (base units) straight from Hyperliquid, at
    no extra request cost.
    """
    now = time.time()
    if _ticker_cache["data"] is not None and (now - _ticker_cache["ts"]) < ttl:
        return _ticker_cache["data"]

    ex = _get_exchange()
    # Hyperliquid also lists tokenized stocks, indices, forex, and
    # pre-IPO equity under hyphenated base symbols across several prefixes
    # (XYZ-TSLA, FLX-NVDA, VNTL-SPACEX, KM-EUR, MKTS-US500, CASH-BABA,
    # PARA-TOTAL2, ...) — every genuine native crypto perp is a plain
    # ticker with no hyphen, so filtering on that keeps this crypto-only
    # without having to enumerate (and keep re-discovering) every prefix.
    swap_symbols = [s for s, m in ex.markets.items()
                    if m.get("swap") and "-" not in s.split("/")[0]]
    tickers = _fetch_with_retry(ex.fetch_tickers, swap_symbols)
    _ticker_cache["data"] = tickers
    _ticker_cache["ts"] = now
    return tickers


def get_top_universe(n_gainers: int = N_GAINERS, n_volume: int = N_VOLUME,
                      min_quote_vol: float = MIN_QUOTE_VOL, all_liquid: bool = ALL_LIQUID_UNIVERSE) -> dict:
    """
    Rank Hyperliquid perpetuals by 24h % change (previousClose -> last)
    and by 24h notional volume (quoteVolume).

    If all_liquid (default, see ALL_LIQUID_UNIVERSE below): returns every
    perp clearing min_quote_vol -- the full liquid tradeable universe
    (~58 markets as of this writing out of 232 total; the other ~174
    don't clear the $1M/24h liquidity bar and are excluded on purpose --
    thin books mean wide spreads, poor fills, and easier manipulation,
    not a real opportunity).

    Otherwise: top n_gainers + top n_volume, deduped and backfilled from
    the volume ranking to reach n_gainers+n_volume total when there's
    overlap (the original, narrower default).

    Returns dict {display_name: exchange_symbol}.
    """
    tickers = get_all_tickers()

    rows = []
    for sym, t in tickers.items():
        prev = t.get("previousClose")
        last = t.get("last") or t.get("close")
        qvol = t.get("quoteVolume") or 0.0
        if not prev or not last or qvol < min_quote_vol:
            continue
        rows.append({
            "symbol": sym,
            "pct_change": (last - prev) / prev,
            "quote_volume": qvol,
        })

    if not rows:
        raise RuntimeError("No valid Hyperliquid tickers returned — check connectivity.")

    df = pd.DataFrame(rows)

    if all_liquid:
        return {row.symbol.split("/")[0]: row.symbol for row in df.itertuples()}

    gainers   = df.sort_values("pct_change", ascending=False)
    by_volume = df.sort_values("quote_volume", ascending=False)

    chosen = list(gainers.head(n_gainers)["symbol"])
    for sym in by_volume["symbol"]:
        if len(chosen) >= n_gainers + n_volume:
            break
        if sym not in chosen:
            chosen.append(sym)

    return {sym.split("/")[0]: sym for sym in chosen}


# SYMBOLS defaults to the live dynamic universe at import time. run.py's
# --tickers flag (or a direct assignment) can replace this with a fixed
# subset; load_all_assets() falls back to this module global whenever it
# isn't given an explicit symbols dict.
try:
    SYMBOLS = get_top_universe()
except Exception as e:
    print(f"[data_loader] Could not fetch live universe at import ({e}) — "
          f"falling back to common majors.")
    SYMBOLS = dict(COMMON_SYMBOLS)


# ============================================================
# DATA FETCHER
# ============================================================

def fetch_ohlcv(symbol: str, period_days: int = LOOKBACK_DAYS,
                 use_cache: bool = True) -> "pd.DataFrame":
    """
    Fetch daily OHLCV for a Hyperliquid perp (e.g. 'BTC/USDC:USDC') via
    ccxt, with parquet caching.

    Returns DataFrame with columns:
        open, high, low, close, volume,
        log_return, simple_return, volatility_20d
    Index: DatetimeIndex (date). Returns None on failure.
    """
    cpath = _cache_path(symbol, period_days)

    # ── Try cache first ──────────────────────────────────────
    if use_cache and _cache_valid(cpath):
        try:
            return pd.read_parquet(cpath)
        except Exception:
            pass  # fall through to fresh fetch

    # ── Fresh download ───────────────────────────────────────
    try:
        ex  = _get_exchange()
        raw = _fetch_with_retry(ex.fetch_ohlcv, symbol, timeframe=TIMEFRAME, limit=period_days + 30)
        if not raw or len(raw) < 30:
            print(f"  ✗ {symbol}: insufficient data")
            return None

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"
        df = df[df["close"] > 0].copy()
        df = df.iloc[-period_days:]

        # Crypto trades 24/7 — annualize with 365, not the 252-trading-day
        # convention used for equities.
        df["log_return"]     = np.log(df["close"]).diff()
        df["simple_return"]  = df["close"].pct_change()
        df["volatility_20d"] = df["log_return"].rolling(20).std() * np.sqrt(365)
        df.dropna(subset=["log_return"], inplace=True)

        # Save to cache
        _try_cache_write(df, cpath)

        return df

    except Exception as e:
        print(f"  ✗ {symbol}: {e}")
        return None


def load_all_assets(
    symbols: dict = None,
    period_days: int = None,
    min_rows: int = 60,
    use_cache: bool = True,
    verbose: bool = None,
) -> dict:
    """
    Fetch data for every symbol. Uses parquet cache (12-hour TTL)
    so repeated calls within the same session are instant.
    Returns dict {name: DataFrame}.
    """
    if symbols is None:
        symbols = SYMBOLS
    if period_days is None:
        period_days = LOOKBACK_DAYS
    if verbose is None:
        verbose = not _QUIET

    if verbose:
        print("=" * 65)
        print("FETCHING MARKET DATA  (Hyperliquid / ccxt)")
        print("=" * 65)

    assets = {}
    for name, symbol in symbols.items():
        df = fetch_ohlcv(symbol, period_days, use_cache=use_cache)
        if df is not None and len(df) >= min_rows:
            assets[name] = df
            if verbose:
                price_str = f"{format_price(df['close'].iloc[-1]):>11}"
                print(f"  ✓ {name:<10} {symbol:<18} {len(df):>3}d  close={price_str}")
        else:
            if verbose:
                rows = len(df) if df is not None else 0
                print(f"  ✗ {name:<10} {symbol:<18} only {rows} rows — skipped")

    if verbose:
        print(f"\n  Loaded {len(assets)} / {len(symbols)} assets.\n")
    return assets


# ============================================================
# INTRADAY DATA
# ============================================================

INTRADAY_INTERVALS = {"1m": 7, "5m": 60, "15m": 60, "1h": 730}
_INTERVAL_MINUTES   = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}

def fetch_intraday(symbol: str, interval: str = "5m",
                   use_cache: bool = True, limit: int = None) -> "pd.DataFrame":
    """
    Fetch intraday OHLCV via ccxt/Hyperliquid.
    Limits: 1m=7 days, 5m/15m=60 days, 1h=730 days (unless `limit` — a
    direct bar count — is given, e.g. for a screener that only needs the
    last couple hundred bars and shouldn't pay for a full day-range pull).
    Returns DataFrame with same derived columns as daily, or None.
    """
    max_days = INTRADAY_INTERVALS.get(interval, 60)
    ckey     = hashlib.md5(f"{symbol}_{interval}_{limit}_intra".encode()).hexdigest()[:10]
    safe     = symbol.replace("/", "_").replace(":", "_")
    cpath    = os.path.join(CACHE_DIR, f"{safe}_{ckey}.parquet")

    if use_cache and _cache_valid(cpath, max_age_hours=1.0):
        try:
            return pd.read_parquet(cpath)
        except Exception:
            pass

    try:
        ex = _get_exchange()
        if limit is None:
            minutes = _INTERVAL_MINUTES.get(interval, 5)
            limit = min(int(max_days * 24 * 60 / minutes), 5000)
        raw = _fetch_with_retry(ex.fetch_ohlcv, symbol, timeframe=interval, limit=limit)
        if not raw or len(raw) < 20:
            return None

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.index.name = "timestamp"
        df = df[df["close"] > 0].copy()

        df["log_return"]    = np.log(df["close"]).diff()
        df["simple_return"] = df["close"].pct_change()
        df["vwap"]          = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
        df.dropna(subset=["log_return"], inplace=True)

        _try_cache_write(df, cpath)
        return df
    except Exception as e:
        print(f"  [intraday] {symbol} {interval}: {e}")
        return None


def load_all_intraday(interval: str = "5m", symbols: dict = None,
                      verbose: bool = True) -> dict:
    """Fetch intraday data for all symbols."""
    if symbols is None:
        symbols = SYMBOLS
    assets = {}
    if verbose:
        print(f"  Fetching intraday ({interval}) data ...")
    for name, symbol in symbols.items():
        df = fetch_intraday(symbol, interval)
        if df is not None and len(df) >= 20:
            assets[name] = df
            if verbose:
                print(f"    {name:<10} {len(df):>5} bars")
    if verbose:
        print(f"  Loaded {len(assets)} / {len(symbols)} intraday.\n")
    return assets


def fetch_intraday_parallel(symbols: dict, interval: str = "15m", limit: int = 200,
                             use_cache: bool = False, max_workers: int = 20,
                             verbose: bool = True) -> dict:
    """
    Concurrent version of load_all_intraday for wide-universe scans (e.g.
    a momentum screener pulling 100-300+ symbols at once). ccxt calls are
    blocking I/O, so a thread pool cuts wall-clock time roughly by
    max_workers vs fetching one symbol at a time.

    use_cache defaults to False: screeners want the freshest bars, and
    fetch_intraday's 1h cache TTL is too stale to be useful against 5m/15m
    candles anyway.
    """
    assets = {}
    if verbose:
        print(f"  Fetching intraday ({interval}, limit={limit}) for "
              f"{len(symbols)} symbols across {max_workers} workers ...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_intraday, symbol, interval, use_cache, limit): name
            for name, symbol in symbols.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                df = fut.result()
            except Exception:
                df = None
            if df is not None and len(df) >= 20:
                assets[name] = df
    if verbose:
        print(f"  Loaded {len(assets)} / {len(symbols)} intraday in "
              f"{time.time()-t0:.1f}s.\n")
    return assets


# ============================================================
# FUNDING RATE HISTORY — perpetual-futures-specific signal (no equities
# equivalent). Hyperliquid's funding rate updates hourly and caps a
# single fetch_funding_rate_history call at ~500 records, so a full
# LOOKBACK_DAYS window needs pagination via `since` cursors.
# ============================================================
FUNDING_MAX_CALLS = 20   # safety cap: 20 calls * 500 records = ~416 days of hourly
                          # funding, comfortably covers LOOKBACK_DAYS=365


def fetch_funding_history(symbol: str, days: int = None, use_cache: bool = True) -> "pd.Series":
    """
    Paginated fetch of hourly funding rate history, resampled to one row
    per day (mean funding rate that day) to align with fetch_ohlcv's
    daily calendar. Returns a pd.Series (index=day, name="funding_rate"),
    or None on failure/no data.
    """
    days = days or LOOKBACK_DAYS
    ckey  = hashlib.md5(f"{symbol}_{days}_funding".encode()).hexdigest()[:10]
    safe  = symbol.replace("/", "_").replace(":", "_")
    cpath = os.path.join(CACHE_DIR, f"{safe}_{ckey}_funding.parquet")

    if use_cache and _cache_valid(cpath, max_age_hours=12.0):
        try:
            return pd.read_parquet(cpath)["funding_rate"]
        except Exception:
            pass

    try:
        ex = _get_exchange()
        since = int((pd.Timestamp.now().timestamp() - days * 86400) * 1000)
        records = []
        for _ in range(FUNDING_MAX_CALLS):
            batch = None
            for attempt in range(7):
                try:
                    batch = ex.fetch_funding_rate_history(symbol, since=since, limit=500)
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 6:
                        time.sleep(2.0 * (attempt + 1))   # backoff -- concurrent tickers
                        continue                            # all paginate at once, so
                    raise                                    # occasional 429s are expected
            if not batch:
                break
            records.extend(batch)
            last_ts = batch[-1]["timestamp"]
            if len(batch) < 500 or last_ts <= since:
                break   # caught up to "now" or exchange has nothing more
            since = last_ts + 1

        if not records:
            return None

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset="timestamp").set_index("timestamp")
        daily = df["fundingRate"].resample("1D").mean().rename("funding_rate")
        daily = daily.dropna()

        _try_cache_write(daily.to_frame(), cpath)
        return daily
    except Exception as e:
        print(f"  [funding] {symbol}: {e}")
        return None


def load_all_funding(symbols: dict = None, days: int = None, use_cache: bool = True,
                      max_workers: int = 4, verbose: bool = False) -> dict:
    """
    Concurrent version of fetch_funding_history for the whole universe --
    a cold sequential fetch is ~18s/ticker (pagination through hourly
    records), which would add 2-3 minutes to a 9-10 ticker run; a thread
    pool cuts that to roughly one ticker's worth of wall-clock time since
    ccxt calls are blocking I/O. Same 12h cache as fetch_funding_history,
    so only the first run after cache expiry pays this cost at all.
    Returns dict {name: pd.Series}, omitting any ticker that failed.
    """
    if symbols is None:
        symbols = SYMBOLS
    days = days or LOOKBACK_DAYS
    t0 = time.time()
    if verbose:
        print(f"  Fetching funding rate history for {len(symbols)} symbols "
              f"across {max_workers} workers ...")
    funding = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fetch_funding_history, symbol, days, use_cache): name
            for name, symbol in symbols.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                series = fut.result()
            except Exception:
                series = None
            if series is not None and len(series) > 0:
                funding[name] = series
    if verbose:
        print(f"  Loaded funding for {len(funding)}/{len(symbols)} symbols "
              f"in {time.time()-t0:.1f}s.\n")
    return funding


def attach_funding(df: "pd.DataFrame", funding: "pd.Series") -> "pd.DataFrame":
    """
    Merges a daily funding-rate series onto `df` as a new 'funding_rate'
    column, forward-filled and left-joined so rows before funding history
    began (or on days funding data is missing) get NaN rather than
    silently dropping rows -- add_features' feature columns handle NaN
    the same way every other rolling-window feature already does.
    """
    out = df.copy()
    out["funding_rate"] = funding.reindex(out.index, method="ffill")
    return out


# ============================================================
# SUMMARY STATS
# ============================================================

def calculate_asset_metrics(data_dict: dict) -> dict:
    metrics = {}
    for name, df in data_dict.items():
        r = df["log_return"].dropna()
        metrics[name] = {
            "current_price":   float(df["close"].iloc[-1]),
            "n_obs":           len(r),
            "mean_return_pct": float(r.mean() * 100),
            "volatility_pct":  float(r.std() * 100),
            "ann_vol_pct":     float(r.std() * np.sqrt(365) * 100),
            "sharpe_ratio":    float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0,
            "skewness":        float(r.skew()),
            "kurtosis":        float(r.kurtosis()),
            "min_return_pct":  float(r.min() * 100),
            "max_return_pct":  float(r.max() * 100),
        }
    return metrics


# ============================================================
# QUICK SELF-TEST
# ============================================================
if __name__ == "__main__":
    assets  = load_all_assets()
    metrics = calculate_asset_metrics(assets)
    print("\nAsset Summary")
    print("-" * 85)
    hdr = f"{'Name':<10} {'Price':>12} {'AnnVol%':>9} {'Sharpe':>7} {'Skew':>7} {'Kurt':>7}"
    print(hdr)
    print("-" * 85)
    for name, m in metrics.items():
        print(
            f"{name:<10} "
            f"{format_price(m['current_price']):>12} "
            f"{m['ann_vol_pct']:>8.1f}% "
            f"{m['sharpe_ratio']:>7.2f} "
            f"{m['skewness']:>7.3f} "
            f"{m['kurtosis']:>7.3f}"
        )
