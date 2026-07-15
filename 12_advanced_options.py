"""
================================================================
Script 12 — Advanced Options Pricing & Analysis
================================================================
CRR binomial tree (European + American), Black-Scholes Greeks,
exotic options (barrier, Asian MC, digital), implied volatility
surface, and delta-hedged P&L analysis.

Adapted from: options_analysis.py (crypto quant suite)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

PLOT_STYLE    = "seaborn-v0_8-darkgrid"
RF            = 0.05
N_STEPS       = 150
HORIZON_DAYS  = 30
T_Y           = HORIZON_DAYS / 365.0


# ============================================================
# Black-Scholes Helpers
# ============================================================
def bs_price(S, K, T, r, sigma, flag="call"):
    if T <= 0 or sigma <= 0:
        return float(max(S - K, 0) if flag == "call" else max(K - S, 0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == "call":
        return float(S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))
    return float(K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))


def bs_greeks(S, K, T, r, sigma, flag="call"):
    if T <= 0 or sigma <= 0:
        return dict(delta=1.0 if flag == "call" else -1.0,
                    gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
    d1     = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2     = d1 - sigma * np.sqrt(T)
    phi_d1 = stats.norm.pdf(d1)
    sqrt_T = np.sqrt(T)
    gamma  = phi_d1 / (S * sigma * sqrt_T)
    vega   = S * phi_d1 * sqrt_T / 100.0
    if flag == "call":
        delta = float(stats.norm.cdf(d1))
        theta = (-(S * phi_d1 * sigma) / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * stats.norm.cdf(d2)) / 365.0
        rho   = K * T * np.exp(-r * T) * stats.norm.cdf(d2) / 100.0
    else:
        delta = float(stats.norm.cdf(d1) - 1)
        theta = (-(S * phi_d1 * sigma) / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * stats.norm.cdf(-d2)) / 365.0
        rho   = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2) / 100.0
    return dict(delta=float(delta), gamma=float(gamma),
                theta=float(theta), vega=float(vega), rho=float(rho))


def implied_vol(market_price, S, K, T, r, flag="call"):
    if T <= 0 or market_price <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if flag == "call" else max(K - S, 0)
    if market_price <= intrinsic + 1e-8:
        return np.nan
    try:
        return float(optimize.brentq(
            lambda s: bs_price(S, K, T, r, s, flag) - market_price,
            1e-4, 10.0, xtol=1e-6, maxiter=200))
    except Exception:
        return np.nan


# ============================================================
# CRR Binomial Tree
# ============================================================
def _crr_price_only(S, K, T, r, sigma, n_steps, flag, style):
    if T <= 0 or sigma <= 0:
        return float(max(S - K, 0) if flag == "call" else max(K - S, 0))
    dt   = T / n_steps
    u    = np.exp(sigma * np.sqrt(dt))
    d    = 1.0 / u
    disc = np.exp(-r * dt)
    p    = float(np.clip((np.exp(r * dt) - d) / (u - d), 0.0, 1.0))

    j  = np.arange(n_steps + 1)
    ST = S * (u ** (n_steps - j)) * (d ** j)
    V  = np.maximum(ST - K, 0.0) if flag == "call" else np.maximum(K - ST, 0.0)

    for i in range(n_steps - 1, -1, -1):
        V_hold = disc * (p * V[:-1] + (1.0 - p) * V[1:])
        if style == "american":
            ji  = np.arange(i + 1)
            S_i = S * (u ** (i - ji)) * (d ** ji)
            ex  = (np.maximum(S_i - K, 0.0) if flag == "call"
                   else np.maximum(K - S_i, 0.0))
            V = np.maximum(V_hold, ex)
        else:
            V = V_hold
    return float(V[0])


def crr_tree(S, K, T, r, sigma, n_steps=150, flag="call", style="european"):
    price = _crr_price_only(S, K, T, r, sigma, n_steps, flag, style)
    eps_S = max(S * 0.01, 1e-4)
    p_up  = _crr_price_only(S + eps_S, K, T, r, sigma, n_steps, flag, style)
    p_dn  = _crr_price_only(S - eps_S, K, T, r, sigma, n_steps, flag, style)
    delta = (p_up - p_dn) / (2.0 * eps_S)
    gamma = (p_up - 2.0 * price + p_dn) / (eps_S ** 2)

    eps_t = max(T * 0.01, 1.0 / 365.0)
    p_t   = _crr_price_only(S, K, max(T - eps_t, 1e-4), r, sigma, n_steps, flag, style)
    theta = (p_t - price) / eps_t / 365.0

    eps_v = 0.01
    p_vup = _crr_price_only(S, K, T, r, sigma + eps_v, n_steps, flag, style)
    p_vdn = _crr_price_only(S, K, T, r, max(sigma - eps_v, 1e-4), n_steps, flag, style)
    vega  = (p_vup - p_vdn) / (2.0 * eps_v) / 100.0

    return dict(price=price, delta=float(delta), gamma=float(gamma),
                theta=float(theta), vega=float(vega))


# ============================================================
# Exotic Options
# ============================================================
def price_barrier(S, K, H, T, r, sigma, barrier_type="down-and-out", flag="call"):
    if T <= 0 or sigma <= 0:
        return 0.0
    mu  = r - 0.5 * sigma**2
    lam = (mu + sigma**2) / sigma**2
    sqT = sigma * np.sqrt(T)
    x1  = np.log(S / K)              / sqT + lam * sqT
    y1  = np.log(H**2 / (S * K))     / sqT + lam * sqT
    phi = 1 if flag == "call" else -1

    def _v(x):
        return phi * (S * stats.norm.cdf(phi * x)
                      - K * np.exp(-r * T) * stats.norm.cdf(phi * x - phi * sqT))

    def _vH(y):
        return phi * ((H / S) ** (2 * lam)
                      * (S * stats.norm.cdf(phi * y)
                         - K * np.exp(-r * T) * stats.norm.cdf(phi * y - phi * sqT)))

    A, C = _v(x1), _vH(y1)
    if barrier_type == "down-and-out":
        return float(max(A - C, 0)) if S > H else 0.0
    elif barrier_type == "down-and-in":
        return float(max(C, 0)) if S > H else float(max(A, 0))
    elif barrier_type == "up-and-out":
        return float(max(A - C, 0)) if S < H else 0.0
    elif barrier_type == "up-and-in":
        return float(max(C, 0)) if S < H else float(max(A, 0))
    return 0.0


def price_asian_mc(S, K, T, r, sigma, n_paths=20_000, n_steps=365, flag="call"):
    dt      = T / n_steps
    Z       = np.random.default_rng(42).standard_normal((n_paths, n_steps))
    log_ret = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    paths   = S * np.exp(np.cumsum(log_ret, axis=1))
    avg     = paths.mean(axis=1)
    payoff  = np.maximum(avg - K, 0.0) if flag == "call" else np.maximum(K - avg, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


def price_digital(S, K, T, r, sigma, flag="call", payout=1.0):
    if T <= 0 or sigma <= 0:
        hit = (S > K) if flag == "call" else (S < K)
        return payout * np.exp(-r * T) if hit else 0.0
    d2   = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    prob = stats.norm.cdf(d2) if flag == "call" else stats.norm.cdf(-d2)
    return float(payout * np.exp(-r * T) * prob)


# ============================================================
# IV Surface
# ============================================================
def build_iv_surface(S, sigma_atm, r=0.0, n_strikes=9, n_expiries=6):
    moneyness  = np.linspace(0.70, 1.30, n_strikes)
    expiries_y = np.array([7, 14, 30, 60, 90, 180])[:n_expiries] / 365.0
    strikes    = moneyness * S
    iv_surface = np.zeros((n_expiries, n_strikes))
    for ei, T in enumerate(expiries_y):
        for ki, km in enumerate(moneyness):
            skew  = -0.15 * (km - 1.0)
            term  = 0.02 * np.sqrt(30 / 365 / max(T, 1e-4))
            smile = 0.08 * (km - 1.0)**2
            iv_surface[ei, ki] = max(sigma_atm + skew + term + smile, 0.01)
    return dict(strikes=strikes, moneyness=moneyness,
                expiries=expiries_y, iv_surface=iv_surface)


# ============================================================
# Per-Asset Analysis
# ============================================================
def analyse_asset(asset, spot, hist_sigma):
    sigma = max(hist_sigma, 0.05)
    T     = T_Y
    K_atm = spot
    K_itm = spot * 0.90
    K_otm = spot * 1.10

    trees = {}
    for label, K, fl, sty in [
        ("Euro ATM Call", K_atm, "call", "european"),
        ("Euro ATM Put",  K_atm, "put",  "european"),
        ("Euro OTM Call", K_otm, "call", "european"),
        ("Euro ITM Call", K_itm, "call", "european"),
        ("Amer ATM Call", K_atm, "call", "american"),
        ("Amer ATM Put",  K_atm, "put",  "american"),
    ]:
        try:
            trees[label] = crr_tree(spot, K, T, RF, sigma, N_STEPS, fl, sty)
            trees[label]["K"] = K
        except Exception:
            trees[label] = {"price": np.nan, "delta": np.nan, "K": K}

    greeks = {fl: bs_greeks(spot, K_atm, T, RF, sigma, fl) for fl in ("call", "put")}

    H_down  = spot * 0.80
    H_up    = spot * 1.20
    exotics = {
        "Barrier Down-Out Call": price_barrier(spot, K_atm, H_down, T, RF, sigma, "down-and-out"),
        "Barrier Down-In Call":  price_barrier(spot, K_atm, H_down, T, RF, sigma, "down-and-in"),
        "Barrier Up-Out Call":   price_barrier(spot, K_atm, H_up,   T, RF, sigma, "up-and-out"),
        "Asian ATM Call (MC)":   price_asian_mc(spot, K_atm, T, RF, sigma),
        "Digital Call":          price_digital(spot, K_atm, T, RF, sigma, "call"),
        "Digital Put":           price_digital(spot, K_atm, T, RF, sigma, "put"),
    }

    iv_surf = build_iv_surface(spot, sigma, r=RF)

    dS      = np.linspace(-0.30, 0.30, 61)
    atm_d   = greeks["call"]["delta"]
    atm_p   = bs_price(spot, K_atm, T, RF, sigma, "call")
    new_S   = spot * (1 + dS)
    raw_pnl = np.array([bs_price(s, K_atm, T, RF, sigma, "call") - atm_p for s in new_S])
    hedge_pnl = raw_pnl - atm_d * (new_S - spot)
    denom   = max(atm_p, 1e-8)

    return dict(
        asset=asset, spot=spot, sigma=sigma,
        trees=trees, greeks=greeks, exotics=exotics,
        iv_surface=iv_surf,
        hedge=dict(dS_pct=dS * 100,
                   raw=raw_pnl / denom * 100,
                   hedged=hedge_pnl / denom * 100),
    )


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(result):
    plt.style.use(PLOT_STYLE)
    asset = result["asset"]
    fig   = plt.figure(figsize=(16, 11))
    gs    = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # [0,0] IV Surface
    ax0    = fig.add_subplot(gs[0, 0])
    ivs    = result["iv_surface"]
    iv_pct = ivs["iv_surface"] * 100
    im     = ax0.imshow(iv_pct, cmap="RdYlGn_r", aspect="auto", origin="lower")
    cb     = plt.colorbar(im, ax=ax0, pad=0.02)
    cb.set_label("IV %", fontsize=7)
    ax0.set_xticks(range(len(ivs["moneyness"])))
    ax0.set_xticklabels([f"{m:.2f}" for m in ivs["moneyness"]], rotation=45, fontsize=6)
    ax0.set_yticks(range(len(ivs["expiries"])))
    ax0.set_yticklabels([f"{int(e*365)}d" for e in ivs["expiries"]], fontsize=7)
    ax0.set_xlabel("K/S Moneyness", fontsize=7)
    for ei in range(iv_pct.shape[0]):
        for ki in range(iv_pct.shape[1]):
            ax0.text(ki, ei, f"{iv_pct[ei,ki]:.0f}", ha="center", va="center", fontsize=5.5)
    ax0.set_title(f"IV Surface  |  {asset}  |  hist_vol={result['sigma']*100:.0f}%", fontsize=9)

    # [0,1] Option Prices
    ax1 = fig.add_subplot(gs[0, 1])
    labels, values = [], []
    for key in list(result["trees"].keys()) + list(result["exotics"].keys()):
        src = result["trees"] if key in result["trees"] else result["exotics"]
        p   = src[key]["price"] if isinstance(src[key], dict) else src[key]
        if np.isfinite(p) and p > 0:
            labels.append(key)
            values.append(p)
    if values:
        y_pos = range(len(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        ax1.barh(y_pos, values, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=6.5)
        for i, v in enumerate(values):
            ax1.text(v * 1.01, i, f"${v:.2f}", va="center", fontsize=6)
    ax1.set_title(f"Option Prices  |  T={HORIZON_DAYS}d  |  CRR + Exotics", fontsize=9)
    ax1.grid(axis="x", alpha=0.3)

    # [1,0] Greek Profiles
    ax2   = fig.add_subplot(gs[1, 0])
    spot  = result["spot"]
    sigma = result["sigma"]
    S_arr = np.linspace(spot * 0.70, spot * 1.30, 80)
    x_pct = (S_arr / spot - 1) * 100
    deltas = [bs_greeks(s, spot, T_Y, RF, sigma, "call")["delta"] for s in S_arr]
    gammas = [bs_greeks(s, spot, T_Y, RF, sigma, "call")["gamma"] * spot for s in S_arr]
    vegas  = [bs_greeks(s, spot, T_Y, RF, sigma, "call")["vega"] for s in S_arr]
    ax2.plot(x_pct, deltas, lw=1.5, label="Delta")
    ax2.plot(x_pct, gammas, lw=1.2, ls="--", label="Gamma x S")
    ax2.plot(x_pct, vegas,  lw=1.2, ls="-.", label="Vega/100")
    ax2.axvline(0, color="gray", lw=0.6, ls=":")
    ax2.axhline(0, color="gray", lw=0.4)
    ax2.set_xlabel("% from spot", fontsize=7)
    ax2.legend(fontsize=7)
    ax2.set_title("Greek Profiles  |  ATM Call vs Spot", fontsize=9)
    ax2.grid(alpha=0.3)

    # [1,1] Delta-Hedged P&L
    ax3 = fig.add_subplot(gs[1, 1])
    h   = result["hedge"]
    ax3.plot(h["dS_pct"], h["raw"],    lw=1.4, label="Unhedged P&L %", color="tomato")
    ax3.plot(h["dS_pct"], h["hedged"], lw=1.4, label="Delta-Hedged P&L %", color="forestgreen")
    ax3.fill_between(h["dS_pct"], h["hedged"], 0, alpha=0.12, color="forestgreen")
    ax3.axvline(0, color="gray", lw=0.6, ls=":")
    ax3.axhline(0, color="gray", lw=0.5)
    ax3.set_xlabel("Spot move %", fontsize=7)
    ax3.set_ylabel("P&L %", fontsize=7)
    ax3.legend(fontsize=7)
    ax3.set_title("Delta-Hedged P&L  |  ATM Call", fontsize=9)
    ax3.grid(alpha=0.3)

    fig.suptitle(f"{asset} — Advanced Options Dashboard", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("ADVANCED OPTIONS PRICING & ANALYSIS")
    print("CRR Binomial Tree + Exotics + IV Surface")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        print(f"\n{'─'*50}")
        print(f"  {ticker}   S={format_price(df['close'].iloc[-1])}")
        print(f"{'─'*50}")

        spot  = float(df["close"].iloc[-1])
        sigma = float(df["log_return"].dropna().std() * np.sqrt(365))   # crypto trades 24/7

        try:
            result = analyse_asset(ticker, spot, sigma)

            # Print CRR prices
            print(f"  Vol={sigma*100:.1f}%  T={HORIZON_DAYS}d  r={RF*100:.1f}%")
            for label, res in result["trees"].items():
                p = res.get("price", np.nan)
                d = res.get("delta", np.nan)
                if np.isfinite(p):
                    print(f"    {label:<16}: ${p:>10.4f}  delta={d:+.4f}")

            # Print exotics
            print("  Exotics:")
            for label, p in result["exotics"].items():
                if np.isfinite(p):
                    print(f"    {label:<24}: ${p:.4f}")

            # Print BS Greeks
            g = result["greeks"]["call"]
            print(f"  BS Greeks (ATM Call):")
            print(f"    Delta={g['delta']:.4f}  Gamma={g['gamma']:.6f}  "
                  f"Theta={g['theta']:.4f}  Vega={g['vega']:.4f}  Rho={g['rho']:.4f}")

            atm_p = result["trees"].get("Euro ATM Call", {}).get("price", np.nan)
            amer_p = result["trees"].get("Amer ATM Put", {}).get("price", np.nan)
            euro_p = result["trees"].get("Euro ATM Put", {}).get("price", np.nan)
            early_ex = amer_p - euro_p if np.isfinite(amer_p) and np.isfinite(euro_p) else np.nan
            summary.append({
                "Ticker": ticker, "Spot": spot, "Vol%": round(sigma * 100, 1),
                "ATM_Call": round(atm_p, 2) if np.isfinite(atm_p) else np.nan,
                "ATM_Put": round(euro_p, 2) if np.isfinite(euro_p) else np.nan,
                "Put_Early_Ex": round(early_ex, 4) if np.isfinite(early_ex) else np.nan,
                "Delta": round(g["delta"], 3),
            })

            plot_dashboard(result)
        except Exception as e:
            print(f"  Error: {e}")

    if summary:
        print("\n" + "=" * 65)
        print("OPTIONS PRICING SUMMARY")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nAdvanced options analysis complete.")


if __name__ == "__main__":
    main()
