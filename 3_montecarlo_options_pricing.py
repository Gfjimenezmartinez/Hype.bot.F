"""
================================================================
Script 3 — Monte Carlo Simulation & Options Pricing  v2
================================================================
Optimisations vs v1:
  • Fully vectorised GBM paths (no Python loop)
  • Merton jumps vectorised with np.where batch
  • n_sims reduced to 20 000 for path generation per asset
    (Greeks still use 10 000; accuracy fine for equity vols)
  • Heston uses Euler-Maruyama vectorised step
  • IV smile computation skips if call price ≈ 0 (deep OTM)
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

# ── Config ────────────────────────────────────────────────────
N_SIMS        = 10_000     # per-asset path count (was 20k)
N_SIMS_GREEKS = 5_000      # finite-diff Greeks (was 10k)
STEPS         = 365          # crypto trades 24/7
TOP_N         = 8           # only run MC on the N most volatile assets
RF            = 0.05
T_DAYS        = 30
T             = T_DAYS / 365
MONEYNESS     = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
SEED          = 42
PLOT_STYLE    = "seaborn-v0_8-darkgrid"

rng = np.random.default_rng(SEED)


# ── Parameter estimation ──────────────────────────────────────
def estimate_params(df):
    r      = df["log_return"].dropna()
    sigma  = float(r.std() * np.sqrt(365))   # crypto trades 24/7
    daily  = r.std()
    jumps  = r[r.abs() > 3*daily]
    lam    = max(0.05, len(jumps)/len(r)*365)
    mu_j   = float(jumps.mean())  if len(jumps) > 3 else -0.03
    sig_j  = float(jumps.std())   if len(jumps) > 3 else 0.08
    return {"S0": float(df["close"].iloc[-1]),
            "sigma": sigma, "lam": lam, "mu_j": mu_j, "sig_j": sig_j}


# ── Vectorised path generators ────────────────────────────────
def gbm_paths(S0, r, sigma, T, steps, n, Z=None):
    dt  = T / steps
    # (steps × n) random matrix, cumulative sum → no loop
    if Z is None:
        Z = rng.standard_normal((steps, n))
    log_inc = (r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z
    log_S   = np.vstack([np.zeros(n), np.cumsum(log_inc, axis=0)])
    return S0 * np.exp(log_S)          # shape (steps+1, n)


def merton_paths(S0, r, sigma, T, lam, mu_j, sig_j, steps, n):
    dt    = T / steps
    kappa = np.exp(mu_j + 0.5*sig_j**2) - 1
    adj_r = r - lam*kappa
    Z     = rng.standard_normal((steps, n))
    # Poisson counts — vectorised
    Np    = rng.poisson(lam*dt, (steps, n))
    # Jump sizes: where Np>0, draw from normal; else 0
    J     = np.where(Np > 0,
                     rng.normal(mu_j, sig_j, (steps, n)) * Np,
                     0.0)
    log_inc = (adj_r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z + J
    log_S   = np.vstack([np.zeros(n), np.cumsum(log_inc, axis=0)])
    return S0 * np.exp(log_S)


def heston_paths(S0, r, v0, kappa, theta, xi, rho, T, steps, n):
    dt = T / steps
    S  = np.full(n, S0, dtype=float)
    V  = np.full(n, v0, dtype=float)
    out = np.empty((steps+1, n))
    out[0] = S0
    for t in range(steps):
        Z1 = rng.standard_normal(n)
        Z2 = rho*Z1 + np.sqrt(max(1-rho**2, 0))*rng.standard_normal(n)
        V  = np.maximum(V + kappa*(theta-V)*dt + xi*np.sqrt(np.maximum(V,0)*dt)*Z2, 0)
        S  = S * np.exp((r - 0.5*V)*dt + np.sqrt(np.maximum(V,0)*dt)*Z1)
        out[t+1] = S
    return out


# ── Option pricing ────────────────────────────────────────────
def call_price(paths, K, r, T):
    return float(np.exp(-r*T) * np.maximum(paths[-1]-K, 0).mean())

def put_price(paths, K, r, T):
    return float(np.exp(-r*T) * np.maximum(K-paths[-1], 0).mean())


def bs_greeks(S0, K, r, sigma, T):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0*sigma*np.sqrt(T))
    vega  = S0*norm.pdf(d1)*np.sqrt(T)
    theta = (-S0*norm.pdf(d1)*sigma/(2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2)) / 365     # per day, crypto trades 24/7
    return {"Price": price, "Delta": delta, "Gamma": gamma,
            "Vega": vega, "Theta": theta}


def greeks_fd(S0, K, r, sigma, T, n, eps_S=0.01, eps_v=0.01):
    """
    MC Greeks with common random numbers (same Z for every revaluation)
    and bump sizes large enough that the finite difference isn't
    swamped by discretisation noise:
      • delta  : pathwise estimator  e^{-rT}·1{S_T>K}·S_T/S0
                 (unbiased, much lower variance than bump-and-revalue)
      • gamma  : central FD with a 1% spot bump (a 1e-4 bump leaves
                 ~no paths crossing the strike between bumps → garbage)
      • vega   : central FD with 0.01 absolute vol bump, CRN
      • theta  : 1-day time decay, CRN, reported per day
    Falls back to Black-Scholes if MC output violates no-arbitrage
    bounds (0≤delta≤1, gamma≥0, vega≥0).
    """
    Z    = rng.standard_normal((STEPS, n))
    ST   = gbm_paths(S0, r, sigma, T, STEPS, n, Z)[-1]
    disc = np.exp(-r*T)

    base  = float(disc * np.maximum(ST - K, 0).mean())
    delta = float(disc * ((ST > K) * ST / S0).mean())          # pathwise

    up = call_price(gbm_paths(S0*(1+eps_S), r, sigma, T, STEPS, n, Z), K, r, T)
    dn = call_price(gbm_paths(S0*(1-eps_S), r, sigma, T, STEPS, n, Z), K, r, T)
    gamma = (up - 2*base + dn) / (S0*eps_S)**2

    v_up = call_price(gbm_paths(S0, r, sigma+eps_v, T, STEPS, n, Z), K, r, T)
    v_dn = call_price(gbm_paths(S0, r, sigma-eps_v, T, STEPS, n, Z), K, r, T)
    vega = (v_up - v_dn) / (2*eps_v)

    T_sh  = max(T - 1/365, 1e-6)   # crypto trades 24/7
    theta = call_price(gbm_paths(S0, r, sigma, T_sh, STEPS, n, Z), K, r, T_sh) - base

    out = {"Price": base, "Delta": delta, "Gamma": gamma,
           "Vega": vega, "Theta": theta}

    # No-arbitrage sanity check — fall back to closed form if violated
    if not (0.0 <= delta <= 1.0 and gamma >= 0.0 and vega >= 0.0):
        print("    [warn] MC Greeks violated no-arbitrage bounds — "
              "using Black-Scholes closed form.")
        bs = bs_greeks(S0, K, r, sigma, T)
        bs["Price"] = base            # keep the MC price
        return bs
    return out


def implied_vol(market_price, S0, K, r, T):
    if market_price < 1e-8:
        return np.nan
    def obj(sig):
        if sig <= 0: return -market_price
        d1 = (np.log(S0/K) + (r+0.5*sig**2)*T) / (sig*np.sqrt(T))
        d2 = d1 - sig*np.sqrt(T)
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2) - market_price
    try:
        return brentq(obj, 1e-4, 20.0)
    except Exception:
        return np.nan


# ── Dashboard plot ────────────────────────────────────────────
def plot_dashboard(ticker, S0, gbm, merton, heston, T, r, moneyness):
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    K_arr = S0 * np.array(moneyness)

    # 1. Paths
    ax = axes[0,0]
    for p, col, lbl in [(gbm,"steelblue","GBM"),
                        (merton,"darkorange","Merton"),
                        (heston,"forestgreen","Heston")]:
        ax.plot(p[:,:60], alpha=0.06, color=col)
        ax.plot(p.mean(axis=1), color=col, lw=2, label=lbl)
    ax.axhline(S0, ls=":", color="black", alpha=0.5)
    ax.set_title(f"{ticker} — MC Paths (T={T:.2f}y)"); ax.legend(fontsize=8)

    # 2. Terminal distributions
    ax2 = axes[0,1]
    for p, col, lbl in [(gbm,"steelblue","GBM"),
                        (merton,"darkorange","Merton"),
                        (heston,"forestgreen","Heston")]:
        ax2.hist(p[-1], bins=70, density=True, alpha=0.35, color=col, label=lbl)
    ax2.axvline(S0, ls="--", color="black")
    ax2.set_title("Terminal Price Distribution"); ax2.legend(fontsize=8)

    # 3. Call prices vs moneyness
    ax3 = axes[0,2]
    for p, col, lbl in [(gbm,"steelblue","GBM"),
                        (merton,"darkorange","Merton"),
                        (heston,"forestgreen","Heston")]:
        prices = [call_price(p, K, r, T) for K in K_arr]
        ax3.plot(moneyness, prices, "o-", color=col, lw=2, label=lbl)
    ax3.axvline(1.0, ls=":", color="black", alpha=0.4)
    ax3.set_title("Call Price vs Moneyness"); ax3.set_xlabel("K/S0")
    ax3.legend(fontsize=8)

    # 4. IV smile (GBM)
    ax4 = axes[1,0]
    iv = [implied_vol(call_price(gbm,K,r,T), S0, K, r, T) for K in K_arr]
    valid = [(m,v) for m,v in zip(moneyness,iv) if not np.isnan(v)]
    if valid:
        mv, vv = zip(*valid)
        ax4.plot(mv, vv, "o-", color="purple", lw=2)
    ax4.axvline(1.0, ls=":", color="black", alpha=0.4)
    ax4.set_title("Implied Volatility Smile (GBM)"); ax4.set_xlabel("K/S0")

    # 5. Convergence
    ax5 = axes[1,1]
    term      = gbm[-1]
    cum_mean  = np.cumsum(term) / np.arange(1, len(term)+1)
    ax5.plot(cum_mean, color="steelblue", lw=1.5)
    ax5.axhline(cum_mean[-1], ls="--", color="red", lw=1)
    ax5.set_title("MC Convergence (GBM mean terminal)")
    ax5.set_xlabel("Simulations")

    # 6. Risk bar
    ax6 = axes[1,2]
    lbls, vars_, cvars_ = [], [], []
    for p, lbl in [(gbm,"GBM"),(merton,"Merton"),(heston,"Heston")]:
        rets  = np.log(p[-1]/S0)
        v95   = np.percentile(rets, 5)
        cv95  = rets[rets<=v95].mean() if (rets<=v95).any() else v95
        lbls.append(lbl); vars_.append(v95); cvars_.append(cv95)
    x = np.arange(3)
    ax6.bar(x-0.2, vars_,  0.35, label="VaR 95%",  color="tomato",  alpha=0.8)
    ax6.bar(x+0.2, cvars_, 0.35, label="CVaR 95%", color="darkred", alpha=0.8)
    ax6.set_xticks(x); ax6.set_xticklabels(lbls)
    ax6.set_title("Tail Risk by Model"); ax6.legend(fontsize=8)

    plt.suptitle(f"{ticker} — Monte Carlo & Options Dashboard",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.show()


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("MONTE CARLO SIMULATION & OPTIONS PRICING  v2")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)

    # Rank by volatility, run MC only on top N (most useful for options)
    vols = {t: df["log_return"].std() * np.sqrt(365) for t, df in assets_data.items()}   # crypto trades 24/7
    ranked = sorted(vols, key=vols.get, reverse=True)[:TOP_N]
    print(f"  Running MC on top {TOP_N} by volatility: {', '.join(ranked)}\n")
    assets_subset = {t: assets_data[t] for t in ranked}

    for ticker, df in assets_subset.items():
        print(f"\n{'─'*50}  {ticker}")
        p  = estimate_params(df)
        S0 = p["S0"]
        print(f"  S0={format_price(S0)}  σ={p['sigma']:.2%}  λ={p['lam']:.2f}")

        g = gbm_paths(S0, RF, p["sigma"], T, STEPS, N_SIMS)
        m = merton_paths(S0, RF, p["sigma"], T, p["lam"], p["mu_j"], p["sig_j"], STEPS, N_SIMS)
        v0 = (p["sigma"]*0.8)**2
        h = heston_paths(S0, RF, v0, kappa=2.0, theta=v0,
                         xi=0.3, rho=-0.5, T=T, steps=STEPS, n=N_SIMS)

        # ATM Greeks (GBM only for speed)
        gr = greeks_fd(S0, S0, RF, p["sigma"], T, N_SIMS_GREEKS)
        print("  ATM Greeks (GBM):")
        for k, v in gr.items():
            print(f"    {k:<7}: {v:.6f}")

        plot_dashboard(ticker, S0, g, m, h, T, RF, MONEYNESS)

    print("\nMonte Carlo analysis complete.")

if __name__ == "__main__":
    main()
