"""
================================================================
Script 34 — Sampling Methods I: Markov Chain Monte Carlo
(Bayesian GARCH(1,1) via Metropolis-Hastings)
================================================================
Covers: Markov chain Monte Carlo (random-walk Metropolis-Hastings).

Script 9 fits GARCH(1,1) by maximum likelihood (the `arch` package) and
treats the fitted (omega, alpha, beta) as known-true when forecasting
tomorrow's volatility. This script instead samples the full posterior
over those parameters via MCMC, so the volatility forecast becomes a
distribution (a credible interval over plausible tomorrow's-vol values)
rather than a single point -- parameter uncertainty that matters most
exactly when you have the least data to pin the GARCH params down.

Sampled in an unconstrained reparameterization (log_omega for positivity;
a logit-persistence split for alpha,beta >= 0 and alpha+beta < 1
stationarity) so a plain random-walk proposal never needs to reject on
constraint violations -- weak Gaussian priors are placed directly on
these unconstrained coordinates, centered near typical GARCH stylized
facts (high persistence, beta > alpha).

Posterior mean (alpha, beta) is cross-checked against Script 9's MLE fit
(same `arch` package call) as a Bayesian-frequentist consistency check:
with a full year of daily bars and weak priors, the two should roughly
agree.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

PLOT_STYLE = "seaborn-v0_8-darkgrid"
N_ITER = 3000
BURN_IN = 1000
STEP_SIZES = np.array([0.0015, 0.075, 0.125, 0.125])   # mu, log_omega, u_persist, u_frac
                                                        # (tuned for ~25-30% acceptance rate)
PRIOR_MEAN = np.array([0.0, -9.0, 2.0, -1.0])
PRIOR_STD  = np.array([0.02, 1.5, 1.5, 1.5])


# ============================================================
# Reparameterization: unconstrained -> (mu, omega, alpha, beta)
# ============================================================
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def unconstrained_to_garch(theta):
    mu, log_omega, u_persist, u_frac = theta
    omega = np.exp(log_omega)
    s = _sigmoid(u_persist)      # alpha + beta, in (0,1)
    frac = _sigmoid(u_frac)      # alpha's share of persistence, in (0,1)
    alpha = frac * s
    beta = (1 - frac) * s
    return mu, omega, alpha, beta


# ============================================================
# GARCH(1,1) log-likelihood (sequential recursion -- inherently non-
# vectorizable across t, same as any GARCH conditional-variance model)
# ============================================================
def garch_log_likelihood(returns, mu, omega, alpha, beta):
    r = returns - mu
    n = len(r)
    sigma2 = np.var(r) if omega / max(1 - alpha - beta, 1e-6) <= 0 else omega / max(1 - alpha - beta, 1e-6)
    sigma2 = max(sigma2, 1e-10)
    ll = 0.0
    for t in range(n):
        sigma2 = max(sigma2, 1e-10)
        ll += -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma2) - 0.5 * r[t] ** 2 / sigma2
        sigma2 = omega + alpha * r[t] ** 2 + beta * sigma2
    return ll


def log_posterior(theta, returns):
    mu, omega, alpha, beta = unconstrained_to_garch(theta)
    if alpha + beta >= 0.9999:
        return -np.inf
    log_prior = -0.5 * np.sum(((theta - PRIOR_MEAN) / PRIOR_STD) ** 2)
    return log_prior + garch_log_likelihood(returns, mu, omega, alpha, beta)


# ============================================================
# Random-Walk Metropolis-Hastings
# ============================================================
def mh_sample_garch_posterior(returns, n_iter=N_ITER, burn_in=BURN_IN,
                               step_sizes=STEP_SIZES, seed=42):
    rng = np.random.default_rng(seed)
    theta = PRIOR_MEAN.copy()
    theta[0] = float(np.mean(returns))
    cur_lp = log_posterior(theta, returns)

    samples = np.zeros((n_iter, 4))
    n_accept = 0
    for i in range(n_iter):
        prop = theta + rng.normal(scale=step_sizes)
        prop_lp = log_posterior(prop, returns)
        if np.log(rng.uniform()) < prop_lp - cur_lp:
            theta, cur_lp = prop, prop_lp
            n_accept += 1
        samples[i] = theta

    post = samples[burn_in:]
    garch_samples = np.array([unconstrained_to_garch(s) for s in post])
    accept_rate = n_accept / n_iter
    return {"unc_samples": post, "garch_samples": garch_samples, "accept_rate": accept_rate,
            "trace": samples}


# ============================================================
# Posterior Predictive Volatility Forecast
# ============================================================
def posterior_vol_forecast(returns, garch_samples, n_draws=500, seed=0):
    """
    For each posterior draw of (mu, omega, alpha, beta), run the GARCH
    recursion forward through the observed data to get that draw's
    filtered sigma_T^2, then take one step ahead -- a fan of plausible
    tomorrow's-vol values instead of Script 9's single point forecast.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(garch_samples), size=min(n_draws, len(garch_samples)), replace=False)
    fc_vols = []
    for i in idx:
        mu, omega, alpha, beta = garch_samples[i]
        r = returns - mu
        sigma2 = omega / max(1 - alpha - beta, 1e-6)
        for t in range(len(r)):
            sigma2 = omega + alpha * r[t] ** 2 + beta * sigma2
        fc_vols.append(np.sqrt(max(sigma2, 1e-12)))
    return np.array(fc_vols)


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, mh_result, fc_vols, mle_params=None):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    gs_samples = mh_result["garch_samples"]

    # [0,0] Trace plot of persistence (alpha+beta) -- convergence check
    ax0 = fig.add_subplot(gs[0, 0])
    persistence_trace = mh_result["trace"][:, 2]
    ax0.plot(_sigmoid(persistence_trace), color="steelblue", lw=0.6)
    ax0.axvline(len(persistence_trace) - len(gs_samples), color="crimson", lw=1.0, ls="--", label="burn-in end")
    ax0.set_xlabel("MCMC iteration"); ax0.set_ylabel("persistence (alpha+beta)")
    ax0.legend(fontsize=7)
    ax0.set_title(f"{ticker} — MH Trace (accept rate={mh_result['accept_rate']:.1%})", fontsize=9.5)
    ax0.grid(alpha=0.3)

    # [0,1] Posterior alpha vs beta scatter
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(gs_samples[:, 2], gs_samples[:, 3], s=4, alpha=0.3, color="steelblue")
    if mle_params is not None:
        ax1.scatter([mle_params[2]], [mle_params[3]], color="crimson", s=80, marker="*",
                    label="MLE (Script 9's `arch` fit)", zorder=5)
        ax1.legend(fontsize=8)
    ax1.set_xlabel("alpha"); ax1.set_ylabel("beta")
    ax1.set_title("Posterior: alpha vs beta", fontsize=10)
    ax1.grid(alpha=0.3)

    # [1,0] Posterior predictive volatility fan
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(fc_vols * 100, bins=30, color="steelblue", alpha=0.75)
    ci_lo, ci_hi = np.percentile(fc_vols * 100, [2.5, 97.5])
    ax2.axvspan(ci_lo, ci_hi, color="crimson", alpha=0.1, label="95% credible interval")
    ax2.axvline(np.median(fc_vols) * 100, color="crimson", lw=1.2, label="posterior median")
    ax2.legend(fontsize=7)
    ax2.set_xlabel("forecast per-bar vol (%)")
    ax2.set_title("Posterior Predictive Volatility Forecast", fontsize=10)
    ax2.grid(alpha=0.3)

    # [1,1] Summary table
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    mu_s, om_s, al_s, be_s = gs_samples[:, 0], gs_samples[:, 1], gs_samples[:, 2], gs_samples[:, 3]
    rows = [
        ["Accept rate", f"{mh_result['accept_rate']:.1%}"],
        ["alpha (mean, 95% CI)", f"{al_s.mean():.4f}  [{np.percentile(al_s,2.5):.4f}, {np.percentile(al_s,97.5):.4f}]"],
        ["beta (mean, 95% CI)", f"{be_s.mean():.4f}  [{np.percentile(be_s,2.5):.4f}, {np.percentile(be_s,97.5):.4f}]"],
        ["Persistence (mean)", f"{(al_s+be_s).mean():.4f}"],
        ["Vol forecast median", f"{np.median(fc_vols)*100:.3f}%"],
        ["Vol forecast 95% CI", f"[{ci_lo:.3f}%, {ci_hi:.3f}%]"],
    ]
    if mle_params is not None:
        rows.append(["MLE alpha, beta (Script 9)", f"{mle_params[2]:.4f}, {mle_params[3]:.4f}"])
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.5)

    fig.suptitle(f"{ticker} — Bayesian GARCH(1,1) via Metropolis-Hastings", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("SAMPLING METHODS I — MARKOV CHAIN MONTE CARLO")
    print("Bayesian GARCH(1,1) via Random-Walk Metropolis-Hastings")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 120:
            print(f"\n  {ticker}: skipped -- need >= 120 return obs")
            continue

        mh_result = mh_sample_garch_posterior(returns)
        gs = mh_result["garch_samples"]
        fc_vols = posterior_vol_forecast(returns, gs)

        mle_params = None
        if ARCH_AVAILABLE:
            try:
                res = arch_model(returns * 100, mean="Constant", vol="GARCH", p=1, q=1,
                                  dist="normal").fit(disp="off")
                mle_params = (res.params["mu"] / 100, res.params["omega"] / 1e4,
                              res.params["alpha[1]"], res.params["beta[1]"])
            except Exception:
                pass

        al_mean, be_mean = gs[:, 2].mean(), gs[:, 3].mean()
        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  MH accept rate: {mh_result['accept_rate']:.1%}")
        print(f"  Posterior alpha={al_mean:.4f}  beta={be_mean:.4f}  "
              f"persistence={al_mean+be_mean:.4f}")
        if mle_params:
            print(f"  MLE (arch pkg) alpha={mle_params[2]:.4f}  beta={mle_params[3]:.4f}  "
                  f"(Bayesian-frequentist agreement check)")
        ci_lo, ci_hi = np.percentile(fc_vols * 100, [2.5, 97.5])
        print(f"  Posterior predictive vol forecast: median={np.median(fc_vols)*100:.3f}%  "
              f"95% CI=[{ci_lo:.3f}%, {ci_hi:.3f}%]")

        summary.append({
            "Ticker": ticker, "AcceptRate": f"{mh_result['accept_rate']:.1%}",
            "Post_Alpha": f"{al_mean:.4f}", "Post_Beta": f"{be_mean:.4f}",
            "MLE_Alpha": f"{mle_params[2]:.4f}" if mle_params else "n/a",
            "MLE_Beta": f"{mle_params[3]:.4f}" if mle_params else "n/a",
            "VolFC_Median%": f"{np.median(fc_vols)*100:.3f}",
            "VolFC_CI%": f"[{ci_lo:.2f},{ci_hi:.2f}]",
        })

        plot_dashboard(ticker, mh_result, fc_vols, mle_params)

    if summary:
        print("\n" + "=" * 70)
        print("BAYESIAN GARCH (MCMC) SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nBayesian GARCH (MCMC) analysis complete.")


if __name__ == "__main__":
    main()
