"""
================================================================
Script 31 — Bayesian Fundamentals
================================================================
Covers: probability review / Bayes' theorem (embodied directly in every
conjugate update below, not a separate toy demo), Bayesian inference and
parameter estimation, Bayesian decision theory, Bayesian model selection.

Every other script in this suite estimates return distribution parameters
as point estimates (MLE mean/std, or best-fit-by-AIC) and treats them as
known-true once fit. This script instead carries a full posterior over
those parameters, so uncertainty in the estimate itself widens the
predictive distribution used for the trade decision -- the actual point
of doing this Bayesianly instead of with a plug-in estimate.

  1. bayesian_return_posterior -- Normal-Inverse-Gamma (NIG) conjugate
     update on daily log-returns. Posterior predictive for tomorrow's
     return is a scaled Student-t (closed form), not a plug-in Normal.
  2. bayes_factor_model_selection -- compares the same three candidate
     distributions Script 1 already tests (Normal/Student-t/Laplace) via
     approximate log-evidence instead of AIC, reporting posterior model
     probabilities. The Normal case has a closed-form conjugate marginal
     likelihood; Student-t/Laplace aren't conjugate, so their evidence
     uses a Laplace-approximated log-evidence (log-lik at the MLE minus
     0.5*k*log(n), i.e. -0.5*BIC) -- explicitly an approximation.
  3. beta_binomial_winrate -- Beta(1,1) prior on P(up day), conjugate
     update to a Beta posterior with a real credible interval, not just
     a point win-rate.
  4. bayes_decision_signal -- LONG/SHORT/FLAT call that minimizes
     posterior expected loss (via numerical integration over the full
     Student-t predictive, not just its sign), under an asymmetric loss
     that can weight downside harder than upside -- true Bayesian
     decision theory, contrasted with every other script's point-
     estimate decision rule.

Return-distribution parameters here are timeframe-agnostic (fit
directly on whatever log_return series is passed in), so this needs no
crypto-specific annualization -- unlike Script 16/25, nothing here
assumes a bar interval.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, integrate, special
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Weak, near-uninformative NIG prior: centered at 0 return, low confidence
# (kappa0=1 means the prior carries the "weight" of one pseudo-observation).
PRIOR_MU0, PRIOR_KAPPA0, PRIOR_ALPHA0, PRIOR_BETA0 = 0.0, 1.0, 1.0, 1e-4
LOSS_ASYMMETRY = 1.5   # downside weighted 1.5x upside -- a real, tunable risk-aversion knob


# ============================================================
# 1. Bayesian Parameter Estimation — Normal-Inverse-Gamma conjugate model
# ============================================================
def bayesian_return_posterior(returns, mu0=PRIOR_MU0, kappa0=PRIOR_KAPPA0,
                               alpha0=PRIOR_ALPHA0, beta0=PRIOR_BETA0):
    """
    NIG conjugate update for i.i.d. Normal(mu, sigma^2) returns. Standard
    result (e.g. Murphy, "Conjugate Bayesian analysis of the Gaussian
    distribution"): posterior predictive for a new observation is a
    Student-t, not a Normal -- because sigma^2 itself is uncertain, not
    known. That extra spread relative to a plug-in Normal IS the value of
    doing this Bayesianly.
    """
    x = np.asarray(returns)
    n = len(x)
    xbar = x.mean()
    ss = np.sum((x - xbar) ** 2)

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
    alpha_n = alpha0 + n / 2
    beta_n = beta0 + 0.5 * ss + (kappa0 * n * (xbar - mu0) ** 2) / (2 * kappa_n)

    pred_df = 2 * alpha_n
    pred_scale = float(np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n)))

    return {
        "mu_n": mu_n, "kappa_n": kappa_n, "alpha_n": alpha_n, "beta_n": beta_n,
        "pred_df": pred_df, "pred_loc": mu_n, "pred_scale": pred_scale,
        "n": n,
    }


def nig_log_evidence(returns, mu0=PRIOR_MU0, kappa0=PRIOR_KAPPA0,
                      alpha0=PRIOR_ALPHA0, beta0=PRIOR_BETA0):
    """Closed-form log marginal likelihood of the NIG-Normal model."""
    x = np.asarray(returns)
    n = len(x)
    post = bayesian_return_posterior(x, mu0, kappa0, alpha0, beta0)
    kappa_n, alpha_n, beta_n = post["kappa_n"], post["alpha_n"], post["beta_n"]
    return (special.gammaln(alpha_n) - special.gammaln(alpha0)
            + alpha0 * np.log(beta0) - alpha_n * np.log(beta_n)
            + 0.5 * (np.log(kappa0) - np.log(kappa_n))
            - (n / 2) * np.log(2 * np.pi))


# ============================================================
# 2. Bayesian Model Selection — Bayes factors (Normal/Student-t/Laplace)
# ============================================================
def bayes_factor_model_selection(returns):
    """
    Posterior model probabilities for the same three candidates Script 1
    ranks by AIC. Normal uses the closed-form NIG evidence above.
    Student-t/Laplace aren't conjugate, so their evidence is Laplace-
    approximated: log p(D|MLE) - 0.5*k*log(n) (equivalent to -0.5*BIC) --
    an explicit approximation. Equal prior model probability (1/3 each)
    assumed.
    """
    x = np.asarray(returns)
    n = len(x)
    log_ev = {}

    log_ev["Normal"] = nig_log_evidence(x)

    t_params = stats.t.fit(x)
    ll_t = np.sum(stats.t.logpdf(x, *t_params))
    log_ev["Student-t"] = ll_t - 0.5 * 3 * np.log(n)   # k=3: df, loc, scale

    lap_params = stats.laplace.fit(x)
    ll_lap = np.sum(stats.laplace.logpdf(x, *lap_params))
    log_ev["Laplace"] = ll_lap - 0.5 * 2 * np.log(n)   # k=2: loc, scale

    names = list(log_ev.keys())
    log_evs = np.array([log_ev[m] for m in names])
    log_evs -= log_evs.max()          # softmax stability
    weights = np.exp(log_evs)
    probs = weights / weights.sum()

    return {name: float(p) for name, p in zip(names, probs)}, log_ev


# ============================================================
# 3. Beta-Binomial — posterior over P(up day)
# ============================================================
def beta_binomial_winrate(returns, a0=1.0, b0=1.0):
    x = np.asarray(returns)
    n_up = int(np.sum(x > 0))
    n_down = int(np.sum(x <= 0))
    a_n, b_n = a0 + n_up, b0 + n_down
    mean = a_n / (a_n + b_n)
    ci_lo, ci_hi = stats.beta.ppf([0.025, 0.975], a_n, b_n)
    return {"a_n": a_n, "b_n": b_n, "mean": mean, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "n_up": n_up, "n_down": n_down}


# ============================================================
# 4. Bayesian Decision Theory — minimize posterior expected loss
# ============================================================
def bayes_decision_signal(post, loss_asymmetry=LOSS_ASYMMETRY):
    """
    Chooses LONG/SHORT/FLAT by minimizing posterior expected loss under
    the full Student-t predictive (numerically integrated), not just the
    sign of its mean. loss_asymmetry > 1 penalizes downside harder than
    it credits equivalent upside -- a real, principled risk-aversion knob
    a point-estimate decision rule doesn't have.
    """
    df, loc, scale = post["pred_df"], post["pred_loc"], post["pred_scale"]
    pdf = lambda r: stats.t.pdf(r, df, loc=loc, scale=scale)

    e_down, _ = integrate.quad(lambda r: max(0.0, -r) * pdf(r), -np.inf, np.inf)
    e_up, _ = integrate.quad(lambda r: max(0.0, r) * pdf(r), -np.inf, np.inf)

    loss_long = loss_asymmetry * e_down - e_up
    loss_short = loss_asymmetry * e_up - e_down
    loss_flat = 0.0

    losses = {"LONG": loss_long, "SHORT": loss_short, "FLAT": loss_flat}
    action = min(losses, key=losses.get)
    p_up = float(1 - stats.t.cdf(0, df, loc=loc, scale=scale))

    return {"action": action, "losses": losses, "p_up": p_up,
            "e_down": e_down, "e_up": e_up}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, post, model_probs, wr, decision):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    # [0,0] Posterior predictive density vs prior predictive, with 95% CI shaded
    ax0 = fig.add_subplot(gs[0, 0])
    df, loc, scale = post["pred_df"], post["pred_loc"], post["pred_scale"]
    r = np.linspace(loc - 6 * scale, loc + 6 * scale, 400)
    ax0.plot(r, stats.t.pdf(r, df, loc=loc, scale=scale), color="crimson", lw=1.8,
              label=f"Posterior predictive (t, df={df:.0f})")
    lo, hi = stats.t.ppf([0.025, 0.975], df, loc=loc, scale=scale)
    mask = (r >= lo) & (r <= hi)
    ax0.fill_between(r[mask], 0, stats.t.pdf(r[mask], df, loc=loc, scale=scale),
                       color="crimson", alpha=0.2, label="95% credible interval")
    ax0.axvline(0, color="gray", lw=0.8, ls="--")
    ax0.set_xlabel("Next-bar log-return"); ax0.set_ylabel("density")
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Posterior Predictive (NIG conjugate)", fontsize=10)
    ax0.grid(alpha=0.3)

    # [0,1] Bayes-factor model probabilities
    ax1 = fig.add_subplot(gs[0, 1])
    names = list(model_probs.keys())
    vals = [model_probs[n] for n in names]
    colors = ["steelblue", "darkorange", "forestgreen"]
    ax1.bar(names, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Posterior model probability")
    ax1.set_title("Bayes-Factor Model Selection", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # [1,0] Beta-Binomial posterior vs flat prior
    ax2 = fig.add_subplot(gs[1, 0])
    p = np.linspace(0.001, 0.999, 300)
    ax2.plot(p, stats.beta.pdf(p, 1, 1), color="gray", lw=1.0, ls="--", label="Prior Beta(1,1)")
    ax2.plot(p, stats.beta.pdf(p, wr["a_n"], wr["b_n"]), color="steelblue", lw=1.8,
              label=f"Posterior Beta({wr['a_n']},{wr['b_n']})")
    ax2.axvline(wr["mean"], color="steelblue", lw=1.0, ls=":")
    ax2.axvspan(wr["ci_lo"], wr["ci_hi"], color="steelblue", alpha=0.15)
    ax2.set_xlabel("P(up bar)"); ax2.set_ylabel("density")
    ax2.legend(fontsize=8)
    ax2.set_title(f"Win-Rate Posterior — mean={wr['mean']:.1%}  "
                  f"95% CI=[{wr['ci_lo']:.1%}, {wr['ci_hi']:.1%}]", fontsize=9.5)
    ax2.grid(alpha=0.3)

    # [1,1] Decision summary panel
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows = [
        ["P(next return > 0)", f"{decision['p_up']:.1%}"],
        ["E[downside | negative]", f"{decision['e_down']:.5f}"],
        ["E[upside | positive]", f"{decision['e_up']:.5f}"],
        ["Loss(LONG)", f"{decision['losses']['LONG']:.5f}"],
        ["Loss(SHORT)", f"{decision['losses']['SHORT']:.5f}"],
        ["Loss(FLAT)", f"{decision['losses']['FLAT']:.5f}"],
        ["Bayes decision", decision["action"]],
    ]
    table = ax3.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax3.set_title("Bayesian Decision Theory", fontsize=10, pad=15)

    fig.suptitle(f"{ticker} — Bayesian Fundamentals", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("BAYESIAN FUNDAMENTALS")
    print("Conjugate Estimation | Bayes-Factor Model Selection | Decision Theory")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 60:
            print(f"\n  {ticker}: skipped -- need >= 60 return obs")
            continue

        post = bayesian_return_posterior(returns)
        model_probs, log_ev = bayes_factor_model_selection(returns)
        wr = beta_binomial_winrate(returns)
        decision = bayes_decision_signal(post)

        best_bayes = max(model_probs, key=model_probs.get)

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}   (n={post['n']})")
        print(f"{'─'*55}")
        print(f"  Posterior predictive: t(df={post['pred_df']:.0f}, "
              f"loc={post['pred_loc']:+.5f}, scale={post['pred_scale']:.5f})")
        print(f"  Bayes-factor model probs: " +
              "  ".join(f"{k}={v:.1%}" for k, v in model_probs.items()) +
              f"  -> best={best_bayes}")
        print(f"  Win-rate posterior: mean={wr['mean']:.1%}  "
              f"95% CI=[{wr['ci_lo']:.1%}, {wr['ci_hi']:.1%}]  "
              f"({wr['n_up']} up / {wr['n_down']} down)")
        print(f"  Decision: {decision['action']}  (P(up)={decision['p_up']:.1%}, "
              f"Loss[L/S/F]={decision['losses']['LONG']:.5f}/"
              f"{decision['losses']['SHORT']:.5f}/{decision['losses']['FLAT']:.5f})")

        summary.append({
            "Ticker": ticker, "N": post["n"],
            "PredMean": f"{post['pred_loc']:+.5f}", "PredScale": f"{post['pred_scale']:.5f}",
            "BestBayesModel": best_bayes,
            "P_Normal": f"{model_probs['Normal']:.1%}",
            "P_StudentT": f"{model_probs['Student-t']:.1%}",
            "P_Laplace": f"{model_probs['Laplace']:.1%}",
            "WinRate": f"{wr['mean']:.1%}", "WinRate_CI": f"[{wr['ci_lo']:.1%},{wr['ci_hi']:.1%}]",
            "Decision": decision["action"],
        })

        plot_dashboard(ticker, post, model_probs, wr, decision)

    if summary:
        print("\n" + "=" * 70)
        print("BAYESIAN FUNDAMENTALS SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nBayesian fundamentals analysis complete.")


if __name__ == "__main__":
    main()
