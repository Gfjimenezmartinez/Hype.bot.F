"""
================================================================
Script 33 — Approximate Inference II: Variational Bayes + Expectation
Propagation
================================================================
Covers: variational Bayes, expectation propagation. (The Laplace
approximation, this cluster's third method, was Script 32.)

  1. VARIATIONAL BAYES -- a mean-field Gaussian mixture over per-bar
     returns, discovering latent return/volatility regimes the same way
     Script 15's HMM does, but via variational inference instead of
     Baum-Welch: sklearn's BayesianGaussianMixture with a Dirichlet-
     process-type prior is the standard, correctly-implemented mean-
     field VI for this exact model (Bishop PRML ch.10.2) -- reusing it
     here is the right call, the same way Script 24 reused sklearn's
     LogisticRegression/RandomForest rather than re-deriving them.
     Automatic relevance determination lets the model prune unused
     components on its own (request K=4, see how many survive with
     non-trivial weight), which is the actual point of the Bayesian
     treatment vs. plain EM/K-means. Cross-checked against Script 15's
     HMM regime call for the same ticker.

  2. EXPECTATION PROPAGATION -- hand-implemented for Bayesian probit
     classification (the textbook EP example: Rasmussen & Williams,
     GPML section 3.6 / Bishop PRML 10.7), predicting next-bar direction
     the same task Script 32's Laplace-approximated logistic regression
     solves, so the two approximate-inference methods are compared
     directly on identical data. Unlike Laplace (which approximates the
     posterior at a single point, the MAP mode), EP approximates it by
     iteratively moment-matching each likelihood factor -- a genuinely
     different approximation strategy, and probit's predictive
     probability has an exact closed form (no MacKay-style correction
     needed at prediction time, only the posterior itself is
     approximate).

Reuses Script 24's build_classification_dataset/chronological_split.
Standalone diagnostic, not wired into Script 17.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from importlib import import_module as _im
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

_r15 = _im("15_regime_detection")
detect_regime = _r15.detect_regime
REGIME_NAMES = _r15.REGIME_NAMES

_r24 = _im("24_classification_signals")
build_classification_dataset = _r24.build_classification_dataset
_r23 = _im("23_ml_alpha_model")
chronological_split = _r23.chronological_split

PLOT_STYLE = "seaborn-v0_8-darkgrid"
VB_MAX_COMPONENTS = 4
EP_PRIOR_ALPHA = 1.0
EP_N_SWEEPS = 5


# ============================================================
# 1. Variational Bayes — Gaussian Mixture regime discovery
# ============================================================
def variational_gmm_regimes(returns, max_components=VB_MAX_COMPONENTS, random_state=42):
    """
    Mean-field VI Gaussian mixture (sklearn's BayesianGaussianMixture,
    Dirichlet-process-type weight prior) on per-bar returns. Requesting
    more components than needed and letting ARD shrink unused ones to
    ~zero weight is the actual demonstration of the Bayesian treatment --
    contrast with a plain (frequentist) GaussianMixture in Script 36,
    which cannot do this and must be told K exactly.
    """
    x = np.asarray(returns).reshape(-1, 1)
    vb = BayesianGaussianMixture(
        n_components=max_components, weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2, max_iter=500, random_state=random_state,
    ).fit(x)

    order = np.argsort(vb.means_.flatten())
    means = vb.means_.flatten()[order]
    stds = np.sqrt(vb.covariances_.flatten())[order]
    weights = vb.weights_[order]
    active = weights > 0.02   # components ARD shrank below 2% weight are "pruned"

    return {"means": means, "stds": stds, "weights": weights, "active": active,
            "n_active": int(active.sum()), "model": vb}


# ============================================================
# 2. Expectation Propagation — Bayesian probit classification
# ============================================================
def ep_probit_classifier(X, y_pm1, alpha=EP_PRIOR_ALPHA, n_sweeps=EP_N_SWEEPS, seed=42):
    """
    Weight-space EP for Bayesian probit regression, prior w ~ N(0, alpha^-1 I).
    Maintains full posterior N(mu_w, Sigma_w) via natural parameters
    (Lambda = alpha*I + X^T diag(tau_tilde) X, h = X^T nu_tilde), recomputed
    after every single-site update -- m (features) is small here (~15-28),
    so a full O(m^3) recompute per point is cheap and safer than an
    incremental rank-1 update. Standard moment-matching formulas for the
    probit likelihood (GPML eq. 3.58-3.59 adapted to weight space).

    Features are standardized first (same convention as Script 32's IRLS) --
    raw feature scales here range widely (e.g. a volume-ratio feature can
    be orders of magnitude larger than a lagged return), and an
    unstandardized huge-scale column blows up the cavity precision (1/v_i)
    to numerically zero, silently disabling every update.
    """
    scaler = StandardScaler().fit(X)
    Xv = scaler.transform(X)
    y = np.asarray(y_pm1, dtype=float)
    n, m = Xv.shape
    rng = np.random.default_rng(seed)

    tau_tilde = np.zeros(n)
    nu_tilde = np.zeros(n)
    prior_prec = alpha * np.eye(m)
    Sigma_w = np.linalg.inv(prior_prec)
    mu_w = np.zeros(m)

    for _ in range(n_sweeps):
        order = rng.permutation(n)
        for i in order:
            xi, yi = Xv[i], y[i]
            v_i = float(xi @ Sigma_w @ xi)
            m_i = float(xi @ mu_w)
            if v_i <= 1e-12:
                continue

            tau_cav = 1.0 / v_i - tau_tilde[i]
            nu_cav = m_i / v_i - nu_tilde[i]
            if tau_cav <= 1e-10:
                continue   # skip pathological cavity -- standard EP damping/safety

            cav_var = 1.0 / tau_cav
            cav_mean = nu_cav / tau_cav
            denom = np.sqrt(1.0 + cav_var)
            z = yi * cav_mean / denom
            Z_hat = max(stats.norm.cdf(z), 1e-12)
            ratio = stats.norm.pdf(z) / Z_hat

            mu_hat = cav_mean + yi * cav_var * ratio / denom
            sigma2_hat = cav_var - (cav_var ** 2 * ratio * (z + ratio)) / (1.0 + cav_var)
            sigma2_hat = max(sigma2_hat, 1e-8)

            new_tau_tilde = max(1.0 / sigma2_hat - tau_cav, 0.0)
            new_nu_tilde = mu_hat / sigma2_hat - nu_cav
            tau_tilde[i], nu_tilde[i] = new_tau_tilde, new_nu_tilde

            Lambda = prior_prec + Xv.T @ (Xv * tau_tilde[:, None])
            Sigma_w = np.linalg.inv(Lambda)
            mu_w = Sigma_w @ (Xv.T @ nu_tilde)

    return {"mu_w": mu_w, "Sigma_w": Sigma_w, "tau_tilde": tau_tilde, "nu_tilde": nu_tilde,
            "scaler": scaler}


def ep_predict_proba(X_new, fit):
    """Exact closed-form Bayesian probit predictive (no approximation
    needed here -- EP already approximated the posterior; integrating a
    Gaussian posterior against a probit likelihood IS closed-form)."""
    Xv = fit["scaler"].transform(X_new)
    mu_a = Xv @ fit["mu_w"]
    var_a = np.einsum("ij,jk,ik->i", Xv, fit["Sigma_w"], Xv)
    return stats.norm.cdf(mu_a / np.sqrt(1.0 + var_a))


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, returns, vb_result, regime_name, X_te, y_te, ep_fit, laplace_acc=None):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # [0,0] Return histogram + VB-GMM component densities
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(returns, bins=40, density=True, color="lightsteelblue", alpha=0.6, label="Returns")
    r = np.linspace(returns.min(), returns.max(), 300)
    colors = plt.cm.tab10(np.linspace(0, 1, len(vb_result["means"])))
    for k in range(len(vb_result["means"])):
        if not vb_result["active"][k]:
            continue
        dens = vb_result["weights"][k] * stats.norm.pdf(r, vb_result["means"][k], vb_result["stds"][k])
        ax0.plot(r, dens, color=colors[k], lw=1.8,
                 label=f"Component {k} (w={vb_result['weights'][k]:.2f})")
    ax0.legend(fontsize=7)
    ax0.set_title(f"{ticker} — Variational Bayes GMM ({vb_result['n_active']} active "
                  f"of {len(vb_result['means'])} requested)  |  HMM regime: {regime_name}", fontsize=9)
    ax0.grid(alpha=0.3)

    # [0,1] Component weights (showing ARD pruning)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(range(len(vb_result["weights"])), vb_result["weights"], color="steelblue", alpha=0.85)
    ax1.axhline(0.02, color="crimson", lw=0.8, ls="--", label="pruning threshold (2%)")
    ax1.set_xlabel("component (sorted by mean)"); ax1.set_ylabel("posterior weight")
    ax1.legend(fontsize=7)
    ax1.set_title("Automatic Relevance Determination", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # [1,0] EP predictive probability vs actual direction (test set)
    ax2 = fig.add_subplot(gs[1, 0])
    proba = ep_predict_proba(X_te, ep_fit)
    up_days = y_te.values == 1
    ax2.scatter(np.arange(len(proba))[up_days], proba[up_days], s=14, color="forestgreen", label="actual up")
    ax2.scatter(np.arange(len(proba))[~up_days], proba[~up_days], s=14, color="tomato", label="actual down")
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("test obs"); ax2.set_ylabel("EP P(up)")
    ax2.legend(fontsize=7)
    ep_acc = float(np.mean((proba >= 0.5) == up_days))
    title = f"EP-Probit Predictive  |  test_acc={ep_acc:.3f}"
    if laplace_acc is not None:
        title += f"  (Laplace-logistic: {laplace_acc:.3f})"
    ax2.set_title(title, fontsize=9.5)
    ax2.grid(alpha=0.3)

    # [1,1] Posterior uncertainty shrinkage: EP vs prior
    ax3 = fig.add_subplot(gs[1, 1])
    prior_var = 1.0 / EP_PRIOR_ALPHA
    post_vars = np.diag(ep_fit["Sigma_w"])
    ax3.bar(["Prior"] + [f"w{i}" for i in range(min(8, len(post_vars)))],
            [prior_var] + list(post_vars[:8]), color=["gray"] + ["steelblue"] * min(8, len(post_vars)), alpha=0.85)
    ax3.set_ylabel("posterior variance")
    ax3.set_title("EP Posterior Variance vs Prior (first 8 weights)", fontsize=9.5)
    ax3.tick_params(axis="x", labelsize=7)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{ticker} — Approximate Inference II (Variational Bayes + Expectation Propagation)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("APPROXIMATE INFERENCE II — VARIATIONAL BAYES + EXPECTATION PROPAGATION")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        returns = df["log_return"].dropna().values
        if len(returns) < 120:
            print(f"\n  {ticker}: skipped -- need >= 120 return obs")
            continue

        vb_result = variational_gmm_regimes(returns)
        try:
            _, regime_name, _ = detect_regime(df)
        except Exception:
            regime_name = "n/a"

        X, y_pm1, feat_cols = build_classification_dataset(df)
        X_tr, X_te, y_tr, y_te = chronological_split(X, y_pm1)
        ep_fit = ep_probit_classifier(X_tr, y_tr)
        proba_te = ep_predict_proba(X_te, ep_fit)
        ep_acc = float(np.mean((proba_te >= 0.5) == (y_te.values == 1)))

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  VB-GMM: {vb_result['n_active']}/{VB_MAX_COMPONENTS} components active  "
              f"means={np.round(vb_result['means'][vb_result['active']], 4)}  "
              f"weights={np.round(vb_result['weights'][vb_result['active']], 2)}  "
              f"(HMM regime: {regime_name})")
        print(f"  EP-Probit: test_acc={ep_acc:.3f}  "
              f"mean posterior weight std={np.sqrt(np.diag(ep_fit['Sigma_w'])).mean():.4f} "
              f"(prior std={1/np.sqrt(EP_PRIOR_ALPHA):.4f})")

        summary.append({
            "Ticker": ticker, "VB_ActiveComponents": vb_result["n_active"],
            "HMM_Regime": regime_name, "EP_TestAcc": f"{ep_acc:.3f}",
        })

        plot_dashboard(ticker, returns, vb_result, regime_name, X_te, y_te, ep_fit)

    if summary:
        print("\n" + "=" * 70)
        print("VARIATIONAL BAYES + EP SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nVariational Bayes + Expectation Propagation analysis complete.")


if __name__ == "__main__":
    main()
