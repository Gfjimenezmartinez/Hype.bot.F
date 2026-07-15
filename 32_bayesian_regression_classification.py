"""
================================================================
Script 32 — Approximate Inference I: Laplace Approximation
(Bayesian Linear Regression w/ Basis Expansion + Bayesian Logistic
Regression via Laplace Approximation)
================================================================
Covers: the Laplace approximation, basis expansions, Bayesian linear
regression, (Bayesian) logistic regression -- GLMs more broadly are
represented by these two (linear = Gaussian-noise GLM, logistic =
Bernoulli-noise GLM); a separate general-GLM script would just be
re-deriving the same IRLS machinery a third time.

Script 23's Ridge/Lasso and Script 24/25's LogReg are all point
estimates: one w vector, chosen by cross-validated lambda, treated as
known-true. This script instead carries a full Gaussian posterior over
the weights:

  1. Bayesian linear regression -- CONJUGATE, no approximation needed:
     Gaussian prior x Gaussian likelihood = Gaussian posterior in closed
     form. Regularization strength (alpha) and noise precision (beta)
     are fit by empirical-Bayes evidence maximization (Bishop PRML
     3.5.2) instead of Script 23's cross-validated grid search -- a
     genuinely different way to answer the same "how much should I
     regularize" question. Includes a quadratic basis expansion of
     Script 23's causal features, with the two feature sets (raw vs.
     expanded) compared by log-evidence -- the same Bayesian model-
     selection idea from Script 31, now applied to regression features
     instead of return distributions.
  2. Bayesian logistic regression -- NOT conjugate (no closed-form
     posterior for a Bernoulli likelihood with a Gaussian prior), which
     is exactly why the Laplace approximation exists: fit the MAP weight
     vector by Newton-Raphson/IRLS, then approximate the posterior as
     Gaussian centered there with covariance equal to the inverse
     Hessian of the negative log-posterior at the mode. The predictive
     probability then uses MacKay's (1992) probit correction, which
     shrinks the raw MAP sigmoid toward 0.5 in proportion to posterior
     uncertainty -- an uncertainty-aware alternative to Script 25's
     point-probability confidence gate.

Reuses Script 23's build_dataset/chronological_split and Script 24's
build_classification_dataset rather than re-deriving the causal feature
pipeline. Standalone diagnostic for now, in the same spirit Scripts 23/24
were before Script 25 wired one of them in -- not wired into Script 17.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from importlib import import_module as _im
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

_r23 = _im("23_ml_alpha_model")
_r24 = _im("24_classification_signals")
build_dataset                 = _r23.build_dataset
chronological_split           = _r23.chronological_split
build_classification_dataset  = _r24.build_classification_dataset

PLOT_STYLE = "seaborn-v0_8-darkgrid"
EB_MAX_ITER = 100
EB_TOL = 1e-6
IRLS_MAX_ITER = 25
IRLS_TOL = 1e-6
LOGISTIC_PRIOR_ALPHA = 1.0   # Gaussian prior precision on logistic weights


# ============================================================
# Basis Expansion
# ============================================================
def basis_expand(X):
    """Quadratic basis expansion: raw features + their squares. Simple,
    interaction-free (keeps dimensionality sane for a few-hundred-row
    dataset)."""
    X2 = X.copy()
    for c in X.columns:
        X2[f"{c}^2"] = X[c] ** 2
    return X2


# ============================================================
# 1. Bayesian Linear Regression — conjugate, empirical-Bayes evidence
# ============================================================
def fit_bayesian_linear_regression(X, y, max_iter=EB_MAX_ITER, tol=EB_TOL):
    """
    Evidence-maximization (empirical Bayes) fit of a Bayesian linear
    model with Gaussian prior w ~ N(0, alpha^-1 I) and noise precision
    beta (Bishop PRML 3.5.2). Standardizes X first (same convention as
    Script 23's Ridge/Lasso pipelines).
    """
    scaler = StandardScaler().fit(X)
    Phi = scaler.transform(X)
    t = y.values if hasattr(y, "values") else np.asarray(y)
    n, m = Phi.shape

    eigvals_base = np.linalg.eigvalsh(Phi.T @ Phi)   # eigenvalues of Phi^T Phi (beta scales them each iter)
    alpha, beta = 1.0, 1.0

    for _ in range(max_iter):
        A = alpha * np.eye(m) + beta * Phi.T @ Phi
        A_inv = np.linalg.inv(A)
        m_N = beta * A_inv @ Phi.T @ t

        lam = beta * eigvals_base
        gamma = float(np.sum(lam / (lam + alpha)))

        alpha_new = gamma / (m_N @ m_N) if m_N @ m_N > 1e-12 else alpha
        resid = t - Phi @ m_N
        beta_new = (n - gamma) / (resid @ resid) if resid @ resid > 1e-12 else beta

        if abs(alpha_new - alpha) < tol and abs(beta_new - beta) < tol:
            alpha, beta = alpha_new, beta_new
            break
        alpha, beta = alpha_new, beta_new

    A = alpha * np.eye(m) + beta * Phi.T @ Phi
    S_N = np.linalg.inv(A)
    m_N = beta * S_N @ Phi.T @ t

    resid = t - Phi @ m_N
    E_mN = 0.5 * beta * (resid @ resid) + 0.5 * alpha * (m_N @ m_N)
    sign, logdet_A = np.linalg.slogdet(A)
    log_evidence = (m / 2) * np.log(alpha) + (n / 2) * np.log(beta) \
        - E_mN - 0.5 * logdet_A - (n / 2) * np.log(2 * np.pi)

    return {"scaler": scaler, "alpha": alpha, "beta": beta, "m_N": m_N, "S_N": S_N,
            "gamma": gamma, "n_features": m, "log_evidence": float(log_evidence)}


def predict_bayesian_linear(X_new, fit):
    """Returns (pred_mean, pred_var) per row -- pred_var = 1/beta (noise)
    + x^T S_N x (parameter uncertainty, grows away from the training data)."""
    Phi = fit["scaler"].transform(X_new)
    mean = Phi @ fit["m_N"]
    var = 1.0 / fit["beta"] + np.einsum("ij,jk,ik->i", Phi, fit["S_N"], Phi)
    return mean, var


# ============================================================
# 2. Bayesian Logistic Regression — Laplace approximation
# ============================================================
def irls_bayesian_logistic(X, y01, alpha=LOGISTIC_PRIOR_ALPHA,
                            max_iter=IRLS_MAX_ITER, tol=IRLS_TOL):
    """
    Newton-Raphson (IRLS) MAP fit under a Gaussian prior w ~ N(0, alpha^-1 I)
    -- equivalent to L2-penalized logistic regression. Also returns the
    Hessian inverse at the mode, which IS the Laplace-approximated
    posterior covariance: p(w|D) ~= N(w_map, H^-1).
    """
    scaler = StandardScaler().fit(X)
    Phi = np.c_[np.ones(len(X)), scaler.transform(X)]   # intercept, unpenalized
    y = np.asarray(y01, dtype=float)
    m = Phi.shape[1]
    w = np.zeros(m)
    prior_prec = alpha * np.eye(m)
    prior_prec[0, 0] = 1e-8   # near-flat prior on the intercept

    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-Phi @ w))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        grad = Phi.T @ (p - y) + prior_prec @ w
        S = p * (1 - p)
        H = Phi.T @ (Phi * S[:, None]) + prior_prec
        step = np.linalg.solve(H, grad)
        w_new = w - step
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    p = 1.0 / (1.0 + np.exp(-Phi @ w))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    S = p * (1 - p)
    H = Phi.T @ (Phi * S[:, None]) + prior_prec
    H_inv = np.linalg.inv(H)

    return {"scaler": scaler, "w_map": w, "H_inv": H_inv, "alpha": alpha}


def laplace_predict_proba(X_new, fit):
    """
    MacKay's (1992) probit approximation to the Bayesian predictive:
    integrating sigmoid(a) over a Gaussian on the logit a ~ N(mu_a, sigma_a^2)
    has no closed form, but sigmoid(mu_a / sqrt(1 + pi*sigma_a^2/8)) is a
    standard, accurate closed-form approximation. sigma_a=0 recovers the
    raw MAP sigmoid exactly; growing sigma_a shrinks the call toward 0.5 --
    the actual value of doing this Bayesianly instead of a point estimate.
    """
    Phi = np.c_[np.ones(len(X_new)), fit["scaler"].transform(X_new)]
    mu_a = Phi @ fit["w_map"]
    sigma_a2 = np.einsum("ij,jk,ik->i", Phi, fit["H_inv"], Phi)
    kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma_a2 / 8.0)
    proba = 1.0 / (1.0 + np.exp(-kappa * mu_a))
    return proba, np.sqrt(sigma_a2)


# ============================================================
# Public APIs (mirror Script 25's get_ml_signal shape)
# ============================================================
def get_bayesian_regression_forecast(df):
    """Next-bar return forecast: picks raw vs. quadratic-basis-expanded
    features by log-evidence (higher = better), reporting predictive
    mean +/- std (parameter + noise uncertainty), not a point number."""
    X, y, feat_cols = build_dataset(df)
    if len(X) < 60:
        return None
    fit_raw = fit_bayesian_linear_regression(X, y)
    X_exp = basis_expand(X)
    fit_exp = fit_bayesian_linear_regression(X_exp, y)

    use_expanded = fit_exp["log_evidence"] > fit_raw["log_evidence"]
    fit = fit_exp if use_expanded else fit_raw
    X_live = X_exp.iloc[[-1]] if use_expanded else X.iloc[[-1]]

    mean, var = predict_bayesian_linear(X_live, fit)
    return {"pred_mean": float(mean[0]), "pred_std": float(np.sqrt(var[0])),
            "used_basis_expansion": use_expanded,
            "log_evidence_raw": fit_raw["log_evidence"], "log_evidence_expanded": fit_exp["log_evidence"],
            "gamma": fit["gamma"], "n_features": fit["n_features"]}


def get_bayesian_logistic_signal(df, conf_threshold=0.60):
    """LONG/SHORT/FLAT via the Laplace-approximated posterior predictive,
    plus the posterior logit-uncertainty sigma_a (Script 25's get_ml_signal
    has no equivalent -- it reports only a point probability)."""
    X, y_pm1, feat_cols = build_classification_dataset(df)
    if len(X) < 120:
        return None
    y01 = (y_pm1.values == 1).astype(float)
    fit = irls_bayesian_logistic(X, y01)
    proba, sigma_a = laplace_predict_proba(X.iloc[[-1]], fit)
    p_up = float(proba[0])

    if p_up >= conf_threshold:
        signal = "LONG"
    elif p_up <= 1 - conf_threshold:
        signal = "SHORT"
    else:
        signal = "FLAT"

    return {"signal": signal, "p_up": p_up, "confidence": abs(p_up - 0.5) * 2,
            "sigma_a": float(sigma_a[0])}


# ============================================================
# Plotting
# ============================================================
def plot_dashboard(ticker, X, reg_fit_raw, reg_fit_exp, X_te, y_te,
                    logit_fit, sigma_grid, proba_grid):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # [0,0] Predicted vs actual (test set) with predictive std error bars
    ax0 = fig.add_subplot(gs[0, 0])
    use_exp = reg_fit_exp["log_evidence"] > reg_fit_raw["log_evidence"]
    fit = reg_fit_exp if use_exp else reg_fit_raw
    X_te_use = basis_expand(X_te) if use_exp else X_te
    mean, var = predict_bayesian_linear(X_te_use, fit)
    std = np.sqrt(var)
    order = np.argsort(y_te.values)
    ax0.errorbar(range(len(y_te)), mean[order], yerr=std[order], fmt="none",
                 ecolor="lightsteelblue", alpha=0.5, zorder=1)
    ax0.scatter(range(len(y_te)), y_te.values[order], s=14, color="gray", label="Actual", zorder=2)
    ax0.scatter(range(len(y_te)), mean[order], s=10, color="crimson", label="Predictive mean", zorder=3)
    ax0.legend(fontsize=8)
    ax0.set_title(f"{ticker} — Bayesian Linear Regression (test set, "
                  f"{'basis-expanded' if use_exp else 'raw'} features)", fontsize=9.5)
    ax0.set_xlabel("test obs (sorted by actual)"); ax0.set_ylabel("next-bar log-return")
    ax0.grid(alpha=0.3)

    # [0,1] Log-evidence comparison: raw vs basis-expanded
    ax1 = fig.add_subplot(gs[0, 1])
    names = ["Raw features", "Quadratic-expanded"]
    vals = [reg_fit_raw["log_evidence"], reg_fit_exp["log_evidence"]]
    colors = ["steelblue", "darkorange"]
    ax1.bar(names, vals, color=colors, alpha=0.85)
    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:.1f}", ha="center",
                 va="bottom" if v > 0 else "top", fontsize=8)
    ax1.set_ylabel("log evidence (higher = better)")
    ax1.set_title("Bayesian Model Selection — Feature Set", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # [1,0] MacKay shrinkage curve: how posterior uncertainty pulls p(up) toward 0.5
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(sigma_grid, proba_grid, color="crimson", lw=1.8)
    ax2.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("posterior logit std (sigma_a)")
    ax2.set_ylabel("P(up) after MacKay correction")
    ax2.set_title(f"Laplace Shrinkage (fixed mu_a={logit_fit['w_map'][0]:+.3f})", fontsize=9.5)
    ax2.grid(alpha=0.3)

    # [1,1] Logistic weight posterior means +/- 1 std (Laplace)
    ax3 = fig.add_subplot(gs[1, 1])
    w = logit_fit["w_map"][1:]   # drop intercept
    sd = np.sqrt(np.diag(logit_fit["H_inv"]))[1:]
    order2 = np.argsort(-np.abs(w))[:10]
    ax3.barh(range(len(order2)), w[order2], xerr=sd[order2], color="steelblue", alpha=0.85)
    ax3.set_yticks(range(len(order2)))
    ax3.set_yticklabels([X.columns[i] for i in order2], fontsize=7)
    ax3.invert_yaxis()
    ax3.axvline(0, color="gray", lw=0.5)
    ax3.set_title("Logistic MAP Weights +/- 1 Posterior Std (Laplace)", fontsize=9.5)
    ax3.grid(axis="x", alpha=0.3)

    fig.suptitle(f"{ticker} — Approximate Inference (Laplace Approximation)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("APPROXIMATE INFERENCE I — LAPLACE APPROXIMATION")
    print("Bayesian Linear Regression (basis expansion) | Bayesian Logistic Regression")
    print("=" * 70)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    summary = []

    for ticker, df in assets_data.items():
        X, y, feat_cols = build_dataset(df)
        if len(X) < 120:
            print(f"\n  {ticker}: skipped -- need >= 120 usable rows")
            continue
        X_tr, X_te, y_tr, y_te = chronological_split(X, y)

        reg_fit_raw = fit_bayesian_linear_regression(X_tr, y_tr)
        reg_fit_exp = fit_bayesian_linear_regression(basis_expand(X_tr), y_tr)
        use_exp = reg_fit_exp["log_evidence"] > reg_fit_raw["log_evidence"]
        best_fit = reg_fit_exp if use_exp else reg_fit_raw
        X_te_use = basis_expand(X_te) if use_exp else X_te
        mean, var = predict_bayesian_linear(X_te_use, best_fit)
        test_rmse = float(np.sqrt(np.mean((mean - y_te.values) ** 2)))

        Xc, yc_pm1, feat_cols_c = build_classification_dataset(df)
        Xc_tr, Xc_te, yc_tr, yc_te = chronological_split(Xc, yc_pm1)
        y01_tr = (yc_tr.values == 1).astype(float)
        logit_fit = irls_bayesian_logistic(Xc_tr, y01_tr)
        proba_te, sigma_te = laplace_predict_proba(Xc_te, logit_fit)
        pred_te = np.where(proba_te >= 0.5, 1, -1)
        test_acc = float(np.mean(pred_te == yc_te.values))

        fc = get_bayesian_regression_forecast(df)
        sig = get_bayesian_logistic_signal(df)

        print(f"\n{'─'*55}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}")
        print(f"{'─'*55}")
        print(f"  Bayesian Linear Regression: log_evidence raw={reg_fit_raw['log_evidence']:.1f}  "
              f"expanded={reg_fit_exp['log_evidence']:.1f}  -> "
              f"{'expanded' if use_exp else 'raw'} features used  "
              f"(gamma={best_fit['gamma']:.1f}/{best_fit['n_features']} eff. params)")
        print(f"  Test RMSE: {test_rmse:.5f}   Live forecast: "
              f"{fc['pred_mean']:+.5f} +/- {fc['pred_std']:.5f}")
        print(f"  Bayesian Logistic Regression (Laplace): test_acc={test_acc:.3f}  "
              f"mean posterior sigma_a={sigma_te.mean():.3f}")
        print(f"  Live signal: {sig['signal']}  P(up)={sig['p_up']:.3f}  "
              f"confidence={sig['confidence']:.0%}  sigma_a={sig['sigma_a']:.3f}")

        summary.append({
            "Ticker": ticker, "FeatureSet": "expanded" if use_exp else "raw",
            "LogEv_Raw": f"{reg_fit_raw['log_evidence']:.1f}",
            "LogEv_Exp": f"{reg_fit_exp['log_evidence']:.1f}",
            "TestRMSE": f"{test_rmse:.5f}",
            "LogisticTestAcc": f"{test_acc:.3f}",
            "LiveSignal": sig["signal"], "P_up": f"{sig['p_up']:.3f}",
            "SigmaA": f"{sig['sigma_a']:.3f}",
        })

        mu_a_fixed = logit_fit["w_map"][0]
        sigma_grid = np.linspace(0, 5, 100)
        kappa = 1.0 / np.sqrt(1.0 + np.pi * sigma_grid ** 2 / 8.0)
        proba_grid = 1.0 / (1.0 + np.exp(-kappa * mu_a_fixed))
        plot_dashboard(ticker, Xc_tr, reg_fit_raw, reg_fit_exp, X_te, y_te,
                       logit_fit, sigma_grid, proba_grid)

    if summary:
        print("\n" + "=" * 70)
        print("APPROXIMATE INFERENCE SUMMARY")
        print("=" * 70)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nApproximate inference (Laplace) analysis complete.")


if __name__ == "__main__":
    main()
