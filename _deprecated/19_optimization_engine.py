"""
================================================================
Script 19 — Optimization Engine
================================================================
Implements optimization methods from your NLP course and applies
them to portfolio problems. Replaces scipy black-box solvers
with transparent, interpretable methods.

Methods (mapped to course lectures):
  L6:  Steepest Descent (gradient method)
  L7:  Newton's Method (quadratic convergence)
  L8:  Conjugate Gradient (Fletcher-Reeves)
  L9:  Least Squares / Projection Theorem
  L10-L11: KKT conditions checker
  L12: Lagrangian Duality
  L13: Penalty and Barrier (interior point) methods

Applications:
  • Min-variance portfolio (compare all solvers)
  • Max-Sharpe portfolio via penalty method
  • Risk parity via Newton's method
  • Efficient frontier via barrier method
  • KKT verification of solutions
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_all_assets, LOOKBACK_DAYS

PLOT_STYLE = "seaborn-v0_8-darkgrid"
ANN = 252
RF  = 0.045


# ============================================================
# L6: Steepest Descent
# ============================================================
def steepest_descent(f, grad_f, x0, lr=0.01, tol=1e-8,
                     max_iter=5000, project=None):
    x = x0.copy()
    history = [float(f(x))]
    for k in range(max_iter):
        g = grad_f(x)
        x = x - lr * g
        if project:
            x = project(x)
        fval = float(f(x))
        history.append(fval)
        if np.linalg.norm(g) < tol:
            break
    return x, history


# ============================================================
# L7: Newton's Method
# ============================================================
def newton_method(f, grad_f, hess_f, x0, tol=1e-8,
                  max_iter=200, project=None):
    x = x0.copy()
    history = [float(f(x))]
    for k in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        try:
            H_reg = H + 1e-8 * np.eye(len(x))
            dx = np.linalg.solve(H_reg, -g)
        except np.linalg.LinAlgError:
            dx = -g
        # Line search (Armijo backtracking)
        alpha = 1.0
        fx = f(x)
        while alpha > 1e-12 and f(x + alpha * dx) > fx - 1e-4 * alpha * g @ dx:
            alpha *= 0.5
        x = x + alpha * dx
        if project:
            x = project(x)
        history.append(float(f(x)))
        if np.linalg.norm(g) < tol:
            break
    return x, history


# ============================================================
# L8: Conjugate Gradient (Fletcher-Reeves)
# ============================================================
def conjugate_gradient(f, grad_f, x0, tol=1e-8,
                       max_iter=5000, project=None):
    x = x0.copy()
    g = grad_f(x)
    d = -g.copy()
    history = [float(f(x))]

    for k in range(max_iter):
        # Line search
        alpha = 0.01
        best_a, best_f = alpha, f(x - alpha * d)
        for a in [0.001, 0.005, 0.01, 0.05, 0.1]:
            trial = f(x + a * d)
            if trial < best_f:
                best_a, best_f = a, trial
        alpha = best_a

        x_new = x + alpha * d
        if project:
            x_new = project(x_new)
        g_new = grad_f(x_new)

        # Fletcher-Reeves beta
        beta = (g_new @ g_new) / max(g @ g, 1e-16)
        d = -g_new + beta * d

        x, g = x_new, g_new
        history.append(float(f(x)))
        if np.linalg.norm(g) < tol:
            break
    return x, history


# ============================================================
# L13: Penalty Method
# ============================================================
def penalty_method(f, grad_f, x0, eq_constraints, ineq_constraints=None,
                   mu_init=1.0, mu_mult=10.0, max_outer=15, tol=1e-8,
                   project=None):
    """
    Quadratic penalty method.
    eq_constraints: list of (c_func, grad_c_func) where c(x) = 0
    ineq_constraints: list of (g_func, grad_g_func) where g(x) >= 0
    """
    mu = mu_init
    x = x0.copy()
    history = []

    for outer in range(max_outer):
        def augmented(w):
            val = f(w)
            for c_fn, _ in eq_constraints:
                val += mu / 2 * c_fn(w) ** 2
            if ineq_constraints:
                for g_fn, _ in ineq_constraints:
                    violation = min(g_fn(w), 0)
                    val += mu / 2 * violation ** 2
            return val

        def aug_grad(w):
            g = grad_f(w)
            for c_fn, gc_fn in eq_constraints:
                g += mu * c_fn(w) * gc_fn(w)
            if ineq_constraints:
                for g_fn, gg_fn in ineq_constraints:
                    if g_fn(w) < 0:
                        g += mu * g_fn(w) * gg_fn(w)
            return g

        x, hist = steepest_descent(augmented, aug_grad, x, lr=0.005,
                                    max_iter=2000, project=project)
        history.extend(hist)

        violation = sum(abs(c(x)) for c, _ in eq_constraints)
        if violation < tol:
            break
        mu *= mu_mult

    return x, history


# ============================================================
# L13: Log-Barrier (Interior Point) Method
# ============================================================
def barrier_method(f, grad_f, x0, ineq_constraints,
                   eq_constraint=None, t_init=1.0, t_mult=5.0,
                   max_outer=20, tol=1e-8, project=None):
    """
    Log-barrier interior point method.
    ineq_constraints: list of (g_func, grad_g_func) where g(x) > 0
    """
    t = t_init
    x = x0.copy()
    N = len(x)
    history = []

    for outer in range(max_outer):
        def barrier_obj(w):
            val = t * f(w)
            for g_fn, _ in ineq_constraints:
                gv = g_fn(w)
                if gv <= 0:
                    return 1e15
                val -= np.log(gv)
            return val

        def barrier_grad(w):
            g = t * grad_f(w)
            for g_fn, gg_fn in ineq_constraints:
                gv = g_fn(w)
                if gv <= 1e-12:
                    gv = 1e-12
                g -= gg_fn(w) / gv
            return g

        x, hist = steepest_descent(barrier_obj, barrier_grad, x, lr=0.001,
                                    max_iter=1000, project=project)
        history.extend(hist)

        if N / t < tol:
            break
        t *= t_mult

    return x, history


# ============================================================
# L10-L11: KKT Conditions Checker
# ============================================================
def check_kkt(x, grad_f_val, eq_vals, ineq_vals, lambdas_eq, lambdas_ineq):
    """
    Verify KKT conditions:
    1. Stationarity: grad_f + sum(lambda_eq * grad_c) + sum(mu * grad_g) = 0
    2. Primal feasibility: c(x) = 0, g(x) >= 0
    3. Dual feasibility: mu >= 0
    4. Complementary slackness: mu * g(x) = 0
    """
    results = {}
    results["stationarity_norm"] = float(np.linalg.norm(grad_f_val))
    results["eq_feasibility"] = float(max(abs(v) for v in eq_vals)) if eq_vals else 0.0
    results["ineq_feasibility"] = float(min(ineq_vals)) if ineq_vals else 0.0
    results["dual_feasibility"] = all(m >= -1e-8 for m in lambdas_ineq) if lambdas_ineq else True
    results["comp_slackness"] = float(max(abs(m * g)
        for m, g in zip(lambdas_ineq, ineq_vals))) if lambdas_ineq else 0.0
    results["kkt_satisfied"] = (
        results["stationarity_norm"] < 5e-3
        and results["eq_feasibility"] < 1e-3
        and results["ineq_feasibility"] >= -1e-6
        and results["dual_feasibility"]
    )
    return results


# ============================================================
# Portfolio Applications
# ============================================================
def project_simplex(w):
    """Project onto probability simplex (w >= 0, sum = 1)."""
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 1e-12 else np.ones(len(w)) / len(w)


def portfolio_var_obj(w, cov):
    return float(w @ cov @ w)


def portfolio_var_grad(w, cov):
    return 2 * cov @ w


def portfolio_var_hess(w, cov):
    return 2 * cov


def sharpe_obj(w, mu, cov):
    ret = float(w @ mu) * ANN
    vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(ANN)
    return -(ret - RF) / max(vol, 1e-8)


def sharpe_grad(w, mu, cov):
    ret = w @ mu * ANN
    vol = np.sqrt(w @ cov @ w) * np.sqrt(ANN)
    if vol < 1e-8:
        return np.zeros(len(w))
    Cw = cov @ w
    d_ret = mu * ANN
    d_vol = Cw / np.sqrt(w @ cov @ w) * np.sqrt(ANN)
    return -(d_ret * vol - (ret - RF) * d_vol) / vol ** 2


def risk_parity_obj(w, cov):
    sigma = np.sqrt(w @ cov @ w)
    if sigma < 1e-12:
        return 0.0
    rc = w * (cov @ w) / sigma
    target = sigma / len(w)
    return float(np.sum((rc - target) ** 2))


def risk_parity_grad(w, cov):
    N = len(w)
    sigma = np.sqrt(w @ cov @ w)
    if sigma < 1e-12:
        return np.zeros(N)
    Cw = cov @ w
    rc = w * Cw / sigma
    target = sigma / N

    grad = np.zeros(N)
    for i in range(N):
        drc_dwi = (Cw[i] + w[i] * cov[i, i]) / sigma - w[i] * Cw[i] * Cw.sum() / sigma ** 3
        grad[i] = 2 * (rc[i] - target) * drc_dwi
    return grad


def solve_all_methods(returns, names):
    T, N = returns.shape
    mu = returns.mean(axis=0)
    cov = np.cov(returns.T, ddof=1)

    # Regularize
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 1e-8:
        cov += (1e-8 - eigvals.min()) * np.eye(N)

    w0 = np.ones(N) / N
    results = {}

    # --- Min Variance ---
    f_var = lambda w: portfolio_var_obj(w, cov)
    g_var = lambda w: portfolio_var_grad(w, cov)
    h_var = lambda w: portfolio_var_hess(w, cov)

    t0 = time.time()
    w_sd, hist_sd = steepest_descent(f_var, g_var, w0.copy(), lr=0.005,
                                      project=project_simplex)
    t_sd = time.time() - t0
    results["MinVar_SteepDescent"] = {
        "w": w_sd, "history": hist_sd, "time": t_sd, "method": "Steepest Descent"}

    # Newton with augmented Lagrangian (penalty on sum=1 + non-negativity)
    pen_mu = 100.0
    f_nt = lambda w: portfolio_var_obj(w, cov) + pen_mu * (w.sum() - 1)**2
    g_nt = lambda w: portfolio_var_grad(w, cov) + 2 * pen_mu * (w.sum() - 1) * np.ones(N)
    h_nt = lambda w: portfolio_var_hess(w, cov) + 2 * pen_mu * np.ones((N, N))

    t0 = time.time()
    w_nt, hist_nt = newton_method(f_nt, g_nt, h_nt, w0.copy(),
                                   project=project_simplex)
    t_nt = time.time() - t0
    results["MinVar_Newton"] = {
        "w": w_nt, "history": hist_nt, "time": t_nt, "method": "Newton"}

    t0 = time.time()
    w_cg, hist_cg = conjugate_gradient(f_var, g_var, w0.copy(),
                                        project=project_simplex)
    t_cg = time.time() - t0
    results["MinVar_ConjGrad"] = {
        "w": w_cg, "history": hist_cg, "time": t_cg, "method": "Conjugate Gradient"}

    # --- Min Variance via Penalty Method ---
    eq_sum = (lambda w: w.sum() - 1.0, lambda w: np.ones(N))
    ineq_pos = [(lambda w, i=i: w[i], lambda w, i=i: np.eye(N)[i]) for i in range(N)]

    t0 = time.time()
    w_pen, hist_pen = penalty_method(f_var, g_var, w0.copy(),
                                      eq_constraints=[eq_sum],
                                      ineq_constraints=ineq_pos)
    w_pen = project_simplex(w_pen)
    t_pen = time.time() - t0
    results["MinVar_Penalty"] = {
        "w": w_pen, "history": hist_pen, "time": t_pen, "method": "Penalty"}

    # --- Max Sharpe via Penalty Method ---
    f_sh = lambda w: sharpe_obj(w, mu, cov)
    g_sh = lambda w: sharpe_grad(w, mu, cov)

    t0 = time.time()
    w_msr, hist_msr = penalty_method(f_sh, g_sh, w0.copy(),
                                      eq_constraints=[eq_sum],
                                      ineq_constraints=ineq_pos)
    w_msr = project_simplex(w_msr)
    t_msr = time.time() - t0
    results["MaxSharpe_Penalty"] = {
        "w": w_msr, "history": hist_msr, "time": t_msr, "method": "Penalty"}

    # --- Risk Parity via Newton ---
    f_rp = lambda w: risk_parity_obj(w, cov)
    g_rp = lambda w: risk_parity_grad(w, cov)
    h_rp = lambda w: np.eye(N) * 0.01  # approximate Hessian

    t0 = time.time()
    w_rp, hist_rp = newton_method(f_rp, g_rp, h_rp, w0.copy(),
                                   project=project_simplex)
    t_rp = time.time() - t0
    results["RiskParity_Newton"] = {
        "w": w_rp, "history": hist_rp, "time": t_rp, "method": "Newton"}

    # KKT check for MinVar Newton solution
    kkt = check_kkt(
        w_nt,
        grad_f_val=g_var(w_nt),
        eq_vals=[float(w_nt.sum() - 1.0)],
        ineq_vals=list(w_nt),
        lambdas_eq=[0.0],
        lambdas_ineq=list(np.zeros(N)),
    )

    return results, mu, cov, kkt


# ============================================================
# Plotting
# ============================================================
def plot_optimization(results, names, mu, cov, kkt):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(18, 11))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    colors = {"MinVar_SteepDescent": "royalblue", "MinVar_Newton": "tomato",
              "MinVar_ConjGrad": "forestgreen", "MinVar_Penalty": "purple",
              "MaxSharpe_Penalty": "gold", "RiskParity_Newton": "teal"}

    # [0,0] Convergence curves
    ax0 = fig.add_subplot(gs[0, 0])
    for name, res in results.items():
        if "MinVar" in name:
            h = res["history"]
            ax0.plot(range(len(h)), h, lw=1.5, label=f"{res['method']} ({res['time']:.3f}s)",
                     color=colors.get(name, "gray"))
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Objective (portfolio variance)")
    ax0.set_title("Convergence: Min-Variance Solvers", fontsize=10)
    ax0.legend(fontsize=7)
    ax0.set_yscale("log")
    ax0.grid(alpha=0.3)

    # [0,1] Weight comparison
    ax1 = fig.add_subplot(gs[0, 1])
    N = len(names)
    x_pos = np.arange(N)
    width = 0.15
    for i, (name, res) in enumerate(results.items()):
        if "MinVar" in name:
            ax1.bar(x_pos + i * width, res["w"] * 100, width,
                    label=res["method"], color=colors.get(name, "gray"), alpha=0.8)
    ax1.set_xticks(x_pos + width * 1.5)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=5)
    ax1.set_ylabel("Weight (%)")
    ax1.set_title("MinVar Weights: Method Comparison", fontsize=10)
    ax1.legend(fontsize=6)
    ax1.grid(axis="y", alpha=0.3)

    # [0,2] Efficient frontier computed with our solver
    ax2 = fig.add_subplot(gs[0, 2])
    targets = np.linspace(float(mu.min()) * 1.05, float(mu.max()) * 0.95, 40)
    ef_v, ef_r = [], []
    for tgt in targets:
        eq_sum = (lambda w: w.sum() - 1.0, lambda w: np.ones(N))
        eq_ret = (lambda w, t=tgt: w @ mu - t, lambda w: mu)
        w0 = np.ones(N) / N
        w, _ = penalty_method(
            lambda w: portfolio_var_obj(w, cov),
            lambda w: portfolio_var_grad(w, cov),
            w0, eq_constraints=[eq_sum, eq_ret])
        w = project_simplex(w)
        ret = float(w @ mu) * ANN * 100
        vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(ANN) * 100
        if vol > 0:
            ef_v.append(vol)
            ef_r.append(ret)

    sharpes = [(r - RF * 100) / max(v, 1e-6) for r, v in zip(ef_r, ef_v)]
    sc = ax2.scatter(ef_v, ef_r, c=sharpes, cmap="RdYlGn", s=8)
    plt.colorbar(sc, ax=ax2, pad=0.02, label="Sharpe")

    for name, res in results.items():
        w = res["w"]
        ret = float(w @ mu) * ANN * 100
        vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(ANN) * 100
        ax2.scatter(vol, ret, s=80, color=colors.get(name, "gray"),
                    marker="D", edgecolors="white", zorder=5)
        short_name = name.split("_")[0]
        ax2.annotate(f" {short_name}", (vol, ret), fontsize=6)
    ax2.set_xlabel("Vol (%)")
    ax2.set_ylabel("Return (%)")
    ax2.set_title("Efficient Frontier (Penalty Method)", fontsize=10)
    ax2.grid(alpha=0.3)

    # [1,0] Solver timing comparison
    ax3 = fig.add_subplot(gs[1, 0])
    solver_names = list(results.keys())
    times = [results[n]["time"] * 1000 for n in solver_names]
    bars_c = [colors.get(n, "gray") for n in solver_names]
    y_pos = range(len(solver_names))
    ax3.barh(y_pos, times, color=bars_c, alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([n.replace("_", "\n") for n in solver_names], fontsize=7)
    for i, t in enumerate(times):
        ax3.text(t + 0.5, i, f"{t:.1f}ms", va="center", fontsize=7)
    ax3.set_xlabel("Time (ms)")
    ax3.set_title("Solver Performance", fontsize=10)
    ax3.grid(axis="x", alpha=0.3)

    # [1,1] KKT conditions
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    kkt_rows = [
        ["Stationarity ||grad||", f"{kkt['stationarity_norm']:.6f}",
         "PASS" if kkt['stationarity_norm'] < 1e-4 else "FAIL"],
        ["Eq Feasibility |c(x)|", f"{kkt['eq_feasibility']:.6f}",
         "PASS" if kkt['eq_feasibility'] < 1e-4 else "FAIL"],
        ["Ineq Feasibility min(g)", f"{kkt['ineq_feasibility']:.6f}",
         "PASS" if kkt['ineq_feasibility'] >= -1e-6 else "FAIL"],
        ["Dual Feasibility", str(kkt['dual_feasibility']),
         "PASS" if kkt['dual_feasibility'] else "FAIL"],
        ["KKT Satisfied", str(kkt['kkt_satisfied']),
         "PASS" if kkt['kkt_satisfied'] else "FAIL"],
    ]
    table = ax4.table(cellText=kkt_rows,
                      colLabels=["Condition", "Value", "Status"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax4.set_title("KKT Conditions (MinVar Newton)", fontsize=10, pad=15)

    # [1,2] Portfolio stats table
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    stat_rows = []
    for name, res in results.items():
        w = res["w"]
        ret = float(w @ mu) * ANN * 100
        vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(ANN) * 100
        sr  = (ret - RF * 100) / max(vol, 1e-6)
        stat_rows.append([name.replace("_", " "), f"{ret:+.1f}%",
                          f"{vol:.1f}%", f"{sr:.2f}", f"{res['time']*1000:.0f}ms"])
    table2 = ax5.table(cellText=stat_rows,
                       colLabels=["Strategy", "Return", "Vol", "Sharpe", "Time"],
                       loc="center", cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(8)
    table2.scale(1.0, 1.5)
    ax5.set_title("Portfolio Stats by Solver", fontsize=10, pad=15)

    fig.suptitle("Optimization Engine — Method Comparison & KKT Verification",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("OPTIMIZATION ENGINE")
    print("Steepest Descent | Newton | Conjugate Gradient | Penalty | Barrier")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    if not assets_data:
        print("\n  No asset data could be loaded; skipping optimization engine.\n")
        return

    names = list(assets_data.keys())
    returns_frame = pd.concat(
        [assets_data[n]["log_return"].dropna().rename(n) for n in names],
        axis=1,
        join="inner",
    ).dropna()

    if returns_frame.empty or returns_frame.shape[1] < 2:
        print("\n  Insufficient aligned return data for optimization engine.\n")
        return

    mat = returns_frame.to_numpy(dtype=float)
    mat = mat[np.isfinite(mat).all(axis=1)]
    T, N = mat.shape
    print(f"\n  Returns: {T} days x {N} assets\n")

    results, mu, cov, kkt = solve_all_methods(mat, names)

    for name, res in results.items():
        w   = res["w"]
        ret = float(w @ mu) * ANN * 100
        vol = float(np.sqrt(w @ cov @ w)) * np.sqrt(ANN) * 100
        sr  = (ret - RF * 100) / max(vol, 1e-6)
        iters = len(res["history"])
        print(f"  {name:<24}: ret={ret:+.1f}%  vol={vol:.1f}%  "
              f"sharpe={sr:.3f}  iters={iters}  time={res['time']*1000:.1f}ms")
        top = sorted(zip(names, w), key=lambda x: -x[1])[:5]
        top_str = "  ".join(f"{n}:{wt:.1%}" for n, wt in top if wt > 0.01)
        print(f"    Top: {top_str}")

    print(f"\n  KKT Conditions (MinVar Newton):")
    for k, v in kkt.items():
        print(f"    {k}: {v}")

    plot_optimization(results, names, mu, cov, kkt)

    print("\nOptimization engine complete.")


if __name__ == "__main__":
    main()
