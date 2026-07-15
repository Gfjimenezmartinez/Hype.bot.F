"""
================================================================
Script 24 — Classification Trading Signals (Perceptron/SVM/MLP/KNN/RF)
================================================================
Script 23 tried to regress next-day *return magnitude* and found near-
zero/negative out-of-sample R^2 -- a hard target. This script instead
predicts *direction* (up/down), a binary classification problem that's
often more tractable for a long/short/flat trading signal, using the
classifier progression from your ML coursework:

  Perceptron : hand-implemented, batch and sequential/online updates.
               Only provably converges to zero error on linearly
               separable data -- financial direction data almost
               certainly isn't, so a non-converging error curve here
               is an honest diagnostic, not a bug. Motivates the rest.
  SVM        : sklearn SVC, kernel (linear/poly/rbf/sigmoid) x C swept
               via GridSearchCV with TimeSeriesSplit CV (not a random
               k-fold -- financial data is autocorrelated).
  MLP        : sklearn MLPClassifier, train-vs-test accuracy swept
               across hidden-layer size -- the same bias-variance
               question Script 23 asked of regularization strength,
               now asked of a nonlinear classifier's capacity.
  KNN        : sklearn KNeighborsClassifier, train-vs-test accuracy
               swept across K -- the textbook small-K/high-variance vs
               large-K/high-bias curve. Distance-based, so it's fit
               inside a StandardScaler Pipeline: the homework's raw
               Euclidean distance on unscaled features (RSI ~0-100 vs
               log-return lags ~0.001-0.02) would be almost entirely
               determined by whichever feature happens to have the
               largest numeric range, not by which one actually
               carries signal.
  RandomForest: sklearn RandomForestClassifier, tuned with
               RandomizedSearchCV(n_iter=15) over a wide parameter
               range. The homework's own grid --
               n_estimators=range(10,10000,10) x max_depth=range(10,
               200,10), ~19,000 combinations x cv=3 -- is why its own
               notebook shows a KeyboardInterrupt. RandomizedSearchCV
               caps the number of fits at n_iter regardless of how
               wide the ranges are, which is the actual fix (not just
               picking smaller numbers).

Naive baseline (not in the original coursework, where synthetic blobs/
moons data made 90-100% accuracy unsurprising): every model's test
accuracy is reported next to the majority-class baseline
(max(P(up), P(down)) on the test set), since that's the bar a
direction-prediction signal actually has to clear to be useful.

Reuses Script 23's causal feature set and chronological-split /
TimeSeriesSplit methodology rather than rebuilding it. Standalone
diagnostic for now -- not wired into Scripts 17/20/21's forecast
inputs yet.
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from importlib import import_module as _im
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from data_loader import load_all_assets, LOOKBACK_DAYS, format_price

_r23 = _im("23_ml_alpha_model")
build_dataset       = _r23.build_dataset
chronological_split = _r23.chronological_split

PLOT_STYLE    = "seaborn-v0_8-darkgrid"
TOP_N         = 8          # SVM/MLP/RF grid-search only on the top-N by vol
                            # (same precedent as Script 3's TOP_N -- these fits
                            # are too slow to run the full 21-ticker universe on)
PERCEPTRON_EPOCHS = 100
PERCEPTRON_LR     = 0.01
SVM_C_GRID    = [0.01, 0.1, 1, 5, 10, 50]
SVM_KERNELS   = ["linear", "poly", "rbf", "sigmoid"]
SVM_GAMMA_GRID = [0.001, 0.01, 0.1, 1]   # unused for linear -- see svm_grid_search
N_CV_SPLITS   = 3   # was 5 -- SVM's grid alone is ~390 fits/ticker (kernels x C x gamma x
                     # folds), the dominant cost in a full-suite run; 3-fold TimeSeriesSplit
                     # is still standard practice and cuts every grid-search's fit count by
                     # 40% without changing which C/gamma/K/n_estimators values get tested
HIDDEN_SIZES  = [5, 10, 20, 40, 80, 160]   # coarser than a 5..300 sweep -- runtime
K_GRID        = [1, 3, 5, 10, 15, 25, 40, 60, 80]
RF_PARAM_DIST = {"n_estimators": [50, 100, 200, 300, 500],
                  "max_depth": [3, 5, 10, 20, None],
                  "min_samples_leaf": [1, 5, 10, 20]}
RF_N_ITER     = 10   # was 15 -- RandomizedSearchCV already samples this space instead of
                     # a full grid; fewer candidates is a direct runtime cut
CONF_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


# ============================================================
# Dataset (reuses Script 23's causal features + chronological split)
# ============================================================
def build_classification_dataset(df):
    X, y_cont, feat_cols = build_dataset(df)
    y = np.where(y_cont.values > 0, 1, -1)   # exact-zero return (rare) -> -1
    return X, pd.Series(y, index=X.index, name="direction"), feat_cols


# ============================================================
# Perceptron (hand-implemented: batch and sequential updates)
# ============================================================
class Perceptron:
    """
    x = [1, features...] @ w  ->  sign(.) in {-1,+1}.
    Convergence to zero error is only guaranteed for linearly separable
    data (Perceptron Convergence Theorem) -- expect the error curve to
    plateau/oscillate on real return-direction data, not reach zero.
    """
    def __init__(self, n_features, learning_rate=PERCEPTRON_LR):
        self.w = np.zeros(n_features + 1)
        self.lr = learning_rate

    @staticmethod
    def _augment(X):
        Xv = X.values if hasattr(X, "values") else X
        return np.c_[np.ones(len(Xv)), Xv]

    def _raw_predict(self, Xa):
        return np.where(Xa @ self.w > 0, 1, -1)

    def predict(self, X):
        return self._raw_predict(self._augment(X))

    def fit_batch(self, X, y, num_epochs=PERCEPTRON_EPOCHS):
        Xa, yv = self._augment(X), np.asarray(y)
        errors = []
        for _ in range(num_epochs):
            pred = self._raw_predict(Xa)
            mis = pred != yv
            errors.append(int(mis.sum()))
            if not mis.any():
                break
            self.w += self.lr * (yv[mis, None] * Xa[mis]).sum(axis=0)
        return errors

    def fit_sequential(self, X, y, num_epochs=PERCEPTRON_EPOCHS):
        Xa, yv = self._augment(X), np.asarray(y)
        errors = []
        for _ in range(num_epochs):
            mis_count = int((self._raw_predict(Xa) != yv).sum())
            errors.append(mis_count)
            if mis_count == 0:
                break
            for i in range(len(Xa)):
                if (1 if Xa[i] @ self.w > 0 else -1) != yv[i]:
                    self.w += self.lr * yv[i] * Xa[i]
        return errors


# ============================================================
# SVM: kernel x C (x gamma, where applicable) grid search, TimeSeriesSplit CV
# ============================================================
def svm_grid_search(X_tr, y_tr):
    """
    Linear kernels don't use gamma at all -- sklearn silently ignores it, so
    sweeping gamma for 'linear' would just re-fit the identical model 4x for
    no reason. GridSearchCV accepts a *list* of param-grid dicts, so linear
    gets its own C-only grid and the other three kernels get C x gamma.
    """
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": SVM_C_GRID},
        {"svc__kernel": ["poly", "rbf", "sigmoid"], "svc__C": SVM_C_GRID, "svc__gamma": SVM_GAMMA_GRID},
    ]
    grid = GridSearchCV(pipe, param_grid, cv=TimeSeriesSplit(n_splits=N_CV_SPLITS), scoring="accuracy")
    grid.fit(X_tr, y_tr)

    res = pd.DataFrame(grid.cv_results_)
    # Best score per kernel (best C, and best gamma where applicable)
    per_kernel = (res.groupby(res["param_svc__kernel"])["mean_test_score"]
                  .max().reindex(SVM_KERNELS))
    # Best CV score per (kernel, C), maximized over gamma where applicable --
    # for the dashboard's "accuracy vs C" panel
    cv_curve = {}
    for kernel in SVM_KERNELS:
        sub = res[res["param_svc__kernel"] == kernel]
        best_per_c = sub.groupby("param_svc__C")["mean_test_score"].max().sort_index()
        cv_curve[kernel] = (best_per_c.index.values.astype(float), best_per_c.values)
    return grid, per_kernel, cv_curve


def svm_pca_boundary(X_tr, y_tr, kernel, C, gamma=None):
    """2D PCA projection + its own small SVC, fit purely for visualization --
    NOT the real high-dimensional decision boundary of the selected model."""
    pca = PCA(n_components=2).fit(StandardScaler().fit_transform(X_tr))
    X2 = pca.transform(StandardScaler().fit_transform(X_tr))
    kwargs = {"kernel": kernel, "C": C}
    if gamma is not None:
        kwargs["gamma"] = gamma
    clf2d = SVC(**kwargs).fit(X2, y_tr)
    return X2, clf2d


# ============================================================
# MLP: train-vs-test accuracy across hidden-layer size
# ============================================================
def mlp_capacity_sweep(X_tr, y_tr, X_te, y_te, sizes=HIDDEN_SIZES):
    train_scores, test_scores = [], []
    for h in sizes:
        mlp = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(h,), activation="logistic",
                                   solver="sgd", alpha=1e-2, max_iter=200,
                                   learning_rate_init=0.1, random_state=42)),
        ])
        mlp.fit(X_tr, y_tr)
        train_scores.append(mlp.score(X_tr, y_tr))
        test_scores.append(mlp.score(X_te, y_te))
    return np.array(train_scores), np.array(test_scores)


# ============================================================
# KNN: train-vs-test accuracy across K (scaled -- see module docstring)
# ============================================================
def knn_k_sweep(X_tr, y_tr, X_te, y_te, k_grid=K_GRID):
    train_scores, test_scores = [], []
    for k in k_grid:
        knn = Pipeline([("scaler", StandardScaler()),
                         ("knn", KNeighborsClassifier(n_neighbors=k))])
        knn.fit(X_tr, y_tr)
        train_scores.append(knn.score(X_tr, y_tr))
        test_scores.append(knn.score(X_te, y_te))
    return np.array(train_scores), np.array(test_scores)


def knn_select_k(X_tr, y_tr, k_grid=K_GRID):
    pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    grid = GridSearchCV(pipe, {"knn__n_neighbors": k_grid},
                         cv=TimeSeriesSplit(n_splits=N_CV_SPLITS), scoring="accuracy")
    grid.fit(X_tr, y_tr)
    return grid


# ============================================================
# Random Forest: bounded RandomizedSearchCV (see module docstring --
# an exhaustive grid over this parameter space never finishes)
# ============================================================
def fit_random_forest(X_tr, y_tr, n_iter=RF_N_ITER):
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42), RF_PARAM_DIST,
        n_iter=n_iter, cv=TimeSeriesSplit(n_splits=N_CV_SPLITS),
        scoring="accuracy", random_state=42,
    )
    search.fit(X_tr, y_tr)
    return search


# ============================================================
# Logistic Regression: interpretable probabilistic baseline
# ============================================================
def fit_logistic(X_tr, y_tr):
    clf = Pipeline([("scaler", StandardScaler()),
                     ("logreg", LogisticRegression(max_iter=1000))])
    clf.fit(X_tr, y_tr)
    return clf


def confidence_threshold_curve(clf, X_te, y_te, thresholds=CONF_THRESHOLDS):
    """
    Selective prediction: at each confidence threshold t, only 'trade' on
    days where P(up) >= t or P(up) <= 1-t. Reports coverage (fraction of
    days confident enough to act) and accuracy restricted to those days --
    answers whether the model's stated confidence actually tracks being
    right, not just whether it's right on average.
    """
    up_idx = list(clf.classes_).index(1)
    proba_up = clf.predict_proba(X_te)[:, up_idx]
    y_te_v = y_te.values
    coverage, accuracy = [], []
    for t in thresholds:
        confident = (proba_up >= t) | (proba_up <= 1 - t)
        coverage.append(float(confident.mean()))
        if confident.sum() > 0:
            pred = np.where(proba_up[confident] >= 0.5, 1, -1)
            accuracy.append(float((pred == y_te_v[confident]).mean()))
        else:
            accuracy.append(np.nan)
    return np.array(coverage), np.array(accuracy)


# ============================================================
# Per-Asset Analysis
# ============================================================
def analyse_asset(df, full_models=True):
    X, y, feat_cols = build_classification_dataset(df)
    if len(X) < 120:
        raise ValueError(f"only {len(X)} usable rows after feature engineering (need >=120)")

    X_tr, X_te, y_tr, y_te = chronological_split(X, y)
    baseline = float(max((y_te == 1).mean(), (y_te == -1).mean()))

    perc = Perceptron(n_features=X_tr.shape[1])
    batch_errors = perc.fit_batch(X_tr, y_tr)
    perc_batch_test_acc = float((perc.predict(X_te) == y_te.values).mean())

    perc_seq = Perceptron(n_features=X_tr.shape[1])
    seq_errors = perc_seq.fit_sequential(X_tr, y_tr)
    perc_seq_test_acc = float((perc_seq.predict(X_te) == y_te.values).mean())

    # Sanity check: batch perceptron shouldn't finish worse than it started
    perc_regressed = batch_errors[-1] > batch_errors[0] and len(batch_errors) > 1

    # KNN runs on every ticker (cheap, like Perceptron)
    knn_train_scores, knn_test_scores = knn_k_sweep(X_tr, y_tr, X_te, y_te)
    knn_grid = knn_select_k(X_tr, y_tr)
    knn_best_k = knn_grid.best_params_["knn__n_neighbors"]
    knn_test_acc = float(knn_grid.score(X_te, y_te))

    # Logistic Regression + confidence-threshold curve also run on every
    # ticker (cheap, like Perceptron/KNN) -- this is the model the
    # confidence-thresholded "only trade when confident" diagnostic uses.
    logreg = fit_logistic(X_tr, y_tr)
    logreg_test_acc = float(logreg.score(X_te, y_te))
    logreg_coef = logreg.named_steps["logreg"].coef_.flatten()
    conf_coverage, conf_accuracy = confidence_threshold_curve(logreg, X_te, y_te)

    result = {
        "feat_cols": feat_cols, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "baseline": baseline,
        "perceptron": {
            "batch_errors": batch_errors, "seq_errors": seq_errors,
            "batch_test_acc": perc_batch_test_acc, "seq_test_acc": perc_seq_test_acc,
            "regressed": perc_regressed,
        },
        "knn": {
            "k_grid": K_GRID, "train_scores": knn_train_scores, "test_scores": knn_test_scores,
            "best_k": knn_best_k, "test_acc": knn_test_acc,
        },
        "logreg": {
            "test_acc": logreg_test_acc, "coef": logreg_coef,
            "conf_thresholds": CONF_THRESHOLDS,
            "conf_coverage": conf_coverage, "conf_accuracy": conf_accuracy,
        },
        "full_models": full_models,
    }

    if not full_models:
        return result

    grid, per_kernel, cv_curve = svm_grid_search(X_tr, y_tr)
    best_kernel = grid.best_params_["svc__kernel"]
    best_C = grid.best_params_["svc__C"]
    best_gamma = grid.best_params_.get("svc__gamma")
    svm_test_acc = float(grid.score(X_te, y_te))
    X2_tr, clf2d = svm_pca_boundary(X_tr, y_tr, best_kernel, best_C, best_gamma)

    train_scores, test_scores = mlp_capacity_sweep(X_tr, y_tr, X_te, y_te)
    best_h_idx = int(np.argmax(test_scores))

    result["svm"] = {
        "best_kernel": best_kernel, "best_C": best_C, "best_gamma": best_gamma, "test_acc": svm_test_acc,
        "per_kernel_cv": per_kernel, "cv_curve": cv_curve, "X2_tr": X2_tr, "clf2d": clf2d,
    }
    result["mlp"] = {
        "sizes": HIDDEN_SIZES, "train_scores": train_scores, "test_scores": test_scores,
        "best_h": HIDDEN_SIZES[best_h_idx], "best_test_acc": float(test_scores[best_h_idx]),
    }

    rf_search = fit_random_forest(X_tr, y_tr)
    rf_pred = rf_search.predict(X_te)
    rf_cm = confusion_matrix(y_te, rf_pred, labels=[1, -1])
    result["rf"] = {
        "best_params": rf_search.best_params_,
        "test_acc": float(rf_search.score(X_te, y_te)),
        "cv_results": pd.DataFrame(rf_search.cv_results_),
        "confusion_matrix": rf_cm,
        "precision_up": float(precision_score(y_te, rf_pred, pos_label=1, zero_division=0)),
        "recall_up": float(recall_score(y_te, rf_pred, pos_label=1, zero_division=0)),
        "precision_down": float(precision_score(y_te, rf_pred, pos_label=-1, zero_division=0)),
        "recall_down": float(recall_score(y_te, rf_pred, pos_label=-1, zero_division=0)),
    }
    return result


# ============================================================
# Plotting
# ============================================================
def _plot_perceptron(ax, ticker, r):
    p = r["perceptron"]
    ax.plot(p["batch_errors"], color="royalblue", lw=1.4, label="Batch")
    ax.plot(p["seq_errors"], color="darkorange", lw=1.2, alpha=0.8, label="Sequential")
    ax.set_xlabel("epoch"); ax.set_ylabel("misclassified (train)")
    ax.legend(fontsize=7)
    ax.set_title(f"{ticker} — Perceptron Convergence", fontsize=9)
    ax.grid(alpha=0.3)


def _plot_knn(ax, r):
    knn_r = r["knn"]
    ax.plot(knn_r["k_grid"], knn_r["train_scores"], "o-", color="royalblue", label="train")
    ax.plot(knn_r["k_grid"], knn_r["test_scores"], "s-", color="darkorange", label="test")
    ax.axhline(r["baseline"], color="gray", lw=1.0, ls="--", label="naive baseline")
    ax.axvline(knn_r["best_k"], color="black", lw=1.0, ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("K (log scale)"); ax.set_ylabel("accuracy")
    ax.legend(fontsize=7)
    ax.set_title(f"KNN: Accuracy vs K  |  small K = high variance, large K = high bias  "
                 f"(CV-selected K={knn_r['best_k']})", fontsize=8.5)
    ax.grid(alpha=0.3)


def _plot_logreg_coef(ax, r, top_n=None):
    logreg_r = r["logreg"]
    feat_cols = r["feat_cols"]
    coef = logreg_r["coef"]
    order = np.argsort(np.abs(coef))[::-1]
    if top_n:
        order = order[:top_n]
    names_sorted = [feat_cols[i] for i in order]
    vals_sorted = coef[order]
    colors = ["forestgreen" if v > 0 else "tomato" for v in vals_sorted]
    ax.barh(range(len(vals_sorted)), vals_sorted, color=colors, alpha=0.85)
    ax.set_yticks(range(len(vals_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(f"Logistic Regression Coefficients  |  test_acc={logreg_r['test_acc']:.3f}", fontsize=8.5)
    ax.grid(axis="x", alpha=0.3)


def _plot_confidence_curve(ax, r):
    logreg_r = r["logreg"]
    thresholds = logreg_r["conf_thresholds"]
    ax2 = ax.twinx()
    l1, = ax.plot(thresholds, logreg_r["conf_accuracy"], "o-", color="royalblue", label="accuracy")
    ax.axhline(r["baseline"], color="gray", lw=0.8, ls="--", alpha=0.7)
    l2, = ax2.plot(thresholds, logreg_r["conf_coverage"], "s--", color="darkorange", label="coverage")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("accuracy", color="royalblue")
    ax2.set_ylabel("coverage (% of days traded)", color="darkorange")
    ax.legend(handles=[l1, l2], fontsize=7, loc="upper left")
    ax.set_title("Confidence-Thresholded Signal (LogReg)", fontsize=9)
    ax.grid(alpha=0.3)


def plot_dashboard(ticker, r):
    plt.style.use(PLOT_STYLE)
    fig = plt.figure(figsize=(17, 19))
    gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.32)
    p, svm_r, mlp_r, knn_r, rf_r, logreg_r = (
        r["perceptron"], r["svm"], r["mlp"], r["knn"], r["rf"], r["logreg"])

    # [0,0] Perceptron convergence
    _plot_perceptron(fig.add_subplot(gs[0, 0]), ticker, r)

    # [0,1] SVM: CV accuracy vs C, per kernel (best over gamma where applicable)
    ax1 = fig.add_subplot(gs[0, 1])
    kernel_colors = {"linear": "royalblue", "poly": "darkorange", "rbf": "forestgreen", "sigmoid": "tomato"}
    for kernel in SVM_KERNELS:
        C_vals, scores = svm_r["cv_curve"][kernel]
        ax1.plot(C_vals, scores, "o-", color=kernel_colors[kernel], lw=1.2, label=kernel)
    ax1.axhline(r["baseline"], color="gray", lw=0.8, ls="--", alpha=0.7)
    ax1.set_xscale("log")
    ax1.set_xlabel("C (log scale)"); ax1.set_ylabel("CV accuracy")
    ax1.legend(fontsize=6.5)
    ax1.set_title("SVM CV Accuracy vs C (per kernel, best gamma)", fontsize=8.5)
    ax1.grid(alpha=0.3)

    # [0,2] SVM decision boundary in 2D PCA space
    ax2 = fig.add_subplot(gs[0, 2])
    X2, clf2d = svm_r["X2_tr"], svm_r["clf2d"]
    xx_min, xx_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    yy_min, yy_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    XX, YY = np.mgrid[xx_min:xx_max:150j, yy_min:yy_max:150j]
    Z = clf2d.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    ax2.contourf(XX, YY, Z, levels=20, cmap=plt.cm.coolwarm, alpha=0.5)
    ax2.contour(XX, YY, Z, colors="k", levels=[0], linewidths=1.2)
    ax2.scatter(X2[:, 0], X2[:, 1], c=r["y_tr"], cmap=plt.cm.coolwarm, s=14, edgecolors="k", lw=0.3)
    gamma_str = f", gamma={svm_r['best_gamma']:.3g}" if svm_r["best_gamma"] is not None else ""
    ax2.set_title(f"SVM Boundary (2D PCA proj.)  |  {svm_r['best_kernel']}, C={svm_r['best_C']}{gamma_str}",
                  fontsize=8)
    ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2")

    # [1,0] KNN bias-variance curve
    _plot_knn(fig.add_subplot(gs[1, 0]), r)

    # [1,1] MLP capacity bias-variance curve
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(mlp_r["sizes"], mlp_r["train_scores"], "o-", color="royalblue", label="train")
    ax3.plot(mlp_r["sizes"], mlp_r["test_scores"], "s-", color="darkorange", label="test")
    ax3.axhline(r["baseline"], color="gray", lw=1.0, ls="--", label="naive baseline")
    ax3.set_xscale("log")
    ax3.set_xlabel("hidden layer size (log scale)"); ax3.set_ylabel("accuracy")
    ax3.legend(fontsize=7)
    ax3.set_title("MLP: Accuracy vs Capacity", fontsize=9)
    ax3.grid(alpha=0.3)

    # [1,2] Logistic Regression coefficients
    _plot_logreg_coef(fig.add_subplot(gs[1, 2]), r)

    # [2,0] Random Forest confusion matrix
    ax_cm = fig.add_subplot(gs[2, 0])
    cm = rf_r["confusion_matrix"]  # rows/cols ordered [1 (up), -1 (down)]
    ax_cm.imshow(cm, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(["Pred Up", "Pred Down"], fontsize=8)
    ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(["True Up", "True Down"], fontsize=8)
    ax_cm.set_title(f"Random Forest Confusion Matrix  |  test_acc={rf_r['test_acc']:.3f}", fontsize=8.5)

    # [2,1] Random Forest hyperparameter search results
    ax5 = fig.add_subplot(gs[2, 1])
    cvres = rf_r["cv_results"]
    depth_labels = cvres["param_max_depth"].fillna("None").astype(str)
    depth_codes = pd.Categorical(depth_labels).codes
    ax5.scatter(cvres["param_n_estimators"], cvres["mean_test_score"],
                c=depth_codes, cmap="viridis", s=40, alpha=0.85)
    ax5.set_xlabel("n_estimators"); ax5.set_ylabel("CV accuracy")
    ax5.set_title(f"RF RandomizedSearchCV ({RF_N_ITER} candidates)\n"
                   f"best: n_est={rf_r['best_params']['n_estimators']}, "
                   f"depth={rf_r['best_params']['max_depth']}", fontsize=8)
    ax5.grid(alpha=0.3)

    # [2,2] Confidence-thresholded signal
    _plot_confidence_curve(fig.add_subplot(gs[2, 2]), r)

    # [3,0] Model comparison bar
    ax4 = fig.add_subplot(gs[3, 0])
    names = ["Baseline", "Perceptron", "SVM", "MLP", "KNN", "RF", "LogReg"]
    vals = [r["baseline"], p["batch_test_acc"], svm_r["test_acc"], mlp_r["best_test_acc"],
            knn_r["test_acc"], rf_r["test_acc"], logreg_r["test_acc"]]
    colors = ["gray", "royalblue", "forestgreen", "darkorange", "purple", "brown", "teal"]
    ax4.bar(names, vals, color=colors, alpha=0.85)
    ax4.axhline(r["baseline"], color="gray", lw=0.8, ls="--")
    ax4.set_ylabel("test accuracy")
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis="x", labelsize=6.5, rotation=20)
    ax4.set_title("Model Comparison vs Naive Baseline", fontsize=9)
    ax4.grid(axis="y", alpha=0.3)

    # [3,1:3] Summary panel
    ax6 = fig.add_subplot(gs[3, 1:])
    ax6.axis("off")
    beats = any(v > r["baseline"] + 1e-9 for v in
                [p["batch_test_acc"], svm_r["test_acc"], mlp_r["best_test_acc"],
                 knn_r["test_acc"], rf_r["test_acc"], logreg_r["test_acc"]])
    best_conf_acc = np.nanmax(logreg_r["conf_accuracy"])
    gamma_disp = f"{svm_r['best_gamma']:.3g}" if svm_r["best_gamma"] is not None else "n/a (linear)"
    rows = [
        ["Naive baseline", f"{r['baseline']:.3f}"],
        ["Perceptron (batch) test acc", f"{p['batch_test_acc']:.3f}"],
        ["SVM best (kernel, C, gamma)", f"{svm_r['best_kernel']}, {svm_r['best_C']}, {gamma_disp}  ({svm_r['test_acc']:.3f})"],
        ["MLP best hidden size", f"{mlp_r['best_h']}  ({mlp_r['best_test_acc']:.3f})"],
        ["KNN best K", f"{knn_r['best_k']}  ({knn_r['test_acc']:.3f})"],
        ["RF best (n_est, depth)", f"{rf_r['best_params']['n_estimators']}, "
                                    f"{rf_r['best_params']['max_depth']}  ({rf_r['test_acc']:.3f})"],
        ["RF precision Up / Down", f"{rf_r['precision_up']:.2f} / {rf_r['precision_down']:.2f}"],
        ["LogReg test acc", f"{logreg_r['test_acc']:.3f}"],
        ["Best confidence-thresholded acc", f"{best_conf_acc:.3f}"],
        ["Any model beats baseline?", "YES" if beats else "NO"],
    ]
    table = ax6.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)

    fig.suptitle(f"{ticker} — Classification Trading Signals (Perceptron/SVM/MLP/KNN/RF/LogReg)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_perceptron_only(ticker, r):
    """Lightweight dashboard for tickers outside the TOP_N full-model subset."""
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    _plot_perceptron(axes[0], ticker, r)
    _plot_knn(axes[1], r)
    _plot_logreg_coef(axes[2], r, top_n=8)
    fig.suptitle(f"{ticker} — Perceptron + KNN + LogReg (baseline={r['baseline']:.3f})", fontsize=11)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 65)
    print("CLASSIFICATION TRADING SIGNALS")
    print("Perceptron/KNN/LogReg (all tickers) + SVM/MLP/RF (top volatility subset)")
    print("=" * 65)

    assets_data = load_all_assets(period_days=LOOKBACK_DAYS)
    vols = {t: df["log_return"].std() * np.sqrt(365) for t, df in assets_data.items()}   # crypto trades 24/7
    top_tickers = set(sorted(vols, key=vols.get, reverse=True)[:TOP_N])
    print(f"  Full SVM/MLP dashboard on top {TOP_N} by volatility: {sorted(top_tickers)}\n")

    summary = []
    for ticker, df in assets_data.items():
        full = ticker in top_tickers
        print(f"\n{'─'*50}")
        print(f"  {ticker}   {format_price(df['close'].iloc[-1])}  ({'full' if full else 'perceptron-only'})")
        print(f"{'─'*50}")

        try:
            r = analyse_asset(df, full_models=full)
            p = r["perceptron"]
            print(f"  Naive baseline (majority class): {r['baseline']:.3f}")
            print(f"  Perceptron batch : final_err={p['batch_errors'][-1]}  "
                  f"epochs_run={len(p['batch_errors'])}  test_acc={p['batch_test_acc']:.3f}"
                  f"{'  [WARNING: ended worse than start]' if p['regressed'] else ''}")
            print(f"  Perceptron seq   : final_err={p['seq_errors'][-1]}  "
                  f"epochs_run={len(p['seq_errors'])}  test_acc={p['seq_test_acc']:.3f}")

            knn_r = r["knn"]
            print(f"  KNN  : best_K={knn_r['best_k']}  test_acc={knn_r['test_acc']:.3f}")

            logreg_r = r["logreg"]
            best_conf_acc = np.nanmax(logreg_r["conf_accuracy"])
            print(f"  LogReg: test_acc={logreg_r['test_acc']:.3f}  "
                  f"best confidence-thresholded acc={best_conf_acc:.3f} "
                  f"(coverage {logreg_r['conf_coverage'][np.nanargmax(logreg_r['conf_accuracy'])]*100:.0f}%)")

            row = {
                "Ticker": ticker, "Mode": "full" if full else "perc-only",
                "Baseline": f"{r['baseline']:.3f}",
                "Perc_Batch_Acc": f"{p['batch_test_acc']:.3f}",
                "KNN_K": knn_r["best_k"], "KNN_Acc": f"{knn_r['test_acc']:.3f}",
                "LogReg_Acc": f"{logreg_r['test_acc']:.3f}",
            }

            if full:
                svm_r, mlp_r, rf_r = r["svm"], r["mlp"], r["rf"]
                gamma_disp = f"{svm_r['best_gamma']:.3g}" if svm_r["best_gamma"] is not None else "n/a"
                print(f"  SVM  : best=({svm_r['best_kernel']}, C={svm_r['best_C']}, gamma={gamma_disp})  "
                      f"test_acc={svm_r['test_acc']:.3f}")
                print(f"  MLP  : best_hidden={mlp_r['best_h']}  test_acc={mlp_r['best_test_acc']:.3f}")
                print(f"  RF   : best={rf_r['best_params']}  test_acc={rf_r['test_acc']:.3f}  "
                      f"precision(Up/Down)={rf_r['precision_up']:.2f}/{rf_r['precision_down']:.2f}")
                row.update({
                    "SVM_Kernel": svm_r["best_kernel"], "SVM_Gamma": gamma_disp, "SVM_Acc": f"{svm_r['test_acc']:.3f}",
                    "MLP_Hidden": mlp_r["best_h"], "MLP_Acc": f"{mlp_r['best_test_acc']:.3f}",
                    "RF_NEst": rf_r["best_params"]["n_estimators"],
                    "RF_Depth": str(rf_r["best_params"]["max_depth"]),
                    "RF_Acc": f"{rf_r['test_acc']:.3f}",
                })
                plot_dashboard(ticker, r)
            else:
                plot_perceptron_only(ticker, r)

            summary.append(row)

        except Exception as e:
            print(f"  Error: {e}")

    if summary:
        print("\n" + "=" * 65)
        print("CLASSIFICATION SIGNALS SUMMARY")
        print("=" * 65)
        print(pd.DataFrame(summary).to_string(index=False))

    print("\nClassification trading signals analysis complete.")


if __name__ == "__main__":
    main()
