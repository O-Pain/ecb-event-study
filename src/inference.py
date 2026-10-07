# -*- coding: utf-8 -*-
"""
Robust inference for linear regressions on grouped data.

Three variance estimators and one bootstrap method.

- `hac_ols`            : OLS with Newey-West standard errors (time series).
- `cluster_vcov`       : CRV1 clustered variance, with the standard small-sample
                         correction G/(G-1) x (N-1)/(N-K).
- `driscoll_kraay`     : OLS with Driscoll-Kraay variance, robust to cross-sectional
                         dependence (via statsmodels, cov_type "nw-groupsum").
- `wild_cluster_bootstrap` : Wild cluster bootstrap with imposed null hypothesis
                         (Cameron, Gelbach, and Miller 2008; Roodman et al. 2019).

Why the bootstrap? With a small number of clusters, asymptotic clustered
standard errors are downward-biased, and the t-test rejects the null hypothesis
far too often. The wild cluster bootstrap corrects this. With Rademacher weights,
the number of distinct sign vectors is 2^G; with G = 4, there are only 16,
resulting in p-values ​​quantified in steps of 1/16. This is the rationale behind
Webb's (2014) six-point weights, which yield 6^G possible draws.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ols", "cluster_vcov", "cluster_t", "wild_cluster_bootstrap",
           "hac_ols", "driscoll_kraay"]


# --------------------------------------------------------------------------- base
def ols(y: np.ndarray, X: np.ndarray):
    """Retourne (beta, residus). Resolution par moindres carres."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def _group_index(groups: np.ndarray):
    """Retourne (codes entiers 0..G-1, G)."""
    _, codes = np.unique(np.asarray(groups), return_inverse=True)
    return codes, int(codes.max()) + 1


def cluster_vcov(X: np.ndarray, u: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Variance clusterisee CRV1."""
    X = np.asarray(X, float)
    u = np.asarray(u, float).ravel()
    codes, G = _group_index(groups)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)

    meat = np.zeros((k, k))
    for g in range(G):
        m = codes == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)

    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return c * xtx_inv @ meat @ xtx_inv


def cluster_t(y, X, groups, j: int, r: float = 0.0):
    """t de Student clusterise pour H0 : beta_j = r."""
    beta, u = ols(np.asarray(y, float).ravel(), np.asarray(X, float))
    V = cluster_vcov(X, u, groups)
    se = float(np.sqrt(V[j, j]))
    return float((beta[j] - r) / se), float(beta[j]), se


# --------------------------------------------------------------------------- bootstrap
_WEIGHTS = {
    # deux points, equiprobables
    "rademacher": (np.array([-1.0, 1.0]), np.array([0.5, 0.5])),
    # six points de Webb (2014)
    "webb": (np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                       np.sqrt(0.5), 1.0, np.sqrt(1.5)]), np.full(6, 1 / 6)),
}


def wild_cluster_bootstrap(y, X, groups, j: int, r: float = 0.0, B: int = 999,
                           weights: str = "rademacher", seed: int = 42) -> dict:
    """
    Bootstrap sauvage clusterise a hypothese nulle imposee, pour H0 : beta_j = r.

    Procedure. On estime le modele contraint en retirant la colonne j et en
    retranchant r * x_j de la variable dependante. Les residus contraints sont
    reechantillonnes en multipliant, cluster par cluster, par un poids aleatoire.
    Sur chaque echantillon bootstrap on reestime le modele complet et on calcule
    le t clusterise centre sur r. La p-value bilaterale symetrique est la part
    des |t*| au moins aussi grands que le |t| observe.

    Retourne un dictionnaire : t_stat, beta, se, p_wcb, p_asymptotique, B, G.
    """
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    n, k = X.shape
    codes, G = _group_index(groups)
    if weights not in _WEIGHTS:
        raise ValueError("poids inconnus : %s" % weights)
    vals, probs = _WEIGHTS[weights]

    # 1. statistique observee
    t_obs, beta_j, se_j = cluster_t(y, X, groups, j, r)

    # 2. modele contraint
    keep = [c for c in range(k) if c != j]
    Xr = X[:, keep]
    y_adj = y - r * X[:, j]
    beta_r, _ = ols(y_adj, Xr)
    fitted = Xr @ beta_r + r * X[:, j]
    u_r = y - fitted

    # 3. quantites reutilisables : X est fixe d'un tirage a l'autre
    xtx_inv = np.linalg.pinv(X.T @ X)
    A = xtx_inv @ X.T                      # beta* = A y*
    c_j = xtx_inv[:, j]                    # V_jj = c_j' meat c_j
    corr = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))

    # Avec peu de clusters, Rademacher n'a que 2^G vecteurs de signes possibles.
    # Les enumerer tous rend la p-value exacte et deterministe, au lieu de tirer
    # 999 fois dans un ensemble de 16 elements.
    exhaustif = (weights == "rademacher") and (2 ** G <= B)
    if exhaustif:
        import itertools
        w_g = np.array(list(itertools.product([-1.0, 1.0], repeat=G)), dtype=float).T
    else:
        rng = np.random.default_rng(seed)
        w_g = rng.choice(vals, size=(G, B), p=probs)    # un poids par cluster et par tirage

    B_eff = w_g.shape[1]
    Ystar = fitted[:, None] + u_r[:, None] * w_g[codes, :]     # (n, B_eff)

    Bhat = A @ Ystar                                    # (k, B_eff)
    U = Ystar - X @ Bhat                                # (n, B_eff)

    # score par cluster projete sur c_j : s_gb = c_j' (X_g' u_gb)
    Xc = X @ c_j                                        # (n,)
    contrib = Xc[:, None] * U                           # (n, B_eff)
    s = np.zeros((G, B_eff))
    np.add.at(s, codes, contrib)                        # somme par cluster
    var_j = corr * (s ** 2).sum(axis=0)                 # (B,)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_star = (Bhat[j, :] - r) / np.sqrt(var_j)
    t_star = t_star[np.isfinite(t_star)]

    p = float((np.abs(t_star) >= abs(t_obs) - 1e-12).mean()) if t_star.size else float("nan")

    from scipy import stats
    p_asy = float(2 * (1 - stats.t.cdf(abs(t_obs), df=max(G - 1, 1))))

    return {"t_stat": t_obs, "beta": beta_j, "se": se_j, "p_wcb": p,
            "p_asymptotique": p_asy, "B": int(t_star.size), "G": G,
            "weights": weights, "exhaustif": bool(exhaustif)}


# --------------------------------------------------------------------------- statsmodels
def hac_ols(y, X, maxlags: int = 4):
    """OLS avec ecarts-types Newey-West."""
    import statsmodels.api as sm
    return sm.OLS(np.asarray(y, float).ravel(), np.asarray(X, float)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})


def driscoll_kraay(y, X, time_ids, maxlags: int = 4):
    """OLS avec variance Driscoll-Kraay (robuste a la dependance transversale)."""
    import statsmodels.api as sm
    t = np.asarray(time_ids)
    _, tcodes = np.unique(t, return_inverse=True)
    return sm.OLS(np.asarray(y, float).ravel(), np.asarray(X, float)).fit(
        cov_type="nw-groupsum",
        cov_kwds={"time": tcodes, "maxlags": maxlags, "groups": None})
