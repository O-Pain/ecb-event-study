# -*- coding: utf-8 -*-
"""
Inference module tests, based solely on simulated data.

The crucial test is the size test: under the null hypothesis, a test at the 5% level should reject the null in approximately 5% of cases.

The results demonstrate this effectively with G = 12 clusters: both methods maintain their nominal level, though the bootstrap is slightly conservative (rejecting in about 2% of cases, compared to 4% for the asymptotic method). The discrepancy between the two becomes significant only with a very small number of clusters, where the asymptotic method over-rejects; the current file does not simulate this scenario, but simply verifies that the bootstrap remains valid and that, at G = 4, it switches to exhaustive enumeration.

    python -m pytest tests -q          (or : python tests/test_inference.py)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import inference as inf                                   # noqa: E402


def simulate(rng, n_per_group=20, G=12, rho=1.0, beta1=0.0):
    """Erreurs a composante commune par cluster : c'est ce qui casse l'OLS naif."""
    codes = np.repeat(np.arange(G), n_per_group)
    n = codes.size
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    u = rho * rng.normal(size=G)[codes] + rng.normal(size=n)
    y = X @ np.array([0.5, beta1, 0.3]) + u
    return y, X, codes


def test_cluster_vcov_coherente():
    """La variance clusterisee complete et la forme vectorisee du bootstrap coincident."""
    rng = np.random.default_rng(0)
    y, X, codes = simulate(rng)
    beta, res = inf.ols(y, X)
    V = inf.cluster_vcov(X, res, codes)

    G, n, k = 12, X.shape[0], X.shape[1]
    xtx_inv = np.linalg.pinv(X.T @ X)
    Xc = X @ xtx_inv[:, 1]
    s = np.zeros(G)
    np.add.at(s, codes, Xc * res)
    v = (G / (G - 1)) * ((n - 1) / (n - k)) * (s ** 2).sum()
    assert abs(V[1, 1] - v) < 1e-12


def test_bootstrap_reproductible():
    rng = np.random.default_rng(1)
    y, X, codes = simulate(rng)
    a = inf.wild_cluster_bootstrap(y, X, codes, j=1, B=299, seed=7)
    b = inf.wild_cluster_bootstrap(y, X, codes, j=1, B=299, seed=7)
    assert a["p_wcb"] == b["p_wcb"]
    assert 0.0 <= a["p_wcb"] <= 1.0


def test_taille_du_test(n_sim=300, B=199, G=12, alpha=0.05):
    """Sous H0, le bootstrap doit rejeter pres de 5 % du temps, l'asymptotique plus."""
    rej_wcb = rej_asy = 0
    for s in range(n_sim):
        rng = np.random.default_rng(1000 + s)
        y, X, codes = simulate(rng, n_per_group=15, G=G, beta1=0.0)
        r = inf.wild_cluster_bootstrap(y, X, codes, j=1, B=B, seed=s)
        rej_wcb += int(r["p_wcb"] < alpha)
        rej_asy += int(r["p_asymptotique"] < alpha)
    taux_wcb = rej_wcb / n_sim
    taux_asy = rej_asy / n_sim
    print("   taille empirique : bootstrap %.3f, asymptotique %.3f (nominal %.2f)"
          % (taux_wcb, taux_asy, alpha))
    # tolerance large : 300 simulations, erreur type d'environ 1,3 point
    assert 0.01 <= taux_wcb <= 0.10, "taille du bootstrap hors de l'intervalle attendu"


def test_peu_de_clusters_enumeration_exhaustive():
    """Avec G = 4, Rademacher n'a que 2^4 = 16 vecteurs de signes : on les enumere
    tous, la p-value est exacte et tombe donc sur la grille des multiples de 1/16."""
    rng = np.random.default_rng(3)
    y, X, codes = simulate(rng, n_per_group=40, G=4)
    r = inf.wild_cluster_bootstrap(y, X, codes, j=1, B=999, weights="rademacher", seed=5)
    w = inf.wild_cluster_bootstrap(y, X, codes, j=1, B=999, weights="webb", seed=5)

    assert r["exhaustif"] is True and r["B"] == 16
    assert w["exhaustif"] is False
    grille = np.arange(0, 17) / 16.0
    assert np.min(np.abs(grille - r["p_wcb"])) < 1e-12, "p-value hors de la grille exacte"
    # deterministe : la graine ne change rien en enumeration exhaustive
    r2 = inf.wild_cluster_bootstrap(y, X, codes, j=1, B=999, weights="rademacher", seed=999)
    assert r2["p_wcb"] == r["p_wcb"]
    print("   G=4 : p Rademacher exacte %.4f (16 vecteurs), p Webb %.4f"
          % (r["p_wcb"], w["p_wcb"]))


def test_puissance():
    """Avec un vrai effet, le bootstrap doit rejeter."""
    rng = np.random.default_rng(11)
    y, X, codes = simulate(rng, n_per_group=30, G=20, beta1=0.8)
    r = inf.wild_cluster_bootstrap(y, X, codes, j=1, B=999, seed=2)
    assert r["p_wcb"] < 0.01


def test_driscoll_kraay_fonctionne():
    rng = np.random.default_rng(4)
    y, X, codes = simulate(rng)
    res = inf.driscoll_kraay(y, X, codes, maxlags=2)
    assert np.all(np.isfinite(res.bse))


if __name__ == "__main__":
    ok = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print("*", name)
            fn()
            ok += 1
    print("\n%d tests passes." % ok)
