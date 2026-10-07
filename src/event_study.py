from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D
from . import inference as inf


def winsorise(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def baseline_by_cell(window: str = "Monetary Event Window",
                     surprise: str = "OIS_1Y",
                     countries=("DE", "FR", "IT", "ES"),
                     maturities=("2Y", "5Y", "10Y"),
                     maxlags: int = 4,
                     winsor: bool = True) -> pd.DataFrame:
    """Une regression par cellule pays x maturite. Retourne un tableau de resultats."""
    df = D.load_eampd(window)
    if surprise not in df.columns:
        raise KeyError("colonne de surprise absente : %s" % surprise)

    rows = []
    for col, c, m in D.sovereign_columns(df, countries, maturities):
        sub = df[[surprise, col]].dropna()
        if len(sub) < 30:
            continue
        s = sub[surprise]
        y = winsorise(sub[col]) if winsor else sub[col]
        X = np.column_stack([np.ones(len(sub)), s.to_numpy()])
        res = inf.hac_ols(y.to_numpy(), X, maxlags=maxlags)
        rows.append({
            "pays": c, "maturite": m, "n": len(sub),
            "beta": res.params[1], "se_hac": res.bse[1],
            "t": res.tvalues[1], "p": res.pvalues[1], "r2": res.rsquared,
        })
    return pd.DataFrame(rows)


def build_panel(window: str = "Monetary Event Window",
                surprise: str = "OIS_1Y",
                countries=("DE", "FR", "IT", "ES"),
                maturities=("2Y", "5Y", "10Y"),
                winsor: bool = True) -> pd.DataFrame:
    """Empile les cellules pays x maturite en un panel long."""
    df = D.load_eampd(window)
    frames = []
    for col, c, m in D.sovereign_columns(df, countries, maturities):
        sub = df[[surprise, col]].dropna().rename(columns={col: "dy"})
        if winsor:
            sub["dy"] = winsorise(sub["dy"])
        sub = sub.assign(pays=c, maturite=m, cellule="%s%s" % (c, m))
        frames.append(sub.reset_index())
    panel = pd.concat(frames, ignore_index=True)
    return panel.rename(columns={surprise: "surprise"})


def panel_with_entity_fe(panel: pd.DataFrame, cluster: str = "cellule") -> dict:
    """
    Panel a effets fixes d'entite, avec les trois inferences sur le meme beta.

    Les effets fixes sont introduits par indicatrices, ce qui garde la matrice X
    explicite et permet de passer exactement le meme X aux trois estimateurs.
    """
    d = panel.dropna(subset=["dy", "surprise"]).copy()
    dummies = pd.get_dummies(d["cellule"], drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(d)), d["surprise"].to_numpy(), dummies.to_numpy()])
    y = d["dy"].to_numpy()
    groups = d[cluster].to_numpy()
    time = pd.to_datetime(d["date"]).astype("int64").to_numpy()

    t_cl, beta, se_cl = inf.cluster_t(y, X, groups, j=1)
    dk = inf.driscoll_kraay(y, X, time, maxlags=4)
    rade = inf.wild_cluster_bootstrap(y, X, groups, j=1, B=999, weights="rademacher")
    webb = inf.wild_cluster_bootstrap(y, X, groups, j=1, B=999, weights="webb")

    return {
        "n": len(d), "n_clusters": int(pd.unique(groups).size), "cluster": cluster,
        "beta": float(beta),
        "se_cluster": float(se_cl), "t_cluster": float(t_cl),
        "se_driscoll_kraay": float(dk.bse[1]), "p_driscoll_kraay": float(dk.pvalues[1]),
        "p_wcb_rademacher": rade["p_wcb"], "p_wcb_webb": webb["p_wcb"],
        "p_asymptotique": rade["p_asymptotique"],
    }
