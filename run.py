# -*- coding: utf-8 -*-
"""
Reproduit la specification de reference et compare trois modes d'inference.

    python run.py                       fenetre Monetary Event, surprise OIS_1Y
    python run.py --window "Press Conference Window"
    python run.py --surprise OIS_3M --no-winsor
    python run.py --cluster pays        clusteriser sur 4 pays au lieu de 12 cellules

Sorties dans outputs/ : baseline_cells.csv, panel_inference.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import event_study as es          # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--window", default="Monetary Event Window")
    p.add_argument("--surprise", default="OIS_1Y")
    p.add_argument("--cluster", default="cellule", choices=["cellule", "pays", "date"])
    p.add_argument("--no-winsor", action="store_true")
    args = p.parse_args()
    winsor = not args.no_winsor

    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 130)

    print("Fenetre  : %s" % args.window)
    print("Surprise : %s" % args.surprise)
    print("Winsorisation 1/99 : %s\n" % ("oui" if winsor else "non"))

    cells = es.baseline_by_cell(window=args.window, surprise=args.surprise, winsor=winsor)
    cells.to_csv(os.path.join(OUT, "baseline_cells.csv"), index=False)
    print("== Reponse par cellule (OLS, ecarts-types Newey-West 4 retards)")
    print(cells.to_string(index=False, float_format=lambda v: "%8.4f" % v))

    print("\n== Gradient de maturite : beta(2Y) > beta(5Y) > beta(10Y) ?")
    piv = cells.pivot(index="pays", columns="maturite", values="beta")
    order = [m for m in ("2Y", "5Y", "10Y") if m in piv.columns]
    piv = piv[order]
    piv["decroissant"] = piv.apply(lambda r: bool(all(
        r.iloc[i] > r.iloc[i + 1] for i in range(len(order) - 1))), axis=1)
    print(piv.to_string(float_format=lambda v: "%8.4f" % v))

    panel = es.build_panel(window=args.window, surprise=args.surprise, winsor=winsor)
    cl = {"cellule": "cellule", "pays": "pays", "date": "date"}[args.cluster]
    res = es.panel_with_entity_fe(panel, cluster=cl)
    pd.DataFrame([res]).to_csv(os.path.join(OUT, "panel_inference.csv"), index=False)

    print("\n== Panel a effets fixes d'entite, clusters sur '%s'" % res["cluster"])
    print("   observations           %d sur %d clusters" % (res["n"], res["n_clusters"]))
    print("   beta                   %.4f" % res["beta"])
    print("   SE clusterisee         %.4f   (t = %.2f)" % (res["se_cluster"], res["t_cluster"]))
    print("   SE Driscoll-Kraay      %.4f   (p = %.4f)"
          % (res["se_driscoll_kraay"], res["p_driscoll_kraay"]))
    print("   p asymptotique t(G-1)  %.4f" % res["p_asymptotique"])
    print("   p wild bootstrap       %.4f  (Rademacher, 999 tirages)" % res["p_wcb_rademacher"])
    print("   p wild bootstrap       %.4f  (Webb, 999 tirages)" % res["p_wcb_webb"])
    if res["n_clusters"] <= 6:
        print("\n   Avec %d clusters, Rademacher n'engendre que 2^%d vecteurs de signes :"
              % (res["n_clusters"], res["n_clusters"]))
        print("   les p-values sont quantifiees, la colonne Webb est celle a lire.")

    print("\nEcrit dans %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
