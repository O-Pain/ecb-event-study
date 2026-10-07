from __future__ import annotations

import os
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

EAMPD_URL = "https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx"
EAMPD_FILE = os.path.join(DATA_DIR, "Dataset_EA-MPD.xlsx")

HEADERS = {"User-Agent": "ecb-event-study/1.0 (academic replication)"}

WINDOWS = ["Press Release Window", "Press Conference Window", "Monetary Event Window"]


def _download(url: str, path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = requests.get(url, timeout=120, headers=HEADERS)
    r.raise_for_status()
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, path)
    return path


def load_eampd(window: str = "Monetary Event Window") -> pd.DataFrame:
    """Charge une fenetre de la base EA-MPD, dates en index datetime, en points de base."""
    if window not in WINDOWS:
        raise ValueError("fenetre inconnue : %s. Valeurs : %s" % (window, WINDOWS))
    _download(EAMPD_URL, EAMPD_FILE)
    df = pd.read_excel(EAMPD_FILE, sheet_name=window)
    # la colonne date melange des dates et, en fin de feuille, des notes
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def sovereign_columns(df: pd.DataFrame, countries=("DE", "FR", "IT", "ES"),
                      maturities=("2Y", "5Y", "10Y")) -> list[tuple[str, str, str]]:
    """Liste des (colonne, pays, maturite) presentes dans la base."""
    out = []
    for c in countries:
        for m in maturities:
            col = "%s%s" % (c, m)
            if col in df.columns:
                out.append((col, c, m))
    return out
