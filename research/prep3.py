"""Cache du tournoi v3 : pour chaque actif, bougies H1 + votants H1/H4/D1 + coûts réels du courtier.

Usage : python -m research.prep3 data/mt5 cache3.pkl
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np

from bot import indicators as ind
from research.mtf import load_h1, lower_tf_votes, mtf_votes

START = np.datetime64("2009-01-01")
# Planchers de spread en prix, niveau d'un courtier grand public (la démo MetaQuotes affiche parfois 0)
MIN_SPREAD = {"XAUUSD": 0.20, "XAGUSD": 0.03, "EURUSD": 0.0001, "GBPUSD": 0.00015, "USDJPY": 0.012,
              "AUDUSD": 0.00012, "USDCHF": 0.00015, "US500": 0.5, "DE40": 1.5, "USTEC": 1.5}
FALLBACK_BP = 1.5e-4  # actif absent du tableau : spread minimum = 1,5 point de base du prix
MIN_BARS = 40_000  # ~6 ans de H1 minimum pour participer


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    with open(os.path.join(src, "specs.json")) as fh:
        specs = json.load(fh)
    assets, names, families = {}, None, None
    for sym, sp in specs.items():
        if sp["bars"] < MIN_BARS:
            print(f"{sym} : historique trop court ({sp['bars']} bougies), écarté")
            continue
        t0 = time.time()
        b, server_ts, utc = load_h1(os.path.join(src, f"{sym}_H1.csv.gz"), sp["point"])
        votes = mtf_votes(b, server_ts)
        m5 = os.path.join(src, f"{sym}_M5.csv.gz")
        if os.path.exists(m5):  # 5 min et 15 min, si l'historique a été téléchargé
            votes.update(lower_tf_votes(m5, sp["point"], server_ts))
        a = ind.atr(b.h, b.lo, b.c, 14)
        med = sp["spread_median_pts"] * sp["point"]
        floor = max(med, MIN_SPREAD.get(sym, FALLBACK_BP * float(np.median(b.c[-5000:]))))
        spread = np.maximum(b.spread, floor)
        keep = (utc >= START) & np.isfinite(a)
        # swap → prix par unité et par nuit : mode 1 = points ; modes 2-4 = devise par lot (÷ taille du contrat)
        if sp["swap_mode"] == 1:
            sw = (sp["swap_long"] * sp["point"], sp["swap_short"] * sp["point"])
        elif sp["swap_mode"] in (2, 3, 4):
            sw = (sp["swap_long"] / sp["contract"], sp["swap_short"] / sp["contract"])
        else:
            sw = (0.0, 0.0)
        if sp["swap_mode"] not in (1, 2, 3, 4):
            print(f"{sym} : swap en mode {sp['swap_mode']} non modélisé")
        if names is None or len(votes) > len(names):
            names, families = list(votes), [votes[k][0] for k in votes]
        assets[sym] = {
            "o": b.o[keep], "h": b.h[keep], "lo": b.lo[keep], "c": b.c[keep], "spread": spread[keep], "atr": a[keep],
            "minute": b.minute[keep], "day": b.day[keep], "weekday": b.weekday[keep], "time": utc[keep],
            "votes": {k: v[1][keep].astype(np.float32) for k, v in votes.items()},
            "slip": 0.25 * floor, "swap_long": sw[0], "swap_short": sw[1],
            "triple_wd": (sp["swap_3days"] - 1) % 7,  # MT5 : 0 = dimanche … 3 = mercredi → lundi = 0
        }
        print(f"{sym} : {keep.sum()} bougies H1, {len(votes)} votants, spread {np.median(spread[keep]):.5f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    for A in assets.values():
        # Les votes ne servent qu'aux bougies de décision (clôtures H1 de la session) : on ne garde que celles-là
        m, d = A["minute"], A["day"]
        dec = np.flatnonzero((m[1:] >= 12 * 60) & (m[1:] <= 17 * 60) & (d[1:] == d[:-1]))
        A["dec"] = dec
        vs = A.pop("votes")  # un actif sans historique 5 min a des votants M5/M15 neutres (0)
        A["V"] = np.column_stack([vs[k][dec] if k in vs else np.zeros(len(dec), np.float32) for k in names])
    with open(out, "wb") as fh:
        pickle.dump({"assets": assets, "names": names, "families": families}, fh)


if __name__ == "__main__":
    main()
