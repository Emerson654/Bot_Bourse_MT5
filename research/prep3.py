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
from research.mtf import load_h1, mtf_votes

START = np.datetime64("2009-01-01")
MIN_SPREAD = {"XAUUSD": 0.20}  # plancher prudent en prix (la démo MetaQuotes est plus serrée qu'un vrai courtier)
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
        a = ind.atr(b.h, b.lo, b.c, 14)
        med = sp["spread_median_pts"] * sp["point"]
        spread = np.maximum(b.spread, max(med, MIN_SPREAD.get(sym, 0.0)))
        keep = (utc >= START) & np.isfinite(a)
        # swap en points (mode 1) → prix par unité et par nuit ; autres modes ignorés (signalés)
        sw = (sp["swap_long"] * sp["point"], sp["swap_short"] * sp["point"]) if sp["swap_mode"] == 1 else (0.0, 0.0)
        if sp["swap_mode"] != 1:
            print(f"{sym} : swap en mode {sp['swap_mode']} non modélisé")
        names = names or list(votes)
        families = families or [votes[k][0] for k in names]
        assets[sym] = {
            "o": b.o[keep], "h": b.h[keep], "lo": b.lo[keep], "c": b.c[keep], "spread": spread[keep], "atr": a[keep],
            "minute": b.minute[keep], "day": b.day[keep], "weekday": b.weekday[keep], "time": utc[keep],
            "V": np.column_stack([votes[k][1][keep] for k in names]).astype(np.float32),
            "slip": 0.25 * max(med, MIN_SPREAD.get(sym, 0.0)), "swap_long": sw[0], "swap_short": sw[1],
            "triple_wd": (sp["swap_3days"] - 1) % 7,  # MT5 : 0 = dimanche … 3 = mercredi → lundi = 0
        }
        print(f"{sym} : {keep.sum()} bougies H1, {len(names)} votants, spread {np.median(spread[keep]):.5f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    with open(out, "wb") as fh:
        pickle.dump({"assets": assets, "names": names, "families": families}, fh)


if __name__ == "__main__":
    main()
