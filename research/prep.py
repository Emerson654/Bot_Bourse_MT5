"""Prépare le cache du tournoi : votes calculés sur tout l'historique, puis restreints aux bougies de session.

Usage : python -m research.prep data/XAUUSD_M5_utc.csv.gz cache.pkl
"""

from __future__ import annotations

import pickle
import sys
import time

import numpy as np

from bot import indicators as ind
from bot.voters import compute_votes
from research.data import load_m5


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    b, utc = load_m5(src)
    print(f"{len(b.c)} bougies chargées ({time.time() - t0:.0f} s)", flush=True)
    votes = compute_votes(b)
    print(f"{len(votes)} votants calculés ({time.time() - t0:.0f} s)", flush=True)
    a = ind.atr(b.h, b.lo, b.c, 14)
    # Bougies utiles : du lundi au vendredi, 11h55-18h00 Paris (la décision de 11h55 ouvre à 12h00)
    keep = (b.weekday < 5) & (b.minute >= 11 * 60 + 55) & (b.minute < 18 * 60) & np.isfinite(a)
    names = list(votes)
    cache = {
        "o": b.o[keep], "h": b.h[keep], "lo": b.lo[keep], "c": b.c[keep], "spread": b.spread[keep],
        "atr": a[keep], "minute": b.minute[keep], "day": b.day[keep], "time": utc[keep],
        "V": np.column_stack([votes[k][1][keep] for k in names]).astype(np.float32),
        "names": names, "families": [votes[k][0] for k in names],
    }
    with open(out, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"{keep.sum()} bougies de session, {len(np.unique(cache['day']))} jours, "
          f"spread médian {np.median(cache['spread']):.3f} $ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
