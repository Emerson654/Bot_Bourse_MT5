"""Lecture multi-temporalités : votants calculés en H1, H4 et D1, ramenés sur la grille H1 sans look-ahead.

Une bougie H4 ou D1 n'est visible qu'à partir de la clôture de sa DERNIÈRE bougie H1 :
à la clôture H1 d'indice t, on voit la dernière bougie supérieure entièrement terminée ≤ t.
H4 et D1 sont reconstruites à partir des H1 (blocs de 4 h et journées en heure serveur).
"""

from __future__ import annotations

import csv
import gzip

import numpy as np

from bot.voters import Bars, compute_votes
from research.data import PARIS, server_to_utc

SESSION_ONLY = ("asia_range", "london_range", "ny_orb")  # n'ont de sens qu'en H1


def load_h1(path: str, point: float) -> tuple[Bars, np.ndarray, np.ndarray]:
    """Bougies H1 (heure de Paris) + horodatage serveur brut + UTC (datetime64[s])."""
    with gzip.open(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    ts = np.array([int(r["time"]) for r in rows])
    o, h, lo, c = (np.array([float(r[k]) for r in rows]) for k in ("open", "high", "low", "close"))
    spread = np.array([float(r["spread"]) for r in rows]) * point
    utc = [server_to_utc(int(t)) for t in ts]
    paris = [u.astimezone(PARIS) for u in utc]
    b = Bars(o, h, lo, c, spread, np.array([p.hour * 60 + p.minute for p in paris]),
             np.array([p.toordinal() for p in paris]), np.array([p.weekday() for p in paris]))
    return b, ts, np.array([u.replace(tzinfo=None) for u in utc], dtype="datetime64[s]")


def aggregate(b: Bars, key: np.ndarray) -> tuple[Bars, np.ndarray]:
    """Regroupe les H1 consécutives de même clé ; retourne les bougies agrégées et l'indice H1 de leur dernière bougie."""
    starts = np.flatnonzero(np.diff(key, prepend=key[0] - 1))
    ends = np.append(starts[1:], len(key)) - 1
    agg = Bars(b.o[starts], np.maximum.reduceat(b.h, starts), np.minimum.reduceat(b.lo, starts), b.c[ends],
               b.spread[ends], b.minute[ends], b.day[ends], b.weekday[ends])
    return agg, ends


def to_h1(values: np.ndarray, ends: np.ndarray, n: int) -> np.ndarray:
    """Valeur de la dernière bougie supérieure terminée, pour chaque bougie H1 (0 avant la première)."""
    out = np.zeros(n)
    idx = np.searchsorted(ends, np.arange(n), side="right") - 1  # dernière bougie dont la fin ≤ t
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def mtf_votes(b: Bars, server_ts: np.ndarray) -> dict[str, tuple[str, np.ndarray]]:
    """{« H1:rsi_14 » : (famille, votes)} pour H1, H4 et D1."""
    n = len(b.c)
    out = {f"H1:{k}": v for k, v in compute_votes(b).items()}
    for tf, key in (("H4", server_ts // (4 * 3600)), ("D1", server_ts // 86400)):
        agg, ends = aggregate(b, key)
        for k, (fam, v) in compute_votes(agg).items():
            if k not in SESSION_ONLY:
                out[f"{tf}:{k}"] = (fam, to_h1(v, ends, n))
    return out
