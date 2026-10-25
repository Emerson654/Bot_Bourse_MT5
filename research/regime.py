"""Détecteur de régime : le marché est-il en tendance ou en yo-yo ? (bougies journalières terminées uniquement)

- Efficacité du mouvement (Kaufman, « Smarter Trading », 1995) : ER = |c_t − c_{t−20}| / Σ |c_i − c_{i−1}|
  sur 20 jours, entre 0 (pur yo-yo) et 1 (ligne droite) ;
- ADX(14) journalier (Wilder, 1978) : force de la tendance, > 25 = tendance nette.
Les valeurs d'un jour ne sont visibles qu'après la clôture de ce jour (même règle que mtf.to_h1).
"""

from __future__ import annotations

import numpy as np

from bot import indicators as ind
from bot.voters import Bars
from research.mtf import aggregate, to_h1


def efficiency_ratio(c: np.ndarray, n: int = 20) -> np.ndarray:
    out = np.zeros(len(c))
    if len(c) <= n:
        return out
    path = np.convolve(np.abs(np.diff(c, prepend=c[0])), np.ones(n), mode="full")[: len(c)]
    move = np.zeros(len(c))
    move[n:] = np.abs(c[n:] - c[:-n])
    with np.errstate(invalid="ignore", divide="ignore"):
        out[n:] = np.where(path[n:] > 0, move[n:] / path[n:], 0.0)
    return out


def regime_h1(b: Bars, server_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ER 20 j, ADX 14 j) de la dernière journée terminée, pour chaque bougie H1."""
    d, ends = aggregate(b, server_ts // 86400)
    er = efficiency_ratio(d.c, 20)
    adx = np.nan_to_num(ind.adx(d.h, d.lo, d.c, 14)[0])
    return to_h1(er, ends, len(b.c)), to_h1(adx, ends, len(b.c))
