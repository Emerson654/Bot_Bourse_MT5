"""Votants « actu et macro » : géopolitique, incertitude économique, taux, dollar, peur, pétrole…

Pour un actif et une série (ex. risque géopolitique) :
- choc_t = variation sur 5 jours de la série / écart-type de ces variations sur 250 jours (z-score) ;
- lien_t = corrélation, sur les 250 jours PRÉCÉDENTS, entre le rendement de l'actif et la variation de la série
  (positive pour l'or face aux tensions, négative pour les actions…) ;
- vote_t = choc_t × lien_t × 5, borné à [-1, 1] : la série vote dans le sens où elle a récemment poussé l'actif.
Sans look-ahead : une série n'est utilisée qu'après sa publication (FRED et EPU : J+2 par prudence ; GPR : J+7, mise à
jour hebdomadaire), et la corrélation n'utilise que des jours déjà terminés.
"""

from __future__ import annotations

import csv
import os
from datetime import date

import numpy as np

LAG_DAYS = {"risque_geopolitique": 7}  # par défaut : 2 jours (publication FRED/EPU parfois tardive)
SMOOTH = {"risque_geopolitique": 5, "incertitude_eco": 5}  # séries d'articles de presse, bruitées : moyenne 5 j


def load_series(src: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for f in sorted(os.listdir(src)) if os.path.isdir(src) else []:
        if f.endswith(".csv"):
            with open(os.path.join(src, f)) as fh:
                rows = list(csv.DictReader(fh))
            d = np.array([date.fromisoformat(r["date"]).toordinal() for r in rows])
            v = np.array([float(r["value"]) for r in rows])
            k = SMOOTH.get(f[:-4], 1)
            if k > 1:
                v = np.convolve(v, np.ones(k) / k, mode="full")[: len(v)]
            out[f[:-4]] = (d, v)
    return out


def _rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        out[i] = np.nanstd(x[i - n:i])
    return out


def _rolling_corr(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Corrélation sur la fenêtre [t-n, t) : le jour t lui-même n'est pas utilisé."""
    out = np.zeros(len(a))
    for i in range(n, len(a)):
        x, y = a[i - n:i], b[i - n:i]
        if x.std() > 0 and y.std() > 0:
            out[i] = np.corrcoef(x, y)[0, 1]
    return out


def macro_votes(day: np.ndarray, close: np.ndarray, series: dict) -> dict[str, tuple[str, np.ndarray]]:
    """`day` = date (ordinal, Paris) de chaque bougie H1 ; retourne les votes ramenés sur les bougies H1."""
    days = np.unique(day)
    last = np.searchsorted(day, days, side="right") - 1
    cd = close[last]
    r = np.zeros(len(days))
    r[1:] = np.log(cd[1:] / cd[:-1])
    pos = np.searchsorted(days, day)  # indice du jour de chaque bougie H1
    out = {}
    for name, (sd, sv) in series.items():
        lag = LAG_DAYS.get(name, 2)
        j = np.searchsorted(sd, days - lag, side="right") - 1  # dernière valeur publiée
        x = np.where(j >= 0, sv[np.maximum(j, 0)], np.nan)
        dx = np.zeros(len(x))
        dx[1:] = np.nan_to_num(x[1:] - x[:-1])
        c5 = np.full(len(x), np.nan)
        c5[5:] = x[5:] - x[:-5]
        z = np.nan_to_num(c5 / _rolling_std(c5, 250))
        link = _rolling_corr(r, dx, 250)
        vote = np.clip(z * link * 5, -1, 1)
        out[f"MACRO:{name}"] = ("macro", vote[pos])
    return out
