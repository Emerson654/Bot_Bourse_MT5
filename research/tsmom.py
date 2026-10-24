"""Test de contrôle : momentum de série temporelle classique, SANS aucune optimisation.

Règle fixée à l'avance (Moskowitz, Ooi & Pedersen, 2012, « Time Series Momentum » ;
Hurst, Ooi & Pedersen, 2017, « A Century of Evidence on Trend-Following Investing ») :
signal = moyenne des signes des rendements 1, 3 et 12 mois ; position ciblée à 10 %/√N de volatilité
annuelle par marché (vol 60 j), revue chaque vendredi ; coûts = spread payé sur chaque changement.
Aucun paramètre n'est choisi sur les données : toute la période est « hors échantillon ».

Usage : python -m research.tsmom data/mt5
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from research.mtf import load_h1
from research.prep3 import FALLBACK_BP, MIN_BARS, MIN_SPREAD

TARGET = float(os.environ.get("TSMOM_VOL", "0.10"))  # volatilité visée du portefeuille (marchés supposés indépendants)
COST = float(os.environ.get("TSMOM_COST", "1"))  # 0 = sans frais (diagnostic)


def daily_closes(path: str, point: float) -> tuple[np.ndarray, np.ndarray]:
    b, ts, _ = load_h1(path, point)
    day = ts // 86400  # journée serveur (clôture New York)
    last = np.flatnonzero(np.diff(day, append=day[-1] + 1))
    return day[last], b.c[last]


def main() -> None:
    src = sys.argv[1]
    specs = json.load(open(os.path.join(src, "specs.json")))
    series = {}
    for sym, sp in specs.items():
        if sp["bars"] < MIN_BARS:
            continue
        d, c = daily_closes(os.path.join(src, f"{sym}_H1.csv.gz"), sp["point"])
        cost = max(sp["spread_median_pts"] * sp["point"], MIN_SPREAD.get(sym, FALLBACK_BP * np.median(c)))
        series[sym] = (d, c, cost)
    days = np.unique(np.concatenate([s[0] for s in series.values()]))
    n = len(series)
    pnl = np.zeros(len(days))
    for sym, (d, c, cost) in series.items():
        idx = np.searchsorted(days, d)
        px = np.full(len(days), np.nan)
        px[idx] = c
        # prix « tenus » les jours sans cotation
        for i in range(1, len(px)):
            if np.isnan(px[i]):
                px[i] = px[i - 1]
        r = np.zeros(len(px))
        r[1:] = np.nan_to_num(px[1:] / px[:-1] - 1)
        valid = np.isfinite(px) & np.isfinite(np.roll(px, 1))
        first = int(np.argmax(valid)) + 1
        r[~valid] = 0.0
        # volatilité 60 j sur les seuls jours cotés ; rien avant 60 jours d'historique de l'actif
        vol = np.full(len(r), np.nan)
        for i in range(first + 60, len(r)):
            vol[i] = r[i - 60:i].std() * np.sqrt(260)
        sig = np.zeros(len(px))
        for lb in (21, 63, 252):
            m = np.zeros(len(px))
            m[lb:] = np.sign(px[lb:] / px[:-lb] - 1)
            sig += m / 3
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(vol > 0.02, sig * TARGET / np.sqrt(n) / vol, 0.0)
        w = np.nan_to_num(np.clip(w, -1, 1))
        # revue hebdomadaire (tous les 5 jours de cotation) : position tenue entre deux revues
        held = np.zeros(len(w))
        for i in range(1, len(w)):
            held[i] = w[i - 1] if i % 5 == 0 else held[i - 1]
        turnover = np.abs(np.diff(held, prepend=0.0))
        pnl += held * r - COST * turnover * cost / np.nan_to_num(px, nan=1.0)
    start = np.searchsorted(days, days[0] + 300)
    pnl = pnl[start:]
    yrs = (days[start:] * 86400).astype("datetime64[s]").astype("datetime64[Y]").astype(int) + 1970
    eq = np.cumprod(1 + pnl)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    print(f"{n} marchés, {yrs[0]}-{yrs[-1]} : rendement annuel {eq[-1] ** (260 / len(pnl)) - 1:.1%}, "
          f"volatilité {pnl.std() * np.sqrt(260):.1%}, Sharpe {pnl.mean() / pnl.std() * np.sqrt(260):.2f}, drawdown max {dd:.1%}")
    for y in np.unique(yrs):
        p = pnl[yrs == y]
        print(f"  {y} : {np.prod(1 + p) - 1:+.1%}")


if __name__ == "__main__":
    main()
