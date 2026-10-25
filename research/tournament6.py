"""Tournoi 6 : un spécialiste par actif, paris de quelques heures à 1 mois, jusqu'à 3 positions simultanées.

Réutilise tout le moteur du tournoi 3-5 (ligues, fusion, finale, walk-forward, votants techniques
5 temporalités + actu/macro) en remplaçant le simulateur (sim4) et en ajoutant deux gènes :
échelle du stop (horizon du pari) et nombre de positions simultanées.

Usage : python -m research.tournament6 CACHE OUT_JSON TEST_YEAR --asset XAUUSD [--gens 60 --pop 300]
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace

import research.tournament as T
import research.tournament3 as T3
from research.sim4 import simulate

T.TSTOPS = (24, 48, 72, 120, 240, 360, 528)  # horizon max en bougies H1 : 1 jour → 1 mois (22 jours de bourse)
SCALES = (1.0, 2.0, 5.0)
_rand, _mut, _init0 = T3.random_genome, T3.mutate, T3._init
ASSET: list[str] = []


def random_genome(rng: random.Random, pool: list[int]) -> T.Genome:
    return replace(_rand(rng, pool), atr_scale=rng.choice(SCALES), max_pos=rng.choice([1, 2, 3]))


def mutate(g: T.Genome, rng: random.Random, pool: list[int], rate: float = 0.3) -> T.Genome:
    g = _mut(g, rng, pool, rate)
    if rng.random() < rate:
        g = replace(g, atr_scale=rng.choice(SCALES))
    if rng.random() < rate / 2:
        g = replace(g, max_pos=rng.choice([1, 2, 3]))
    return g


def run_asset(A: dict, V, g: T.Genome, a: int, b: int):
    import numpy as np

    dec = A["dec"]
    da, db = np.searchsorted(dec, a), np.searchsorted(dec, b)
    s = np.zeros(b - a)
    s[dec[da:db] - a] = T3.score_of(V[da:db], g)
    sl = slice(a, b)
    return simulate(A["o"][sl], A["h"][sl], A["lo"][sl], A["c"][sl], A["spread"][sl], A["atr"][sl], A["minute"][sl],
                    A["day"][sl], A["weekday"][sl], s, g.entry, g.exit, g.stop_k, g.tp_r, max(g.tstop, 24), bool(g.short),
                    A["slip"], A["swap_long"], A["swap_short"], A["triple_wd"], g.risk_lo, g.risk_hi, g.risk_pow,
                    g.atr_scale, g.max_pos)


def _init(cache: str) -> None:
    """Charge le cache puis ne garde que l'actif du spécialiste."""
    _init0(cache)
    if ASSET:
        T3._G["assets"] = {k: v for k, v in T3._G["assets"].items() if k in ASSET}


T3.random_genome, T3.mutate, T3.run_asset, T3._init = random_genome, mutate, run_asset, _init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("test_year", type=int)
    ap.add_argument("--asset", required=True)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--risk-max", type=float, default=0.02, help="risque max par pari (tournoi 7 : 0,01)")
    ap.add_argument("--dd-max", type=float, default=0.15, help="drawdown toléré par spécialiste (tournoi 7 : 0,10)")
    a = ap.parse_args()
    T3.RISKS = tuple(r for r in (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02) if r <= a.risk_max)
    T3.MAX_DD = a.dd_max
    ASSET.append(a.asset)
    rep = T3.run_fold(a.cache, a.test_year, a.gens, a.pop, a.workers)
    rep["asset"] = a.asset
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    w = rep["winner"]
    print(json.dumps({"actif": a.asset, "poids": w["weights"], "valid": w["valid"], "test": w.get("test")}, indent=1, default=float))


if __name__ == "__main__":
    main()
