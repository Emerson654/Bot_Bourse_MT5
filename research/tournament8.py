"""Tournoi 8 : spécialistes simples, robustes, qui ne tradent que les tendances et laissent courir les gains.

Corrections des défauts mesurés au tournoi 7 :
1. suroptimisation (Sharpe 1,7 à l'entraînement → −0,3 en test) : 5 indicateurs maximum par bot, et la note
   d'entraînement privilégie la PIRE des 3 périodes (0,3 × moyenne + 0,7 × pire) : un bot doit marcher partout ;
2. marchés en yo-yo (EUR/USD, GBP/USD 2019-2023) : filtre de régime (efficacité du mouvement, ADX journalier) ;
3. gains trop petits (0,54 R gagné contre 0,60 R perdu) : objectif ≥ 1,5 × le stop, et stop remonté derrière
   le prix une fois 1 R atteint pour laisser courir les gagnants.
Moteur et protocole identiques aux tournois 6-7 (walk-forward, risque ≤ 1 %, drawdown ≤ 10 % par actif).

Usage : python -m research.tournament8 CACHE OUT_JSON TEST_YEAR --asset XAUUSD [--gens 60 --pop 300]
"""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np

import research.tournament as T
import research.tournament3 as T3
import research.tournament6 as T6
from research.sim5 import simulate

T.TPS = (1.5, 2.0, 3.0, 4.0, 6.0)  # objectif ≥ 1,5 × stop
MAX_VOTERS = 5
ER_MINS, ADX_MINS, TRAILS = (0.0, 0.2, 0.3, 0.4), (0.0, 20.0, 25.0), (0.0, 1.5, 2.5, 4.0)
_rand6, _mut6, _cross = T6.random_genome, T6.mutate, T.crossover


def _simple(g: T.Genome) -> T.Genome:
    """Garde au plus 5 indicateurs (les plus pondérés)."""
    if len(g.idx) <= MAX_VOTERS:
        return g
    keep = sorted(np.argsort(g.w)[::-1][:MAX_VOTERS])
    return replace(g, idx=tuple(g.idx[k] for k in keep), w=tuple(g.w[k] for k in keep))


def random_genome(rng: random.Random, pool: list[int]) -> T.Genome:
    g = _rand6(rng, pool)
    return _simple(replace(g, tp_r=rng.choice(T.TPS), er_min=rng.choice(ER_MINS), adx_min=rng.choice(ADX_MINS),
                           trail_k=rng.choice(TRAILS)))


def mutate(g: T.Genome, rng: random.Random, pool: list[int], rate: float = 0.3) -> T.Genome:
    g = _mut6(g, rng, pool, rate)
    if rng.random() < rate:
        g = replace(g, er_min=rng.choice(ER_MINS))
    if rng.random() < rate:
        g = replace(g, adx_min=rng.choice(ADX_MINS))
    if rng.random() < rate:
        g = replace(g, trail_k=rng.choice(TRAILS))
    return _simple(replace(g, tp_r=max(g.tp_r, 1.5)))


def crossover(a: T.Genome, b: T.Genome, rng: random.Random) -> T.Genome:
    return _simple(_cross(a, b, rng))


def run_asset(A: dict, V, g: T.Genome, a: int, b: int):
    dec = A["dec"]
    da, db = np.searchsorted(dec, a), np.searchsorted(dec, b)
    s = np.zeros(b - a)
    s[dec[da:db] - a] = T3.score_of(V[da:db], g)
    sl = slice(a, b)
    return simulate(A["o"][sl], A["h"][sl], A["lo"][sl], A["c"][sl], A["spread"][sl], A["atr"][sl], A["minute"][sl],
                    A["day"][sl], A["weekday"][sl], s, g.entry, g.exit, g.stop_k, max(g.tp_r, 1.5), max(g.tstop, 24),
                    bool(g.short), A["slip"], A["swap_long"], A["swap_short"], A["triple_wd"], g.risk_lo, g.risk_hi,
                    g.risk_pow, g.atr_scale, g.max_pos, A["er"][sl], A["adx"][sl], g.er_min, g.adx_min, g.trail_k)


def era_fitness(g: T.Genome, eras, vkey: str = "V") -> float:
    """Robustesse : 0,3 × moyenne + 0,7 × pire des 3 périodes d'entraînement."""
    fs = [T3.fitness_of(T3.metrics(T3.portfolio(g, e, vkey))) for e in eras]
    return 0.3 * float(np.mean(fs)) + 0.7 * float(np.min(fs))


T3.random_genome, T3.mutate, T3.run_asset, T3.era_fitness = random_genome, mutate, run_asset, era_fitness
T.crossover = T3.crossover = crossover

if __name__ == "__main__":
    T6.main()
