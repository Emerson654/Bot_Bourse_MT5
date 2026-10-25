"""Simulateur v8 : filtre de régime et stop remonté derrière le prix (jamais redescendu)."""

import numpy as np

from research.sim5 import simulate


def _run(c, er_val, er_min, trail_k):
    n = len(c)
    o = np.concatenate([[c[0]], c[:-1]])
    h, lo = np.maximum(o, c) + 0.01, np.minimum(o, c) - 0.01
    hours = np.arange(n)
    minute, day = (hours % 24) * 60, hours // 24
    weekday = np.zeros(n, np.int64)  # toujours lundi : décisions possibles chaque jour
    score = np.zeros(n)
    score[11] = 1.0  # signal d'achat à la clôture de 11h → entrée à 12h
    return simulate(o, h, lo, c, np.zeros(n), np.ones(n), minute, day, weekday, score, 0.5, 1.0, 2.0, 50.0, 500, True,
                    0.0, 0.0, 0.0, 2, 0.01, 0.01, 1.0, 1.0, 1, np.full(n, er_val), np.full(n, 30.0), er_min, 0.0, trail_k)


def test_filtre_de_regime_bloque_les_entrees():
    c = np.concatenate([np.full(12, 100.0), np.linspace(100, 90, 36)])  # chute : le stop ferme le trade
    assert len(_run(c, er_val=0.1, er_min=0.3, trail_k=0.0)[1]) == 0
    assert len(_run(c, er_val=0.5, er_min=0.3, trail_k=0.0)[1]) == 1


def test_stop_remonte_pour_garder_les_gains():
    # hausse de +6 (3 R) puis chute : sans sécurisation, on retombe au stop initial (−1 R)
    c = np.concatenate([np.full(12, 100.0), np.linspace(100, 106, 6), np.linspace(106, 90, 30)])
    sans = _run(c, 0.5, 0.0, 0.0)[1]
    avec = _run(c, 0.5, 0.0, 1.5)[1]
    assert sans[0, 3] < 0 < avec[0, 3]  # avec le stop remonté, le trade finit gagnant
