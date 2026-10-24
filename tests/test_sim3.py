"""Risque selon la conviction : un signal unanime mise plus qu'un signal tout juste au-dessus du seuil."""

import numpy as np

from research.sim3 import simulate


def _one_trade(score_value: float) -> float:
    """Un achat à 12h00 (Paris), stop touché la bougie suivante → perte = risque engagé."""
    n = 4
    o = np.array([100.0, 100.0, 100.0, 100.0])
    h = np.array([100.5, 100.5, 100.5, 100.5])
    lo = np.array([99.5, 99.5, 90.0, 99.5])  # la 3e bougie traverse le stop
    c = o.copy()
    minute = np.array([11 * 60, 12 * 60, 13 * 60, 14 * 60])
    day, wd = np.zeros(n, np.int64), np.zeros(n, np.int64)
    score = np.array([score_value, 0.0, 0.0, 0.0])
    _, tr = simulate(o, h, lo, c, np.zeros(n), np.ones(n), minute, day, wd, score, 0.5, 1.0, 2.0, 3.0, 0, True,
                     0.0, 0.0, 0.0, 2, 0.005, 0.02, 1.0)
    return float(-tr[0, 4])


def test_mise_proportionnelle_a_la_conviction():
    assert abs(_one_trade(0.5001) - 0.005) < 1e-4  # tout juste au-dessus du seuil : risque minimum
    assert abs(_one_trade(1.0) - 0.02) < 1e-9  # signal unanime : risque maximum
    assert abs(_one_trade(0.75) - 0.0125) < 1e-9  # à mi-chemin : au milieu
