"""Simulateur v6 : horizon maximum respecté et nombre de positions simultanées plafonné."""

import numpy as np

from research.sim4 import simulate


def _market(n_days: int = 60):
    hours = np.arange(n_days * 24)
    minute = (hours % 24) * 60
    day = hours // 24
    weekday = day % 7  # 0 = lundi
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 0.05, len(hours)))
    o = np.roll(c, 1)
    o[0] = c[0]
    return o, np.maximum(o, c) + 0.01, np.minimum(o, c) - 0.01, c, minute, day, weekday


def _run(score, max_pos, tstop):
    o, h, lo, c, minute, day, wd = _market()
    n = len(c)
    return simulate(o, h, lo, c, np.zeros(n), np.full(n, 0.05), minute, day, wd, score, 0.5, 1.0, 50.0, 50.0, tstop,
                    True, 0.0, 0.0, 0.0, 2, 0.01, 0.01, 1.0, 1.0, max_pos)


def test_horizon_et_positions_simultanees():
    n = 60 * 24
    score = np.zeros(n)
    score[::7] = 1.0  # signaux d'achat répétés (le score retombe entre deux)
    _, tr = _run(score, max_pos=3, tstop=120)
    assert len(tr) > 0
    # sortie à la première décision après l'horizon (au pire le lundi 12h si l'échéance tombe le week-end)
    assert (tr[:, 1] - tr[:, 0]).max() <= 120 + 72
    # jamais plus de 3 positions ouvertes en même temps
    events = sorted([(a, 1) for a in tr[:, 0]] + [(b, -1) for b in tr[:, 1]], key=lambda x: (x[0], x[1]))
    live, peak = 0, 0
    for _, e in events:
        live += e
        peak = max(peak, live)
    assert peak <= 3
    _, tr1 = _run(score, max_pos=1, tstop=120)
    assert len(tr1) < len(tr)
