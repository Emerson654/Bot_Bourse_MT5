"""Multi-temporalités : une bougie H4/D1 n'est visible qu'une fois terminée (aucun look-ahead)."""

import numpy as np

from research.mtf import aggregate, to_h1
from bot.voters import Bars


def test_bougie_superieure_visible_seulement_apres_sa_cloture():
    n = 12
    b = Bars(*(np.arange(n, dtype=float),) * 5, np.zeros(n), np.zeros(n), np.zeros(n))
    key = np.arange(n) // 4  # blocs de 4 bougies H1 = 1 bougie H4
    agg, ends = aggregate(b, key)
    assert list(ends) == [3, 7, 11]
    assert list(agg.c) == [3.0, 7.0, 11.0]
    v = to_h1(np.array([10.0, 20.0, 30.0]), ends, n)
    # avant la fin de la 1re bougie H4 (indices 0-2) : rien ; à partir de 3 : sa valeur
    assert list(v) == [0, 0, 0, 10, 10, 10, 10, 20, 20, 20, 20, 30]
