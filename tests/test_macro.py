"""Votants actu/macro : une donnée publiée le jour D n'influence rien avant D + délai de publication."""

import numpy as np

from research.macro import macro_votes


def test_aucune_donnee_avant_sa_publication():
    rng = np.random.default_rng(0)
    days = np.arange(738000, 738700)
    day = np.repeat(days, 3)  # 3 bougies H1 par jour
    close = np.exp(np.cumsum(rng.normal(0, 0.01, len(day))))
    sv = np.cumsum(rng.normal(0, 1, len(days)))
    base = macro_votes(day, close, {"taux": (days, sv)})["MACRO:taux"][1]
    choc = sv.copy()
    choc[600:] += 50.0  # énorme choc publié le jour 600
    after = macro_votes(day, close, {"taux": (days, choc)})["MACRO:taux"][1]
    first_seen = np.flatnonzero(base != after)
    assert first_seen.size and day[first_seen[0]] >= days[600] + 2  # visible au plus tôt à J+2
