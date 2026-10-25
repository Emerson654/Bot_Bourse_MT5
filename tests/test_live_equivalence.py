"""Le bot live (historique limité à 30 000 bougies H1) voit les mêmes votes que le backtest (historique complet)."""

import json
import os

import numpy as np
import pytest

from research.macro import load_series, macro_votes
from research.mtf import load_h1, lower_tf_votes_from, mtf_votes
from bot.specialist import decision_rows

SYM = "USDJPY"
SRC = "data/mt5"


@pytest.mark.skipif(not os.path.exists(f"{SRC}/{SYM}_M5.csv.gz"), reason="historique absent")
def test_votes_live_egaux_au_backtest():
    point = json.load(open(f"{SRC}/specs.json"))[SYM]["point"]
    b, ts, _ = load_h1(f"{SRC}/{SYM}_H1.csv.gz", point)
    b5, ts5, _ = load_h1(f"{SRC}/{SYM}_M5.csv.gz", point)
    series = load_series("data/macro")

    def votes(bb, tt, bb5, tt5):
        v = mtf_votes(bb, tt)
        v.update(lower_tf_votes_from(bb5, tt5, tt))
        v.update(macro_votes(bb.day, bb.c, series))
        return v

    full = votes(b, ts, b5, ts5)
    n, n5 = len(b.c) - 30_000, len(b5.c) - 20_000  # fenêtre « live » (mêmes tailles que bot/live.py)
    sl = lambda x, k: type(x)(*(getattr(x, f)[k:] for f in x.__dataclass_fields__))  # noqa: E731
    live = votes(sl(b, n), ts[n:], sl(b5, n5), ts5[n5:])
    last = decision_rows(sl(b, n), 0, 6, -1)[-60:]  # 60 dernières décisions (~2 semaines)
    diffs = {k: np.abs(full[k][1][n + last] - live[k][1][last]).max() for k in full}
    bad = {k: round(float(d), 3) for k, d in diffs.items() if d > 0.05}
    assert not bad, f"votes différents entre live et backtest : {bad}"
