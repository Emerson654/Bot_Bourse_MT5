"""Mode simulation (Mac, sans MT5) : rejoue la dernière séance de l'historique bougie par bougie.

Le faux courtier avance d'une bougie M5 à chaque décision du bot et déclenche SL/TP sur le plus
haut / plus bas de la bougie, comme le ferait le serveur. Usage : python -m bot.main --sim
"""

from __future__ import annotations

import csv
import gzip
from datetime import UTC, datetime, timedelta

import numpy as np

from bot.sim_broker import SimBroker

DATA = "data/XAUUSD_M5_utc.csv.gz"


class SimMarket(SimBroker):
    TIMEFRAME_M5 = 5

    def __init__(self, rates: np.ndarray, start: int):
        super().__init__()
        self.rates, self.i = rates, start
        self._sync()

    def _sync(self) -> None:
        self.bid = float(self.rates["close"][self.i - 1])

    def copy_rates_from_pos(self, symbol, tf, pos, n):
        return self.rates[max(0, self.i - pos - n + 1): self.i - pos + 1]

    def advance(self) -> None:
        """Bougie suivante : SL/TP déclenchés côté « serveur »."""
        r = self.rates[self.i]
        for p in list(self.positions):
            buy = p.type == self.POSITION_TYPE_BUY
            hit = (r["low"] <= p.sl or r["high"] >= p.tp) if buy else \
                  (r["high"] + self.spread >= p.sl or r["low"] + self.spread <= p.tp)
            if hit:
                self.positions.remove(p)
        self.i += 1
        self._sync()


def load_rates(path: str = DATA) -> np.ndarray:
    with gzip.open(path, "rt") as fh:
        rows = list(csv.DictReader(fh))[-12_000:]
    dt = [("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"), ("close", "f8"), ("spread", "f8")]
    return np.array([(int(r["time_utc"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
                      float(r["spread"]) / 0.01) for r in rows], dtype=dt)


def run_simulation(params: dict, rep) -> None:
    from bot.main import run_session
    from bot.session import PARIS

    rates = load_rates()
    # Première bougie de 12h00 (Paris) de la dernière séance complète
    times = [datetime.fromtimestamp(int(t), UTC).astimezone(PARIS) for t in rates["time"]]
    starts = [i for i, t in enumerate(times) if t.hour == 12 and t.minute == 0 and i > 6000]
    if not starts:
        raise SystemExit("Pas assez d'historique pour simuler une séance")
    mk = SimMarket(rates, starts[-2])

    def clock() -> datetime:
        return datetime.fromtimestamp(int(rates["time"][mk.i - 1]), UTC) + timedelta(minutes=5)

    def tick_clock() -> datetime:
        t = clock()
        if mk.i < len(rates) - 1:
            mk.advance()
        return t

    run_session(mk, params, rep, sim_clock=tick_clock)
    print(f"Simulation terminée : {sum(1 for t in rep.trades if t.get('evenement') == 'sortie')} trades")
