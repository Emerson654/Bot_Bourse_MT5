"""Faux courtier qui imite l'API `MetaTrader5` : tests unitaires et mode simulation sur Mac.

Les constantes ont les mêmes valeurs que le module officiel. Des pannes peuvent être simulées
(`drop_sltp` : le serveur ignore SL/TP à l'ouverture ; `fail_modify` : la modification échoue).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace


@dataclass
class Position:
    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    sl: float
    tp: float
    magic: int
    profit: float = 0.0


@dataclass
class SimBroker:
    bid: float = 2400.0
    spread: float = 0.30
    equity: float = 100_000.0
    drop_sltp: bool = False
    fail_modify: bool = False
    positions: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    _ticket: int = 1000

    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_REJECT = 10006

    def symbol_info(self, symbol):
        return SimpleNamespace(name=symbol, point=0.01, digits=2, volume_min=0.01, volume_step=0.01, volume_max=100.0,
                               trade_stops_level=10, filling_mode=1, spread=int(self.spread / 0.01))

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=self.bid, ask=self.bid + self.spread, time=0)

    def account_info(self):
        return SimpleNamespace(equity=self.equity, balance=self.equity, currency="EUR")

    def order_calc_profit(self, kind, symbol, volume, p0, p1):
        # 1 lot = 100 onces ; profit en USD converti 1:1 (suffisant pour la simulation)
        return (p1 - p0) * 100 * volume * (1 if kind == self.ORDER_TYPE_BUY else -1)

    def positions_get(self, symbol=None, ticket=None):
        return [p for p in self.positions if (symbol is None or p.symbol == symbol) and (ticket is None or p.ticket == ticket)]

    def order_send(self, req):
        self.requests.append(req)
        done = SimpleNamespace(retcode=self.TRADE_RETCODE_DONE)
        if req["action"] == self.TRADE_ACTION_SLTP:
            if self.fail_modify:
                return SimpleNamespace(retcode=self.TRADE_RETCODE_REJECT)
            for p in self.positions_get(ticket=req["position"]):
                p.sl, p.tp = req["sl"], req["tp"]
            return done
        if "position" in req:  # fermeture
            self.positions = [p for p in self.positions if p.ticket != req["position"]]
            return done
        self._ticket += 1
        kind = self.POSITION_TYPE_BUY if req["type"] == self.ORDER_TYPE_BUY else self.POSITION_TYPE_SELL
        sl, tp = (0.0, 0.0) if self.drop_sltp else (req.get("sl", 0.0), req.get("tp", 0.0))
        self.positions.append(Position(self._ticket, req["symbol"], kind, req["volume"], req["price"], sl, tp, req["magic"]))
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=self._ticket)
