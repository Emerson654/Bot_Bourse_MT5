"""Bot live XAUUSD : une session de 12h à 17h50 (Paris), décision à chaque clôture de bougie M5.

Usage :
  python -m bot.main            # compte MT5 (identifiants dans MT5_LOGIN / MT5_PASSWORD / MT5_SERVER)
  python -m bot.main --sim      # simulation locale sans MT5 (Mac), marché aléatoire accéléré
Logs limités à : heure, sens, prix, SL, TP, résultat (en R et en %). Jamais de solde ni d'identifiant.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

import numpy as np

from bot import execution as ex
from bot import news, risk, session
from bot.strategy import final_score, last_atr, load_params
from bot.voters import Bars

SYMBOL = os.environ.get("MT5_SYMBOL", "XAUUSD")
N_BARS = 6000  # ~3 semaines de M5 : chauffe suffisante pour tous les indicateurs
log = logging.getLogger("bot")


# ---------------------------------------------------------------------------------------------
# Connexion


def connect_mt5():
    import MetaTrader5 as mt5

    ok = mt5.initialize(path=os.environ.get("MT5_PATH"), login=int(os.environ["MT5_LOGIN"].strip()),
                        password=os.environ["MT5_PASSWORD"].strip(), server=os.environ["MT5_SERVER"].strip(),
                        timeout=120_000, portable=True)
    if not ok:
        sys.exit(f"Connexion MT5 impossible (code {mt5.last_error()[0]})")
    if not mt5.symbol_select(SYMBOL, True):
        sys.exit(f"Symbole {SYMBOL} indisponible")
    return mt5


def server_offset_hours(api) -> int:
    """Décalage heure serveur − UTC, mesuré sur le dernier tick (arrondi à l'heure)."""
    t = api.symbol_info_tick(SYMBOL).time
    return round((t - time.time()) / 3600) if t else 0


def fetch_bars(api, offset_h: int) -> Bars:
    """Bougies M5 clôturées (position 1 = dernière bougie terminée), horodatées en heure de Paris."""
    r = api.copy_rates_from_pos(SYMBOL, api.TIMEFRAME_M5, 1, N_BARS)
    point = api.symbol_info(SYMBOL).point
    paris = [datetime.fromtimestamp(int(t) - offset_h * 3600, UTC).astimezone(session.PARIS) for t in r["time"]]
    return Bars(r["open"].astype(float), r["high"].astype(float), r["low"].astype(float), r["close"].astype(float),
                r["spread"].astype(float) * point, np.array([p.hour * 60 + p.minute for p in paris]),
                np.array([p.toordinal() for p in paris]), np.array([p.weekday() for p in paris]))


# ---------------------------------------------------------------------------------------------
# Statut et journal (aucune donnée de compte)


class Reporter:
    def __init__(self, publish: bool):
        self.publish = publish
        self.trades: list[dict] = []
        self.status: dict = {}
        os.makedirs("status_site", exist_ok=True)

    def trade(self, **kw) -> None:
        self.trades.append(kw)
        log.info(" | ".join(f"{k}={v}" for k, v in kw.items()))

    def update(self, state: str, **kw) -> None:
        self.status = {"etat": state, "maj": datetime.now(session.PARIS).strftime("%Y-%m-%d %H:%M"),
                       "trades_du_jour": self.trades[-30:], **kw}
        with open("status_site/status.json", "w", encoding="utf-8") as fh:
            json.dump(self.status, fh, ensure_ascii=False, indent=1)
        if self.publish:  # branche « status » servie par GitHub Pages ; un seul commit écrasé à chaque fois
            subprocess.run("git add -A && git commit -q --amend -m statut --allow-empty && git push -q -f origin HEAD:status",
                           shell=True, cwd="status_site", check=False, capture_output=True)

    def save_journal(self, path: str = "journal/trades.csv") -> None:
        if not self.trades:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = [t for t in self.trades if t.get("evenement") == "sortie"]
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["heure", "sens", "entree", "sortie", "sl", "tp", "r", "pct", "raison"],
                               extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(rows)


# ---------------------------------------------------------------------------------------------
# Boucle de session


def today_start_equity(api) -> float:
    """Capital au début de la journée = capital actuel − résultat du jour (réalisé + latent) du bot."""
    eq = api.account_info().equity
    midnight = datetime.now(session.PARIS).replace(hour=0, minute=0, second=0, microsecond=0)
    deals = getattr(api, "history_deals_get", lambda *a: [])(midnight.astimezone(UTC), datetime.now(UTC)) or []
    realized = sum(d.profit + d.commission + d.swap for d in deals if d.magic == ex.MAGIC)
    floating = sum(p.profit for p in ex.bot_positions(api, SYMBOL))
    return eq - realized - floating


def run_session(api, params: dict, rep: Reporter, sim_clock=None) -> None:
    clock = sim_clock or (lambda: datetime.now(UTC))
    job_start = clock()
    events = news.load_events()
    offset = server_offset_hours(api) if sim_clock is None else 0
    ex.check_all_positions(api, SYMBOL)
    last_exit: datetime | None = None
    entry_info: dict[int, dict] = {}
    weights = params["winner"]["weights"]
    while True:
        now = clock()
        if session.must_stop(now, job_start):
            for p in ex.bot_positions(api, SYMBOL):
                _exit(api, p, entry_info, rep, "fin de session")
            rep.update("arrêté (fin de session)", poids=weights)
            return
        b = fetch_bars(api, offset)
        s, rules = final_score(b, params)
        score = float(s[-1])
        a = last_atr(b)
        start_eq = today_start_equity(api)
        halted = risk.daily_loss_hit(start_eq, api.account_info().equity)
        pos = ex.bot_positions(api, SYMBOL)
        for p in pos:
            ex.ensure_protection(api, p)
            info = entry_info.get(p.ticket, {})
            d = 1 if p.type == api.POSITION_TYPE_BUY else -1
            held = (now - info.get("t", now)).total_seconds() / 300
            reverse = (d == 1 and score < -rules.exit) or (d == -1 and score > rules.exit)
            if reverse or (rules.tstop and held >= rules.tstop):
                _exit(api, p, entry_info, rep, "signal inverse" if reverse else "stop temporel")
                last_exit = now
        # Sorties chez le courtier (SL/TP touchés) : la position a disparu
        for t in [t for t in entry_info if t not in {p.ticket for p in ex.bot_positions(api, SYMBOL)}]:
            _record_broker_exit(api, t, entry_info, rep)
            last_exit = now
        state = "en pause (perte du jour ≥ 3 %)" if halted else "en marche"
        can = (session.can_open(now, job_start) and not halted and not ex.bot_positions(api, SYMBOL)
               and not (api.positions_get(symbol=SYMBOL) or [])
               and not (rules.news and news.blocked(now, events))
               and (last_exit is None or now - last_exit >= timedelta(minutes=5 * rules.cooldown)))
        if can:
            d = 1 if score > rules.entry else (-1 if rules.short and score < -rules.entry else 0)
            if d:
                _enter(api, d, a, rules, score, now, entry_info, rep)
        elif rules.news and news.blocked(now, events):
            state = "en pause (annonce économique)"
        _status(api, rep, state, score, weights, entry_info)
        _sleep_to_next_bar(clock, sim_clock)


def _enter(api, d: int, atr: float, rules, score: float, now: datetime, entry_info: dict, rep: Reporter) -> None:
    info = api.symbol_info(SYMBOL)
    tick = api.symbol_info_tick(SYMBOL)
    rules_b = risk.SymbolRules(info.point, info.volume_min, info.volume_step, info.volume_max,
                               info.trade_stops_level, info.filling_mode)
    spread = tick.ask - tick.bid
    dist = risk.stop_distance(atr, rules.stop_k, rules_b, spread)
    price = tick.ask if d == 1 else tick.bid
    sl = round(price - d * dist, info.digits)
    tp = round(price + d * rules.tp_r * dist, info.digits)
    value = abs(api.order_calc_profit(api.ORDER_TYPE_BUY, SYMBOL, 1.0, price, price + 1.0))
    vol = risk.lot_size(api.account_info().equity, risk.RISK_PER_TRADE, dist, value, rules_b)
    if vol <= 0:
        log.info("trade ignoré : volume minimal supérieur au risque autorisé")
        return
    ticket = ex.open_position(api, SYMBOL, d, vol, sl, tp)
    if ticket:
        entry_info[ticket] = {"t": now, "price": price, "dist": dist, "d": d, "sl": sl, "tp": tp}
        rep.trade(evenement="entree", heure=_hm(now), sens="achat" if d == 1 else "vente", prix=price, sl=sl, tp=tp,
                  score=round(score, 2))


def _exit(api, p, entry_info: dict, rep: Reporter, why: str) -> None:
    tick = api.symbol_info_tick(SYMBOL)
    if ex.close_position(api, p, why):
        price = tick.bid if p.type == api.POSITION_TYPE_BUY else tick.ask
        _log_exit(entry_info.pop(p.ticket, None), price, why, rep)


def _record_broker_exit(api, ticket: int, entry_info: dict, rep: Reporter) -> None:
    info = entry_info.pop(ticket)
    deals = getattr(api, "history_deals_get", lambda **k: [])(position=ticket) or []
    price = deals[-1].price if deals else info["sl"]
    why = "take-profit" if abs(price - info["tp"]) < abs(price - info["sl"]) else "stop-loss"
    _log_exit(info, price, why, rep)


def _log_exit(info: dict | None, price: float, why: str, rep: Reporter) -> None:
    if not info:
        return
    r = info["d"] * (price - info["price"]) / info["dist"]
    rep.trade(evenement="sortie", heure=_hm(datetime.now(UTC)), sens="achat" if info["d"] == 1 else "vente",
              entree=info["price"], sortie=price, sl=info["sl"], tp=info["tp"], r=round(r, 2),
              pct=round(r * risk.RISK_PER_TRADE * 100, 2), raison=why)


def _status(api, rep: Reporter, state: str, score: float, weights: dict, entry_info: dict) -> None:
    pos = None
    for p in ex.bot_positions(api, SYMBOL):
        info = entry_info.get(p.ticket)
        tick = api.symbol_info_tick(SYMBOL)
        if info:
            px = tick.bid if info["d"] == 1 else tick.ask
            pos = {"sens": "achat" if info["d"] == 1 else "vente", "entree": info["price"], "sl": p.sl, "tp": p.tp,
                   "r_latent": round(info["d"] * (px - info["price"]) / info["dist"], 2)}
    rep.update(state, score=round(score, 3), position=pos, poids=weights)


def _hm(t: datetime) -> str:
    return t.astimezone(session.PARIS).strftime("%Y-%m-%d %H:%M")


def _sleep_to_next_bar(clock, sim_clock) -> None:
    if sim_clock is not None:
        return
    now = time.time()
    time.sleep(300 - now % 300 + 3)  # 3 s après la clôture de la bougie


def backtest_validated(path: str = "results/backtest.json") -> bool:
    try:
        with open(path, encoding="utf-8") as fh:
            return bool(json.load(fh).get("validated"))
    except FileNotFoundError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="simulation locale sans MT5")
    ap.add_argument("--publish", action="store_true", help="publier le statut sur la branche status")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M")
    rep = Reporter(a.publish)
    if not backtest_validated():
        log.info("Le dernier backtest n'est pas validé (results/backtest.json) : le bot ne démarre pas.")
        rep.update("arrêté (backtest non validé)")
        return
    params = load_params()
    if a.sim:
        from bot.sim_market import run_simulation
        run_simulation(params, rep)
        return
    if not session.is_trading_day(session.now_paris()):
        rep.update("hors session (week-end)")
        return
    api = connect_mt5()
    wait = session.seconds_until_open(datetime.now(UTC))
    if wait > 0:
        rep.update(f"en attente de l'ouverture (12h00), dans {int(wait // 60)} min")
        time.sleep(wait)
    try:
        run_session(api, params, rep)
    finally:
        rep.save_journal()
        api.shutdown()


if __name__ == "__main__":
    main()
