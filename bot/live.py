"""Bot live (tournois 6-7) : un spécialiste par actif, paris de quelques heures à 1 mois.

Une session par jour ouvré, 12h00-17h50 (Paris), heures communes de Londres et New York :
- à chaque heure pleine de 12h à 17h, chaque spécialiste recalcule son score (5 temporalités + actu/macro) ;
  il ferme ses positions sur signal inverse ou horizon atteint, et ouvre un pari sur un nouveau signal ;
- SL et TP sont envoyés DANS l'ordre d'ouverture (gardés par le courtier la nuit et le week-end) ;
- toutes les 5 min : détection des positions fermées par SL/TP, statut, notifications Telegram ;
- garde-fous : perte du jour ≤ 3 %, risque total ouvert ≤ 6 %, levier ≤ 20, positions max par actif.
Démarrage : bot validé par le backtest, OU compte de démo (argent fictif, accord d'Emerson du 25/10/2026).
En réel, un bot non validé ne démarre jamais.

Usage : python -m bot.live [--publish]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import time
import traceback
from datetime import UTC, datetime

from bot import execution as ex
from bot import risk, session
from bot.specialist import decide, decision_rows, order_plan, regime_ok, signal_of, trailed_stop
from bot.telegram import Telegram, esc, price
from research.macro import load_series
from research.mtf import bars_from_arrays

DAILY_LOSS, TOTAL_RISK_CAP, MAX_LEV = 0.03, 0.06, 20.0
H1_BARS, M5_BARS = 30_000, 20_000  # ~5 ans de H1 (moyennes journalières longues) et ~10 semaines de M5
log = logging.getLogger("bot")


class Live:
    def __init__(self, api, params: dict, tg: Telegram, publish: bool):
        self.api, self.params, self.tg, self.publish = api, params, tg, publish
        self.series = load_series("data/macro")
        self.known: dict[int, dict] = {}  # positions suivies : ticket → infos d'ouverture
        self.closed: list[dict] = []
        self.ccy = api.account_info().currency
        os.makedirs("status_site", exist_ok=True)

    # ------------------------------------------------------------------ outils MT5
    def value_per_unit(self, sym: str, px: float) -> float:
        """Gain/perte en devise du compte pour 1 lot et 1,00 de prix."""
        return abs(self.api.order_calc_profit(self.api.ORDER_TYPE_BUY, sym, 1.0, px, px + 1.0) or 0.0)

    def risk_money(self, p) -> float:
        return abs(self.api.order_calc_profit(self.api.ORDER_TYPE_BUY, p.symbol, p.volume, p.sl, p.price_open) or 0.0) if p.sl else 0.0

    def bot_positions(self) -> list:
        return [p for p in self.api.positions_get() or [] if p.magic == ex.MAGIC]

    def start_equity(self) -> float:
        eq = self.api.account_info().equity
        midnight = datetime.now(session.PARIS).replace(hour=0, minute=0, second=0, microsecond=0)
        deals = self.api.history_deals_get(midnight.astimezone(UTC), datetime.now(UTC)) or []
        realized = sum(d.profit + d.commission + d.swap for d in deals if d.magic == ex.MAGIC)
        floating = sum(p.profit for p in self.bot_positions())
        return eq - realized - floating

    # ------------------------------------------------------------------ suivi des positions
    def track_existing(self) -> None:
        for p in self.bot_positions():
            self.known.setdefault(p.ticket, {"sym": p.symbol, "d": 1 if p.type == self.api.POSITION_TYPE_BUY else -1,
                                             "entry": p.price_open, "risk_money": self.risk_money(p), "sl": p.sl, "tp": p.tp})

    def detect_closures(self) -> None:
        live = {p.ticket for p in self.bot_positions()}
        for t in [t for t in self.known if t not in live]:
            info = self.known.pop(t)
            deals = self.api.history_deals_get(position=t) or []
            out = [d for d in deals if d.entry == self.api.DEAL_ENTRY_OUT]
            pnl = sum(d.profit + d.commission + d.swap for d in deals)
            px = out[-1].price if out else info["entry"]
            why = info.get("why") or ("take-profit" if info["tp"] and abs(px - info["tp"]) < abs(px - info["sl"]) else "stop-loss")
            r = pnl / info["risk_money"] if info["risk_money"] else 0.0
            row = {"heure": now_str(), "actif": info["sym"], "sens": "achat" if info["d"] == 1 else "vente",
                   "entree": info["entry"], "sortie": px, "r": round(r, 2), "raison": why}
            self.closed.append(row)
            log.info("sortie %s %s %s R=%.2f (%s)", row["heure"], info["sym"], row["sens"], r, why)
            icon = "✅" if pnl >= 0 else "❌"
            self.tg.send(f"{icon} <b>{esc(info['sym'])}</b> {row['sens']} fermé ({esc(why)})\n"
                         f"Entrée {price(info['entry'])} → sortie {price(px)}\n"
                         f"Résultat : <b>{pnl:+,.2f} {self.ccy}</b> ({r:+.2f} R)".replace(",", " "))

    # ------------------------------------------------------------------ décision horaire
    def decide_symbol(self, sym: str, params: dict, start_eq: float) -> None:
        api = self.api
        info = api.symbol_info(sym)
        for _ in range(6):  # le terminal télécharge l'historique à la demande : on lui laisse le temps
            h1 = api.copy_rates_from_pos(sym, api.TIMEFRAME_H1, 1, H1_BARS)
            m5 = api.copy_rates_from_pos(sym, api.TIMEFRAME_M5, 1, M5_BARS)
            if h1 is not None and m5 is not None and len(h1) >= 0.9 * H1_BARS and len(m5) >= 0.9 * M5_BARS:
                break
            time.sleep(10)
        cur = api.copy_rates_from_pos(sym, api.TIMEFRAME_H1, 0, 1)
        if h1 is None or m5 is None or cur is None or len(h1) < 2000:
            log.warning("%s : historique indisponible", sym)
            return
        b, ts, _ = bars_from_arrays(h1["time"], h1["open"], h1["high"], h1["low"], h1["close"], h1["spread"], info.point)
        b5, ts5, _ = bars_from_arrays(m5["time"], m5["open"], m5["high"], m5["low"], m5["close"], m5["spread"], info.point)
        nb, _, _ = bars_from_arrays(cur["time"], cur["open"], cur["high"], cur["low"], cur["close"], cur["spread"], info.point)
        dec = decision_rows(b, int(nb.minute[0]), int(nb.weekday[0]), int(nb.day[0]))
        if len(dec) < 50 or dec[-1] != len(b.c) - 1:
            return  # la bougie en cours n'est pas un moment de décision
        D = decide(params, b, ts, b5, ts5, self.series, dec)
        g = D.genome
        now_ts = int(cur["time"][0])
        mine = [p for p in self.bot_positions() if p.symbol == sym]
        # 1) sorties : signal inverse ou horizon atteint
        for p in mine:
            d = 1 if p.type == api.POSITION_TYPE_BUY else -1
            held_h = (now_ts - int(p.time)) / 3600
            reverse = (d == 1 and D.score < -g["exit"]) or (d == -1 and D.score > g["exit"])
            if reverse or held_h >= max(g["tstop"], 24):
                why = "signal inverse" if reverse else "horizon atteint"
                if p.ticket in self.known:
                    self.known[p.ticket]["why"] = why
                ex.close_position(api, p, why)
        mine = [p for p in self.bot_positions() if p.symbol == sym]
        # 1 bis) laisser courir les gagnants : stop remonté CHEZ LE COURTIER dès 1 R de gain (jamais redescendu)
        tick = api.symbol_info_tick(sym)
        for p in mine:
            d = 1 if p.type == api.POSITION_TYPE_BUY else -1
            init = init_distance(p)
            new = trailed_stop(D, d, tick.bid if d == 1 else tick.ask, p.price_open, init, p.sl)
            if new is None:
                continue
            new = round(new, info.digits)
            res = api.order_send({"action": api.TRADE_ACTION_SLTP, "symbol": sym, "position": p.ticket,
                                  "sl": new, "tp": p.tp, "magic": ex.MAGIC})
            if res is not None and res.retcode == api.TRADE_RETCODE_DONE:
                if p.ticket in self.known:
                    self.known[p.ticket]["sl"] = new
                self.tg.send(f"🔒 <b>{esc(sym)}</b> : stop remonté à {price(new)} pour garder les gains")
        mine = [p for p in self.bot_positions() if p.symbol == sym]
        # 2) entrée sur signal frais, seulement si le marché est en tendance (filtre de régime)
        if not regime_ok(D):
            return
        sig = signal_of(D.score, g)
        if sig == 0 or sig == D.prev_signal or len(mine) >= g["max_pos"] or any(
                (1 if p.type == api.POSITION_TYPE_BUY else -1) != sig for p in mine):
            return
        acc = api.account_info()
        if risk.daily_loss_hit(start_eq, acc.equity, DAILY_LOSS):
            log.info("pause : perte du jour ≥ 3 %")
            return
        open_risk = sum(self.risk_money(p) for p in self.bot_positions()) / acc.equity
        tick = api.symbol_info_tick(sym)
        px = tick.ask if sig == 1 else tick.bid
        dist, tp_dist, r, force = order_plan(D, sig, px, tick.ask - tick.bid)
        rules = risk.SymbolRules(info.point, info.volume_min, info.volume_step, info.volume_max,
                                 info.trade_stops_level, info.filling_mode)
        dist = max(dist, risk.min_stop_distance(rules, tick.ask - tick.bid))
        r = min(r, TOTAL_RISK_CAP - open_risk)
        if r <= 0.0005:
            log.info("%s : risque total ouvert au plafond (6 %%)", sym)
            return
        vpu = self.value_per_unit(sym, px)
        vol = risk.lot_size(acc.equity, r, dist, vpu, rules)
        lev = sum(abs(p.volume * p.price_open * self.value_per_unit(p.symbol, p.price_open)) for p in self.bot_positions())
        if vol <= 0 or lev + vol * px * vpu > MAX_LEV * acc.equity:
            log.info("%s : volume nul ou levier max atteint", sym)
            return
        sl = round(px - sig * dist, info.digits)
        tp = round(px + sig * tp_dist, info.digits)
        ticket = ex.open_position(api, sym, sig, vol, sl, tp, comment=f"bot d={dist:.6g}")
        if not ticket:
            self.tg.send(f"⚠️ <b>{esc(sym)}</b> : ordre refusé par le courtier")
            return
        p = next(p for p in self.bot_positions() if p.ticket == ticket)
        self.known[ticket] = {"sym": sym, "d": sig, "entry": p.price_open, "risk_money": self.risk_money(p), "sl": sl, "tp": tp}
        sens = "achat" if sig == 1 else "vente"
        log.info("entrée %s %s %s prix=%s sl=%s tp=%s", now_str(), sym, sens, p.price_open, sl, tp)
        self.tg.send(f"{'🟢' if sig == 1 else '🔴'} <b>{esc(sym)}</b> : {sens} à {price(p.price_open)}\n"
                     f"SL {price(sl)} · TP {price(tp)} · {vol} lot(s)\n"
                     f"Risque {r:.2%} du capital · conviction {force:.0%} (score {D.score:+.2f})\n"
                     f"Horizon max {max(g['tstop'], 24) // 24} j")

    # ------------------------------------------------------------------ statut et journal
    def status(self, state: str) -> None:
        pos = []
        acc = self.api.account_info()
        for p in self.bot_positions():
            rm = self.risk_money(p)
            pos.append({"actif": p.symbol, "sens": "achat" if p.type == self.api.POSITION_TYPE_BUY else "vente",
                        "entree": p.price_open, "sl": p.sl, "tp": p.tp, "r_latent": round(p.profit / rm, 2) if rm else 0.0,
                        "risque_pct": round(rm / acc.equity * 100, 2) if acc.equity else 0.0})
        se = getattr(self, "start_eq", 0.0)
        st = {"etat": state, "maj": now_str(), "positions": pos, "trades_du_jour": self.closed[-30:],
              "actifs": sorted(self.params), "perf_jour_pct": round((acc.equity - se) / se * 100, 3) if se else 0.0}
        with open("status_site/status.json", "w", encoding="utf-8") as fh:
            json.dump(st, fh, ensure_ascii=False, indent=1)
        if self.publish:
            subprocess.run("git add -A && git commit -q --amend -m statut --allow-empty && git push -q -f origin HEAD:status",
                           shell=True, cwd="status_site", check=False, capture_output=True)

    def save_journal(self, path: str = "journal/trades.csv") -> None:
        if not self.closed:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["heure", "actif", "sens", "entree", "sortie", "r", "raison"])
            if new:
                w.writeheader()
            w.writerows(self.closed)

    def save_equity(self, start_eq: float, path: str = "journal/equity.csv") -> None:
        """Courbe de performance en INDICE (1 = premier jour), sans aucun montant : le dépôt est public."""
        acc = self.api.account_info()
        day = (acc.equity - start_eq) / start_eq if start_eq else 0.0
        prev = 1.0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                prev = float(rows[-1]["indice"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["date", "indice", "jour_pct", "positions"])
            w.writerow([datetime.now(session.PARIS).strftime("%Y-%m-%d"), round(prev * (1 + day), 6),
                        round(day * 100, 3), len(self.bot_positions())])

    def summary(self, start_eq: float) -> None:
        acc = self.api.account_info()
        day = (acc.equity - start_eq) / start_eq if start_eq else 0.0
        lines = [f"📊 <b>Fin de session</b> · résultat du jour <b>{day:+.2%}</b> ({acc.equity - start_eq:+,.2f} {self.ccy})",
                 f"Trades fermés aujourd'hui : {len(self.closed)}"]
        for p in self.bot_positions():
            lines.append(f"• {esc(p.symbol)} {'achat' if p.type == self.api.POSITION_TYPE_BUY else 'vente'} "
                         f"{price(p.price_open)} → latent {p.profit:+,.2f} {self.ccy}")
        self.tg.send("\n".join(lines).replace(",", " "))

    # ------------------------------------------------------------------ boucle
    def run(self) -> None:
        job_start = datetime.now(UTC)
        start_eq = self.start_eq = self.start_equity()
        for sym in self.params:
            self.api.symbol_select(sym, True)
            ex.check_all_positions(self.api, sym)
        self.track_existing()
        done_hours: set[int] = set()
        while True:
            now = datetime.now(UTC)
            p = now.astimezone(session.PARIS)
            if p.time() >= session.FLAT or now - job_start >= session.JOB_STOP:
                self.detect_closures()
                self.status("arrêté (fin de session, positions gardées avec SL/TP)")
                self.summary(start_eq)
                self.save_equity(start_eq)
                return
            self.detect_closures()
            if 12 <= p.hour <= 17 and p.hour not in done_hours and p.minute < 30:
                done_hours.add(p.hour)
                for sym, prm in self.params.items():
                    try:
                        self.decide_symbol(sym, prm, start_eq)
                    except Exception as e:  # noqa: BLE001 - un actif en erreur ne bloque pas les autres
                        log.warning("%s : erreur %s", sym, type(e).__name__)
                        self.tg.send(f"⚠️ Erreur sur <b>{esc(sym)}</b> : {esc(type(e).__name__)}: {esc(str(e)[:300])}")
                self.detect_closures()
            self.status("en marche")
            time.sleep(max(5.0, 300 - time.time() % 300 + 20))  # toutes les 5 min, 20 s après la clôture


def init_distance(p) -> float:
    """Distance du stop à l'ouverture (notée dans le commentaire de l'ordre) : c'est la référence de 1 R."""
    try:
        return float(str(p.comment).split("d=")[1])
    except (IndexError, ValueError):
        return abs(p.price_open - p.sl)


def now_str() -> str:
    return datetime.now(session.PARIS).strftime("%Y-%m-%d %H:%M")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M")
    tg = Telegram()
    if not session.is_trading_day(session.now_paris()):
        return
    try:
        with open("results/live/bot_params.json", encoding="utf-8") as fh:
            params = json.load(fh)
        with open("results/live/backtest.json", encoding="utf-8") as fh:
            validated = bool(json.load(fh).get("validated"))
    except FileNotFoundError:
        tg.send("⚠️ Bot : aucun résultat de tournoi trouvé, pas de session aujourd'hui.")
        return
    from bot.mt5_setup import connect

    api = connect()
    demo = api.account_info().trade_mode == api.ACCOUNT_TRADE_MODE_DEMO
    if not validated and not demo:
        tg.send("⛔ Bot non validé par le backtest : il ne trade jamais un compte réel.")
        api.shutdown()
        return
    live = Live(api, params, tg, a.publish)
    wait = session.seconds_until_open(datetime.now(UTC))
    tg.send(f"🤖 <b>Session du {datetime.now(session.PARIS):%d/%m}</b> · {'démo' if demo else 'RÉEL'}"
            f"{'' if validated else ' (bot non validé : démo uniquement)'}\n"
            f"Spécialistes : {', '.join(sorted(params))}\nDécisions à chaque heure de 12h à 17h (Paris).")
    if wait > 0:
        live.status(f"en attente de l'ouverture (12h00), dans {int(wait // 60)} min")
        time.sleep(wait)
    try:
        live.run()
    except Exception as e:
        tg.send(f"🛑 Le bot s'est arrêté sur une erreur : {esc(type(e).__name__)}: {esc(str(e)[:500])}")
        log.error("arrêt sur erreur : %s", traceback.format_exc().splitlines()[-1])
        raise
    finally:
        live.save_journal()
        api.shutdown()


if __name__ == "__main__":
    main()
