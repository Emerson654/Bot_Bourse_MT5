"""Étape 0 : vérifie que MT5 tourne sur le runner, lit les règles XAUUSD et télécharge l'historique.

Aucune donnée sensible n'est affichée : ni numéro de compte, ni solde, ni identifiants.
Usage (runner Windows) : python scripts/probe.py
"""

from __future__ import annotations

import csv
import gzip
import os
import sys
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

SYMBOL_CANDIDATES = ("XAUUSD", "GOLD", "XAUUSD.", "XAUUSDm")
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1,
}
START = datetime(2008, 1, 1, tzinfo=timezone.utc)


def connect() -> None:
    ok = mt5.initialize(
        path=os.environ.get("MT5_PATH"),
        login=int(os.environ["MT5_LOGIN"]),
        password=os.environ["MT5_PASSWORD"],
        server=os.environ["MT5_SERVER"],
        timeout=120_000,
    )
    if not ok:
        sys.exit(f"Connexion MT5 impossible : {mt5.last_error()}")
    acc = mt5.account_info()
    # Seulement des infos non sensibles
    print(f"Connecté. Devise du compte : {acc.currency}, levier 1:{acc.leverage}, "
          f"mode {'hedge' if acc.margin_mode == mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING else 'netting'}, "
          f"trading autorisé : {acc.trade_allowed}, algo autorisé : {mt5.terminal_info().trade_allowed}")


def find_symbol() -> str:
    for s in SYMBOL_CANDIDATES:
        if mt5.symbol_select(s, True):
            return s
    golds = [s.name for s in (mt5.symbols_get("*XAU*") or [])]
    sys.exit(f"Aucun symbole or trouvé parmi {SYMBOL_CANDIDATES} ; disponibles : {golds}")


def print_specs(sym: str) -> None:
    i = mt5.symbol_info(sym)
    fill = {1: "FOK", 2: "IOC", 4: "RETURN"}
    print(f"Symbole {sym} : digits={i.digits} point={i.point} contrat={i.trade_contract_size} "
          f"vol_min={i.volume_min} vol_step={i.volume_step} vol_max={i.volume_max} "
          f"stops_level={i.trade_stops_level} freeze_level={i.trade_freeze_level} "
          f"spread={i.spread} spread_flottant={i.spread_float} "
          f"filling={[n for b, n in fill.items() if i.filling_mode & b]} "
          f"devise_profit={i.currency_profit} mode_exec={i.trade_exemode} swap_long={i.swap_long} swap_short={i.swap_short}")
    # Valeur d'un mouvement de 1 $ sur 1 lot, en devise du compte
    tick = mt5.symbol_info_tick(sym)
    if tick:
        p = mt5.order_calc_profit(mt5.ORDER_TYPE_BUY, sym, 1.0, tick.ask, tick.ask + 1.0)
        print(f"1 lot, +1.00 de prix = {p} (devise du compte) ; heure serveur du dernier tick : "
              f"{datetime.fromtimestamp(tick.time, timezone.utc):%Y-%m-%d %H:%M}")


def download(sym: str, name: str, tf: int, out_dir: str) -> None:
    """Télécharge par tranches de 30 jours (contourne la limite de bougies par requête)."""
    now = datetime.now(timezone.utc)
    rows: dict[int, tuple] = {}
    t = START
    while t < now:
        t2 = min(t + timedelta(days=30), now)
        r = mt5.copy_rates_range(sym, tf, t, t2)
        if r is not None:
            for x in r:
                rows[int(x["time"])] = (int(x["time"]), x["open"], x["high"], x["low"], x["close"],
                                        int(x["tick_volume"]), int(x["spread"]))
        t = t2
    if not rows:
        print(f"{name} : aucune bougie ({mt5.last_error()})")
        return
    keys = sorted(rows)
    path = os.path.join(out_dir, f"XAUUSD_{name}.csv.gz")
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "tick_volume", "spread"])
        w.writerows(rows[k] for k in keys)
    first = datetime.fromtimestamp(keys[0], timezone.utc)
    last = datetime.fromtimestamp(keys[-1], timezone.utc)
    print(f"{name} : {len(keys)} bougies du {first:%Y-%m-%d} au {last:%Y-%m-%d %H:%M}")


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    os.makedirs(out_dir, exist_ok=True)
    connect()
    sym = find_symbol()
    print_specs(sym)
    for name, tf in TIMEFRAMES.items():
        download(sym, name, tf, out_dir)
    mt5.shutdown()


if __name__ == "__main__":
    main()
