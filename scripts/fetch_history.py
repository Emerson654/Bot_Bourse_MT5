"""Liste les actifs disponibles sur le serveur et télécharge l'historique H1 de ceux demandés.

Usage (runner Windows) : python scripts/fetch_history.py data/mt5 XAUUSD EURUSD USDJPY ...
Pour chaque actif : règles du courtier (spread, swap, lots) dans data/mt5/specs.json et bougies H1.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from datetime import UTC, datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bot import mt5_setup  # noqa: E402

START = datetime(2008, 1, 1, tzinfo=UTC)


def list_groups(mt5) -> None:
    """Affiche les actifs hors Forex « exotique », regroupés par catégorie du serveur."""
    groups: dict[str, list[str]] = {}
    for s in mt5.symbols_get() or []:
        g = s.path.rsplit("\\", 1)[0]
        groups.setdefault(g, []).append(s.name)
    for g, names in sorted(groups.items()):
        print(f"[{g}] {len(names)} : {' '.join(sorted(names)[:40])}")


def fetch(mt5, sym: str, out_dir: str) -> dict | None:
    if not mt5.symbol_select(sym, True):
        print(f"{sym} : indisponible")
        return None
    i = mt5.symbol_info(sym)
    rows: dict[int, tuple] = {}
    t, now = START, datetime.now(UTC)
    while t < now:
        t2 = min(t + timedelta(days=180), now)
        r = mt5.copy_rates_range(sym, mt5.TIMEFRAME_H1, t, t2)
        if r is not None:
            for x in r:
                rows[int(x["time"])] = (int(x["time"]), x["open"], x["high"], x["low"], x["close"], int(x["spread"]))
        t = t2
    if not rows:
        print(f"{sym} : aucune bougie")
        return None
    keys = sorted(rows)
    with gzip.open(os.path.join(out_dir, f"{sym}_H1.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "open", "high", "low", "close", "spread"])
        w.writerows(rows[k] for k in keys)
    spreads = sorted(rows[k][5] for k in keys[-5000:])
    spec = {"point": i.point, "digits": i.digits, "contract": i.trade_contract_size, "vol_min": i.volume_min,
            "vol_step": i.volume_step, "stops_level": i.trade_stops_level, "swap_long": i.swap_long,
            "swap_short": i.swap_short, "swap_mode": i.swap_mode, "swap_3days": i.swap_rollover3days,
            "spread_median_pts": spreads[len(spreads) // 2], "currency_profit": i.currency_profit,
            "first": datetime.fromtimestamp(keys[0], UTC).strftime("%Y-%m-%d"), "bars": len(keys)}
    print(f"{sym} : {len(keys)} bougies H1 depuis {spec['first']}, spread médian {spec['spread_median_pts']} pts")
    return spec


def main() -> None:
    out_dir, symbols = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    mt5 = mt5_setup.connect()
    list_groups(mt5)
    specs = {s: sp for s in symbols if (sp := fetch(mt5, s, out_dir))}
    with open(os.path.join(out_dir, "specs.json"), "w") as fh:
        json.dump(specs, fh, indent=1)
    mt5.shutdown()


if __name__ == "__main__":
    main()
