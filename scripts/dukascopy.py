"""Télécharge l'historique XAUUSD minute de Dukascopy (gratuit, en UTC) et l'agrège en M5.

Format .bi5 : LZMA, enregistrements de 24 octets big-endian
(décalage en s depuis minuit UTC, open, close, low, high en entiers ×1000, volume float32).
Le spread de chaque bougie M5 = moyenne (ask − bid) des clôtures minute.
Usage : python scripts/dukascopy.py 2012 data/XAUUSD_M5_utc.csv.gz
"""

from __future__ import annotations

import csv
import gzip
import lzma
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta

URL = "https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m:02d}/{d:02d}/{side}_candles_min_1.bi5"
SCALE = 1000.0


def fetch_day(day: date, side: str) -> list[tuple[int, float, float, float, float]]:
    url = URL.format(y=day.year, m=day.month - 1, d=day.day, side=side)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            break
        except Exception:  # noqa: BLE001 - réessai réseau
            time.sleep(2 ** attempt)
    else:
        return []
    if not raw:
        return []
    data = lzma.decompress(raw)
    base = int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())
    out = []
    for off in range(0, len(data), 24):
        t, o, c, lo, h, v = struct.unpack(">5if", data[off:off + 24])
        if v > 0:
            out.append((base + t, o / SCALE, h / SCALE, lo / SCALE, c / SCALE))
    return out


def fetch_both(day: date) -> list[tuple]:
    bid = fetch_day(day, "BID")
    ask = {t: c for t, _, _, _, c in fetch_day(day, "ASK")}
    return [(t, o, h, lo, c, ask.get(t, c) - c) for t, o, h, lo, c in bid]


def to_m5(rows: list[tuple]) -> list[tuple]:
    out: dict[int, list] = {}
    for t, o, h, lo, c, sp in rows:
        k = t - t % 300
        b = out.get(k)
        if b is None:
            out[k] = [k, o, h, lo, c, sp, 1]
        else:
            b[2], b[3], b[4] = max(b[2], h), min(b[3], lo), c
            b[5] += sp
            b[6] += 1
    return [(k, b[1], b[2], b[3], b[4], round(b[5] / b[6], 4)) for k, b in sorted(out.items())]


def main() -> None:
    start_year, path = int(sys.argv[1]), sys.argv[2]
    d0, d1 = date(start_year, 1, 1), datetime.now(UTC).date() - timedelta(days=1)
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1) if (d0 + timedelta(days=i)).weekday() != 5]
    rows: list[tuple] = []
    with ThreadPoolExecutor(16) as ex:
        for i, r in enumerate(ex.map(fetch_both, days)):
            rows.extend(r)
            if i % 250 == 0:
                print(f"{days[i]} : {len(rows)} minutes", flush=True)
    m5 = to_m5(rows)
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_utc", "open", "high", "low", "close", "spread"])
        w.writerows(m5)
    print(f"{len(m5)} bougies M5 du {datetime.fromtimestamp(m5[0][0], UTC):%Y-%m-%d} "
          f"au {datetime.fromtimestamp(m5[-1][0], UTC):%Y-%m-%d}")


if __name__ == "__main__":
    main()
