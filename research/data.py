"""Chargement de l'historique XAUUSD M5 et conversion de l'heure serveur MT5 en heure de Paris.

L'heure serveur de MetaQuotes-Demo suit la convention « clôture de New York » : 00:00 serveur =
17:00 à New York, été comme hiver, donc heure serveur = heure de New York + 7 h.
`check_server_offset` le vérifie sur les données (la pause quotidienne de l'or tombe à 00:00 serveur).
"""

from __future__ import annotations

import csv
import gzip
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from bot.voters import Bars

NY = ZoneInfo("America/New_York")
PARIS = ZoneInfo("Europe/Paris")


def server_to_utc(ts: int) -> datetime:
    """Horodatage serveur (secondes, faussement « UTC ») → datetime UTC réel."""
    naive_ny = datetime.fromtimestamp(ts, UTC).replace(tzinfo=None) - timedelta(hours=7)
    return naive_ny.replace(tzinfo=NY).astimezone(UTC)


def load_m5(path: str) -> tuple[Bars, np.ndarray]:
    """Retourne les bougies et les horodatages UTC (datetime64[s])."""
    with gzip.open(path, "rt") as fh:
        rows = list(csv.DictReader(fh))
    o, h, lo, c = (np.array([float(r[k]) for r in rows]) for k in ("open", "high", "low", "close"))
    if "time_utc" in rows[0]:  # Dukascopy : déjà en UTC, spread en prix
        utc = [datetime.fromtimestamp(int(r["time_utc"]), UTC) for r in rows]
        spread = np.array([float(r["spread"]) for r in rows])
    else:  # export MT5 : heure serveur, spread en points
        utc = [server_to_utc(int(r["time"])) for r in rows]
        spread = np.array([float(r["spread"]) for r in rows]) * (0.01 if np.median(c) > 100 else 0.0001)
    paris = [u.astimezone(PARIS) for u in utc]
    minute = np.array([p.hour * 60 + p.minute for p in paris])
    weekday = np.array([p.weekday() for p in paris])
    # Jour de session = date de Paris (la session 12h-18h est entièrement dans la même date)
    day = np.array([p.toordinal() for p in paris])
    bars = Bars(o, h, lo, c, spread, minute, day, weekday)
    return bars, np.array([u.replace(tzinfo=None) for u in utc], dtype="datetime64[s]")


def check_server_offset(path: str) -> dict[int, int]:
    """Histogramme des heures serveur où il manque des bougies M5 : doit pointer sur 23h/00h."""
    with gzip.open(path, "rt") as fh:
        ts = np.array([int(r["time"]) for r in csv.DictReader(fh)])
    gaps = ts[1:] - ts[:-1]
    # trous de 10 min à 3 h (hors week-end)
    idx = np.flatnonzero((gaps > 600) & (gaps < 3 * 3600))
    hours = (ts[idx + 1] // 3600) % 24
    vals, counts = np.unique(hours, return_counts=True)
    return dict(zip(vals.tolist(), counts.tolist()))
