"""Filtre des annonces économiques : pas de nouveau trade 15 min avant/après une annonce USD à fort impact.

Source : calendrier ForexFactory de la semaine (JSON public). Si le calendrier est indisponible,
on bloque par prudence les créneaux habituels des statistiques US : 14h30 et 16h00 (Paris) ± 15 min,
les mêmes que le backtest.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timedelta

from bot.session import PARIS

URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MARGIN = timedelta(minutes=15)
log = logging.getLogger("bot")


def load_events() -> list[datetime] | None:
    try:
        with urllib.request.urlopen(URL, timeout=20) as r:
            data = json.load(r)
    except Exception as e:  # noqa: BLE001 - on retombe sur les créneaux par défaut
        log.info("calendrier indisponible (%s) : créneaux US par défaut", type(e).__name__)
        return None
    return [datetime.fromisoformat(e["date"]) for e in data
            if e.get("country") == "USD" and e.get("impact") == "High" and "T" in e.get("date", "")]


def blocked(t: datetime, events: list[datetime] | None) -> bool:
    if events is None:
        p = t.astimezone(PARIS)
        m = p.hour * 60 + p.minute
        return 14 * 60 + 15 <= m < 14 * 60 + 45 or 15 * 60 + 45 <= m < 16 * 60 + 15
    return any(abs(t - e) <= MARGIN for e in events)
