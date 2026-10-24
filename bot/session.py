"""Horaires de session : 12h-18h heure de Paris, du lundi au vendredi (changement d'heure géré par zoneinfo).

Le cron GitHub est en UTC et peut démarrer en retard : les limites sont donc à la fois horaires
(Paris) et relatives au démarrage du job (tué par GitHub à 6 h) :
- plus d'ouverture après 17h30 OU après 5h30 de job ;
- fermeture de toutes les positions et arrêt propre à 17h50 OU après 5h50 de job.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

PARIS = ZoneInfo("Europe/Paris")
OPEN, LAST_ENTRY, FLAT = time(12, 0), time(17, 30), time(17, 50)
JOB_NO_ENTRY, JOB_STOP = timedelta(hours=5, minutes=30), timedelta(hours=5, minutes=50)


def now_paris() -> datetime:
    return datetime.now(PARIS)


def is_trading_day(t: datetime) -> bool:
    return t.astimezone(PARIS).weekday() < 5


def can_open(t: datetime, job_start: datetime) -> bool:
    p = t.astimezone(PARIS)
    return is_trading_day(p) and OPEN <= p.time() < LAST_ENTRY and t - job_start < JOB_NO_ENTRY


def must_stop(t: datetime, job_start: datetime) -> bool:
    """Vrai à partir de 17h50 (Paris) ou après 5h50 de job : tout fermer et s'arrêter."""
    p = t.astimezone(PARIS)
    return p.time() >= FLAT or t - job_start >= JOB_STOP or not is_trading_day(p)


def seconds_until_open(t: datetime) -> float:
    p = t.astimezone(PARIS)
    start = p.replace(hour=OPEN.hour, minute=0, second=0, microsecond=0)
    return max(0.0, (start - p).total_seconds())
