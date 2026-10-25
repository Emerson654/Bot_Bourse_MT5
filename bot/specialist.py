"""Calcul live du score d'un spécialiste (tournois 6-7), identique au backtest.

Le score n'est calculé qu'aux « bougies de décision » (clôtures H1 dont la bougie suivante ouvre entre
12h00 et 17h00, Paris, du lundi au vendredi), lissé sur ces seules bougies, comme dans research/prep3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bot import indicators as ind
from bot.voters import Bars
from research.macro import macro_votes
from research.mtf import lower_tf_votes_from, mtf_votes

FIRST_OPEN, LAST_OPEN = 12 * 60, 17 * 60
MONTH_ATR = np.sqrt(22.0 * 24.0)


@dataclass
class Decision:
    score: float  # score du bot à la clôture qui vient de se terminer
    prev_signal: int  # signal (1, -1, 0) à la décision précédente : une entrée exige un signal « frais »
    atr: float  # ATR(14) H1
    genome: dict  # règles du gagnant (seuils, stop, TP, horizon, risque, positions max)


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    out = np.empty_like(x)
    a, e = 2.0 / (n + 1), x[0]
    for i, v in enumerate(x):
        e += a * (v - e)
        out[i] = e
    return out


def _score(V: np.ndarray, g: dict) -> np.ndarray:
    w = np.asarray(g["w"], dtype=float)
    return _ema(V[:, list(g["idx"])].astype(float) @ w / max(w.sum(), 1e-9), g["smooth"])


def decision_rows(b: Bars, next_minute: int, next_weekday: int, next_day: int) -> np.ndarray:
    """Indices des bougies de décision ; la dernière bougie terminée l'est si la bougie en cours ouvre dans la session."""
    m, d, wd = b.minute, b.day, b.weekday
    dec = list(np.flatnonzero((m[1:] >= FIRST_OPEN) & (m[1:] <= LAST_OPEN) & (d[1:] == d[:-1]) & (wd[1:] < 5)))
    if FIRST_OPEN <= next_minute <= LAST_OPEN and next_weekday < 5 and next_day == d[-1]:
        dec.append(len(b.c) - 1)
    return np.array(dec, dtype=np.int64)


def signal_of(s: float, g: dict) -> int:
    return 1 if s > g["entry"] else (-1 if (g["short"] and s < -g["entry"]) else 0)


def decide(params: dict, b: Bars, ts: np.ndarray, b5: Bars, ts5: np.ndarray, series: dict, dec: np.ndarray) -> Decision:
    """`params` = rapport du pli de production du spécialiste (results/bot_params.json[actif])."""
    votes = mtf_votes(b, ts)
    votes.update(lower_tf_votes_from(b5, ts5, ts))
    if series:
        votes.update(macro_votes(b.day, b.c, series))
    names = params["voters"]
    V = np.column_stack([votes[k][1][dec] if k in votes else np.zeros(len(dec)) for k in names])
    cols = []
    for lg in params["champions"]:
        c = params["leagues"][lg]
        cols.append(_score(V, c["genome"]) if c["type"] == "seul" else np.mean([_score(V, m) for m in c["members"]], axis=0))
    g = params["winner"]["genome"]
    s = _score(np.column_stack(cols), g)
    a = ind.atr(b.h, b.lo, b.c, 14)
    return Decision(float(s[-1]), signal_of(float(s[-2]), g) if len(s) > 1 else 0, float(a[-1]), g)


def order_plan(dec: Decision, d: int, price: float, spread: float) -> tuple[float, float, float, float]:
    """(distance du stop, objectif, risque en fraction du capital, force du signal), règles du simulateur sim4."""
    g = dec.genome
    dist = max(g["stop_k"] * dec.atr * g["atr_scale"], 3.0 * spread)
    tp_dist = min(g["tp_r"] * dist, MONTH_ATR * dec.atr)
    force = min(1.0, max(0.0, (abs(dec.score) - g["entry"]) / max(1e-9, 1.0 - g["entry"])))
    risk = g["risk_lo"] + (g["risk_hi"] - g["risk_lo"]) * force ** g["risk_pow"]
    return dist, tp_dist, min(risk, 0.02), force
