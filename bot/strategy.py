"""Stratégie live : recalcule exactement le score du bot gagnant du tournoi (results/bot_params.json).

score_champion = EMA( Σ w_i · vote_i / Σ w_i ) ; un champion « fusion » = moyenne de ses membres ;
score_final = EMA( Σ W_j · score_champion_j / Σ W_j ) avec les pondérations de la grande finale.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from bot import indicators as ind
from bot.voters import Bars, compute_votes


@dataclass
class Rules:
    entry: float
    exit: float
    stop_k: float
    tp_r: float
    tstop: int
    cooldown: int
    short: int
    news: int


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    out = np.empty_like(x)
    a, e = 2.0 / (n + 1), x[0]
    for i, v in enumerate(x):
        e += a * (v - e)
        out[i] = e
    return out


def _genome_score(V: np.ndarray, g: dict) -> np.ndarray:
    w = np.asarray(g["w"], dtype=float)
    return _ema(V[:, list(g["idx"])].astype(float) @ w / max(w.sum(), 1e-9), g["smooth"])


def load_params(path: str = "results/bot_params.json") -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def final_score(b: Bars, params: dict) -> tuple[np.ndarray, Rules]:
    votes = compute_votes(b)
    names = list(votes)
    if names != params["voters"]:
        raise RuntimeError("Les votants du code ne correspondent plus au tournoi : relancer le tournoi")
    V = np.column_stack([votes[k][1] for k in names])
    cols = []
    for lg in params["champions"]:
        c = params["leagues"][lg]
        if c["type"] == "seul":
            cols.append(_genome_score(V, c["genome"]))
        else:
            cols.append(np.mean([_genome_score(V, m) for m in c["members"]], axis=0))
    g = params["winner"]["genome"]
    s = _genome_score(np.column_stack(cols), g)
    rules = Rules(**{k: g[k] for k in Rules.__dataclass_fields__})
    return s, rules


def last_atr(b: Bars) -> float:
    return float(ind.atr(b.h, b.lo, b.c, 14)[-1])
