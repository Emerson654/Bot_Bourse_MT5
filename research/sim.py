"""Simulateur intraday XAUUSD fidèle aux règles du bot live (compilé avec numba).

Règles reproduites :
- décision à la clôture de la bougie t, exécution à l'ouverture de t+1 ;
- prix des bougies = bid ; achat au ask (bid + spread), vente au bid ; stop et TP chez le courtier,
  testés sur le plus haut / plus bas de chaque bougie, le stop d'abord si les deux sont touchés ;
- 1 position max, risque 1 % du capital par trade (taille = risque / distance du stop),
  levier plafonné ; glissement ajouté à chaque exécution ;
- entrées de 12h00 à 17h30 (Paris), fermeture forcée à 17h50, perte journalière max 3 % ;
- fenêtres d'annonces US (14h30 et 16h00 ±15 min) optionnellement bloquées pour les entrées.
"""

from __future__ import annotations

import numpy as np
from numba import njit

ENTRY_START, ENTRY_END, FLAT_AT = 12 * 60, 17 * 60 + 30, 17 * 60 + 50
NEWS = ((14 * 60 + 15, 14 * 60 + 45), (15 * 60 + 45, 16 * 60 + 15))
RISK, DAILY_STOP, MAX_LEV = 0.01, 0.03, 20.0


@njit(cache=True)
def _in_news(m: int) -> bool:
    return (14 * 60 + 15 <= m < 14 * 60 + 45) or (15 * 60 + 45 <= m < 16 * 60 + 15)


@njit(cache=True)
def simulate(o, h, lo, c, spread, atr, minute, day, score, entry, exit_, stop_k, tp_r, tstop, cooldown,
             allow_short, news_block, slip):
    """Retourne (equity par bougie, trades : tableau (n, 6) = [i_entrée, i_sortie, sens, R, rendement, raison]).

    Raisons de sortie : 0 stop, 1 TP, 2 signal inverse, 3 stop temporel, 4 fin de session.
    """
    n = len(c)
    eq = np.ones(n)
    trades = np.zeros((n // 2 + 1, 6))
    nt = 0
    equity = 1.0
    pos = 0  # +1 achat, −1 vente
    ep = sl = tp = size = 0.0  # size = fraction du capital par unité de prix
    i_in = 0
    last_exit = -10_000
    cur_day = -1
    day_start = 1.0
    halted = False
    for i in range(n - 1):
        if day[i] != cur_day:
            cur_day = day[i]
            day_start = equity
            halted = False
        # 1) Gestion de la position ouverte sur la bougie i (stops chez le courtier)
        if pos != 0 and i > i_in:
            px = -1.0
            why = -1
            if pos == 1:
                if lo[i] <= sl:
                    px, why = min(sl, o[i]) - slip, 0
                elif h[i] >= tp:
                    px, why = max(tp, o[i]), 1
            else:
                if h[i] + spread[i] >= sl:
                    px, why = max(sl, o[i] + spread[i]) + slip, 0
                elif lo[i] + spread[i] <= tp:
                    px, why = min(tp, o[i] + spread[i]), 1
            if why < 0 and (minute[i] >= FLAT_AT or day[i] != day[i_in]):
                px = (o[i] if pos == 1 else o[i] + spread[i]) - pos * slip
                why = 4
            if why >= 0:
                ret = pos * (px - ep) * size
                equity *= 1.0 + ret
                trades[nt, 0], trades[nt, 1], trades[nt, 2] = i_in, i, pos
                trades[nt, 3] = pos * (px - ep) / (stop_k * atr[i_in - 1])
                trades[nt, 4], trades[nt, 5] = ret, why
                nt += 1
                pos = 0
                last_exit = i
                if equity <= day_start * (1.0 - DAILY_STOP):
                    halted = True
        eq[i] = equity
        # 2) Décision à la clôture de i, exécutée à l'ouverture de i+1
        m_next = minute[i + 1]
        if pos != 0:
            s = score[i]
            reverse = (pos == 1 and s < -exit_) or (pos == -1 and s > exit_)
            timeout = tstop > 0 and (i + 1 - i_in) >= tstop
            last_bar = day[i + 1] != day[i] or m_next >= FLAT_AT
            if reverse or timeout or last_bar:
                if last_bar and day[i + 1] != day[i]:
                    px = (c[i] if pos == 1 else c[i] + spread[i]) - pos * slip
                else:
                    px = (o[i + 1] if pos == 1 else o[i + 1] + spread[i + 1]) - pos * slip
                ret = pos * (px - ep) * size
                equity *= 1.0 + ret
                trades[nt, 0], trades[nt, 1], trades[nt, 2] = i_in, i + 1, pos
                trades[nt, 3] = pos * (px - ep) / (stop_k * atr[i_in - 1])
                trades[nt, 4] = ret
                trades[nt, 5] = 2 if reverse else (3 if timeout else 4)
                nt += 1
                pos = 0
                last_exit = i + 1
                if equity <= day_start * (1.0 - DAILY_STOP):
                    halted = True
        elif (not halted and day[i + 1] == day[i] and ENTRY_START <= m_next < ENTRY_END
              and i + 1 - last_exit >= cooldown and not (news_block and _in_news(m_next))):
            s = score[i]
            d = 1 if s > entry else (-1 if (allow_short and s < -entry) else 0)
            if d != 0 and atr[i] > 0:
                dist = stop_k * atr[i]
                ep = (o[i + 1] + spread[i + 1] if d == 1 else o[i + 1]) + d * slip
                sl = ep - d * dist
                tp = ep + d * tp_r * dist
                # taille : risque 1 % / distance, plafonnée par le levier max
                size = min(RISK / dist, MAX_LEV / ep)
                pos = d
                i_in = i + 1
    eq[n - 1] = equity
    return eq, trades[:nt]


def metrics(eq: np.ndarray, trades: np.ndarray, day: np.ndarray) -> dict:
    """Mesures de performance : profit factor, drawdown, Sharpe journalier, trades/jour, espérance en R."""
    if len(trades) == 0:
        return {"trades": 0, "pf": 0.0, "max_dd": 0.0, "sharpe": 0.0, "tpd": 0.0, "exp_r": 0.0, "ret": 0.0, "win": 0.0}
    r = trades[:, 4]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    peak = np.maximum.accumulate(eq)
    dd = float((eq / peak - 1).min())
    # Rendement journalier : équité à la dernière bougie de chaque jour
    last = np.flatnonzero(np.diff(day, append=day[-1] + 1))
    de = eq[last]
    dr = np.diff(de) / de[:-1]
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    n_days = len(np.unique(day))
    return {
        "trades": len(trades),
        "pf": float(gains / losses) if losses > 0 else 99.0,
        "max_dd": dd,
        "sharpe": sharpe,
        "tpd": len(trades) / max(n_days, 1),
        "exp_r": float(trades[:, 3].mean()),
        "ret": float(eq[-1] - 1),
        "win": float((r > 0).mean()),
        "avg_bars": float((trades[:, 1] - trades[:, 0]).mean()),
    }
