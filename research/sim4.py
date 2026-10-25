"""Simulateur v6 : paris de quelques heures à 1 mois, plusieurs positions simultanées par actif.

- décisions (entrées et sorties sur signal) aux clôtures H1 de la session 12h-17h, du lundi au vendredi ;
- jusqu'à `max_pos` positions ouvertes en même temps sur l'actif, chacune avec SL/TP chez le courtier et un
  horizon maximum (`tstop` bougies H1, ≤ 1 mois) ; une nouvelle entrée exige un nouveau signal ;
- stop = stop_k × ATR(H1) × atr_scale (1 ≈ horaire, 2 ≈ 4 h, 5 ≈ journalier) ;
  objectif = tp_r × stop, plafonné au mouvement mensuel typique (ATR H1 × √(22 × 24)) ;
- positions gardées la nuit et le week-end : swap facturé, et un gap d'ouverture au-delà du stop est
  exécuté au prix d'ouverture (perte supérieure au risque prévu, comme en réel) ;
- risque du pari selon la conviction (voir sim3), levier total plafonné.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MAX_LEV = 20.0
RISK_CAP = 0.02
FIRST_OPEN, LAST_OPEN = 12 * 60, 17 * 60
MONTH_ATR = np.sqrt(22.0 * 24.0)  # mouvement mensuel typique ≈ ATR horaire × √(nombre d'heures)
MAXP = 3


@njit(cache=True)
def simulate(o, h, lo, c, spread, atr, minute, day, weekday, score, entry, exit_, stop_k, tp_r, tstop,
             allow_short, slip, swap_long, swap_short, triple_wd, risk_lo, risk_hi, risk_pow, atr_scale, max_pos):
    """Retourne (equity par bougie, trades (n, 6) = [i_entrée, i_sortie, sens, R, rendement, raison]).

    Raisons : 0 stop, 1 TP, 2 signal inverse, 3 horizon atteint.
    """
    n = len(c)
    eq = np.ones(n)
    trades = np.zeros((MAXP * n + 1, 6))
    nt = 0
    equity = 1.0
    pos = np.zeros(MAXP, np.int64)
    ep = np.zeros(MAXP)
    sl = np.zeros(MAXP)
    tp = np.zeros(MAXP)
    size = np.zeros(MAXP)
    dist = np.zeros(MAXP)
    i_in = np.zeros(MAXP, np.int64)
    last_sig = 0
    for i in range(n - 1):
        for k in range(MAXP):
            if pos[k] == 0 or i <= i_in[k]:
                continue
            p = pos[k]
            if day[i] != day[i - 1]:
                mult = 3.0 if weekday[i - 1] == triple_wd else 1.0
                equity *= 1.0 + (swap_long if p == 1 else swap_short) * size[k] * mult
            px = -1.0
            why = -1
            if p == 1:
                if lo[i] <= sl[k]:
                    px, why = min(sl[k], o[i]) - slip, 0
                elif h[i] >= tp[k]:
                    px, why = max(tp[k], o[i]), 1
            else:
                if h[i] + spread[i] >= sl[k]:
                    px, why = max(sl[k], o[i] + spread[i]) + slip, 0
                elif lo[i] + spread[i] <= tp[k]:
                    px, why = min(tp[k], o[i] + spread[i]), 1
            if why >= 0:
                ret = p * (px - ep[k]) * size[k]
                equity *= 1.0 + ret
                trades[nt, 0], trades[nt, 1], trades[nt, 2] = i_in[k], i, p
                trades[nt, 3], trades[nt, 4], trades[nt, 5] = p * (px - ep[k]) / dist[k], ret, why
                nt += 1
                pos[k] = 0
        eq[i] = equity
        m_next = minute[i + 1]
        if not (FIRST_OPEN <= m_next <= LAST_OPEN and day[i + 1] == day[i] and weekday[i + 1] < 5):
            continue
        s = score[i]
        # Sorties sur signal inverse ou horizon atteint
        for k in range(MAXP):
            if pos[k] == 0:
                continue
            p = pos[k]
            reverse = (p == 1 and s < -exit_) or (p == -1 and s > exit_)
            timeout = (i + 1 - i_in[k]) >= tstop
            if reverse or timeout:
                px = (o[i + 1] if p == 1 else o[i + 1] + spread[i + 1]) - p * slip
                ret = p * (px - ep[k]) * size[k]
                equity *= 1.0 + ret
                trades[nt, 0], trades[nt, 1], trades[nt, 2] = i_in[k], i + 1, p
                trades[nt, 3], trades[nt, 4] = p * (px - ep[k]) / dist[k], ret
                trades[nt, 5] = 2 if reverse else 3
                nt += 1
                pos[k] = 0
        # Nouvelle entrée : signal qui (re)franchit le seuil, place libre, même sens que les positions ouvertes
        d = 1 if s > entry else (-1 if (allow_short and s < -entry) else 0)
        fresh = d != 0 and d != last_sig
        last_sig = d
        if not fresh or atr[i] <= 0:
            continue
        open_n = 0
        same = True
        free = -1
        lev = 0.0
        for k in range(MAXP):
            if pos[k] != 0:
                open_n += 1
                lev += size[k] * ep[k]
                if pos[k] != d:
                    same = False
            elif free < 0:
                free = k
        if open_n >= max_pos or not same or free < 0:
            continue
        dk = max(stop_k * atr[i] * atr_scale, 3.0 * spread[i + 1])
        e = (o[i + 1] + spread[i + 1] if d == 1 else o[i + 1]) + d * slip
        force = min(1.0, max(0.0, (abs(s) - entry) / max(1e-9, 1.0 - entry)))
        risk = min(RISK_CAP, risk_lo + (risk_hi - risk_lo) * force ** risk_pow)
        sz = min(risk / dk, max(0.0, MAX_LEV - lev) / e)
        if sz <= 0:
            continue
        pos[free], ep[free], dist[free], size[free], i_in[free] = d, e, dk, sz, i + 1
        sl[free] = e - d * dk
        tp[free] = e + d * min(tp_r * dk, MONTH_ATR * atr[i])
    eq[n - 1] = equity
    return eq, trades[:nt]
