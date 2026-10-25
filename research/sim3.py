"""Simulateur v3 (bougies H1, positions de quelques heures à quelques jours), fidèle au bot live :

- le bot ne tourne que pendant la session GitHub : décisions (entrées ET sorties sur signal) seulement
  aux clôtures H1 dont la bougie suivante ouvre entre 12h00 et 17h00 (Paris) ;
- SL et TP sont chez le courtier : testés sur CHAQUE bougie, nuit comprise (stop d'abord si les deux) ;
- achat au ask (bid + spread), vente au bid, glissement à chaque exécution ;
- swap facturé à chaque nuit passée (triple la nuit indiquée par le courtier) ;
- vendredi : tout est fermé à la dernière décision de la session (pas de position le week-end) ;
- risque variable selon la conviction du signal (taille = risque / distance du stop), levier plafonné.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MAX_LEV = 20.0
RISK_CAP = 0.025  # garde-fou : jamais plus de 2,5 % du capital risqué sur un trade
FIRST_OPEN, LAST_OPEN = 12 * 60, 17 * 60  # ouverture de la bougie suivante, en minutes Paris


@njit(cache=True)
def simulate(o, h, lo, c, spread, atr, minute, day, weekday, score, entry, exit_, stop_k, tp_r, tstop,
             allow_short, slip, swap_long, swap_short, triple_wd, risk_lo, risk_hi, risk_pow):
    """Retourne (equity par bougie, trades (n, 6) = [i_entrée, i_sortie, sens, R, rendement, raison]).

    Raisons : 0 stop, 1 TP, 2 signal inverse, 3 stop temporel, 4 vendredi soir.
    Risque du trade = risk_lo + (risk_hi − risk_lo) × force^risk_pow, force = (|score| − entry)/(1 − entry) ∈ [0, 1] :
    plus les votants sont d'accord, plus la mise est forte (plafonnée à RISK_CAP).
    swap_* = coût (ou gain) en prix par unité et par nuit ; triple_wd = jour (0 = lundi) du swap triple.
    """
    n = len(c)
    eq = np.ones(n)
    trades = np.zeros((n // 2 + 1, 6))
    nt = 0
    equity = 1.0
    pos = 0
    ep = sl = tp = size = dist = 0.0
    i_in = 0
    for i in range(n - 1):
        # 1) Stops chez le courtier sur la bougie i, et swap si on change de jour
        if pos != 0 and i > i_in:
            if day[i] != day[i - 1]:
                mult = 3.0 if weekday[i - 1] == triple_wd else 1.0
                equity *= 1.0 + (swap_long if pos == 1 else swap_short) * size * mult
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
            if why >= 0:
                ret = pos * (px - ep) * size
                equity *= 1.0 + ret
                trades[nt, 0], trades[nt, 1], trades[nt, 2] = i_in, i, pos
                trades[nt, 3], trades[nt, 4], trades[nt, 5] = pos * (px - ep) / dist, ret, why
                nt += 1
                pos = 0
        eq[i] = equity
        # 2) Décision à la clôture de i (seulement pendant la session du bot)
        m_next = minute[i + 1]
        if not (FIRST_OPEN <= m_next <= LAST_OPEN and day[i + 1] == day[i] and weekday[i + 1] < 5):
            continue
        friday_end = weekday[i + 1] == 4 and m_next >= LAST_OPEN
        s = score[i]
        if pos != 0:
            reverse = (pos == 1 and s < -exit_) or (pos == -1 and s > exit_)
            timeout = tstop > 0 and (i + 1 - i_in) >= tstop
            if reverse or timeout or friday_end:
                px = (o[i + 1] if pos == 1 else o[i + 1] + spread[i + 1]) - pos * slip
                ret = pos * (px - ep) * size
                equity *= 1.0 + ret
                trades[nt, 0], trades[nt, 1], trades[nt, 2] = i_in, i + 1, pos
                trades[nt, 3], trades[nt, 4] = pos * (px - ep) / dist, ret
                trades[nt, 5] = 2 if reverse else (3 if timeout else 4)
                nt += 1
                pos = 0
        elif not friday_end and atr[i] > 0:
            d = 1 if s > entry else (-1 if (allow_short and s < -entry) else 0)
            if d != 0:
                dist = max(stop_k * atr[i], 3.0 * spread[i + 1])
                ep = (o[i + 1] + spread[i + 1] if d == 1 else o[i + 1]) + d * slip
                sl = ep - d * dist
                tp = ep + d * tp_r * dist
                force = min(1.0, max(0.0, (abs(s) - entry) / max(1e-9, 1.0 - entry)))
                risk = min(RISK_CAP, risk_lo + (risk_hi - risk_lo) * force ** risk_pow)
                size = min(risk / dist, MAX_LEV / ep)
                pos = d
                i_in = i + 1
    eq[n - 1] = equity
    return eq, trades[:nt]
