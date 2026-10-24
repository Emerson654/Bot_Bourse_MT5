"""Bibliothèque de « votants » : chaque votant donne un avis v_t ∈ [-1, 1] (vendre ↔ acheter).

Calculés sur les bougies M5 clôturées ; les unités de temps supérieures sont simulées par des
fenêtres plus longues (12 bougies M5 = 1 h), ce qui évite tout look-ahead lié au rééchantillonnage.
Chaque votant appartient à une famille (= ligue du tournoi).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bot import indicators as ind


@dataclass
class Bars:
    """Bougies M5. `minute` = minutes depuis minuit heure de Paris ; `day` = identifiant du jour de session."""

    o: np.ndarray
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    spread: np.ndarray  # spread en prix (déjà multiplié par le point)
    minute: np.ndarray
    day: np.ndarray
    weekday: np.ndarray


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(x), -1.0, 1.0)


def _z(x: np.ndarray, n: int) -> np.ndarray:
    sd = ind.rolling_std(x, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (x - ind.sma(x, n)) / sd, 0.0)


def _session_anchor(b: Bars, start_min: int, end_min: int) -> tuple[np.ndarray, np.ndarray]:
    """Plus haut / plus bas de la fenêtre [start, end) du jour, connus seulement après `end` (sinon NaN)."""
    hi = np.full(len(b.c), np.nan)
    lo = np.full(len(b.c), np.nan)
    cur_day, h, l, done = -1, -np.inf, np.inf, False
    for i in range(len(b.c)):
        if b.day[i] != cur_day:
            cur_day, h, l, done = b.day[i], -np.inf, np.inf, False
        m = b.minute[i]
        if start_min <= m < end_min:
            h, l = max(h, b.h[i]), min(l, b.lo[i])
        elif m >= end_min and np.isfinite(h):
            done = True
        if done:
            hi[i], lo[i] = h, l
    return hi, lo


def _prev_day_hl(b: Bars) -> tuple[np.ndarray, np.ndarray]:
    """Plus haut / plus bas de la veille (journée serveur complète)."""
    hi = np.full(len(b.c), np.nan)
    lo = np.full(len(b.c), np.nan)
    days = b.day
    starts = np.flatnonzero(np.diff(days, prepend=days[0] - 1))
    ends = np.append(starts[1:], len(days))
    for k in range(1, len(starts)):
        s0, e0 = starts[k - 1], ends[k - 1]
        hi[starts[k]:ends[k]] = b.h[s0:e0].max()
        lo[starts[k]:ends[k]] = b.lo[s0:e0].min()
    return hi, lo


def _vwap_day(b: Bars) -> np.ndarray:
    """Prix moyen du jour (sans volume réel en CFD : moyenne des prix typiques depuis l'ouverture du jour)."""
    tp = (b.h + b.lo + b.c) / 3
    out = np.empty(len(tp))
    s, k, d = 0.0, 0, -1
    for i in range(len(tp)):
        if b.day[i] != d:
            s, k, d = 0.0, 0, b.day[i]
        s += tp[i]
        k += 1
        out[i] = s / k
    return out


def _breakout(c: np.ndarray, hi: np.ndarray, lo: np.ndarray, a: np.ndarray) -> np.ndarray:
    """+1 au-dessus du plus haut, −1 sous le plus bas, entre les deux : position relative en [-1, 1] atténuée."""
    with np.errstate(invalid="ignore", divide="ignore"):
        up = (c - hi) / a
        dn = (lo - c) / a
    v = np.where(up > 0, np.minimum(1.0, 0.5 + up), np.where(dn > 0, -np.minimum(1.0, 0.5 + dn), 0.0))
    return np.nan_to_num(v)


def compute_votes(b: Bars) -> dict[str, tuple[str, np.ndarray]]:
    """Retourne {nom: (famille, votes)}. Toutes les fenêtres sont en bougies M5."""
    c, h, lo = b.c, b.h, b.lo
    a = ind.atr(h, lo, c, 14)
    a = np.where(np.isnan(a) | (a <= 0), np.nanmedian(a), a)
    a_h1 = ind.atr(h, lo, c, 12 * 14)
    a_h1 = np.where(np.isnan(a_h1) | (a_h1 <= 0), a, a_h1)
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    out: dict[str, tuple[str, np.ndarray]] = {}

    # --- Tendance -----------------------------------------------------------------
    for f, s in ((9, 21), (20, 50), (12 * 9, 12 * 21), (50, 200)):
        out[f"ema_{f}_{s}"] = ("tendance", _clip((ind.ema(c, f) - ind.ema(c, s)) / a_h1))
    for n in (12, 48, 144):
        out[f"tsmom_{n}"] = ("tendance", _clip((c - np.roll(c, n)) / (a * np.sqrt(n)) * 1.5))
    out["macd"] = ("tendance", _clip(ind.macd_hist(c) / a * 4))
    out["macd_h1"] = ("tendance", _clip(ind.macd_hist(c, 144, 312, 108) / a_h1 * 2))
    for n in (24, 96):
        sl, r2 = ind.linreg_slope(c, n)
        out[f"linreg_{n}"] = ("tendance", _clip(sl * n / a * np.nan_to_num(r2)))
    adx_v, pdi, mdi = ind.adx(h, lo, c, 14)
    out["dmi"] = ("tendance", _clip((pdi - mdi) / 25.0 * (np.nan_to_num(adx_v) > 20)))
    # Supertrend simplifié : position du prix par rapport à la bande ATR autour de la médiane
    mid = (ind.rolling_max(h, 20) + ind.rolling_min(lo, 20)) / 2
    out["channel_mid"] = ("tendance", _clip((c - mid) / (2 * a)))

    # --- Retour à la moyenne ----------------------------------------------------------
    out["rsi_2"] = ("retour", _clip((50 - ind.rsi(c, 2)) / 40))
    out["rsi_14"] = ("retour", _clip((50 - ind.rsi(c, 14)) / 25))
    out["rsi_14_h1"] = ("retour", _clip((50 - ind.rsi(c, 168)) / 15))
    for n in (20, 60):
        out[f"bollinger_{n}"] = ("retour", _clip(-_z(c, n) / 2))
    out["stoch"] = ("retour", _clip((0.5 - ind.stochastic(h, lo, c, 14)) * 2))
    out["cci"] = ("retour", _clip(-ind.cci(h, lo, c, 20) / 200))
    vw = _vwap_day(b)
    out["vwap_dev"] = ("retour", _clip(-(c - vw) / (2 * a)))
    out["zret_12"] = ("retour", _clip(-(c - np.roll(c, 12)) / (a * np.sqrt(12))))

    # --- Cassures et volatilité --------------------------------------------------------
    for n in (12, 36, 96):
        hh, ll = ind.rolling_max(h, n), ind.rolling_min(lo, n)
        out[f"donchian_{n}"] = ("cassure", _breakout(c, np.roll(hh, 1), np.roll(ll, 1), a))
    pdh, pdl = _prev_day_hl(b)
    out["prev_day"] = ("cassure", _breakout(c, pdh, pdl, a))
    ah, al = _session_anchor(b, 2 * 60, 9 * 60)  # range asiatique 02h-09h Paris
    out["asia_range"] = ("cassure", _breakout(c, ah, al, a))
    lh, ll2 = _session_anchor(b, 9 * 60, 12 * 60)  # matinée Londres 09h-12h
    out["london_range"] = ("cassure", _breakout(c, lh, ll2, a))
    nh, nl = _session_anchor(b, 15 * 60 + 30, 16 * 60)  # 30 premières minutes de New York
    out["ny_orb"] = ("cassure", _breakout(c, nh, nl, a))
    kc_up, kc_dn = ind.ema(c, 20) + 1.5 * a, ind.ema(c, 20) - 1.5 * a
    out["keltner"] = ("cassure", _breakout(c, kc_up, kc_dn, a))
    bb_sd = ind.rolling_std(c, 20)
    squeeze = np.roll((2 * bb_sd) < (1.5 * a), 1)  # compression à la bougie précédente
    out["squeeze"] = ("cassure", _clip(np.sign(c - ind.sma(c, 20)) * squeeze * np.abs(_z(c, 20)) / 2))

    # --- Statistiques -------------------------------------------------------------------
    hu = ind.hurst_rs(r, 96)
    mom = np.sign(c - np.roll(c, 24))
    out["hurst_mom"] = ("stat", _clip(mom * (np.nan_to_num(hu, nan=0.5) - 0.5) * 8))
    vr = ind.variance_ratio(r, 144, 4)
    out["vr_mom"] = ("stat", _clip(mom * (vr - 1) * 3))
    ac = ind.autocorr(r, 96)
    out["autocorr"] = ("stat", _clip(np.sign(r) * ac * 5))
    _, ks = ind.kalman_trend(c, 1e-3)
    out["kalman"] = ("stat", _clip(ks / a * 20))
    _, ks2 = ind.kalman_trend(c, 1e-5)
    out["kalman_lent"] = ("stat", _clip(ks2 / a * 60))
    out["zscore_96"] = ("stat", _clip(-_z(c, 96) / 2.5))
    return out


FAMILIES = ("tendance", "retour", "cassure", "stat")
