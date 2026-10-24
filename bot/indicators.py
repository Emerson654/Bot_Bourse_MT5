"""Indicateurs techniques en numpy, sans look-ahead : la valeur en t n'utilise que les bougies ≤ t.

Conventions : tableaux float64 de même longueur, NaN tant que la fenêtre n'est pas remplie.
"""

from __future__ import annotations

import numpy as np


def ema(x: np.ndarray, n: int) -> np.ndarray:
    """Moyenne exponentielle : e_t = e_{t-1} + α (x_t − e_{t-1}), α = 2/(n+1), amorcée par la moyenne simple."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    a = 2.0 / (n + 1)
    e = np.mean(x[:n])
    out[n - 1] = e
    for i in range(n, len(x)):
        e += a * (x[i] - e)
        out[i] = e
    return out


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Moyenne de Wilder (α = 1/n), utilisée par RSI, ATR et ADX."""
    out = np.full(len(x), np.nan)
    v = np.where(np.isnan(x), 0.0, x)
    if len(x) < n:
        return out
    e = np.mean(v[:n])
    out[n - 1] = e
    for i in range(n, len(x)):
        e += (v[i] - e) / n
        out[i] = e
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    """Écart-type glissant : √(E[x²] − E[x]²)."""
    m = sma(x, n)
    m2 = sma(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).max(axis=1)
    return out


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    return -rolling_max(-x, n)


def true_range(h: np.ndarray, lo: np.ndarray, c: np.ndarray) -> np.ndarray:
    """TR_t = max(h − l, |h − c_{t-1}|, |l − c_{t-1}|)."""
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - lo, np.maximum(np.abs(h - pc), np.abs(lo - pc)))


def atr(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    return rma(true_range(h, lo, c), n)


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI de Wilder : 100 − 100/(1 + RMA(gains)/RMA(pertes))."""
    d = np.diff(c, prepend=c[0])
    g = rma(np.maximum(d, 0.0), n)
    p = rma(np.maximum(-d, 0.0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p == 0, 100.0, 100.0 - 100.0 / (1.0 + g / p))


def macd_hist(c: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> np.ndarray:
    m = ema(c, fast) - ema(c, slow)
    out = np.full(len(c), np.nan)
    ok = ~np.isnan(m)
    if ok.sum() >= sig:
        out[ok] = m[ok] - ema(m[ok], sig)
    return out


def adx(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, +DI, −DI de Wilder."""
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(lo, prepend=lo[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(h, lo, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * rma(pdm, n) / tr
        mdi = 100 * rma(mdm, n) / tr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    return rma(np.nan_to_num(dx), n), pdi, mdi


def stochastic(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    """%K = (c − min_n(l)) / (max_n(h) − min_n(l)) ∈ [0, 1]."""
    hh, ll = rolling_max(h, n), rolling_min(lo, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(hh > ll, (c - ll) / (hh - ll), 0.5)


def cci(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 20) -> np.ndarray:
    """CCI approché par l'écart-type : (tp − SMA(tp)) / (0,015 · σ·√(2/π)) (écart absolu moyen ≈ σ√(2/π))."""
    tp = (h + lo + c) / 3
    sd = rolling_std(tp, n) * np.sqrt(2 / np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (tp - sma(tp, n)) / (0.015 * sd), 0.0)


def linreg_slope(c: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Pente MCO de c sur [t−n+1, t] et R². Pente = Cov(x, y)/Var(x)."""
    x = np.arange(n, dtype=float)
    xm, xv = x.mean(), ((x - x.mean()) ** 2).sum()
    from numpy.lib.stride_tricks import sliding_window_view
    slope = np.full(len(c), np.nan)
    r2 = np.full(len(c), np.nan)
    if len(c) < n:
        return slope, r2
    w = sliding_window_view(c, n)
    ym = w.mean(axis=1)
    cov = ((w - ym[:, None]) * (x - xm)).sum(axis=1)
    b = cov / xv
    yv = ((w - ym[:, None]) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(yv > 0, cov * cov / (xv * yv), 0.0)
    slope[n - 1:] = b
    r2[n - 1:] = rr
    return slope, r2


def kalman_trend(c: np.ndarray, q: float = 1e-4, r: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Filtre de Kalman niveau + pente (modèle à vitesse constante), bruits relatifs à la variance des rendements.

    État x = [niveau, pente] ; F = [[1, 1], [0, 1]] ; observation y = niveau + bruit.
    Réf. : Harvey (1989), « Forecasting, Structural Time Series Models and the Kalman Filter ».
    """
    n = len(c)
    lvl, slp = np.full(n, np.nan), np.full(n, np.nan)
    var = np.nanvar(np.diff(c[: min(n, 2000)])) or 1.0
    Q = np.array([[q, 0.0], [0.0, q * 0.01]]) * var
    R = r * var
    x = np.array([c[0], 0.0])
    P = np.eye(2) * var
    for i in range(n):
        x = np.array([x[0] + x[1], x[1]])
        P = np.array([[P[0, 0] + 2 * P[0, 1] + P[1, 1], P[0, 1] + P[1, 1]], [P[0, 1] + P[1, 1], P[1, 1]]]) + Q
        k0 = P[0, 0] / (P[0, 0] + R)
        k1 = P[0, 1] / (P[0, 0] + R)
        e = c[i] - x[0]
        x = np.array([x[0] + k0 * e, x[1] + k1 * e])
        P = np.array([[(1 - k0) * P[0, 0], (1 - k0) * P[0, 1]], [P[0, 1] - k1 * P[0, 0], P[1, 1] - k1 * P[0, 1]]])
        lvl[i], slp[i] = x
    return lvl, slp


def hurst_rs(r: np.ndarray, n: int) -> np.ndarray:
    """Exposant de Hurst glissant par la variance des sommes agrégées : Var(Σ_k r) ∝ k^{2H}.

    H estimé par régression log-log sur k ∈ {1, 2, 4, 8} dans une fenêtre de n rendements.
    H > 0,5 : persistance (tendance) ; H < 0,5 : antipersistance (retour à la moyenne).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(r), np.nan)
    if len(r) < n:
        return out
    w = sliding_window_view(np.nan_to_num(r), n)
    ks = np.array([1, 2, 4, 8])
    lv = []
    for k in ks:
        m = n // k
        agg = w[:, : m * k].reshape(len(w), m, k).sum(axis=2)
        lv.append(np.log(agg.var(axis=1) + 1e-18))
    lv = np.array(lv)  # (4, T)
    lk = np.log(ks)
    b = ((lk - lk.mean())[:, None] * (lv - lv.mean(axis=0))).sum(axis=0) / ((lk - lk.mean()) ** 2).sum()
    out[n - 1:] = b / 2
    return out


def variance_ratio(r: np.ndarray, n: int, k: int = 4) -> np.ndarray:
    """VR(k) = Var(r_t + … + r_{t−k+1}) / (k · Var(r_t)) sur une fenêtre de n (Lo & MacKinlay, 1988)."""
    rk = np.convolve(np.nan_to_num(r), np.ones(k), mode="full")[: len(r)]
    v1 = rolling_std(np.nan_to_num(r), n) ** 2
    vk = rolling_std(rk, n) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v1 > 0, vk / (k * v1), 1.0)


def autocorr(r: np.ndarray, n: int) -> np.ndarray:
    """Autocorrélation d'ordre 1 glissante des rendements."""
    x = np.nan_to_num(r)
    y = np.roll(x, 1)
    y[0] = 0
    mx, my = sma(x, n), sma(y, n)
    cov = sma(x * y, n) - mx * my
    sx, sy = rolling_std(x, n), rolling_std(y, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((sx > 0) & (sy > 0), cov / (sx * sy), 0.0)
