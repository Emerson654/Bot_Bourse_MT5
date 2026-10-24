"""Gestion du risque : taille de position, distance minimale du stop, perte journalière."""

from __future__ import annotations

import math
from dataclasses import dataclass

RISK_PER_TRADE = 0.01
DAILY_LOSS_MAX = 0.03
MAX_POSITIONS = 1


@dataclass
class SymbolRules:
    """Règles du courtier lues dans `symbol_info` avant chaque ordre."""

    point: float
    volume_min: float
    volume_step: float
    volume_max: float
    stops_level: int  # distance minimale SL/TP en points
    filling_mode: int  # masque : 1 = FOK, 2 = IOC


def min_stop_distance(rules: SymbolRules, spread: float) -> float:
    """Distance minimale autorisée entre prix et stop : stops_level (points) + spread + 2 points de marge."""
    return rules.stops_level * rules.point + spread + 2 * rules.point


def stop_distance(atr: float, stop_k: float, rules: SymbolRules, spread: float) -> float:
    """Stop = k × ATR, élargi si le courtier impose une distance minimale plus grande."""
    return max(stop_k * atr, min_stop_distance(rules, spread))


def lot_size(equity: float, risk: float, dist: float, value_per_lot: float, rules: SymbolRules) -> float:
    """Volume = (capital × risque) / (distance du stop × valeur d'un mouvement de 1 sur 1 lot).

    `value_per_lot` = gain/perte en devise du compte pour 1 lot et 1,00 de prix (order_calc_profit).
    Arrondi vers le BAS au pas de volume (on ne risque jamais plus que prévu) ; 0 si < volume_min.
    """
    if dist <= 0 or value_per_lot <= 0 or equity <= 0:
        return 0.0
    raw = equity * risk / (dist * value_per_lot)
    steps = math.floor(raw / rules.volume_step + 1e-9)
    vol = min(steps * rules.volume_step, rules.volume_max)
    vol = round(vol, 8)
    return vol if vol >= rules.volume_min else 0.0


def daily_loss_hit(start_equity: float, equity: float, limit: float = DAILY_LOSS_MAX) -> bool:
    """Vrai si la perte du jour (réalisée + latente) atteint la limite."""
    return start_equity > 0 and (equity - start_equity) / start_equity <= -limit
