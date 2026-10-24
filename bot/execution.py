"""Exécution des ordres : SL et TP toujours envoyés DANS l'ordre d'ouverture (stockés chez le courtier).

Interdits : trailing stop MT5 et tout SL/TP « virtuel » géré par le script.
Après chaque ordre, la position est relue : SL ou TP manquant → modification TRADE_ACTION_SLTP ;
si elle échoue → fermeture immédiate. `api` = le module MetaTrader5, ou `SimBroker` en local.
"""

from __future__ import annotations

import logging

log = logging.getLogger("bot")
MAGIC = 330_470
DEVIATION = 30  # glissement max accepté, en points


def filling_type(api, symbol: str) -> int:
    """Mode de remplissage accepté par le courtier : FOK, sinon IOC, sinon RETURN."""
    mode = api.symbol_info(symbol).filling_mode
    if mode & 1:
        return api.ORDER_FILLING_FOK
    if mode & 2:
        return api.ORDER_FILLING_IOC
    return api.ORDER_FILLING_RETURN


def _ok(res, api) -> bool:
    return res is not None and res.retcode == api.TRADE_RETCODE_DONE


def open_position(api, symbol: str, direction: int, volume: float, sl: float, tp: float, comment: str = "bot") -> int | None:
    """Ouvre au marché avec SL et TP dans la même requête ; retourne le ticket de position vérifié, ou None."""
    tick = api.symbol_info_tick(symbol)
    price = tick.ask if direction == 1 else tick.bid
    req = {
        "action": api.TRADE_ACTION_DEAL, "symbol": symbol, "volume": volume,
        "type": api.ORDER_TYPE_BUY if direction == 1 else api.ORDER_TYPE_SELL,
        "price": price, "sl": sl, "tp": tp, "deviation": DEVIATION, "magic": MAGIC,
        "comment": comment, "type_time": api.ORDER_TIME_GTC, "type_filling": filling_type(api, symbol),
    }
    res = api.order_send(req)
    if not _ok(res, api):
        log.warning("ordre refusé (code %s)", getattr(res, "retcode", None))
        return None
    pos = next((p for p in api.positions_get(symbol=symbol) or [] if p.magic == MAGIC), None)
    if pos is None:
        log.warning("position introuvable après l'ordre")
        return None
    return pos.ticket if ensure_protection(api, pos, sl, tp) else None


def ensure_protection(api, pos, sl: float | None = None, tp: float | None = None) -> bool:
    """Vérifie que la position a un SL et un TP non nuls côté serveur ; sinon les pose, sinon ferme."""
    if pos.sl and pos.tp:
        return True
    sl = sl or pos.sl
    tp = tp or pos.tp
    if sl and tp:
        res = api.order_send({"action": api.TRADE_ACTION_SLTP, "symbol": pos.symbol, "position": pos.ticket,
                              "sl": sl, "tp": tp, "magic": MAGIC})
        fresh = next((p for p in api.positions_get(ticket=pos.ticket) or []), None)
        if _ok(res, api) and fresh is not None and fresh.sl and fresh.tp:
            log.info("SL/TP reposés sur la position")
            return True
    log.warning("SL/TP impossibles à poser : fermeture immédiate")
    close_position(api, pos)
    return False


def close_position(api, pos, comment: str = "fermeture") -> bool:
    """Ferme au marché (ordre opposé sur le même ticket)."""
    tick = api.symbol_info_tick(pos.symbol)
    buy = pos.type == api.POSITION_TYPE_BUY
    res = api.order_send({
        "action": api.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume, "position": pos.ticket,
        "type": api.ORDER_TYPE_SELL if buy else api.ORDER_TYPE_BUY, "price": tick.bid if buy else tick.ask,
        "deviation": DEVIATION, "magic": MAGIC, "comment": comment, "type_time": api.ORDER_TIME_GTC,
        "type_filling": filling_type(api, pos.symbol),
    })
    return _ok(res, api)


def bot_positions(api, symbol: str) -> list:
    return [p for p in api.positions_get(symbol=symbol) or [] if p.magic == MAGIC]


def check_all_positions(api, symbol: str) -> None:
    """Au démarrage : toutes les positions du bot doivent avoir SL et TP, sinon correction ou fermeture."""
    for p in bot_positions(api, symbol):
        ensure_protection(api, p)
