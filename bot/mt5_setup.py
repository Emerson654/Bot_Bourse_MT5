"""Connexion au terminal MT5 du runner, avec les réglages nécessaires au bot.

Un terminal neuf a le bouton « Algo Trading » désactivé (order_send refusé) et limite l'historique
à 100 000 bougies. Ces réglages vivent dans <dossier de données>/config/common.ini, dossier créé au
premier démarrage : on démarre une fois, on écrit le fichier, puis on redémarre.
"""

from __future__ import annotations

import os
import sys
import time

SETTINGS = "[Experts]\nEnabled=1\nAllowLiveTrading=1\nAllowDllImport=0\n[Charts]\nMaxBars=5000000\n"


def _init(mt5) -> bool:
    return mt5.initialize(
        path=os.environ.get("MT5_PATH"),
        login=int(os.environ["MT5_LOGIN"].strip()),
        password=os.environ["MT5_PASSWORD"].strip(),
        server=os.environ["MT5_SERVER"].strip(),
        timeout=120_000,
    )


def connect():
    """Retourne le module MetaTrader5 connecté, Algo Trading activé si possible."""
    import MetaTrader5 as mt5

    if not _init(mt5):
        sys.exit(f"Connexion MT5 impossible : {mt5.last_error()}")
    info = mt5.terminal_info()
    if not info.trade_allowed or info.maxbars < 5_000_000:
        cfg = os.path.join(info.data_path, "config", "common.ini")
        mt5.shutdown()
        time.sleep(3)
        os.system("taskkill /F /IM terminal64.exe >NUL 2>&1")  # le terminal réécrit sa config en quittant
        time.sleep(2)
        old = open(cfg, encoding="utf-16", errors="ignore").read() if os.path.exists(cfg) else ""
        # On retire les anciennes sections [Experts] et [Charts], puis on ajoute les nôtres
        keep, skip = [], False
        for line in old.splitlines():
            if line.startswith("["):
                skip = line.strip() in ("[Experts]", "[Charts]")
            if not skip:
                keep.append(line)
        with open(cfg, "w", encoding="utf-16") as fh:
            fh.write("\n".join(keep).strip() + "\n" + SETTINGS)
        if not _init(mt5):
            sys.exit(f"Reconnexion MT5 impossible : {mt5.last_error()}")
    return mt5
