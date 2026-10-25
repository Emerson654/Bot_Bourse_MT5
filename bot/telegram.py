"""Notifications Telegram (repris du bot T-B, sans dépendance externe).

Identifiants dans les secrets GitHub : TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID. Sans eux, rien n'est envoyé.
Les messages ne sont jamais écrits dans les logs (dépôt public).
"""

from __future__ import annotations

import html
import json
import os
import urllib.request


class Telegram:
    def __init__(self) -> None:
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    @property
    def ready(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        """Envoie un message HTML ; ne lève jamais (une notification ratée ne doit pas arrêter le bot)."""
        if not self.ready:
            return False
        body = json.dumps({"chat_id": self.chat_id, "text": text[:4096], "parse_mode": "HTML",
                           "disable_web_page_preview": True}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return bool(json.load(r).get("ok"))
        except Exception:  # noqa: BLE001
            return False


def esc(x) -> str:
    return html.escape(str(x))


def price(x: float) -> str:
    return f"{x:,.2f}".replace(",", " ") if x >= 100 else f"{x:.5g}"
