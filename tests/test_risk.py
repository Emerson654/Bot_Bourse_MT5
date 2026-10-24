"""Taille de position et distance minimale du stop."""

import pytest

from bot.risk import SymbolRules, daily_loss_hit, lot_size, stop_distance

GOLD = SymbolRules(point=0.01, volume_min=0.01, volume_step=0.01, volume_max=100.0, stops_level=10, filling_mode=1)


def test_lot_size_risque_un_pourcent():
    # 100 000 € × 1 % = 1 000 € ; stop à 5,00 $ ; 1 lot = 100 € par 1,00 de prix → 1000 / (5 × 100) = 2 lots
    assert lot_size(100_000, 0.01, 5.0, 100.0, GOLD) == pytest.approx(2.0)


def test_lot_size_arrondi_vers_le_bas():
    # 1000 / (3 × 100) = 3,333… → 3,33 (jamais plus de risque que prévu)
    assert lot_size(100_000, 0.01, 3.0, 100.0, GOLD) == pytest.approx(3.33)


def test_lot_size_sous_le_minimum_renvoie_zero():
    assert lot_size(100, 0.01, 50.0, 100.0, GOLD) == 0.0


def test_lot_size_plafonne_au_volume_max():
    assert lot_size(1e9, 0.01, 1.0, 100.0, GOLD) == 100.0


def test_lot_size_entrees_invalides():
    assert lot_size(100_000, 0.01, 0.0, 100.0, GOLD) == 0.0
    assert lot_size(100_000, 0.01, 5.0, 0.0, GOLD) == 0.0


def test_stop_elargi_au_minimum_du_courtier():
    # ATR minuscule : stop = stops_level (0,10) + spread (0,30) + 2 points
    assert stop_distance(0.01, 2.0, GOLD, 0.30) == pytest.approx(0.42)
    assert stop_distance(3.0, 2.0, GOLD, 0.30) == pytest.approx(6.0)


def test_perte_journaliere():
    assert daily_loss_hit(100_000, 97_000)
    assert not daily_loss_hit(100_000, 97_100)
