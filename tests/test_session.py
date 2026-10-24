"""Horaires de session en heure de Paris, été comme hiver."""

from datetime import UTC, datetime, timedelta

from bot.session import can_open, must_stop

START = datetime(2026, 7, 1, 10, 0, tzinfo=UTC)  # mercredi 12h00 Paris (été, UTC+2)


def test_ouverture_ete_et_hiver():
    assert can_open(datetime(2026, 7, 1, 10, 5, tzinfo=UTC), START)
    winter = datetime(2026, 1, 7, 11, 5, tzinfo=UTC)  # 12h05 Paris (hiver, UTC+1)
    assert can_open(winter, winter)
    assert not can_open(datetime(2026, 1, 7, 10, 5, tzinfo=UTC), winter)  # 11h05 Paris


def test_plus_d_entree_apres_17h30():
    assert not can_open(datetime(2026, 7, 1, 15, 31, tzinfo=UTC), START)


def test_plus_d_entree_apres_5h30_de_job():
    early = START - timedelta(minutes=10)  # job lancé à 11h50 : 5h30 plus tard = 17h20, avant 17h30
    assert can_open(early + timedelta(hours=5, minutes=29), early)
    assert not can_open(early + timedelta(hours=5, minutes=30), early)


def test_arret_17h50_et_week_end():
    assert must_stop(datetime(2026, 7, 1, 15, 50, tzinfo=UTC), START)
    assert not must_stop(datetime(2026, 7, 1, 15, 49, tzinfo=UTC), START)
    assert must_stop(datetime(2026, 7, 4, 12, 0, tzinfo=UTC), START)  # samedi
