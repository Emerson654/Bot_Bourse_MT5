"""Vérification des SL/TP : présents dans l'ordre, reposés si absents, sinon fermeture."""

from bot import execution as ex
from bot.sim_broker import Position, SimBroker


def test_sl_tp_envoyes_dans_l_ordre_d_ouverture():
    api = SimBroker()
    t = ex.open_position(api, "XAUUSD", 1, 0.5, 2395.0, 2410.0)
    assert t is not None
    req = api.requests[0]
    assert req["sl"] == 2395.0 and req["tp"] == 2410.0 and req["action"] == api.TRADE_ACTION_DEAL
    p = api.positions[0]
    assert (p.sl, p.tp) == (2395.0, 2410.0)


def test_sl_tp_perdus_sont_reposes():
    api = SimBroker(drop_sltp=True)
    t = ex.open_position(api, "XAUUSD", -1, 0.5, 2405.0, 2390.0)
    assert t is not None
    assert any(r["action"] == api.TRADE_ACTION_SLTP for r in api.requests)
    assert (api.positions[0].sl, api.positions[0].tp) == (2405.0, 2390.0)


def test_position_fermee_si_modification_impossible():
    api = SimBroker(drop_sltp=True, fail_modify=True)
    assert ex.open_position(api, "XAUUSD", 1, 0.5, 2395.0, 2410.0) is None
    assert api.positions == []


def test_demarrage_position_sans_stop_fermee():
    api = SimBroker()
    api.positions.append(Position(1, "XAUUSD", 0, 0.1, 2400.0, 0.0, 0.0, ex.MAGIC))
    api.positions.append(Position(2, "XAUUSD", 0, 0.1, 2400.0, 2390.0, 2420.0, ex.MAGIC))
    api.positions.append(Position(3, "XAUUSD", 0, 0.1, 2400.0, 0.0, 0.0, 999))  # pas au bot : intouchée
    ex.check_all_positions(api, "XAUUSD")
    assert sorted(p.ticket for p in api.positions) == [2, 3]


def test_mode_de_remplissage():
    api = SimBroker()
    assert ex.filling_type(api, "XAUUSD") == api.ORDER_FILLING_FOK


def test_plusieurs_positions_sur_le_meme_actif():
    api = SimBroker()
    t1 = ex.open_position(api, "XAUUSD", 1, 0.5, 2395.0, 2410.0)
    t2 = ex.open_position(api, "XAUUSD", 1, 0.3, 2390.0, 2420.0)
    assert t1 != t2 and len(api.positions) == 2
    p2 = next(p for p in api.positions if p.ticket == t2)
    assert (p2.volume, p2.sl, p2.tp) == (0.3, 2390.0, 2420.0)
