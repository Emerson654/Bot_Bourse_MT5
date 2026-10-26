"""Backtest du bot EN SERVICE (results/live/bot_params.json) sur tout l'historique et les 12 dernières semaines.

Lancé chaque samedi (workflow « Backtest hebdomadaire ») ou à la demande depuis le dashboard.
Rejoue exactement les règles des spécialistes (simulateur v6, ou v8 si le génome a les gènes du tournoi 8).
Sortie : results/weekly/latest.json (+ copie datée) avec courbe de capital (indice), mesures et détail par actif.

Usage : python -m research.backtest_live CACHE results/weekly
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from datetime import UTC, datetime

import numpy as np

import research.tournament as T
import research.tournament3 as T3
from research import sim4, sim5


def champion_scores(A: dict, params: dict) -> np.ndarray:
    cols = []
    for lg in params["champions"]:
        c = params["leagues"][lg]
        if c["type"] == "seul":
            cols.append(T3.score_of(A["V"], T.Genome(**c["genome"])))
        else:
            cols.append(np.mean([T3.score_of(A["V"], T.Genome(**m)) for m in c["members"]], axis=0))
    return np.column_stack(cols)


def run(A: dict, params: dict):
    g = T.Genome(**params["winner"]["genome"])
    C = champion_scores(A, params)
    s = np.zeros(len(A["c"]))
    s[A["dec"]] = T3.score_of(C, g)
    args = (A["o"], A["h"], A["lo"], A["c"], A["spread"], A["atr"], A["minute"], A["day"], A["weekday"], s, g.entry,
            g.exit, g.stop_k, g.tp_r, max(g.tstop, 24), bool(g.short), A["slip"], A["swap_long"], A["swap_short"],
            A["triple_wd"], g.risk_lo, g.risk_hi, g.risk_pow, g.atr_scale, g.max_pos)
    if "er_min" in params["winner"]["genome"]:
        return sim5.simulate(*args, A["er"], A["adx"], g.er_min, g.adx_min, g.trail_k)
    return sim4.simulate(*args)


def main() -> None:
    cache, out = sys.argv[1], sys.argv[2]
    with open(cache, "rb") as fh:
        G = pickle.load(fh)
    with open("results/live/bot_params.json") as fh:
        live = json.load(fh)
    daily: dict[int, float] = {}
    per_asset = {}
    all_tr = []
    for sym, params in live.items():
        A = G["assets"].get(sym)
        if A is None:
            continue
        names = G["names"]
        if names != params["voters"]:  # même ordre de votants que le tournoi
            idx = [names.index(v) if v in names else -1 for v in params["voters"]]
            A["V"] = np.column_stack([A["V"][:, i] if i >= 0 else np.zeros(len(A["dec"])) for i in idx])
        eq, tr = run(A, params)
        day = A["day"]
        last = np.flatnonzero(np.diff(day, append=day[-1] + 1))
        e = np.concatenate([[1.0], eq[last]])
        r = np.diff(e) / e[:-1]
        for d, x in zip(day[last], r):
            daily[int(d)] = daily.get(int(d), 0.0) + x
        all_tr.append(tr)
        per_asset[sym] = T3.metrics({"r": r, "trades": tr})
    days = np.array(sorted(daily))
    r = np.array([daily[d] for d in days])
    tr = np.concatenate(all_tr) if all_tr else np.zeros((0, 6))
    eq = np.cumprod(1 + r)
    recent = days >= days[-1] - 84
    wd = np.array([datetime.fromordinal(int(d)).weekday() for d in days])
    rep = {
        "date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "actifs": sorted(per_asset),
        "global": T3.metrics({"r": r, "trades": tr}),
        "douze_semaines": T3.metrics({"r": r[recent], "trades": tr[:0]}) | {"ret": float(np.prod(1 + r[recent]) - 1)},
        "par_actif": per_asset,
        # rendement moyen par jour de la semaine (lundi → vendredi), en %
        "jours_semaine": [round(float(r[wd == k].mean() * 100), 4) if (wd == k).any() else 0.0 for k in range(5)],
        # courbe de capital (indice 1 = départ), un point par semaine pour rester léger
        "courbe": [[datetime.fromordinal(int(d)).strftime("%Y-%m-%d"), round(float(x), 5)] for d, x in zip(days[::5], eq[::5])],
    }
    os.makedirs(out, exist_ok=True)
    for name in ("latest.json", f"{rep['date'][:10]}.json"):
        with open(os.path.join(out, name), "w") as fh:
            json.dump(rep, fh, indent=1, default=float)
    g = rep["global"]
    print(f"Backtest du bot en service : {g['trades']} trades, rendement annuel {g['cagr']:+.1%}, "
          f"drawdown max {g['max_dd']:.1%}, 12 dernières semaines {rep['douze_semaines']['ret']:+.1%}")


if __name__ == "__main__":
    main()
