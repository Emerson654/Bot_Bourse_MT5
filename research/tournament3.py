"""Tournoi v3 : multi-temporalités (H1/H4/D1) et multi-actifs, sans volume imposé.

Différences avec la v1/v2 (intraday M5) :
- un bot = UN génome appliqué à TOUS les actifs (un même jeu de règles qui marche partout est bien
  moins suroptimisé qu'un réglage par actif) ; la note porte sur le portefeuille ;
- votants vus sur trois temporalités ; décisions aux clôtures H1 de la session 12h-17h ;
- positions de quelques heures à quelques jours, SL/TP chez le courtier la nuit, swap compté ;
- ligues par famille + ligue libre → fusion → grande finale (pondérations entre champions, combo
  non imposé) → sélection des actifs où le bot gagne à l'entraînement.
Même protocole walk-forward que la v1 : l'année testée n'est jamais vue.

Usage : python -m research.tournament3 CACHE OUT_JSON TEST_YEAR [--gens 60 --pop 300 --workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

import research.tournament as T
from research.sim3 import simulate
from research.tournament import Genome, _ema_smooth, crossover, mutate, random_genome

T.TSTOPS = (0, 6, 12, 24, 48, 72, 120)  # en bougies H1 (0 = pas de stop temporel)
MAX_DD = 0.15
_G: dict = {}


def _init(cache: str) -> None:
    with open(cache, "rb") as fh:
        _G.update(pickle.load(fh))


# ---------------------------------------------------------------------------------------------
# Évaluation d'un génome sur le portefeuille


def score_of(V: np.ndarray, g: Genome) -> np.ndarray:
    w = np.asarray(g.w, dtype=np.float64)
    return _ema_smooth(V[:, list(g.idx)].astype(np.float64) @ w / max(w.sum(), 1e-9), g.smooth)


def run_asset(A: dict, V: np.ndarray, g: Genome, a: int, b: int):
    s = score_of(V[a:b], g)
    sl = slice(a, b)
    return simulate(A["o"][sl], A["h"][sl], A["lo"][sl], A["c"][sl], A["spread"][sl], A["atr"][sl], A["minute"][sl],
                    A["day"][sl], A["weekday"][sl], s, g.entry, g.exit, g.stop_k, g.tp_r, g.tstop, bool(g.short),
                    A["slip"], A["swap_long"], A["swap_short"], A["triple_wd"])


def bounds(A: dict, start: str, end: str) -> tuple[int, int]:
    t = A["time"]
    return int(np.searchsorted(t, np.datetime64(start))), int(np.searchsorted(t, np.datetime64(end)))


def portfolio(g: Genome, period: tuple[str, str], vkey: str = "V", assets: list[str] | None = None) -> dict:
    """Rendements journaliers du portefeuille (somme des poches, 1 % de risque par trade et par actif) + trades."""
    daily: dict[int, float] = {}
    trades = []
    per_asset = {}
    for name, A in _G["assets"].items():
        if assets is not None and name not in assets:
            continue
        a, b = bounds(A, *period)
        if b - a < 500:
            continue
        eq, tr = run_asset(A, A[vkey], g, a, b)
        day = A["day"][a:b]
        last = np.flatnonzero(np.diff(day, append=day[-1] + 1))
        e = np.concatenate([[1.0], eq[last]])
        r = np.diff(e) / e[:-1]
        for d, x in zip(day[last], r):
            daily[int(d)] = daily.get(int(d), 0.0) + x
        trades.append(tr)
        per_asset[name] = (r, tr)
    days = sorted(daily)
    return {"days": np.array(days), "r": np.array([daily[d] for d in days]),
            "trades": np.concatenate(trades) if trades else np.zeros((0, 6)), "per_asset": per_asset}


def metrics(p: dict) -> dict:
    r, tr = p["r"], p["trades"]
    if len(r) < 20 or len(tr) == 0:
        return {"trades": int(len(tr)), "pf": 0.0, "max_dd": 0.0, "sharpe": 0.0, "tpd": 0.0, "ret": 0.0, "cagr": 0.0,
                "win": 0.0, "exp_r": 0.0}
    eq = np.cumprod(1 + r)
    g, lo = tr[tr[:, 4] > 0, 4].sum(), -tr[tr[:, 4] < 0, 4].sum()
    years = len(r) / 260
    return {"trades": int(len(tr)), "pf": float(g / lo) if lo > 0 else 99.0,
            "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(260)) if r.std() > 0 else 0.0,
            "tpd": len(tr) / len(r), "ret": float(eq[-1] - 1), "cagr": float(eq[-1] ** (1 / years) - 1),
            "win": float((tr[:, 4] > 0).mean()), "exp_r": float(tr[:, 3].mean())}


def fitness_of(m: dict) -> float:
    """Sharpe du portefeuille − pénalités (drawdown > 15 %, profit factor < 1, trop peu de trades)."""
    if m["trades"] < 30:
        return -9.0
    f = m["sharpe"]
    f -= max(0.0, -MAX_DD - m["max_dd"]) * 10
    f -= max(0.0, 1.0 - m["pf"]) * 2
    f -= max(0.0, 0.1 - m["tpd"]) * 5  # moins d'un trade tous les 10 jours : trop peu pour juger
    return f


def era_fitness(g: Genome, eras, vkey: str = "V") -> float:
    fs = [fitness_of(metrics(portfolio(g, e, vkey))) for e in eras]
    return 0.6 * float(np.mean(fs)) + 0.4 * float(np.min(fs))


# ---------------------------------------------------------------------------------------------
# Algorithme génétique (même mécanique que la v1)


def ga(args: tuple) -> list[tuple[float, Genome]]:
    vkey, pool, eras, gens, pop_n, seed, seeds_in = args
    rng = random.Random(seed)
    pop = list(seeds_in) + [random_genome(rng, pool) for _ in range(pop_n - len(seeds_in))]
    cache: dict = {}

    def fit(g: Genome) -> float:
        if g not in cache:
            cache[g] = era_fitness(g, eras, vkey)
        return cache[g]

    for _ in range(gens):
        ranked = sorted(pop, key=fit, reverse=True)
        elite, top = ranked[:3], ranked[:20]
        children = []
        while len(children) < int(pop_n * 0.87) - 3:
            c = crossover(rng.choice(elite), rng.choice(top), rng) if rng.random() < 0.7 else rng.choice(elite)
            children.append(mutate(c, rng, pool))
        pop = elite + children + [random_genome(rng, pool) for _ in range(pop_n - 3 - len(children))]
    return sorted(((fit(g), g) for g in set(pop)), key=lambda x: -x[0])


def select(cands: list[Genome], tr, va, eras, vkey: str = "V") -> list[Genome]:
    ft = np.array([era_fitness(g, eras, vkey) for g in cands])
    fv = np.array([fitness_of(metrics(portfolio(g, va, vkey))) for g in cands])
    rt, rv = np.argsort(np.argsort(-ft)), np.argsort(np.argsort(-fv))
    return [cands[i] for i in np.argsort(rt + rv + 1e-3 * rt)]


def significantly_better(r1: np.ndarray, r2: np.ndarray) -> bool:
    d = r1 - r2
    return d.std() > 0 and d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) > 1.2816


# ---------------------------------------------------------------------------------------------
# Déroulé d'un pli


def run_fold(cache: str, test_year: int, gens: int, pop: int, workers: int, seeds: int = 2) -> dict:
    _init(cache)
    last = max(A["time"][-1] for A in _G["assets"].values()) + np.timedelta64(1, "s")
    train_end = f"{test_year}-01-01" if test_year else str(last)
    test_end = f"{test_year + 1}-01-01" if test_year else str(last)
    val_start = str(np.datetime64(train_end[:10]) - np.timedelta64(365, "D"))
    tr, va, te = ("2009-06-01", val_start), (val_start, train_end), (train_end, test_end)
    d0, d1 = np.datetime64(tr[0]), np.datetime64(tr[1])
    step = (d1 - d0) // 3
    eras = [(str(d0 + i * step), str(d0 + (i + 1) * step)) for i in range(3)]
    fam = _G["families"]
    leagues = {f: [i for i, x in enumerate(fam) if x == f] for f in sorted(set(fam))}
    leagues["libre"] = list(range(len(fam)))
    t0 = time.time()

    jobs = [("V", pool, eras, gens, pop, test_year * 100 + s * 10 + li, ()) for li, pool in enumerate(leagues.values())
            for s in range(seeds)]
    with Pool(workers, initializer=_init, initargs=(cache,)) as p:
        res = p.map(ga, jobs)
    report: dict = {"test_year": test_year, "leagues": {}, "voters": _G["names"]}
    champs, names = [], []
    for li, lg in enumerate(leagues):
        pop_all = sorted((x for s in range(seeds) for x in res[li * seeds + s]), key=lambda x: -x[0])
        ranked = select(list(dict.fromkeys(g for _, g in pop_all))[:12], tr, va, eras)
        best = ranked[0]
        rv = [portfolio(g, va)["r"] for g in ranked[:3]]
        fused = len(ranked) >= 3 and len({len(r) for r in rv}) == 1 and \
            not any(significantly_better(rv[0], r) for r in rv[1:])
        if fused:  # comité des 3 premiers : moyenne de leurs scores, règles du champion
            for A in _G["assets"].values():
                A["F"] = np.column_stack([score_of(A["V"], g) for g in ranked[:3]]).astype(np.float32)
            fg = replace(best, idx=(0, 1, 2), w=(1.0, 1.0, 1.0), smooth=1)
            if era_fitness(fg, eras, "F") + fitness_of(metrics(portfolio(fg, va, "F"))) > \
                    era_fitness(best, eras) + fitness_of(metrics(portfolio(best, va))):
                report["leagues"][lg] = {"type": "fusion", "members": [asdict(g) for g in ranked[:3]], "rules": asdict(fg)}
                champs.append(("fusion", ranked[:3], fg))
                names.append(lg)
                continue
        report["leagues"][lg] = {"type": "seul", "genome": asdict(best), "valid": metrics(portfolio(best, va))}
        champs.append(("seul", [best], None))
        names.append(lg)

    # Scores des champions = votants de la finale
    for A in _G["assets"].values():
        cols = [score_of(A["V"], m[0]) if k == "seul" else np.mean([score_of(A["V"], x) for x in m], axis=0)
                for k, m, _ in champs]
        A["C"] = np.column_stack(cols).astype(np.float32)
        A.pop("F", None)
    with open(cache + ".final", "wb") as fh:
        pickle.dump(_G, fh)
    k = len(champs)
    singles = []
    for j, (kind, m, rules) in enumerate(champs):
        g0 = m[0] if kind == "seul" else rules
        singles.append(replace(g0, idx=(j,), w=(1.0,), smooth=1))
    combos = [replace(g, idx=tuple(range(k)), w=(1.0,) * k) for g in singles]
    jobs = [("C", list(range(k)), eras, gens, pop, test_year * 100 + 90 + s, tuple(singles + combos)) for s in range(seeds)]
    with Pool(workers, initializer=_init, initargs=(cache + ".final",)) as p:
        fin = p.map(ga, jobs)
    finalists = list(dict.fromkeys([g for r in fin for _, g in r[:10]] + singles + combos))
    winner = select(finalists, tr, va, eras, "C")[0]

    # Actifs retenus : ceux où le gagnant est rentable sur l'entraînement
    ptr = portfolio(winner, tr, "C")
    kept = [a for a, (r, t) in ptr["per_asset"].items() if len(t) >= 10 and r.mean() > 0]
    kept = kept or list(ptr["per_asset"])
    tot = sum(winner.w)
    report["champions"] = names
    report["winner"] = {"genome": asdict(winner), "weights": {names[i]: round(w / tot, 3) for i, w in zip(winner.idx, winner.w)},
                        "assets": kept, "train": metrics(portfolio(winner, tr, "C", kept)),
                        "valid": metrics(portfolio(winner, va, "C", kept))}
    report["per_asset_train"] = {a: metrics({"r": r, "trades": t}) for a, (r, t) in ptr["per_asset"].items()}
    if test_year:
        pt = portfolio(winner, te, "C", kept)
        report["winner"]["test"] = metrics(pt)
        report["test_daily"] = pt["r"].tolist()
        report["test_trades"] = pt["trades"][:, [3, 4]].tolist()
        report["per_asset_test"] = {a: metrics({"r": r, "trades": t}) for a, (r, t) in pt["per_asset"].items()}
    report["minutes"] = round((time.time() - t0) / 60, 1)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("test_year", type=int)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    rep = run_fold(a.cache, a.test_year, a.gens, a.pop, a.workers, a.seeds)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    w = rep["winner"]
    print(json.dumps({"poids": w["weights"], "actifs": w["assets"], "valid": w["valid"], "test": w.get("test")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
