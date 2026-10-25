"""Tournoi des bots XAUUSD intraday : ligues par famille → fusion → grande finale pondérée.

Un « bot » = un génome : sous-ensemble de votants + poids, lissage du score, règles de trading
(seuils d'entrée/sortie, stop k·ATR, take-profit R·stop, stop temporel, délai entre trades,
vente à découvert autorisée ou non, blocage des annonces US).

Déroulé d'un « pli » (fold) : entraînement [début, fin_train], validation = 12 derniers mois de
l'entraînement (jamais utilisés par l'algorithme génétique), test = période suivante, jamais vue.
1. Tour 1 : une ligue par famille de votants (+ une ligue libre), algorithme génétique avec
   élitisme, croisement, mutation et immigrés (repris du tournoi de T-B).
2. Fusion : si les 3 meilleurs d'une ligue ne sont pas significativement différents sur la
   validation, on crée le bot « comité » qui moyenne leurs scores ; il remplace le champion s'il
   fait mieux (rang entraînement + rang validation).
3. Grande finale : les champions de ligue deviennent des votants ; le génétique cherche leurs
   pondérations et de nouvelles règles de trading. Le gagnant (combo ou champion seul) est le bot du pli.
Réf. : Pardo (2008) ch. 11 (walk-forward) ; Aronson (2006) ; Bailey & López de Prado (2014) (Sharpe dégonflé).

Usage : python -m research.tournament CACHE OUT_JSON TEST_YEAR [--gens 60 --pop 400 --workers 4]
        TEST_YEAR = 0 → pli de production (test = néant, validation = 12 derniers mois).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool

import numpy as np
from numba import njit

from research.sim import metrics, simulate

SLIP = 0.05  # glissement en $ par exécution
MIN_TPD = 2.0  # trades par jour visés (intraday, volume)
HARD_TPD = 1.0  # règle stricte : sous 1 trade/jour, le bot est classé derrière tous les autres
COMBO_MIN = 3  # la grande finale doit combiner au moins 3 champions
MAX_DD = 0.15
SMOOTHS = (1, 2, 3, 5)
TSTOPS = (0, 6, 12, 24, 36, 48)  # en bougies M5 (0 = pas de stop temporel)
TPS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class Genome:
    idx: tuple  # indices des votants (colonnes de la matrice de votes ou des champions en finale)
    w: tuple  # poids ≥ 0
    smooth: int
    entry: float
    exit: float
    stop_k: float
    tp_r: float
    tstop: int
    cooldown: int
    short: int
    news: int
    # Risque selon la conviction (v4) : risque = lo + (hi − lo) × force^pow, force = (|score| − seuil)/(1 − seuil)
    risk_lo: float = 0.01
    risk_hi: float = 0.01
    risk_pow: float = 1.0
    # v6 : échelle du stop (1 ≈ ATR horaire, 2 ≈ 4 h, 5 ≈ journalier) et positions simultanées max
    atr_scale: float = 1.0
    max_pos: int = 1


# ---------------------------------------------------------------------------------------------
# Score et évaluation


@njit(cache=True)
def _ema_smooth(x, n):
    if n <= 1:
        return x
    out = np.empty_like(x)
    a = 2.0 / (n + 1)
    e = x[0]
    for i in range(len(x)):
        e += a * (x[i] - e)
        out[i] = e
    return out


def score_of(V: np.ndarray, g: Genome) -> np.ndarray:
    """score_t = EMA_n( Σ w_i v_{i,t} / Σ w_i ) ∈ [-1, 1]."""
    w = np.asarray(g.w, dtype=np.float64)
    s = V[:, list(g.idx)].astype(np.float64) @ w / max(w.sum(), 1e-9)
    return _ema_smooth(s, g.smooth)


def run(D: dict, V: np.ndarray, g: Genome, lo_i: int, hi_i: int):
    """Simule le génome sur les bougies [lo_i, hi_i)."""
    s = score_of(V[lo_i:hi_i], g)
    sl = slice(lo_i, hi_i)
    return simulate(D["o"][sl], D["h"][sl], D["lo"][sl], D["c"][sl], D["spread"][sl], D["atr"][sl],
                    D["minute"][sl], D["day"][sl], s, g.entry, g.exit, g.stop_k, g.tp_r, g.tstop,
                    g.cooldown, bool(g.short), bool(g.news), SLIP)


def fitness_of(m: dict) -> float:
    """Note = Sharpe journalier − pénalités (drawdown > 15 %, < 2 trades/jour, échantillon trop petit)."""
    if m["trades"] < 30:
        return -9.0
    if m["tpd"] < HARD_TPD:  # note graduée pour que le génétique remonte vers le volume visé
        return -5.0 - (HARD_TPD - m["tpd"]) * 3
    f = m["sharpe"]
    f -= max(0.0, -MAX_DD - m["max_dd"]) * 10
    f -= max(0.0, MIN_TPD - m["tpd"]) / MIN_TPD * 1.5
    f -= max(0.0, 1.0 - m["pf"]) * 2
    return f


def era_fitness(D: dict, V: np.ndarray, g: Genome, eras: list[tuple[int, int]]) -> float:
    """Cohérence dans le temps : 0,6 × moyenne + 0,4 × pire des tranches d'entraînement."""
    fs = []
    for a, b in eras:
        eq, tr = run(D, V, g, a, b)
        fs.append(fitness_of(metrics(eq, tr, D["day"][a:b])))
    return 0.6 * float(np.mean(fs)) + 0.4 * float(np.min(fs))


def daily_returns(D: dict, V: np.ndarray, g: Genome, a: int, b: int) -> np.ndarray:
    eq, _ = run(D, V, g, a, b)
    day = D["day"][a:b]
    last = np.flatnonzero(np.diff(day, append=day[-1] + 1))
    e = np.concatenate([[1.0], eq[last]])
    return np.diff(e) / e[:-1]


# ---------------------------------------------------------------------------------------------
# Algorithme génétique


def random_genome(rng: random.Random, pool: list[int]) -> Genome:
    k = rng.choice([1, 2, 3, 4, 5, 6, 8])
    idx = tuple(sorted(rng.sample(pool, min(k, len(pool)))))
    entry = round(rng.uniform(0.05, 0.7), 3)
    return Genome(idx, tuple(round(rng.uniform(0.2, 1.0), 3) for _ in idx), rng.choice(SMOOTHS), entry,
                  round(rng.uniform(-0.3, entry), 3), round(rng.uniform(0.8, 4.0), 2), rng.choice(TPS),
                  rng.choice(TSTOPS), rng.choice([0, 1, 2, 3, 6]), rng.choice([0, 1, 1]), rng.choice([0, 1]))


def mutate(g: Genome, rng: random.Random, pool: list[int], rate: float = 0.3) -> Genome:
    d = asdict(g)
    idx, w = list(g.idx), list(g.w)
    if rng.random() < rate and len(pool) > len(idx):  # ajout d'un votant
        i = rng.choice([p for p in pool if p not in idx])
        idx.append(i)
        w.append(round(rng.uniform(0.2, 1.0), 3))
    if rng.random() < rate and len(idx) > 1:  # retrait
        j = rng.randrange(len(idx))
        idx.pop(j)
        w.pop(j)
    for j in range(len(w)):
        if rng.random() < rate:
            w[j] = round(min(1.0, max(0.02, w[j] * rng.uniform(0.5, 1.6))), 3)
    order = np.argsort(idx)
    d["idx"], d["w"] = tuple(idx[o] for o in order), tuple(w[o] for o in order)
    if rng.random() < rate:
        d["smooth"] = rng.choice(SMOOTHS)
    if rng.random() < rate:
        d["entry"] = round(min(0.9, max(0.02, d["entry"] + rng.gauss(0, 0.08))), 3)
    if rng.random() < rate:
        d["exit"] = round(min(d["entry"], max(-0.5, d["exit"] + rng.gauss(0, 0.08))), 3)
    d["exit"] = min(d["exit"], d["entry"])
    if rng.random() < rate:
        d["stop_k"] = round(min(5.0, max(0.6, d["stop_k"] * rng.uniform(0.75, 1.3))), 2)
    if rng.random() < rate:
        d["tp_r"] = rng.choice(TPS)
    if rng.random() < rate:
        d["tstop"] = rng.choice(TSTOPS)
    if rng.random() < rate / 2:
        d["cooldown"] = rng.choice([0, 1, 2, 3, 6])
    if rng.random() < rate / 3:
        d["short"] = 1 - d["short"]
    if rng.random() < rate / 3:
        d["news"] = 1 - d["news"]
    return Genome(**d)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    da, db = asdict(a), asdict(b)
    votes = dict(zip(a.idx, a.w))
    for i, w in zip(b.idx, b.w):
        if i not in votes or rng.random() < 0.5:
            if i in votes or rng.random() < 0.5:
                votes[i] = w
    keys = sorted(votes)[:10]
    child = {k: (da if rng.random() < 0.5 else db)[k] for k in da if k not in ("idx", "w")}
    child["exit"] = min(child["exit"], child["entry"])
    return Genome(tuple(keys), tuple(votes[k] for k in keys), **child)


_G: dict = {}


def _init(cache: str) -> None:
    with open(cache, "rb") as fh:
        _G.update(pickle.load(fh))


def ga(args: tuple) -> list[tuple[float, Genome]]:
    """Un tournoi génétique (une île) ; retourne la population finale triée."""
    V_name, pool, eras, gens, pop_n, seed, seeds_in = args
    D, V = _G, _G[V_name]
    rng = random.Random(seed)
    pop = list(seeds_in) + [random_genome(rng, pool) for _ in range(pop_n - len(seeds_in))]
    cache: dict = {}

    def fit(g: Genome) -> float:
        if g not in cache:
            combo_ok = V_name != "C" or len(g.idx) >= COMBO_MIN
            cache[g] = era_fitness(D, V, g, eras) if combo_ok else -9.0
        return cache[g]

    for gen in range(gens):
        ranked = sorted(pop, key=fit, reverse=True)
        elite = ranked[:3]
        top = ranked[:20]
        children = []
        while len(children) < int(pop_n * 0.87) - 3:
            p1 = rng.choice(elite)
            p2 = rng.choice(top)
            c = crossover(p1, p2, rng) if rng.random() < 0.7 else p1
            children.append(mutate(c, rng, pool))
        immigrants = [random_genome(rng, pool) for _ in range(pop_n - 3 - len(children))]
        pop = elite + children + immigrants
    return sorted(((fit(g), g) for g in set(pop)), key=lambda x: -x[0])


# ---------------------------------------------------------------------------------------------
# Déroulé d'un pli


def index_range(D: dict, start: str, end: str) -> tuple[int, int]:
    t = D["time"]
    return int(np.searchsorted(t, np.datetime64(start))), int(np.searchsorted(t, np.datetime64(end)))


def eras_of(a: int, b: int, k: int = 3) -> list[tuple[int, int]]:
    step = (b - a) // k
    return [(a + i * step, a + (i + 1) * step) for i in range(k)]


def evaluate(D: dict, V: np.ndarray, g: Genome, a: int, b: int) -> dict:
    eq, tr = run(D, V, g, a, b)
    return metrics(eq, tr, D["day"][a:b])


def significantly_better(r1: np.ndarray, r2: np.ndarray) -> bool:
    """Test t apparié sur les rendements journaliers : r1 meilleur que r2 avec p < 0,1 (unilatéral)."""
    d = r1 - r2
    if d.std() == 0:
        return False
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    return t > 1.2816


def select(cands: list[Genome], D: dict, V: np.ndarray, tr: tuple[int, int], va: tuple[int, int], eras) -> list[Genome]:
    """Classement = rang entraînement + rang validation (fitness)."""
    ft = [era_fitness(D, V, g, eras) for g in cands]
    fv = [fitness_of(evaluate(D, V, g, *va)) for g in cands]
    rt = np.argsort(np.argsort(-np.array(ft)))
    rv = np.argsort(np.argsort(-np.array(fv)))
    order = np.argsort(rt + rv + 1e-3 * rt)
    return [cands[i] for i in order]


def champion_matrix(D: dict, champs: list[Genome]) -> np.ndarray:
    """Scores des champions de ligue, empilés en colonnes : ce sont les votants de la finale."""
    return np.column_stack([score_of(D["V"], g) for g in champs]).astype(np.float32)


def run_fold(cache: str, test_year: int, gens: int, pop: int, workers: int, seeds: int = 2) -> dict:
    _init(cache)
    D = _G
    end_data = str(D["time"][-1] + np.timedelta64(1, "s"))
    if test_year:
        train_end, test_end = f"{test_year}-01-01", f"{test_year + 1}-01-01"
    else:
        train_end, test_end = end_data, end_data
    a, b = index_range(D, "2010-01-01", train_end)
    val_start = str(np.datetime64(train_end[:10]) - np.timedelta64(365, "D"))
    v0 = index_range(D, val_start, train_end)[0]
    tr, va = (a, v0), (v0, b)
    te = index_range(D, train_end, test_end)
    eras = eras_of(*tr)
    fam = D["families"]
    leagues = {f: [i for i, x in enumerate(fam) if x == f] for f in sorted(set(fam))}
    leagues["libre"] = list(range(len(fam)))
    t0 = time.time()

    # Tour 1 : ligues par famille, plusieurs îles (graines) en parallèle
    jobs = [("V", pool, eras, gens, pop, test_year * 100 + s * 10 + li, ()) for li, (_, pool) in enumerate(leagues.items())
            for s in range(seeds)]
    with Pool(workers, initializer=_init, initargs=(cache,)) as p:
        res = p.map(ga, jobs)
    report: dict = {"test_year": test_year, "leagues": {}}
    champs: list[Genome] = []
    names: list[str] = []
    for li, (lg, _) in enumerate(leagues.items()):
        pop_all = sorted((x for s in range(seeds) for x in res[li * seeds + s]), key=lambda x: -x[0])
        uniq = list(dict.fromkeys(g for _, g in pop_all))[:12]
        ranked = select(uniq, D, D["V"], tr, va, eras)
        best = ranked[0]
        # Fusion : comité des 3 premiers si le champion ne domine pas significativement
        rv = [daily_returns(D, D["V"], g, *va) for g in ranked[:3]]
        fused = None
        if len(ranked) >= 3 and not any(significantly_better(rv[0], r) for r in rv[1:]):
            sub = ranked[:3]
            M = champion_matrix(D, sub)
            Dm = dict(D, V=M)
            fused_g = replace(best, idx=(0, 1, 2), w=(1.0, 1.0, 1.0), smooth=1)
            f_tr = era_fitness(Dm, M, fused_g, eras)
            f_va = fitness_of(evaluate(Dm, M, fused_g, *va))
            b_tr = era_fitness(D, D["V"], best, eras)
            b_va = fitness_of(evaluate(D, D["V"], best, *va))
            fused = {"members": [asdict(g) for g in sub], "rules": asdict(fused_g), "train": f_tr, "valid": f_va}
            if f_tr + f_va > b_tr + b_va:
                report["leagues"][lg] = {"type": "fusion", **fused}
                champs.append(("fusion", sub, fused_g))
                names.append(lg)
                continue
        report["leagues"][lg] = {"type": "seul", "genome": asdict(best), "fusion_testee": fused is not None,
                                 "valid": evaluate(D, D["V"], best, *va)}
        champs.append(("seul", [best], None))
        names.append(lg)

    # Matrice des scores des champions (un fusionné = moyenne de ses membres puis lissage de ses règles)
    cols = []
    for kind, members, rules in champs:
        if kind == "seul":
            cols.append(score_of(D["V"], members[0]))
        else:
            cols.append(champion_matrix(D, members).mean(axis=1))
    C = np.column_stack(cols).astype(np.float32)
    _G["C"] = C
    with open(cache + ".final", "wb") as fh:
        pickle.dump({**{k: v for k, v in D.items() if k != "C"}, "C": C}, fh)

    # Grande finale : pondérations entre champions + règles de trading
    singles = []
    for j, (kind, members, rules) in enumerate(champs):
        g0 = members[0] if kind == "seul" else rules
        singles.append(Genome((j,), (1.0,), 1, g0.entry, g0.exit, g0.stop_k, g0.tp_r, g0.tstop, g0.cooldown,
                              g0.short, g0.news))
    k = C.shape[1]
    combos = [replace(g, idx=tuple(range(k)), w=(1.0,) * k) for g in singles]  # comité équipondéré, règles de chaque champion
    jobs = [("C", list(range(k)), eras, gens, pop, test_year * 100 + 90 + s, tuple(combos)) for s in range(seeds)]
    with Pool(workers, initializer=_init, initargs=(cache + ".final",)) as p:
        fin = p.map(ga, jobs)
    finalists = list(dict.fromkeys([g for r in fin for _, g in sorted(r, key=lambda x: -x[0])[:10]] + combos))
    finalists = [g for g in finalists if len(g.idx) >= COMBO_MIN]
    ranked = select(finalists, D, C, tr, va, eras)
    winner = ranked[0]
    best_single = select(singles, D, C, tr, va, eras)[0]

    def describe(g: Genome) -> dict:
        tot = sum(g.w)
        return {names[i]: round(w / tot, 3) for i, w in zip(g.idx, g.w)}

    report["champions"] = names
    report["winner"] = {"genome": asdict(winner), "weights": describe(winner),
                        "train": evaluate(D, C, winner, *tr), "valid": evaluate(D, C, winner, *va)}
    report["best_single"] = {"champion": names[best_single.idx[0]], "valid": evaluate(D, C, best_single, *va)}
    if te[1] > te[0]:
        report["best_single"]["test"] = evaluate(D, C, best_single, *te)
        report["winner"]["test"] = evaluate(D, C, winner, *te)
        _, trades = run(D, C, winner, *te)
        report["test_trades"] = trades[:, [3, 4]].tolist()  # [R, rendement] de chaque trade du test
        report["test_daily"] = daily_returns(D, C, winner, *te).tolist()
    report["minutes"] = round((time.time() - t0) / 60, 1)
    report["n_bots_tested"] = (len(leagues) * seeds + seeds) * gens * pop
    report["voters"] = D["names"]
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("test_year", type=int)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=400)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    rep = run_fold(a.cache, a.test_year, a.gens, a.pop, a.workers, a.seeds)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    w = rep["winner"]
    print(json.dumps({"poids": w["weights"], "valid": w["valid"], "test": w.get("test")}, indent=1, default=float))


if __name__ == "__main__":
    main()
