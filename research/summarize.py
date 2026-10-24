"""Fusionne les plis du tournoi : performance hors échantillon de la méthode et verdict de validation.

Critères (bot live autorisé seulement si tous passent, sur les années de test jamais vues) :
profit factor > 1,3 ; drawdown max < 15 % ; au moins 100 trades.
Usage : python -m research.summarize DOSSIER_PLIS results/
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

CRITERIA = {"pf_min": 1.3, "max_dd_max": 0.15, "trades_min": 100}


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    folds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(src, "fold_*.json")))]
    wfo = [f for f in folds if f["test_year"]]
    prod = next((f for f in folds if not f["test_year"]), None)
    rets = np.array([r for f in wfo for _, r in f.get("test_trades", [])])
    daily = np.array([d for f in wfo for d in f.get("test_daily", [])])
    eq = np.cumprod(1 + daily) if len(daily) else np.ones(1)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    gains, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    pf = float(gains / losses) if losses > 0 else 0.0
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    years = len(daily) / 252 if len(daily) else 1
    oos = {"trades": len(rets), "pf": pf, "max_dd": dd, "sharpe": sharpe,
           "cagr": float(eq[-1] ** (1 / years) - 1), "tpd": len(rets) / max(len(daily), 1),
           "win": float((rets > 0).mean()) if len(rets) else 0.0,
           "par_annee": {f["test_year"]: f["winner"]["test"] for f in wfo}}
    ok = pf > CRITERIA["pf_min"] and -dd < CRITERIA["max_dd_max"] and len(rets) >= CRITERIA["trades_min"]
    os.makedirs(out, exist_ok=True)
    verdict = {"validated": bool(ok), "criteria": CRITERIA, "out_of_sample": oos,
               "production": prod["winner"] if prod else None}
    json.dump(verdict, open(os.path.join(out, "backtest.json"), "w"), indent=1, default=float)
    if prod:
        json.dump(prod, open(os.path.join(out, "bot_params.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in oos.items() if k != "par_annee"}, indent=1))
    print("VALIDÉ" if ok else "NON VALIDÉ")


if __name__ == "__main__":
    main()
