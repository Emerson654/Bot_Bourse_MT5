"""Fusion du tournoi 6 : portefeuille des spécialistes (un par actif) sur les années jamais vues, et verdict.

Un spécialiste n'entre dans le portefeuille d'une année que si son entraînement (avant cette année) était
rentable : décision prise sans regarder l'année testée.
Usage : python -m research.summarize6 DOSSIER_PLIS results/
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from research.summarize import CRITERIA


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    folds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(src, "*.json")))]
    years = sorted({f["test_year"] for f in folds if f["test_year"]})
    daily: dict[int, float] = {}
    rets, per_year, per_asset = [], {}, {}
    for y in years:
        yd: dict[int, float] = {}
        for f in (f for f in folds if f["test_year"] == y):
            if f["winner"]["train"]["sharpe"] <= 0:
                continue  # spécialiste perdant à l'entraînement : écarté cette année-là
            for d, r in zip(f.get("test_days", []), f.get("test_daily", [])):
                yd[d] = yd.get(d, 0.0) + r
            tr = [r for _, r in f.get("test_trades", [])]
            rets += tr
            pa = per_asset.setdefault(f["asset"], {"trades": 0, "gain": 0.0})
            pa["trades"] += len(tr)
            pa["gain"] += float(sum(tr))
        r = np.array([yd[d] for d in sorted(yd)])
        per_year[y] = {"ret": float(np.prod(1 + r) - 1) if len(r) else 0.0, "jours": len(r)}
        daily.update(yd)
    r = np.array([daily[d] for d in sorted(daily)])
    eq = np.cumprod(1 + r) if len(r) else np.ones(1)
    rets = np.array(rets)
    gains, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()
    oos = {"trades": int(len(rets)), "pf": float(gains / losses) if losses > 0 else 0.0,
           "max_dd": float((eq / np.maximum.accumulate(eq) - 1).min()),
           "sharpe": float(r.mean() / r.std() * np.sqrt(260)) if len(r) and r.std() > 0 else 0.0,
           "cagr": float(eq[-1] ** (260 / max(len(r), 1)) - 1), "par_annee": per_year, "par_actif": per_asset}
    ok = oos["pf"] > CRITERIA["pf_min"] and -oos["max_dd"] < CRITERIA["max_dd_max"] and oos["trades"] >= CRITERIA["trades_min"]
    prod = {f["asset"]: f for f in folds if not f["test_year"] and f["winner"]["train"]["sharpe"] > 0}
    os.makedirs(out, exist_ok=True)
    json.dump({"validated": bool(ok), "criteria": CRITERIA, "out_of_sample": oos,
               "production_assets": sorted(prod)}, open(os.path.join(out, "backtest.json"), "w"), indent=1, default=float)
    json.dump(prod, open(os.path.join(out, "bot_params.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in oos.items()}, indent=1, default=float))
    print("VALIDÉ" if ok else "NON VALIDÉ")


if __name__ == "__main__":
    main()
