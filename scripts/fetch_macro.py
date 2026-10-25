"""Collecte des données « actu et macro » quotidiennes, avec historique (indispensable pour le backtest).

Trois outils :
1. Risque géopolitique (GPR) — Caldara & Iacoviello : comptage quotidien d'articles de presse sur les
   tensions, guerres, terrorisme (journaux US/UK/CA). https://www.matteoiacoviello.com/gpr.htm
2. Incertitude économique (EPU) — Baker, Bloom & Davis : comptage quotidien d'articles sur l'incertitude
   de politique économique. https://www.policyuncertainty.com
3. Macro de marché (FRED, Réserve fédérale de St-Louis) : taux réel 10 ans, taux 10 ans et 2 ans,
   inflation anticipée, dollar, VIX (peur sur les actions), volatilité de l'or, pétrole.

Sortie : data/macro/<nom>.csv (date, valeur). Usage : python scripts/fetch_macro.py data/macro
"""

from __future__ import annotations

import csv
import io
import os
import sys
import urllib.request

FRED = {
    "taux_reel_10a": "DFII10", "taux_10a": "DGS10", "taux_2a": "DGS2", "inflation_anticipee": "T10YIE",
    "dollar": "DTWEXBGS", "vix": "VIXCLS", "vol_or": "GVZCLS", "petrole": "DCOILWTICO",
}
GPR = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
EPU = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
UA = {"User-Agent": "Mozilla/5.0 (bot de recherche)"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read()


def save(out_dir: str, name: str, rows: list[tuple[str, float]]) -> None:
    with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "value"])
        w.writerows(rows)
    print(f"{name} : {len(rows)} jours, {rows[0][0]} → {rows[-1][0]}")


def main() -> None:
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, sid in FRED.items():
        txt = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}").decode()
        rows = [(r[0], float(r[1])) for r in list(csv.reader(io.StringIO(txt)))[1:] if r[1] not in (".", "")]
        save(out, name, rows)
    txt = get(EPU).decode()
    rd = list(csv.DictReader(io.StringIO(txt)))
    save(out, "incertitude_eco", [(f"{int(r['year']):04d}-{int(r['month']):02d}-{int(r['day']):02d}",
                                    float(r["daily_policy_index"])) for r in rd if r.get("daily_policy_index")])
    import xlrd  # le fichier GPR est un .xls

    book = xlrd.open_workbook(file_contents=get(GPR))
    sh = book.sheet_by_index(0)
    head = [str(c.value).strip() for c in sh.row(0)]
    print("colonnes GPR :", head[:12])
    i_date = next(i for i, h in enumerate(head) if h.lower() in ("date", "day"))
    i_gpr = head.index("GPRD") if "GPRD" in head else next(i for i, h in enumerate(head) if h.upper().startswith("GPR"))
    rows = []
    for k in range(1, sh.nrows):
        d, v = sh.cell_value(k, i_date), sh.cell_value(k, i_gpr)
        if v in ("", None) or d in ("", None):
            continue
        txt = str(d).strip()
        if (isinstance(d, float) and d > 19_000_000) or (txt.replace(".0", "").isdigit() and len(txt.replace(".0", "")) == 8):
            n = int(float(d))  # aaaammjj
            y, m, dd = n // 10000, n // 100 % 100, n % 100
        elif isinstance(d, float):  # date Excel
            y, m, dd, *_ = xlrd.xldate_as_tuple(d, book.datemode)
        else:
            y, m, dd = map(int, str(d)[:10].split("-"))
        rows.append((f"{y:04d}-{m:02d}-{dd:02d}", float(v)))
    save(out, "risque_geopolitique", rows)


if __name__ == "__main__":
    main()
