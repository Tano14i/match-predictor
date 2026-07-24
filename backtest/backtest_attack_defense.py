"""
backtest_attack_defense.py — stesso backtest walk-forward di
backtest_football.py (stesse quote, stesso EV, stesso MIN_EDGE, stesso
Dixon-Coles tau/rho per i punteggi bassi), ma con l'xG stimato dal modello
attacco/difesa (dixon_coles_model.py) invece del proxy "media tiri in
porta ultime 5 partite". Unica variabile cambiata, per un confronto
onesto mela-con-mela contro i risultati gia' ottenuti col modello attuale.

Walk-forward: rifit del modello attacco/difesa una volta per ogni nuova
data di match incontrata (usando solo risultati precedenti — nessun
leakage), poi le stesse identiche funzioni di prob/EV del resto del
progetto.

Uso:
    python backtest_attack_defense.py --files E0_2020-21.csv ... E0_2025-26.csv --label "Premier League"
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime

from backtest_football import (
    DC_RHO,
    MIN_EDGE,
    evaluate_value,
    parse_date,
    prob_over_under_goals,
    probs_1x2_raw,
    summarize,
    to_float,
    to_int,
)
from dixon_coles_model import MIN_HISTORY_MATCHES, fit_attack_defense


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8-sig") as fh:
            file_rows = list(csv.DictReader(fh))
        file_rows = [r for r in file_rows if r.get("Date")]
        file_rows.sort(key=lambda r: parse_date(r["Date"]) or datetime.min)
        rows.extend(file_rows)
    return rows


def run_backtest_ad(rows: list[dict], min_edge: float = MIN_EDGE, odds_source: str = "Avg", rho: float = DC_RHO, alpha: float = 1.0) -> list[dict]:
    """alpha: peso del gol REALE nel target di training del fit attacco/difesa
    (1.0 = solo gol, comportamento originale; 0.0 = solo proxy tiri in porta;
    valori intermedi = media pesata). Il grading delle bet (vinta/persa) usa
    SEMPRE il risultato reale — alpha cambia solo cosa il modello impara
    dalla storia, non come le bet vengono giudicate."""
    hcol, dcol, acol = f"{odds_source}H", f"{odds_source}D", f"{odds_source}A"
    ocol, ucol = f"{odds_source}>2.5", f"{odds_source}<2.5"

    history: list[tuple] = []
    current_fit = None
    fit_date = None
    bets = []

    for r in rows:
        home, away = r.get("HomeTeam"), r.get("AwayTeam")
        if not home or not away:
            continue
        fthg, ftag = to_int(r.get("FTHG")), to_int(r.get("FTAG"))
        if fthg is None or ftag is None:
            continue
        date = parse_date(r.get("Date"))
        if date is None:
            continue

        if date != fit_date:
            current_fit = fit_attack_defense(history, date, prev_fit=current_fit)
            fit_date = date

        if current_fit is not None:
            xg_h, xg_a = current_fit.expected_goals(home, away)
            if xg_h is not None and xg_a is not None:
                w, d, l = probs_1x2_raw(xg_h, xg_a, rho)
                over, under = prob_over_under_goals(xg_h, xg_a, 2.5, rho)

                odds_1x2 = [to_float(r.get(hcol)), to_float(r.get(dcol)), to_float(r.get(acol))]
                odds_goals = [to_float(r.get(ocol)), to_float(r.get(ucol))]

                pick_1x2 = evaluate_value([w, d, l], odds_1x2, ["H", "D", "A"], min_edge)
                pick_goals = evaluate_value([over, under], odds_goals, ["Over", "Under"], min_edge)

                real_result = "H" if fthg > ftag else ("A" if ftag > fthg else "D")
                real_total = fthg + ftag
                season_year = date.year

                if pick_1x2:
                    won = pick_1x2["label"] == real_result
                    profit = (pick_1x2["odds"] - 1) if won else -1
                    bets.append({"market": "1x2", "year": season_year, "won": won, "profit": profit, "edge": pick_1x2["edge"]})
                if pick_goals:
                    won = (pick_goals["label"] == "Over" and real_total > 2.5) or (pick_goals["label"] == "Under" and real_total < 2.5)
                    profit = (pick_goals["odds"] - 1) if won else -1
                    bets.append({"market": "goals", "year": season_year, "won": won, "profit": profit, "edge": pick_goals["edge"]})

        if alpha < 1.0:
            hst, ast = to_int(r.get("HST")), to_int(r.get("AST"))
            target_h = alpha * fthg + (1 - alpha) * hst * 0.33 if hst is not None else fthg
            target_a = alpha * ftag + (1 - alpha) * ast * 0.33 if ast is not None else ftag
        else:
            target_h, target_a = fthg, ftag
        history.append((date, home, away, target_h, target_a))

    return bets


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest walk-forward del modello attacco/difesa Dixon-Coles (MLE) al posto del proxy xG attuale.")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE)
    ap.add_argument("--odds-source", default="Avg", choices=["Avg", "B365", "Max", "PS"])
    ap.add_argument("--rho", type=float, default=DC_RHO)
    ap.add_argument("--alpha", type=float, default=1.0, help="Peso del gol reale nel target di training (1.0=solo gol, 0.0=solo proxy tiri in porta)")
    args = ap.parse_args()

    rows = load_rows(args.files)
    label = args.label or " + ".join(args.files)
    print(f"Match totali ({label}): {len(rows)}")
    print(f"Fonte quote: {args.odds_source} | Margine minimo: {args.min_edge*100:.0f}% | rho: {args.rho} | alpha: {args.alpha} | soglia storia minima: {MIN_HISTORY_MATCHES}")

    bets = run_backtest_ad(rows, args.min_edge, args.odds_source, args.rho, args.alpha)
    result = summarize(bets)

    print(f"\n── RISULTATO COMPLESSIVO ({label}) — modello attacco/difesa ──")
    o = result["overall"]
    print(f"  bet: {o['bets']} | hit rate: {o['hit_rate']}% | ROI: {o['roi_pct']}% | profitto: {o['profit_units']:+.2f}u")
    print("\n── Per mercato ──")
    for label_m, key in (("1X2", "1x2"), ("Over/Under 2.5 gol", "goals")):
        r = result[key]
        if r["bets"]:
            print(f"  {label_m}: bet {r['bets']} | hit rate {r['hit_rate']}% | ROI {r['roi_pct']}% | profitto {r['profit_units']:+.2f}u")
        else:
            print(f"  {label_m}: nessuna bet")
    print("\n── Per stagione (anno) ──")
    for y, r in result["by_year"].items():
        print(f"  {y}: bet {r['bets']} | hit rate {r['hit_rate']}% | ROI {r['roi_pct']}% | profitto {r['profit_units']:+.2f}u")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
