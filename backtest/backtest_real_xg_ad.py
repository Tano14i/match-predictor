"""
backtest_real_xg_ad.py — come backtest_real_xg.py, ma usa l'xG reale come
TARGET DI TRAINING del modello attacco/difesa (dixon_coles_model.py) invece
che come input diretto di una media mobile. Isola la domanda "l'xG reale
aiuta con un'architettura migliore?" da "l'xG reale aiuta con la media
mobile debole?" (risposta gia' negativa in backtest_real_xg.py).
"""
from __future__ import annotations

import argparse

from backtest_football import (
    DC_RHO,
    MIN_EDGE,
    evaluate_value,
    prob_over_under_goals,
    probs_1x2_raw,
    summarize,
    to_float,
)
from dixon_coles_model import fit_attack_defense
from real_xg_data import load_xg_json, merge_xg_with_odds


def run_backtest_real_xg_ad(merged_rows: list[dict], min_edge: float = MIN_EDGE, odds_source: str = "Avg", rho: float = DC_RHO) -> list[dict]:
    hcol, dcol, acol = f"{odds_source}H", f"{odds_source}D", f"{odds_source}A"
    ocol, ucol = f"{odds_source}>2.5", f"{odds_source}<2.5"

    rows_sorted = sorted(merged_rows, key=lambda m: m["date"])
    history: list[tuple] = []  # (date, home, away, xg_home, xg_away) - target di training = xG reale
    current_fit = None
    fit_date = None
    bets = []

    for r in rows_sorted:
        home, away = r["home"], r["away"]
        date = r["date"]

        if date != fit_date:
            current_fit = fit_attack_defense(history, date, prev_fit=current_fit)
            fit_date = date

        if current_fit is not None:
            xg_h_pred, xg_a_pred = current_fit.expected_goals(home, away)
            if xg_h_pred is not None and xg_a_pred is not None:
                w, d, l = probs_1x2_raw(xg_h_pred, xg_a_pred, rho)
                over, under = prob_over_under_goals(xg_h_pred, xg_a_pred, 2.5, rho)

                odds = r["odds"]
                odds_1x2 = [to_float(odds.get(hcol)), to_float(odds.get(dcol)), to_float(odds.get(acol))]
                odds_goals = [to_float(odds.get(ocol)), to_float(odds.get(ucol))]

                pick_1x2 = evaluate_value([w, d, l], odds_1x2, ["H", "D", "A"], min_edge)
                pick_goals = evaluate_value([over, under], odds_goals, ["Over", "Under"], min_edge)

                real_result = "H" if r["fthg"] > r["ftag"] else ("A" if r["ftag"] > r["fthg"] else "D")
                real_total = r["fthg"] + r["ftag"]
                year = date.year

                if pick_1x2:
                    won = pick_1x2["label"] == real_result
                    profit = (pick_1x2["odds"] - 1) if won else -1
                    bets.append({"market": "1x2", "year": year, "won": won, "profit": profit, "edge": pick_1x2["edge"]})
                if pick_goals:
                    won = (pick_goals["label"] == "Over" and real_total > 2.5) or (pick_goals["label"] == "Under" and real_total < 2.5)
                    profit = (pick_goals["odds"] - 1) if won else -1
                    bets.append({"market": "goals", "year": year, "won": won, "profit": profit, "edge": pick_goals["edge"]})

        history.append((date, home, away, r["xg_home"], r["xg_away"]))

    return bets


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest walk-forward: attacco/difesa allenato su xG reale invece che sui gol.")
    ap.add_argument("--league", required=True, choices=["premier_league", "serie_a", "eredivisie"])
    ap.add_argument("--xg", nargs="+", required=True)
    ap.add_argument("--odds", nargs="+", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE)
    ap.add_argument("--odds-source", default="Avg", choices=["Avg", "B365", "Max", "PS"])
    ap.add_argument("--rho", type=float, default=DC_RHO)
    args = ap.parse_args()

    all_xg_rows = []
    for path in args.xg:
        all_xg_rows.extend(load_xg_json(path, args.league))
    merged, unmatched = merge_xg_with_odds(all_xg_rows, args.odds)

    label = args.label or args.league
    print(f"Match totali ({label}): {len(merged)} (con xG reale + quote unite, {unmatched} scartati)")
    print(f"Fonte quote: {args.odds_source} | Margine minimo: {args.min_edge*100:.0f}% | rho: {args.rho}")

    bets = run_backtest_real_xg_ad(merged, args.min_edge, args.odds_source, args.rho)
    result = summarize(bets)

    print(f"\n── RISULTATO COMPLESSIVO ({label}) — attacco/difesa su xG reale ──")
    o = result["overall"]
    print(f"  bet: {o['bets']} | hit rate: {o['hit_rate']}% | ROI: {o['roi_pct']}% | profitto: {o['profit_units']:+.2f}u")
    print("\n── Per mercato ──")
    for label_m, key in (("1X2", "1x2"), ("Over/Under 2.5 gol", "goals")):
        r = result[key]
        if r["bets"]:
            print(f"  {label_m}: bet {r['bets']} | hit rate {r['hit_rate']}% | ROI {r['roi_pct']}% | profitto {r['profit_units']:+.2f}u")
        else:
            print(f"  {label_m}: nessuna bet")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
