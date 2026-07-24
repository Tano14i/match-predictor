"""
backtest_real_xg.py — backtest walk-forward con xG REALE (data provider via
api-football, non piu' un proxy) al posto sia del proxy tiri-in-porta che
del target "gol nudi" del modello attacco/difesa. Stessa disciplina di
sempre: nessun leakage (solo storia precedente al match), stesso motore
EV/quote/Dixon-Coles di backtest_football.py — unica variabile cambiata e'
la fonte dell'xG usato per stimare lambda/mu pre-match.

Modello: media mobile delle ultime N partite di xG REALE per squadra, con
split casa/trasferta quando il campione lo permette — stessa architettura
di team_xg() in backtest_football.py, sorgente diversa (xG vero invece di
tiri in porta * 0.33).

LIMITE STRUTTURALE (non aggirabile): l'xG reale non e' disponibile
retroattivamente da api-football — copertura da meta' stagione 2022-23
(PL/Serie A) o da 2023-24 (Eredivisie). Il campione e' quindi molto piu'
piccolo di quello usato per gli altri modelli (6 stagioni PL li' vs ~3.5
qui): i risultati vanno letti con piu' cautela statistica, non meno.

Uso:
    python backtest_real_xg.py --league premier_league \
      --xg xg_data/PremierLeague_2022.json xg_data/PremierLeague_2023.json xg_data/PremierLeague_2024.json xg_data/PremierLeague_2025.json \
      --odds E0_2022-23.csv E0_2023-24.csv E0_2024-25.csv E0_2025-26.csv \
      --label "Premier League (xG reale)"
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque

from backtest_football import (
    DC_RHO,
    MIN_EDGE,
    evaluate_value,
    prob_over_under_goals,
    probs_1x2_raw,
    summarize,
    to_float,
)
from real_xg_data import load_xg_json, merge_xg_with_odds

LAST_N = 5
MIN_SPLIT = 2


def new_state() -> dict:
    return {"matches": deque(maxlen=LAST_N)}


def team_xg_real(state: dict, is_home: bool) -> float | None:
    matches = list(state["matches"])
    if len(matches) < LAST_N:
        return None
    same_ctx = [m for m in matches if m["is_home"] == is_home]
    if len(same_ctx) >= MIN_SPLIT:
        return max(0.05, sum(m["xg"] for m in same_ctx) / len(same_ctx))
    return max(0.05, sum(m["xg"] for m in matches) / len(matches))


def update_state(state: dict, xg: float, is_home: bool) -> None:
    state["matches"].append({"xg": xg, "is_home": is_home})


def run_backtest_real_xg(merged_rows: list[dict], min_edge: float = MIN_EDGE, odds_source: str = "Avg", rho: float = DC_RHO) -> list[dict]:
    hcol, dcol, acol = f"{odds_source}H", f"{odds_source}D", f"{odds_source}A"
    ocol, ucol = f"{odds_source}>2.5", f"{odds_source}<2.5"

    state: dict[str, dict] = defaultdict(new_state)
    bets = []

    for r in sorted(merged_rows, key=lambda m: m["date"]):
        home, away = r["home"], r["away"]
        hs, as_ = state[home], state[away]
        xg_h_pred, xg_a_pred = team_xg_real(hs, True), team_xg_real(as_, False)

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
            year = r["date"].year

            if pick_1x2:
                won = pick_1x2["label"] == real_result
                profit = (pick_1x2["odds"] - 1) if won else -1
                bets.append({"market": "1x2", "year": year, "won": won, "profit": profit, "edge": pick_1x2["edge"]})
            if pick_goals:
                won = (pick_goals["label"] == "Over" and real_total > 2.5) or (pick_goals["label"] == "Under" and real_total < 2.5)
                profit = (pick_goals["odds"] - 1) if won else -1
                bets.append({"market": "goals", "year": year, "won": won, "profit": profit, "edge": pick_goals["edge"]})

        update_state(hs, r["xg_home"], True)
        update_state(as_, r["xg_away"], False)

    return bets


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest walk-forward con xG reale (api-football) invece del proxy.")
    ap.add_argument("--league", required=True, choices=["premier_league", "serie_a", "eredivisie"])
    ap.add_argument("--xg", nargs="+", required=True, help="File xg_*.json in ordine cronologico")
    ap.add_argument("--odds", nargs="+", required=True, help="File CSV football-data.co.uk corrispondenti (stesso ordine)")
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
    print(f"Match totali ({label}): {len(merged)} (con xG reale + quote unite, {unmatched} scartati per mancanza quote)")
    print(f"Fonte quote: {args.odds_source} | Margine minimo: {args.min_edge*100:.0f}% | rho: {args.rho}")

    bets = run_backtest_real_xg(merged, args.min_edge, args.odds_source, args.rho)
    result = summarize(bets)

    print(f"\n── RISULTATO COMPLESSIVO ({label}) — xG reale ──")
    o = result["overall"]
    print(f"  bet: {o['bets']} | hit rate: {o['hit_rate']}% | ROI: {o['roi_pct']}% | profitto: {o['profit_units']:+.2f}u")
    print("\n── Per mercato ──")
    for label_m, key in (("1X2", "1x2"), ("Over/Under 2.5 gol", "goals")):
        r = result[key]
        if r["bets"]:
            print(f"  {label_m}: bet {r['bets']} | hit rate {r['hit_rate']}% | ROI {r['roi_pct']}% | profitto {r['profit_units']:+.2f}u")
        else:
            print(f"  {label_m}: nessuna bet")
    print("\n── Per anno ──")
    for y, r in result["by_year"].items():
        if r["bets"]:
            print(f"  {y}: bet {r['bets']} | hit rate {r['hit_rate']}% | ROI {r['roi_pct']}% | profitto {r['profit_units']:+.2f}u")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
