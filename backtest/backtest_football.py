"""
backtest_football.py — backtest offline del modello di match-predictor-5.html
contro risultati e quote storiche reali (football-data.co.uk).

Stessa disciplina di tennisanalytics/backtest.py: cammina i match in ordine
cronologico, ricostruisce lo stato PRE-match di ogni squadra (ultime 5
partite con dati tiri, xG proxy = tiri in porta * 0.33, split casa/trasferta
quando il campione lo permette — stesse regole di buildTeamProfile nel tool
live), applica la STESSA matematica Poisson + Dixon-Coles del tool
(porting diretto, stessi coefficienti), confronta con le quote REALI del
giorno, e misura ROI/hit-rate REALIZZATI (non solo accuracy).

LIMITI ONESTI:
- Solo 1X2 e Over/Under 2.5 gol sono backtestabili: football-data.co.uk ha i
  corner REALI (per controllare se la distribuzione Poisson ipotizzata nel
  tool e' ragionevole) ma NON le quote storiche sui corner — quel mercato
  del tool resta quindi senza validazione di profittabilita' reale.
- Nessun Bet365/Sisal/Toto specifico come nel tool live: questo dataset non
  traccia bookmaker italiani. Uso 'Avg' (media di mercato, la stima piu'
  robusta) come riferimento di default; --odds-source permette di provare
  anche B365 o Max (il prezzo migliore tra tutti quelli tracciati dal sito).
- Riposo/calendario e forza avversari (aggiunti nel tool live dopo) NON sono
  replicati qui: richiederebbero classifica storica giorno-per-giorno che
  questi file non forniscono in un formato utilizzabile per un walk-forward
  pulito. Questo backtest valida il NUCLEO del modello (xG da tiri in porta
  + Poisson/Dixon-Coles + EV), non gli ultimi due aggiustamenti.

Uso:
    python backtest_football.py --files E0.csv --label "Premier League"
    python backtest_football.py --files N1_2020-21.csv N1_2021-22.csv N1_2022-23.csv N1_2023-24.csv --label "Eredivisie"
    python backtest_football.py --files E0.csv --odds-source B365
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from datetime import datetime

# ─── stessa matematica del tool live (match-predictor-5.html) ──────────────
DC_RHO = -0.1
MIN_EDGE = 0.05
LAST_N = 5
MIN_SPLIT = 2


def poisson_p(lam: float, k: int) -> float:
    r = math.exp(-lam)
    for i in range(k):
        r *= lam / (i + 1)
    return r


def dixon_coles_tau(x: int, y: int, lam: float, mu: float, rho: float = DC_RHO) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_prob(xg_h: float, xg_a: float, h: int, a: int) -> float:
    return poisson_p(xg_h, h) * poisson_p(xg_a, a) * dixon_coles_tau(h, a, xg_h, xg_a)


def probs_1x2_raw(xg_h: float, xg_a: float) -> tuple[float, float, float]:
    w = d = l = 0.0
    for h in range(11):
        for a in range(11):
            p = score_prob(xg_h, xg_a, h, a)
            if h > a:
                w += p
            elif h == a:
                d += p
            else:
                l += p
    t = w + d + l
    return w / t, d / t, l / t


def prob_over_under_goals(xg_h: float, xg_a: float, line: float) -> tuple[float, float]:
    max_under = math.floor(line)
    under = over = 0.0
    for h in range(11):
        for a in range(11):
            p = score_prob(xg_h, xg_a, h, a)
            if h + a <= max_under:
                under += p
            else:
                over += p
    t = under + over
    return over / t, under / t


def implied_probs(odds: list[float | None]) -> list[float | None]:
    raw = [1 / o if o and o > 1 else 0 for o in odds]
    total = sum(raw)
    if total <= 0:
        return [None] * len(odds)
    return [r / total for r in raw]


def expected_value(prob: float, odds: float | None) -> float | None:
    if not odds or odds <= 1:
        return None
    return prob * odds - 1


def evaluate_value(probs: list[float], odds: list[float | None], labels: list[str], min_edge: float = MIN_EDGE) -> dict | None:
    if not any(o and o > 1 for o in odds):
        return None
    evs = [expected_value(p, o) for p, o in zip(probs, odds)]
    best_i, best_edge = None, None
    for i, e in enumerate(evs):
        if e is not None and (best_edge is None or e > best_edge):
            best_i, best_edge = i, e
    if best_i is not None and best_edge >= min_edge:
        return {"label": labels[best_i], "edge": best_edge, "odds": odds[best_i]}
    return None


# ─── stato squadra walk-forward (stesse regole di buildTeamProfile) ────────
def new_state() -> dict:
    return {"matches": deque(maxlen=LAST_N)}


def team_xg(state: dict, is_home: bool) -> float | None:
    matches = list(state["matches"])
    if len(matches) < LAST_N:
        return None  # non abbastanza storia (stesso "ultimi 5" del tool live)
    same_ctx = [m for m in matches if m["is_home"] == is_home]
    if len(same_ctx) >= MIN_SPLIT:
        avg_shots_on = sum(m["shots_on"] for m in same_ctx) / len(same_ctx)
    else:
        avg_shots_on = sum(m["shots_on"] for m in matches) / len(matches)
    return max(0.05, avg_shots_on * 0.33)


def update_state(state: dict, shots_on: int, is_home: bool) -> None:
    state["matches"].append({"shots_on": shots_on, "is_home": is_home})


def to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def parse_date(s: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


# ─── backtest walk-forward ──────────────────────────────────────────────────
def run_backtest(rows: list[dict], min_edge: float = MIN_EDGE, odds_source: str = "Avg") -> list[dict]:
    state: dict[str, dict] = defaultdict(new_state)
    bets = []

    hcol, dcol, acol = f"{odds_source}H", f"{odds_source}D", f"{odds_source}A"
    ocol, ucol = f"{odds_source}>2.5", f"{odds_source}<2.5"

    for r in rows:
        home, away = r.get("HomeTeam"), r.get("AwayTeam")
        if not home or not away:
            continue
        fthg, ftag = to_int(r.get("FTHG")), to_int(r.get("FTAG"))
        hst, ast = to_int(r.get("HST")), to_int(r.get("AST"))
        if fthg is None or ftag is None:
            continue

        hs, as_ = state[home], state[away]
        xg_h, xg_a = team_xg(hs, True), team_xg(as_, False)

        if xg_h is not None and xg_a is not None:
            w, d, l = probs_1x2_raw(xg_h, xg_a)
            over, under = prob_over_under_goals(xg_h, xg_a, 2.5)

            odds_1x2 = [to_float(r.get(hcol)), to_float(r.get(dcol)), to_float(r.get(acol))]
            odds_goals = [to_float(r.get(ocol)), to_float(r.get(ucol))]

            pick_1x2 = evaluate_value([w, d, l], odds_1x2, ["H", "D", "A"], min_edge)
            pick_goals = evaluate_value([over, under], odds_goals, ["Over", "Under"], min_edge)

            real_result = "H" if fthg > ftag else ("A" if ftag > fthg else "D")
            real_total = fthg + ftag
            date = r.get("Date")
            season_year = (parse_date(date) or datetime.min).year

            if pick_1x2:
                won = pick_1x2["label"] == real_result
                profit = (pick_1x2["odds"] - 1) if won else -1
                bets.append({"market": "1x2", "year": season_year, "won": won, "profit": profit, "edge": pick_1x2["edge"]})
            if pick_goals:
                won = (pick_goals["label"] == "Over" and real_total > 2.5) or (pick_goals["label"] == "Under" and real_total < 2.5)
                profit = (pick_goals["odds"] - 1) if won else -1
                bets.append({"market": "goals", "year": season_year, "won": won, "profit": profit, "edge": pick_goals["edge"]})

        if hst is not None:
            update_state(hs, hst, True)
        if ast is not None:
            update_state(as_, ast, False)

    return bets


def summarize(bets: list[dict]) -> dict:
    def agg(subset):
        n = len(subset)
        if n == 0:
            return {"bets": 0, "hit_rate": None, "roi_pct": None, "profit_units": 0.0}
        wins = sum(1 for b in subset if b["won"])
        profit = sum(b["profit"] for b in subset)
        return {"bets": n, "hit_rate": round(wins / n * 100, 1), "roi_pct": round(profit / n * 100, 1), "profit_units": round(profit, 2)}

    by_year = {}
    for y in sorted({b["year"] for b in bets}):
        by_year[y] = agg([b for b in bets if b["year"] == y])

    return {
        "overall": agg(bets),
        "1x2": agg([b for b in bets if b["market"] == "1x2"]),
        "goals": agg([b for b in bets if b["market"] == "goals"]),
        "by_year": by_year,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest del modello match-predictor-5.html contro risultati e quote storiche reali (football-data.co.uk).")
    ap.add_argument("--files", nargs="+", required=True, help="File CSV football-data.co.uk, in ordine cronologico se piu' stagioni della stessa lega")
    ap.add_argument("--label", default=None, help="Etichetta per il report (es. 'Eredivisie')")
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE, help="Margine EV minimo per piazzare una bet (default 0.05)")
    ap.add_argument("--odds-source", default="Avg", choices=["Avg", "B365", "Max", "PS"], help="Colonna quote da usare (default Avg = media di mercato)")
    args = ap.parse_args()

    rows: list[dict] = []
    for path in args.files:
        with open(path, encoding="utf-8-sig") as fh:
            file_rows = list(csv.DictReader(fh))
        file_rows = [r for r in file_rows if r.get("Date")]
        file_rows.sort(key=lambda r: parse_date(r["Date"]) or datetime.min)
        rows.extend(file_rows)
        print(f"[ok] {path}: {len(file_rows)} match")

    label = args.label or " + ".join(args.files)
    print(f"\nMatch totali ({label}): {len(rows)}")
    print(f"Fonte quote: {args.odds_source} | Margine minimo: {args.min_edge*100:.0f}%")

    bets = run_backtest(rows, args.min_edge, args.odds_source)
    result = summarize(bets)

    print(f"\n── RISULTATO COMPLESSIVO ({label}) ──")
    o = result["overall"]
    print(f"  bet: {o['bets']} | hit rate: {o['hit_rate']}% | ROI: {o['roi_pct']}% | profitto: {o['profit_units']:+.2f}u")
    print("\n── Per mercato ──")
    for label_m, key in (("1X2", "1x2"), ("Over/Under 2.5 gol", "goals")):
        r = result[key]
        if r["bets"]:
            print(f"  {label_m}: bet {r['bets']} | hit rate {r['hit_rate']}% | ROI {r['roi_pct']}% | profitto {r['profit_units']:+.2f}u")
        else:
            print(f"  {label_m}: nessuna bet")
    print("\n── Per stagione (anno di inizio) ──")
    for y, r in result["by_year"].items():
        print(f"  {y}: bet {r['bets']} | hit rate {r['hit_rate']}% | ROI {r['roi_pct']}% | profitto {r['profit_units']:+.2f}u")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
