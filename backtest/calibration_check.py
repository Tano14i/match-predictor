"""
calibration_check.py — NON e' un test di EV/edge (quello e' backtest_*.py).
Risponde a una domanda diversa: quando il modello dice "55% vittoria casa",
la casa vince davvero circa il 55% delle volte nei dati reali? Questa e' la
domanda giusta per un tool che deve dare un'ipotesi affidabile (1X2/Over-
Under/risultato probabile), non per un tool che deve battere il bookmaker
— sono obiettivi diversi, il primo e' molto piu' facile da soddisfare del
secondo (un mercato efficiente puo' benissimo "sapere" quello che sa anche
il modello, senza che questo renda le previsioni del modello sbagliate).

Per ogni partita con storia sufficiente, registra il pick FAVORITO del
modello (quello con probabilita' piu' alta tra le opzioni del mercato) e
se ha vinto — SEMPRE, non solo quando c'e' un edge >=5% contro le quote
(quello e' un filtro per scommettere, qui vogliamo giudicare la previsione
in se'). Poi:
  1. Curva di calibrazione: raggruppa i pick per fascia di probabilita'
     prevista, confronta con la frequenza reale di vittoria in quella
     fascia (dovrebbero coincidere se il modello e' onesto).
  2. Win rate e ROI di "segui SEMPRE il pick favorito del modello" con le
     quote medie di mercato — non e' una strategia di value betting (nessun
     filtro edge), e' la domanda pratica "se mi fido del tool, cosa
     succede".

Usa lo stesso modello attacco/difesa gia' validato (dixon_coles_model.py),
stesso motore Poisson/Dixon-Coles, stesse quote storiche reali.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from backtest_football import (
    DC_RHO,
    parse_date,
    prob_over_under_goals,
    probs_1x2_raw,
    to_float,
    to_int,
)
from backtest_attack_defense import load_rows
from dixon_coles_model import fit_attack_defense

BUCKETS = [(0.0, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 1.01)]


def bucket_for(p: float) -> tuple[float, float]:
    for lo, hi in BUCKETS:
        if lo <= p < hi:
            return (lo, hi)
    return BUCKETS[-1]


def run(rows: list[dict], odds_source: str = "Avg", rho: float = DC_RHO):
    hcol, dcol, acol = f"{odds_source}H", f"{odds_source}D", f"{odds_source}A"
    ocol, ucol = f"{odds_source}>2.5", f"{odds_source}<2.5"

    history: list[tuple] = []
    current_fit = None
    fit_date = None
    picks_1x2, picks_goals = [], []

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

                real_result = "H" if fthg > ftag else ("A" if ftag > fthg else "D")
                real_total = fthg + ftag

                # Pick 1X2: il favorito del modello, SEMPRE (nessun filtro edge)
                probs_map = {"H": w, "D": d, "A": l}
                fav_label = max(probs_map, key=probs_map.get)
                fav_prob = probs_map[fav_label]
                odds_map = {"H": to_float(r.get(hcol)), "D": to_float(r.get(dcol)), "A": to_float(r.get(acol))}
                fav_odds = odds_map[fav_label]
                if fav_odds:
                    picks_1x2.append({"prob": fav_prob, "won": fav_label == real_result, "odds": fav_odds})

                # Pick Over/Under: il favorito del modello, SEMPRE
                og_label = "Over" if over >= under else "Under"
                og_prob = over if og_label == "Over" else under
                og_odds = to_float(r.get(ocol)) if og_label == "Over" else to_float(r.get(ucol))
                if og_odds:
                    won = (og_label == "Over" and real_total > 2.5) or (og_label == "Under" and real_total < 2.5)
                    picks_goals.append({"prob": og_prob, "won": won, "odds": og_odds})

        history.append((date, home, away, fthg, ftag))

    return picks_1x2, picks_goals


def report(picks: list[dict], label: str):
    if not picks:
        print(f"{label}: nessun pick registrato")
        return

    n = len(picks)
    wins = sum(1 for p in picks if p["won"])
    profit = sum((p["odds"] - 1) if p["won"] else -1 for p in picks)
    print(f"\n=== {label} — segui SEMPRE il pick favorito del modello (no filtro edge) ===")
    print(f"  pick totali: {n} | win rate: {wins/n*100:.1f}% | ROI: {profit/n*100:+.1f}% | profitto: {profit:+.2f}u")

    print(f"\n  Calibrazione (probabilita' prevista vs frequenza reale di vittoria):")
    by_bucket = defaultdict(list)
    for p in picks:
        by_bucket[bucket_for(p["prob"])].append(p)
    for lo, hi in BUCKETS:
        bucket_picks = by_bucket.get((lo, hi), [])
        if not bucket_picks:
            continue
        avg_pred = sum(p["prob"] for p in bucket_picks) / len(bucket_picks)
        actual_rate = sum(1 for p in bucket_picks if p["won"]) / len(bucket_picks)
        n_b = len(bucket_picks)
        diff = (actual_rate - avg_pred) * 100
        flag = "  <-- scostamento notevole" if abs(diff) > 5 and n_b >= 30 else ""
        print(f"    [{lo*100:.0f}-{hi*100:.0f}%) n={n_b:<5} previsto medio {avg_pred*100:.1f}%  reale {actual_rate*100:.1f}%  (scarto {diff:+.1f} punti){flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Verifica se le probabilita' previste dal modello sono calibrate (coincidono con la frequenza reale), non se sono profittevoli.")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--odds-source", default="Avg", choices=["Avg", "B365", "Max", "PS"])
    args = ap.parse_args()

    rows = load_rows(args.files)
    label = args.label or " + ".join(args.files)
    print(f"Match totali ({label}): {len(rows)} | fonte quote: {args.odds_source}")

    picks_1x2, picks_goals = run(rows, args.odds_source)
    report(picks_1x2, f"{label} — 1X2")
    report(picks_goals, f"{label} — Over/Under 2.5 gol")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
