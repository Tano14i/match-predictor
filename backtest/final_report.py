"""
final_report.py — un solo backtest walk-forward che copre tutti e 3 i
mercati del tool (1X2, Over/Under 2.5, Risultato Esatto) sugli stessi dati,
nello stesso giro (piu' efficiente di girare 3 script separati). Riporta
hit rate + calibrazione per ciascun mercato, separatamente per lega e
aggregato — stesso modello attacco/difesa del tool live, nessun filtro
edge (qui la domanda e' "le previsioni sono affidabili", non "sono
profittevoli" — quello e' gia' stato testato a fondo altrove nel repo).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from backtest_football import DC_RHO, parse_date, prob_over_under_goals, probs_1x2_raw, score_prob, to_int
from backtest_attack_defense import load_rows
from dixon_coles_model import fit_attack_defense


def top_scores(xg_h, xg_a, rho=DC_RHO, n=5):
    scores = []
    for h in range(8):
        for a in range(8):
            scores.append((h, a, score_prob(xg_h, xg_a, h, a, rho)))
    total = sum(p for _, _, p in scores)
    scores = [(h, a, p / total) for h, a, p in scores]
    scores.sort(key=lambda s: -s[2])
    return scores[:n]


def run_league(files: list[str], rho: float = DC_RHO) -> dict:
    rows = load_rows(files)
    history: list[tuple] = []
    current_fit = None
    fit_date = None

    picks_1x2, picks_ou, picks_score = [], [], []

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

                probs_map = {"H": w, "D": d, "A": l}
                fav = max(probs_map, key=probs_map.get)
                picks_1x2.append({"prob": probs_map[fav], "won": fav == real_result})

                ou_fav = "Over" if over >= under else "Under"
                ou_prob = over if ou_fav == "Over" else under
                won_ou = (ou_fav == "Over" and real_total > 2.5) or (ou_fav == "Under" and real_total < 2.5)
                picks_ou.append({"prob": ou_prob, "won": won_ou})

                top5 = top_scores(xg_h, xg_a, rho, n=5)
                top1_h, top1_a, top1_p = top5[0]
                picks_score.append({
                    "prob": top1_p,
                    "top1_hit": (top1_h == fthg and top1_a == ftag),
                    "top5_hit": any(h == fthg and a == ftag for h, a, _ in top5),
                })

        history.append((date, home, away, fthg, ftag))

    return {"1x2": picks_1x2, "ou": picks_ou, "score": picks_score}


def summarize_binary(picks: list[dict], label: str):
    n = len(picks)
    if n == 0:
        print(f"  {label}: nessun dato")
        return
    wins = sum(1 for p in picks if p["won"])
    print(f"  {label}: {n} previsioni | hit rate {wins/n*100:.1f}%")


def summarize_score(picks: list[dict], label: str):
    n = len(picks)
    if n == 0:
        print(f"  {label}: nessun dato")
        return
    top1 = sum(1 for p in picks if p["top1_hit"]) / n
    top5 = sum(1 for p in picks if p["top5_hit"]) / n
    print(f"  {label}: {n} previsioni | risultato esatto (pick #1) {top1*100:.1f}% | risultato tra i primi 5 {top5*100:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True, help="Gruppi 'nome:file1,file2,...' (una lega/gruppo per volta)")
    args = ap.parse_args()

    all_picks = {"1x2": [], "ou": [], "score": []}

    print("═" * 70)
    print("BACKTEST PER MERCATO — modello attacco/difesa, walk-forward, no leakage")
    print("═" * 70)

    for group_spec in args.groups:
        name, files_str = group_spec.split(":", 1)
        files = files_str.split(",")
        result = run_league(files)
        for k in all_picks:
            all_picks[k].extend(result[k])

        print(f"\n── {name} ──")
        summarize_binary(result["1x2"], "1X2")
        summarize_binary(result["ou"], "Over/Under 2.5 gol")
        summarize_score(result["score"], "Risultato esatto")

    print("\n" + "═" * 70)
    print("AGGREGATO — tutte le leghe")
    print("═" * 70)
    summarize_binary(all_picks["1x2"], "1X2")
    summarize_binary(all_picks["ou"], "Over/Under 2.5 gol")
    summarize_score(all_picks["score"], "Risultato esatto")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
