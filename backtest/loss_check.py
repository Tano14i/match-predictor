"""
loss_check.py — risposta diretta a "sarei in perdita?": hit rate alto NON
implica profitto, perche' le quote del bookmaker sono prezzate proprio
sulla base di probabilita' simili (o migliori) a quelle del modello, piu'
un margine. Qui si calcola il ROI REALE di "scommetti sempre il pick
preferito del modello, quota media di mercato, puntata fissa 1 unita',
NESSUN filtro di edge" — la domanda pratica "se mi fido ciecamente del
tool e scommetto su tutto quello che dice, cosa succede ai miei soldi".
"""
from __future__ import annotations

from backtest_football import DC_RHO, parse_date, prob_over_under_goals, probs_1x2_raw, to_float, to_int
from backtest_attack_defense import load_rows
from dixon_coles_model import fit_attack_defense


def run_league(files: list[str], odds_source: str = "Avg", rho: float = DC_RHO) -> dict:
    hcol, dcol, acol = f"{odds_source}H", f"{odds_source}D", f"{odds_source}A"
    ocol, ucol = f"{odds_source}>2.5", f"{odds_source}<2.5"

    rows = load_rows(files)
    history: list[tuple] = []
    current_fit = None
    fit_date = None
    bets_1x2, bets_ou = [], []

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
                odds_map = {"H": to_float(r.get(hcol)), "D": to_float(r.get(dcol)), "A": to_float(r.get(acol))}
                fav_odds = odds_map[fav]
                if fav_odds:
                    won = fav == real_result
                    bets_1x2.append((fav_odds - 1) if won else -1)

                ou_fav = "Over" if over >= under else "Under"
                ou_odds = to_float(r.get(ocol)) if ou_fav == "Over" else to_float(r.get(ucol))
                if ou_odds:
                    won = (ou_fav == "Over" and real_total > 2.5) or (ou_fav == "Under" and real_total < 2.5)
                    bets_ou.append((ou_odds - 1) if won else -1)

        history.append((date, home, away, fthg, ftag))

    return {"1x2": bets_1x2, "ou": bets_ou}


def report(profits: list[float], label: str):
    n = len(profits)
    if n == 0:
        print(f"  {label}: nessuna scommessa")
        return
    total = sum(profits)
    roi = total / n * 100
    print(f"  {label}: {n} scommesse da 1 unita' | risultato netto {total:+.1f}u | ROI {roi:+.1f}%")


def main():
    groups = {
        "Premier League": ["E0_2020-21.csv", "E0_2021-22.csv", "E0_2022-23.csv", "E0_2023-24.csv", "E0_2024-25.csv", "E0_2025-26.csv"],
        "Serie A": ["I1_2021-22.csv", "I1_2022-23.csv", "I1_2023-24.csv", "I1_2024-25.csv", "I1_2025-26.csv"],
        "Eredivisie": ["N1_2020-21.csv", "N1_2021-22.csv", "N1_2022-23.csv", "N1_2023-24.csv"],
        "Championship": ["minor_leagues/E1_2023-24.csv", "minor_leagues/E1_2024-25.csv", "minor_leagues/E1_2025-26.csv"],
        "League One": ["minor_leagues/E2_2023-24.csv", "minor_leagues/E2_2024-25.csv", "minor_leagues/E2_2025-26.csv"],
        "Belgio": ["minor_leagues/B1_2023-24.csv", "minor_leagues/B1_2024-25.csv", "minor_leagues/B1_2025-26.csv"],
        "Grecia": ["minor_leagues/G1_2023-24.csv", "minor_leagues/G1_2024-25.csv", "minor_leagues/G1_2025-26.csv"],
        "Segunda": ["minor_leagues/SP2_2023-24.csv", "minor_leagues/SP2_2024-25.csv", "minor_leagues/SP2_2025-26.csv"],
        "Portogallo": ["minor_leagues/P1_2023-24.csv", "minor_leagues/P1_2024-25.csv", "minor_leagues/P1_2025-26.csv"],
        "Serie B": ["minor_leagues/I2_2023-24.csv", "minor_leagues/I2_2024-25.csv", "minor_leagues/I2_2025-26.csv"],
    }

    print("Se scommetti SEMPRE il pick preferito del tool, 1 unita' a botta, quote medie di mercato reali:\n")

    all_1x2, all_ou = [], []
    for name, files in groups.items():
        result = run_league(files)
        all_1x2.extend(result["1x2"])
        all_ou.extend(result["ou"])
        print(f"── {name} ──")
        report(result["1x2"], "1X2")
        report(result["ou"], "Over/Under 2.5")
        print()

    print("═" * 60)
    print("TOTALE — tutte le leghe, tutte le scommesse")
    print("═" * 60)
    report(all_1x2, "1X2")
    report(all_ou, "Over/Under 2.5")
    combined = all_1x2 + all_ou
    report(combined, "COMPLESSIVO (1X2 + Over/Under insieme)")


if __name__ == "__main__":
    main()
