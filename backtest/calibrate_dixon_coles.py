"""
calibrate_dixon_coles.py — calibra il parametro rho di Dixon-Coles (1997) sui
dati storici reali, invece di usare il valore -0.1 preso dalla letteratura
(match-predictor-5.html usa DC_RHO=-0.1 fisso, mai fittato sui propri dati).

Metodo: stima di massima verosimiglianza (MLE). Il fattore di correzione
tau(x,y,lambda,mu,rho) di Dixon-Coles modifica SOLO le 4 celle a punteggio
basso (0-0, 1-0, 0-1, 1-1); per ogni altro risultato tau=1 e non dipende da
rho. Quindi massimizzare la log-verosimiglianza del modello equivale a
massimizzare sum(log(tau(h,a,lambda,mu,rho))) sui match osservati con quei
punteggi, dove lambda/mu sono gli xG PRE-match calcolati walk-forward
(stessa logica di backtest_football.team_xg — nessun leakage: per ogni match
si usa solo la storia precedente a quel match).

Fa due cose:
1. Fit IN-SAMPLE per lega: quale rho massimizza la verosimiglianza sui dati
   disponibili, e quanto e' stabile tra leghe diverse.
2. Validazione OUT-OF-SAMPLE walk-forward: rho e' stimato SOLO sulle
   stagioni di allenamento (--files), poi il ROI reale viene ricalcolato
   sulla stagione di test tenuta fuori (--test-files) usando il rho
   congelato — stesso rigore no-leakage del resto del progetto. Questo e'
   il numero che conta: un rho che massimizza la verosimiglianza in-sample
   ma non migliora il ROI out-of-sample non e' un vantaggio reale.

Uso:
    python calibrate_dixon_coles.py --files E0_2020-21.csv E0_2021-22.csv E0_2022-23.csv E0_2023-24.csv E0_2024-25.csv --test-files E0_2025-26.csv --label "Premier League"
    python calibrate_dixon_coles.py --files N1_2020-21.csv N1_2021-22.csv N1_2022-23.csv --test-files N1_2023-24.csv --label "Eredivisie"
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime

from scipy.optimize import minimize_scalar

from backtest_football import (
    DC_RHO,
    MIN_EDGE,
    dixon_coles_tau,
    new_state,
    parse_date,
    run_backtest,
    summarize,
    team_xg,
    to_float,
    to_int,
    update_state,
)

LOW_SCORES = {(0, 0), (1, 0), (0, 1), (1, 1)}


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8-sig") as fh:
            file_rows = list(csv.DictReader(fh))
        file_rows = [r for r in file_rows if r.get("Date")]
        file_rows.sort(key=lambda r: parse_date(r["Date"]) or datetime.min)
        rows.extend(file_rows)
    return rows


def walk_forward_lambdas(rows: list[dict]) -> list[tuple[float, float, int, int]]:
    """Ricostruisce (lambda, mu, gol_casa, gol_trasferta) PRE-match, walk-forward — stessa logica di run_backtest."""
    state: dict[str, dict] = defaultdict(new_state)
    samples = []
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
            samples.append((xg_h, xg_a, fthg, ftag))
        if hst is not None:
            update_state(hs, hst, True)
        if ast is not None:
            update_state(as_, ast, False)
    return samples


def neg_log_likelihood(rho: float, low_score_samples: list[tuple[float, float, int, int]]) -> float:
    total = 0.0
    for lam, mu, h, a in low_score_samples:
        tau = dixon_coles_tau(h, a, lam, mu, rho)
        if tau <= 0:
            return 1e9
        total += math.log(tau)
    return -total


def fit_rho(samples: list[tuple[float, float, int, int]], bounds=(-0.5, 0.5)):
    low_score = [s for s in samples if (s[2], s[3]) in LOW_SCORES]
    res = minimize_scalar(neg_log_likelihood, bounds=bounds, method="bounded", args=(low_score,))
    return res.x, -res.fun, len(low_score), len(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibra Dixon-Coles rho via MLE sui dati storici (walk-forward, no leakage).")
    ap.add_argument("--files", nargs="+", required=True, help="Stagioni di TRAIN su cui stimare rho")
    ap.add_argument("--test-files", nargs="+", default=None, help="Stagione(i) di TEST tenute fuori: ROI ricalcolato qui col rho congelato")
    ap.add_argument("--label", default=None)
    ap.add_argument("--odds-source", default="Avg", choices=["Avg", "B365", "Max", "PS"])
    ap.add_argument("--min-edge", type=float, default=MIN_EDGE)
    args = ap.parse_args()

    label = args.label or " + ".join(args.files)
    rows = load_rows(args.files)
    samples = walk_forward_lambdas(rows)
    rho_hat, ll_hat, n_low, n_tot = fit_rho(samples)
    low_score = [s for s in samples if (s[2], s[3]) in LOW_SCORES]
    ll_zero = -neg_log_likelihood(0.0, low_score)
    ll_base = -neg_log_likelihood(DC_RHO, low_score)

    print(f"=== Calibrazione Dixon-Coles rho — {label} (train) ===")
    print(f"Match con storia sufficiente: {n_tot} (punteggio 0-0/1-0/0-1/1-1: {n_low})")
    print(f"rho calibrato (MLE):        {rho_hat:+.4f}")
    print(f"rho letteratura (baseline): {DC_RHO:+.4f}")
    print(f"log-verosimiglianza: rho=0 {ll_zero:.2f}  |  rho={DC_RHO} {ll_base:.2f}  |  rho={rho_hat:+.4f} (calibrato) {ll_hat:.2f}")

    if args.test_files:
        print(f"\n--- Validazione OUT-OF-SAMPLE su {' + '.join(args.test_files)} (rho congelato dal train, zero leakage) ---")
        test_rows = load_rows(args.test_files)
        for rho_val, tag in [
            (0.0, "rho=0 (nessuna correzione DC)"),
            (DC_RHO, f"rho={DC_RHO} (letteratura, quello usato dal tool live)"),
            (rho_hat, f"rho={rho_hat:+.4f} (calibrato sul train)"),
        ]:
            bets = run_backtest(test_rows, args.min_edge, args.odds_source, rho_val)
            res = summarize(bets)
            o, x, g = res["overall"], res["1x2"], res["goals"]
            print(f"  {tag}")
            print(f"    complessivo: bet {o['bets']} | ROI {o['roi_pct']}%   (1X2 {x['roi_pct']}% | gol O/U {g['roi_pct']}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
