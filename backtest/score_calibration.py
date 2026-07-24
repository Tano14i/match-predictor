"""
score_calibration.py — stessa domanda di calibration_check.py/fit_calibration.py
ma per il RISULTATO ESATTO (topScores nel tool live), non 1X2/Over-Under.

Due cose diverse da misurare:
  1. Calibrazione: quando il modello dice "il punteggio X-Y e' il piu'
     probabile, con probabilita' P%", il punteggio X-Y esce davvero circa
     P% delle volte? (stessa logica di calibration_check.py, applicata al
     pick #1 della griglia Poisson+Dixon-Coles invece che a H/D/A o O/U)
  2. Hit rate onesto: quanto spesso il punteggio esatto indovinato e'
     quello vero (aspettativa realistica: anche un modello perfetto avra'
     un hit rate basso, il calcio ha troppi punteggi possibili — non e' un
     difetto del modello, e' la natura del problema). Riportato anche
     "il risultato vero era tra i primi 5 suggeriti?" come metrica piu'
     morbida e realistica per l'uso pratico del tool.

Stessa disciplina train/test di fit_calibration.py: la correzione (se
serve) viene fittata SOLO sul train, validata sul test tenuto fuori.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

from backtest_football import DC_RHO, parse_date, score_prob, to_int
from backtest_attack_defense import load_rows
from dixon_coles_model import fit_attack_defense


def top_scores(xg_h: float, xg_a: float, rho: float = DC_RHO, n: int = 5):
    scores = []
    for h in range(8):
        for a in range(8):
            scores.append((h, a, score_prob(xg_h, xg_a, h, a, rho)))
    total = sum(p for _, _, p in scores)
    scores = [(h, a, p / total) for h, a, p in scores]
    scores.sort(key=lambda s: -s[2])
    return scores[:n]


def collect_predictions(rows: list[dict], rho: float = DC_RHO) -> list[dict]:
    history: list[tuple] = []
    current_fit = None
    fit_date = None
    out = []

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
                top5 = top_scores(xg_h, xg_a, rho, n=5)
                top1_h, top1_a, top1_p = top5[0]
                out.append({
                    "date": date,
                    "top1_prob": top1_p,
                    "top1_hit": (top1_h == fthg and top1_a == ftag),
                    "top5_hit": any(h == fthg and a == ftag for h, a, _ in top5),
                })

        history.append((date, home, away, fthg, ftag))

    return out


def eval_calibration(probs: np.ndarray, hits: np.ndarray, label: str, buckets=((0,.08),(.08,.10),(.10,.12),(.12,.15),(.15,1.01))):
    print(f"  {label}:")
    for lo, hi in buckets:
        mask = (probs >= lo) & (probs < hi)
        n = mask.sum()
        if n < 20:
            continue
        pred = probs[mask].mean()
        real = hits[mask].mean()
        print(f"    [{lo*100:.0f}-{hi*100:.0f}%) n={n:<5} previsto {pred*100:.1f}%  reale {real*100:.1f}%  scarto {(real-pred)*100:+.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrazione del risultato esatto (pick #1 della griglia Poisson+Dixon-Coles).")
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--test-files", nargs="+", required=True)
    ap.add_argument("--out", default="score_calibration_map.json")
    args = ap.parse_args()

    train_rows = load_rows(args.train_files)
    test_rows = load_rows(args.test_files)
    train_preds = collect_predictions(train_rows)
    test_preds = collect_predictions(test_rows)
    print(f"Previsioni utilizzabili — train: {len(train_preds)} | test: {len(test_preds)}")

    top1_hit_rate = sum(1 for p in test_preds if p["top1_hit"]) / len(test_preds)
    top5_hit_rate = sum(1 for p in test_preds if p["top5_hit"]) / len(test_preds)
    print(f"\n=== Hit rate onesto (dati di TEST, mai visti nel fit) ===")
    print(f"  Risultato esatto = pick #1 del modello: {top1_hit_rate*100:.1f}% delle volte")
    print(f"  Risultato esatto tra i primi 5 suggeriti: {top5_hit_rate*100:.1f}% delle volte")

    probs_tr = np.array([p["top1_prob"] for p in train_preds])
    hits_tr = np.array([p["top1_hit"] for p in train_preds], dtype=float)
    probs_te = np.array([p["top1_prob"] for p in test_preds])
    hits_te = np.array([p["top1_hit"] for p in test_preds], dtype=float)

    print(f"\n=== Calibrazione PRIMA (dati di TEST) ===")
    eval_calibration(probs_te, hits_te, "P(pick #1 esatto) grezza")

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.60)
    iso.fit(probs_tr, hits_tr)
    probs_te_cal = iso.predict(probs_te)

    print(f"\n=== Calibrazione DOPO (stessi dati di TEST) ===")
    eval_calibration(probs_te_cal, hits_te, "P(pick #1 esatto) calibrata")

    grid = np.linspace(0, 0.5, 101)
    calib_map = {"x": grid.tolist(), "y": iso.predict(grid).tolist()}
    with open(args.out, "w") as f:
        json.dump(calib_map, f)
    print(f"\nMappa salvata in {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
