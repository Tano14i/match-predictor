"""
fit_calibration.py — corregge l'overconfidence del modello attacco/difesa
scoperta da calibration_check.py (specialmente Over/Under alto: previsto
76.8%, reale 61.6%). Fitta una mappa di ricalibrazione per isotonic
regression (monotona, non parametrica: non assume una forma specifica
della distorsione) su P(1), P(X), P(2), P(Over) separatamente, usando
TUTTE le previsioni del modello (non solo il pick favorito, per avere piu'
dati e coprire l'intero range 0-100%).

Disciplina train/test identica al resto del progetto: la mappa viene
fittata SOLO sulle stagioni di train, poi validata sulla stagione di test
tenuta fuori (l'ultima disponibile per ogni lega) — se la calibrazione
migliora anche li', e' un effetto reale, non overfitting sui dati usati
per costruirla.

Output: mappa esportata come tabella (probabilita' grezza -> corretta) in
calibration_map.json, pronta per essere applicata sia in Python (backtest)
che in JS (tool live) senza bisogno di sklearn a runtime.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
from sklearn.isotonic import IsotonicRegression

from backtest_football import DC_RHO, parse_date, prob_over_under_goals, probs_1x2_raw, to_int
from backtest_attack_defense import load_rows
from dixon_coles_model import fit_attack_defense


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
                w, d, l = probs_1x2_raw(xg_h, xg_a, rho)
                over, under = prob_over_under_goals(xg_h, xg_a, 2.5, rho)
                real_result = "H" if fthg > ftag else ("A" if ftag > fthg else "D")
                real_total = fthg + ftag
                out.append({
                    "date": date, "pH": w, "pD": d, "pA": l, "pOver": over,
                    "isH": real_result == "H", "isD": real_result == "D", "isA": real_result == "A",
                    "isOver": real_total > 2.5,
                })

        history.append((date, home, away, fthg, ftag))

    return out


def fit_isotonic(probs: np.ndarray, outcomes: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    iso.fit(probs, outcomes)
    return iso


def eval_calibration(probs: np.ndarray, outcomes: np.ndarray, label: str, buckets=((0,.4),(.4,.5),(.5,.6),(.6,.7),(.7,1.01))):
    print(f"  {label}:")
    for lo, hi in buckets:
        mask = (probs >= lo) & (probs < hi)
        n = mask.sum()
        if n < 20:
            continue
        pred = probs[mask].mean()
        real = outcomes[mask].mean()
        print(f"    [{lo*100:.0f}-{hi*100:.0f}%) n={n:<5} previsto {pred*100:.1f}%  reale {real*100:.1f}%  scarto {(real-pred)*100:+.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fitta e valida (train/test) una correzione di calibrazione isotonic per le probabilita' del modello.")
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--test-files", nargs="+", required=True)
    ap.add_argument("--out", default="calibration_map.json")
    args = ap.parse_args()

    train_rows = load_rows(args.train_files)
    test_rows = load_rows(args.test_files)
    print(f"Train: {len(train_rows)} match | Test (tenuto fuori): {len(test_rows)} match")

    train_preds = collect_predictions(train_rows)
    test_preds = collect_predictions(test_rows)
    print(f"Previsioni utilizzabili — train: {len(train_preds)} | test: {len(test_preds)}")

    pH_tr = np.array([p["pH"] for p in train_preds]); isH_tr = np.array([p["isH"] for p in train_preds], dtype=float)
    pD_tr = np.array([p["pD"] for p in train_preds]); isD_tr = np.array([p["isD"] for p in train_preds], dtype=float)
    pA_tr = np.array([p["pA"] for p in train_preds]); isA_tr = np.array([p["isA"] for p in train_preds], dtype=float)
    pOver_tr = np.array([p["pOver"] for p in train_preds]); isOver_tr = np.array([p["isOver"] for p in train_preds], dtype=float)

    iso_H = fit_isotonic(pH_tr, isH_tr)
    iso_D = fit_isotonic(pD_tr, isD_tr)
    iso_A = fit_isotonic(pA_tr, isA_tr)
    iso_Over = fit_isotonic(pOver_tr, isOver_tr)

    pH_te = np.array([p["pH"] for p in test_preds]); isH_te = np.array([p["isH"] for p in test_preds], dtype=float)
    pOver_te = np.array([p["pOver"] for p in test_preds]); isOver_te = np.array([p["isOver"] for p in test_preds], dtype=float)

    print("\n=== PRIMA della calibrazione (dati di TEST, mai visti nel fit) ===")
    eval_calibration(pH_te, isH_te, "P(vittoria casa) grezza")
    eval_calibration(pOver_te, isOver_te, "P(Over 2.5) grezza")

    pH_te_cal = iso_H.predict(pH_te)
    pOver_te_cal = iso_Over.predict(pOver_te)

    print("\n=== DOPO la calibrazione (stessi dati di TEST) ===")
    eval_calibration(pH_te_cal, isH_te, "P(vittoria casa) calibrata")
    eval_calibration(pOver_te_cal, isOver_te, "P(Over 2.5) calibrata")

    grid = np.linspace(0, 1, 101)
    calib_map = {
        "H": {"x": grid.tolist(), "y": iso_H.predict(grid).tolist()},
        "D": {"x": grid.tolist(), "y": iso_D.predict(grid).tolist()},
        "A": {"x": grid.tolist(), "y": iso_A.predict(grid).tolist()},
        "Over": {"x": grid.tolist(), "y": iso_Over.predict(grid).tolist()},
    }
    with open(args.out, "w") as f:
        json.dump(calib_map, f)
    print(f"\nMappa di calibrazione salvata in {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
