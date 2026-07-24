"""
dixon_coles_model.py — modello attacco/difesa Dixon-Coles (1997) via MLE,
alternativa al proxy "media tiri in porta ultime 5 partite" usato oggi dal
tool live (buildTeamProfile in match-predictor-5.html).

Per ogni squadra stima una forza d'attacco e una di difesa (Poisson
regression log-lineare), con home-advantage globale e pesatura a
decadimento temporale (i risultati piu' vecchi contano meno):

    log(lambda_home) = mu + home_adv + attack[home] - defense[away]
    log(lambda_away)  = mu + attack[away] - defense[home]

Identificabilita': penalita' ridge (L2) su attack/defense invece di un
vincolo rigido sum(attack)=0 — numericamente piu' stabile, effetto
equivalente in pratica (le squadre non-informative restano vicine a 0).

XI (decadimento/giorno) e' un valore tipico da letteratura Dixon-Coles,
NON calibrato sui dati (stessa onesta' con cui DC_RHO e' stato lasciato a
-0.1 dopo la calibrazione: qui la priorita' era validare se il modello
attacco/difesa in se' migliora il ROI, non ottimizzare ogni iperparametro).

Walk-forward: un fit per ogni NUOVA data di match incontrata (non per ogni
singolo match: tutti i match della stessa giornata condividono lo stesso
stato "pre-match", rifittare per ognuno sarebbe ridondante e piu' lento).
Usa solo risultati con data < data del fit — nessun leakage. Sotto
MIN_HISTORY_MATCHES partite di storia disponibile non produce previsioni
(troppi pochi dati per stimare in modo affidabile ~2*N_squadre parametri).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

XI = 0.0018  # decadimento temporale giornaliero (letteratura Dixon-Coles 1997, non fittato)
RIDGE = 0.05  # penalita' L2 per identificabilita' attack/defense
MIN_HISTORY_MATCHES = 60


class DixonColesFit:
    __slots__ = ("attack", "defense", "mu", "home_adv")

    def __init__(self, mu: float, home_adv: float, attack: dict, defense: dict):
        self.mu = mu
        self.home_adv = home_adv
        self.attack = attack
        self.defense = defense

    def expected_goals(self, home: str, away: str):
        if home not in self.attack or away not in self.attack:
            return None, None
        lam = math.exp(self.mu + self.home_adv + self.attack[home] - self.defense[away])
        mu_away = math.exp(self.mu + self.attack[away] - self.defense[home])
        return lam, mu_away


def fit_attack_defense(history: list[tuple], as_of_date, prev_fit: "DixonColesFit | None" = None) -> "DixonColesFit | None":
    """history: lista di (date, home, away, fthg, ftag), TUTTE con date < as_of_date."""
    if len(history) < MIN_HISTORY_MATCHES:
        return None

    teams = sorted({h for _, h, _, _, _ in history} | {a for _, _, a, _, _ in history})
    idx = {t: i for i, t in enumerate(teams)}
    K = len(teams)

    weights = np.array([math.exp(-XI * (as_of_date - d).days) for d, _, _, _, _ in history])
    home_idx = np.array([idx[h] for _, h, _, _, _ in history])
    away_idx = np.array([idx[a] for _, _, a, _, _ in history])
    fthg = np.array([g for _, _, _, g, _ in history], dtype=float)
    ftag = np.array([g for _, _, _, _, g in history], dtype=float)

    def loss_and_grad(theta):
        mu, home_adv = theta[0], theta[1]
        attack = theta[2:2 + K]
        defense = theta[2 + K:2 + 2 * K]

        log_lam = mu + home_adv + attack[home_idx] - defense[away_idx]
        log_mu_ = mu + attack[away_idx] - defense[home_idx]
        lam = np.exp(log_lam)
        mu_ = np.exp(log_mu_)

        ll = weights * (fthg * log_lam - lam + ftag * log_mu_ - mu_)
        penalty = RIDGE * (np.sum(attack ** 2) + np.sum(defense ** 2))
        neg_ll = -np.sum(ll) + penalty

        resid_home = weights * (fthg - lam)
        resid_away = weights * (ftag - mu_)

        g_mu = -np.sum(resid_home + resid_away)
        g_home_adv = -np.sum(resid_home)
        g_attack = -(np.bincount(home_idx, weights=resid_home, minlength=K) +
                      np.bincount(away_idx, weights=resid_away, minlength=K)) + 2 * RIDGE * attack
        g_defense = (np.bincount(away_idx, weights=resid_home, minlength=K) +
                     np.bincount(home_idx, weights=resid_away, minlength=K)) + 2 * RIDGE * defense

        grad = np.concatenate(([g_mu, g_home_adv], g_attack, g_defense))
        return neg_ll, grad

    x0 = np.zeros(2 + 2 * K)
    x0[0] = 0.3
    if prev_fit is not None:
        x0[1] = prev_fit.home_adv
        for t in teams:
            i = idx[t]
            x0[2 + i] = prev_fit.attack.get(t, 0.0)
            x0[2 + K + i] = prev_fit.defense.get(t, 0.0)

    res = minimize(loss_and_grad, x0, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    mu, home_adv = res.x[0], res.x[1]
    attack = res.x[2:2 + K]
    defense = res.x[2 + K:2 + 2 * K]
    return DixonColesFit(mu, home_adv, {t: attack[idx[t]] for t in teams}, {t: defense[idx[t]] for t in teams})
