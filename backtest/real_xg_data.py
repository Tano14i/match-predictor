"""
real_xg_data.py — unisce l'xG reale (scaricato da api-football, /fixtures/
statistics, campo "expected_goals") con le quote storiche reali (football-
data.co.uk, gia' usate nel resto del backtest) per data + squadre.

I nomi squadra differiscono tra le due fonti (es. "Manchester City" vs
"Man City"): TEAM_ALIASES mappa nome api-football -> nome football-data.co.uk
per ogni lega, costruito a mano confrontando le liste di squadre delle due
fonti per le stagioni disponibili (vedi commit).

Copertura xG reale (verificata via /fixtures/statistics live, luglio 2026):
inizia a meta' stagione 2022-23 per Premier League/Serie A (da gennaio 2023),
stagione 2023-24 per Eredivisie (2022-23 e' 0% coperta) — NON e' disponibile
retroattivamente per stagioni precedenti. Le partite senza xG valido vengono
scartate, non riempite con un fallback: e' meglio un campione piu' piccolo
ma pulito che uno grande con buchi silenziosi.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

TEAM_ALIASES = {
    "premier_league": {
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Nottingham Forest": "Nott'm Forest",
        "Sheffield Utd": "Sheffield United",
    },
    "serie_a": {
        "AC Milan": "Milan",
        "AS Roma": "Roma",
        "Hellas Verona": "Verona",
    },
    "eredivisie": {
        "Almere City FC": "Almere City",
        "FC Volendam": "Volendam",
        "Fortuna Sittard": "For Sittard",
        "GO Ahead Eagles": "Go Ahead Eagles",
        "NEC Nijmegen": "Nijmegen",
        "PEC Zwolle": "Zwolle",
        "FC Emmen": "Emmen",
    },
}


def normalize_team(name: str, league: str) -> str:
    return TEAM_ALIASES.get(league, {}).get(name, name)


def load_xg_json(path: str, league: str) -> list[dict]:
    """Carica un file xg_*.json (vedi script di fetch), normalizza i nomi
    squadra alla convenzione football-data.co.uk, scarta le partite senza
    xG valido su entrambe le squadre."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for m in raw:
        if m["xgHome"] is None or m["xgAway"] is None:
            continue
        date = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
        out.append({
            "date": date,
            "home": normalize_team(m["home"], league),
            "away": normalize_team(m["away"], league),
            "fthg": m["fthg"],
            "ftag": m["ftag"],
            "xg_home": m["xgHome"],
            "xg_away": m["xgAway"],
        })
    return out


def load_odds_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("Date")]


def parse_csv_date(s: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def merge_xg_with_odds(xg_rows: list[dict], odds_csv_paths: list[str]) -> list[dict]:
    """Unisce le righe xG (gia' filtrate/normalizzate) con le quote storiche
    dei CSV football-data.co.uk, per (data, casa, trasferta). Match sulla
    sola DATA (non l'orario, i due dataset non lo condividono) + nomi
    squadra normalizzati. Le partite xG senza corrispondenza nei CSV quote
    (nessuna quota storica disponibile) vengono scartate con un conteggio,
    non silenziosamente."""
    odds_by_key: dict[tuple, dict] = {}
    for path in odds_csv_paths:
        for r in load_odds_csv(path):
            d = parse_csv_date(r["Date"])
            if d is None or not r.get("HomeTeam") or not r.get("AwayTeam"):
                continue
            key = (d.date(), r["HomeTeam"], r["AwayTeam"])
            odds_by_key[key] = r

    merged, unmatched = [], 0
    for m in xg_rows:
        key = (m["date"].date(), m["home"], m["away"])
        odds_row = odds_by_key.get(key)
        if odds_row is None:
            unmatched += 1
            continue
        merged.append({**m, "odds": odds_row})
    return merged, unmatched
