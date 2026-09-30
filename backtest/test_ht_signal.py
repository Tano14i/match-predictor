"""
test_ht_signal.py — verifica onesta, su dati reali, dell'ipotesi di fondo
del video "pronostico leggendario": le partite gia' "vive" a meta' primo
tempo hanno piu' probabilita' di un altro gol nel resto della partita?

Usiamo il risultato all'intervallo (HTHG/HTAG, disponibile in tutti i CSV
football-data.co.uk) come proxy per "partita viva al minuto ~47" — non
abbiamo i tiri in porta minuto-per-minuto che usa il tipster, ma il suo
stesso dato (7 delle sue 8 partite perse partivano 0-0 all'intervallo)
suggerisce che il risultato all'intervallo catturi gia' buona parte del
segnale.

LIMITE ONESTO, dichiarato subito: questo test misura solo la PROBABILITA'
di un altro gol, non se le quote LIVE in quel momento sono sottoprezzate.
Non abbiamo dati storici di quote live (nessuna fonte gratuita/economica
le offre) — quindi anche se il test conferma "sì, più probabile", questo
NON dimostra un vantaggio scommettendo: il bookmaker probabilmente aggiusta
gia' le quote live esattamente su questa stessa informazione (e' il dato
piu' ovvio possibile per chi prezza una partita in corso).
"""
import csv
import glob


def load_all(pattern):
    rows = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8-sig") as f:
            rows.extend(csv.DictReader(f))
    return rows


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main():
    files = load_all("*.csv") + load_all("minor_leagues/*.csv")
    print(f"Match totali caricati: {len(files)}")

    tied_0_0 = {"n": 0, "second_half_goal": 0}
    not_tied = {"n": 0, "second_half_goal": 0}

    for r in files:
        hthg, htag = to_int(r.get("HTHG")), to_int(r.get("HTAG"))
        fthg, ftag = to_int(r.get("FTHG")), to_int(r.get("FTAG"))
        if None in (hthg, htag, fthg, ftag):
            continue

        ht_total = hthg + htag
        second_half_goals = (fthg + ftag) - ht_total
        had_goal = second_half_goals >= 1

        bucket = tied_0_0 if ht_total == 0 else not_tied
        bucket["n"] += 1
        if had_goal:
            bucket["second_half_goal"] += 1

    print("\n=== Probabilita' di almeno un altro gol nel resto della partita ===")
    for label, b in [("0-0 all'intervallo", tied_0_0), ("Gia' sbloccata all'intervallo (qualsiasi risultato diverso da 0-0)", not_tied)]:
        pct = b["second_half_goal"] / b["n"] * 100
        print(f"  {label}: {b['n']} partite | almeno un altro gol: {pct:.1f}%")

    diff = (not_tied["second_half_goal"] / not_tied["n"]) - (tied_0_0["second_half_goal"] / tied_0_0["n"])
    print(f"\nDifferenza: {diff*100:+.1f} punti percentuali")


if __name__ == "__main__":
    main()
