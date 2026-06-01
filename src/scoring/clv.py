"""CLV — Closing Line Value. Le juge de paix du projet.

Mesure si la cote à laquelle on a "parié" (journal, cote précoce capturée via The
Odds API) était meilleure que la cote de CLÔTURE Pinnacle (football-data,
PSCH/PSCD/PSCA). Battre la ligne de clôture régulièrement est le meilleur
prédicteur de profitabilité — bien plus stable que les gains/pertes à court
terme, noyés dans la variance.

Deux mesures par pari :
  - CLV brut       = cote_pari / cote_clôture − 1
  - CLV dévigorisé = p_clôture_devig / p_pari_devig − 1
    (rigoureux : compare des probabilités sans marge ; nécessite les 3 cotes
     de clôture pour dévigoriser. C'est la mesure de référence.)

Jointure pari → clôture : via TeamResolver (oddsapi → canonique vs football-data
→ canonique) + date du match. Un pari sans match football-data correspondant
(pas encore joué, ou non trouvé) est laissé "en attente", jamais deviné.

Pinnacle peut manquer sur certains matchs : repli sur la moyenne marché de
clôture (AvgCH/AvgCD/AvgCA), signalé dans le rapport.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from common.teams import TeamResolver, UnknownTeamError
from scoring.betlog import BetRecord, load_log, DEFAULT_LOG

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHES = _ROOT / "data" / "processed" / "matches.csv"
DEFAULT_OUT = _ROOT / "data" / "bets" / "clv_report.csv"

# Colonnes de clôture football-data, par issue.
CLOSE_PINNACLE = {"home": "PSCH", "draw": "PSCD", "away": "PSCA"}
CLOSE_MARKET = {"home": "AvgCH", "draw": "AvgCD", "away": "AvgCA"}


@dataclass
class ClvResult:
    bet: BetRecord
    close_odds: float | None
    close_source: str            # 'pinnacle' | 'market' | 'none'
    clv_raw: float | None
    clv_devig: float | None
    status: str                  # 'matched' | 'pending' | 'unresolved'


# ---------------------------------------------------------------------- #
# Chargement des clôtures football-data, indexées par (date, home, away)
# ---------------------------------------------------------------------- #
def _parse_date(s: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def load_closings(matches_path: Path):
    """Indexe les cotes de clôture par (date_iso, home_canon, away_canon).
    Les équipes football-data sont déjà canoniques dans matches.csv."""
    closings = {}
    with Path(matches_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            h = (row.get("HomeTeamCanonical") or "").strip()
            a = (row.get("AwayTeamCanonical") or "").strip()
            d = _parse_date(row.get("Date", ""))
            if not (h and a and d):
                continue
            key = (d.strftime("%Y-%m-%d"), h, a)
            closings[key] = row
    return closings


def _read_close(row, issue: str) -> tuple[float | None, str]:
    """Cote de clôture pour une issue : Pinnacle d'abord, moyenne en repli."""
    for cols, src in ((CLOSE_PINNACLE, "pinnacle"), (CLOSE_MARKET, "market")):
        v = (row.get(cols[issue]) or "").strip()
        if v:
            try:
                f = float(v)
                if f > 1.0:
                    return f, src
            except ValueError:
                pass
    return None, "none"


def _all_close_odds(row, source_cols) -> dict | None:
    """Les 3 cotes de clôture d'une même source (pour dévigoriser)."""
    out = {}
    for issue, col in source_cols.items():
        v = (row.get(col) or "").strip()
        try:
            f = float(v)
        except (ValueError, TypeError):
            return None
        if f <= 1.0:
            return None
        out[issue] = f
    return out


def _devig(odds: dict[str, float]) -> dict[str, float]:
    inv = {k: 1.0 / v for k, v in odds.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


# ---------------------------------------------------------------------- #
# Calcul du CLV d'un pari
# ---------------------------------------------------------------------- #
def compute_clv_for_bet(
    bet: BetRecord, closings: dict, resolver: TeamResolver
) -> ClvResult:
    # Le journal stocke déjà des noms canoniques (le scoring travaille en
    # canonique). On tente d'abord tel quel ; sinon on résout via oddsapi.
    home, away = bet.home, bet.away
    md = _parse_date(bet.match_date)
    if md is None:
        return ClvResult(bet, None, "none", None, None, "pending")
    key = (md.strftime("%Y-%m-%d"), home, away)
    row = closings.get(key)

    if row is None:
        # Pas trouvé : match pas encore joué/ingéré, ou noms à résoudre.
        # On tente une résolution oddsapi -> canonique au cas où le journal
        # contiendrait un nom brut oddsapi.
        try:
            hc = resolver.to_canonical(home, "oddsapi")
            ac = resolver.to_canonical(away, "oddsapi")
            row = closings.get((md.strftime("%Y-%m-%d"), hc, ac))
        except UnknownTeamError:
            row = None
    if row is None:
        return ClvResult(bet, None, "none", None, None, "pending")

    close_odds, src = _read_close(row, bet.issue)
    if close_odds is None:
        return ClvResult(bet, None, "none", None, None, "unresolved")

    clv_raw = bet.odds / close_odds - 1.0

    # CLV dévigorisé : nécessite les 3 cotes de clôture de la même source.
    clv_devig = None
    src_cols = CLOSE_PINNACLE if src == "pinnacle" else CLOSE_MARKET
    close_all = _all_close_odds(row, src_cols)
    if close_all is not None:
        p_close = _devig(close_all)[bet.issue]
        # proba implicite de la cote du pari : on n'a qu'une cote (pas les 3 du
        # même book au même instant), donc on compare la proba implicite brute
        # du pari à la proba dévigorisée de clôture. Approximation raisonnable
        # et conservatrice (le pari garde sa marge, la clôture non).
        p_bet = 1.0 / bet.odds
        if p_bet > 0:
            clv_devig = p_close / p_bet - 1.0

    return ClvResult(bet, close_odds, src, clv_raw, clv_devig, "matched")


# ---------------------------------------------------------------------- #
# Rapport agrégé
# ---------------------------------------------------------------------- #
def run(
    log_path: Path = DEFAULT_LOG,
    matches_path: Path = DEFAULT_MATCHES,
    out_path: Path = DEFAULT_OUT,
    resolver: TeamResolver | None = None,
):
    bets = load_log(log_path)
    print(f"Paris journalisés : {len(bets)}")
    if not bets:
        print("Aucun pari à évaluer (journal vide — normal hors saison).")
        return []

    closings = load_closings(matches_path)
    resolver = resolver or TeamResolver.from_csv()

    results = [compute_clv_for_bet(b, closings, resolver) for b in bets]
    matched = [r for r in results if r.status == "matched"]
    pending = [r for r in results if r.status == "pending"]

    print(f"Appariés (match joué + clôture dispo) : {len(matched)}")
    print(f"En attente (match pas encore joué/ingéré) : {len(pending)}")

    if matched:
        raw_vals = [r.clv_raw for r in matched if r.clv_raw is not None]
        dv_vals = [r.clv_devig for r in matched if r.clv_devig is not None]
        avg_raw = sum(raw_vals) / len(raw_vals) if raw_vals else float("nan")
        pos = sum(1 for v in raw_vals if v > 0)
        print("\n=== CLV (sur paris appariés) ===")
        print(f"  CLV brut moyen       : {avg_raw:+.4f} ({avg_raw*100:+.2f} %)")
        if dv_vals:
            avg_dv = sum(dv_vals) / len(dv_vals)
            print(f"  CLV dévigorisé moyen : {avg_dv:+.4f} ({avg_dv*100:+.2f} %)")
        print(f"  Paris battant la clôture : {pos}/{len(raw_vals)} "
              f"({100*pos/len(raw_vals):.0f} %)")
        print("\n=== VERDICT ===")
        if avg_raw > 0:
            print("  ✓ CLV moyen POSITIF : le scoring obtient de meilleures cotes")
            print("    que la clôture. C'est le signe d'un avantage réel (à")
            print("    confirmer sur un échantillon plus grand).")
        else:
            print("  ✗ CLV moyen négatif : le scoring n'obtient pas de meilleures")
            print("    cotes que la clôture. Le modèle ne repère pas de valeur")
            print("    exploitable — ou le seuil/les réglages sont à revoir.")
        print(f"\n  ⚠ Échantillon : {len(raw_vals)} paris. Le CLV n'est fiable")
        print("    qu'à partir de ~100-200 paris. Patience avant de conclure.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_date", "home", "away", "issue", "bet_odds",
                    "close_odds", "close_source", "clv_raw", "clv_devig", "status"])
        for r in results:
            w.writerow([
                r.bet.match_date, r.bet.home, r.bet.away, r.bet.issue,
                f"{r.bet.odds:.4f}",
                f"{r.close_odds:.4f}" if r.close_odds is not None else "",
                r.close_source,
                f"{r.clv_raw:.4f}" if r.clv_raw is not None else "",
                f"{r.clv_devig:.4f}" if r.clv_devig is not None else "",
                r.status,
            ])
    print(f"\nÉcrit : {out_path}")
    return results


if __name__ == "__main__":  # pragma: no cover
    run()
