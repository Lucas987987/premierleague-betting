"""Orchestrateur du scoring — chaîne complète pour les matchs à venir.

Enchaîne : dernier snapshot de cotes (Odds API) → résolution des noms →
rechargement du modèle bayésien figé (params_bayes_full.json, découplage
fit/predict) → prédiction 1/N/2 avec intervalles → scoring EV + Signal×Fiabilité
→ journalisation des paris dépassant le seuil.

Ne réajuste PAS le modèle : il recharge les forces figées (cf. ARCHITECTURE.md,
découplage fit/predict). Réajuster relève de fit_bayes.yml, à un autre rythme.

Hors saison, le snapshot de cotes est vide → 0 match scoré, rien à journaliser.
Comportement normal, pas une erreur.

Agrégation des cotes : pour chaque match et chaque issue, on retient la MÉDIANE
des cotes des bookmakers (robuste aux books aberrants), et on conserve la liste
complète des cotes par issue pour mesurer la convergence (fiabilité).
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

from common.teams import TeamResolver, UnknownTeamError
from model.dixoncoles_bayes import BayesParams, predict_1x2_bayes
from scoring.value import score_match, ISSUES
from scoring.betlog import record_bets, DEFAULT_SCORE_THRESHOLD

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ODDS_DIR = _ROOT / "data" / "raw" / "oddsapi"
DEFAULT_MODEL_FULL = _ROOT / "data" / "model" / "params_bayes_full.json"
DEFAULT_MATCHES = _ROOT / "data" / "processed" / "matches.csv"

# Correspondance des noms d'issue Odds API -> nos clés internes.
# Odds API donne les outcomes par NOM d'équipe + "Draw".
OUTCOME_DRAW = "Draw"


def latest_snapshot(odds_dir: Path) -> Path | None:
    snaps = sorted(Path(odds_dir).glob("odds_*.json"))
    return snaps[-1] if snaps else None


def _team_match_counts(matches_path: Path) -> dict[str, int]:
    """Nombre de matchs joués par équipe canonique (pour la fiabilité données)."""
    counts: dict[str, int] = {}
    p = Path(matches_path)
    if not p.exists():
        return counts
    import csv
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for col in ("HomeTeamCanonical", "AwayTeamCanonical"):
                t = (row.get(col) or "").strip()
                if t:
                    counts[t] = counts.get(t, 0) + 1
    return counts


def _extract_odds(event: dict, resolver: TeamResolver):
    """D'un événement Odds API, extrait les cotes par issue (médiane + liste).

    Retourne (home_canon, away_canon, match_date, consensus, book_prices) ou None
    si les équipes ne se résolvent pas.
    """
    home_raw = event.get("home_team", "")
    away_raw = event.get("away_team", "")
    try:
        home = resolver.to_canonical(home_raw, "oddsapi")
        away = resolver.to_canonical(away_raw, "oddsapi")
    except UnknownTeamError:
        return None  # nom non mappé : on signale plus haut, on ne devine pas

    commence = event.get("commence_time", "")
    match_date = commence[:10] if commence else ""  # ISO -> YYYY-MM-DD

    # Collecte des cotes : prices[issue] = liste sur tous les books.
    prices: dict[str, list[float]] = {k: [] for k in ISSUES}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "h2h":
                continue
            for oc in mk.get("outcomes", []):
                name, price = oc.get("name"), oc.get("price")
                if price is None or price <= 1.0:
                    continue
                if name == OUTCOME_DRAW:
                    prices["draw"].append(float(price))
                elif name == home_raw:
                    prices["home"].append(float(price))
                elif name == away_raw:
                    prices["away"].append(float(price))

    if any(len(prices[k]) == 0 for k in ISSUES):
        return None  # cotes incomplètes pour ce match

    consensus = {k: statistics.median(prices[k]) for k in ISSUES}
    return home, away, match_date, consensus, prices


def run(
    odds_dir: Path = DEFAULT_ODDS_DIR,
    model_full: Path = DEFAULT_MODEL_FULL,
    matches_path: Path = DEFAULT_MATCHES,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    resolver: TeamResolver | None = None,
    now: datetime | None = None,
):
    snap = latest_snapshot(odds_dir)
    if snap is None:
        print("Aucun snapshot de cotes — rien à scorer (normal hors saison).")
        return []
    events = json.loads(snap.read_text(encoding="utf-8"))
    print(f"Snapshot : {snap.name} ({len(events)} matchs)")
    if not events:
        print("Snapshot vide (intersaison) — rien à scorer.")
        return []

    if not Path(model_full).exists():
        raise FileNotFoundError(
            f"Modèle complet introuvable : {model_full}. "
            f"Lancer d'abord fit_bayes (workflow Modèle bayésien)."
        )
    params = BayesParams.from_dict(
        json.loads(Path(model_full).read_text(encoding="utf-8"))
    )
    resolver = resolver or TeamResolver.from_csv()
    counts = _team_match_counts(matches_path)
    known = set(params.teams)

    match_scores = []
    match_dates: dict[tuple[str, str], str] = {}
    n_unresolved = 0
    n_unknown_model = 0

    for ev in events:
        parsed = _extract_odds(ev, resolver)
        if parsed is None:
            n_unresolved += 1
            continue
        home, away, mdate, consensus, prices = parsed
        if home not in known or away not in known:
            n_unknown_model += 1  # équipe absente du modèle (promu sans historique)
            continue
        pred = predict_1x2_bayes(params, home, away, n_samples=300)
        ms = score_match(home, away, pred, consensus,
                         book_prices=prices, n_matches=counts)
        match_scores.append(ms)
        match_dates[(home, away)] = mdate

    print(f"Matchs scorés : {len(match_scores)}")
    if n_unresolved:
        print(f"  Non résolus (noms/cotes incomplètes) : {n_unresolved}")
    if n_unknown_model:
        print(f"  Hors modèle (équipe sans historique) : {n_unknown_model}")

    # Journalisation des paris dépassant le seuil.
    added = record_bets(match_scores, match_dates, threshold=threshold, now=now)
    print(f"\nNouveaux paris journalisés (score ≥ {threshold}) : {len(added)}")
    for r in added:
        print(f"  {r.match_date}  {r.home} vs {r.away}  [{r.issue}]  "
              f"cote={r.odds}  EV={r.ev:+.3f}  score={r.score:.3f}")

    # Aperçu des meilleurs scores du run (même non journalisés).
    if match_scores:
        ranked = sorted(
            (s for ms in match_scores for s in ms.issues.values()),
            key=lambda s: s.score, reverse=True,
        )[:5]
        print("\nTop opportunités du run :")
        for s in ranked:
            print(f"  [{s.issue}] EV={s.ev:+.3f} signal={s.signal:+.3f} "
                  f"fiab={s.reliability:.3f} score={s.score:.3f}")

    return match_scores


if __name__ == "__main__":  # pragma: no cover
    run()
