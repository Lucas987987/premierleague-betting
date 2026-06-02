"""Orchestrateur du scoring — chaîne complète pour les matchs à venir.

Enchaîne : dernier snapshot de cotes (Odds API) → résolution des noms →
rechargement du modèle bayésien figé → prédiction 1/N/2 + over/under avec
intervalles → scoring EV + Signal×Fiabilité → journalisation (2 journaux séparés).

Ne réajuste PAS le modèle (découplage fit/predict). Hors saison : snapshot vide
→ rien à scorer. Agrégation des cotes : médiane des books par issue, liste
complète conservée pour la convergence (fiabilité).
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
from scoring.totals import predict_over_under
from scoring.totals_odds import extract_totals
from scoring.score_totals import score_over_under
from scoring.betlog_ou import record_bets_ou

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ODDS_DIR = _ROOT / "data" / "raw" / "oddsapi"
DEFAULT_MODEL_FULL = _ROOT / "data" / "model" / "params_bayes_full.json"
DEFAULT_MATCHES = _ROOT / "data" / "processed" / "matches.csv"
OUTCOME_DRAW = "Draw"


def latest_snapshot(odds_dir):
    snaps = sorted(Path(odds_dir).glob("odds_*.json"))
    return snaps[-1] if snaps else None


def _team_match_counts(matches_path):
    counts = {}
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


def _extract_odds(event, resolver):
    home_raw = event.get("home_team", "")
    away_raw = event.get("away_team", "")
    try:
        home = resolver.to_canonical(home_raw, "oddsapi")
        away = resolver.to_canonical(away_raw, "oddsapi")
    except UnknownTeamError:
        return None
    commence = event.get("commence_time", "")
    match_date = commence[:10] if commence else ""
    prices = {k: [] for k in ISSUES}
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
        return None
    consensus = {k: statistics.median(prices[k]) for k in ISSUES}
    return home, away, match_date, consensus, prices


def run(odds_dir=DEFAULT_ODDS_DIR, model_full=DEFAULT_MODEL_FULL,
        matches_path=DEFAULT_MATCHES, threshold=DEFAULT_SCORE_THRESHOLD,
        resolver=None, now=None):
    snap = latest_snapshot(odds_dir)
    if snap is None:
        print("Aucun snapshot de cotes — rien à scorer (normal hors saison).")
        return [], []
    events = json.loads(snap.read_text(encoding="utf-8"))
    print(f"Snapshot : {snap.name} ({len(events)} matchs)")
    if not events:
        print("Snapshot vide (intersaison) — rien à scorer.")
        return [], []
    if not Path(model_full).exists():
        raise FileNotFoundError(
            f"Modèle complet introuvable : {model_full}. Lancer d'abord fit_bayes.")
    params = BayesParams.from_dict(json.loads(Path(model_full).read_text(encoding="utf-8")))
    resolver = resolver or TeamResolver.from_csv()
    counts = _team_match_counts(matches_path)
    known = set(params.teams)
    match_scores = []
    ou_match_scores = []
    match_dates = {}
    n_unresolved = 0
    n_unknown_model = 0
    for ev in events:
        parsed = _extract_odds(ev, resolver)
        if parsed is None:
            n_unresolved += 1
            continue
        home, away, mdate, consensus, prices = parsed
        if home not in known or away not in known:
            n_unknown_model += 1
            continue
        pred = predict_1x2_bayes(params, home, away, n_samples=300)
        ms = score_match(home, away, pred, consensus, book_prices=prices, n_matches=counts)
        match_scores.append(ms)
        match_dates[(home, away)] = mdate
        # Over/Under : cotes totals du même événement + prédiction du modèle.
        ou_odds = extract_totals(ev)
        if ou_odds:
            ou_pred = predict_over_under(params, home, away, n_samples=300)
            ou_ms = score_over_under(home, away, ou_pred, ou_odds, n_matches=counts)
            if ou_ms.issues:
                ou_match_scores.append(ou_ms)
    print(f"Matchs scorés (1/N/2) : {len(match_scores)}")
    print(f"Matchs avec over/under coté : {len(ou_match_scores)}")
    if n_unresolved:
        print(f"  Non résolus (noms/cotes incomplètes) : {n_unresolved}")
    if n_unknown_model:
        print(f"  Hors modèle (équipe sans historique) : {n_unknown_model}")
    added = record_bets(match_scores, match_dates, threshold=threshold, now=now)
    print(f"\nNouveaux paris 1/N/2 journalisés (score >= {threshold}) : {len(added)}")
    for r in added:
        print(f"  {r.match_date}  {r.home} vs {r.away}  [{r.issue}]  cote={r.odds}  EV={r.ev:+.3f}  score={r.score:.3f}")
    added_ou = record_bets_ou(ou_match_scores, match_dates, threshold=threshold, now=now)
    print(f"\nNouveaux paris over/under journalisés (score >= {threshold}) : {len(added_ou)}")
    for r in added_ou:
        print(f"  {r.match_date}  {r.home} vs {r.away}  [{r.side} {r.threshold}]  cote={r.odds}  EV={r.ev:+.3f}  score={r.score:.3f}")
    if match_scores:
        ranked = sorted((s for ms in match_scores for s in ms.issues.values()),
                        key=lambda s: s.score, reverse=True)[:5]
        print("\nTop opportunités 1/N/2 du run :")
        for s in ranked:
            print(f"  [{s.issue}] EV={s.ev:+.3f} signal={s.signal:+.3f} fiab={s.reliability:.3f} score={s.score:.3f}")
    return match_scores, ou_match_scores


if __name__ == "__main__":  # pragma: no cover
    run()
