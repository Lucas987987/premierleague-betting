"""Scoring Over/Under — EV + Signal × Fiabilité par seuil coté.

Réutilise les briques de value.py (devig, composantes de fiabilité). Un marché
over/under est un marché à 2 issues (over, under) ; la mécanique est celle du
1/N/2, juste sur 2 issues. La Fiabilité combine :
  - certitude bayésienne (largeur de l'intervalle de crédibilité de la proba)
  - convergence des books (dispersion des cotes over/under)
  - qualité des données (nombre de matchs des deux équipes)

Fiabilité = rel_ci × rel_books × rel_data (produit direct, SANS plancher) —
identique à score_match (1/N/2) pour que les scores des deux marchés soient
comparables sur le même repo.

On ne score QUE les seuils réellement cotés (EV/Signal impossibles sans cote).
"""
from __future__ import annotations
from dataclasses import dataclass
from scoring.value import (
    devig, reliability_from_ci, reliability_from_books, reliability_from_data,
    MATCHES_FULL,
)


@dataclass
class OUIssueScore:
    threshold: float
    side: str            # 'over' | 'under'
    p_model: float
    p_market: float
    odds: float
    ev: float
    signal: float
    reliability: float
    score: float


@dataclass
class OUMatchScore:
    home: str
    away: str
    issues: list[OUIssueScore]

    def best_by_score(self):
        return max(self.issues, key=lambda s: s.score) if self.issues else None

    def best_by_ev(self):
        return max(self.issues, key=lambda s: s.ev) if self.issues else None


def score_over_under(home, away, ou_pred, ou_odds, n_matches=None):
    """Score chaque seuil présent À LA FOIS dans le modèle et dans les cotes."""
    n_home = (n_matches or {}).get(home, MATCHES_FULL)
    n_away = (n_matches or {}).get(away, MATCHES_FULL)
    rel_data = reliability_from_data(n_home, n_away)

    issues = []
    for seuil in sorted(set(ou_pred) & set(ou_odds)):
        pred = ou_pred[seuil]
        odds = ou_odds[seuil]
        p_market = devig({"over": odds["over"], "under": odds["under"]})

        for side in ("over", "under"):
            p_m = pred[side]
            p_k = p_market[side]
            cote = odds[side]
            ev = p_m * cote - 1.0
            signal = (p_m - p_k) / p_k if p_k > 0 else 0.0

            ci = pred.get(f"{side}_ci")
            rel_ci = reliability_from_ci(ci["lo"], ci["hi"]) if ci else 0.5
            prices = odds.get(f"{side}_prices", [])
            rel_books = reliability_from_books(prices)
            reliability = rel_ci * rel_books * rel_data

            score = max(signal, 0.0) * reliability
            issues.append(OUIssueScore(
                threshold=seuil, side=side,
                p_model=p_m, p_market=p_k, odds=cote,
                ev=ev, signal=signal, reliability=reliability, score=score,
            ))

    return OUMatchScore(home=home, away=away, issues=issues)
