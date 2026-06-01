"""Étage scoring — EV + Signal × Fiabilité, pour chaque match à venir.

Croise la prédiction bayésienne (proba + intervalle de crédibilité) avec les
cotes live (The Odds API) et produit, POUR CHAQUE ISSUE (1/N/2) :

  - EV       = p_modèle × cote − 1   (combien le pari paie en espérance)
  - Signal   = (p_modèle − p_marché) / p_marché   (divergence relative au marché)
  - Fiabilité ∈ [0,1] = certitude bayésienne × convergence books × qualité données
  - Score    = Signal × Fiabilité   (signal tempéré par la confiance)

Philosophie (héritée du projet tennis) : Signal et Fiabilité sont SÉPARÉS et
indépendants de l'EV. On veut pouvoir inspecter chaque composante :
  - un score faible vient-il d'un petit écart (Signal) ou d'une grande
    incertitude (Fiabilité) ? On garde les deux visibles.
  - l'EV dit "combien ça paie", le Score dit "à quel point j'y crois".

L'apport du bayésien : la Fiabilité intègre l'intervalle de crédibilité. Un gros
signal sur un match incertain (intervalle large, ex. promu) est tempéré.

Ce module NE décide PAS de parier. Il classe et éclaire. La décision (seuils,
mise) viendra après, informée par le CLV.
"""

from __future__ import annotations

from dataclasses import dataclass

# Bornes de normalisation (réglables). Choisies pour que la fiabilité soit
# discriminante sur les ordres de grandeur réels observés.
CI_WIDTH_FULL = 0.06      # intervalle <= 6 pts => certitude ~1
CI_WIDTH_ZERO = 0.35      # intervalle >= 35 pts => certitude ~0
BOOK_SPREAD_FULL = 0.02   # books à <=2 pts d'écart => convergence ~1
BOOK_SPREAD_ZERO = 0.15   # books à >=15 pts d'écart => convergence ~0
MATCHES_FULL = 60         # >=60 matchs sur l'équipe => qualité données ~1
MATCHES_ZERO = 10         # <=10 matchs => qualité ~0

ISSUES = ("home", "draw", "away")


@dataclass
class IssueScore:
    issue: str
    p_model: float
    p_market: float
    odds: float
    ev: float
    signal: float
    reliability: float
    score: float


@dataclass
class MatchScore:
    home: str
    away: str
    issues: dict[str, IssueScore]

    def best_by_score(self) -> IssueScore:
        return max(self.issues.values(), key=lambda s: s.score)

    def best_by_ev(self) -> IssueScore:
        return max(self.issues.values(), key=lambda s: s.ev)


# ---------------------------------------------------------------------- #
# Dévigorisation du marché
# ---------------------------------------------------------------------- #
def devig(odds: dict[str, float]) -> dict[str, float]:
    """Probas implicites dévigorisées (méthode proportionnelle)."""
    inv = {k: 1.0 / v for k, v in odds.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


# ---------------------------------------------------------------------- #
# Composantes de la fiabilité (chacune dans [0,1])
# ---------------------------------------------------------------------- #
def _ramp(x: float, full: float, zero: float) -> float:
    """Rampe linéaire : 1 quand x<=full, 0 quand x>=zero, interpolé entre.
    (full < zero attendu.)"""
    if x <= full:
        return 1.0
    if x >= zero:
        return 0.0
    return (zero - x) / (zero - full)


def reliability_from_ci(ci_lo: float, ci_hi: float) -> float:
    """Certitude bayésienne : intervalle étroit => proche de 1."""
    return _ramp(ci_hi - ci_lo, CI_WIDTH_FULL, CI_WIDTH_ZERO)


def reliability_from_books(book_prices: list[float]) -> float:
    """Convergence bookmakers : cotes serrées => proche de 1.
    On mesure la dispersion en probabilité implicite (1/cote)."""
    if len(book_prices) < 2:
        return 0.5  # un seul book : information faible, fiabilité neutre
    imp = [1.0 / p for p in book_prices if p > 1.0]
    if len(imp) < 2:
        return 0.5
    spread = max(imp) - min(imp)
    return _ramp(spread, BOOK_SPREAD_FULL, BOOK_SPREAD_ZERO)


def reliability_from_data(n_home: int, n_away: int) -> float:
    """Qualité des données : assez de matchs sur les DEUX équipes => proche de 1.
    On prend le minimum des deux (l'équipe la moins connue limite la confiance)."""
    n = min(n_home, n_away)
    return _ramp(-n, -MATCHES_FULL, -MATCHES_ZERO)  # rampe croissante via négation


# ---------------------------------------------------------------------- #
# Scoring d'un match
# ---------------------------------------------------------------------- #
def score_match(
    home: str,
    away: str,
    model_pred: dict,          # {'home','draw','away', 'home_ci':{lo,hi}, ...}
    consensus_odds: dict[str, float],   # cote retenue par issue (ex. meilleure ou médiane)
    book_prices: dict[str, list[float]] | None = None,  # cotes par book et par issue
    n_matches: dict[str, int] | None = None,  # {home: n, away: n}
) -> MatchScore:
    """Calcule EV, Signal, Fiabilité et Score pour les 3 issues d'un match."""
    p_market = devig(consensus_odds)
    n_home = (n_matches or {}).get(home, MATCHES_FULL)
    n_away = (n_matches or {}).get(away, MATCHES_FULL)
    rel_data = reliability_from_data(n_home, n_away)

    issues: dict[str, IssueScore] = {}
    for issue in ISSUES:
        p_m = model_pred[issue]
        p_k = p_market[issue]
        odds = consensus_odds[issue]

        ev = p_m * odds - 1.0
        signal = (p_m - p_k) / p_k if p_k > 0 else 0.0

        ci = model_pred.get(f"{issue}_ci")
        rel_ci = reliability_from_ci(ci["lo"], ci["hi"]) if ci else 0.5
        rel_books = (
            reliability_from_books(book_prices[issue])
            if book_prices and issue in book_prices else 0.5
        )
        reliability = rel_ci * rel_books * rel_data

        # Le score ne récompense que les signaux POSITIFS (value côté joueur) :
        # un signal négatif (modèle sous le marché) n'est pas une opportunité.
        score = max(signal, 0.0) * reliability

        issues[issue] = IssueScore(
            issue=issue, p_model=p_m, p_market=p_k, odds=odds,
            ev=ev, signal=signal, reliability=reliability, score=score,
        )

    return MatchScore(home=home, away=away, issues=issues)
