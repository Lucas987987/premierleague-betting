"""Journal de paris théoriques — couche immuable pour la mesure du CLV.

Le scoring repère des opportunités (issues dont le score Signal×Fiabilité dépasse
un seuil). Pour mesurer honnêtement le CLV, on inscrit ces paris AVANT la clôture,
avec la cote du moment et l'horodatage, et on n'y touche plus jamais.

Pourquoi automatique et non manuel : on veut valider le MODÈLE, pas le flair
humain. Un journal auto, non biaisé par des décisions a posteriori, répond à la
vraie question : "le scoring lui-même repère-t-il de la valeur ?"

Discipline temporelle (cruciale) :
  - Un pari est inscrit au moment où le scoring le repère (cote précoce).
  - Dédup : si le même pari (match+issue) ressort à un run ultérieur, on GARDE
    la PREMIÈRE occurrence — la cote la plus précoce est la plus intéressante
    pour le CLV (on veut battre la clôture, donc parier tôt).
  - Immuable : une ligne inscrite n'est jamais réécrite.

Le journal est un CSV append-only : data/bets/bet_log.csv.
Identité d'un pari = (date_match, home, away, issue) — indépendante du run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = _ROOT / "data" / "bets" / "bet_log.csv"

# Seuil de score au-delà duquel un pari théorique est inscrit. Réglable.
# Volontairement bas au début : on veut un échantillon pour mesurer le CLV,
# pas (encore) une sélection agressive. Le CLV dira si ce seuil est pertinent.
DEFAULT_SCORE_THRESHOLD = 0.05

FIELDS = [
    "logged_at_utc",   # quand le pari a été inscrit (horodatage du run)
    "match_date",      # date du match (issue du scoring / cotes)
    "home", "away", "issue",
    "odds",            # cote au moment de l'inscription (cote précoce)
    "p_model", "p_market", "ev", "signal", "reliability", "score",
]


@dataclass(frozen=True)
class BetRecord:
    logged_at_utc: str
    match_date: str
    home: str
    away: str
    issue: str
    odds: float
    p_model: float
    p_market: float
    ev: float
    signal: float
    reliability: float
    score: float

    @property
    def identity(self) -> tuple[str, str, str, str]:
        """Identité stable d'un pari, indépendante du run et de la cote."""
        return (self.match_date, self.home, self.away, self.issue)


# ---------------------------------------------------------------------- #
# Lecture du journal existant
# ---------------------------------------------------------------------- #
def load_log(path: Path = DEFAULT_LOG) -> list[BetRecord]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(BetRecord(
                logged_at_utc=row["logged_at_utc"],
                match_date=row["match_date"],
                home=row["home"], away=row["away"], issue=row["issue"],
                odds=float(row["odds"]),
                p_model=float(row["p_model"]), p_market=float(row["p_market"]),
                ev=float(row["ev"]), signal=float(row["signal"]),
                reliability=float(row["reliability"]), score=float(row["score"]),
            ))
    return out


def existing_identities(path: Path = DEFAULT_LOG) -> set[tuple[str, str, str, str]]:
    return {r.identity for r in load_log(path)}


# ---------------------------------------------------------------------- #
# Inscription
# ---------------------------------------------------------------------- #
def record_bets(
    match_scores: list,                # list[MatchScore] du module value
    match_dates: dict[tuple[str, str], str],  # (home,away) -> date du match
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    log_path: Path = DEFAULT_LOG,
    now: datetime | None = None,
) -> list[BetRecord]:
    """Inscrit les paris dont le score dépasse le seuil, en évitant les doublons.

    Retourne la liste des paris EFFECTIVEMENT ajoutés (hors doublons déjà connus).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    seen = existing_identities(log_path)

    new_records: list[BetRecord] = []
    for ms in match_scores:
        md = match_dates.get((ms.home, ms.away), "")
        for issue, s in ms.issues.items():
            if s.score < threshold:
                continue
            rec = BetRecord(
                logged_at_utc=stamp, match_date=md,
                home=ms.home, away=ms.away, issue=issue,
                odds=round(s.odds, 4), p_model=round(s.p_model, 4),
                p_market=round(s.p_market, 4), ev=round(s.ev, 4),
                signal=round(s.signal, 4), reliability=round(s.reliability, 4),
                score=round(s.score, 4),
            )
            if rec.identity in seen:
                continue  # déjà inscrit (occurrence plus précoce conservée)
            seen.add(rec.identity)
            new_records.append(rec)

    if new_records:
        _append(log_path, new_records)
    return new_records


def _append(path: Path, records: list[BetRecord]) -> None:
    """Ajoute des lignes au journal (append-only). Écrit l'en-tête si nouveau."""
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        for r in records:
            w.writerow(asdict(r))
