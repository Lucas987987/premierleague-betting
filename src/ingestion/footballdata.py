"""Étage 1 — Ingestion football-data.co.uk (Premier League).

Télécharge les CSV de résultats + cotes de la Premier League (code `E0`) et les
écrit dans data/raw/footballdata/ en respectant l'IMMUTABILITÉ de la couche raw
(cf. ARCHITECTURE.md §0 et discipline ELT raw/staging/marts) :

  - Chaque téléchargement = un fichier horodaté en UTC, JAMAIS réécrit.
    Ex. E0_2526_20260531T143005Z.csv
  - Refus d'écrasement : si un nom de fichier existe déjà, on lève une erreur.
    L'immutabilité est garantie par le code, pas par la discipline humaine.
  - Déduplication par hash : si le contenu téléchargé est identique (octet pour
    octet) à la dernière capture de la même saison, on ne crée pas de doublon.

Pourquoi c'est vital : la couche raw est la source de vérité rejouable. Pour le
CLV (plus tard, sur les cotes), une capture est une preuve historique qu'on ne
doit jamais altérer. On pose la discipline ici, on la réutilisera pour les cotes.

Ce module NE normalise PAS les noms d'équipes : le raw reste brut (noms tels
que football-data les écrit, typos comprises). La normalisation est l'affaire
de l'étage 2 (consolidation/normalize.py).

Réseau requis. En l'absence de réseau, fournir `downloader=` (injection) pour
les tests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request

# Code football-data de la Premier League.
LEAGUE_CODE = "E0"
BASE_URL = "https://www.football-data.co.uk/mmz4281"

DEFAULT_RAW_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "footballdata"
)

# User-agent explicite : courtoisie minimale envers le serveur.
_USER_AGENT = "premierleague-betting/0.1 (ingestion footballdata)"


class RawImmutabilityError(FileExistsError):
    """Levée si on tente d'écraser un fichier de la couche raw."""


@dataclass(frozen=True)
class IngestionResult:
    """Résultat d'une ingestion d'une saison."""

    season: str                 # code saison, ex. "2526"
    path: Path | None           # fichier écrit, ou None si dédupliqué
    sha256: str                 # hash du contenu téléchargé
    deduplicated: bool          # True si identique à la dernière capture
    n_bytes: int


# ---------------------------------------------------------------------- #
# Calcul des saisons
# ---------------------------------------------------------------------- #
def recent_seasons(n: int = 5, today: datetime | None = None) -> list[str]:
    """Renvoie les `n` derniers codes saison football-data, ex. ['2122', ...].

    Une saison de Premier League va d'août à mai. Convention football-data : la
    saison 2025-26 est codée '2526'. On considère qu'une saison "démarre" en
    juillet : avant juillet, la saison courante est (année-1)/année.
    """
    today = today or datetime.now(timezone.utc)
    # Année de début de la saison courante.
    start_year = today.year if today.month >= 7 else today.year - 1
    seasons = []
    for k in range(n):
        y0 = start_year - k
        y1 = y0 + 1
        seasons.append(f"{y0 % 100:02d}{y1 % 100:02d}")
    return list(reversed(seasons))  # de la plus ancienne à la plus récente


def season_url(season: str) -> str:
    return f"{BASE_URL}/{season}/{LEAGUE_CODE}.csv"


# ---------------------------------------------------------------------- #
# Téléchargement (injectable pour les tests)
# ---------------------------------------------------------------------- #
def _default_downloader(url: str, timeout: int = 30) -> bytes:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL maîtrisée)
        return resp.read()


# ---------------------------------------------------------------------- #
# Helpers immutabilité
# ---------------------------------------------------------------------- #
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_stamp(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def _capture_name(season: str, stamp: str) -> str:
    return f"{LEAGUE_CODE}_{season}_{stamp}.csv"


def _latest_capture(raw_dir: Path, season: str) -> Path | None:
    """Dernière capture connue pour une saison (par ordre de nom = ordre temps)."""
    pattern = f"{LEAGUE_CODE}_{season}_*.csv"
    captures = sorted(raw_dir.glob(pattern))
    return captures[-1] if captures else None


# ---------------------------------------------------------------------- #
# Ingestion
# ---------------------------------------------------------------------- #
def ingest_season(
    season: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    downloader: Callable[[str], bytes] = _default_downloader,
    now: datetime | None = None,
) -> IngestionResult:
    """Télécharge une saison et l'écrit en respectant l'immutabilité.

    - Si le contenu est identique à la dernière capture : pas de nouveau fichier
      (deduplicated=True, path=None).
    - Sinon : écrit un fichier horodaté. Refuse d'écraser un nom existant.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    data = downloader(season_url(season))
    if not data or not data.strip():
        raise ValueError(f"Téléchargement vide pour la saison {season}.")
    digest = _sha256(data)

    # Déduplication : comparer au hash de la dernière capture.
    latest = _latest_capture(raw_dir, season)
    if latest is not None and _sha256(latest.read_bytes()) == digest:
        return IngestionResult(
            season=season,
            path=None,
            sha256=digest,
            deduplicated=True,
            n_bytes=len(data),
        )

    stamp = _utc_stamp(now)
    target = raw_dir / _capture_name(season, stamp)

    # Garde-fou d'immutabilité : ne jamais écraser.
    if target.exists():
        raise RawImmutabilityError(
            f"Refus d'écrasement (couche raw immuable) : {target.name} "
            f"existe déjà."
        )

    target.write_bytes(data)
    return IngestionResult(
        season=season,
        path=target,
        sha256=digest,
        deduplicated=False,
        n_bytes=len(data),
    )


def ingest_recent(
    n_seasons: int = 5,
    raw_dir: Path = DEFAULT_RAW_DIR,
    downloader: Callable[[str], bytes] = _default_downloader,
    now: datetime | None = None,
) -> list[IngestionResult]:
    """Ingestion des `n_seasons` dernières saisons (V1 : 5)."""
    results = []
    for season in recent_seasons(n_seasons, today=now):
        results.append(
            ingest_season(season, raw_dir=raw_dir, downloader=downloader, now=now)
        )
    return results


if __name__ == "__main__":  # pragma: no cover
    for res in ingest_recent():
        if res.deduplicated:
            print(f"{res.season} : inchangé (hash {res.sha256[:12]}…)")
        else:
            print(f"{res.season} : écrit {res.path.name} ({res.n_bytes} o)")
