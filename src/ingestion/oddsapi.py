"""Étage 1 — Ingestion The Odds API (cotes live Premier League).

Interroge l'endpoint /odds de The Odds API pour les matchs Premier League à venir
et capture chaque snapshot de cotes 1X2 (marché h2h, région eu) de façon IMMUABLE :
chaque appel = un fichier JSON horodaté en UTC, jamais réécrit, dédupliqué par
hash (cf. discipline raw football-data).

Pourquoi capturer des snapshots répétés : pour le CLV, on veut suivre l'évolution
de la cote entre l'ouverture et le coup d'envoi. Chaque snapshot est une photo
horodatée du marché à un instant donné — une preuve qu'on n'altère jamais.

Coût quota : marché h2h + région eu = 1 crédit par appel. Un appel ramène TOUS
les matchs Premier League à venir. Avec 5 clés × 500 crédits/mois, large marge.

Note : The Odds API ne fournit PAS Pinnacle. La closing line Pinnacle vient de
football-data (PSCH/PSCD/PSCA). Ici on capture les cotes grand public au moment
du "pari" ; le CLV croisera les deux sources.

Clé API : lue depuis la variable d'environnement ODDS_API_KEY (jamais en clair).
En GitHub Actions, fournie via un secret de repo.

Réseau requis. Downloader injectable pour les tests (comme footballdata).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SPORT_KEY = "soccer_epl"
REGION = "eu"
MARKET = "h2h,totals"  # h2h (1/N/2) + totals (over/under). 2 crédits/appel.
BASE_URL = "https://api.the-odds-api.com/v4"

DEFAULT_RAW_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "oddsapi"
)

_USER_AGENT = "premierleague-betting/0.1 (ingestion oddsapi)"
API_KEY_ENV = "ODDS_API_KEY"


class MissingApiKeyError(RuntimeError):
    """Levée si la clé API n'est pas disponible dans l'environnement."""


class RawImmutabilityError(FileExistsError):
    """Levée si on tente d'écraser un snapshot de cotes (couche raw immuable)."""


class OddsApiError(RuntimeError):
    """Erreur renvoyée par l'API (quota dépassé, clé invalide, etc.)."""


@dataclass(frozen=True)
class OddsSnapshot:
    path: Path | None         # fichier écrit, ou None si dédupliqué
    sha256: str
    deduplicated: bool
    n_events: int             # nombre de matchs dans le snapshot
    requests_remaining: str | None  # quota restant (en-tête API), si connu


# ---------------------------------------------------------------------- #
# Clé API
# ---------------------------------------------------------------------- #
def get_api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise MissingApiKeyError(
            f"Clé API absente. Définir la variable d'environnement "
            f"{API_KEY_ENV} (en Actions : secret de repo)."
        )
    return key


def odds_url(api_key: str) -> str:
    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": MARKET,
        "oddsFormat": "decimal",
    }
    return f"{BASE_URL}/sports/{SPORT_KEY}/odds?{urlencode(params)}"


# ---------------------------------------------------------------------- #
# Téléchargement (injectable pour tests)
# ---------------------------------------------------------------------- #
def _default_downloader(url: str, timeout: int = 30) -> tuple[bytes, dict]:
    """Renvoie (corps, en-têtes). Les en-têtes portent le quota restant."""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        body = resp.read()
        headers = {k.lower(): v for k, v in resp.getheaders()}
    return body, headers


# ---------------------------------------------------------------------- #
# Helpers immutabilité
# ---------------------------------------------------------------------- #
def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_stamp(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def _latest_snapshot(raw_dir: Path) -> Path | None:
    snaps = sorted(raw_dir.glob("odds_*.json"))
    return snaps[-1] if snaps else None


# ---------------------------------------------------------------------- #
# Validation de la réponse
# ---------------------------------------------------------------------- #
def _validate_payload(body: bytes) -> list:
    """La réponse /odds doit être une liste JSON d'événements. Détecte les
    erreurs API (qui arrivent en objet JSON avec un message)."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise OddsApiError(f"Réponse non-JSON : {body[:200]!r}") from e
    if isinstance(data, dict):
        # L'API renvoie un objet {"message": "..."} en cas d'erreur.
        msg = data.get("message") or str(data)
        raise OddsApiError(f"Erreur API : {msg}")
    if not isinstance(data, list):
        raise OddsApiError(f"Format inattendu : {type(data).__name__}")
    return data


# ---------------------------------------------------------------------- #
# Ingestion
# ---------------------------------------------------------------------- #
def ingest_odds(
    raw_dir: Path = DEFAULT_RAW_DIR,
    api_key: str | None = None,
    downloader: Callable[[str], tuple[bytes, dict]] = _default_downloader,
    now: datetime | None = None,
) -> OddsSnapshot:
    """Capture un snapshot des cotes Premier League, immuable et horodaté.

    - Valide que la réponse est bien une liste d'événements (sinon OddsApiError).
    - Si identique au dernier snapshot (hash), ne crée pas de doublon.
    - Refuse d'écraser un fichier existant.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    key = get_api_key(api_key)
    body, headers = downloader(odds_url(key))
    events = _validate_payload(body)  # lève si erreur API

    # On reformate en JSON canonique (clés triées) pour un hash stable :
    # deux snapshots au contenu identique donnent le même hash même si l'ordre
    # des clés varie d'un appel à l'autre.
    canonical = json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    digest = _sha256(canonical)
    remaining = headers.get("x-requests-remaining")

    latest = _latest_snapshot(raw_dir)
    if latest is not None:
        prev = json.loads(latest.read_text(encoding="utf-8"))
        prev_canon = json.dumps(
            prev, sort_keys=True, separators=(",", ":")
        ).encode()
        if _sha256(prev_canon) == digest:
            return OddsSnapshot(
                path=None, sha256=digest, deduplicated=True,
                n_events=len(events), requests_remaining=remaining,
            )

    stamp = _utc_stamp(now)
    target = raw_dir / f"odds_{stamp}.json"
    if target.exists():
        raise RawImmutabilityError(
            f"Refus d'écrasement (couche raw immuable) : {target.name}."
        )

    # On écrit la réponse telle quelle (lisible), pas la version canonique.
    target.write_text(
        json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return OddsSnapshot(
        path=target, sha256=digest, deduplicated=False,
        n_events=len(events), requests_remaining=remaining,
    )


if __name__ == "__main__":  # pragma: no cover
    snap = ingest_odds()
    if snap.deduplicated:
        print(f"Cotes inchangées ({snap.n_events} matchs). "
              f"Quota restant : {snap.requests_remaining}")
    else:
        print(f"Snapshot écrit : {snap.path.name} "
              f"({snap.n_events} matchs). Quota restant : {snap.requests_remaining}")
