"""Résolution des noms d'équipes vers une forme canonique.

Vit dans common/ car utilisé à la fois par l'ingestion (étage 1) et la
consolidation (étage 2). C'est le SEUL module transverse autorisé avec io.py
et dates.py.

La logique métier est volontairement minimale : tout repose sur la table de
correspondance config/team_mapping.csv, qui est la source de vérité, éditée à
la main. Ce module ne "devine" jamais un nom — si un nom est inconnu, il lève
une erreur explicite. C'est voulu : un nom non résolu doit casser bruyamment,
jamais disparaître en silence (cf. ARCHITECTURE.md, "le bug du match perdu").
"""

from __future__ import annotations

import csv
from pathlib import Path

# Les colonnes du mapping qui correspondent à une source de données.
# (toutes les colonnes sauf 'canonical' et la colonne d'état 'verified')
SOURCE_COLUMNS = ("footballdata", "clubelo", "fbref", "oddsapi")

DEFAULT_MAPPING_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "team_mapping.csv"
)


class UnknownTeamError(KeyError):
    """Levée quand un nom d'équipe n'existe pas dans la table pour une source.

    On la fait dériver de KeyError mais avec un message explicite, pour qu'un
    nom non résolu soit immédiatement visible et traçable, jamais avalé.
    """


class TeamResolver:
    """Résout les noms d'une source donnée vers leur forme canonique.

    Usage :
        resolver = TeamResolver.from_csv()
        resolver.to_canonical("Paris Saint-Germain", source="fbref")  -> "Paris SG"
    """

    def __init__(self, lookup: dict[str, dict[str, str]], canonicals: set[str]):
        # lookup[source][nom_normalisé_source] = nom_canonique
        self._lookup = lookup
        self._canonicals = canonicals

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_MAPPING_PATH) -> "TeamResolver":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Table de mapping introuvable : {path}")

        lookup: dict[str, dict[str, str]] = {src: {} for src in SOURCE_COLUMNS}
        canonicals: set[str] = set()

        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            _validate_header(reader.fieldnames, path)

            for line_no, row in enumerate(reader, start=2):
                canonical = _clean(row["canonical"])
                if not canonical:
                    raise ValueError(
                        f"{path}:{line_no} : colonne 'canonical' vide."
                    )
                if canonical in canonicals:
                    raise ValueError(
                        f"{path}:{line_no} : nom canonique en double "
                        f"'{canonical}'."
                    )
                canonicals.add(canonical)

                for src in SOURCE_COLUMNS:
                    raw = _clean(row.get(src, ""))
                    if not raw:
                        # Cellule vide = cette source n'a pas encore été mappée
                        # pour cette équipe (ex. oddsapi hors saison). Toléré.
                        continue
                    key = _normalize_key(raw)
                    if key in lookup[src]:
                        raise ValueError(
                            f"{path}:{line_no} : le nom source '{raw}' "
                            f"({src}) est déjà mappé vers "
                            f"'{lookup[src][key]}'."
                        )
                    lookup[src][key] = canonical

        return cls(lookup=lookup, canonicals=canonicals)

    # ------------------------------------------------------------------ #
    # Résolution
    # ------------------------------------------------------------------ #
    def to_canonical(self, name: str, source: str) -> str:
        """Renvoie le nom canonique pour un nom brut d'une source donnée.

        Lève UnknownTeamError si le nom n'est pas dans la table : un nom non
        résolu ne doit jamais passer silencieusement.
        """
        if source not in self._lookup:
            raise ValueError(
                f"Source inconnue : '{source}'. "
                f"Sources valides : {', '.join(SOURCE_COLUMNS)}."
            )
        key = _normalize_key(name)
        try:
            return self._lookup[source][key]
        except KeyError:
            raise UnknownTeamError(
                f"Nom d'équipe non mappé pour la source '{source}' : "
                f"'{name}'. Ajouter une ligne dans team_mapping.csv."
            ) from None

    def known_names(self, source: str) -> set[str]:
        """Noms bruts connus pour une source (utile aux tests/diagnostics)."""
        if source not in self._lookup:
            raise ValueError(f"Source inconnue : '{source}'.")
        return set(self._lookup[source].keys())

    @property
    def canonicals(self) -> set[str]:
        return set(self._canonicals)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _clean(value: str | None) -> str:
    """Retire les espaces parasites en début/fin (fréquents dans ces CSV)."""
    return (value or "").strip()


def _normalize_key(name: str) -> str:
    """Clé de comparaison tolérante aux variations d'espaces et de casse.

    On normalise UNIQUEMENT pour la comparaison (espaces multiples, casse).
    On ne touche pas aux accents ni aux tirets : ce sont des distinctions
    réelles entre sources, gérées explicitement par la table, pas devinées.
    """
    return " ".join(name.split()).casefold()


def _validate_header(fieldnames, path) -> None:
    if fieldnames is None:
        raise ValueError(f"{path} : fichier vide ou sans en-tête.")
    missing = {"canonical", *SOURCE_COLUMNS} - set(fieldnames)
    if missing:
        raise ValueError(
            f"{path} : colonnes manquantes dans l'en-tête : "
            f"{', '.join(sorted(missing))}."
        )
