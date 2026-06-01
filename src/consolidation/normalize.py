"""Étage 2 — Normalisation des noms d'équipes dans les données brutes.

Prend les résultats bruts d'une source (pour l'instant football-data.co.uk) et
remplace les noms d'équipes par leur forme canonique, via TeamResolver.

Contrat de sortie : mêmes lignes, mêmes colonnes, mais les colonnes d'équipes
contiennent des noms canoniques. Aucune ligne n'est silencieusement supprimée :
si un nom ne se résout pas, on lève une erreur (cf. ARCHITECTURE.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass

from common.teams import TeamResolver, UnknownTeamError


@dataclass
class NormalizationReport:
    """Bilan d'une passe de normalisation, pour traçabilité et tests."""

    rows_in: int
    rows_out: int
    unresolved: list[tuple[int, str, str]]  # (ligne, nom, source)

    @property
    def ok(self) -> bool:
        return not self.unresolved and self.rows_in == self.rows_out


def normalize_matches(
    rows: list[dict[str, str]],
    resolver: TeamResolver,
    source: str,
    home_col: str = "HomeTeam",
    away_col: str = "AwayTeam",
    strict: bool = True,
) -> tuple[list[dict[str, str]], NormalizationReport]:
    """Normalise les noms d'équipes domicile/extérieur d'une liste de matchs.

    Args:
        rows: lignes brutes (ex. issues d'un csv.DictReader sur F1.csv).
        resolver: TeamResolver chargé depuis team_mapping.csv.
        source: clé de source ('footballdata', 'clubelo', ...).
        home_col / away_col: noms de colonnes d'équipes dans la source.
        strict: si True (défaut), lève dès le premier nom non résolu. Si False,
                collecte tous les non-résolus dans le rapport sans interrompre
                (utile pour diagnostiquer un mapping incomplet d'un coup).

    Returns:
        (lignes_normalisées, rapport). En mode strict, ne renvoie que si tout
        s'est résolu.
    """
    out: list[dict[str, str]] = []
    unresolved: list[tuple[int, str, str]] = []

    for line_no, row in enumerate(rows, start=2):
        new_row = dict(row)
        for col in (home_col, away_col):
            raw = (row.get(col) or "").strip()
            if not raw:
                unresolved.append((line_no, f"<colonne {col} vide>", source))
                continue
            try:
                new_row[col] = resolver.to_canonical(raw, source=source)
            except UnknownTeamError:
                if strict:
                    raise
                unresolved.append((line_no, raw, source))
        out.append(new_row)

    report = NormalizationReport(
        rows_in=len(rows), rows_out=len(out), unresolved=unresolved
    )
    return out, report
