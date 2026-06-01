"""Étage 2 — Consolidation : raw football-data → data/processed/matches.csv.

Pour chaque saison présente dans data/raw/footballdata/, prend la DERNIÈRE
capture (la plus récente, immuable), normalise les noms d'équipes via
TeamResolver, et fusionne tout en un seul matches.csv.

Décisions (cf. ARCHITECTURE.md + choix V1) :
  - TOUTES les colonnes football-data sont préservées (on filtrera plus tard,
    près du modèle). Les colonnes diffèrent entre saisons (ex. 'Referee' absent
    certaines années, nombre de cotes variable) : on prend l'UNION des colonnes,
    les cellules manquantes restent vides.
  - On ajoute 3 colonnes dérivées en tête : `season` (ex. '2425'),
    `HomeTeamCanonical`, `AwayTeamCanonical`. Les colonnes brutes HomeTeam/
    AwayTeam sont CONSERVÉES telles quelles (traçabilité : on voit le nom source).
  - Aucune ligne n'est silencieusement supprimée. Les lignes dont une équipe
    n'est pas résolue sont signalées dans le rapport. En mode non-strict (défaut
    ici), on garde la ligne avec un canonique vide pour pouvoir tout diagnostiquer
    d'un coup ; en strict, on lève.

Le but de ce module en V1 : confronter team_mapping.csv aux 5 vraies saisons.
Il va lister les promus/relégués manquants (Leeds, Sunderland, etc.) pour
qu'on complète la table — c'est le garde-fou anti "match perdu" en action.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from common.teams import TeamResolver, UnknownTeamError

LEAGUE_CODE = "E0"
HOME_COL = "HomeTeam"
AWAY_COL = "AwayTeam"
SOURCE = "footballdata"

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = _ROOT / "data" / "raw" / "footballdata"
DEFAULT_OUT = _ROOT / "data" / "processed" / "matches.csv"
DEFAULT_MAPPING = _ROOT / "config" / "team_mapping.csv"

DERIVED_COLS = ["season", "HomeTeamCanonical", "AwayTeamCanonical"]


@dataclass
class ConsolidationReport:
    seasons: list = field(default_factory=list)
    rows_per_season: dict = field(default_factory=dict)
    total_rows: int = 0
    unresolved_teams: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unresolved_teams

    def summary(self) -> str:
        lines = [
            f"Saisons consolidées : {', '.join(self.seasons)}",
            f"Total matchs : {self.total_rows}",
            "Matchs par saison : "
            + ", ".join(f"{s}={self.rows_per_season[s]}" for s in self.seasons),
            f"Colonnes (union) : {len(self.columns)}",
        ]
        if self.unresolved_teams:
            lines.append("")
            lines.append("⚠ ÉQUIPES NON RÉSOLUES (à ajouter dans team_mapping.csv) :")
            for name, count in sorted(
                self.unresolved_teams.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"   - {name!r}  ({count} matchs concernés)")
        else:
            lines.append("✓ Toutes les équipes sont résolues.")
        return "\n".join(lines)


def _season_from_filename(path: Path) -> str:
    return path.name.split("_")[1]


def latest_capture_per_season(raw_dir: Path) -> dict:
    by_season = {}
    for p in raw_dir.glob(f"{LEAGUE_CODE}_*.csv"):
        by_season.setdefault(_season_from_filename(p), []).append(p)
    return {season: sorted(paths)[-1] for season, paths in by_season.items()}


def _read_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = [
            r for r in reader
            if (r.get("Date") or "").strip() and (r.get(HOME_COL) or "").strip()
        ]
    return fieldnames, rows


def build_matches(raw_dir=DEFAULT_RAW_DIR, out_path=DEFAULT_OUT,
                  mapping_path=DEFAULT_MAPPING, strict=False, write=True):
    raw_dir = Path(raw_dir)
    resolver = TeamResolver.from_csv(mapping_path)
    captures = latest_capture_per_season(raw_dir)
    if not captures:
        raise FileNotFoundError(
            f"Aucune capture {LEAGUE_CODE}_*.csv dans {raw_dir}. "
            f"Lancer d'abord l'ingestion."
        )
    report = ConsolidationReport()
    all_columns = list(DERIVED_COLS)
    consolidated = []
    for season in sorted(captures):
        fieldnames, rows = _read_rows(captures[season])
        for col in fieldnames:
            if col not in all_columns:
                all_columns.append(col)
        for row in rows:
            out_row = dict(row)
            out_row["season"] = season
            for raw_col, canon_col in (
                (HOME_COL, "HomeTeamCanonical"),
                (AWAY_COL, "AwayTeamCanonical"),
            ):
                raw_name = (row.get(raw_col) or "").strip()
                try:
                    out_row[canon_col] = resolver.to_canonical(raw_name, source=SOURCE)
                except UnknownTeamError:
                    if strict:
                        raise
                    out_row[canon_col] = ""
                    report.unresolved_teams[raw_name] = (
                        report.unresolved_teams.get(raw_name, 0) + 1
                    )
            consolidated.append(out_row)
        report.seasons.append(season)
        report.rows_per_season[season] = len(rows)
    report.total_rows = len(consolidated)
    report.columns = all_columns
    if write:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(consolidated)
    return report


if __name__ == "__main__":  # pragma: no cover
    rep = build_matches(strict=False)
    print(rep.summary())
    if not rep.ok:
        raise SystemExit(
            "\nConsolidation incomplète : compléter team_mapping.csv "
            "avec les équipes ci-dessus, puis relancer."
        )
