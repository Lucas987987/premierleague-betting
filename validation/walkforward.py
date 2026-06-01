"""Validation walk-forward (hors pipeline de prod — cf. ARCHITECTURE.md §5).

Répond à LA question : le modèle est-il bon ? = bien calibré ET bat le marché
en log-loss. Pas "prédit souvent juste" (piège de l'accuracy).

Règle d'or : on ne prédit JAMAIS un match avec une information postérieure à son
coup d'envoi. Pour prédire une date D, on entraîne uniquement sur les matchs
joués strictement avant D.

Protocole (V1) :
  - Amorçage : les 2 premières saisons servent uniquement à l'entraînement initial.
  - Walk-forward : à partir de la 3e saison, on prédit par "fenêtre" de dates
    (regroupées par journée/date), en réentraînant sur tout le passé à chaque pas.
  - Pas de pondération temporelle ξ en V1 (tous matchs à poids égal).

Comparaisons :
  - Modèle Dixon-Coles
  - Marché : cotes de clôture Pinnacle (PSCH/D/A), dévigorisées ; repli sur la
    moyenne marché (AvgCH/D/A) si Pinnacle manque.
  - Baseline naïve : fréquences de base 1/N/2 du jeu d'amorçage (constantes).

Ce module se lance à la main / via workflow dédié, jamais dans la prod.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from model.dixoncoles import fit, predict_1x2

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHES = _ROOT / "data" / "processed" / "matches.csv"
DEFAULT_OUT = _ROOT / "data" / "validation"

HOME, AWAY = "HomeTeamCanonical", "AwayTeamCanonical"
HG, AG, RES = "FTHG", "FTAG", "FTR"
# Cotes de clôture : Pinnacle d'abord, moyenne marché en repli.
ODDS_PINNACLE = ("PSCH", "PSCD", "PSCA")
ODDS_MARKET = ("AvgCH", "AvgCD", "AvgCA")

# Saisons d'amorçage (entraînement initial, non évaluées).
BURN_IN_SEASONS = {"2122", "2223"}

EPS = 1e-15  # borne pour éviter log(0)


@dataclass
class Match:
    date: datetime
    season: str
    home: str
    away: str
    hg: int
    ag: int
    result: str               # 'H' / 'D' / 'A'
    odds: tuple[float, float, float] | None  # (H, D, A) clôture, ou None


@dataclass
class Metrics:
    n: int = 0
    log_loss_sum: float = 0.0
    brier_sum: float = 0.0
    # Pour la calibration : on enregistre CHAQUE issue de CHAQUE match sous la
    # forme (proba prédite pour cette issue, 1 si cette issue s'est réalisée).
    # C'est la bonne façon de mesurer la calibration : on veut savoir si, parmi
    # toutes les fois où le modèle a dit "60 %", l'issue est arrivée ~60 % du temps.
    calib: list[tuple[float, int]] = field(default_factory=list)

    def add(self, probs: dict[str, float], result: str):
        p = {k: min(max(v, EPS), 1 - EPS) for k, v in probs.items()}
        key = {"H": "home", "D": "draw", "A": "away"}[result]
        self.log_loss_sum += -math.log(p[key])
        # Brier multiclasse + calibration : on parcourt les 3 issues.
        for k in ("home", "draw", "away"):
            realized = 1 if {"home": "H", "draw": "D", "away": "A"}[k] == result else 0
            self.brier_sum += (p[k] - realized) ** 2
            self.calib.append((p[k], realized))
        self.n += 1

    @property
    def log_loss(self) -> float:
        return self.log_loss_sum / self.n if self.n else float("nan")

    @property
    def brier(self) -> float:
        return self.brier_sum / self.n if self.n else float("nan")


# ---------------------------------------------------------------------- #
# Chargement
# ---------------------------------------------------------------------- #
def _parse_date(s: str) -> datetime:
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date illisible : {s!r}")


def _parse_odds(row) -> tuple[float, float, float] | None:
    for cols in (ODDS_PINNACLE, ODDS_MARKET):
        vals = [(row.get(c) or "").strip() for c in cols]
        if all(vals):
            try:
                o = tuple(float(v) for v in vals)
                if all(x > 1.0 for x in o):
                    return o  # type: ignore
            except ValueError:
                pass
    return None


def load_matches(path: Path) -> list[Match]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            h, a = (row.get(HOME) or "").strip(), (row.get(AWAY) or "").strip()
            gh, ga = (row.get(HG) or "").strip(), (row.get(AG) or "").strip()
            res = (row.get(RES) or "").strip()
            if not (h and a and gh and ga and res in ("H", "D", "A")):
                continue
            try:
                out.append(Match(
                    date=_parse_date(row["Date"]),
                    season=(row.get("season") or "").strip(),
                    home=h, away=a,
                    hg=int(float(gh)), ag=int(float(ga)),
                    result=res,
                    odds=_parse_odds(row),
                ))
            except (ValueError, KeyError):
                continue
    out.sort(key=lambda m: m.date)
    return out


# ---------------------------------------------------------------------- #
# Marché : dévigorisation (normalisation des probas implicites)
# ---------------------------------------------------------------------- #
def market_probs(odds: tuple[float, float, float]) -> dict[str, float]:
    """Probas implicites dévigorisées (méthode proportionnelle)."""
    inv = [1.0 / o for o in odds]
    s = sum(inv)
    return {"home": inv[0] / s, "draw": inv[1] / s, "away": inv[2] / s}


# ---------------------------------------------------------------------- #
# Walk-forward
# ---------------------------------------------------------------------- #
def _naive_probs(history: list[Match]) -> dict[str, float]:
    """Fréquences de base 1/N/2 sur l'historique d'amorçage."""
    n = len(history)
    h = sum(1 for m in history if m.result == "H") / n
    d = sum(1 for m in history if m.result == "D") / n
    a = sum(1 for m in history if m.result == "A") / n
    return {"home": h, "draw": d, "away": a}


def walk_forward(matches: list[Match]):
    """Exécute la validation. Retourne (metrics_model, metrics_market,
    metrics_naive, n_skipped_no_odds, n_skipped_unknown_team)."""
    m_model, m_market, m_naive = Metrics(), Metrics(), Metrics()
    n_no_odds = 0
    n_unknown = 0

    # Groupes de dates à prédire = toutes les dates hors amorçage, triées.
    eval_matches = [m for m in matches if m.season not in BURN_IN_SEASONS]
    eval_dates = sorted({m.date for m in eval_matches})

    naive_ref = _naive_probs([m for m in matches if m.season in BURN_IN_SEASONS])

    for d in eval_dates:
        # Entraînement : tous les matchs joués STRICTEMENT avant cette date.
        train = [m for m in matches if m.date < d]
        if len(train) < 100:
            continue
        params = fit(
            [m.home for m in train], [m.away for m in train],
            [m.hg for m in train], [m.ag for m in train],
        )
        known = set(params.teams)

        # Prédiction de tous les matchs de cette date.
        for m in (mt for mt in eval_matches if mt.date == d):
            if m.home not in known or m.away not in known:
                n_unknown += 1  # équipe jamais vue (promu sans historique)
                continue
            probs = predict_1x2(params, m.home, m.away)
            m_model.add(probs, m.result)
            m_naive.add(naive_ref, m.result)
            if m.odds is not None:
                m_market.add(market_probs(m.odds), m.result)
            else:
                n_no_odds += 1

    return m_model, m_market, m_naive, n_no_odds, n_unknown


# ---------------------------------------------------------------------- #
# Calibration : binning
# ---------------------------------------------------------------------- #
def calibration_table(metrics: Metrics, n_bins: int = 10):
    """Regroupe les prédictions par tranche de proba et compare au taux réel."""
    bins = [[] for _ in range(n_bins)]
    for p, y in metrics.calib:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    rows = []
    for i, b in enumerate(bins):
        if not b:
            continue
        avg_pred = sum(p for p, _ in b) / len(b)
        frac_real = sum(y for _, y in b) / len(b)
        rows.append((f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}", len(b), avg_pred, frac_real))
    return rows


# ---------------------------------------------------------------------- #
# Rapport
# ---------------------------------------------------------------------- #
def run(matches_path: Path = DEFAULT_MATCHES, out_dir: Path = DEFAULT_OUT):
    matches = load_matches(matches_path)
    print(f"Matchs chargés : {len(matches)}")

    m_model, m_market, m_naive, n_no_odds, n_unknown = walk_forward(matches)

    print(f"\nMatchs évalués (modèle) : {m_model.n}")
    print(f"Matchs avec cotes clôture : {m_market.n}  (sans cotes : {n_no_odds})")
    print(f"Ignorés (équipe sans historique) : {n_unknown}")

    print("\n=== LOG-LOSS (plus bas = meilleur) ===")
    print(f"  Modèle Dixon-Coles : {m_model.log_loss:.4f}")
    print(f"  Marché (clôture)   : {m_market.log_loss:.4f}")
    print(f"  Baseline naïve     : {m_naive.log_loss:.4f}")

    print("\n=== BRIER (plus bas = meilleur) ===")
    print(f"  Modèle : {m_model.brier:.4f}")
    print(f"  Marché : {m_market.brier:.4f}")
    print(f"  Naïve  : {m_naive.brier:.4f}")

    # Verdict.
    print("\n=== VERDICT ===")
    if m_model.log_loss < m_naive.log_loss:
        print("  ✓ Le modèle bat la baseline naïve (il apprend quelque chose).")
    else:
        print("  ✗ Le modèle ne bat MÊME PAS la baseline naïve — problème sérieux.")
    if m_market.n > 0:
        gap = m_model.log_loss - m_market.log_loss
        if gap < 0:
            print(f"  ✓✓ Le modèle BAT le marché en log-loss (écart {gap:+.4f}) — rare !")
        else:
            print(f"  ~ Le modèle ne bat pas le marché (écart {gap:+.4f}).")
            print(f"    Normal : battre la clôture est très difficile. "
                  f"L'écart mesure le chemin restant.")

    print("\n=== CALIBRATION DU MODÈLE (proba prédite vs taux réel) ===")
    print("  tranche      n     prédit   réel")
    for label, n, pred, real in calibration_table(m_model):
        flag = "" if abs(pred - real) < 0.05 else "  <-- écart"
        print(f"  {label:<10} {n:>4}   {pred:.3f}   {real:.3f}{flag}")

    # Écriture d'un CSV de calibration pour usage ultérieur (frontend, suivi).
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "calibration.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bin", "count", "avg_predicted", "fraction_observed"])
        for label, n, pred, real in calibration_table(m_model):
            w.writerow([label, n, f"{pred:.4f}", f"{real:.4f}"])
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "model", "market", "naive"])
        w.writerow(["log_loss", f"{m_model.log_loss:.4f}",
                    f"{m_market.log_loss:.4f}", f"{m_naive.log_loss:.4f}"])
        w.writerow(["brier", f"{m_model.brier:.4f}",
                    f"{m_market.brier:.4f}", f"{m_naive.brier:.4f}"])
        w.writerow(["n_evaluated", m_model.n, m_market.n, m_naive.n])

    print("\nÉcrit : validation/summary.csv, validation/calibration.csv")
    return m_model, m_market, m_naive


if __name__ == "__main__":  # pragma: no cover
    run()
