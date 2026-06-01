"""Validation walk-forward du modèle BAYÉSIEN — comparé au fréquentiste et au marché.

Réutilise toute la mécanique de walkforward.py (chargement, métriques, marché,
calibration) pour garantir une comparaison HONNÊTE : exactement les mêmes matchs
évalués, exactement les mêmes métriques calculées de la même façon. La seule
différence est le modèle utilisé pour prédire.

Même protocole rigoureux : amorçage = 2 premières saisons, walk-forward sans
fuite temporelle (on entraîne uniquement sur le passé strict de chaque date).

À chaque date, on ajuste DEUX modèles sur le même passé :
  - le fréquentiste (rapide, référence : log-loss connue ~1.0037)
  - le bayésien MAP+Laplace
puis on prédit les mêmes matchs avec les deux. Le marché (cotes clôture) sert de
juge de paix commun.

MAP+Laplace étant rapide, le walk-forward complet est faisable (contrairement à
PyMC). Pour la validation, on n'a pas besoin des intervalles de crédibilité : on
évalue la prédiction ponctuelle (au MAP), donc on n'échantillonne pas (rapide).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from model.dixoncoles import fit as fit_freq, predict_1x2 as predict_freq
from model.dixoncoles_bayes import fit_bayes
from model.dixoncoles_bayes import _params_from_theta, _predict_one

# On réutilise tout le code commun de walkforward.
from validation.walkforward import (
    BURN_IN_SEASONS,
    DEFAULT_MATCHES,
    DEFAULT_OUT,
    EPS,
    Metrics,
    calibration_table,
    load_matches,
    market_probs,
)


def _predict_bayes_map(params, home: str, away: str) -> dict[str, float]:
    """Prédiction ponctuelle bayésienne au MAP (sans échantillonner :
    pour la validation on n'a besoin que du point, pas de l'intervalle)."""
    i, j = params.index(home), params.index(away)
    att, dfc, ha, rho, inter = _params_from_theta(
        params.mean_vector, params.teams, params._layout
    )
    h, d, a = _predict_one(att, dfc, ha, rho, inter, i, j)
    return {"home": h, "draw": d, "away": a}


def walk_forward_compare(matches):
    """Walk-forward évaluant bayésien ET fréquentiste sur les mêmes matchs.

    Retourne (m_bayes, m_freq, m_market, n_unknown).
    """
    m_bayes, m_freq, m_market = Metrics(), Metrics(), Metrics()
    n_unknown = 0

    eval_matches = [m for m in matches if m.season not in BURN_IN_SEASONS]
    eval_dates = sorted({m.date for m in eval_matches})

    for d in eval_dates:
        train = [m for m in matches if m.date < d]
        if len(train) < 100:
            continue

        ht = [m.home for m in train]
        at = [m.away for m in train]
        hg = [m.hg for m in train]
        ag = [m.ag for m in train]

        # Deux fits sur le MÊME passé.
        p_freq = fit_freq(ht, at, hg, ag)
        p_bayes = fit_bayes(ht, at, hg, ag)
        known = set(p_freq.teams)  # mêmes équipes pour les deux

        for m in (mt for mt in eval_matches if mt.date == d):
            if m.home not in known or m.away not in known:
                n_unknown += 1
                continue
            m_freq.add(predict_freq(p_freq, m.home, m.away), m.result)
            m_bayes.add(_predict_bayes_map(p_bayes, m.home, m.away), m.result)
            if m.odds is not None:
                m_market.add(market_probs(m.odds), m.result)

    return m_bayes, m_freq, m_market, n_unknown


def run(matches_path: Path = DEFAULT_MATCHES, out_dir: Path = DEFAULT_OUT):
    matches = load_matches(matches_path)
    print(f"Matchs chargés : {len(matches)}")

    m_bayes, m_freq, m_market, n_unknown = walk_forward_compare(matches)

    print(f"\nMatchs évalués : {m_bayes.n}  (ignorés équipe inconnue : {n_unknown})")
    print(f"Matchs avec cotes clôture : {m_market.n}")

    print("\n=== LOG-LOSS (plus bas = meilleur) ===")
    print(f"  Bayésien      : {m_bayes.log_loss:.4f}")
    print(f"  Fréquentiste  : {m_freq.log_loss:.4f}")
    print(f"  Marché        : {m_market.log_loss:.4f}")

    print("\n=== BRIER (plus bas = meilleur) ===")
    print(f"  Bayésien      : {m_bayes.brier:.4f}")
    print(f"  Fréquentiste  : {m_freq.brier:.4f}")
    print(f"  Marché        : {m_market.brier:.4f}")

    print("\n=== VERDICT ===")
    diff = m_freq.log_loss - m_bayes.log_loss
    if abs(diff) < 0.0010:
        print(f"  ~ Bayésien ≈ fréquentiste (écart {diff:+.4f}, négligeable).")
        print("    L'apport du bayésien est surtout l'incertitude quantifiée,")
        print("    pas un gain de précision. Les deux se valent en prédiction.")
    elif diff > 0:
        print(f"  ✓ Le bayésien BAT le fréquentiste de {diff:+.4f} en log-loss.")
        print("    La régularisation hiérarchique améliore la prédiction.")
    else:
        print(f"  ✗ Le bayésien fait MOINS bien que le fréquentiste ({diff:+.4f}).")
        print("    Régularisation peut-être trop forte ; à investiguer.")

    gap_market = m_bayes.log_loss - m_market.log_loss
    if m_market.n:
        print(f"\n  Écart bayésien vs marché : {gap_market:+.4f}")
        if gap_market < 0:
            print("    ✓✓ Le bayésien bat le marché — rare !")
        else:
            print("    ~ Ne bat pas le marché (normal). Comparer à l'écart "
                  "fréquentiste (+0.0225 précédemment).")

    print("\n=== CALIBRATION DU BAYÉSIEN ===")
    print("  tranche      n     prédit   réel")
    for label, n, pred, real in calibration_table(m_bayes):
        flag = "" if abs(pred - real) < 0.05 else "  <-- écart"
        print(f"  {label:<10} {n:>4}   {pred:.3f}   {real:.3f}{flag}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary_bayes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "bayes", "freq", "market"])
        w.writerow(["log_loss", f"{m_bayes.log_loss:.4f}",
                    f"{m_freq.log_loss:.4f}", f"{m_market.log_loss:.4f}"])
        w.writerow(["brier", f"{m_bayes.brier:.4f}",
                    f"{m_freq.brier:.4f}", f"{m_market.brier:.4f}"])
        w.writerow(["n_evaluated", m_bayes.n, m_freq.n, m_market.n])
    with (out_dir / "calibration_bayes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["bin", "count", "avg_predicted", "fraction_observed"])
        for label, n, pred, real in calibration_table(m_bayes):
            w.writerow([label, n, f"{pred:.4f}", f"{real:.4f}"])

    print("\nÉcrit : validation/summary_bayes.csv, validation/calibration_bayes.csv")
    return m_bayes, m_freq, m_market


if __name__ == "__main__":  # pragma: no cover
    run()
