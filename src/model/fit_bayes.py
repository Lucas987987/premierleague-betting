"""Fit bayésien (MAP + Laplace) sur matches.csv — étape 1 : un fit unique.

Ajuste le Dixon-Coles hiérarchique sur tout l'historique joué et écrit :
  - data/model/params_bayes.json     : paramètres MAP + sigma estimés
  - data/model/team_ratings_bayes.csv : forces AVEC incertitude (écart-type)
  - data/model/sample_predictions_bayes.csv : prédictions 1/N/2 + intervalles

But de cette étape : vérifier que le bayésien tourne, que les sigma hiérarchiques
sont sensés, et surtout que les intervalles de crédibilité reflètent le volume de
données (promus = intervalles larges). La validation walk-forward viendra ensuite.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from model.dixoncoles_bayes import BayesParams, fit_bayes, predict_1x2_bayes

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHES = _ROOT / "data" / "processed" / "matches.csv"
DEFAULT_MODEL_DIR = _ROOT / "data" / "model"

HOME, AWAY = "HomeTeamCanonical", "AwayTeamCanonical"
HG, AG = "FTHG", "FTAG"


def load_played(path: Path):
    home, away, hg, ag = [], [], [], []
    counts: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            h, a = (row.get(HOME) or "").strip(), (row.get(AWAY) or "").strip()
            gh, ga = (row.get(HG) or "").strip(), (row.get(AG) or "").strip()
            if not (h and a and gh and ga):
                continue
            try:
                hh, gg = int(float(gh)), int(float(ga))
            except ValueError:
                continue
            home.append(h); away.append(a); hg.append(hh); ag.append(gg)
            counts[h] = counts.get(h, 0) + 1
            counts[a] = counts.get(a, 0) + 1
    return home, away, hg, ag, counts


def _param_std(params: BayesParams):
    """Écart-type de chaque attaque/défense, depuis la covariance de Laplace.
    Pour les n-1 premières équipes (paramètres libres) ; la dernière est dérivée."""
    lay = params._layout
    diag = np.sqrt(np.clip(np.diag(params.cov_matrix), 0, None))
    n = len(params.teams)
    att_std = np.zeros(n)
    def_std = np.zeros(n)
    att_std[: n - 1] = diag[lay["att"]]
    def_std[: n - 1] = diag[lay["def"]]
    # Dernière équipe = -somme : son incertitude ~ somme quadratique (approx).
    att_std[n - 1] = np.sqrt(np.sum(diag[lay["att"]] ** 2))
    def_std[n - 1] = np.sqrt(np.sum(diag[lay["def"]] ** 2))
    return att_std, def_std


def run(matches_path: Path = DEFAULT_MATCHES, model_dir: Path = DEFAULT_MODEL_DIR):
    home, away, hg, ag, counts = load_played(matches_path)
    n = len(hg)
    if n < 100:
        raise ValueError(f"Trop peu de matchs joués ({n}).")
    print(f"Matchs joués : {n}")

    params = fit_bayes(home, away, hg, ag)
    print(f"Équipes : {len(params.teams)}")
    print(f"home_adv = {params.home_adv:.3f} | rho = {params.rho:.3f}")
    print(f"sigma_att = {params.sigma_att:.3f} | sigma_def = {params.sigma_def:.3f}")

    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Paramètres (résumé lisible).
    (model_dir / "params_bayes.json").write_text(
        json.dumps({
            "teams": params.teams,
            "attack": params.attack.tolist(),
            "defence": params.defence.tolist(),
            "home_adv": params.home_adv, "rho": params.rho,
            "intercept": params.intercept,
            "sigma_att": params.sigma_att, "sigma_def": params.sigma_def,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 1b. Paramètres COMPLETS (avec covariance) pour le scoring : permet de
    # reconstruire les prédictions avec intervalles sans réajuster (découplage
    # fit/predict, cf. ARCHITECTURE.md).
    (model_dir / "params_bayes_full.json").write_text(
        json.dumps(params.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    # 2. Forces avec incertitude + nombre de matchs (pour voir le lien).
    att_std, def_std = _param_std(params)
    ratings = sorted(
        range(len(params.teams)),
        key=lambda k: -(params.attack[k] + params.defence[k]),
    )
    with (model_dir / "team_ratings_bayes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["team", "matches", "attack", "attack_std",
                    "defence", "defence_std", "net"])
        for k in ratings:
            t = params.teams[k]
            w.writerow([
                t, counts.get(t, 0),
                f"{params.attack[k]:.4f}", f"{att_std[k]:.4f}",
                f"{params.defence[k]:.4f}", f"{def_std[k]:.4f}",
                f"{params.attack[k] + params.defence[k]:.4f}",
            ])

    # 3. Prédictions d'affiche AVEC intervalles.
    top = [params.teams[k] for k in ratings[:2]]
    bottom = [params.teams[k] for k in ratings[-2:]]
    pairs = [(top[0], top[1]), (top[0], bottom[0]), (bottom[0], top[0])]
    with (model_dir / "sample_predictions_bayes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["home", "away", "P_home", "home_lo", "home_hi",
                    "P_draw", "P_away", "away_lo", "away_hi"])
        for h, a in pairs:
            pr = predict_1x2_bayes(params, h, a, n_samples=400)
            w.writerow([
                h, a,
                f"{pr['home']:.3f}", f"{pr['home_ci']['lo']:.3f}", f"{pr['home_ci']['hi']:.3f}",
                f"{pr['draw']:.3f}", f"{pr['away']:.3f}",
                f"{pr['away_ci']['lo']:.3f}", f"{pr['away_ci']['hi']:.3f}",
            ])

    # Aperçu : top 5 + les équipes à plus faible historique.
    print("\nTop 5 forces (avec écart-type d'attaque) :")
    for k in ratings[:5]:
        t = params.teams[k]
        print(f"   {t:<16} matchs={counts.get(t,0):>3}  net={params.attack[k]+params.defence[k]:+.3f}"
              f"  att_std={att_std[k]:.3f}")

    least = sorted(params.teams, key=lambda t: counts.get(t, 0))[:3]
    print("\nÉquipes à plus faible historique (intervalles attendus plus larges) :")
    for t in least:
        k = params.index(t)
        print(f"   {t:<16} matchs={counts.get(t,0):>3}  att_std={att_std[k]:.3f}")

    print("\nÉcrit : params_bayes.json, team_ratings_bayes.csv, "
          "sample_predictions_bayes.csv")
    return params


if __name__ == "__main__":  # pragma: no cover
    run()
