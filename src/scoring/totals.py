"""Prédiction Over/Under (totaux de buts) — lecture de la matrice des scores.

Le modèle bayésien construit déjà la matrice P(score exact) pour chaque match.
Le 1/N/2 en est une agrégation (signe de buts_dom − buts_ext). L'Over/Under en
est une AUTRE agrégation (somme buts_dom + buts_ext vs un seuil). Même modèle,
même matrice, aucun réajustement.

Pour chaque seuil S (0.5, 1.5, 2.5, 3.5) :
  P(over S)  = somme des cases où (i + j) > S
  P(under S) = somme des cases où (i + j) < S
  (S est demi-entier, donc pas d'égalité possible : over + under = 1)

Incertitude : on échantillonne dans la gaussienne de Laplace (comme pour le
1/N/2) et on propage jusqu'aux probas over/under → moyenne + intervalle crédible.

ATTENTION (calibration) : la correction Dixon-Coles τ agit sur les 4 scores
faibles (0-0,1-0,0-1,1-1). Elle a été calibrée pour le 1/N/2. Les seuils bas
(0.5, 1.5) dépendent fortement de ces scores → leur calibration doit être
VÉRIFIÉE séparément, pas supposée. Ce module calcule ; la validation tranche.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from model.dixoncoles import MAX_GOALS, tau
from model.dixoncoles_bayes import BayesParams, _params_from_theta

# Seuils standard du marché (demi-entiers : over+under = 1, pas de push).
THRESHOLDS = (0.5, 1.5, 2.5, 3.5)


def _score_matrix(attack, defence, home_adv, rho, intercept, i, j):
    """Matrice P(score exact) normalisée — identique à la logique 1/N/2."""
    lam = np.exp(intercept + home_adv + attack[i] - defence[j])
    mu = np.exp(intercept + attack[j] - defence[i])
    goals = np.arange(MAX_GOALS + 1)
    mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= tau(x, y, lam, mu, rho)
    mat /= mat.sum()
    return mat


def _over_probs_from_matrix(mat) -> dict[float, float]:
    """P(over S) pour chaque seuil, depuis une matrice de scores.
    total[k] = P(somme des buts == k) ; P(over S) = somme des total[k] pour k>S."""
    n = mat.shape[0]
    totals = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            totals[i + j] += mat[i, j]
    out = {}
    for s in THRESHOLDS:
        # over S = total de buts strictement supérieur à S (S demi-entier).
        out[s] = float(totals[int(np.ceil(s)):].sum())
    return out


def predict_over_under(
    params: BayesParams, home: str, away: str,
    n_samples: int = 300, seed: int = 0,
) -> dict:
    """Probas over/under tous seuils, avec intervalle de crédibilité (Laplace).

    Retourne, par seuil S :
      {'over': p, 'under': 1-p, 'over_ci': {lo, hi}, 'under_ci': {lo, hi}}
    """
    i, j = params.index(home), params.index(away)
    lay = params._layout
    rng = np.random.default_rng(seed)

    # Échantillons de paramètres (même mécanique que predict_1x2_bayes).
    try:
        draws = rng.multivariate_normal(
            params.mean_vector, params.cov_matrix, size=n_samples
        )
    except np.linalg.LinAlgError:
        draws = np.tile(params.mean_vector, (n_samples, 1))

    # Distribution échantillonnée de P(over S) pour chaque seuil.
    samples = {s: [] for s in THRESHOLDS}
    for theta in draws:
        att, dfc, ha, rho, inter = _params_from_theta(theta, params.teams, lay)
        mat = _score_matrix(att, dfc, ha, rho, inter, i, j)
        op = _over_probs_from_matrix(mat)
        for s in THRESHOLDS:
            samples[s].append(op[s])

    # Prédiction ponctuelle au MAP.
    att, dfc, ha, rho, inter = _params_from_theta(params.mean_vector, params.teams, lay)
    point = _over_probs_from_matrix(_score_matrix(att, dfc, ha, rho, inter, i, j))

    result = {}
    for s in THRESHOLDS:
        arr = np.array(samples[s])
        over_p = point[s]
        result[s] = {
            "over": over_p,
            "under": 1.0 - over_p,
            "over_ci": {
                "lo": float(np.percentile(arr, 5)),
                "hi": float(np.percentile(arr, 95)),
            },
            "under_ci": {
                "lo": float(1.0 - np.percentile(arr, 95)),
                "hi": float(1.0 - np.percentile(arr, 5)),
            },
        }
    return result
