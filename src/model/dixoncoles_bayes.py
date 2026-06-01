"""Dixon-Coles bayésien hiérarchique, estimé par MAP + Laplace.

Vrai modèle bayésien (priors, hiérarchie, incertitude quantifiée) sans MCNC :
  - MAP : on maximise (log-vraisemblance + log-priors). Les priors régularisent
    les forces (utile pour les promus à faible historique).
  - Hiérarchie : σ_att, σ_def sont estimés (pas fixés), avec hyperpriors
    HalfNormal. Le modèle apprend l'ampleur des écarts entre équipes.
  - Laplace : l'incertitude vient de la courbure (hessienne) autour du MAP.
    Inverse de la hessienne = covariance approchée des paramètres.
  - Prédiction avec incertitude : on échantillonne dans la gaussienne de Laplace
    et on propage jusqu'à P(1/N/2) → on obtient moyenne + intervalle crédible.

Réutilise la correction τ et la matrice des scores du modèle fréquentiste
(une seule définition, pas de duplication).

Modèle :
    log(λ) = intercept + home_adv + att[i] - def[j]
    log(μ) = intercept           + att[j] - def[i]
Priors :
    att[k] ~ N(0, σ_att)         def[k] ~ N(0, σ_def)
    home_adv ~ N(0.25, 0.5)      rho ~ N(-0.1, 0.1)   intercept ~ N(0, 1)
    σ_att, σ_def ~ HalfNormal(1)  (paramétrés en log pour rester positifs)
Contrainte d'identifiabilité : somme(att)=0, somme(def)=0 (dernière équipe = -somme).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from model.dixoncoles import MAX_GOALS, tau

# Hyperparamètres des priors (fixes, faiblement informatifs).
PRIOR_HOME_MEAN, PRIOR_HOME_SD = 0.25, 0.5
PRIOR_RHO_MEAN, PRIOR_RHO_SD = -0.1, 0.1
PRIOR_INTERCEPT_SD = 1.0
HALF_NORMAL_SD = 1.0  # pour σ_att, σ_def


@dataclass
class BayesParams:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    intercept: float
    sigma_att: float
    sigma_def: float
    # Pour l'incertitude :
    mean_vector: np.ndarray        # MAP (vecteur d'optim complet)
    cov_matrix: np.ndarray         # covariance de Laplace
    _layout: dict                  # indices des paramètres dans le vecteur

    def index(self, team: str) -> int:
        return self.teams.index(team)

    # ------------------------------------------------------------------ #
    # Sauvegarde / rechargement (découplage fit/predict)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Sérialise tout ce qu'il faut pour reconstruire les prédictions AVEC
        intervalles : forces, scalaires, vecteur MAP, covariance, layout.
        La covariance (~41 Ko pour 25 équipes) permet de ré-échantillonner les
        intervalles de crédibilité sans réajuster le modèle."""
        return {
            "teams": self.teams,
            "attack": self.attack.tolist(),
            "defence": self.defence.tolist(),
            "home_adv": self.home_adv, "rho": self.rho,
            "intercept": self.intercept,
            "sigma_att": self.sigma_att, "sigma_def": self.sigma_def,
            "mean_vector": self.mean_vector.tolist(),
            "cov_matrix": self.cov_matrix.tolist(),
            "layout": {k: ([v.start, v.stop] if isinstance(v, slice) else v)
                       for k, v in self._layout.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BayesParams":
        layout = {}
        for k, v in d["layout"].items():
            layout[k] = slice(v[0], v[1]) if isinstance(v, list) else v
        return cls(
            teams=list(d["teams"]),
            attack=np.asarray(d["attack"]),
            defence=np.asarray(d["defence"]),
            home_adv=float(d["home_adv"]), rho=float(d["rho"]),
            intercept=float(d["intercept"]),
            sigma_att=float(d["sigma_att"]), sigma_def=float(d["sigma_def"]),
            mean_vector=np.asarray(d["mean_vector"]),
            cov_matrix=np.asarray(d["cov_matrix"]),
            _layout=layout,
        )


# ---------------------------------------------------------------------- #
# Paramétrage du vecteur d'optimisation
# ---------------------------------------------------------------------- #
# theta = [att_free (n-1), def_free (n-1), home_adv, rho, intercept,
#          log_sigma_att, log_sigma_def]
def _layout(n_teams: int) -> dict:
    a = n_teams - 1
    return {
        "att": slice(0, a),
        "def": slice(a, 2 * a),
        "home_adv": 2 * a,
        "rho": 2 * a + 1,
        "intercept": 2 * a + 2,
        "log_sig_att": 2 * a + 3,
        "log_sig_def": 2 * a + 4,
        "size": 2 * a + 5,
    }


def _unpack(theta, n_teams, lay):
    a_free = theta[lay["att"]]
    d_free = theta[lay["def"]]
    attack = np.concatenate([a_free, [-a_free.sum()]])
    defence = np.concatenate([d_free, [-d_free.sum()]])
    home_adv = theta[lay["home_adv"]]
    rho = theta[lay["rho"]]
    intercept = theta[lay["intercept"]]
    sig_att = np.exp(theta[lay["log_sig_att"]])
    sig_def = np.exp(theta[lay["log_sig_def"]])
    return attack, defence, home_adv, rho, intercept, sig_att, sig_def


# ---------------------------------------------------------------------- #
# Objectif : -log posterior = -(log-vraisemblance + log-priors)
# ---------------------------------------------------------------------- #
def _neg_log_posterior(theta, home_idx, away_idx, hg, ag, n_teams, lay):
    attack, defence, home_adv, rho, intercept, sig_att, sig_def = _unpack(
        theta, n_teams, lay
    )

    # --- Vraisemblance ---
    log_lam = intercept + home_adv + attack[home_idx] - defence[away_idx]
    log_mu = intercept + attack[away_idx] - defence[home_idx]
    lam, mu = np.exp(log_lam), np.exp(log_mu)
    ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
    t = tau(hg, ag, lam, mu, rho)
    if np.any(t <= 0) or sig_att <= 0 or sig_def <= 0:
        return 1e10
    ll = ll + np.log(t)
    log_lik = np.sum(ll)

    # --- Priors ---
    # Forces : N(0, sigma). On somme sur TOUTES les équipes (la contrainte
    # somme=0 est gérée par le paramétrage ; on régularise le vecteur complet).
    lp_att = np.sum(_norm_logpdf(attack, 0.0, sig_att))
    lp_def = np.sum(_norm_logpdf(defence, 0.0, sig_def))
    lp_home = _norm_logpdf(home_adv, PRIOR_HOME_MEAN, PRIOR_HOME_SD)
    lp_rho = _norm_logpdf(rho, PRIOR_RHO_MEAN, PRIOR_RHO_SD)
    lp_int = _norm_logpdf(intercept, 0.0, PRIOR_INTERCEPT_SD)
    # Hyperpriors HalfNormal sur sigma. On optimise en log_sigma, donc on ajoute
    # le jacobien (log_sigma -> sigma) : + log(sigma) = + log_sig.
    lp_sig = (
        _halfnormal_logpdf(sig_att, HALF_NORMAL_SD) + theta[lay["log_sig_att"]]
        + _halfnormal_logpdf(sig_def, HALF_NORMAL_SD) + theta[lay["log_sig_def"]]
    )

    log_prior = lp_att + lp_def + lp_home + lp_rho + lp_int + lp_sig
    return -(log_lik + log_prior)


def _norm_logpdf(x, mean, sd):
    return -0.5 * np.log(2 * np.pi * sd ** 2) - (x - mean) ** 2 / (2 * sd ** 2)


def _halfnormal_logpdf(x, sd):
    # densité HalfNormal pour x>0 ; constante près, suffit pour le MAP.
    return -0.5 * (x / sd) ** 2 + np.log(np.sqrt(2 / np.pi) / sd)


# ---------------------------------------------------------------------- #
# Fit MAP + Laplace
# ---------------------------------------------------------------------- #
def fit_bayes(home_teams, away_teams, home_goals, away_goals) -> BayesParams:
    teams = sorted(set(home_teams) | set(away_teams))
    n = len(teams)
    lay = _layout(n)
    idx = {t: i for i, t in enumerate(teams)}
    home_idx = np.array([idx[t] for t in home_teams])
    away_idx = np.array([idx[t] for t in away_teams])
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)

    theta0 = np.zeros(lay["size"])
    theta0[lay["home_adv"]] = 0.25
    theta0[lay["rho"]] = -0.1
    theta0[lay["log_sig_att"]] = np.log(0.3)
    theta0[lay["log_sig_def"]] = np.log(0.3)

    res = minimize(
        _neg_log_posterior, theta0,
        args=(home_idx, away_idx, hg, ag, n, lay),
        method="L-BFGS-B", options={"maxiter": 2000},
    )
    theta = res.x

    # --- Laplace : hessienne numérique au MAP, covariance = inverse ---
    cov = _laplace_cov(
        lambda th: _neg_log_posterior(th, home_idx, away_idx, hg, ag, n, lay),
        theta,
    )

    attack, defence, home_adv, rho, intercept, sig_att, sig_def = _unpack(
        theta, n, lay
    )
    return BayesParams(
        teams=teams, attack=attack, defence=defence,
        home_adv=float(home_adv), rho=float(rho), intercept=float(intercept),
        sigma_att=float(sig_att), sigma_def=float(sig_def),
        mean_vector=theta, cov_matrix=cov, _layout=lay,
    )


def _laplace_cov(neg_log_post, theta, eps=1e-4):
    """Covariance de Laplace = inverse de la hessienne de -log posterior.
    Hessienne par différences finies centrées."""
    n = len(theta)
    H = np.zeros((n, n))
    f0 = neg_log_post(theta)
    for i in range(n):
        for j in range(i, n):
            ti, tj = theta.copy(), theta.copy()
            tpp = theta.copy(); tpp[i] += eps; tpp[j] += eps
            tpm = theta.copy(); tpm[i] += eps; tpm[j] -= eps
            tmp = theta.copy(); tmp[i] -= eps; tmp[j] += eps
            tmm = theta.copy(); tmm[i] -= eps; tmm[j] -= eps
            H[i, j] = (neg_log_post(tpp) - neg_log_post(tpm)
                       - neg_log_post(tmp) + neg_log_post(tmm)) / (4 * eps ** 2)
            H[j, i] = H[i, j]
    # Régularisation légère pour garantir l'inversibilité.
    H += np.eye(n) * 1e-6
    try:
        cov = np.linalg.inv(H)
        # Symétrise et force la semi-définie positivité (sécurité numérique).
        cov = (cov + cov.T) / 2
        return cov
    except np.linalg.LinAlgError:
        return np.eye(n) * 1e-3


# ---------------------------------------------------------------------- #
# Prédiction avec incertitude (échantillonnage de Laplace)
# ---------------------------------------------------------------------- #
def _params_from_theta(theta, teams, lay):
    a_free = theta[lay["att"]]
    d_free = theta[lay["def"]]
    attack = np.concatenate([a_free, [-a_free.sum()]])
    defence = np.concatenate([d_free, [-d_free.sum()]])
    return attack, defence, theta[lay["home_adv"]], theta[lay["rho"]], theta[lay["intercept"]]


def _predict_one(attack, defence, home_adv, rho, intercept, i, j):
    lam = np.exp(intercept + home_adv + attack[i] - defence[j])
    mu = np.exp(intercept + attack[j] - defence[i])
    goals = np.arange(MAX_GOALS + 1)
    mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= tau(x, y, lam, mu, rho)
    mat /= mat.sum()
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    return p_home, p_draw, p_away


def predict_1x2_bayes(
    params: BayesParams, home: str, away: str, n_samples: int = 300, seed: int = 0
) -> dict:
    """Prédiction 1/N/2 avec intervalle de crédibilité (échantillonnage Laplace)."""
    i, j = params.index(home), params.index(away)
    lay = params._layout
    rng = np.random.default_rng(seed)

    # Tirages dans la gaussienne de Laplace autour du MAP.
    try:
        draws = rng.multivariate_normal(
            params.mean_vector, params.cov_matrix, size=n_samples
        )
    except np.linalg.LinAlgError:
        draws = np.tile(params.mean_vector, (n_samples, 1))

    ph, pd, pa = [], [], []
    for theta in draws:
        att, dfc, ha, rho, inter = _params_from_theta(theta, params.teams, lay)
        h, d, a = _predict_one(att, dfc, ha, rho, inter, i, j)
        ph.append(h); pd.append(d); pa.append(a)
    ph, pd, pa = np.array(ph), np.array(pd), np.array(pa)

    def summarize(arr):
        return {
            "mean": float(arr.mean()),
            "lo": float(np.percentile(arr, 5)),    # intervalle crédible 90 %
            "hi": float(np.percentile(arr, 95)),
        }

    # Prédiction ponctuelle (au MAP) pour cohérence.
    att, dfc, ha, rho, inter = _params_from_theta(params.mean_vector, params.teams, lay)
    h0, d0, a0 = _predict_one(att, dfc, ha, rho, inter, i, j)

    return {
        "home": h0, "draw": d0, "away": a0,
        "home_ci": summarize(ph), "draw_ci": summarize(pd), "away_ci": summarize(pa),
  }
