"""Cœur du modèle : Dixon-Coles fréquentiste (baseline)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def tau(x, y, lam, mu, rho):
    x = np.asarray(x); y = np.asarray(y)
    out = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


@dataclass
class DixonColesParams:
    teams: list
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    intercept: float
    def index(self, team): return self.teams.index(team)


def _pack(attack_free, defence_free, home_adv, rho, intercept):
    return np.concatenate([attack_free, defence_free, [home_adv, rho, intercept]])


def _unpack(theta, n_teams):
    a_free = theta[: n_teams - 1]
    d_free = theta[n_teams - 1 : 2 * (n_teams - 1)]
    home_adv, rho, intercept = theta[-3], theta[-2], theta[-1]
    attack = np.concatenate([a_free, [-a_free.sum()]])
    defence = np.concatenate([d_free, [-d_free.sum()]])
    return attack, defence, home_adv, rho, intercept


def _neg_log_likelihood(theta, home_idx, away_idx, hg, ag, n_teams, weights):
    attack, defence, home_adv, rho, intercept = _unpack(theta, n_teams)
    log_lam = intercept + home_adv + attack[home_idx] - defence[away_idx]
    log_mu = intercept + attack[away_idx] - defence[home_idx]
    lam = np.exp(log_lam); mu = np.exp(log_mu)
    ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
    t = tau(hg, ag, lam, mu, rho)
    if np.any(t <= 0): return 1e10
    ll = ll + np.log(t)
    return -np.sum(weights * ll)


def fit(home_teams, away_teams, home_goals, away_goals, weights=None):
    teams = sorted(set(home_teams) | set(away_teams))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    home_idx = np.array([idx[t] for t in home_teams])
    away_idx = np.array([idx[t] for t in away_teams])
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    weights = np.ones(len(hg)) if weights is None else np.asarray(weights, dtype=float)
    theta0 = _pack(np.zeros(n - 1), np.zeros(n - 1), home_adv=0.25, rho=-0.1, intercept=0.0)
    res = minimize(_neg_log_likelihood, theta0,
                   args=(home_idx, away_idx, hg, ag, n, weights),
                   method="L-BFGS-B", options={"maxiter": 1000})
    attack, defence, home_adv, rho, intercept = _unpack(res.x, n)
    return DixonColesParams(teams=teams, attack=attack, defence=defence,
                            home_adv=float(home_adv), rho=float(rho), intercept=float(intercept))


def score_matrix(params, home, away):
    i, j = params.index(home), params.index(away)
    lam = np.exp(params.intercept + params.home_adv + params.attack[i] - params.defence[j])
    mu = np.exp(params.intercept + params.attack[j] - params.defence[i])
    goals = np.arange(MAX_GOALS + 1)
    px = poisson.pmf(goals, lam); py = poisson.pmf(goals, mu)
    mat = np.outer(px, py)
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= tau(x, y, lam, mu, params.rho)
    return mat / mat.sum()


def predict_1x2(params, home, away):
    mat = score_matrix(params, home, away)
    p_home = float(np.tril(mat, -1).sum())
    p_draw = float(np.trace(mat))
    p_away = float(np.triu(mat, 1).sum())
    return {"home": p_home, "draw": p_draw, "away": p_away}
