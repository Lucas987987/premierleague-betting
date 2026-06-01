"""ROI walk-forward sans fuite, filtré par score de confiance bayésien.

ATTENTION MÉTHODOLOGIQUE (à lire avant d'interpréter) :
  - On parie aux cotes de CLÔTURE football-data (seules dispo en historique).
    Parier à la clôture = AUCUN CLV par construction. La clôture est la ligne
    qu'on cherche à battre, pas une cote à laquelle on obtient de la value.
    => Le ROI attendu est au mieux neutre, probablement négatif (marge du book).
    Ce backtest mesure : "en pariant à la clôture selon le modèle, perd-on plus
    ou moins selon le score de confiance ?" — PAS une promesse de gain.
  - Walk-forward strict : à chaque date, le modèle n'est entraîné que sur le
    passé. Pas de fuite temporelle. (Réutilise walkforward.py.)

Règle de pari :
  - Pour chaque match hors amorçage : EV_issue = p_model_bayes × cote_clôture − 1.
  - On parie l'issue de meilleur EV si EV > 0 ET score_confiance >= seuil.
  - Mise fixe = 1 unité. Gain si l'issue se réalise = cote − 1, sinon −1.
  - ROI = profit_total / nb_paris.

Score de confiance (0-100) : basé sur la largeur moyenne des intervalles de
crédibilité bayésiens des issues (mêmes bornes que le frontend : FULL=0.04,
ZERO=0.30). Mesure la CERTITUDE DU MODÈLE, pas la value.
"""
from __future__ import annotations
import csv
from pathlib import Path
from model.dixoncoles_bayes import fit_bayes, _params_from_theta, _predict_one, predict_1x2_bayes
from validation.walkforward import (
    BURN_IN_SEASONS, DEFAULT_MATCHES, load_matches, market_probs,
)

CONF_FULL, CONF_ZERO = 0.04, 0.30  # mêmes bornes que le frontend


def _confidence(pred: dict) -> float | None:
    """Score 0-100 depuis la largeur moyenne des intervalles dispo (home/away)."""
    widths = []
    for iss in ("home", "away"):
        ci = pred.get(f"{iss}_ci")
        if ci and ci.get("lo") is not None:
            widths.append(ci["hi"] - ci["lo"])
    if not widths:
        return None
    avg = sum(widths) / len(widths)
    c = 1.0 if avg <= CONF_FULL else 0.0 if avg >= CONF_ZERO else (CONF_ZERO - avg) / (CONF_ZERO - CONF_FULL)
    return c * 100


def run(matches_path: Path = DEFAULT_MATCHES, thresholds=(0, 40, 60, 70)):
    matches = load_matches(matches_path)
    print(f"Matchs chargés : {len(matches)}")
    eval_matches = [m for m in matches if m.season not in BURN_IN_SEASONS]
    eval_dates = sorted({m.date for m in eval_matches})

    # Accumulateurs par seuil de confiance.
    by_thr = {t: {"n": 0, "profit": 0.0, "wins": 0} for t in thresholds}
    n_no_odds = 0
    n_bets_considered = 0

    for d in eval_dates:
        train = [m for m in matches if m.date < d]
        if len(train) < 100:
            continue
        params = fit_bayes(
            [m.home for m in train], [m.away for m in train],
            [m.hg for m in train], [m.ag for m in train],
        )
        known = set(params.teams)
        for m in (mt for mt in eval_matches if mt.date == d):
            if m.home not in known or m.away not in known:
                continue
            if m.odds is None:
                n_no_odds += 1
                continue
            pred = predict_1x2_bayes(params, m.home, m.away, n_samples=200)
            conf = _confidence(pred)
            if conf is None:
                continue
            # EV par issue avec la cote de clôture.
            cote = {"home": m.odds[0], "draw": m.odds[1], "away": m.odds[2]}
            p = {"home": pred["home"], "draw": pred["draw"], "away": pred["away"]}
            evs = {k: p[k] * cote[k] - 1.0 for k in ("home", "draw", "away")}
            best = max(evs, key=evs.get)
            if evs[best] <= 0:
                continue  # pas de value côté joueur
            n_bets_considered += 1
            # Le pari gagne si l'issue 'best' correspond au résultat réel.
            realized = {"home": "H", "draw": "D", "away": "A"}[best] == m.result
            payoff = (cote[best] - 1.0) if realized else -1.0
            # Comptabilise dans chaque seuil que la confiance dépasse.
            for t in thresholds:
                if conf >= t:
                    by_thr[t]["n"] += 1
                    by_thr[t]["profit"] += payoff
                    by_thr[t]["wins"] += 1 if realized else 0

    print(f"Dates évaluées : {len(eval_dates)} | sans cotes : {n_no_odds}")
    print(f"Paris à EV>0 considérés (avant filtre confiance) : {n_bets_considered}")
    print("\n=== ROI par seuil de confiance (mise fixe 1 unité, cotes clôture) ===")
    print(f"  {'seuil':>6} {'paris':>7} {'gagnés':>7} {'%gagn':>7} {'profit':>9} {'ROI':>8}")
    for t in thresholds:
        s = by_thr[t]
        if s["n"] == 0:
            print(f"  {t:>5}+ {0:>7}      —       —         —        —")
            continue
        roi = s["profit"] / s["n"]
        winrate = 100 * s["wins"] / s["n"]
        print(f"  {t:>5}+ {s['n']:>7} {s['wins']:>7} {winrate:>6.1f}% {s['profit']:>+9.2f} {roi*100:>+7.2f}%")
    return by_thr


if __name__ == "__main__":  # pragma: no cover
    run()
