"""Validation de la calibration Over/Under, en walk-forward sans fuite.

Vérifie SEUIL PAR SEUIL si les probas over/under du modèle sont calibrées.
Vérité de terrain : total réel = FTHG + FTAG vs seuil (demi-entier).

Le modèle est calibré sur le 1/N/2 ; ça ne garantit pas les totaux, surtout aux
seuils bas (0.5, 1.5) où la correction Dixon-Coles pèse. On mesure, on ne suppose pas.

Comparaison au marché : football-data fournit des cotes O/U pour le seuil 2.5.
On utilise ici les cotes PRÉ-MATCH P>2.5/P<2.5 (Pinnacle), repli Avg>2.5/Avg<2.5
(cohérence avec le projet L1 ; la vraie clôture PC>2.5 est réservée au CLV).
Comparaison marché UNIQUEMENT sur 2.5 ; les autres seuils n'ont pas de référence.

Protocole : même walk-forward que le 1/N/2 (amorçage 2 saisons, passé strict).
Modèle FRÉQUENTISTE de base pour la vitesse (on valide la calibration de la
matrice de scores, pas l'incertitude).
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np
from scipy.stats import poisson
from model.dixoncoles import fit as fit_base, MAX_GOALS, tau
from validation.walkforward import (
    BURN_IN_SEASONS, DEFAULT_MATCHES, DEFAULT_OUT, EPS, load_matches,
)
from scoring.totals import THRESHOLDS

COL_P_OVER, COL_P_UNDER = "P>2.5", "P<2.5"          # Pinnacle pré-match
COL_AVG_OVER, COL_AVG_UNDER = "Avg>2.5", "Avg<2.5"  # moyenne marché (repli)


def _over_probs(params, home, away):
    i = params.teams.index(home)
    j = params.teams.index(away)
    lam = np.exp(params.intercept + params.home_adv + params.attack[i] - params.defence[j])
    mu = np.exp(params.intercept + params.attack[j] - params.defence[i])
    g = np.arange(MAX_GOALS + 1)
    mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= tau(x, y, lam, mu, params.rho)
    mat /= mat.sum()
    n = mat.shape[0]
    totals = np.zeros(2 * n - 1)
    for a in range(n):
        for b in range(n):
            totals[a + b] += mat[a, b]
    return {s: float(totals[int(np.ceil(s)):].sum()) for s in THRESHOLDS}


def _market_over_25(row):
    for co, cu in ((COL_P_OVER, COL_P_UNDER), (COL_AVG_OVER, COL_AVG_UNDER)):
        ov, un = (row.get(co) or "").strip(), (row.get(cu) or "").strip()
        if ov and un:
            try:
                o, u = float(ov), float(un)
                if o > 1 and u > 1:
                    inv_o, inv_u = 1/o, 1/u
                    return inv_o / (inv_o + inv_u)
            except ValueError:
                pass
    return None


class OUMetrics:
    def __init__(self):
        self.ll = 0.0; self.brier = 0.0; self.n = 0; self.calib = []
    def add(self, p_over, over_real):
        p = min(max(p_over, EPS), 1 - EPS)
        self.ll += -(over_real * math.log(p) + (1 - over_real) * math.log(1 - p))
        self.brier += (p - over_real) ** 2
        self.calib.append((p, over_real)); self.n += 1
    @property
    def log_loss(self): return self.ll / self.n if self.n else float("nan")
    @property
    def brier_score(self): return self.brier / self.n if self.n else float("nan")


def _calib_table(calib, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for p, y in calib:
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    rows = []
    for i, b in enumerate(bins):
        if not b: continue
        ap = sum(p for p, _ in b) / len(b)
        fr = sum(y for _, y in b) / len(b)
        rows.append((f"{i/n_bins:.1f}-{(i+1)/n_bins:.1f}", len(b), ap, fr))
    return rows


def _max_calib_gap(calib, n_bins=10):
    rows = _calib_table(calib, n_bins)
    gaps = [abs(p - r) for _, n, p, r in rows if n >= 20]
    return max(gaps) if gaps else 0.0


def _ece(calib, n_bins=10):
    rows = _calib_table(calib, n_bins)
    total = sum(n for _, n, _, _ in rows)
    if total == 0: return 0.0
    return sum(n * abs(p - r) for _, n, p, r in rows) / total


def m_date_str(m):
    return m.date.strftime("%d/%m/%Y")


def run(matches_path=DEFAULT_MATCHES, out_dir=DEFAULT_OUT):
    matches = load_matches(matches_path)
    raw_by_key = {}
    with Path(matches_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            k = (row.get("Date", ""), row.get("HomeTeamCanonical", ""),
                 row.get("AwayTeamCanonical", ""))
            raw_by_key[k] = row
    print(f"Matchs chargés : {len(matches)}")
    model_m = {s: OUMetrics() for s in THRESHOLDS}
    market_m = OUMetrics()
    eval_matches = [m for m in matches if m.season not in BURN_IN_SEASONS]
    eval_dates = sorted({m.date for m in eval_matches})
    for d in eval_dates:
        train = [m for m in matches if m.date < d]
        if len(train) < 150: continue
        params = fit_base([m.home for m in train], [m.away for m in train],
                          [m.hg for m in train], [m.ag for m in train])
        known = set(params.teams)
        for m in (mt for mt in eval_matches if mt.date == d):
            if m.home not in known or m.away not in known: continue
            total = m.hg + m.ag
            probs = _over_probs(params, m.home, m.away)
            for s in THRESHOLDS:
                model_m[s].add(probs[s], 1 if total > s else 0)
            row = raw_by_key.get((m_date_str(m), m.home, m.away))
            if row is not None:
                mp = _market_over_25(row)
                if mp is not None:
                    market_m.add(mp, 1 if total > 2.5 else 0)
    print(f"\nMatchs évalués : {model_m[2.5].n}")
    print(f"Avec cotes O/U 2.5 (marché) : {market_m.n}")
    print("\n=== CALIBRATION DU MODÈLE PAR SEUIL ===")
    for s in THRESHOLDS:
        mm = model_m[s]
        print(f"\n--- Over/Under {s}  (log-loss {mm.log_loss:.4f}, Brier {mm.brier_score:.4f}) ---")
        print("  tranche      n    prédit  réel")
        for label, n, pred, real in _calib_table(mm.calib):
            flag = "" if abs(pred - real) < 0.05 else "  <-- écart"
            print(f"  {label:<10} {n:>4}  {pred:.3f}  {real:.3f}{flag}")
    print("\n=== MODÈLE vs MARCHÉ (seuil 2.5) ===")
    print(f"  Modèle log-loss : {model_m[2.5].log_loss:.4f}")
    print(f"  Marché log-loss : {market_m.log_loss:.4f}")
    gap = model_m[2.5].log_loss - market_m.log_loss
    print(f"  Écart : {gap:+.4f}  " + ("(le modèle s'approche du marché)" if gap > 0
          else "(le modèle bat le marché — rare, à vérifier)"))
    print("\n=== VERDICT ===")
    print("  (ECE = écart moyen pondéré par effectif ; max = pire tranche)")
    for s in THRESHOLDS:
        ece = _ece(model_m[s].calib)
        gap = _max_calib_gap(model_m[s].calib)
        if ece < 0.03: verdict = "bien calibré"
        elif ece < 0.05: verdict = "acceptable"
        else: verdict = "DÉCALÉ — prudence"
        note = "  (défaut localisé, ensemble OK)" if ece < 0.05 <= gap else ""
        print(f"  Over/Under {s} : ECE {ece:.3f} · max {gap:.3f} → {verdict}{note}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "calibration_ou.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "bin", "count", "avg_predicted", "fraction_over"])
        for s in THRESHOLDS:
            for label, n, pred, real in _calib_table(model_m[s].calib):
                w.writerow([s, label, n, f"{pred:.4f}", f"{real:.4f}"])
    print("\nÉcrit : validation/calibration_ou.csv")
    return model_m, market_m


if __name__ == "__main__":  # pragma: no cover
    run()
