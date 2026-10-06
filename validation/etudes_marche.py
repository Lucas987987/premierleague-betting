"""Études de structure de marché Premier League — transposition des études tennis.

Lit les hypothèses gelées (frozen_hypotheses.json) et les évalue sur un bloc
temporel : IS (exploration) ou OOS (une seule fois, Holm sur la famille).
Aucun paramètre n'est réglable en ligne de commande : tout vient du gel.

Usage :
    python validation/etudes/etudes_marche.py --bloc IS
    python validation/etudes/etudes_marche.py --bloc OOS

Sorties : data/validation/etudes_<bloc>.csv + résumé markdown sur stdout
(redirigé vers $GITHUB_STEP_SUMMARY par le workflow).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FREEZE = Path(__file__).with_name("frozen_hypotheses.json")
MATCHES = ROOT / "data" / "processed" / "matches.csv"
OUT_DIR = ROOT / "data" / "validation"

ISSUES = ("H", "D", "A")
SOFT_BOOKS = ("B365", "BW", "IW", "WH", "VC")   # hors exchange
MOVE = 0.03
LONGSHOT = 0.20
SEED = 20261006


# ------------------------------------------------------------------ données
def _fair(df: pd.DataFrame, prefix: str, close: bool) -> pd.DataFrame:
    c = "C" if close else ""
    inv = pd.DataFrame({i: 1 / df[f"{prefix}{c}{i}"] for i in ISSUES})
    s = inv.sum(axis=1)
    out = inv.div(s, axis=0)
    out["margin"] = s - 1
    return out


def load(bloc: str, freeze: dict) -> pd.DataFrame:
    seasons = freeze["decoupage"][bloc]
    d = pd.read_csv(MATCHES, dtype={"season": str})
    d = d[d["season"].isin(seasons)].copy()
    need = [f"PS{c}{i}" for c in ("", "C") for i in ISSUES]
    d = d.dropna(subset=need + ["FTR"]).reset_index(drop=True)
    d["mid"] = np.arange(len(d))
    return d


def long_format(d: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (match, issue) avec probas juste ouverture/clôture."""
    po, pc = _fair(d, "PS", False), _fair(d, "PS", True)
    rows = []
    for i in ISSUES:
        best = pd.concat([d.get(f"{b}C{i}") for b in SOFT_BOOKS
                          if f"{b}C{i}" in d], axis=1).max(axis=1)
        rows.append(pd.DataFrame({
            "mid": d["mid"], "issue": i,
            "p_open": po[i], "p_close": pc[i],
            "won": (d["FTR"] == i).astype(float),
            "b365c": d.get(f"B365C{i}"), "best_soft_c": best,
        }))
    L = pd.concat(rows, ignore_index=True)
    L["dp"] = L["p_close"] - L["p_open"]
    L["ev_b365"] = L["b365c"] * L["p_close"] - 1
    L["ev_best"] = L["best_soft_c"] * L["p_close"] - 1
    return L


# ---------------------------------------------------------------- stats
def cluster_boot(values: pd.Series, clusters: pd.Series, n: int, rng) -> np.ndarray:
    """Bootstrap de la moyenne, rééchantillonné par match (issues corrélées)."""
    g = pd.DataFrame({"v": values.values, "c": clusters.values}).groupby("c")["v"]
    sums, cnts = g.sum().values, g.count().values
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def summarize(stat: float, boots: np.ndarray, side: str) -> dict:
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = float((boots >= 0).mean() if side == "<" else (boots <= 0).mean())
    return {"stat": stat, "ic_lo": lo, "ic_hi": hi, "p_un_cote": max(p, 1 / len(boots))}


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    e = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def logloss(d: pd.DataFrame, close: bool) -> float:
    pf = _fair(d, "PS", close)
    p = np.array([pf.loc[k, r] for k, r in zip(d.index, d["FTR"])])
    return float(-np.log(p).mean())


# -------------------------------------------------------------- études
def run(bloc: str) -> pd.DataFrame:
    fz = json.loads(FREEZE.read_text())
    rng = np.random.default_rng(SEED)
    B = fz["bootstrap"]
    d = load(bloc, fz)
    L = long_format(d)
    res = []

    # M1 — calibration
    e = ece(L["p_close"].values, L["won"].values)
    ll_o, ll_c = logloss(d, False), logloss(d, True)
    res.append({"id": "M1", "n": len(d), "stat": e, "ic_lo": np.nan, "ic_hi": np.nan,
                "p_un_cote": np.nan, "detail": f"ECE={e:.4f} LL_open={ll_o:.5f} LL_close={ll_c:.5f}",
                "verdict_brut": "OK" if (e < 0.02 and ll_c < ll_o) else "NON"})

    # M2 — dérive du favori
    po = _fair(d, "PS", False)[list(ISSUES)]
    pc = _fair(d, "PS", True)[list(ISSUES)]
    fav = po.values.argmax(1)
    drift = pd.Series(pc.values[np.arange(len(d)), fav] - po.values[np.arange(len(d)), fav])
    b = cluster_boot(drift, d["mid"], B, rng)
    res.append({"id": "M2", "n": len(d), **summarize(drift.mean(), b, "<"), "detail": "Δp favori"})

    # M3 — marges
    for pre, tag in (("B365", "B365"), ("PS", "PS")):
        mo, mc = _fair(d, pre, False)["margin"], _fair(d, pre, True)["margin"]
        dm = (mc - mo).dropna()
        b = cluster_boot(dm, d.loc[dm.index, "mid"], B, rng)
        r = summarize(dm.mean(), b, "<")
        r.update({"id": f"M3_{tag}", "n": len(dm),
                  "detail": f"marge open={mo.mean():.4f} close={mc.mean():.4f} médiane Δ={dm.median():.4f}"})
        res.append(r)

    # M4 — prime de sélection du nul
    E = L.dropna(subset=["ev_best"]).pivot(index="mid", columns="issue", values="ev_best").dropna()
    diff = E["D"] - (E["H"] + E["A"]) / 2
    b = cluster_boot(diff, pd.Series(E.index), B, rng)
    res.append({"id": "M4", "n": len(E), **summarize(diff.mean(), b, ">"),
                "detail": f"EV_best H={E['H'].mean():.4f} D={E['D'].mean():.4f} A={E['A'].mean():.4f}"})

    # M5 — retard bet365 sur gros mouvements
    Lb = L.dropna(subset=["ev_b365"])
    mv = Lb[Lb["dp"] >= MOVE]
    base = Lb["ev_b365"].mean()
    # stat = moyenne(EV|move) − moyenne(EV|tout) ; bootstrap conjoint par match
    vals = Lb["ev_b365"] - base
    w = (Lb["dp"] >= MOVE).astype(float)
    g = pd.DataFrame({"m": Lb["mid"], "v": Lb["ev_b365"], "w": w}).groupby("m")
    s_all, c_all = g["v"].sum().values, g["v"].count().values
    s_mv = g.apply(lambda x: (x["v"] * x["w"]).sum()).values
    c_mv = g["w"].sum().values
    idx = rng.integers(0, len(s_all), size=(B, len(s_all)))
    boots = s_mv[idx].sum(1) / np.maximum(c_mv[idx].sum(1), 1) - s_all[idx].sum(1) / c_all[idx].sum(1)
    roi = (mv["won"] * mv["b365c"] - 1).mean()
    r = summarize(mv["ev_b365"].mean() - base, boots, ">")
    r.update({"id": "M5", "n": len(mv),
              "detail": f"EV|move={mv['ev_b365'].mean():.4f} EV|tout={base:.4f} "
                        f"part EV>0|move={(mv['ev_b365'] > 0).mean():.3f} ROI réalisé={roi:.4f}"})
    res.append(r)

    # M6 — gros-move résiduel
    mv = L[L["dp"] >= MOVE]
    resid = mv["won"] - mv["p_close"]
    b = cluster_boot(resid, mv["mid"], B, rng)
    res.append({"id": "M6", "n": len(mv), **summarize(resid.mean(), b, ">"),
                "detail": f"fréq={mv['won'].mean():.4f} p_close={mv['p_close'].mean():.4f}"})

    # M7 — outsiders
    ls = L[L["p_close"] < LONGSHOT]
    resid = ls["won"] - ls["p_close"]
    b = cluster_boot(resid, ls["mid"], B, rng)
    res.append({"id": "M7", "n": len(ls), **summarize(resid.mean(), b, "<"),
                "detail": f"fréq={ls['won'].mean():.4f} p_close={ls['p_close'].mean():.4f}"})

    R = pd.DataFrame(res)
    R.insert(0, "bloc", bloc)

    # Holm sur les tests (pas M1 descriptif, pas M3_PS qui est le témoin de M3)
    tests = R["id"].isin(["M2", "M3_B365", "M4", "M5", "M6", "M7"])
    p = R.loc[tests, "p_un_cote"].sort_values()
    m, holm, prev = len(p), {}, 0.0
    for k, (ix, pv) in enumerate(p.items()):
        prev = max(prev, min(1.0, (m - k) * pv))
        holm[ix] = prev
    R["p_holm"] = pd.Series(holm)
    return R


def verdicts(R: pd.DataFrame) -> pd.DataFrame:
    def v(r):
        if r["id"] == "M1":
            return r["verdict_brut"]
        if r["id"] == "M3_B365":
            ok = abs(float(r["detail"].split("médiane Δ=")[1])) < 0.005
            return "CONFIRMÉE" if ok else "REJETÉE"
        if r["id"] == "M3_PS":
            return "témoin: compression" if r["ic_hi"] < 0 else "témoin: pas de compression"
        sig = r["p_holm"] < 0.05 if r["bloc"] == "OOS" else r["p_un_cote"] < 0.05
        return "CONFIRMÉE" if sig else "NON CONFIRMÉE"
    R["verdict"] = R.apply(v, axis=1)
    return R


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloc", choices=["IS", "OOS"], required=True)
    a = ap.parse_args()
    R = verdicts(run(a.bloc))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT_DIR / f"etudes_{a.bloc}.csv", index=False, float_format="%.5f")
    print(f"## Études de marché Premier League — bloc {a.bloc}\n")
    print("| id | n | stat | IC95 | p | p Holm | verdict | détail |\n|---|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        ic = "" if pd.isna(r["ic_lo"]) else f"[{r['ic_lo']:+.4f}; {r['ic_hi']:+.4f}]"
        ph = "" if pd.isna(r.get("p_holm")) else f"{r['p_holm']:.3f}"
        pu = "" if pd.isna(r["p_un_cote"]) else f"{r['p_un_cote']:.3f}"
        print(f"| {r['id']} | {r['n']} | {r['stat']:+.4f} | {ic} | {pu} | {ph} | {r['verdict']} | {r['detail']} |")


if __name__ == "__main__":
    main()
