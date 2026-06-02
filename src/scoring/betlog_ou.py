"""Journal de paris OVER/UNDER — couche immuable pour le CLV des totaux."""
from __future__ import annotations
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_OU = _ROOT / "data" / "bets" / "bet_log_ou.csv"
DEFAULT_SCORE_THRESHOLD = 0.05

FIELDS = [
    "logged_at_utc", "match_date", "home", "away",
    "threshold", "side",
    "odds", "p_model", "p_market", "ev", "signal", "reliability", "score",
]


@dataclass(frozen=True)
class OUBetRecord:
    logged_at_utc: str
    match_date: str
    home: str
    away: str
    threshold: float
    side: str
    odds: float
    p_model: float
    p_market: float
    ev: float
    signal: float
    reliability: float
    score: float

    @property
    def identity(self):
        return (self.match_date, self.home, self.away, self.threshold, self.side)


def load_log_ou(path=DEFAULT_LOG_OU):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(OUBetRecord(
                logged_at_utc=row["logged_at_utc"], match_date=row["match_date"],
                home=row["home"], away=row["away"],
                threshold=float(row["threshold"]), side=row["side"],
                odds=float(row["odds"]),
                p_model=float(row["p_model"]), p_market=float(row["p_market"]),
                ev=float(row["ev"]), signal=float(row["signal"]),
                reliability=float(row["reliability"]), score=float(row["score"]),
            ))
    return out


def existing_identities_ou(path=DEFAULT_LOG_OU):
    return {r.identity for r in load_log_ou(path)}


def record_bets_ou(ou_match_scores, match_dates, threshold=DEFAULT_SCORE_THRESHOLD,
                   log_path=DEFAULT_LOG_OU, now=None):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    seen = existing_identities_ou(log_path)
    new_records = []
    for ms in ou_match_scores:
        md = match_dates.get((ms.home, ms.away), "")
        for s in ms.issues:
            if s.score < threshold:
                continue
            rec = OUBetRecord(
                logged_at_utc=stamp, match_date=md,
                home=ms.home, away=ms.away,
                threshold=s.threshold, side=s.side,
                odds=round(s.odds, 4), p_model=round(s.p_model, 4),
                p_market=round(s.p_market, 4), ev=round(s.ev, 4),
                signal=round(s.signal, 4), reliability=round(s.reliability, 4),
                score=round(s.score, 4),
            )
            if rec.identity in seen:
                continue
            seen.add(rec.identity)
            new_records.append(rec)
    if new_records:
        is_new = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if is_new:
                w.writeheader()
            for r in new_records:
                w.writerow(asdict(r))
    return new_records
