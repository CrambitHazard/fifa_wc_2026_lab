"""Baseline training smoke test on synthetic schedules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.elo import attach_pre_match_elo
from features.match_baseline import build_baseline_feature_table
from models.baseline_match import train_baselines


def _fake_schedule(n: int = 60) -> pd.DataFrame:
    """Build a random double round-robin–like list of fixtures."""
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(8)]
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rows: list[dict] = []
    for mid in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        rows.append(
            {
                "match_id": str(1000 + mid),
                "date": idx[mid].strftime("%Y-%m-%d"),
                "home_team": h,
                "away_team": a,
                "home_score": hg,
                "away_score": ag,
                "venue": "V",
                "neutral_ground": False,
                "stage": "L",
                "competition": "Synth",
                "attendance": None,
                "weather": None,
            },
        )
    return pd.DataFrame(rows)


def test_train_baselines_runs() -> None:
    """Enough history should let both models fit without error."""
    frame = _fake_schedule(60)
    with_elo = attach_pre_match_elo(frame)
    feats = build_baseline_feature_table(with_elo)
    suite = train_baselines(feats, test_fraction=0.3, random_state=0)
    assert np.isfinite(suite.report_logistic.log_loss)
    assert suite.report_logistic.accuracy >= 0.0
