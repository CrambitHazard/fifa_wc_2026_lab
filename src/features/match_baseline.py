"""Rolling form and goals for baseline match models."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd


def outcome_label(home_goals: int, away_goals: int) -> int:
    """Multiclass label: ``0`` away win, ``1`` draw, ``2`` home win.

    Args:
        home_goals: Full-time home goals.
        away_goals: Full-time away goals.

    Returns:
        Integer class for sklearn ``predict`` style models.
    """
    if home_goals > away_goals:
        return 2
    if home_goals < away_goals:
        return 0
    return 1


def _team_points_from_fixture(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def build_baseline_feature_table(
    matches: pd.DataFrame,
    *,
    window: int = 5,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    hg_col: str = "home_score",
    ag_col: str = "away_score",
    neutral_col: str = "neutral_ground",
    match_id_col: str = "match_id",
) -> pd.DataFrame:
    """Chronological rolling form / goals joined to each match row.

    Requires columns from :func:`features.elo.attach_pre_match_elo`
    (``elo_home_before``, ``elo_away_before``).

    Args:
        matches: Table sorted by date with Elo columns already attached.
        window: Look-back length for rolling means / sums.
        date_col: Date column name.
        home_col: Home team column.
        away_col: Away team column.
        hg_col: Home goals column.
        ag_col: Away goals column.
        neutral_col: Neutral-ground flag.
        match_id_col: Primary key column.

    Returns:
        Modeling table with ``y`` and numeric features prior to kickoff.

    """
    frame = matches.copy()
    frame["_d"] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values("_d").reset_index(drop=True)

    history: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        h = str(row[home_col])
        a = str(row[away_col])
        hg = int(row[hg_col])
        ag = int(row[ag_col])

        def _roll(team: str) -> tuple[float, float, float]:
            seq = history[team][-window:]
            if not seq:
                return 0.0, 0.0, 0.0
            pts = sum(p for _, __, p in seq)
            gfs = [gf for gf, _, __ in seq]
            gas = [ga for _, ga, __ in seq]
            return pts, sum(gfs) / len(gfs), sum(gas) / len(gas)

        ph, gfh, gah = _roll(h)
        pa, gfa, gaa = _roll(a)

        rows.append(
            {
                match_id_col: row[match_id_col],
                date_col: row[date_col],
                home_col: h,
                away_col: a,
                "y": outcome_label(hg, ag),
                "form_points_home": ph,
                "form_points_away": pa,
                "avg_gf_home": gfh,
                "avg_ga_home": gah,
                "avg_gf_away": gfa,
                "avg_ga_away": gaa,
                "elo_home_before": float(row["elo_home_before"]),
                "elo_away_before": float(row["elo_away_before"]),
                "elo_diff": float(row["elo_home_before"] - row["elo_away_before"]),
                neutral_col: float(bool(row.get(neutral_col, False))),
            },
        )

        history[h].append((hg, ag, _team_points_from_fixture(hg, ag)))
        history[a].append((ag, hg, _team_points_from_fixture(ag, hg)))

    out = pd.DataFrame(rows)
    return out
