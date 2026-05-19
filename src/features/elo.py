"""Elo ratings over a chronological match list with optional home advantage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    """Hyperparameters for sequential Elo updates.

    Attributes:
        base_rating: Starting rating for unseen teams.
        k_factor: Update strength after each match.
        home_adv: Extra rating points credited to the home side before
            expected score (set to ``0`` when ``neutral_ground`` is true).
    """

    base_rating: float = 1500.0
    k_factor: float = 20.0
    home_adv: float = 100.0


def _expected_score(r_a: float, r_b: float) -> float:
    """Expected score for side A (``1 -`` for side B)."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _score_pair_for_home(home_goals: int, away_goals: int) -> tuple[float, float]:
    """Return (s_home, s_away) in ``{0, 0.5, 1}``."""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def attach_pre_match_elo(
    matches: pd.DataFrame,
    *,
    cfg: EloConfig | None = None,
    date_col: str = "date",
    home_col: str = "home_team",
    away_col: str = "away_team",
    hg_col: str = "home_score",
    ag_col: str = "away_score",
    neutral_col: str = "neutral_ground",
) -> pd.DataFrame:
    """Sort by ``date`` and add pre-match Elo columns.

    Args:
        matches: Match-level table (one row per fixture).
        cfg: Elo configuration; default is standard recreational constants.
        date_col: Column parseable as datetimes.
        home_col: Home team label column.
        away_col: Away team label column.
        hg_col: Full-time home goals.
        ag_col: Full-time away goals.
        neutral_col: If true, home advantage is disabled for that row.

    Returns:
        Copy of ``matches`` with ``elo_home_before``, ``elo_away_before``,
        and sorted chronologically.

    """
    c = cfg or EloConfig()
    frame = matches.copy()
    frame["_sort_date"] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values("_sort_date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    elo_h: list[float] = []
    elo_a: list[float] = []

    for _, row in frame.iterrows():
        h = str(row[home_col])
        a = str(row[away_col])
        rh = ratings.get(h, c.base_rating)
        ra = ratings.get(a, c.base_rating)
        neutral = bool(row.get(neutral_col, False))
        adv = 0.0 if neutral else c.home_adv
        rh_eff = rh + adv
        eh = _expected_score(rh_eff, ra)
        ea = 1.0 - eh
        s_h, s_a = _score_pair_for_home(int(row[hg_col]), int(row[ag_col]))

        elo_h.append(ratings.get(h, c.base_rating))
        elo_a.append(ratings.get(a, c.base_rating))

        rht = rh + c.k_factor * (s_h - eh)
        rat = ra + c.k_factor * (s_a - ea)
        ratings[h] = rht
        ratings[a] = rat

    out = frame.drop(columns=["_sort_date"])
    out["elo_home_before"] = elo_h
    out["elo_away_before"] = elo_a
    return out
