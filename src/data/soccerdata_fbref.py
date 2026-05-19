"""Optional FBref access via soccerdata (network required)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def try_read_fbref_team_season_shooting(
    *,
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
) -> pd.DataFrame | None:
    """Best-effort season shooting table from FBref / soccerdata.

    Args:
        leagues: soccerdata league ids; default single Premier League season.
        seasons: Five-digit seasons like ``["2324"]``.

    Returns:
        A DataFrame when soccerdata succeeds; ``None`` on import or IO errors.
    """
    try:
        import soccerdata as sd
    except ImportError:
        return None
    lg = leagues or ["ENG-Premier League"]
    ss = seasons or ["2324"]
    try:
        fb = sd.FBref(leagues=lg, seasons=ss)
        return fb.read_team_season_stats(stat_type="shooting")
    except (OSError, ValueError, RuntimeError):
        return None
