"""FBref ingestion via soccerdata for non–World Cup match volume."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from features.tactical_matrix import TACTICAL_MATRIX_COLUMNS


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from soccerdata if present."""
    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = [
            "_".join(str(x) for x in col if str(x) != "").strip()
            for col in frame.columns.values
        ]
    return frame


def _pick_column(frame: pd.DataFrame, patterns: list[str]) -> str | None:
    """Find first column whose name contains any pattern (case-insensitive)."""
    for pat in patterns:
        for col in frame.columns:
            if pat.lower() in str(col).lower():
                return col
    return None


def fetch_fbref_team_match_stats(
    *,
    leagues: list[str],
    seasons: list[str],
) -> pd.DataFrame | None:
    """Download team-level match stats (passing + defense) from FBref.

    Args:
        leagues: soccerdata league keys (e.g. ``ENG-Premier League``).
        seasons: Season codes like ``2324``.

    Returns:
        Long-format team-match table or ``None`` on failure.
    """
    try:
        import soccerdata as sd
    except ImportError:
        return None

    try:
        fb = sd.FBref(leagues=leagues, seasons=seasons)
        passing = _flatten_columns(fb.read_team_match_stats(stat_type="passing"))
        defense = _flatten_columns(fb.read_team_match_stats(stat_type="defense"))
    except (OSError, ValueError, RuntimeError, KeyError):
        return None

    passing = passing.reset_index()
    defense = defense.reset_index()
    merge_keys = [c for c in passing.columns if c in defense.columns and c not in ("team",)]
    on_cols = [c for c in merge_keys if c in ("game", "date", "team", "opponent", "league", "season")]
    if not on_cols:
        on_cols = [c for c in ("game", "team") if c in passing.columns and c in defense.columns]
    merged = passing.merge(defense, on=on_cols, how="left", suffixes=("", "_def"))
    return merged


def fbref_stats_to_tactical_rows(stats: pd.DataFrame) -> pd.DataFrame:
    """Map FBref team-match stats into :data:`TACTICAL_MATRIX_COLUMNS` (partial)."""
    frame = stats.copy()
    poss_col = _pick_column(frame, ["poss", "possession"])
    prog_col = _pick_column(frame, ["prog", "prgp", "progressive"])
    long_col = _pick_column(frame, ["long", "lng"])
    cross_col = _pick_column(frame, ["cross"])
    third_col = _pick_column(frame, ["1/3", "final", "att3rd", "att third"])
    press_col = _pick_column(frame, ["press", "pressures"])
    xg_col = _pick_column(frame, ["xg", "xG", "npxg"])
    team_col = _pick_column(frame, ["team"])
    game_col = _pick_column(frame, ["game", "match"])

    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        team = str(r.get(team_col, "")) if team_col else ""
        game = str(r.get(game_col, "")) if game_col else ""
        if not team or not game:
            continue
        match_id = f"fbref_{re.sub(r'[^a-zA-Z0-9]+', '_', game)}_{re.sub(r'[^a-zA-Z0-9]+', '_', team)}"
        rows.append(
            {
                "team": team,
                "match_id": match_id,
                "possession": float(r[poss_col]) if poss_col and pd.notna(r.get(poss_col)) else None,
                "ppda": None,
                "high_turnovers": None,
                "progressive_passes": float(r[prog_col]) if prog_col and pd.notna(r.get(prog_col)) else None,
                "progressive_carries": None,
                "long_balls": float(r[long_col]) if long_col and pd.notna(r.get(long_col)) else None,
                "crosses": float(r[cross_col]) if cross_col and pd.notna(r.get(cross_col)) else None,
                "counter_attacks": None,
                "transition_speed": None,
                "final_third_entries": float(r[third_col]) if third_col and pd.notna(r.get(third_col)) else None,
                "defensive_line_height": None,
                "passes_per_sequence": None,
                "xg": float(r[xg_col]) if xg_col and pd.notna(r.get(xg_col)) else None,
                "xga": None,
                "shot_distance": None,
                "press_success_rate": None,
                "pressing_intensity": float(r[press_col]) if press_col and pd.notna(r.get(press_col)) else None,
                "data_source": "fbref",
            },
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=list(TACTICAL_MATRIX_COLUMNS) + ["data_source"])
    for col in TACTICAL_MATRIX_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out.reindex(columns=list(TACTICAL_MATRIX_COLUMNS) + ["data_source"])


def fetch_fbref_schedule(
    *,
    leagues: list[str],
    seasons: list[str],
) -> pd.DataFrame | None:
    """Fetch match schedule (scores, teams, date) from FBref."""
    try:
        import soccerdata as sd
    except ImportError:
        return None
    try:
        fb = sd.FBref(leagues=leagues, seasons=seasons)
        sched = _flatten_columns(fb.read_schedule())
        return sched.reset_index()
    except (OSError, ValueError, RuntimeError, KeyError):
        return None


def schedule_to_matches(schedule: pd.DataFrame, *, competition_label: str) -> pd.DataFrame:
    """Map FBref schedule rows to canonical match schema columns."""
    frame = schedule.copy()
    date_col = _pick_column(frame, ["date"])
    home_col = _pick_column(frame, ["home"])
    away_col = _pick_column(frame, ["away"])
    hg_col = _pick_column(frame, ["home_score", "score_home"])
    ag_col = _pick_column(frame, ["away_score", "score_away"])
    game_col = _pick_column(frame, ["game", "match"])

    rows: list[dict[str, Any]] = []
    for _, r in frame.iterrows():
        game = str(r.get(game_col, "")) if game_col else ""
        home = str(r.get(home_col, "")) if home_col else ""
        away = str(r.get(away_col, "")) if away_col else ""
        if not home or not away:
            continue
        mid = f"fbref_{re.sub(r'[^a-zA-Z0-9]+', '_', game)}"
        rows.append(
            {
                "match_id": mid,
                "date": str(r.get(date_col, ""))[:10] if date_col else "",
                "competition": competition_label,
                "home_team": home,
                "away_team": away,
                "home_score": int(r[hg_col]) if hg_col and pd.notna(r.get(hg_col)) else 0,
                "away_score": int(r[ag_col]) if ag_col and pd.notna(r.get(ag_col)) else 0,
                "venue": "",
                "neutral_ground": False,
                "stage": "League",
                "attendance": None,
                "weather": None,
                "data_source": "fbref",
            },
        )
    return pd.DataFrame(rows)


def ingest_fbref_bundle(
    *,
    leagues: list[str],
    seasons: list[str],
    competition_labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch matches + tactical proxy rows for configured leagues.

    Args:
        leagues: soccerdata league identifiers.
        seasons: Season codes.
        competition_labels: Optional map league key -> competition name.

    Returns:
        ``(matches, tactical_matrix)`` DataFrames (may be empty).
    """
    all_matches: list[pd.DataFrame] = []
    all_tactical: list[pd.DataFrame] = []

    for league in leagues:
        label = (competition_labels or {}).get(league, league)
        sched = fetch_fbref_schedule(leagues=[league], seasons=seasons)
        if sched is not None and not sched.empty:
            all_matches.append(schedule_to_matches(sched, competition_label=label))
        stats = fetch_fbref_team_match_stats(leagues=[league], seasons=seasons)
        if stats is not None and not stats.empty:
            all_tactical.append(fbref_stats_to_tactical_rows(stats))

    matches = pd.concat(all_matches, ignore_index=True) if all_matches else pd.DataFrame()
    tactical = pd.concat(all_tactical, ignore_index=True) if all_tactical else pd.DataFrame()
    return matches, tactical
