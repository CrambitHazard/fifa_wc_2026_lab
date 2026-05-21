"""Build the Phase 2 tactical feature matrix from matches + Open Data events."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.scope_filters import is_statsbomb_event_match_id
from data.statsbomb import default_open_data_root, events_path, load_events_file
from features.tactical_events import counts_to_row, extract_match_tactical_counts

TACTICAL_MATRIX_COLUMNS: tuple[str, ...] = (
    "team",
    "match_id",
    "possession",
    "ppda",
    "high_turnovers",
    "progressive_passes",
    "progressive_carries",
    "long_balls",
    "crosses",
    "counter_attacks",
    "transition_speed",
    "final_third_entries",
    "defensive_line_height",
    "passes_per_sequence",
    "xg",
    "xga",
    "shot_distance",
    "press_success_rate",
    "pressing_intensity",
)

RATE_COLUMNS: tuple[str, ...] = tuple(
    c for c in TACTICAL_MATRIX_COLUMNS if c not in ("team", "match_id")
)


def build_tactical_matrix(
    matches: pd.DataFrame,
    *,
    open_data_root: Path | None = None,
) -> pd.DataFrame:
    """Extract one row per team per match from StatsBomb events.

    Args:
        matches: ``matches.parquet`` with ``match_id``, ``home_team``, ``away_team``.
        open_data_root: Open Data checkout; defaults to :func:`data.statsbomb.default_open_data_root`.

    Returns:
        Tactical matrix aligned to :data:`RATE_COLUMNS`.
    """
    root = open_data_root or default_open_data_root()
    rows: list[dict] = []
    for _, m in matches.iterrows():
        m_id_s = str(m["match_id"])
        if not is_statsbomb_event_match_id(m_id_s):
            continue
        mid = int(m_id_s)
        home = str(m["home_team"])
        away = str(m["away_team"])
        ev_path = events_path(root, mid)
        events = load_events_file(ev_path)
        if not events:
            continue
        counts = extract_match_tactical_counts(events, home_team=home, away_team=away)
        rows.append(counts_to_row(home, m_id_s, counts[home], counts[away]))
        rows.append(counts_to_row(away, m_id_s, counts[away], counts[home]))
    if not rows:
        return pd.DataFrame(columns=list(TACTICAL_MATRIX_COLUMNS))
    return pd.DataFrame(rows).reindex(columns=list(TACTICAL_MATRIX_COLUMNS))
