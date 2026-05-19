"""Orchestrate ingestion into ``data/processed`` parquet bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.schemas import (
    LINEUP_COLUMNS,
    MATCH_COLUMNS,
    PENALTY_COLUMNS,
    SCHEMA_VERSION,
    SHOT_COLUMNS,
    TEAM_TACTICAL_COLUMNS,
)
from data.statsbomb import (
    default_open_data_root,
    events_path,
    events_to_shots_and_penalties,
    events_to_team_tactical,
    lineups_path,
    lineups_to_dataframe,
    load_events_file,
    load_matches_file,
    matches_path,
    matches_to_dataframe,
    raw_match_competition,
)


def run_statsbomb_etl(
    *,
    open_data_root: Path | None,
    competition_id: int,
    season_id: int,
    processed_dir: Path,
    skip_missing_events: bool = True,
) -> dict[str, Any]:
    """Build unified parquet tables for one competition-season.

    Args:
        open_data_root: StatsBomb Open Data checkout (uses
            :func:`data.statsbomb.default_open_data_root` if omitted).
        competition_id: StatsBomb competition id (e.g. ``43`` for WC).
        season_id: StatsBomb season id (e.g. ``3`` for 2018).
        processed_dir: Output directory (typically ``data/processed``).
        skip_missing_events: If True, matches without event files produce empty
            shot/tactical tables for that id instead of raising.

    Returns:
        Metadata dict with row counts and schema version.
    """
    root = open_data_root or default_open_data_root()
    mp = matches_path(root, competition_id, season_id)
    raw = load_matches_file(mp)

    matches_df = matches_to_dataframe(raw)

    shots_parts: list[pd.DataFrame] = []
    pens_parts: list[pd.DataFrame] = []
    lineup_parts: list[pd.DataFrame] = []
    tact_parts: list[pd.DataFrame] = []

    for rec in raw:
        mid = int(rec["match_id"])
        m_id_s = str(mid)
        comp_label = raw_match_competition(rec)
        home = str((rec.get("home_team") or {}).get("home_team_name", ""))
        away = str((rec.get("away_team") or {}).get("away_team_name", ""))
        stage = str((rec.get("competition_stage") or {}).get("name", ""))

        ev_path = events_path(root, mid)
        if not ev_path.is_file():
            if not skip_missing_events:
                msg = f"Missing events for match {mid}: {ev_path}"
                raise FileNotFoundError(msg)
            events: list = []
        else:
            events = load_events_file(ev_path)

        sf, pf = events_to_shots_and_penalties(
            events,
            match_id=m_id_s,
            competition=str(comp_label),
            match_label=stage,
        )
        shots_parts.append(sf)
        pens_parts.append(pf)

        lp = lineups_path(root, mid)
        if lp.is_file():
            with lp.open(encoding="utf-8") as fh:
                lineup_raw = json.load(fh)
        else:
            lineup_raw = []
        lineup_parts.append(lineups_to_dataframe(lineup_raw, m_id_s))
        tact_parts.append(
            events_to_team_tactical(
                events,
                match_id=m_id_s,
                home_team=home,
                away_team=away,
            ),
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    def _concat(parts: list[pd.DataFrame], columns: tuple[str, ...]) -> pd.DataFrame:
        non_empty = [p for p in parts if len(p) > 0]
        if not non_empty:
            return pd.DataFrame(columns=list(columns))
        out = pd.concat(non_empty, ignore_index=True)
        return out.reindex(columns=list(columns))

    shots_df = _concat(shots_parts, SHOT_COLUMNS)
    pens_df = _concat(pens_parts, PENALTY_COLUMNS)
    lineup_df = _concat(lineup_parts, LINEUP_COLUMNS)
    tact_df = _concat(tact_parts, TEAM_TACTICAL_COLUMNS)

    matches_out = processed_dir / "matches.parquet"
    shots_out = processed_dir / "shots_open_play.parquet"
    pens_out = processed_dir / "penalties.parquet"
    lineup_out = processed_dir / "lineups.parquet"
    tact_out = processed_dir / "team_tactical.parquet"

    matches_df.reindex(columns=list(MATCH_COLUMNS)).to_parquet(matches_out, index=False)
    shots_df.to_parquet(shots_out, index=False)
    pens_df.to_parquet(pens_out, index=False)
    lineup_df.to_parquet(lineup_out, index=False)
    tact_df.to_parquet(tact_out, index=False)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "competition_id": competition_id,
        "season_id": season_id,
        "open_data_root": str(root),
        "rows": {
            "matches": len(matches_df),
            "shots_open_play": len(shots_df),
            "penalties": len(pens_df),
            "lineups": len(lineup_df),
            "team_tactical": len(tact_df),
        },
    }
    with (processed_dir / "etl_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta
