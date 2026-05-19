"""StatsBomb Open Data JSON -> canonical :mod:`data.schemas` tables.

Expects the public repo layout (``data/matches/{competition_id}/{season_id}.json``,
``data/events/{match_id}.json``, ``data/lineups/{match_id}.json``).

Pitch convention: StatsBomb **120×80** units; goal line at ``x = 120``, center
``y = 40``. Distance and opening angle use that frame for consistency across
matches ingested from this source.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from data.schemas import (
    LINEUP_COLUMNS,
    LINEUP_JERSEY,
    LINEUP_MATCH_ID,
    LINEUP_PLAYER,
    LINEUP_PLAYER_ID,
    LINEUP_POSITION,
    LINEUP_STARTER,
    LINEUP_TEAM,
    MATCH_ATTENDANCE,
    MATCH_AWAY_SCORE,
    MATCH_AWAY_TEAM,
    MATCH_COLUMNS,
    MATCH_COMPETITION,
    MATCH_DATE,
    MATCH_HOME_SCORE,
    MATCH_HOME_TEAM,
    MATCH_ID,
    MATCH_NEUTRAL,
    MATCH_STAGE,
    MATCH_VENUE,
    MATCH_WEATHER,
    PENALTY_COLUMNS,
    PEN_COMPETITION,
    PEN_FOOT,
    PEN_ID,
    PEN_KEEPER,
    PEN_MATCH_PRESSURE,
    PEN_MINUTE,
    PEN_SCORE_STATE,
    PEN_SCORED,
    PEN_SHOOTER,
    PEN_ZONE,
    SHOT_ANGLE,
    SHOT_ASSIST_TYPE,
    SHOT_BODY_PART,
    SHOT_COLUMNS,
    SHOT_DISTANCE,
    SHOT_ID,
    SHOT_IS_GOAL,
    SHOT_PLAYER,
    SHOT_TEAM,
    SHOT_UNDER_PRESSURE,
    SHOT_X,
    SHOT_Y,
    TEAM_TACTICAL_COLUMNS,
    TACT_COUNTER,
    TACT_MATCH_ID,
    TACT_POSSESSION,
    TACT_PPDA,
    TACT_PRESSING,
    TACT_PROG_PASSES,
    TACT_TEAM,
    TACT_TRANSITION,
    TACT_XGA,
    TACT_XG,
)


def resolve_open_data_data_dir(open_data_root: Path) -> Path:
    """Return the directory that contains ``matches/``, ``events/``, ``lineups/``.

    Supports both official clone layouts:

    - ``{root}/data/matches/...`` (full GitHub repo checkout)
    - ``{root}/matches/...`` (inner ``data/`` folder extracted to ``data/external``)
    """
    root = open_data_root.resolve()
    nested = root / "data"
    if (nested / "matches").is_dir():
        return nested
    if (root / "matches").is_dir():
        return root
    return nested


def default_open_data_root() -> Path:
    """Resolve Open Data root: env, flat ``data/external``, or nested clone path."""
    env = os.environ.get("STATSBOMB_OPEN_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    external = repo / "data" / "external"
    nested_clone = external / "statsbomb-open-data"
    for candidate in (external, nested_clone):
        if (candidate / "matches").is_dir() or (candidate / "data" / "matches").is_dir():
            return candidate.resolve()
    return nested_clone.resolve()


def matches_path(open_data_root: Path, competition_id: int, season_id: int) -> Path:
    """Path to the StatsBomb matches JSON file."""
    base = resolve_open_data_data_dir(open_data_root)
    return base / "matches" / str(competition_id) / f"{season_id}.json"


def events_path(open_data_root: Path, match_id: int) -> Path:
    """Path to per-match events JSON."""
    base = resolve_open_data_data_dir(open_data_root)
    return base / "events" / f"{match_id}.json"


def lineups_path(open_data_root: Path, match_id: int) -> Path:
    """Path to per-match lineups JSON."""
    base = resolve_open_data_data_dir(open_data_root)
    return base / "lineups" / f"{match_id}.json"


def load_matches_file(path: Path) -> list[dict[str, Any]]:
    """Load a StatsBomb matches JSON array.

    Args:
        path: File produced by Open Data under ``data/matches/``.

    Returns:
        List of raw match dicts.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        msg = f"matches file not found: {path}"
        raise FileNotFoundError(msg)
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_events_file(path: Path) -> list[dict[str, Any]]:
    """Load events for one match."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def raw_match_competition(match: dict[str, Any]) -> str:
    comp = match.get("competition") or {}
    sea = match.get("season") or {}
    cname = comp.get("competition_name") or "unknown_competition"
    sname = sea.get("season_name") or ""
    return f"{cname} {sname}".strip()


def matches_to_dataframe(raw_matches: list[dict[str, Any]]) -> pd.DataFrame:
    """Map raw StatsBomb matches to :data:`~data.schemas.MATCH_COLUMNS`."""
    rows: list[dict[str, Any]] = []
    for m in raw_matches:
        home = m.get("home_team") or {}
        away = m.get("away_team") or {}
        stadium = m.get("stadium") or {}
        stage = m.get("competition_stage") or {}
        rows.append(
            {
                MATCH_ID: str(m.get("match_id", "")),
                MATCH_DATE: str(m.get("match_date", "")),
                MATCH_COMPETITION: raw_match_competition(m),
                MATCH_HOME_TEAM: str(home.get("home_team_name", "")),
                MATCH_AWAY_TEAM: str(away.get("away_team_name", "")),
                MATCH_HOME_SCORE: int(m.get("home_score", 0)),
                MATCH_AWAY_SCORE: int(m.get("away_score", 0)),
                MATCH_VENUE: str(stadium.get("name", "")),
                MATCH_NEUTRAL: bool(m.get("neutral_ground", False)),
                MATCH_STAGE: str(stage.get("name", "")),
                MATCH_ATTENDANCE: m.get("attendance"),
                MATCH_WEATHER: None,
            },
        )
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=list(MATCH_COLUMNS))


def sb_distance_angle(x: float, y: float) -> tuple[float, float]:
    """Distance to goal mouth center and opening angle (radians) in SB frame.

    Args:
        x: Pitch x (0–120).
        y: Pitch y (0–80).

    Returns:
        Euclidean distance to ``(120, 40)`` and angle between vectors to goal
        posts placed at ``y ≈ 36.34`` and ``y ≈ 43.66``.
    """
    p = np.array([float(x), float(y)], dtype=float)
    left = np.array([120.0, 36.34])
    right = np.array([120.0, 43.66])
    v1 = left - p
    v2 = right - p
    dist = float(np.hypot(120.0 - x, 40.0 - y))
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    ang = float(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)))
    return dist, ang


def _pick_keeper_from_freeze(shot_block: dict[str, Any]) -> str:
    ff = shot_block.get("freeze_frame") or []
    for ent in ff:
        pl = ent.get("player") or {}
        pos = ent.get("position") or {}
        name = pos.get("name") or ""
        if not ent.get("teammate") and "Goalkeeper" in name:
            return str(pl.get("name") or "")
    return ""


def _penalty_zone(end_loc: list[float] | None) -> str | None:
    if not end_loc or len(end_loc) < 2:
        return None
    ex, ey = float(end_loc[0]), float(end_loc[1])
    # Very coarse grid for Week-1 EDA (refine later).
    if ex >= 118:
        tier = "mouth"
    elif ex >= 114:
        tier = "close"
    else:
        tier = "far"
    side = "center"
    if ey < 38:
        side = "low_y"
    elif ey > 42:
        side = "high_y"
    return f"{tier}_{side}"


def _is_penalty_shot(shot_block: dict[str, Any]) -> bool:
    st = shot_block.get("type") or {}
    return st.get("name") == "Penalty"


def _is_open_play_shot(shot_block: dict[str, Any]) -> bool:
    return not _is_penalty_shot(shot_block)


def iter_shot_and_penalty_rows(
    events: list[dict[str, Any]],
    *,
    match_id: str,
    competition: str,
    match_label: str | None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (\"shot\"|\"penalty\", row_dict) from event stream."""
    for ev in events:
        if (ev.get("type") or {}).get("name") != "Shot":
            continue
        shot = ev.get("shot") or {}
        if not shot:
            continue
        loc = ev.get("location") or [0.0, 0.0]
        if len(loc) < 2:
            continue
        x, y = float(loc[0]), float(loc[1])
        dist, ang = sb_distance_angle(x, y)
        outcome = (shot.get("outcome") or {}).get("name") or ""
        is_goal = outcome == "Goal"
        body = (shot.get("body_part") or {}).get("name") or ""
        player = (ev.get("player") or {}).get("name") or ""
        team = (ev.get("team") or {}).get("name") or ""
        under = ev.get("under_pressure")
        assist = "key_pass" if shot.get("key_pass_id") else ""

        if _is_penalty_shot(shot):
            keeper = _pick_keeper_from_freeze(shot)
            end_loc = shot.get("end_location")
            if isinstance(end_loc, list) and len(end_loc) >= 2:
                zone = _penalty_zone([float(end_loc[0]), float(end_loc[1])])
            else:
                zone = None
            prow = {
                PEN_ID: str(ev.get("id", "")),
                PEN_SHOOTER: player,
                PEN_KEEPER: keeper,
                PEN_FOOT: body,
                PEN_ZONE: zone,
                PEN_SCORED: is_goal,
                PEN_MATCH_PRESSURE: match_label,
                PEN_SCORE_STATE: None,
                PEN_MINUTE: ev.get("minute"),
                PEN_COMPETITION: competition,
            }
            yield "penalty", prow
        elif _is_open_play_shot(shot):
            srow = {
                SHOT_ID: str(ev.get("id", "")),
                SHOT_PLAYER: player,
                SHOT_TEAM: team,
                SHOT_X: x,
                SHOT_Y: y,
                SHOT_DISTANCE: dist,
                SHOT_ANGLE: ang,
                SHOT_BODY_PART: body,
                SHOT_ASSIST_TYPE: assist,
                SHOT_UNDER_PRESSURE: under,
                SHOT_IS_GOAL: is_goal,
            }
            yield "shot", srow


def events_to_shots_and_penalties(
    events: list[dict[str, Any]],
    *,
    match_id: str,
    competition: str,
    match_label: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split open-play shots and penalties into two aligned frames."""
    shots: list[dict[str, Any]] = []
    pens: list[dict[str, Any]] = []
    for kind, row in iter_shot_and_penalty_rows(
        events,
        match_id=match_id,
        competition=competition,
        match_label=match_label,
    ):
        if kind == "shot":
            shots.append(row)
        else:
            pens.append(row)
    sf = pd.DataFrame(shots)
    pf = pd.DataFrame(pens)
    if sf.empty:
        sf = pd.DataFrame(columns=list(SHOT_COLUMNS))
    else:
        sf = sf.reindex(columns=list(SHOT_COLUMNS))
    if pf.empty:
        pf = pd.DataFrame(columns=list(PENALTY_COLUMNS))
    else:
        pf = pf.reindex(columns=list(PENALTY_COLUMNS))
    return sf, pf


def lineups_to_dataframe(raw: list[dict[str, Any]], match_id: str) -> pd.DataFrame:
    """Flatten StatsBomb lineups JSON to :data:`~data.schemas.LINEUP_COLUMNS`."""
    rows: list[dict[str, Any]] = []
    for team_block in raw:
        tname = str(team_block.get("team_name", ""))
        for pl in team_block.get("lineup") or []:
            positions = pl.get("positions") or []
            starter = False
            pos_label = ""
            if positions:
                p0 = positions[0]
                pos_label = str(p0.get("position", ""))
                starter = str(p0.get("start_reason", "")) == "Starting XI"
            rows.append(
                {
                    LINEUP_MATCH_ID: match_id,
                    LINEUP_TEAM: tname,
                    LINEUP_PLAYER_ID: int(pl.get("player_id", 0)),
                    LINEUP_PLAYER: str(pl.get("player_name", "")),
                    LINEUP_POSITION: pos_label,
                    LINEUP_STARTER: starter,
                    LINEUP_JERSEY: pl.get("jersey_number"),
                },
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(LINEUP_COLUMNS))
    return frame.reindex(columns=list(LINEUP_COLUMNS))


def _possession_shares(
    events: list[dict[str, Any]],
    home: str,
    away: str,
) -> tuple[float | None, float | None]:
    acc: dict[str, float] = defaultdict(float)
    for ev in events:
        t = ev.get("team") or {}
        name = t.get("name")
        if not name:
            continue
        acc[str(name)] += float(ev.get("duration") or 0.0)
    h, a = acc.get(home, 0.0), acc.get(away, 0.0)
    tot = h + a
    if tot <= 0:
        return None, None
    return 100.0 * h / tot, 100.0 * a / tot


def _count_progressive_passes(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        if (ev.get("type") or {}).get("name") != "Pass":
            continue
        loc = ev.get("location")
        p = ev.get("pass") or {}
        eloc = p.get("end_location")
        if not loc or not eloc or len(loc) < 2 or len(eloc) < 2:
            continue
        if float(eloc[0]) - float(loc[0]) < 10.0:
            continue
        team = (ev.get("team") or {}).get("name")
        if team:
            counts[str(team)] += 1
    return dict(counts)


def _count_counters(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ev in events:
        pat = ev.get("play_pattern") or {}
        if pat.get("name") != "From Counter":
            continue
        team = (ev.get("team") or {}).get("name")
        if team:
            counts[str(team)] += 1
    return dict(counts)


def _xg_by_team(events: list[dict[str, Any]]) -> dict[str, float]:
    xg: dict[str, float] = defaultdict(float)
    for ev in events:
        if (ev.get("type") or {}).get("name") != "Shot":
            continue
        shot = ev.get("shot") or {}
        if _is_penalty_shot(shot):
            continue
        team = (ev.get("team") or {}).get("name")
        if not team:
            continue
        xg[str(team)] += float(shot.get("statsbomb_xg") or 0.0)
    return dict(xg)


def events_to_team_tactical(
    events: list[dict[str, Any]],
    *,
    match_id: str,
    home_team: str,
    away_team: str,
) -> pd.DataFrame:
    """Aggregate simple per-team tactical proxies from events (Week 1 scope)."""
    pos_h, pos_a = _possession_shares(events, home_team, away_team)
    prog = _count_progressive_passes(events)
    ctr = _count_counters(events)
    xg_map = _xg_by_team(events)
    xg_h = xg_map.get(home_team, 0.0)
    xg_a = xg_map.get(away_team, 0.0)

    rows = [
        {
            TACT_TEAM: home_team,
            TACT_MATCH_ID: match_id,
            TACT_POSSESSION: pos_h,
            TACT_PPDA: None,
            TACT_PROG_PASSES: float(prog.get(home_team, 0)),
            TACT_COUNTER: float(ctr.get(home_team, 0)),
            TACT_PRESSING: None,
            TACT_TRANSITION: None,
            TACT_XG: xg_h,
            TACT_XGA: xg_a,
        },
        {
            TACT_TEAM: away_team,
            TACT_MATCH_ID: match_id,
            TACT_POSSESSION: pos_a,
            TACT_PPDA: None,
            TACT_PROG_PASSES: float(prog.get(away_team, 0)),
            TACT_COUNTER: float(ctr.get(away_team, 0)),
            TACT_PRESSING: None,
            TACT_TRANSITION: None,
            TACT_XG: xg_a,
            TACT_XGA: xg_h,
        },
    ]
    return pd.DataFrame(rows).reindex(columns=list(TEAM_TACTICAL_COLUMNS))


def iter_match_bundle(
    open_data_root: Path,
    competition_id: int,
    season_id: int,
    *,
    skip_missing_events: bool = True,
) -> Iterator[dict[str, pd.DataFrame | str]]:
    """Yield per-match dicts with tables for streaming ETL."""
    path = matches_path(open_data_root, competition_id, season_id)
    raw_matches = load_matches_file(path)
    for m in raw_matches:
        mid = int(m["match_id"])
        m_id_s = str(mid)
        comp = raw_match_competition(m)
        home = str((m.get("home_team") or {}).get("home_team_name", ""))
        away = str((m.get("away_team") or {}).get("away_team_name", ""))
        stage = str((m.get("competition_stage") or {}).get("name", ""))
        ev_path = events_path(open_data_root, mid)
        events = load_events_file(ev_path)
        if not events and not skip_missing_events:
            msg = f"missing events for match {mid}"
            raise FileNotFoundError(msg)
        shots, pens = events_to_shots_and_penalties(
            events,
            match_id=m_id_s,
            competition=comp,
            match_label=stage,
        )
        lineup_raw: list[dict[str, Any]] = []
        lp = lineups_path(open_data_root, mid)
        if lp.is_file():
            with lp.open(encoding="utf-8") as fh:
                lineup_raw = json.load(fh)
        lineups = lineups_to_dataframe(lineup_raw, m_id_s)
        tactical = events_to_team_tactical(
            events,
            match_id=m_id_s,
            home_team=home,
            away_team=away,
        )
        yield {
            "match_id": m_id_s,
            "shots": shots,
            "penalties": pens,
            "lineups": lineups,
            "team_tactical": tactical,
        }
