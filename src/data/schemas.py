"""Canonical column contracts shared across ingestion, features, and models.

Every loader (StatsBomb, soccerdata, FBref, etc.) should map into these
names so downstream modules can join and stack without ad-hoc renames.

Notes:
    - Dates are ISO strings or pandas datetime64; choose one per table in ETL.
    - ``neutral_ground`` is boolean once normalized.
    - Coordinates for shots follow the same pitch convention in all sources
      after normalization (document the convention in the ETL module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

SCHEMA_VERSION: Final[str] = "0.1.0"

# ---------------------------------------------------------------------------
# Match table
# ---------------------------------------------------------------------------

MATCH_ID: Final[str] = "match_id"
MATCH_DATE: Final[str] = "date"
MATCH_COMPETITION: Final[str] = "competition"
MATCH_HOME_TEAM: Final[str] = "home_team"
MATCH_AWAY_TEAM: Final[str] = "away_team"
MATCH_HOME_SCORE: Final[str] = "home_score"
MATCH_AWAY_SCORE: Final[str] = "away_score"
MATCH_VENUE: Final[str] = "venue"
MATCH_NEUTRAL: Final[str] = "neutral_ground"
MATCH_STAGE: Final[str] = "stage"
MATCH_ATTENDANCE: Final[str] = "attendance"
MATCH_WEATHER: Final[str] = "weather"

MATCH_COLUMNS: Final[tuple[str, ...]] = (
    MATCH_ID,
    MATCH_DATE,
    MATCH_COMPETITION,
    MATCH_HOME_TEAM,
    MATCH_AWAY_TEAM,
    MATCH_HOME_SCORE,
    MATCH_AWAY_SCORE,
    MATCH_VENUE,
    MATCH_NEUTRAL,
    MATCH_STAGE,
    MATCH_ATTENDANCE,
    MATCH_WEATHER,
)

# ---------------------------------------------------------------------------
# Team tactical features (one row per team per match)
# ---------------------------------------------------------------------------

TACT_TEAM: Final[str] = "team"
TACT_MATCH_ID: Final[str] = "match_id"
TACT_POSSESSION: Final[str] = "possession"
TACT_PPDA: Final[str] = "ppda"
TACT_PROG_PASSES: Final[str] = "progressive_passes"
TACT_COUNTER: Final[str] = "counter_attacks"
TACT_PRESSING: Final[str] = "pressing_intensity"
TACT_TRANSITION: Final[str] = "transition_speed"
TACT_XG: Final[str] = "xg"
TACT_XGA: Final[str] = "xga"

TEAM_TACTICAL_COLUMNS: Final[tuple[str, ...]] = (
    TACT_TEAM,
    TACT_MATCH_ID,
    TACT_POSSESSION,
    TACT_PPDA,
    TACT_PROG_PASSES,
    TACT_COUNTER,
    TACT_PRESSING,
    TACT_TRANSITION,
    TACT_XG,
    TACT_XGA,
)

# ---------------------------------------------------------------------------
# Shot table
# ---------------------------------------------------------------------------

SHOT_ID: Final[str] = "shot_id"
SHOT_PLAYER: Final[str] = "player"
SHOT_TEAM: Final[str] = "team"
SHOT_X: Final[str] = "x"
SHOT_Y: Final[str] = "y"
SHOT_DISTANCE: Final[str] = "distance"
SHOT_ANGLE: Final[str] = "angle"
SHOT_BODY_PART: Final[str] = "body_part"
SHOT_ASSIST_TYPE: Final[str] = "assist_type"
SHOT_UNDER_PRESSURE: Final[str] = "under_pressure"
SHOT_IS_GOAL: Final[str] = "is_goal"

SHOT_COLUMNS: Final[tuple[str, ...]] = (
    SHOT_ID,
    SHOT_PLAYER,
    SHOT_TEAM,
    SHOT_X,
    SHOT_Y,
    SHOT_DISTANCE,
    SHOT_ANGLE,
    SHOT_BODY_PART,
    SHOT_ASSIST_TYPE,
    SHOT_UNDER_PRESSURE,
    SHOT_IS_GOAL,
)

# ---------------------------------------------------------------------------
# Penalty table
# ---------------------------------------------------------------------------

PEN_ID: Final[str] = "penalty_id"
PEN_SHOOTER: Final[str] = "shooter"
PEN_KEEPER: Final[str] = "keeper"
PEN_FOOT: Final[str] = "footedness"
PEN_ZONE: Final[str] = "zone"
PEN_SCORED: Final[str] = "scored"
PEN_MATCH_PRESSURE: Final[str] = "match_pressure"
PEN_SCORE_STATE: Final[str] = "score_state"
PEN_MINUTE: Final[str] = "minute"
PEN_COMPETITION: Final[str] = "competition"

PENALTY_COLUMNS: Final[tuple[str, ...]] = (
    PEN_ID,
    PEN_SHOOTER,
    PEN_KEEPER,
    PEN_FOOT,
    PEN_ZONE,
    PEN_SCORED,
    PEN_MATCH_PRESSURE,
    PEN_SCORE_STATE,
    PEN_MINUTE,
    PEN_COMPETITION,
)


@dataclass(frozen=True)
class MatchRow:
    """Typed view of one normalized match record."""

    match_id: str
    date: str
    competition: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    venue: str
    neutral_ground: bool
    stage: str
    attendance: int | None
    weather: str | None


@dataclass(frozen=True)
class TeamTacticalRow:
    """Typed view of one team-match tactical feature row."""

    team: str
    match_id: str
    possession: float | None
    ppda: float | None
    progressive_passes: float | None
    counter_attacks: float | None
    pressing_intensity: float | None
    transition_speed: float | None
    xg: float | None
    xga: float | None


@dataclass(frozen=True)
class ShotRow:
    """Typed view of one normalized shot event."""

    shot_id: str
    player: str
    team: str
    x: float
    y: float
    distance: float
    angle: float
    body_part: str
    assist_type: str
    under_pressure: bool | None
    is_goal: bool


@dataclass(frozen=True)
class PenaltyRow:
    """Typed view of one normalized penalty event."""

    penalty_id: str
    shooter: str
    keeper: str
    footedness: str
    zone: str | None
    scored: bool
    match_pressure: str | None
    score_state: str | None
    minute: int | None
    competition: str
