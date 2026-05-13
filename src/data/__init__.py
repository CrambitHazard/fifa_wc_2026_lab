"""Data ingestion, normalization, and schema contracts."""

from .schemas import (
    PENALTY_COLUMNS,
    SHOT_COLUMNS,
    MATCH_COLUMNS,
    TEAM_TACTICAL_COLUMNS,
)

__all__ = [
    "MATCH_COLUMNS",
    "PENALTY_COLUMNS",
    "SHOT_COLUMNS",
    "TEAM_TACTICAL_COLUMNS",
]
