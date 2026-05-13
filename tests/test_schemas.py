"""Guardrails for canonical schema contracts."""

from __future__ import annotations

from data.schemas import (
    MATCH_COLUMNS,
    PENALTY_COLUMNS,
    SHOT_COLUMNS,
    TEAM_TACTICAL_COLUMNS,
)


def _assert_unique(columns: tuple[str, ...]) -> None:
    assert len(columns) == len(set(columns)), f"duplicate columns in {columns}"


def test_match_columns_unique_and_stable_order() -> None:
    """Match table column names must stay stable for parquet/DB exports."""
    _assert_unique(MATCH_COLUMNS)
    assert MATCH_COLUMNS[0] == "match_id"


def test_team_tactical_columns_unique() -> None:
    """Team tactical feature names must align with modeling joins."""
    _assert_unique(TEAM_TACTICAL_COLUMNS)
    assert "match_id" in TEAM_TACTICAL_COLUMNS
    assert "team" in TEAM_TACTICAL_COLUMNS


def test_shot_columns_unique() -> None:
    """Shot event contract supports xG and audit workflows."""
    _assert_unique(SHOT_COLUMNS)


def test_penalty_columns_unique() -> None:
    """Penalty contract supports specialist models later."""
    _assert_unique(PENALTY_COLUMNS)
