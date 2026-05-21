"""Tests for national / World Cup match scope filters."""

from __future__ import annotations

import pandas as pd

from data.scope_filters import (
    apply_match_scope,
    is_statsbomb_event_match_id,
)
from features.tactical_matrix import build_tactical_matrix


def test_is_statsbomb_event_match_id() -> None:
    assert is_statsbomb_event_match_id("7584")
    assert not is_statsbomb_event_match_id("fbref_2022_08_05_A_B")


def test_world_cup_scope() -> None:
    matches = pd.DataFrame(
        {
            "match_id": ["1", "2"],
            "competition": ["FIFA World Cup 2018", "ENG Premier League 23/24"],
        },
    )
    out = apply_match_scope(matches, "world_cup")
    assert len(out) == 1
    assert "World Cup" in out.iloc[0]["competition"]


def test_mens_senior_scope_excludes_womens() -> None:
    matches = pd.DataFrame(
        {
            "match_id": ["1", "2", "3"],
            "competition": [
                "FIFA World Cup 2022",
                "Women's World Cup 2023",
                "FIFA U20 World Cup 1979",
            ],
        },
    )
    out = apply_match_scope(matches, "mens_senior_national")
    assert len(out) == 1
    assert "2022" in out.iloc[0]["competition"]


def test_wc_scope_excludes_womens() -> None:
    matches = pd.DataFrame(
        {
            "match_id": ["1", "2"],
            "competition": ["FIFA World Cup 2022", "Women's World Cup 2023"],
        },
    )
    out = apply_match_scope(matches, "world_cup")
    assert len(out) == 1


def test_build_matrix_skips_fbref_ids() -> None:
    """Non-numeric match_id rows must not crash event extraction."""
    matches = pd.DataFrame(
        {
            "match_id": ["fbref_x", "99999"],
            "home_team": ["A", "H"],
            "away_team": ["B", "A"],
        },
    )
    mat = build_tactical_matrix(matches, open_data_root=__import__("pathlib").Path("missing"))
    assert len(mat) == 0
