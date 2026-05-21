"""Tests for openfootball and API-Football ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.api_football import fixtures_to_matches, statistics_to_tactical_rows
from data.merge_sources import dedupe_matches_by_fixture
from data.openfootball import parse_openfootball_text

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "openfootball" / "sample_euro.txt"


def test_parse_openfootball_sample() -> None:
    """Sample Euro file should yield three fixtures with scores and stages."""
    text = FIXTURE.read_text(encoding="utf-8")
    frame = parse_openfootball_text(text, competition="UEFA Euro 2024", default_year=2024)
    assert len(frame) == 3
    assert set(frame["home_team"]) >= {"Germany", "Spain"}
    final = frame[frame["stage"] == "Final"].iloc[0]
    assert final["home_score"] == 2
    assert final["away_score"] == 1
    assert str(frame["match_id"].iloc[0]).startswith("of_")


def test_dedupe_prefers_statsbomb() -> None:
    """Duplicate fixture keys should keep StatsBomb over openfootball."""
    sb = pd.DataFrame(
        [
            {
                "match_id": "7584",
                "date": "2024-06-14",
                "home_team": "Germany",
                "away_team": "Scotland",
                "home_score": 5,
                "away_score": 1,
                "competition": "UEFA Euro 2024",
                "venue": "",
                "neutral_ground": True,
                "stage": "Group A",
                "attendance": None,
                "weather": None,
                "data_source": "statsbomb",
            },
        ],
    )
    of = pd.DataFrame(
        [
            {
                "match_id": "of_dup",
                "date": "2024-06-14",
                "home_team": "Germany",
                "away_team": "Scotland",
                "home_score": 5,
                "away_score": 1,
                "competition": "UEFA Euro 2024",
                "venue": "Munich",
                "neutral_ground": True,
                "stage": "Group A",
                "attendance": None,
                "weather": None,
                "data_source": "openfootball",
            },
        ],
    )
    out = dedupe_matches_by_fixture(pd.concat([of, sb], ignore_index=True))
    assert len(out) == 1
    assert out.iloc[0]["data_source"] == "statsbomb"


def test_api_football_statistics_mapping() -> None:
    """Statistics payload should map possession and xG into tactical columns."""
    payload = {
        "response": [
            {
                "team": {"name": "Spain"},
                "statistics": [
                    {"type": "Ball Possession", "value": "55%"},
                    {"type": "expected_goals", "value": "1.8"},
                    {"type": "Total Shots", "value": 12},
                ],
            },
            {
                "team": {"name": "England"},
                "statistics": [
                    {"type": "Ball Possession", "value": "45%"},
                    {"type": "expected_goals", "value": "1.1"},
                    {"type": "Total Shots", "value": 8},
                ],
            },
        ],
    }
    tact = statistics_to_tactical_rows(
        999,
        payload,
        home_team="Spain",
        away_team="England",
    )
    assert len(tact) == 2
    spain = tact[tact["team"] == "Spain"].iloc[0]
    assert spain["possession"] == 55.0
    assert spain["xga"] == 1.1


def test_api_football_fixtures_mapping() -> None:
    """Fixture JSON should become canonical match rows."""
    payload = [
        {
            "fixture": {"id": 123, "date": "2022-11-20T16:00:00+00:00", "status": {"short": "FT"}},
            "league": {"name": "World Cup", "round": "Group A"},
            "teams": {"home": {"name": "Qatar"}, "away": {"name": "Ecuador"}},
            "goals": {"home": 0, "away": 2},
        },
    ]
    matches = fixtures_to_matches(payload, competition_name="FIFA World Cup 2022")
    assert len(matches) == 1
    assert matches.iloc[0]["match_id"] == "apif_123"
