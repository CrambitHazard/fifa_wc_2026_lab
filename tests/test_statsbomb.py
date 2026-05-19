"""StatsBomb loader and ETL tests (use local fixtures, no download)."""

from __future__ import annotations

from pathlib import Path

from data.pipeline import run_statsbomb_etl
from data.statsbomb import (
    events_to_shots_and_penalties,
    load_events_file,
    raw_match_competition,
    resolve_open_data_data_dir,
    sb_distance_angle,
)

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "statsbomb"


def test_sb_distance_angle_basic() -> None:
    """Opening angle in front of goal should be positive."""
    dist, ang = sb_distance_angle(100.0, 40.0)
    assert dist > 0.4
    assert 0 < ang < 3.15


def test_open_data_fixture_penalty_split() -> None:
    """Penalty shots belong only in the penalty frame."""
    ev_path = FIXTURE_ROOT / "data" / "events" / "90001.json"
    ev = load_events_file(ev_path)
    shots, pens = events_to_shots_and_penalties(
        ev,
        match_id="90001",
        competition="Test Cup 2018",
        match_label="Group Stage",
    )
    assert len(shots) == 1
    assert len(pens) == 1
    assert bool(shots.iloc[0]["is_goal"])
    assert not bool(pens.iloc[0]["scored"])


def test_run_etl_on_fixtures(tmp_path: Path) -> None:
    """End-to-end parquet write from minimal Open Data layout."""
    out = tmp_path / "processed"
    meta = run_statsbomb_etl(
        open_data_root=FIXTURE_ROOT,
        competition_id=99,
        season_id=1,
        processed_dir=out,
        skip_missing_events=False,
    )
    assert meta["rows"]["matches"] == 2
    assert meta["rows"]["shots_open_play"] >= 1
    assert meta["rows"]["penalties"] >= 1


def test_resolve_flat_open_data_layout() -> None:
    """Fixture tree uses matches/ at root (no nested data/ folder)."""
    data_dir = resolve_open_data_data_dir(FIXTURE_ROOT)
    assert (data_dir / "matches" / "99" / "1.json").is_file()


def test_raw_match_competition_string() -> None:
    """Smoke test helper used by the pipeline."""
    import json

    p = FIXTURE_ROOT / "data" / "matches" / "99" / "1.json"
    m0 = json.loads(p.read_text(encoding="utf-8"))[0]
    assert "Test Cup" in raw_match_competition(m0)
