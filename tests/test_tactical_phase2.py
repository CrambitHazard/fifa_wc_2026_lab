"""Phase 2 tactical extraction and ablation smoke tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.statsbomb import load_events_file, load_matches_file, matches_to_dataframe
from features.tactical_events import extract_match_tactical_counts
from features.tactical_matrix import build_tactical_matrix
from features.tactical_matchup import MATCHUP_COLUMNS, build_matchup_features, build_pre_match_tactical_profiles
from features.tactical_matrix import RATE_COLUMNS
from features.tactical_normalize import normalize_tactical_matrix

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "statsbomb"


def test_extract_counts_from_fixture() -> None:
    """Fixture match should yield progressive passes and possession by possession_team."""
    ev = load_events_file(FIXTURE_ROOT / "data" / "events" / "90001.json")
    counts = extract_match_tactical_counts(ev, home_team="Alpha FC", away_team="Beta FC")
    assert counts["Alpha FC"].progressive_passes >= 1
    assert counts["Alpha FC"].possession_seconds > 0
    assert counts["Beta FC"].possession_seconds > 0
    assert counts["Alpha FC"].pressures >= 1
    assert counts["Alpha FC"].pressure_wins >= 1


def test_ablation_model_b_includes_embeddings() -> None:
    """Model B must include PCA/cluster embedding columns, not only matchup gaps."""
    from models.match_ablation import build_tactical_model_frame

    matches = pd.DataFrame(
        [
            {
                "match_id": str(i),
                "date": f"2020-06-{i:02d}",
                "home_team": "A",
                "away_team": "B",
                "home_score": 1,
                "away_score": 0,
                "competition": "T",
                "stage": "Group Stage",
            }
            for i in range(1, 13)
        ],
    )
    rows = []
    for i in range(1, 13):
        for team in ("A", "B"):
            row = {
                "match_id": str(i),
                "team": team,
                "date": f"2020-06-{min(i, 28):02d}",
            }
            for col in RATE_COLUMNS:
                row[f"{col}_norm"] = float(i % 3) * 0.1
                row[f"{col}_raw"] = 1.0
                row[f"{col}_opp_adj"] = 1.0
            rows.append(row)
    normalized = pd.DataFrame(rows)
    frame, _ = build_tactical_model_frame(matches, normalized, profile_window=2)
    emb_cols = [
        c
        for c in frame.columns
        if c.startswith(
            ("gap_emb_", "embedding_", "cluster_", "home_emb_", "away_emb_"),
        )
    ]
    assert len(emb_cols) >= 3


def test_wc_holdout_split() -> None:
    """Train on intl friendlies/tournaments; validate on WC 2022 only."""
    from models.match_ablation import split_match_ids

    matches = pd.DataFrame(
        {
            "match_id": ["1", "2", "3"],
            "date": ["2020-06-01", "2022-11-01", "2024-06-01"],
            "competition": [
                "UEFA Euro 2020",
                "FIFA World Cup 2022",
                "UEFA Euro 2024",
            ],
        },
    )
    train_ids, valid_ids = split_match_ids(matches, eval_mode="wc_holdout_2022")
    assert valid_ids == {"2"}
    assert train_ids == {"1", "3"}


def test_walk_forward_ablation_smoke() -> None:
    """Walk-forward should return folds on synthetic data."""
    from models.match_ablation import run_walk_forward_ablation

    matches = pd.DataFrame(
        [
            {
                "match_id": str(i),
                "date": f"2020-01-{min(i, 28):02d}",
                "home_team": "A",
                "away_team": "B",
                "home_score": i % 3,
                "away_score": (i + 1) % 3,
                "competition": "T",
                "stage": "Group Stage",
            }
            for i in range(1, 80)
        ],
    )
    rows = []
    for i in range(1, 80):
        for team in ("A", "B"):
            row = {"match_id": str(i), "team": team, "date": f"2020-01-{min(i, 28):02d}"}
            for col in RATE_COLUMNS:
                row[f"{col}_norm"] = float(i % 5) * 0.05
                row[f"{col}_raw"] = 1.0
                row[f"{col}_opp_adj"] = 1.0
            rows.append(row)
    normalized = pd.DataFrame(rows)
    wf = run_walk_forward_ablation(
        matches,
        normalized,
        n_folds=3,
        min_train_matches=20,
        profile_window=2,
        n_clusters=3,
    )
    assert wf is not None
    assert len(wf.folds) >= 1


def test_build_matrix_on_fixture_matches() -> None:
    """Matrix builder should read events for synthetic competition."""
    raw = load_matches_file(FIXTURE_ROOT / "data" / "matches" / "99" / "1.json")
    matches = matches_to_dataframe(raw)
    mat = build_tactical_matrix(matches, open_data_root=FIXTURE_ROOT)
    assert len(mat) >= 2
    assert "ppda" in mat.columns or mat["progressive_passes"].notna().any()


def test_normalize_accepts_rate_tuple_columns() -> None:
    """Pandas must not treat RATE_COLUMNS tuple as a single column key."""
    raw = load_matches_file(FIXTURE_ROOT / "data" / "matches" / "99" / "1.json")
    matches = matches_to_dataframe(raw)
    mat = build_tactical_matrix(matches, open_data_root=FIXTURE_ROOT)
    norm = normalize_tactical_matrix(mat, matches)
    assert "possession_norm" in norm.columns


def test_matchup_columns_present() -> None:
    """Matchup builder returns all interaction columns."""
    matches = pd.DataFrame(
        [
            {
                "match_id": "1",
                "date": "2020-01-01",
                "home_team": "A",
                "away_team": "B",
                "competition": "T",
                "stage": "G",
            },
            {
                "match_id": "2",
                "date": "2020-01-02",
                "home_team": "B",
                "away_team": "A",
                "competition": "T",
                "stage": "G",
            },
        ],
    )
    norm = pd.DataFrame(
        {
            "team": ["A", "B", "A", "B"],
            "match_id": ["1", "1", "2", "2"],
            "possession_norm": [0.1, -0.1, 0.2, -0.2],
            "ppda_norm": [0.0, 0.0, 0.0, 0.0],
            "high_turnovers_norm": [0.0, 0.0, 0.0, 0.0],
            "progressive_passes_norm": [0.0, 0.0, 0.0, 0.0],
            "progressive_carries_norm": [0.0, 0.0, 0.0, 0.0],
            "long_balls_norm": [0.0, 0.0, 0.0, 0.0],
            "crosses_norm": [0.0, 0.0, 0.0, 0.0],
            "counter_attacks_norm": [0.0, 0.0, 0.0, 0.0],
            "transition_speed_norm": [0.0, 0.0, 0.0, 0.0],
            "final_third_entries_norm": [0.0, 0.0, 0.0, 0.0],
            "defensive_line_height_norm": [0.0, 0.0, 0.0, 0.0],
            "passes_per_sequence_norm": [0.0, 0.0, 0.0, 0.0],
            "xg_norm": [0.0, 0.0, 0.0, 0.0],
            "xga_norm": [0.0, 0.0, 0.0, 0.0],
            "shot_distance_norm": [0.0, 0.0, 0.0, 0.0],
            "press_success_rate_norm": [0.0, 0.0, 0.0, 0.0],
            "pressing_intensity_norm": [0.0, 0.0, 0.0, 0.0],
        },
    )
    for col in [
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
    ]:
        if f"{col}_raw" not in norm.columns:
            norm[f"{col}_raw"] = 0.0
        if f"{col}_opp_adj" not in norm.columns:
            norm[f"{col}_opp_adj"] = norm[f"{col}_norm"]
    profiles = build_pre_match_tactical_profiles(norm, matches, window=1)
    matchup = build_matchup_features(profiles, matches)
    for c in MATCHUP_COLUMNS:
        assert c in matchup.columns
