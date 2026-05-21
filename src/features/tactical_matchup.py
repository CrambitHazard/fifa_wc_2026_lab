"""Style matchup and embedding features for match-outcome models."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from features.tactical_matrix import RATE_COLUMNS
from features.tactical_normalize import competition_priors

PROFILE_SUFFIX = "_norm"


def _profile_cols(normalized: pd.DataFrame) -> list[str]:
    return [f"{c}{PROFILE_SUFFIX}" for c in RATE_COLUMNS if f"{c}{PROFILE_SUFFIX}" in normalized.columns]


def build_pre_match_tactical_profiles(
    normalized: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    window: int = 5,
) -> pd.DataFrame:
    """Rolling mean of normalized tactics per team; impute cold starts from competition priors.

    Args:
        normalized: Context-normalized tactical matrix.
        matches: Match table with ``date`` and ``match_id``.
        window: Number of prior team-matches to average.

    Returns:
        One row per (match_id, team) with ``profile_*`` columns.
    """
    cols = _profile_cols(normalized)
    priors = competition_priors(normalized, matches)
    meta_cols = ["match_id"]
    if "date" not in normalized.columns:
        meta_cols.append("date")
    meta = matches[meta_cols].drop_duplicates()
    meta["match_id"] = meta["match_id"].astype(str)
    frame = normalized.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    if "date" not in frame.columns:
        frame = frame.merge(meta, on="match_id", how="left")
    frame = frame.merge(
        priors,
        on=["match_id", "team"],
        how="left",
    )
    frame["_d"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["team", "_d"]).reset_index(drop=True)

    history: dict[str, list[dict[str, float]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        team = str(row["team"])
        mid = str(row["match_id"])
        prof: dict[str, float] = {}
        seq = history[team][-window:]
        for col in cols:
            key = f"profile_{col}"
            if seq:
                prof[key] = float(sum(h.get(col, 0.0) for h in seq) / len(seq))
            else:
                prior_col = f"prior_{col}"
                prof[key] = float(row.get(prior_col, 0.0))

        rows.append({"match_id": mid, "team": team, **prof})

        snap = {col: float(row[col]) if pd.notna(row[col]) else 0.0 for col in cols}
        history[team].append(snap)

    return pd.DataFrame(rows)


def build_matchup_features(
    profiles: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Construct home-vs-away style interaction gaps per match."""
    prof = profiles.copy()
    prof["match_id"] = prof["match_id"].astype(str)
    prof = prof.set_index(["match_id", "team"])

    def _get(mid: str, team: str, key: str) -> float:
        try:
            return float(prof.loc[(mid, team), key])
        except KeyError:
            return 0.0

    rows: list[dict[str, Any]] = []
    for _, m in matches.iterrows():
        mid = str(m["match_id"])
        h = str(m["home_team"])
        a = str(m["away_team"])

        h_press = _get(mid, h, "profile_press_success_rate_norm")
        a_press = _get(mid, a, "profile_press_success_rate_norm")
        h_build = _get(mid, h, "profile_passes_per_sequence_norm")
        a_build = _get(mid, a, "profile_passes_per_sequence_norm")
        h_poss = _get(mid, h, "profile_possession_norm")
        a_poss = _get(mid, a, "profile_possession_norm")
        h_trans = _get(mid, h, "profile_transition_speed_norm")
        a_trans = _get(mid, a, "profile_transition_speed_norm")
        h_aerial = _get(mid, h, "profile_long_balls_norm") + _get(mid, h, "profile_crosses_norm")
        a_aerial = _get(mid, a, "profile_long_balls_norm") + _get(mid, a, "profile_crosses_norm")
        h_wing = _get(mid, h, "profile_crosses_norm")
        a_wing = _get(mid, a, "profile_crosses_norm")
        h_ppda = _get(mid, h, "profile_ppda_norm")
        a_ppda = _get(mid, a, "profile_ppda_norm")

        rows.append(
            {
                "match_id": mid,
                "press_resistance_gap": h_build - a_press,
                "transition_vulnerability_gap": a_trans - h_poss,
                "possession_asymmetry": h_poss - a_poss,
                "aerial_advantage": h_aerial - a_aerial,
                "wing_dependency_mismatch": h_wing - a_wing,
                "pressing_intensity_gap": h_ppda - a_ppda,
                "style_press_vs_build": (h_press - a_build) - (a_press - h_build),
            },
        )
    return pd.DataFrame(rows)


MATCHUP_COLUMNS: tuple[str, ...] = (
    "press_resistance_gap",
    "transition_vulnerability_gap",
    "possession_asymmetry",
    "aerial_advantage",
    "wing_dependency_mismatch",
    "pressing_intensity_gap",
    "style_press_vs_build",
)


def build_embedding_match_features(
    embeddings: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    profile_window: int = 5,
) -> pd.DataFrame:
    """Pre-match embedding summaries and cluster matchup signals per fixture.

    Uses rolling mean of prior PCA coordinates and cluster modes per team.

    Args:
        embeddings: Rows with ``match_id``, ``team``, ``pca_*``, ``cluster_kmeans``.
        matches: Match table with home/away teams.
        profile_window: Prior matches to average for embedding profiles.

    Returns:
        One row per match with embedding-based features for Model B.
    """
    emb = embeddings.copy()
    emb["match_id"] = emb["match_id"].astype(str)
    pca_cols = [c for c in emb.columns if c.startswith("pca_")]
    meta_cols = ["match_id", "date", "home_team", "away_team"]
    if "competition" in matches.columns:
        meta_cols.append("competition")
    meta = matches[meta_cols].drop_duplicates()
    meta["match_id"] = meta["match_id"].astype(str)
    frame = emb.merge(meta, on="match_id", how="left")
    frame["_d"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["team", "_d"]).reset_index(drop=True)

    comp_history: dict[str, list[pd.Series]] = defaultdict(list)
    history: dict[str, list[pd.Series]] = defaultdict(list)
    team_profiles: dict[tuple[str, str], dict[str, float]] = {}

    for _, row in frame.iterrows():
        team = str(row["team"])
        mid = str(row["match_id"])
        comp = str(row.get("competition", ""))
        seq = history[team][-profile_window:]
        comp_seq = comp_history[comp][-profile_window * 4 :]
        prof: dict[str, float] = {}
        if seq:
            for col in pca_cols:
                prof[f"emb_{col}"] = float(np.mean([s[col] for s in seq]))
            clusters = [int(s["cluster_kmeans"]) for s in seq if "cluster_kmeans" in s.index]
            prof["emb_cluster"] = float(np.mean(clusters)) if clusters else 0.0
        elif comp_seq:
            for col in pca_cols:
                prof[f"emb_{col}"] = float(np.mean([s[col] for s in comp_seq]))
            clusters = [
                int(s["cluster_kmeans"])
                for s in comp_seq
                if "cluster_kmeans" in s.index
            ]
            prof["emb_cluster"] = float(np.mean(clusters)) if clusters else 0.0
        else:
            for col in pca_cols:
                prof[f"emb_{col}"] = 0.0
            prof["emb_cluster"] = 0.0
        team_profiles[(mid, team)] = prof
        history[team].append(row)
        comp_history[comp].append(row)

    rows: list[dict[str, Any]] = []
    for _, m in matches.iterrows():
        mid = str(m["match_id"])
        h = str(m["home_team"])
        a = str(m["away_team"])
        hp = team_profiles.get((mid, h), {})
        ap = team_profiles.get((mid, a), {})
        row_out: dict[str, Any] = {"match_id": mid}
        for col in pca_cols:
            key = f"emb_{col}"
            row_out[f"home_{key}"] = hp.get(key, 0.0)
            row_out[f"away_{key}"] = ap.get(key, 0.0)
            row_out[f"gap_{key}"] = hp.get(key, 0.0) - ap.get(key, 0.0)
        row_out["home_emb_cluster"] = hp.get("emb_cluster", 0.0)
        row_out["away_emb_cluster"] = ap.get("emb_cluster", 0.0)
        row_out["cluster_mismatch"] = abs(
            hp.get("emb_cluster", 0.0) - ap.get("emb_cluster", 0.0),
        )
        if pca_cols:
            dist = 0.0
            for col in pca_cols:
                key = f"emb_{col}"
                dist += (hp.get(key, 0.0) - ap.get(key, 0.0)) ** 2
            row_out["embedding_style_distance"] = float(np.sqrt(dist))
        else:
            row_out["embedding_style_distance"] = 0.0
        rows.append(row_out)

    return pd.DataFrame(rows)


def embedding_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Numeric embedding feature columns present in ``frame``."""
    skip = {"match_id"}
    cols = [
        c
        for c in frame.columns
        if c not in skip and frame[c].dtype.kind in "iuf"
    ]
    return tuple(cols)
