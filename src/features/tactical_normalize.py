"""Context-adjust tactical rates (opponent strength, leak-free expanding norms)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.elo import attach_pre_match_elo
from features.tactical_matrix import RATE_COLUMNS


def _stage_weight(stage: str) -> float:
    """Rough importance weight for tournament stage (knockout > group)."""
    s = (stage or "").lower()
    if "final" in s:
        return 1.4
    if "semi" in s:
        return 1.3
    if "quarter" in s or "round of 16" in s or "round of 8" in s:
        return 1.2
    if "group" in s:
        return 1.0
    return 1.05


def attach_opponent_elo(
    tactical: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``opponent_elo`` using pre-match Elo from the same fixture."""
    with_elo = attach_pre_match_elo(matches)
    home_map = with_elo.set_index("match_id", drop=False)[
        ["match_id", "home_team", "elo_home_before", "elo_away_before"]
    ]
    rows: list[dict] = []
    for _, r in tactical.iterrows():
        mid = str(r["match_id"])
        team = str(r["team"])
        hm = home_map[home_map["match_id"].astype(str) == mid]
        if hm.empty:
            continue
        row = hm.iloc[0]
        if team == str(row["home_team"]):
            opp_elo = float(row["elo_away_before"])
        else:
            opp_elo = float(row["elo_home_before"])
        rows.append({**r.to_dict(), "opponent_elo": opp_elo})
    return pd.DataFrame(rows)


def _expanding_zscore(series: pd.Series) -> pd.Series:
    """Z-score using only prior rows in the same group (no future leakage)."""
    prior_mean = series.expanding(min_periods=3).mean().shift(1)
    prior_std = series.expanding(min_periods=3).std().shift(1).replace(0, np.nan)
    return ((series - prior_mean) / prior_std).fillna(0.0)


def normalize_tactical_matrix(
    tactical: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Opponent-adjust and leak-free expanding z-scores by competition × stage.

    Args:
        tactical: Raw matrix from :func:`features.tactical_matrix.build_tactical_matrix`.
        matches: Match metadata with ``competition``, ``stage``, teams, ``match_id``.

    Returns:
        Copy with ``*_raw``, ``*_opp_adj``, and ``*_norm`` columns for each rate.
    """
    meta = matches[
        ["match_id", "date", "competition", "stage", "home_team", "away_team"]
    ].drop_duplicates()
    meta["match_id"] = meta["match_id"].astype(str)
    frame = tactical.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    frame = frame.merge(meta, on="match_id", how="left")
    frame = attach_opponent_elo(frame, matches)
    frame["_d"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("_d").reset_index(drop=True)

    out = frame.copy()
    rate_cols = [c for c in RATE_COLUMNS if c in out.columns]

    for col in rate_cols:
        out[f"{col}_raw"] = out[col].astype(float)

    elo = out["opponent_elo"].fillna(1500.0)
    elo_z = (elo - 1500.0) / 200.0
    for col in rate_cols:
        base = out[f"{col}_raw"]
        expected_shift = elo_z * float(base.std(ddof=0)) * 0.15
        out[f"{col}_opp_adj"] = base + expected_shift

    for col in rate_cols:
        adj_col = f"{col}_opp_adj"
        out[f"{col}_norm"] = (
            out.groupby(["competition", "stage"], dropna=False)[adj_col]
            .transform(_expanding_zscore)
        )

    out["stage_weight"] = out["stage"].map(lambda s: _stage_weight(str(s)))
    return out


def competition_priors(
    normalized: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """Expanding competition-level means of norm features (for cold-start imputation).

    Args:
        normalized: Output of :func:`normalize_tactical_matrix`.
        matches: Match table with ``date`` and ``competition``.

    Returns:
        Same row index as ``normalized`` with ``prior_{col}`` columns.
    """
    norm_cols = [c for c in normalized.columns if c.endswith("_norm")]
    need = [c for c in ("date", "competition") if c not in normalized.columns]
    frame = normalized.copy()
    frame["match_id"] = frame["match_id"].astype(str)
    if need:
        meta = matches[["match_id", *need]].drop_duplicates()
        meta["match_id"] = meta["match_id"].astype(str)
        frame = frame.merge(meta, on="match_id", how="left")
    if "date" not in frame.columns:
        raise ValueError("normalized frame missing date (merge matches metadata)")
    if "competition" not in frame.columns:
        frame["competition"] = "unknown"
    frame["_d"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("_d").reset_index(drop=True)

    priors = frame[["match_id", "team"]].copy()
    for col in norm_cols:
        prior_name = f"prior_{col}"
        priors[prior_name] = (
            frame.groupby("competition", dropna=False)[col]
            .transform(lambda s: s.expanding(min_periods=1).mean().shift(1))
            .fillna(0.0)
        )
    return priors
