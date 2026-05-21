"""Model A vs Model B ablation: traditional vs tactical + embedding features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from embeddings.tactical import fit_tactical_embeddings
from features.elo import attach_pre_match_elo
from features.match_baseline import build_baseline_feature_table
from features.tactical_matchup import (
    MATCHUP_COLUMNS,
    build_embedding_match_features,
    build_matchup_features,
    build_pre_match_tactical_profiles,
    embedding_feature_columns,
)
from models.baseline_match import FEATURE_COLUMNS as TRADITIONAL_COLUMNS
from models.metrics import ClassificationReport, match_outcome_metrics, to_serializable_report

EVAL_MODE_CHOICES: tuple[str, ...] = (
    "chronological",
    "wc_holdout_2022",
    "wc_holdout_recent",
)

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[misc, assignment]

_OUTCOME_LABELS = [0, 1, 2]


def _align_proba(
    proba: np.ndarray,
    classes: np.ndarray,
    n_classes: int = 3,
) -> np.ndarray:
    """Map ``predict_proba`` output to fixed ``0..n_classes-1`` columns."""
    out = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(classes):
        idx = int(cls)
        if 0 <= idx < n_classes:
            out[:, idx] = proba[:, j]
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return out / row_sums


@dataclass(frozen=True)
class AblationResult:
    """Holdout metrics for traditional-only vs tactical-augmented models."""

    traditional_lr: ClassificationReport
    tactical_lr: ClassificationReport
    traditional_xgb: ClassificationReport | None
    tactical_xgb: ClassificationReport | None
    delta_log_loss_lr: float
    delta_log_loss_xgb: float | None
    n_valid: int
    n_train: int
    tactical_feature_columns: tuple[str, ...]
    tactical_lr_variant: str | None = None
    tactical_xgb_variant: str | None = None


@dataclass(frozen=True)
class WalkForwardFoldResult:
    """Single chronological fold metrics."""

    fold: int
    n_train: int
    n_valid: int
    delta_log_loss_lr: float
    delta_log_loss_xgb: float | None


@dataclass(frozen=True)
class WalkForwardAblationResult:
    """Aggregated walk-forward comparison."""

    folds: tuple[WalkForwardFoldResult, ...]
    mean_delta_log_loss_lr: float
    std_delta_log_loss_lr: float
    mean_delta_log_loss_xgb: float | None
    std_delta_log_loss_xgb: float | None


@dataclass(frozen=True)
class FeatureAblationRow:
    """One feature-set variant vs traditional baseline."""

    name: str
    n_features: int
    delta_log_loss_lr: float
    delta_log_loss_xgb: float | None
    tactical_log_loss_lr: float


def _fit_lr(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    cols: tuple[str, ...],
    random_state: int,
    *,
    C: float = 1.0,
) -> ClassificationReport:
    """Train scaled logistic regression and score on validation."""
    X_tr = train.loc[:, cols].astype(float).fillna(0.0).values
    X_va = valid.loc[:, cols].astype(float).fillna(0.0).values
    y_tr = train["y"].to_numpy(dtype=int)
    y_va = valid["y"].to_numpy(dtype=int)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
        C=C,
    )
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_va_s)
    proba = _align_proba(model.predict_proba(X_va_s), model.classes_)
    return match_outcome_metrics(y_va, pred, proba, labels=_OUTCOME_LABELS)


def _xgb_hyperparams(
    n_train: int,
    n_features: int,
    random_state: int,
) -> dict[str, Any]:
    """Scale regularization and capacity with training size and feature count."""
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": min(500, max(150, n_train * 2)),
        "max_depth": 3 if n_features > 15 else 4,
        "learning_rate": 0.05,
        "reg_alpha": 1.0,
        "reg_lambda": 2.0,
        "min_child_weight": max(1, n_train // 80),
        "subsample": 0.8,
        "colsample_bytree": 0.7 if n_features > 15 else 0.85,
        "random_state": random_state,
        "eval_metric": "mlogloss",
    }


def _fit_xgb(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    cols: tuple[str, ...],
    random_state: int,
) -> ClassificationReport | None:
    """Train XGBoost multiclass model when dependency is available."""
    if XGBClassifier is None:
        return None
    y_tr = train["y"].to_numpy(dtype=int)
    if len(np.unique(y_tr)) < 3:
        return None
    X_tr = train.loc[:, cols].astype(float).fillna(0.0).values
    X_va = valid.loc[:, cols].astype(float).fillna(0.0).values
    y_va = valid["y"].to_numpy(dtype=int)
    params = _xgb_hyperparams(len(train), len(cols), random_state)
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    proba = _align_proba(model.predict_proba(X_va), model.classes_)
    return match_outcome_metrics(y_va, pred, proba, labels=_OUTCOME_LABELS)


def _train_match_ids(
    matches: pd.DataFrame,
    test_fraction: float,
) -> tuple[set[str], set[str]]:
    """Return train and validation match_id sets (chronological split)."""
    ordered = matches.copy()
    ordered["_t"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values("_t").reset_index(drop=True)
    n = len(ordered)
    cut = max(1, int(np.floor(n * (1.0 - test_fraction))))
    train_ids = set(ordered.iloc[:cut]["match_id"].astype(str))
    valid_ids = set(ordered.iloc[cut:]["match_id"].astype(str))
    return train_ids, valid_ids


def _wc_competition_mask(matches: pd.DataFrame, *, year: str | None = None) -> pd.Series:
    """Boolean mask for men's World Cup fixtures, optionally filtered by year."""
    comp = matches["competition"].astype(str)
    wc = comp.str.contains("world cup", case=False, na=False)
    womens = comp.str.contains("women", case=False, na=False)
    youth = comp.str.contains(r"u\d{2}|youth", case=False, na=False)
    mask = wc & ~womens & ~youth
    if year is not None:
        mask = mask & comp.str.contains(str(year), na=False)
    return mask


def split_match_ids(
    matches: pd.DataFrame,
    *,
    eval_mode: str = "chronological",
    test_fraction: float = 0.25,
) -> tuple[set[str], set[str]]:
    """Return train/validation match ids for the requested evaluation protocol.

    Args:
        matches: Match table with ``match_id``, ``date``, ``competition``.
        eval_mode: ``chronological``, ``wc_holdout_2022``, or ``wc_holdout_recent``.
        test_fraction: Used only for ``chronological`` splits.

    Returns:
        Train and validation match id sets.

    Raises:
        ValueError: Unknown eval mode or empty split.
    """
    frame = matches.copy()
    frame["match_id"] = frame["match_id"].astype(str)

    if eval_mode == "chronological":
        return _train_match_ids(frame, test_fraction)

    if eval_mode == "wc_holdout_2022":
        valid = frame[_wc_competition_mask(frame, year="2022")]
        train = frame[~_wc_competition_mask(frame, year="2022")]
    elif eval_mode == "wc_holdout_recent":
        recent = _wc_competition_mask(frame) & (
            frame["competition"].astype(str).str.contains("2018|2022", na=False)
        )
        valid = frame[recent]
        train = frame[~recent]
    else:
        msg = f"unknown eval_mode: {eval_mode!r} (use {', '.join(EVAL_MODE_CHOICES)})"
        raise ValueError(msg)

    train_ids = set(train["match_id"].astype(str))
    valid_ids = set(valid["match_id"].astype(str))
    if len(train_ids) == 0 or len(valid_ids) == 0:
        msg = (
            f"empty train ({len(train_ids)}) or valid ({len(valid_ids)}) "
            f"for eval_mode={eval_mode!r}"
        )
        raise ValueError(msg)
    return train_ids, valid_ids


def build_tactical_model_frame(
    matches: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    profile_window: int = 5,
    n_clusters: int = 5,
    random_state: int = 42,
    train_match_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build matchup + embedding features merged to baseline labels.

    Args:
        matches: Match table.
        normalized: Leak-free normalized tactical matrix.
        profile_window: Rolling window for profiles.
        n_clusters: KMeans clusters.
        random_state: RNG seed.
        train_match_ids: If set, fit PCA/KMeans only on these matches.

    Returns:
        Tuple of (modeling frame with ``y``, cluster validation dict).
    """
    normalized = normalized.copy()
    normalized["match_id"] = normalized["match_id"].astype(str)

    if train_match_ids:
        fit_rows = normalized[normalized["match_id"].isin(train_match_ids)]
    else:
        fit_rows = normalized

    emb = fit_tactical_embeddings(
        normalized,
        n_clusters=n_clusters,
        random_state=random_state,
        fit_frame=fit_rows,
    )
    profiles = build_pre_match_tactical_profiles(
        normalized,
        matches,
        window=profile_window,
    )
    matchup = build_matchup_features(profiles, matches)
    emb_match = build_embedding_match_features(
        emb.frame,
        matches,
        profile_window=profile_window,
    )
    emb_cols = embedding_feature_columns(emb_match)

    with_elo = attach_pre_match_elo(matches)
    base = build_baseline_feature_table(with_elo)
    frame = base.merge(matchup, on="match_id", how="inner")
    frame = frame.merge(emb_match, on="match_id", how="inner")
    tact_cols = tuple(list(MATCHUP_COLUMNS) + list(emb_cols))
    return frame, emb.validation


def _tactical_column_variants(
    frame: pd.DataFrame,
    emb_cols: tuple[str, ...],
) -> list[tuple[str, tuple[str, ...]]]:
    """Candidate Model B feature sets ordered from smallest to largest."""
    return [
        ("matchup_only", (*TRADITIONAL_COLUMNS, *MATCHUP_COLUMNS)),
        ("embeddings_only", (*TRADITIONAL_COLUMNS, *emb_cols)),
        ("full_tactical", (*TRADITIONAL_COLUMNS, *MATCHUP_COLUMNS, *emb_cols)),
    ]


def _chronological_inner_split(
    train: pd.DataFrame,
    inner_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest fraction of train rows for inner model selection."""
    ordered = train.copy()
    ordered["_t"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values("_t").reset_index(drop=True)
    cut = max(8, int(np.floor(len(ordered) * (1.0 - inner_fraction))))
    if cut >= len(ordered):
        cut = max(1, len(ordered) - 8)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _select_tactical_columns(
    train: pd.DataFrame,
    emb_cols: tuple[str, ...],
    *,
    random_state: int,
    model: str,
) -> tuple[str, tuple[str, ...]]:
    """Pick the tactical variant with lowest inner-validation log loss."""
    variants = _tactical_column_variants(train, emb_cols)
    if len(train) < 32 or "date" not in train.columns:
        use = tuple(c for c in variants[-1][1] if c in train.columns)
        return variants[-1][0], use

    inner_tr, inner_va = _chronological_inner_split(train)
    if len(inner_va) < 4:
        use = tuple(c for c in variants[0][1] if c in train.columns)
        return variants[0][0], use

    best_name = variants[0][0]
    best_cols = tuple(c for c in variants[0][1] if c in train.columns)
    best_ll = float("inf")
    tact_c = 0.1 if len(train) >= 120 else 0.05

    for name, cols in variants:
        use_cols = tuple(c for c in cols if c in train.columns)
        if len(use_cols) <= len(TRADITIONAL_COLUMNS):
            continue
        if model == "lr":
            report = _fit_lr(inner_tr, inner_va, use_cols, random_state, C=tact_c)
        else:
            report = _fit_xgb(inner_tr, inner_va, use_cols, random_state)
            if report is None:
                continue
        if report.log_loss < best_ll:
            best_ll = report.log_loss
            best_name = name
            best_cols = use_cols

    return best_name, best_cols


def _embedding_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        c
        for c in frame.columns
        if c.startswith(
            ("gap_emb_", "embedding_", "cluster_", "home_emb_", "away_emb_"),
        )
    )


def _run_ablation_on_frame(
    frame: pd.DataFrame,
    train_ids: set[str],
    valid_ids: set[str],
    *,
    random_state: int,
) -> AblationResult:
    """Fit models on ``train_ids`` and score ``valid_ids`` from a prebuilt frame."""
    train = frame[frame["match_id"].astype(str).isin(train_ids)].copy()
    valid = frame[frame["match_id"].astype(str).isin(valid_ids)].copy()
    if len(train) == 0 or len(valid) == 0:
        msg = f"empty train ({len(train)}) or valid ({len(valid)}) split"
        raise ValueError(msg)

    emb_cols = _embedding_columns(frame)
    trad_cols = TRADITIONAL_COLUMNS

    lr_variant, lr_cols = _select_tactical_columns(
        train,
        emb_cols,
        random_state=random_state,
        model="lr",
    )
    xgb_variant, xgb_cols = _select_tactical_columns(
        train,
        emb_cols,
        random_state=random_state,
        model="xgb",
    )

    trad_lr = _fit_lr(train, valid, trad_cols, random_state, C=1.0)
    tact_c = 0.1 if len(train) >= 120 else 0.05
    tact_lr = _fit_lr(train, valid, lr_cols, random_state, C=tact_c)
    trad_xgb = _fit_xgb(train, valid, trad_cols, random_state)
    tact_xgb = _fit_xgb(train, valid, xgb_cols, random_state)

    delta_lr = trad_lr.log_loss - tact_lr.log_loss
    delta_xgb: float | None = None
    if trad_xgb is not None and tact_xgb is not None:
        delta_xgb = trad_xgb.log_loss - tact_xgb.log_loss

    return AblationResult(
        traditional_lr=trad_lr,
        tactical_lr=tact_lr,
        traditional_xgb=trad_xgb,
        tactical_xgb=tact_xgb,
        delta_log_loss_lr=delta_lr,
        delta_log_loss_xgb=delta_xgb,
        n_valid=len(valid),
        n_train=len(train),
        tactical_feature_columns=lr_cols,
        tactical_lr_variant=lr_variant,
        tactical_xgb_variant=xgb_variant,
    )


def run_tactical_ablation(
    matches: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    profile_window: int = 5,
    n_clusters: int = 5,
    test_fraction: float = 0.25,
    eval_mode: str = "chronological",
    random_state: int = 42,
) -> AblationResult:
    """Compare traditional baseline vs tactical + embedding features (train-only cluster fit)."""
    train_ids, valid_ids = split_match_ids(
        matches,
        eval_mode=eval_mode,
        test_fraction=test_fraction,
    )
    frame, _ = build_tactical_model_frame(
        matches,
        normalized,
        profile_window=profile_window,
        n_clusters=n_clusters,
        random_state=random_state,
        train_match_ids=train_ids,
    )
    return _run_ablation_on_frame(
        frame,
        train_ids,
        valid_ids,
        random_state=random_state,
    )


def run_walk_forward_ablation(
    matches: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    n_folds: int = 5,
    min_train_matches: int = 40,
    profile_window: int = 5,
    n_clusters: int = 5,
    random_state: int = 42,
) -> WalkForwardAblationResult | None:
    """Chronological walk-forward folds; each fold fits on all prior matches.

    Args:
        matches: Match table sorted internally by date.
        normalized: Leak-free normalized tactical matrix.
        n_folds: Number of validation windows.
        min_train_matches: Skip folds with fewer prior matches for training.
        profile_window: Rolling profile window.
        n_clusters: KMeans clusters.
        random_state: RNG seed.

    Returns:
        Aggregated fold metrics, or ``None`` if not enough data for folds.
    """
    ordered = matches.copy()
    ordered["_t"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values("_t").reset_index(drop=True)
    n = len(ordered)
    if n < min_train_matches + n_folds:
        return None

    fold_size = max(1, (n - min_train_matches) // n_folds)
    fold_results: list[WalkForwardFoldResult] = []

    for fold_idx in range(n_folds):
        valid_start = min_train_matches + fold_idx * fold_size
        valid_end = min(n, valid_start + fold_size)
        if valid_start >= n or valid_start >= valid_end:
            break
        train_ids = set(ordered.iloc[:valid_start]["match_id"].astype(str))
        valid_ids = set(ordered.iloc[valid_start:valid_end]["match_id"].astype(str))
        frame, _ = build_tactical_model_frame(
            matches,
            normalized,
            profile_window=profile_window,
            n_clusters=n_clusters,
            random_state=random_state,
            train_match_ids=train_ids,
        )
        try:
            result = _run_ablation_on_frame(
                frame,
                train_ids,
                valid_ids,
                random_state=random_state,
            )
        except ValueError:
            continue
        fold_results.append(
            WalkForwardFoldResult(
                fold=fold_idx + 1,
                n_train=result.n_train,
                n_valid=result.n_valid,
                delta_log_loss_lr=result.delta_log_loss_lr,
                delta_log_loss_xgb=result.delta_log_loss_xgb,
            ),
        )

    if not fold_results:
        return None

    lr_deltas = [f.delta_log_loss_lr for f in fold_results]
    xgb_deltas = [f.delta_log_loss_xgb for f in fold_results if f.delta_log_loss_xgb is not None]
    mean_xgb = float(np.mean(xgb_deltas)) if xgb_deltas else None
    std_xgb = float(np.std(xgb_deltas)) if len(xgb_deltas) > 1 else None

    return WalkForwardAblationResult(
        folds=tuple(fold_results),
        mean_delta_log_loss_lr=float(np.mean(lr_deltas)),
        std_delta_log_loss_lr=float(np.std(lr_deltas)) if len(lr_deltas) > 1 else 0.0,
        mean_delta_log_loss_xgb=mean_xgb,
        std_delta_log_loss_xgb=std_xgb,
    )


def run_feature_ablation(
    matches: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    profile_window: int = 5,
    n_clusters: int = 5,
    test_fraction: float = 0.25,
    eval_mode: str = "chronological",
    random_state: int = 42,
) -> list[FeatureAblationRow]:
    """Compare traditional vs matchup-only vs embedding-only vs full Model B."""
    train_ids, valid_ids = split_match_ids(
        matches,
        eval_mode=eval_mode,
        test_fraction=test_fraction,
    )
    frame, _ = build_tactical_model_frame(
        matches,
        normalized,
        profile_window=profile_window,
        n_clusters=n_clusters,
        random_state=random_state,
        train_match_ids=train_ids,
    )
    train = frame[frame["match_id"].astype(str).isin(train_ids)]
    valid = frame[frame["match_id"].astype(str).isin(valid_ids)]
    emb_cols = _embedding_columns(frame)

    variants: list[tuple[str, tuple[str, ...], float]] = [
        ("matchup_only", (*TRADITIONAL_COLUMNS, *MATCHUP_COLUMNS), 0.05),
        ("embeddings_only", (*TRADITIONAL_COLUMNS, *emb_cols), 0.05),
        ("full_tactical", (*TRADITIONAL_COLUMNS, *MATCHUP_COLUMNS, *emb_cols), 0.05),
    ]

    trad_lr = _fit_lr(train, valid, TRADITIONAL_COLUMNS, random_state, C=1.0)
    rows: list[FeatureAblationRow] = []

    for name, cols, c_reg in variants:
        use_cols = tuple(c for c in cols if c in frame.columns)
        tact_lr = _fit_lr(train, valid, use_cols, random_state, C=c_reg)
        tact_xgb = _fit_xgb(train, valid, use_cols, random_state)
        delta_xgb: float | None = None
        if tact_xgb is not None:
            trad_xgb = _fit_xgb(train, valid, TRADITIONAL_COLUMNS, random_state)
            if trad_xgb is not None:
                delta_xgb = trad_xgb.log_loss - tact_xgb.log_loss
        rows.append(
            FeatureAblationRow(
                name=name,
                n_features=len(use_cols) - len(TRADITIONAL_COLUMNS),
                delta_log_loss_lr=trad_lr.log_loss - tact_lr.log_loss,
                delta_log_loss_xgb=delta_xgb,
                tactical_log_loss_lr=tact_lr.log_loss,
            ),
        )
    return rows


def save_ablation_report(
    result: AblationResult,
    path: Path,
    *,
    eval_mode: str | None = None,
) -> dict[str, Any]:
    """Write JSON ablation summary for reports."""
    payload: dict[str, Any] = {
        "n_train": result.n_train,
        "n_valid": result.n_valid,
        "delta_log_loss_lr": result.delta_log_loss_lr,
        "delta_log_loss_xgb": result.delta_log_loss_xgb,
        "n_tactical_features": len(result.tactical_feature_columns),
        "n_embedding_features": sum(
            1
            for c in result.tactical_feature_columns
            if c.startswith(
                ("gap_emb_", "embedding_", "cluster_", "home_emb_", "away_emb_"),
            )
        ),
        "interpretation": (
            "Positive delta_log_loss means tactical model has LOWER log loss (better)."
        ),
        "traditional_lr": to_serializable_report(result.traditional_lr),
        "tactical_lr": to_serializable_report(result.tactical_lr),
    }
    if eval_mode is not None:
        payload["eval_mode"] = eval_mode
    if result.tactical_lr_variant is not None:
        payload["tactical_lr_variant"] = result.tactical_lr_variant
    if result.tactical_xgb_variant is not None:
        payload["tactical_xgb_variant"] = result.tactical_xgb_variant
    if result.traditional_xgb is not None:
        payload["traditional_xgb"] = to_serializable_report(result.traditional_xgb)
    if result.tactical_xgb is not None:
        payload["tactical_xgb"] = to_serializable_report(result.tactical_xgb)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def save_walk_forward_report(
    result: WalkForwardAblationResult,
    path: Path,
) -> dict[str, Any]:
    """Write walk-forward ablation JSON."""
    payload: dict[str, Any] = {
        "n_folds": len(result.folds),
        "mean_delta_log_loss_lr": result.mean_delta_log_loss_lr,
        "std_delta_log_loss_lr": result.std_delta_log_loss_lr,
        "mean_delta_log_loss_xgb": result.mean_delta_log_loss_xgb,
        "std_delta_log_loss_xgb": result.std_delta_log_loss_xgb,
        "interpretation": (
            "Positive mean_delta_log_loss_lr means tactical model wins on average across folds."
        ),
        "folds": [
            {
                "fold": f.fold,
                "n_train": f.n_train,
                "n_valid": f.n_valid,
                "delta_log_loss_lr": f.delta_log_loss_lr,
                "delta_log_loss_xgb": f.delta_log_loss_xgb,
            }
            for f in result.folds
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def save_feature_ablation_report(
    rows: list[FeatureAblationRow],
    path: Path,
) -> dict[str, Any]:
    """Write feature-decomposition ablation JSON."""
    payload = {
        "interpretation": (
            "Positive delta_log_loss means variant beats traditional-only on holdout."
        ),
        "variants": [
            {
                "name": r.name,
                "n_extra_features": r.n_features,
                "delta_log_loss_lr": r.delta_log_loss_lr,
                "delta_log_loss_xgb": r.delta_log_loss_xgb,
                "log_loss_lr": r.tactical_log_loss_lr,
            }
            for r in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload
