"""Week-1 baseline match outcome models (multinomial logistic + XGBoost)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.metrics import ClassificationReport, match_outcome_metrics

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None  # type: ignore[misc, assignment]


FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_home_before",
    "elo_away_before",
    "elo_diff",
    "form_points_home",
    "form_points_away",
    "avg_gf_home",
    "avg_ga_home",
    "avg_gf_away",
    "avg_ga_away",
    "neutral_ground",
)


@dataclass(frozen=True)
class BaselineSuite:
    """Trained baselines plus holdout metrics."""

    logistic: Any
    log_reg_scaler: StandardScaler
    xgboost: Any | None
    report_logistic: ClassificationReport
    report_xgb: ClassificationReport | None
    feature_columns: tuple[str, ...]


def time_ordered_split(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    test_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut data chronologically (no shuffling).

    Args:
        frame: Modeling table including ``date_col``.
        date_col: Column used for ordering.
        test_fraction: Fraction of newest matches assigned to validation.

    Returns:
        ``(train, valid)`` partitions.

    """
    ordered = frame.copy()
    ordered["_t"] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values("_t").reset_index(drop=True)
    n = len(ordered)
    cut = max(1, int(np.floor(n * (1.0 - test_fraction))))
    train, valid = ordered.iloc[:cut], ordered.iloc[cut:]
    return train.drop(columns=["_t"]), valid.drop(columns=["_t"])


def train_baselines(
    feature_table: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    date_col: str = "date",
    random_state: int = 42,
) -> BaselineSuite:
    """Fit logistic regression and (if available) XGBoost multiclass models.

    Args:
        feature_table: Output of :func:`features.match_baseline.build_baseline_feature_table`.
        test_fraction: Chronological validation fraction.
        date_col: Date column name on ``feature_table``.
        random_state: Seed for tree model.

    Returns:
        BaselineSuite with sklearn / xgb estimators and metric reports.

    Raises:
        ValueError: If training split is too small or features are missing.
    """
    missing = [c for c in FEATURE_COLUMNS if c not in feature_table.columns]
    if missing:
        msg = f"feature_table missing columns: {missing}"
        raise ValueError(msg)

    train, valid = time_ordered_split(
        feature_table,
        date_col=date_col,
        test_fraction=test_fraction,
    )
    if len(train) < 10:
        msg = f"training split too small (n={len(train)})"
        raise ValueError(msg)

    X_tr = train.loc[:, FEATURE_COLUMNS].values
    X_va = valid.loc[:, FEATURE_COLUMNS].values
    y_tr = train["y"].to_numpy(dtype=int)
    y_va = valid["y"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    log_reg = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
    )
    log_reg.fit(X_tr_s, y_tr)

    pred_lr = log_reg.predict(X_va_s)
    proba_lr = log_reg.predict_proba(X_va_s)
    report_lr = match_outcome_metrics(y_va, pred_lr, proba_lr, labels=[0, 1, 2])

    report_xgb: ClassificationReport | None = None
    xgb_model: Any | None = None
    if XGBClassifier is not None:
        xgb_model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="mlogloss",
        )
        xgb_model.fit(X_tr, y_tr)
        pred_x = xgb_model.predict(X_va)
        proba_x = xgb_model.predict_proba(X_va)
        report_xgb = match_outcome_metrics(y_va, pred_x, proba_x, labels=[0, 1, 2])

    return BaselineSuite(
        logistic=log_reg,
        log_reg_scaler=scaler,
        xgboost=xgb_model,
        report_logistic=report_lr,
        report_xgb=report_xgb,
        feature_columns=FEATURE_COLUMNS,
    )
