"""Baseline evaluation utilities for classifiers (match outcome, etc.).

Keeps a single place for accuracy, proper scoring rules, and calibration
diagnostics so advanced modules can be compared to baselines fairly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


@dataclass(frozen=True)
class ClassificationReport:
    """Bundle standard metrics for multiclass or binary match models.

    Attributes:
        accuracy: Fraction of correct class predictions.
        log_loss: Negative log-likelihood of true labels under predicted probs.
        brier_score: Mean squared error of predicted win probabilities (binary).
        calibration_bins: Fraction of positives per probability bin.
        calibration_mean_pred: Mean predicted prob per bin.
    """

    accuracy: float
    log_loss: float
    brier_score: float | None
    calibration_bins: np.ndarray | None
    calibration_mean_pred: np.ndarray | None


def match_outcome_metrics(
    y_true: np.ndarray,
    y_pred_labels: np.ndarray,
    y_pred_proba: np.ndarray | None,
    *,
    labels: list[int] | None = None,
) -> ClassificationReport:
    """Compute baseline metrics for a match-outcome classifier.

    Args:
        y_true: Ground-truth integer class indices (shape ``(n_samples,)``).
        y_pred_labels: Predicted class indices from ``predict``.
        y_pred_proba: Predicted probabilities from ``predict_proba`` when
            available; required for log loss and calibration summaries.
        labels: Optional explicit class label list for log loss.

    Returns:
        ClassificationReport with scalar metrics and optional calibration curve
        arrays for the positive class in binary problems.
    """
    acc = float(accuracy_score(y_true, y_pred_labels))
    ll = float("nan")
    brier: float | None = None
    cal_bins: np.ndarray | None = None
    cal_pred: np.ndarray | None = None

    if y_pred_proba is not None:
        ll = float(log_loss(y_true, y_pred_proba, labels=labels))
        if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
            pos_prob = y_pred_proba[:, 1]
            brier = float(brier_score_loss(y_true, pos_prob))
            cal_bins, cal_pred = calibration_curve(
                y_true,
                pos_prob,
                n_bins=10,
                strategy="uniform",
            )

    return ClassificationReport(
        accuracy=acc,
        log_loss=ll,
        brier_score=brier,
        calibration_bins=cal_bins,
        calibration_mean_pred=cal_pred,
    )


def to_serializable_report(report: ClassificationReport) -> dict[str, Any]:
    """Convert a report into JSON-friendly primitives for logging or MLflow.

    Args:
        report: Output from :func:`match_outcome_metrics`.

    Returns:
        Dict with floats and nested lists instead of numpy arrays.
    """
    payload: dict[str, Any] = {
        "accuracy": report.accuracy,
        "log_loss": report.log_loss,
        "brier_score": report.brier_score,
    }
    if report.calibration_bins is not None:
        payload["calibration_bins"] = report.calibration_bins.tolist()
    if report.calibration_mean_pred is not None:
        payload["calibration_mean_pred"] = report.calibration_mean_pred.tolist()
    return payload
