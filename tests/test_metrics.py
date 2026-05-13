"""Smoke tests for baseline evaluation helpers."""

from __future__ import annotations

import numpy as np
from models.metrics import match_outcome_metrics, to_serializable_report


def test_match_outcome_metrics_binary() -> None:
    """Binary probs should yield finite log loss and calibration objects."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.6, 0.4], [0.4, 0.6]])
    report = match_outcome_metrics(y_true, y_pred, proba)
    assert report.accuracy == 0.75
    assert np.isfinite(report.log_loss)
    assert report.brier_score is not None
    assert report.calibration_bins is not None


def test_to_serializable_roundtrip_keys() -> None:
    """Serialized reports should omit numpy containers."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])
    report = match_outcome_metrics(y_true, y_pred, proba)
    payload = to_serializable_report(report)
    assert isinstance(payload["calibration_bins"], list)
