"""PCA / UMAP / clustering for tactical style discovery (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from features.tactical_matrix import RATE_COLUMNS

try:
    import umap

    HAS_UMAP = True
except ImportError:  # pragma: no cover
    HAS_UMAP = False


NORM_SUFFIX = "_norm"


def norm_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return normalized tactical columns present in ``frame``."""
    cols = [f"{c}{NORM_SUFFIX}" for c in RATE_COLUMNS]
    return [c for c in cols if c in frame.columns]


@dataclass(frozen=True)
class TacticalEmbeddingResult:
    """Fitted reducers, cluster labels, and validation summary."""

    frame: pd.DataFrame
    pca: PCA
    scaler: StandardScaler
    kmeans: KMeans
    gmm: GaussianMixture
    validation: dict[str, Any]
    feature_columns: tuple[str, ...]


def _anova_style_score(merged: pd.DataFrame, col: str, cluster_col: str = "cluster") -> float | None:
    """Eta-squared style separation of ``col`` across clusters (0–1, higher = better)."""
    if merged[col].notna().sum() < 5:
        return None
    groups = [g[col].dropna().values for _, g in merged.groupby(cluster_col)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return None
    overall = merged[col].dropna().values
    if len(overall) < 5:
        return None
    grand_mean = float(np.mean(overall))
    ss_total = float(np.sum((overall - grand_mean) ** 2))
    if ss_total <= 1e-9:
        return None
    ss_between = 0.0
    for g in groups:
        ss_between += len(g) * (float(np.mean(g)) - grand_mean) ** 2
    return ss_between / ss_total


def _cluster_validation(
    embedded: pd.DataFrame,
    labels: np.ndarray,
    raw: pd.DataFrame,
) -> dict[str, Any]:
    """Measure whether clusters separate interpretable tactical rates."""
    check_cols = [
        "possession_raw",
        "ppda_raw",
        "counter_attacks_raw",
        "defensive_line_height_raw",
        "press_success_rate_raw",
    ]
    available = [c for c in check_cols if c in raw.columns]
    merged = embedded.copy()
    merged["cluster"] = labels
    merged = merged.merge(
        raw[["team", "match_id"] + available],
        on=["team", "match_id"],
        how="left",
    )
    separation: dict[str, float | None] = {}
    for col in available:
        separation[col] = _anova_style_score(merged, col)
    return {
        "cluster_eta_squared": separation,
        "n_clusters": int(len(np.unique(labels))),
        "n_samples": int(len(labels)),
    }


def fit_tactical_embeddings(
    normalized: pd.DataFrame,
    *,
    n_clusters: int = 5,
    random_state: int = 42,
    fit_frame: pd.DataFrame | None = None,
) -> TacticalEmbeddingResult:
    """PCA (+ optional UMAP) and KMeans / GMM on normalized tactical rows.

    Args:
        normalized: Context-normalized tactical matrix.
        n_clusters: KMeans / GMM cluster count.
        random_state: RNG seed.
        fit_frame: If set, fit reducers/clusters only on these rows (train split).

    Returns:
        TacticalEmbeddingResult with enriched dataframe and validation dict.
    """
    feat_cols = norm_feature_columns(normalized)
    if not feat_cols:
        msg = "normalized frame missing *_norm tactical columns"
        raise ValueError(msg)

    full = normalized.copy()
    full["match_id"] = full["match_id"].astype(str)
    train = fit_frame if fit_frame is not None else full
    train = train.copy()
    train["match_id"] = train["match_id"].astype(str)

    X_train = train[feat_cols].astype(float).fillna(0.0).values
    X_all = full[feat_cols].astype(float).fillna(0.0).values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_all_s = scaler.transform(X_all)

    n_comp = min(3, X_train_s.shape[1], max(1, X_train_s.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    pcs_train = pca.fit_transform(X_train_s)
    pcs_all = pca.transform(X_all_s)

    out = full[["team", "match_id"]].copy()
    for i in range(pcs_all.shape[1]):
        out[f"pca_{i}"] = pcs_all[:, i]

    if HAS_UMAP and X_train_s.shape[0] >= 15:
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        reducer.fit(X_train_s)
        um = reducer.transform(X_all_s)
        out["umap_0"] = um[:, 0]
        out["umap_1"] = um[:, 1]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_train_s)
    labels = kmeans.predict(X_all_s)
    out["cluster_kmeans"] = labels

    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm.fit(X_train_s)
    out["cluster_gmm"] = gmm.predict(X_all_s)

    validation = _cluster_validation(out, labels, normalized)
    return TacticalEmbeddingResult(
        frame=out,
        pca=pca,
        scaler=scaler,
        kmeans=kmeans,
        gmm=gmm,
        validation=validation,
        feature_columns=tuple(feat_cols),
    )
