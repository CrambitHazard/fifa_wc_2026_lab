"""Phase 2: tactical matrix, embeddings, matchup features, Model A vs B ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

from embeddings.tactical import fit_tactical_embeddings
from features.tactical_matchup import (
    build_embedding_match_features,
    build_matchup_features,
    build_pre_match_tactical_profiles,
)
from data.scope_filters import apply_match_scope, is_statsbomb_event_match_id, MATCH_SCOPE_CHOICES
from data.env_loader import load_repo_env
from data.merge_sources import build_unified_tactical_matrix
from features.tactical_normalize import normalize_tactical_matrix
from models.match_ablation import (
    EVAL_MODE_CHOICES,
    run_feature_ablation,
    run_tactical_ablation,
    run_walk_forward_ablation,
    save_ablation_report,
    save_feature_ablation_report,
    save_walk_forward_report,
)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _plot_clusters(embedded: pd.DataFrame, reports_dir: Path) -> None:
    """Save PCA and optional UMAP cluster scatter plots."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    if "pca_0" in embedded.columns and "pca_1" in embedded.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=embedded,
            x="pca_0",
            y="pca_1",
            hue="cluster_kmeans",
            palette="tab10",
            ax=ax,
            legend=True,
        )
        ax.set_title("Tactical styles (PCA projection)")
        fig.tight_layout()
        fig.savefig(reports_dir / "phase2_pca_clusters.png", dpi=150)
        plt.close(fig)

    if "umap_0" in embedded.columns and "umap_1" in embedded.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=embedded,
            x="umap_0",
            y="umap_1",
            hue="cluster_kmeans",
            palette="tab10",
            ax=ax,
        )
        ax.set_title("Tactical styles (UMAP projection)")
        fig.tight_layout()
        fig.savefig(reports_dir / "phase2_umap_clusters.png", dpi=150)
        plt.close(fig)


def _load_tactical_matrix(
    proc: Path,
    matches: pd.DataFrame,
    *,
    open_root: Path | None,
    rebuild: bool = False,
) -> pd.DataFrame:
    """Load or build tactical matrix (StatsBomb events + optional FBref rows)."""
    from data.statsbomb import default_open_data_root

    prebuilt = proc / "tactical_matrix.parquet"
    fbref_tactical: pd.DataFrame | None = None
    if prebuilt.is_file():
        cached = pd.read_parquet(prebuilt)
        non_sb_ids = set(
            matches.loc[
                ~matches["match_id"].astype(str).map(is_statsbomb_event_match_id),
                "match_id",
            ].astype(str),
        )
        if non_sb_ids:
            fbref_tactical = cached[
                cached["match_id"].astype(str).isin(non_sb_ids)
            ].copy()
        if not rebuild:
            press = cached.get("press_success_rate")
            if press is not None and (press.fillna(0) > 0).any():
                match_ids = set(matches["match_id"].astype(str))
                return cached[cached["match_id"].astype(str).isin(match_ids)].copy()
            print(
                "Cached tactical_matrix has zero press_success_rate; rebuilding SB rows...",
            )

    root = open_root or default_open_data_root()
    return build_unified_tactical_matrix(
        matches,
        open_data_root=root,
        fbref_tactical=fbref_tactical,
    )


def main() -> None:
    """Run full Phase 2 pipeline and write artifacts under data/processed and reports/."""
    load_repo_env(REPO_ROOT)
    parser = argparse.ArgumentParser(description="Phase 2 tactical intelligence pipeline")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase2.yaml")
    parser.add_argument("--open-data", type=Path, default=None)
    parser.add_argument("--processed", type=Path, default=None)
    parser.add_argument(
        "--rebuild-matrix",
        action="store_true",
        help="Force rebuild tactical_matrix from StatsBomb events",
    )
    parser.add_argument(
        "--scope",
        choices=MATCH_SCOPE_CHOICES,
        default=None,
        help=(
            "Filter matches: mens_senior_national (recommended), world_cup, "
            "national, all. Default from configs/phase2.yaml."
        ),
    )
    parser.add_argument(
        "--eval-mode",
        choices=EVAL_MODE_CHOICES,
        default=None,
        help=(
            "Evaluation split: wc_holdout_2022 trains on Euro/Copa/AFCON/etc. "
            "and validates on WC 2022 (default from config)."
        ),
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = _load_config(cfg_path)
    proc = args.processed or Path(cfg["paths"]["processed_dir"])
    if not proc.is_absolute():
        proc = REPO_ROOT / proc
    reports = Path(cfg["paths"]["reports_dir"])
    if not reports.is_absolute():
        reports = REPO_ROOT / reports

    tact_cfg = cfg.get("tactical", {})
    abl_cfg = cfg.get("ablation", {})
    window = int(tact_cfg.get("profile_window", 5))
    n_clusters = int(tact_cfg.get("n_clusters", 5))
    random_state = int(tact_cfg.get("random_state", 42))
    test_fraction = float(abl_cfg.get("test_fraction", 0.25))
    eval_mode = args.eval_mode or abl_cfg.get("eval_mode", "chronological")

    matches = pd.read_parquet(proc / "matches.parquet")
    scope = args.scope or cfg.get("scope", "world_cup")
    n_before = len(matches)
    matches = apply_match_scope(matches, scope)
    print(f"Scope={scope}: {len(matches)} matches (from {n_before})")
    if len(matches) == 0:
        raise SystemExit(
            f"No matches after scope={scope!r}. Use data/processed_all or "
            "scripts/build_wc_stack.py for World Cup-only runs.",
        )
    open_root = args.open_data

    print("Loading / building tactical feature matrix...")
    tactical = _load_tactical_matrix(
        proc,
        matches,
        open_root=open_root,
        rebuild=args.rebuild_matrix,
    )
    tactical.to_parquet(proc / "tactical_matrix.parquet", index=False)

    print("Normalizing (leak-free expanding z-scores)...")
    normalized = normalize_tactical_matrix(tactical, matches)
    normalized.to_parquet(proc / "tactical_normalized.parquet", index=False)

    print("Fitting PCA / clustering (full sample for visualization)...")
    emb_result = fit_tactical_embeddings(
        normalized,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    emb_result.frame.to_parquet(proc / "tactical_embeddings.parquet", index=False)
    with (reports / "phase2_cluster_validation.json").open("w", encoding="utf-8") as fh:
        json.dump(emb_result.validation, fh, indent=2)

    _plot_clusters(emb_result.frame, reports)

    print("Building pre-match profiles, matchup + embedding match features...")
    profiles = build_pre_match_tactical_profiles(normalized, matches, window=window)
    profiles.to_parquet(proc / "tactical_profiles.parquet", index=False)
    matchup = build_matchup_features(profiles, matches)
    matchup.to_parquet(proc / "tactical_matchup.parquet", index=False)
    emb_match = build_embedding_match_features(
        emb_result.frame,
        matches,
        profile_window=window,
    )
    emb_match.to_parquet(proc / "tactical_embedding_match.parquet", index=False)

    print(f"Running Model A vs Model B ablation (eval_mode={eval_mode})...")
    ablation = run_tactical_ablation(
        matches,
        normalized,
        profile_window=window,
        n_clusters=n_clusters,
        test_fraction=test_fraction,
        eval_mode=eval_mode,
        random_state=random_state,
    )
    save_ablation_report(
        ablation,
        reports / "phase2_ablation.json",
        eval_mode=eval_mode,
    )

    secondary_eval = abl_cfg.get("secondary_eval_mode")
    if secondary_eval and secondary_eval in EVAL_MODE_CHOICES and secondary_eval != eval_mode:
        print(f"Secondary ablation (eval_mode={secondary_eval})...")
        secondary = run_tactical_ablation(
            matches,
            normalized,
            profile_window=window,
            n_clusters=n_clusters,
            test_fraction=test_fraction,
            eval_mode=secondary_eval,
            random_state=random_state,
        )
        save_ablation_report(
            secondary,
            reports / f"phase2_ablation_{secondary_eval}.json",
            eval_mode=secondary_eval,
        )
        print(
            f"  {secondary_eval}: delta_ll_lr={secondary.delta_log_loss_lr:.4f} "
            f"delta_ll_xgb={secondary.delta_log_loss_xgb}",
        )

    if abl_cfg.get("run_feature_ablation", True):
        print("Feature ablation (matchup vs embeddings vs full)...")
        feat_rows = run_feature_ablation(
            matches,
            normalized,
            profile_window=window,
            n_clusters=n_clusters,
            test_fraction=test_fraction,
            eval_mode=eval_mode,
            random_state=random_state,
        )
        feat_report = save_feature_ablation_report(
            feat_rows,
            reports / "phase2_feature_ablation.json",
        )
        for v in feat_report["variants"]:
            print(
                f"  {v['name']}: delta_ll_lr={v['delta_log_loss_lr']:.4f} "
                f"({v['n_extra_features']} extra features)",
            )

    wf_folds = int(abl_cfg.get("walk_forward_folds", 0))
    if wf_folds > 0:
        print(f"Walk-forward ablation ({wf_folds} folds)...")
        wf = run_walk_forward_ablation(
            matches,
            normalized,
            n_folds=wf_folds,
            min_train_matches=int(abl_cfg.get("min_train_matches", 40)),
            profile_window=window,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        if wf is None:
            print("  Skipped (not enough matches for walk-forward).")
        else:
            save_walk_forward_report(wf, reports / "phase2_walkforward.json")
            print(
                f"  Mean delta LR: {wf.mean_delta_log_loss_lr:.4f} "
                f"(±{wf.std_delta_log_loss_lr:.4f})",
            )
            if wf.mean_delta_log_loss_xgb is not None:
                print(
                    f"  Mean delta XGB: {wf.mean_delta_log_loss_xgb:.4f}",
                )

    meta = {
        "scope": scope,
        "eval_mode": eval_mode,
        "n_matches": len(matches),
        "n_tactical_rows": len(tactical),
        "test_fraction": test_fraction,
        "profile_window": window,
        "n_clusters": n_clusters,
    }
    with (reports / "phase2_run_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    press = tactical["press_success_rate"].fillna(0)
    print("\n=== Phase 2 summary ===")
    print(f"Eval mode: {eval_mode}")
    print(f"Matches: {len(matches)} | Tactical rows: {len(tactical)}")
    print(f"press_success_rate mean: {press.mean():.3f} (nonzero share {(press > 0).mean():.3f})")
    print(f"Cluster validation (eta²): {emb_result.validation}")
    print(
        f"  LR variant={ablation.tactical_lr_variant} "
        f"XGB variant={ablation.tactical_xgb_variant}",
    )
    print(f"Log-loss delta LR (+ = tactical better): {ablation.delta_log_loss_lr:.4f}")
    if ablation.delta_log_loss_xgb is not None:
        print(f"Log-loss delta XGB: {ablation.delta_log_loss_xgb:.4f}")
    print(f"Reports: {reports}")


if __name__ == "__main__":
    main()
