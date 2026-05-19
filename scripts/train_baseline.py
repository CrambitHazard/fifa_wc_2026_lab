"""Train Week-1 baseline outcome models from processed parquets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import yaml

from features.elo import attach_pre_match_elo
from features.match_baseline import build_baseline_feature_table
from models.baseline_match import train_baselines
from models.metrics import to_serializable_report


def main() -> None:
    """Load ``matches.parquet``, build features, print validation metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "week1.yaml",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Override processed dir (default from config)",
    )
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = yaml.safe_load(cfg_path.open(encoding="utf-8"))
    proc = args.processed or Path(cfg["paths"]["processed_dir"])
    if not proc.is_absolute():
        proc = REPO_ROOT / proc
    bl = cfg.get("baseline", {})
    test_fraction = float(bl.get("test_fraction", 0.25))
    random_state = int(bl.get("random_state", 42))

    matches = pd.read_parquet(proc / "matches.parquet")
    with_elo = attach_pre_match_elo(matches)
    feats = build_baseline_feature_table(with_elo)
    suite = train_baselines(
        feats,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    print("Logistic regression:", to_serializable_report(suite.report_logistic))
    if suite.report_xgb is not None:
        print("XGBoost:", to_serializable_report(suite.report_xgb))


if __name__ == "__main__":
    main()
