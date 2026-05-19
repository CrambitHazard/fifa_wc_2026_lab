"""Run StatsBomb -> parquet ETL using :data:`configs/week1.yaml`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml

from data.pipeline import run_statsbomb_etl
from data.statsbomb import default_open_data_root


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    """CLI entry for Week 1 ingestion."""
    parser = argparse.ArgumentParser(description="StatsBomb Open Data ETL")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "week1.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--open-data",
        type=Path,
        default=None,
        help="Override Open Data root (else env STATSBOMB_OPEN_DATA_DIR or default path)",
    )
    parser.add_argument(
        "--processed",
        type=Path,
        default=None,
        help="Override output processed dir (default from config)",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        default=None,
        help="Override StatsBomb competition id from config",
    )
    parser.add_argument(
        "--season-id",
        type=int,
        default=None,
        help="Override StatsBomb season id from config",
    )
    args = parser.parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = _load_config(cfg_path)
    sb = cfg["statsbomb"]
    proc = args.processed or Path(cfg["paths"]["processed_dir"])
    if not proc.is_absolute():
        proc = REPO_ROOT / proc

    root = args.open_data or default_open_data_root()
    comp_id = int(args.competition_id if args.competition_id is not None else sb["competition_id"])
    season_id = int(args.season_id if args.season_id is not None else sb["season_id"])
    meta = run_statsbomb_etl(
        open_data_root=root,
        competition_id=comp_id,
        season_id=season_id,
        processed_dir=proc.resolve(),
    )
    print(meta)


if __name__ == "__main__":
    main()
