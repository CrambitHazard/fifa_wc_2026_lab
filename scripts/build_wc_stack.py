"""Stack FIFA World Cup 2018 + 2022 StatsBomb ETL into one processed folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from data.merge_sources import merge_match_tables
from data.pipeline import run_statsbomb_etl
from data.statsbomb import default_open_data_root

WC_SEASONS = (
    (43, 3, "FIFA World Cup 2018"),
    (43, 106, "FIFA World Cup 2022"),
)


def main() -> None:
    """Run ETL for each WC season on disk and merge into ``data/processed_all``."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed_all",
    )
    parser.add_argument("--open-data", type=Path, default=None)
    args = parser.parse_args()

    root = args.open_data or default_open_data_root()
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    scratch = out / "_wc_etl"
    scratch.mkdir(parents=True, exist_ok=True)

    match_parts: list[pd.DataFrame] = []
    for cid, sid, label in WC_SEASONS:
        tag = f"wc_{cid}_{sid}"
        proc = scratch / tag
        try:
            run_statsbomb_etl(
                open_data_root=root,
                competition_id=cid,
                season_id=sid,
                processed_dir=proc,
            )
        except FileNotFoundError:
            print(f"Skip {label}: matches file missing on disk")
            continue
        mp = proc / "matches.parquet"
        if mp.is_file():
            df = pd.read_parquet(mp)
            df["data_source"] = "statsbomb"
            df["competition"] = label
            match_parts.append(df)
            print(f"{label}: {len(df)} matches")

    if not match_parts:
        raise SystemExit("No WC match files found under data/external")

    matches = merge_match_tables(*match_parts)
    out.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out / "matches.parquet", index=False)

    meta = {
        "sources": ["statsbomb"],
        "seasons": [label for _, _, label in WC_SEASONS],
        "n_matches": len(matches),
    }
    with (out / "etl_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Wrote {len(matches)} matches -> {out / 'matches.parquet'}")


if __name__ == "__main__":
    main()
