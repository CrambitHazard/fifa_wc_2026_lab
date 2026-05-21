"""Merge StatsBomb (all open competitions) + FBref into data/processed_unified."""

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
import yaml

from data.merge_sources import build_unified_tactical_matrix, merge_match_tables
from data.scope_filters import apply_match_scope
from data.pipeline import run_statsbomb_etl
from data.env_loader import load_repo_env
from data.openfootball import ingest_openfootball_bundle
from data.api_football import api_football_key, build_api_tactical_for_matches
from data.soccerdata_fbref import ingest_fbref_bundle
from data.statsbomb import default_open_data_root
from data.statsbomb_catalog import list_etl_candidates


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    """Build unified matches + tactical_matrix parquets for Phase 2."""
    load_repo_env(REPO_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "data_sources.yaml")
    parser.add_argument("--skip-fbref", action="store_true", help="Skip soccerdata FBref scrape")
    parser.add_argument("--skip-openfootball", action="store_true")
    parser.add_argument("--skip-api-football", action="store_true")
    parser.add_argument("--skip-statsbomb-extra", action="store_true")
    parser.add_argument(
        "--scope",
        choices=("all", "national", "mens_senior_national", "world_cup"),
        default=None,
        help="Filter competitions (default: scope in config or all)",
    )
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    cfg = _load_config(cfg_path)
    paths = cfg.get("paths", {})
    out_dir = Path(paths.get("processed_unified", "data/processed_unified"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wc_dir = Path(paths.get("processed_wc", "data/processed"))
    if not wc_dir.is_absolute():
        wc_dir = REPO_ROOT / wc_dir

    root = default_open_data_root()
    match_parts: list[pd.DataFrame] = []

    if (wc_dir / "matches.parquet").is_file():
        wc = pd.read_parquet(wc_dir / "matches.parquet")
        wc["data_source"] = "statsbomb"
        match_parts.append(wc)
        print(f"Loaded WC matches: {len(wc)}")

    if not args.skip_statsbomb_extra:
        sb_cfg = cfg.get("statsbomb_open_data", {})
        exclude = [
            (int(x["competition_id"]), int(x["season_id"]))
            for x in sb_cfg.get("exclude_competitions", [])
        ]
        scope = args.scope or cfg.get("scope", "all")
        candidates = (
            list_etl_candidates(root, exclude=exclude, scope=scope)
            if sb_cfg.get("auto_discover")
            else []
        )
        for item in sb_cfg.get("extra", []):
            candidates.append(
                (int(item[0]), int(item[1]), "custom", str(item[1])),
            )
        scratch = out_dir / "_sb_extra"
        scratch.mkdir(parents=True, exist_ok=True)
        for cid, sid, name, season in candidates:
            tag = f"sb_{cid}_{sid}"
            meta_path = scratch / tag / "etl_meta.json"
            if not meta_path.is_file():
                try:
                    run_statsbomb_etl(
                        open_data_root=root,
                        competition_id=cid,
                        season_id=sid,
                        processed_dir=scratch / tag,
                    )
                except FileNotFoundError:
                    continue
            mp = scratch / tag / "matches.parquet"
            if mp.is_file():
                df = pd.read_parquet(mp)
                df["data_source"] = "statsbomb"
                df["competition"] = f"{name} {season}"
                match_parts.append(df)
                print(f"StatsBomb extra: {name} {season} ({len(df)} matches)")

    fbref_tactical: pd.DataFrame | None = None
    api_football_tactical: pd.DataFrame | None = None

    if not args.skip_openfootball and cfg.get("openfootball", {}).get("enabled", False):
        of_cfg = cfg["openfootball"]
        cache = Path(of_cfg.get("cache_dir", "data/external/openfootball_cache"))
        if not cache.is_absolute():
            cache = REPO_ROOT / cache
        print("Fetching openfootball/internationals (GitHub)...")
        of_matches = ingest_openfootball_bundle(
            tournaments=list(of_cfg.get("tournaments", [])) or None,
            min_year=int(of_cfg.get("min_year", 2018)),
            cache_dir=cache,
        )
        if len(of_matches) > 0:
            match_parts.append(of_matches)
            print(f"OpenFootball matches: {len(of_matches)}")

    if not args.skip_fbref and cfg.get("fbref_soccerdata", {}).get("enabled"):
        fb = cfg["fbref_soccerdata"]
        print("Fetching FBref via soccerdata (cached after first run)...")
        fb_matches, fbref_tactical = ingest_fbref_bundle(
            leagues=list(fb.get("leagues", [])),
            seasons=list(fb.get("seasons", [])),
            competition_labels=dict(fb.get("competition_labels", {})),
        )
        if len(fb_matches) > 0:
            match_parts.append(fb_matches)
            print(f"FBref matches: {len(fb_matches)}")

    matches = merge_match_tables(*match_parts)
    scope = args.scope or cfg.get("scope", "all")
    if scope != "all":
        n0 = len(matches)
        matches = apply_match_scope(matches, scope)
        print(f"Applied scope={scope}: {len(matches)} matches (from {n0})")

    if not args.skip_api_football and cfg.get("api_football", {}).get("enabled", False):
        af_cfg = cfg["api_football"]
        if not api_football_key() and not af_cfg.get("api_key"):
            print("API-Football skipped: set API_FOOTBALL_KEY in .env")
        else:
            cache = Path(af_cfg.get("cache_dir", "data/external/api_football_cache"))
            if not cache.is_absolute():
                cache = REPO_ROOT / cache
            max_stats = af_cfg.get("max_statistics_fixtures")
            max_stats_i = int(max_stats) if max_stats is not None else None
            existing_api: pd.DataFrame | None = None
            skip_ids: set[str] = set()
            tact_path = out_dir / "tactical_matrix.parquet"
            if tact_path.is_file():
                prev_tact = pd.read_parquet(tact_path)
                if "data_source" in prev_tact.columns:
                    existing_api = prev_tact[prev_tact["data_source"] == "api_football"].copy()
                    skip_ids = set(existing_api["match_id"].astype(str).unique())
            print(
                "Fetching API-Football statistics for canonical matches "
                f"(cap={max_stats_i or 'none'}, skip={len(skip_ids)} existing)...",
            )
            _, api_new = build_api_tactical_for_matches(
                matches,
                leagues=list(af_cfg.get("leagues", [])) or None,
                cache_dir=cache,
                api_key=af_cfg.get("api_key"),
                skip_statsbomb_ids=bool(af_cfg.get("skip_statsbomb_statistics", True)),
                max_statistics_fixtures=max_stats_i,
                skip_match_ids=skip_ids,
            )
            if existing_api is not None and len(existing_api) > 0:
                api_football_tactical = pd.concat(
                    [existing_api, api_new],
                    ignore_index=True,
                )
            else:
                api_football_tactical = api_new
            print(f"API-Football tactical rows: {len(api_football_tactical)}")

    matches.to_parquet(out_dir / "matches.parquet", index=False)
    print(f"Unified matches: {len(matches)} -> {out_dir / 'matches.parquet'}")

    tactical = build_unified_tactical_matrix(
        matches,
        open_data_root=root,
        fbref_tactical=fbref_tactical,
        api_football_tactical=api_football_tactical,
    )
    tactical.to_parquet(out_dir / "tactical_matrix.parquet", index=False)
    print(f"Unified tactical rows: {len(tactical)} -> {out_dir / 'tactical_matrix.parquet'}")

    meta = {
        "n_matches": len(matches),
        "n_tactical_rows": len(tactical),
        "sources": matches["data_source"].value_counts().to_dict() if "data_source" in matches.columns else {},
        "tactical_sources": tactical["data_source"].value_counts().to_dict()
        if "data_source" in tactical.columns
        else {},
    }
    with (out_dir / "unified_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
