# FIFA World Cup 2026 — Interaction-Aware Football Intelligence

Research and engineering workspace for an **interaction-aware football intelligence system** aimed at FIFA World Cup 2026: ingestion through simulation, with explicit baselines before advanced modeling.

## Repository layout

| Path | Purpose |
| --- | --- |
| `data/raw` | Immutable pulls (StatsBomb Open Data, FBref exports, etc.) |
| `data/processed` | Normalized parquet/CSV aligned to shared schemas |
| `data/external` | Third-party reference tables (fifa ids, stadiums, …) |
| `notebooks/exploratory` | EDA answering possession/xG/shot-zone questions |
| `notebooks/experiments` | Hypothesis checks and temporary model drafts |
| `src/data` | Ingestion + **canonical schemas** (`schemas.py`) |
| `src/features` | Feature pipelines (tactical, form, Elo, …) |
| `src/models` | Baselines (logistic, GBDT) + metrics (`metrics.py`) |
| `src/embeddings` | Tactical / sequence embeddings (later phase) |
| `src/simulation` | Tournament and volatility simulations |
| `src/api` | FastAPI surface |
| `src/dashboard` | Dashboard helpers |
| `configs` | YAML/JSON parameters (paths, seasons, feature flags) |
| `tests` | Pytest suite |
| `models` | Serialized checkpoints and exports (gitignored) |
| `reports` | Figures and tables for writeups |

Supporting folders: `ideas`, `scratch`, `backlog`.

## Shared data schema

Single vocabulary for matches, team–match tactics, shots, and penalties lives in `src/data/schemas.py` (`SCHEMA_VERSION` + column tuples + row dataclasses). **Ingestion code must normalize to these names** before feature work.

## Phase discipline

1. **Phase 0–1** — Unified database, EDA notebooks, baseline match outcome model (Elo, form, goals, home), evaluation pipeline (accuracy, log loss, calibration). **No deep learning until baselines work.**
2. **Later** — xG engine, embeddings, PyTorch modules, penalty intelligence, simulation at scale.

## Environment (Week 1 Task A)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Torch wheels are platform-specific; adjust index URL if you use CUDA.

## Week 1 pipeline (StatsBomb + baseline)

1. **Place Open Data** — clone [statsbomb/open-data](https://github.com/statsbomb/open-data) to `data/external/statsbomb-open-data`, **or** set `STATSBOMB_OPEN_DATA_DIR` to that folder. The default competition in `configs/week1.yaml` is FIFA World Cup 2018 (`competition_id: 43`, `season_id: 3`).

2. **ETL to parquet** (from repo root; scripts add `src` to `sys.path` automatically):

   ```bash
   python scripts/run_week1_etl.py
   ```

   Options: `--open-data DIR`, `--competition-id N`, `--season-id M`, `--processed DIR` (override paths and ids from config—handy for local checks with the tiny `tests/fixtures/statsbomb` tree).

   Outputs: `data/processed/matches.parquet`, `shots_open_play.parquet`, `penalties.parquet`, `lineups.parquet`, `team_tactical.parquet`, plus `etl_meta.json` (or the same filenames under `--processed`).

3. **EDA** — open `notebooks/exploratory/week1_eda.ipynb` and run all cells (set kernel working directory to repo root or `notebooks/exploratory`; the notebook resolves `REPO` accordingly).

4. **Baseline match model** (chronological holdout, logistic regression + XGBoost if installed):

   ```bash
   python scripts/train_baseline.py
   ```

5. **Optional FBref** — `src/data/soccerdata_fbref.py` exposes `try_read_fbref_team_season_shooting()` (network + `soccerdata`); use as a second source after the unified schema is stable.

## Tests

```bash
pytest
```

`pyproject.toml` sets `pythonpath = ["src"]` so packages `data`, `features`, and `models` resolve cleanly.

## Data sources

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [soccerdata](https://soccerdata.readthedocs.io)
- [FBref](https://fbref.com)

Goal: matches, shots, lineups, and penalties in pandas, aligned to `schemas.py`.

## Phase 2 — Tactical intelligence (research milestone)

**Question:** Do tactical styles measurably improve football prediction?

### Data scope (important for WC 2026)

**Train on men's senior national-team tournaments; evaluate on World Cup holdouts.**

| Scope | Command | Use when |
| --- | --- | --- |
| **Men's senior national** (default) | `build_unified_dataset.py --config configs/data_sources_mens_national.yaml` | Train pool: Euro, Copa, AFCON, WC, … |
| **World Cup only** | `data/processed_all` from `build_wc_stack.py` | Small-N diagnostic only |
| **All national (incl. women's)** | `--scope national` | Legacy; not recommended for men's WC |

1. Place StatsBomb Open Data under `data/external/` (flat `matches/`, `events/`, `lineups/`).
2. **Recommended for WC 2026 work:**

```bash
python scripts/build_wc_stack.py
python scripts/build_unified_dataset.py --config configs/data_sources_mens_national.yaml
python scripts/run_phase2.py --rebuild-matrix
# Default: scope=mens_senior_national, eval_mode=wc_holdout_2022
```

This trains on ~330 men's senior international matches (Euro, Copa, AFCON, historical WC, …) and validates on **FIFA World Cup 2022** — the evaluation protocol closest to predicting a future World Cup.

3. Optional chronological holdout (also written when `secondary_eval_mode: chronological`):

```bash
python scripts/run_phase2.py --eval-mode chronological
```

### Extra data sources (beyond StatsBomb)

| Source | Config block | Notes |
| --- | --- | --- |
| **OpenFootball** | `openfootball:` in `data_sources_mens_national.yaml` | Free GitHub JSON/TXT — friendlies, qualifiers, Nations League, … |
| **API-Football** | `api_football:` + `API_FOOTBALL_KEY` in `.env` | Fixtures + possession/xG team stats ([dashboard](https://dashboard.api-football.com)) |

Rebuild with all sources:

```bash
python scripts/build_unified_dataset.py --config configs/data_sources_mens_national.yaml
# Skip API if no key: --skip-api-football
```

### What Model B actually uses

- **Traditional:** Elo, form, goals, home (`build_baseline_feature_table`)
- **Matchup:** 7 pre-match style gap features (`tactical_matchup.parquet`)
- **Embeddings:** rolling PCA means, cluster mismatch, style distance (`build_embedding_match_features`)
- **Leak-free norms:** expanding z-scores per competition×stage (`tactical_normalize.py`)
- **Press success:** regain within 8 events after pressure (not only `counterpress`)
- **Possession:** summed duration on `possession_team`, not event `team`

Produces:

- `tactical_matrix.parquet`, `tactical_normalized.parquet`, `tactical_embeddings.parquet`
- `tactical_matchup.parquet`, `tactical_embedding_match.parquet`
- `reports/phase2_ablation.json` — Model A vs Model B (`n_embedding_features` in report)
- `reports/phase2_feature_ablation.json` — matchup-only vs embeddings-only vs full Model B
- `reports/phase2_walkforward.json` — chronological walk-forward folds (config: `walk_forward_folds`)
- `reports/phase2_cluster_validation.json` — eta-squared cluster separation (not Pearson on cluster ids)
- `reports/phase2_pca_clusters.png`, `phase2_umap_clusters.png`
- `reports/phase2_run_meta.json` — scope, match counts, hyperparameters

Positive `delta_log_loss` means the tactical model has **lower** log loss on the chronological holdout.
Clusters are fit on **train matches only** inside the ablation.

**Phase 2 complete** when `wc_holdout_2022` ablation deltas are positive and walk-forward mean delta is stable on the men's senior pool. Next milestone: xG engine (Phase 3).
