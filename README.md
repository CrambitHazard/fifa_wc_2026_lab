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
