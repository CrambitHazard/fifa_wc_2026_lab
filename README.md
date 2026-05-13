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

## Tests

```bash
pytest
```

`pyproject.toml` sets `pythonpath = ["src"]` so packages `data` and `models` resolve cleanly.

## Data sources (Task B)

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
- [soccerdata](https://soccerdata.readthedocs.io)
- [FBref](https://fbref.com)

Goal: matches, shots, lineups, penalties in pandas, aligned to `schemas.py`.
