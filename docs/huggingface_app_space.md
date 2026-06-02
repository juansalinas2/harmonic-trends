# Hugging Face App Space

The live harmonic explorer needs a Python API and read-only DuckDB access, so it
should run as a Hugging Face Docker Space. The older `spaces/harmonic-trends/`
bundle remains a static Space for the standalone chart modules.

## Runtime Files

The Docker Space uses:

- `Dockerfile`
- `requirements-app.txt`
- `app.py`
- `static/`
- `scripts/start_hf_space.py`
- `README.hf.md` as the Space `README.md`

The deployment helper uploads only those files.

```bash
HF_TOKEN=... python3 scripts/deploy_hf_app_space.py --repo-id USERNAME/harmonic-trends-explorer
```

## Data Strategy

Do not commit DuckDB files directly to Git. The full local research database is
large, and the Space should treat data as runtime artifacts.

Build the compact app database first:

```bash
python3 scripts/build_app_duckdb.py --replace
```

This writes `data/processed/harmonic_trends_app.duckdb`, which keeps only the
tables needed by the explorer.

Recommended free setup:

1. Create the Space with Docker SDK.
2. Create a private Dataset at `juansalinas2/harmonic-trends-data`.
3. Upload `harmonic_trends_app.duckdb`.
4. Add `HF_TOKEN` as a Space secret with read access to the Dataset.
5. Let startup download the file into `/data/harmonic_trends.duckdb`.

Lower-latency paid setup:

1. Enable persistent Space storage.
2. Keep the same Dataset source of truth.
3. Use `/data` as a persistent cache so the DB is downloaded only once.

## Space Variables

```text
HARMONIC_DB_PATH=/data/harmonic_trends.duckdb
HARMONIC_DB_REPO_ID=juansalinas2/harmonic-trends-data
HARMONIC_DB_FILENAME=harmonic_trends_app.duckdb
SPOTIFY_CACHE_PATH=/data/spotify_metadata_cache.sqlite
SPOTIFY_CACHE_REPO_ID=juansalinas2/harmonic-trends-data
SPOTIFY_CACHE_FILENAME=spotify_metadata_cache.sqlite
SONGS_MASTER_PARQUET_PATH=/data/songs_master.parquet
SONGS_MASTER_REPO_ID=juansalinas2/harmonic-trends-data
SONGS_MASTER_FILENAME=songs_master.parquet
HARMONIC_DB_URL=
SPOTIFY_CACHE_URL=
DUCKDB_THREADS=4
```

The app exposes `/api/health` so the Space can report whether the database is
present without running a heavy similarity query.
