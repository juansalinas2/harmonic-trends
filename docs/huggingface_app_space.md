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

Do not commit `data/processed/harmonic_trends.duckdb` directly to Git. The local
database is large, and the Space should treat it as runtime data.

Recommended production setup:

1. Create the Space with Docker SDK.
2. Enable persistent storage.
3. Put the database at `/data/harmonic_trends.duckdb`.
4. Optionally put `spotify_metadata_cache.sqlite` and `songs_master.parquet` in
   `/data`.

Alternative setup:

1. Upload the DuckDB file to a Hugging Face Dataset or object store.
2. Set `HARMONIC_DB_URL` in the Space variables.
3. Keep persistent storage enabled so the startup download is not repeated on
   every cold start.

## Space Variables

```text
HARMONIC_DB_PATH=/data/harmonic_trends.duckdb
SPOTIFY_CACHE_PATH=/data/spotify_metadata_cache.sqlite
SONGS_MASTER_PARQUET_PATH=/data/songs_master.parquet
HARMONIC_DB_URL=
SPOTIFY_CACHE_URL=
DUCKDB_THREADS=4
```

The app exposes `/api/health` so the Space can report whether the database is
present without running a heavy similarity query.
