---
title: Harmonic Trends Explorer
sdk: docker
app_port: 7860
pinned: false
---

# Harmonic Trends Explorer

Can harmonic structures be learned by a computer? **Yes.**

This Space serves the DuckDB-backed harmonic explorer. The database stores
learned harmonic vocabulary statistics; the app searches songs, retrieves
neighbors, and explains similarity through shared transposition-invariant chord
n-grams.

The app expects the similarity index built by notebook 12:

- `song_harmonic_terms`
- `song_harmonic_tfidf`
- `harmonic_song_document_frequency`
- `song_harmonic_norm_components`
- `song_harmonic_totals`
- `song_metadata`
- `song_harmonic_similarity_index_metadata`

## Core Data

The app DuckDB is the core deployment artifact. It is built from the full local
research database with:

```bash
python3 scripts/build_app_duckdb.py --replace
```

The built file is `data/processed/harmonic_trends_app.duckdb`. It is kept out of
Git and stored in a private Hugging Face Dataset. The container downloads it to:

```text
/data/harmonic_trends.duckdb
```

Recommended setup:

- Store the DuckDB in the private Dataset
  `juansalinas2/harmonic-trends-data`.
- Add a Space secret named `HF_TOKEN` with read access to that Dataset.
- On startup, the container downloads the DB to `/data/harmonic_trends.duckdb`
  if it is missing.

Optional files:

- `/data/spotify_metadata_cache.sqlite` for cached Spotify title/artist data.
- `/data/songs_master.parquet` for chord excerpts in the song detail panel.

## Environment

Useful runtime variables:

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

If the database is missing, the app will still start and `/api/health` will
report `missing_database`.
