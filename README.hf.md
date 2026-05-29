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

The DuckDB database is the core deployment artifact. It is too large for normal
Git history, so the container looks for it at:

```text
/data/harmonic_trends.duckdb
```

Recommended options:

- Enable persistent storage for the Space and place the DuckDB file at
  `/data/harmonic_trends.duckdb`.
- Host the DuckDB file in a Hugging Face Dataset or another durable object
  store, then set `HARMONIC_DB_URL` so the container downloads it on startup.
  Persistent storage is strongly recommended so this only happens once.

Optional files:

- `/data/spotify_metadata_cache.sqlite` for cached Spotify title/artist data.
- `/data/songs_master.parquet` for chord excerpts in the song detail panel.

## Environment

Useful runtime variables:

```text
HARMONIC_DB_PATH=/data/harmonic_trends.duckdb
SPOTIFY_CACHE_PATH=/data/spotify_metadata_cache.sqlite
SONGS_MASTER_PARQUET_PATH=/data/songs_master.parquet
HARMONIC_DB_URL=
SPOTIFY_CACHE_URL=
DUCKDB_THREADS=4
```

If the database is missing, the app will still start and `/api/health` will
report `missing_database`.
