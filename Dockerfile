FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=7860
ENV HARMONIC_DB_PATH=/data/harmonic_trends.duckdb
ENV SPOTIFY_CACHE_PATH=/data/spotify_metadata_cache.sqlite
ENV SONGS_MASTER_PARQUET_PATH=/data/songs_master.parquet
ENV HF_HOME=/data/.huggingface
ENV HARMONIC_DB_REPO_ID=juansalinas2/harmonic-trends-data
ENV HARMONIC_DB_FILENAME=harmonic_trends_app.duckdb
ENV SPOTIFY_CACHE_REPO_ID=juansalinas2/harmonic-trends-data
ENV SPOTIFY_CACHE_FILENAME=spotify_metadata_cache.sqlite
ENV SONGS_MASTER_REPO_ID=juansalinas2/harmonic-trends-data
ENV SONGS_MASTER_FILENAME=songs_master.parquet
ENV DUCKDB_THREADS=4

WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY app.py .
COPY static ./static
COPY scripts/start_hf_space.py ./scripts/start_hf_space.py

RUN mkdir -p /data

EXPOSE 7860

CMD ["python", "scripts/start_hf_space.py"]
