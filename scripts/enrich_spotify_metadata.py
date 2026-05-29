from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from pathlib import Path
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from app import (  # noqa: E402
    DB_PATH,
    cache_spotify_artist_metadata,
    cache_spotify_metadata,
    cached_spotify_artist_metadata,
    cached_spotify_metadata,
    spotify_embed_url,
    spotify_artist_page_metadata,
    spotify_artist_url,
    spotify_page_metadata,
    spotify_url,
)


def spotify_ids(limit: int, offset: int) -> list[str]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT
                m.spotify_song_id,
                SUM(t.unique_indexed_harmonic_classes) AS indexed_features
            FROM song_metadata m
            JOIN song_harmonic_totals t ON t.song_id = m.id
            WHERE m.spotify_song_id IS NOT NULL
              AND m.spotify_song_id <> ''
            GROUP BY m.spotify_song_id
            ORDER BY indexed_features DESC, m.spotify_song_id
            LIMIT ? OFFSET ?
            """,
            [limit, offset],
        ).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def spotify_artist_ids(limit: int, offset: int) -> list[str]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = con.execute(
            """
            SELECT
                m.spotify_artist_id,
                COUNT(DISTINCT m.id) AS song_count,
                SUM(t.unique_indexed_harmonic_classes) AS indexed_features
            FROM song_metadata m
            JOIN song_harmonic_totals t ON t.song_id = m.id
            WHERE m.spotify_artist_id IS NOT NULL
              AND m.spotify_artist_id <> ''
            GROUP BY m.spotify_artist_id
            ORDER BY song_count DESC, indexed_features DESC, m.spotify_artist_id
            LIMIT ? OFFSET ?
            """,
            [limit, offset],
        ).fetchall()
    finally:
        con.close()
    return [row[0] for row in rows]


def fetch_one(spotify_song_id: str, refresh: bool) -> tuple[str, str, str | None]:
    if not refresh:
        cached = cached_spotify_metadata(spotify_song_id)
        if cached and (cached.get("title") or cached.get("artist")):
            return spotify_song_id, "cached", None

    try:
        page = spotify_page_metadata(spotify_song_id)
        payload = {
            "spotify_song_id": spotify_song_id,
            "spotify_url": spotify_url(spotify_song_id),
            "spotify_embed_url": spotify_embed_url(spotify_song_id),
            "title": page.get("title"),
            "artist": page.get("artist"),
            "description": page.get("description"),
            "preview_url": page.get("preview_url"),
        }
        cache_spotify_metadata(payload)
        return spotify_song_id, "fetched", None
    except Exception as exc:
        cache_spotify_metadata(
            {
                "spotify_song_id": spotify_song_id,
                "spotify_url": spotify_url(spotify_song_id),
                "spotify_embed_url": spotify_embed_url(spotify_song_id),
                "error": str(exc),
            }
        )
        return spotify_song_id, "error", str(exc)


def fetch_artist_one(spotify_artist_id: str, refresh: bool) -> tuple[str, str, str | None]:
    if not refresh:
        cached = cached_spotify_artist_metadata(spotify_artist_id)
        if cached and cached.get("name"):
            return spotify_artist_id, "cached", None

    try:
        page = spotify_artist_page_metadata(spotify_artist_id)
    except HTTPError as exc:
        if exc.code == 429:
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else 8
            time.sleep(delay)
            try:
                page = spotify_artist_page_metadata(spotify_artist_id)
            except Exception as retry_exc:
                return cache_artist_error(spotify_artist_id, retry_exc)
        else:
            return cache_artist_error(spotify_artist_id, exc)
    except Exception as exc:
        return cache_artist_error(spotify_artist_id, exc)

    payload = {
        "spotify_artist_id": spotify_artist_id,
        "spotify_url": spotify_artist_url(spotify_artist_id),
        "name": page.get("name"),
        "description": page.get("description"),
        "thumbnail_url": page.get("thumbnail_url"),
    }
    cache_spotify_artist_metadata(payload)
    return spotify_artist_id, "fetched", None


def cache_artist_error(
    spotify_artist_id: str,
    exc: Exception,
) -> tuple[str, str, str | None]:
    cache_spotify_artist_metadata(
        {
            "spotify_artist_id": spotify_artist_id,
            "spotify_url": spotify_artist_url(spotify_artist_id),
            "error": str(exc),
        }
    )
    return spotify_artist_id, "error", str(exc)


def progress(done: int, total: int, counts: dict[str, int], started: float) -> None:
    width = 28
    filled = int(width * done / max(total, 1))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - started, 0.001)
    rate = done / elapsed
    sys.stdout.write(
        f"\r[{bar}] {done}/{total} "
        f"fetched={counts['fetched']} cached={counts['cached']} "
        f"errors={counts['error']} {rate:.2f}/s"
    )
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preload Spotify title/artist metadata for app name search."
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--artists",
        action="store_true",
        help="Preload Spotify artist names instead of track titles.",
    )
    args = parser.parse_args()

    ids = (
        spotify_artist_ids(args.limit, args.offset)
        if args.artists
        else spotify_ids(args.limit, args.offset)
    )
    fetch = fetch_artist_one if args.artists else fetch_one
    counts = {"fetched": 0, "cached": 0, "error": 0}
    started = time.time()
    progress(0, len(ids), counts, started)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch, spotify_id, args.refresh) for spotify_id in ids]
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            _, status, _ = future.result()
            counts[status] += 1
            progress(done, len(ids), counts, started)

    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
