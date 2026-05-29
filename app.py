from __future__ import annotations

import json
import math
import mimetypes
import html
import os
import re
import sqlite3
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parent


def configured_path(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default))).expanduser()


def configured_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{env_name} must be an integer, got {raw!r}") from exc


DB_PATH = configured_path(
    "HARMONIC_DB_PATH",
    ROOT / "data" / "processed" / "harmonic_trends.duckdb",
)
STATIC_DIR = ROOT / "static"
SPOTIFY_CACHE_PATH = configured_path(
    "SPOTIFY_CACHE_PATH",
    ROOT / "data" / "processed" / "spotify_metadata_cache.sqlite",
)
SONGS_MASTER_PATH = configured_path(
    "SONGS_MASTER_PARQUET_PATH",
    ROOT / "data" / "processed" / "songs_master.parquet",
)
DUCKDB_THREADS = max(1, configured_int("DUCKDB_THREADS", 4))
SERVER_HOST = os.environ.get("HOST", "127.0.0.1")
SERVER_PORT = configured_int("PORT", 8000)

NS = tuple(range(3, 9))
DEFAULT_N_WEIGHTS = {n: n / min(NS) for n in NS}
DEFAULT_WEIGHT_COLUMN = "tfidf_log_count"
SPOTIFY_OEMBED_CACHE: dict[str, dict] = {}
SPOTIFY_ID_RE = re.compile(r"(?:spotify:track:|open\.spotify\.com/track/)?([A-Za-z0-9]{22})")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def connect() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise ApiError(500, f"Missing DuckDB database at {DB_PATH}")
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    return con


def rows_to_records(df: pd.DataFrame) -> list[dict]:
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    return records


def spotify_cache() -> sqlite3.Connection:
    SPOTIFY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(SPOTIFY_CACHE_PATH)
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS spotify_track_metadata (
            spotify_song_id TEXT PRIMARY KEY,
            title TEXT,
            artist TEXT,
            description TEXT,
            thumbnail_url TEXT,
            preview_url TEXT,
            spotify_url TEXT,
            spotify_embed_url TEXT,
            error TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS spotify_track_metadata_search_idx
        ON spotify_track_metadata(title, artist)
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS spotify_artist_metadata (
            spotify_artist_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            thumbnail_url TEXT,
            spotify_url TEXT,
            error TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS spotify_artist_metadata_search_idx
        ON spotify_artist_metadata(name)
        """
    )
    return con


def cached_spotify_metadata(spotify_song_id: str) -> dict | None:
    with spotify_cache() as con:
        row = con.execute(
            """
            SELECT spotify_song_id, title, artist, description, thumbnail_url,
                   preview_url, spotify_url, spotify_embed_url, error
            FROM spotify_track_metadata
            WHERE spotify_song_id = ?
            """,
            [spotify_song_id],
        ).fetchone()
    return dict(row) if row else None


def cache_spotify_metadata(payload: dict) -> None:
    if not payload.get("spotify_song_id"):
        return
    with spotify_cache() as con:
        con.execute(
            """
            INSERT INTO spotify_track_metadata (
                spotify_song_id, title, artist, description, thumbnail_url,
                preview_url, spotify_url, spotify_embed_url, error, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(spotify_song_id) DO UPDATE SET
                title = excluded.title,
                artist = excluded.artist,
                description = excluded.description,
                thumbnail_url = excluded.thumbnail_url,
                preview_url = excluded.preview_url,
                spotify_url = excluded.spotify_url,
                spotify_embed_url = excluded.spotify_embed_url,
                error = excluded.error,
                updated_at = CURRENT_TIMESTAMP
            """,
            [
                payload.get("spotify_song_id"),
                payload.get("title"),
                payload.get("artist") or payload.get("author_name"),
                payload.get("description"),
                payload.get("thumbnail_url"),
                payload.get("preview_url"),
                payload.get("spotify_url"),
                payload.get("spotify_embed_url"),
                payload.get("error") or payload.get("page_error"),
            ],
        )


def cached_spotify_artist_metadata(spotify_artist_id: str) -> dict | None:
    with spotify_cache() as con:
        row = con.execute(
            """
            SELECT spotify_artist_id, name, description, thumbnail_url,
                   spotify_url, error
            FROM spotify_artist_metadata
            WHERE spotify_artist_id = ?
            """,
            [spotify_artist_id],
        ).fetchone()
    return dict(row) if row else None


def cache_spotify_artist_metadata(payload: dict) -> None:
    if not payload.get("spotify_artist_id"):
        return
    with spotify_cache() as con:
        con.execute(
            """
            INSERT INTO spotify_artist_metadata (
                spotify_artist_id, name, description, thumbnail_url,
                spotify_url, error, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(spotify_artist_id) DO UPDATE SET
                name = excluded.name,
                description = excluded.description,
                thumbnail_url = excluded.thumbnail_url,
                spotify_url = excluded.spotify_url,
                error = excluded.error,
                updated_at = CURRENT_TIMESTAMP
            """,
            [
                payload.get("spotify_artist_id"),
                payload.get("name"),
                payload.get("description"),
                payload.get("thumbnail_url"),
                payload.get("spotify_url"),
                payload.get("error"),
            ],
        )


def lookup_cached_spotify_ids(query: str, limit: int) -> list[str]:
    like = f"%{query.lower()}%"
    with spotify_cache() as con:
        rows = con.execute(
            """
            SELECT spotify_song_id
            FROM spotify_track_metadata
            WHERE LOWER(COALESCE(title, '')) LIKE ?
               OR LOWER(COALESCE(artist, '')) LIKE ?
               OR LOWER(COALESCE(description, '')) LIKE ?
            ORDER BY
                CASE
                    WHEN LOWER(COALESCE(title, '')) = LOWER(?) THEN 0
                    WHEN LOWER(COALESCE(artist, '')) = LOWER(?) THEN 1
                    WHEN LOWER(COALESCE(title, '')) LIKE LOWER(?) THEN 2
                    WHEN LOWER(COALESCE(artist, '')) LIKE LOWER(?) THEN 3
                    ELSE 4
                END,
                title,
                artist
            LIMIT ?
            """,
            [like, like, like, query, query, f"{query}%", f"{query}%", limit],
        ).fetchall()
    return [row["spotify_song_id"] for row in rows]


def lookup_cached_spotify_artist_ids(query: str, limit: int) -> list[str]:
    like = f"%{query.lower()}%"
    with spotify_cache() as con:
        rows = con.execute(
            """
            SELECT spotify_artist_id
            FROM spotify_artist_metadata
            WHERE LOWER(COALESCE(name, '')) LIKE ?
               OR LOWER(COALESCE(description, '')) LIKE ?
            ORDER BY
                CASE
                    WHEN LOWER(COALESCE(name, '')) = LOWER(?) THEN 0
                    WHEN LOWER(COALESCE(name, '')) LIKE LOWER(?) THEN 1
                    ELSE 2
                END,
                name
            LIMIT ?
            """,
            [like, like, query, f"{query}%", limit],
        ).fetchall()
    return [row["spotify_artist_id"] for row in rows]


def add_cached_spotify_fields(records: list[dict]) -> list[dict]:
    spotify_ids = sorted(
        {
            str(record.get("spotify_song_id"))
            for record in records
            if record.get("spotify_song_id")
        }
    )
    if not spotify_ids:
        return records
    placeholders = ",".join("?" for _ in spotify_ids)
    with spotify_cache() as con:
        rows = con.execute(
            f"""
            SELECT spotify_song_id, title, artist, description, thumbnail_url, preview_url
            FROM spotify_track_metadata
            WHERE spotify_song_id IN ({placeholders})
            """,
            spotify_ids,
        ).fetchall()
    cache = {row["spotify_song_id"]: dict(row) for row in rows}
    for record in records:
        meta = cache.get(str(record.get("spotify_song_id")))
        if meta:
            record["spotify_title"] = meta.get("title")
            record["spotify_artist"] = meta.get("artist")
            record["spotify_description"] = meta.get("description")
            record["spotify_thumbnail_url"] = meta.get("thumbnail_url")
            record["spotify_preview_url"] = meta.get("preview_url")

    spotify_artist_ids = sorted(
        {
            str(record.get("spotify_artist_id"))
            for record in records
            if record.get("spotify_artist_id")
        }
    )
    if not spotify_artist_ids:
        return records
    placeholders = ",".join("?" for _ in spotify_artist_ids)
    with spotify_cache() as con:
        rows = con.execute(
            f"""
            SELECT spotify_artist_id, name, description, thumbnail_url, spotify_url
            FROM spotify_artist_metadata
            WHERE spotify_artist_id IN ({placeholders})
            """,
            spotify_artist_ids,
        ).fetchall()
    artist_cache = {row["spotify_artist_id"]: dict(row) for row in rows}
    for record in records:
        meta = artist_cache.get(str(record.get("spotify_artist_id")))
        if meta:
            record["spotify_artist_name"] = meta.get("name")
            record["spotify_artist_description"] = meta.get("description")
            record["spotify_artist_thumbnail_url"] = meta.get("thumbnail_url")
            record["spotify_artist_url"] = meta.get("spotify_url")
            if not record.get("spotify_artist"):
                record["spotify_artist"] = meta.get("name")
    return records


def parse_spotify_track_id(value: str) -> str | None:
    match = SPOTIFY_ID_RE.search(value)
    return match.group(1) if match else None


def spotify_url(spotify_song_id: str | None) -> str | None:
    if not spotify_song_id:
        return None
    return f"https://open.spotify.com/track/{spotify_song_id}"


def spotify_embed_url(spotify_song_id: str | None) -> str | None:
    if not spotify_song_id:
        return None
    return f"https://open.spotify.com/embed/track/{spotify_song_id}"


def spotify_artist_url(spotify_artist_id: str | None) -> str | None:
    if not spotify_artist_id:
        return None
    return f"https://open.spotify.com/artist/{spotify_artist_id}"


def fetch_json(url: str, timeout: float = 4) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "harmonic-trends-local-explorer/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str, timeout: float = 4) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html",
            "Accept-Encoding": "identity",
            "User-Agent": "harmonic-trends-local-explorer/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def html_meta(content: str, attr_name: str, attr_value: str) -> str | None:
    attr = re.escape(attr_name)
    value = re.escape(attr_value)
    patterns = (
        rf'<meta[^>]+{attr}=["\']{value}["\'][^>]+content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+{attr}=["\']{value}["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return html.unescape(match.group(1)).strip()
    return None


def spotify_page_metadata(spotify_song_id: str) -> dict:
    track_url = spotify_url(spotify_song_id)
    assert track_url is not None
    content = fetch_text(track_url)
    title = html_meta(content, "property", "og:title")
    description = html_meta(content, "property", "og:description")
    artist = html_meta(content, "name", "music:musician_description")
    if not artist and description:
        artist = description.split(" · ", 1)[0].strip() or None
    return {
        "title": title,
        "artist": artist,
        "description": description,
        "preview_url": html_meta(content, "property", "og:audio"),
    }


def spotify_artist_page_metadata(spotify_artist_id: str) -> dict:
    artist_url = spotify_artist_url(spotify_artist_id)
    assert artist_url is not None
    content = fetch_text(artist_url)
    return {
        "name": html_meta(content, "property", "og:title"),
        "description": html_meta(content, "property", "og:description"),
        "thumbnail_url": html_meta(content, "property", "og:image"),
    }


def chord_excerpt(con: duckdb.DuckDBPyConnection, song_id: int) -> str | None:
    if not SONGS_MASTER_PATH.exists():
        return None
    try:
        row = con.execute(
            """
            SELECT chords_nosections
            FROM read_parquet(?)
            WHERE id = ?
            LIMIT 1
            """,
            [str(SONGS_MASTER_PATH), song_id],
        ).fetchone()
    except duckdb.Error:
        return None
    if not row or not row[0]:
        return None
    tokens = str(row[0]).split()
    excerpt = " ".join(tokens[:24])
    return f"{excerpt}..." if len(tokens) > 24 else excerpt


def parse_int(value: str | None, default: int | None = None) -> int:
    if value in (None, ""):
        if default is None:
            raise ApiError(400, "Missing integer parameter")
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ApiError(400, f"Invalid integer parameter: {value}") from exc


def bounded_int(
    params: dict[str, list[str]],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = parse_int(params.get(key, [str(default)])[0], default)
    return max(minimum, min(value, maximum))


def parse_ns(params: dict[str, list[str]]) -> tuple[int, ...]:
    raw = params.get("ns", ["3,4,5,6,7,8"])[0]
    try:
        selected = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ApiError(400, "ns must be a comma-separated subset of 3,4,5,6,7,8") from exc
    if not selected or any(n not in NS for n in selected):
        raise ApiError(400, "ns must be a non-empty subset of 3,4,5,6,7,8")
    return selected


def parse_weights(params: dict[str, list[str]], selected_ns: tuple[int, ...]) -> dict[int, float]:
    raw = params.get("weights", [""])[0]
    if not raw:
        return {n: DEFAULT_N_WEIGHTS[n] for n in selected_ns}

    values: dict[int, float] = {}
    try:
        for assignment in raw.split(","):
            if not assignment.strip():
                continue
            n_text, weight_text = assignment.split(":", 1)
            n = int(n_text)
            weight = float(weight_text)
            if n not in NS or not math.isfinite(weight) or weight < 0:
                raise ValueError
            values[n] = min(weight, 10.0)
    except ValueError as exc:
        raise ApiError(400, "weights must look like 3:1,4:1.33,5:1.67") from exc

    return {n: values.get(n, DEFAULT_N_WEIGHTS[n]) for n in selected_ns}


def parse_spotify_mode(params: dict[str, list[str]]) -> str:
    mode = params.get("spotify", ["all"])[0].strip().lower()
    if mode not in {"all", "only", "prefer"}:
        raise ApiError(400, "spotify must be all, only, or prefer")
    return mode


def parse_bool(params: dict[str, list[str]], key: str, default: bool = False) -> bool:
    raw = params.get(key, ["1" if default else "0"])[0].strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    raise ApiError(400, f"{key} must be boolean")


def dedupe_spotify_records(
    records: list[dict],
    *,
    id_key: str,
    limit: int,
) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for record in records:
        spotify_song_id = record.get("spotify_song_id")
        key = (
            ("spotify", str(spotify_song_id))
            if spotify_song_id
            else ("local", str(record.get(id_key)))
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
        if len(deduped) >= limit:
            break
    return deduped


def candidate_filter_sql(
    spotify_mode: str,
    *,
    exclude_same_artist: bool,
    cross_genre: bool,
) -> str:
    conditions: list[str] = []
    if spotify_mode == "only":
        conditions.append("m.spotify_song_id IS NOT NULL AND m.spotify_song_id <> ''")
    if exclude_same_artist:
        conditions.append(
            """
            (qm.artist_id IS NULL OR m.artist_id IS NULL OR qm.artist_id <> m.artist_id)
            AND (qm.spotify_artist_id IS NULL OR m.spotify_artist_id IS NULL
                 OR qm.spotify_artist_id <> m.spotify_artist_id)
            """
        )
    if cross_genre:
        conditions.append(
            "qm.main_genre IS NULL OR m.main_genre IS NULL OR qm.main_genre <> m.main_genre"
        )
    return "WHERE " + " AND ".join(f"({condition})" for condition in conditions) if conditions else ""


def norm_column(weight_column: str) -> str:
    if weight_column == "tfidf_log_count":
        return "norm_sq_tfidf_log_count"
    if weight_column == "tfidf_frequency":
        return "norm_sq_tfidf_frequency"
    raise ApiError(400, "weight_column must be tfidf_log_count or tfidf_frequency")


def register_weights(con: duckdb.DuckDBPyConnection, n_weights: dict[int, float]) -> None:
    alpha = pd.DataFrame(
        [{"n": int(n), "alpha": float(alpha)} for n, alpha in n_weights.items()]
    )
    con.register("selected_n_weights", alpha)


def api_stats() -> dict:
    con = connect()
    try:
        table_counts = rows_to_records(
            con.execute(
                """
                SELECT 'song_harmonic_terms' AS table_name, COUNT(*) AS rows FROM song_harmonic_terms
                UNION ALL SELECT 'song_harmonic_tfidf', COUNT(*) FROM song_harmonic_tfidf
                UNION ALL SELECT 'harmonic_song_document_frequency', COUNT(*) FROM harmonic_song_document_frequency
                UNION ALL SELECT 'song_harmonic_norm_components', COUNT(*) FROM song_harmonic_norm_components
                """
            ).fetchdf()
        )
        coverage = rows_to_records(
            con.execute(
                """
                SELECT
                    n,
                    SUM(indexed_windows)::DOUBLE / NULLIF(SUM(total_windows), 0) AS weighted_coverage,
                    AVG(indexed_window_coverage) AS avg_song_coverage,
                    MEDIAN(indexed_window_coverage) AS median_song_coverage,
                    AVG(unique_indexed_harmonic_classes) AS avg_unique_classes
                FROM song_harmonic_totals
                WHERE total_windows > 0
                GROUP BY n
                ORDER BY n
                """
            ).fetchdf()
        )
        metadata = {
            row["key"]: row["value"]
            for row in rows_to_records(
                con.execute(
                    "SELECT key, value FROM song_harmonic_similarity_index_metadata"
                ).fetchdf()
            )
        }
    finally:
        con.close()
    return {"table_counts": table_counts, "coverage": coverage, "metadata": metadata}


def api_health() -> dict:
    return {
        "status": "ok" if DB_PATH.exists() else "missing_database",
        "database_path": str(DB_PATH),
        "database_exists": DB_PATH.exists(),
        "database_size_bytes": DB_PATH.stat().st_size if DB_PATH.exists() else None,
        "spotify_cache_path": str(SPOTIFY_CACHE_PATH),
        "static_dir": str(STATIC_DIR),
    }


def api_search(params: dict[str, list[str]]) -> dict:
    q = params.get("q", [""])[0].strip()
    limit = bounded_int(params, "limit", 20, minimum=1, maximum=50)
    sql_limit = min(limit * 6, 300)
    spotify_mode = parse_spotify_mode(params)
    cached_spotify_ids = lookup_cached_spotify_ids(q, limit * 4) if q else []
    cached_spotify_artist_ids = lookup_cached_spotify_artist_ids(q, limit * 4) if q else []
    parsed_spotify_id = parse_spotify_track_id(q) if q else None
    if parsed_spotify_id and parsed_spotify_id not in cached_spotify_ids:
        cached_spotify_ids.insert(0, parsed_spotify_id)
    con = connect()
    try:
        if q:
            like = f"%{q.lower()}%"
            exact_id = int(q) if q.isdigit() else -1
            spotify_filter = cached_spotify_ids or ["__no_cached_spotify_matches__"]
            spotify_artist_filter = cached_spotify_artist_ids or [
                "__no_cached_spotify_artist_matches__"
            ]
            spotify_required = (
                "AND m.spotify_song_id IS NOT NULL AND m.spotify_song_id <> ''"
                if spotify_mode == "only"
                else ""
            )
            spotify_order = (
                "CASE WHEN m.spotify_song_id IS NOT NULL AND m.spotify_song_id <> '' THEN 0 ELSE 1 END,"
                if spotify_mode == "prefer"
                else ""
            )
            con.register(
                "spotify_search_ids",
                pd.DataFrame({"spotify_song_id": spotify_filter}),
            )
            con.register(
                "spotify_artist_search_ids",
                pd.DataFrame({"spotify_artist_id": spotify_artist_filter}),
            )
            df = con.execute(
                f"""
                SELECT
                    m.id AS song_id,
                    m.release_year,
                    m.decade,
                    m.main_genre,
                    m.artist_id,
                    m.spotify_artist_id,
                    m.spotify_song_id,
                    SUM(t.indexed_windows) AS indexed_windows,
                    SUM(t.unique_indexed_harmonic_classes) AS indexed_features
                FROM song_metadata m
                JOIN song_harmonic_totals t ON t.song_id = m.id
                WHERE (
                       m.id = ?
                   OR LOWER(COALESCE(m.spotify_song_id, '')) LIKE ?
                   OR LOWER(COALESCE(m.spotify_artist_id, '')) LIKE ?
                   OR LOWER(COALESCE(m.artist_id, '')) LIKE ?
                   OR LOWER(COALESCE(m.main_genre, '')) LIKE ?
                   OR m.spotify_song_id IN (SELECT * FROM spotify_search_ids)
                   OR m.spotify_artist_id IN (SELECT * FROM spotify_artist_search_ids)
                )
                {spotify_required}
                GROUP BY m.id, m.release_year, m.decade, m.main_genre,
                         m.artist_id, m.spotify_artist_id, m.spotify_song_id
                HAVING SUM(t.unique_indexed_harmonic_classes) > 0
                ORDER BY
                    CASE WHEN m.id = ? THEN 0 ELSE 1 END,
                    CASE WHEN m.spotify_artist_id IN (SELECT * FROM spotify_artist_search_ids) THEN 0 ELSE 1 END,
                    CASE WHEN m.spotify_song_id IN (SELECT * FROM spotify_search_ids) THEN 0 ELSE 1 END,
                    {spotify_order}
                    indexed_features DESC,
                    song_id
                LIMIT ?
                """,
                [exact_id, like, like, like, like, exact_id, sql_limit],
            ).fetchdf()
            con.unregister("spotify_search_ids")
            con.unregister("spotify_artist_search_ids")
        else:
            spotify_required = (
                "WHERE m.spotify_song_id IS NOT NULL AND m.spotify_song_id <> ''"
                if spotify_mode == "only"
                else ""
            )
            spotify_order = (
                "CASE WHEN m.spotify_song_id IS NOT NULL AND m.spotify_song_id <> '' THEN 0 ELSE 1 END,"
                if spotify_mode == "prefer"
                else ""
            )
            df = con.execute(
                f"""
                SELECT
                    m.id AS song_id,
                    m.release_year,
                    m.decade,
                    m.main_genre,
                    m.artist_id,
                    m.spotify_artist_id,
                    m.spotify_song_id,
                    SUM(t.indexed_windows) AS indexed_windows,
                    SUM(t.unique_indexed_harmonic_classes) AS indexed_features
                FROM song_metadata m
                JOIN song_harmonic_totals t ON t.song_id = m.id
                {spotify_required}
                GROUP BY m.id, m.release_year, m.decade, m.main_genre,
                         m.artist_id, m.spotify_artist_id, m.spotify_song_id
                HAVING SUM(t.unique_indexed_harmonic_classes) > 0
                ORDER BY {spotify_order} indexed_features DESC, song_id
                LIMIT ?
                """,
                [sql_limit],
            ).fetchdf()
    finally:
        con.close()
    records = dedupe_spotify_records(
        add_cached_spotify_fields(rows_to_records(df)),
        id_key="song_id",
        limit=limit,
    )
    return {
        "results": records,
        "spotify_cache_matches": len(cached_spotify_ids),
        "spotify_artist_cache_matches": len(cached_spotify_artist_ids),
    }


def api_song(song_id: int) -> dict:
    con = connect()
    try:
        song = con.execute(
            """
            SELECT
                m.id AS song_id,
                m.release_year,
                m.decade,
                m.main_genre,
                m.rock_genre,
                m.genres,
                m.artist_id,
                m.spotify_song_id,
                m.spotify_artist_id
            FROM song_metadata m
            WHERE m.id = ?
            """,
            [song_id],
        ).fetchdf()
        if song.empty:
            raise ApiError(404, f"Song {song_id} not found")

        totals = con.execute(
            """
            SELECT
                n,
                total_windows,
                indexed_windows,
                indexed_window_coverage,
                unique_indexed_harmonic_classes
            FROM song_harmonic_totals
            WHERE song_id = ?
            ORDER BY n
            """,
            [song_id],
        ).fetchdf()
        excerpt = chord_excerpt(con, song_id)
    finally:
        con.close()
    song_record = rows_to_records(song)[0]
    song_record["spotify_url"] = spotify_url(song_record.get("spotify_song_id"))
    song_record["spotify_embed_url"] = spotify_embed_url(song_record.get("spotify_song_id"))
    song_record["chord_excerpt"] = excerpt
    add_cached_spotify_fields([song_record])
    return {"song": song_record, "totals": rows_to_records(totals)}


def api_spotify(params: dict[str, list[str]]) -> dict:
    spotify_song_id = params.get("spotify_song_id", [""])[0].strip()
    if not spotify_song_id:
        raise ApiError(400, "Missing spotify_song_id")
    cached = cached_spotify_metadata(spotify_song_id)
    if cached and (cached.get("title") or cached.get("artist")):
        SPOTIFY_OEMBED_CACHE[spotify_song_id] = cached
        return cached
    if spotify_song_id in SPOTIFY_OEMBED_CACHE:
        return SPOTIFY_OEMBED_CACHE[spotify_song_id]

    track_url = spotify_url(spotify_song_id)
    assert track_url is not None
    oembed_url = "https://open.spotify.com/oembed?" + urllib.parse.urlencode({"url": track_url})
    page_error = None
    try:
        oembed_payload = fetch_json(oembed_url)
    except Exception as exc:
        payload = {
            "spotify_song_id": spotify_song_id,
            "spotify_url": track_url,
            "spotify_embed_url": spotify_embed_url(spotify_song_id),
            "title": None,
            "artist": None,
            "author_name": None,
            "thumbnail_url": None,
            "error": str(exc),
        }
    else:
        payload = {
            "spotify_song_id": spotify_song_id,
            "spotify_url": track_url,
            "spotify_embed_url": spotify_embed_url(spotify_song_id),
            "title": oembed_payload.get("title"),
            "artist": oembed_payload.get("author_name"),
            "author_name": oembed_payload.get("author_name"),
            "thumbnail_url": oembed_payload.get("thumbnail_url"),
        }
    if not payload.get("artist") or not payload.get("title"):
        try:
            page_payload = spotify_page_metadata(spotify_song_id)
        except Exception as exc:
            page_error = str(exc)
        else:
            payload["title"] = payload.get("title") or page_payload.get("title")
            payload["artist"] = payload.get("artist") or page_payload.get("artist")
            payload["author_name"] = payload.get("author_name") or page_payload.get("artist")
            payload["description"] = page_payload.get("description")
            payload["preview_url"] = page_payload.get("preview_url")
    if page_error:
        payload["page_error"] = page_error
    cache_spotify_metadata(payload)
    SPOTIFY_OEMBED_CACHE[spotify_song_id] = payload
    return payload


def api_similar(params: dict[str, list[str]]) -> dict:
    song_id = parse_int(params.get("song_id", [None])[0])
    selected_ns = parse_ns(params)
    weights = parse_weights(params, selected_ns)
    top_k = bounded_int(params, "top_k", 20, minimum=1, maximum=100)
    sql_limit = min(top_k * 6, 600)
    min_shared = bounded_int(params, "min_shared", 4, minimum=1, maximum=500)
    spotify_mode = parse_spotify_mode(params)
    exclude_same_artist = parse_bool(params, "exclude_same_artist")
    cross_genre = parse_bool(params, "cross_genre")
    weight_column = params.get("weight_column", [DEFAULT_WEIGHT_COLUMN])[0]
    norm_col = norm_column(weight_column)
    candidate_where = candidate_filter_sql(
        spotify_mode,
        exclude_same_artist=exclude_same_artist,
        cross_genre=cross_genre,
    )
    spotify_order = (
        "CASE WHEN m.spotify_song_id IS NOT NULL AND m.spotify_song_id <> '' THEN 0 ELSE 1 END,"
        if spotify_mode == "prefer"
        else ""
    )

    con = connect()
    register_weights(con, weights)
    try:
        df = con.execute(
            f"""
            WITH q AS (
                SELECT
                    w.song_id,
                    w.n,
                    w.harmonic_id,
                    w.{weight_column} * a.alpha AS weight
                FROM song_harmonic_tfidf w
                JOIN selected_n_weights a USING (n)
                WHERE w.song_id = ?
            ), candidate_dots AS (
                SELECT
                    c.song_id AS candidate_song_id,
                    SUM(q.weight * c.{weight_column} * a.alpha) AS dot_product,
                    COUNT(*) AS shared_features
                FROM q
                JOIN song_harmonic_tfidf c USING (n, harmonic_id)
                JOIN selected_n_weights a ON a.n = c.n
                WHERE c.song_id <> ?
                GROUP BY c.song_id
                HAVING COUNT(*) >= ?
            ), norms AS (
                SELECT
                    nc.song_id,
                    SQRT(SUM(nc.{norm_col} * a.alpha * a.alpha)) AS norm
                FROM song_harmonic_norm_components nc
                JOIN selected_n_weights a USING (n)
                GROUP BY nc.song_id
            ), q_norm AS (
                SELECT norm FROM norms WHERE song_id = ?
            ), q_meta AS (
                SELECT artist_id, spotify_artist_id, main_genre
                FROM song_metadata
                WHERE id = ?
            )
            SELECT
                d.candidate_song_id,
                d.dot_product / NULLIF(q_norm.norm * norms.norm, 0) AS similarity_score,
                d.shared_features,
                m.release_year,
                m.decade,
                m.main_genre,
                m.artist_id,
                m.spotify_artist_id,
                m.spotify_song_id
            FROM candidate_dots d
            JOIN norms ON norms.song_id = d.candidate_song_id
            CROSS JOIN q_norm
            CROSS JOIN q_meta qm
            LEFT JOIN song_metadata m ON m.id = d.candidate_song_id
            {candidate_where}
            ORDER BY {spotify_order} similarity_score DESC, shared_features DESC, candidate_song_id
            LIMIT ?
            """,
            [song_id, song_id, min_shared, song_id, song_id, sql_limit],
        ).fetchdf()
    finally:
        con.unregister("selected_n_weights")
        con.close()
    records = dedupe_spotify_records(
        add_cached_spotify_fields(rows_to_records(df)),
        id_key="candidate_song_id",
        limit=top_k,
    )
    return {
        "song_id": song_id,
        "ns": selected_ns,
        "weights": weights,
        "results": records,
    }


def similar_for_ns(
    con: duckdb.DuckDBPyConnection,
    song_id: int,
    selected_ns: tuple[int, ...],
    weights: dict[int, float],
    top_k: int,
    min_shared: int,
    weight_column: str = DEFAULT_WEIGHT_COLUMN,
) -> pd.DataFrame:
    norm_col = norm_column(weight_column)
    register_weights(con, weights)
    try:
        return con.execute(
            f"""
            WITH q AS (
                SELECT
                    w.song_id,
                    w.n,
                    w.harmonic_id,
                    w.{weight_column} * a.alpha AS weight
                FROM song_harmonic_tfidf w
                JOIN selected_n_weights a USING (n)
                WHERE w.song_id = ?
            ), candidate_dots AS (
                SELECT
                    c.song_id AS candidate_song_id,
                    SUM(q.weight * c.{weight_column} * a.alpha) AS dot_product,
                    COUNT(*) AS shared_features
                FROM q
                JOIN song_harmonic_tfidf c USING (n, harmonic_id)
                JOIN selected_n_weights a ON a.n = c.n
                WHERE c.song_id <> ?
                GROUP BY c.song_id
                HAVING COUNT(*) >= ?
            ), norms AS (
                SELECT
                    nc.song_id,
                    SQRT(SUM(nc.{norm_col} * a.alpha * a.alpha)) AS norm
                FROM song_harmonic_norm_components nc
                JOIN selected_n_weights a USING (n)
                GROUP BY nc.song_id
            ), q_norm AS (
                SELECT norm FROM norms WHERE song_id = ?
            )
            SELECT
                d.candidate_song_id,
                d.dot_product / NULLIF(q_norm.norm * norms.norm, 0) AS similarity_score,
                d.shared_features,
                m.release_year,
                m.decade,
                m.main_genre,
                m.artist_id,
                m.spotify_song_id
            FROM candidate_dots d
            JOIN norms ON norms.song_id = d.candidate_song_id
            CROSS JOIN q_norm
            LEFT JOIN song_metadata m ON m.id = d.candidate_song_id
            ORDER BY similarity_score DESC, shared_features DESC, candidate_song_id
            LIMIT ?
            """,
            [song_id, song_id, min_shared, song_id, top_k],
        ).fetchdf()
    finally:
        con.unregister("selected_n_weights")


def api_compare_n(params: dict[str, list[str]]) -> dict:
    song_id = parse_int(params.get("song_id", [None])[0])
    top_k = bounded_int(params, "top_k", 5, minimum=1, maximum=20)
    sql_limit = min(top_k * 6, 120)
    min_shared = bounded_int(params, "min_shared", 2, minimum=1, maximum=500)
    spotify_mode = parse_spotify_mode(params)
    exclude_same_artist = parse_bool(params, "exclude_same_artist")
    cross_genre = parse_bool(params, "cross_genre")
    candidate_where = candidate_filter_sql(
        spotify_mode,
        exclude_same_artist=exclude_same_artist,
        cross_genre=cross_genre,
    )
    spotify_order = (
        "CASE WHEN m.spotify_song_id IS NOT NULL AND m.spotify_song_id <> '' THEN 0 ELSE 1 END,"
        if spotify_mode == "prefer"
        else ""
    )

    con = connect()
    try:
        df = con.execute(
            f"""
            WITH q AS (
                SELECT song_id, n, harmonic_id, tfidf_log_count AS weight
                FROM song_harmonic_tfidf
                WHERE song_id = ?
            ), q_meta AS (
                SELECT artist_id, spotify_artist_id, main_genre
                FROM song_metadata
                WHERE id = ?
            ), candidate_dots AS (
                SELECT
                    q.n,
                    c.song_id AS candidate_song_id,
                    SUM(q.weight * c.tfidf_log_count) AS dot_product,
                    COUNT(*) AS shared_features
                FROM q
                JOIN song_harmonic_tfidf c USING (n, harmonic_id)
                WHERE c.song_id <> ?
                GROUP BY q.n, c.song_id
                HAVING COUNT(*) >= ?
            ), scored AS (
                SELECT
                    d.n,
                    d.candidate_song_id,
                    d.dot_product / NULLIF(
                        SQRT(qn.norm_sq_tfidf_log_count) * SQRT(cn.norm_sq_tfidf_log_count),
                        0
                    ) AS similarity_score,
                    d.shared_features,
                    m.release_year,
                    m.decade,
                    m.main_genre,
                    m.artist_id,
                    m.spotify_artist_id,
                    m.spotify_song_id
                FROM candidate_dots d
                JOIN song_harmonic_norm_components qn
                    ON qn.song_id = ? AND qn.n = d.n
                JOIN song_harmonic_norm_components cn
                    ON cn.song_id = d.candidate_song_id AND cn.n = d.n
                CROSS JOIN q_meta qm
                LEFT JOIN song_metadata m ON m.id = d.candidate_song_id
                {candidate_where}
            )
            SELECT *
            FROM scored
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY n
                ORDER BY {spotify_order} similarity_score DESC, shared_features DESC, candidate_song_id
            ) <= ?
            ORDER BY n, {spotify_order} similarity_score DESC
            """,
            [song_id, song_id, song_id, min_shared, song_id, sql_limit],
        ).fetchdf()
    finally:
        con.close()

    comparisons = {}
    for n in NS:
        records = add_cached_spotify_fields(rows_to_records(df[df["n"].eq(n)].drop(columns=["n"])))
        comparisons[str(n)] = dedupe_spotify_records(
            records,
            id_key="candidate_song_id",
            limit=top_k,
        )
    return {"song_id": song_id, "top_k": top_k, "results_by_n": comparisons}


def api_explain(params: dict[str, list[str]]) -> dict:
    song_id = parse_int(params.get("song_id", [None])[0])
    candidate_id = parse_int(params.get("candidate_id", [None])[0])
    selected_ns = parse_ns(params)
    weights = parse_weights(params, selected_ns)
    top_features = bounded_int(params, "top_features", 40, minimum=1, maximum=200)
    weight_column = params.get("weight_column", [DEFAULT_WEIGHT_COLUMN])[0]
    norm_col = norm_column(weight_column)

    con = connect()
    register_weights(con, weights)
    try:
        by_n = con.execute(
            f"""
            WITH norms AS (
                SELECT
                    nc.song_id,
                    SQRT(SUM(nc.{norm_col} * a.alpha * a.alpha)) AS norm
                FROM song_harmonic_norm_components nc
                JOIN selected_n_weights a USING (n)
                WHERE nc.song_id IN (?, ?)
                GROUP BY nc.song_id
            ), shared AS (
                SELECT
                    q.n,
                    q.{weight_column} * c.{weight_column} * a.alpha * a.alpha AS raw_contribution
                FROM song_harmonic_tfidf q
                JOIN song_harmonic_tfidf c USING (n, harmonic_id)
                JOIN selected_n_weights a USING (n)
                WHERE q.song_id = ?
                  AND c.song_id = ?
            ), denom AS (
                SELECT q.norm * c.norm AS norm_product
                FROM norms q
                JOIN norms c ON c.song_id = ?
                WHERE q.song_id = ?
            )
            SELECT
                n,
                COUNT(*) AS shared_features,
                SUM(raw_contribution) AS raw_contribution,
                SUM(raw_contribution) / NULLIF(MAX(norm_product), 0) AS cosine_contribution
            FROM shared
            CROSS JOIN denom
            GROUP BY n
            ORDER BY n
            """,
            [song_id, candidate_id, song_id, candidate_id, candidate_id, song_id],
        ).fetchdf()

        features = con.execute(
            f"""
            WITH norms AS (
                SELECT
                    nc.song_id,
                    SQRT(SUM(nc.{norm_col} * a.alpha * a.alpha)) AS norm
                FROM song_harmonic_norm_components nc
                JOIN selected_n_weights a USING (n)
                WHERE nc.song_id IN (?, ?)
                GROUP BY nc.song_id
            ), denom AS (
                SELECT q.norm * c.norm AS norm_product
                FROM norms q
                JOIN norms c ON c.song_id = ?
                WHERE q.song_id = ?
            )
            SELECT
                q.n,
                q.harmonic_id,
                df.example_ngram,
                df.song_df,
                df.song_df_rate,
                df.idf,
                q.count AS query_count,
                c.count AS candidate_count,
                q.{weight_column} * a.alpha AS query_weight,
                c.{weight_column} * a.alpha AS candidate_weight,
                q.{weight_column} * c.{weight_column} * a.alpha * a.alpha / NULLIF(denom.norm_product, 0)
                    AS cosine_contribution
            FROM song_harmonic_tfidf q
            JOIN song_harmonic_tfidf c USING (n, harmonic_id)
            JOIN selected_n_weights a USING (n)
            JOIN harmonic_song_document_frequency df USING (n, harmonic_id)
            CROSS JOIN denom
            WHERE q.song_id = ?
              AND c.song_id = ?
            ORDER BY cosine_contribution DESC, n DESC, harmonic_id
            LIMIT ?
            """,
            [song_id, candidate_id, candidate_id, song_id, song_id, candidate_id, top_features],
        ).fetchdf()

        unique_query = con.execute(
            f"""
            SELECT
                q.n,
                q.harmonic_id,
                df.example_ngram,
                df.song_df,
                df.idf,
                q.count,
                q.{weight_column} * a.alpha AS feature_weight
            FROM song_harmonic_tfidf q
            JOIN selected_n_weights a USING (n)
            JOIN harmonic_song_document_frequency df USING (n, harmonic_id)
            ANTI JOIN song_harmonic_tfidf c
                ON c.song_id = ? AND c.n = q.n AND c.harmonic_id = q.harmonic_id
            WHERE q.song_id = ?
            ORDER BY feature_weight DESC
            LIMIT 20
            """,
            [candidate_id, song_id],
        ).fetchdf()
    finally:
        con.unregister("selected_n_weights")
        con.close()

    return {
        "song_id": song_id,
        "candidate_id": candidate_id,
        "by_n": rows_to_records(by_n),
        "features": rows_to_records(features),
        "unique_query": rows_to_records(unique_query),
    }


def handle_api(path: str, params: dict[str, list[str]]) -> dict:
    if path == "/api/health":
        return api_health()
    if path == "/api/stats":
        return api_stats()
    if path == "/api/search":
        return api_search(params)
    if path == "/api/song":
        return api_song(parse_int(params.get("song_id", [None])[0]))
    if path == "/api/spotify":
        return api_spotify(params)
    if path == "/api/similar":
        return api_similar(params)
    if path == "/api/compare-n":
        return api_compare_n(params)
    if path == "/api/explain":
        return api_explain(params)
    raise ApiError(404, "Unknown API route")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.send_json_response(parsed.path, parse_qs(parsed.query))
            return
        self.send_static(parsed.path)

    def send_json_response(self, path: str, params: dict[str, list[str]]) -> None:
        try:
            payload = handle_api(path, params)
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
        except ApiError as exc:
            body = json.dumps({"error": exc.message}).encode("utf-8")
            self.send_response(exc.status)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(500)

        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, request_path: str) -> None:
        path = request_path.strip("/")
        if not path:
            path = "index.html"
        file_path = (STATIC_DIR / path).resolve()
        try:
            file_path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self.send_error(404)
            return
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404)
            return

        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")


def main() -> None:
    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), Handler)
    display_host = "127.0.0.1" if SERVER_HOST in {"0.0.0.0", "::"} else SERVER_HOST
    print(f"Harmonic explorer running at http://{display_host}:{SERVER_PORT}")
    print(f"DuckDB path: {DB_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
