"""Build the smaller DuckDB artifact used by the web app."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "processed" / "harmonic_trends.duckdb"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "harmonic_trends_app.duckdb"


def sql_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an app-only DuckDB from the full harmonic research database."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 4),
        help="DuckDB worker threads.",
    )
    parser.add_argument(
        "--with-indexes",
        action="store_true",
        help="Add ART indexes. Faster for some queries, but much larger and slower to cold-start.",
    )
    return parser.parse_args()


def run_step(con: duckdb.DuckDBPyConnection, label: str, sql: str) -> None:
    start = time.monotonic()
    print(f"[build] {label}...")
    con.execute(sql)
    elapsed = time.monotonic() - start
    print(f"[build] {label} done in {elapsed:.1f}s")


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    tmp_output = output.with_suffix(output.suffix + ".tmp")

    if not source.exists():
        raise SystemExit(f"Missing source database: {source}")
    if output.exists() and not args.replace:
        raise SystemExit(f"Output exists: {output}. Use --replace to rebuild it.")
    if tmp_output.exists():
        tmp_output.unlink()
    if output.exists():
        output.unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(tmp_output))
    try:
        con.execute(f"PRAGMA threads={max(1, args.threads)}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"ATTACH {sql_literal(source)} AS src (READ_ONLY)")

        run_step(
            con,
            "feature map",
            """
            CREATE TABLE harmonic_feature_map AS
            SELECT
                CAST(row_number() OVER (ORDER BY n, harmonic_id) AS INTEGER) AS feature_id,
                CAST(n AS UTINYINT) AS n,
                harmonic_id AS source_harmonic_id
            FROM src.harmonic_song_document_frequency
            """,
        )
        run_step(
            con,
            "document frequencies",
            """
            CREATE TABLE harmonic_song_document_frequency AS
            SELECT
                m.n,
                m.feature_id AS harmonic_id,
                CAST(df.song_df AS INTEGER) AS song_df,
                CAST(df.n_songs AS INTEGER) AS n_songs,
                CAST(df.song_df_rate AS FLOAT) AS song_df_rate,
                CAST(df.idf AS FLOAT) AS idf,
                CAST(df.global_count AS INTEGER) AS global_count,
                CAST(df.global_frequency AS FLOAT) AS global_frequency,
                df.example_ngram,
                df.harmonic_key,
                CAST(df.n_exact_ngrams AS INTEGER) AS n_exact_ngrams
            FROM src.harmonic_song_document_frequency df
            JOIN harmonic_feature_map m
              ON m.n = df.n AND m.source_harmonic_id = df.harmonic_id
            ORDER BY m.n, m.feature_id
            """,
        )
        run_step(
            con,
            "song feature weights",
            """
            CREATE TABLE song_harmonic_tfidf AS
            SELECT
                CAST(t.song_id AS INTEGER) AS song_id,
                CAST(t.n AS UTINYINT) AS n,
                m.feature_id AS harmonic_id,
                CAST(t.count AS INTEGER) AS count,
                CAST(t.tfidf_frequency AS FLOAT) AS tfidf_frequency,
                CAST(t.tfidf_log_count AS FLOAT) AS tfidf_log_count
            FROM src.song_harmonic_tfidf t
            JOIN harmonic_feature_map m
              ON m.n = t.n AND m.source_harmonic_id = t.harmonic_id
            ORDER BY t.n, m.feature_id, t.song_id
            """,
        )
        run_step(
            con,
            "norm components",
            """
            CREATE TABLE song_harmonic_norm_components AS
            SELECT
                CAST(song_id AS INTEGER) AS song_id,
                CAST(n AS UTINYINT) AS n,
                CAST(norm_sq_tfidf_frequency AS FLOAT) AS norm_sq_tfidf_frequency,
                CAST(norm_sq_tfidf_log_count AS FLOAT) AS norm_sq_tfidf_log_count
            FROM src.song_harmonic_norm_components
            ORDER BY n, song_id
            """,
        )
        run_step(
            con,
            "song totals",
            """
            CREATE TABLE song_harmonic_totals AS
            SELECT
                CAST(song_id AS INTEGER) AS song_id,
                CAST(n AS UTINYINT) AS n,
                CAST(total_windows AS INTEGER) AS total_windows,
                CAST(unique_indexed_harmonic_classes AS INTEGER) AS unique_indexed_harmonic_classes,
                CAST(indexed_windows AS INTEGER) AS indexed_windows,
                CAST(indexed_window_coverage AS FLOAT) AS indexed_window_coverage
            FROM src.song_harmonic_totals
            ORDER BY n, song_id
            """,
        )
        run_step(
            con,
            "song metadata",
            """
            CREATE TABLE song_metadata AS
            SELECT
                CAST(id AS INTEGER) AS id,
                release_date,
                CAST(release_year AS INTEGER) AS release_year,
                CAST(decade AS INTEGER) AS decade,
                main_genre,
                rock_genre,
                genres,
                artist_id,
                spotify_song_id,
                spotify_artist_id
            FROM src.song_metadata
            ORDER BY id
            """,
        )
        default_weights = ", ".join(
            f"({n}, {n / 3.0})"
            for n in range(3, 9)
        )
        run_step(
            con,
            "default weighted norms",
            f"""
            CREATE TABLE song_harmonic_norm_default AS
            WITH selected_n_weights(n, alpha) AS (
                VALUES {default_weights}
            )
            SELECT
                CAST(nc.song_id AS INTEGER) AS song_id,
                CAST(SQRT(SUM(nc.norm_sq_tfidf_log_count * a.alpha * a.alpha)) AS FLOAT)
                    AS norm_tfidf_log_count,
                CAST(SQRT(SUM(nc.norm_sq_tfidf_frequency * a.alpha * a.alpha)) AS FLOAT)
                    AS norm_tfidf_frequency
            FROM song_harmonic_norm_components nc
            JOIN selected_n_weights a USING (n)
            GROUP BY nc.song_id
            ORDER BY nc.song_id
            """,
        )
        run_step(
            con,
            "search summary",
            """
            CREATE TABLE song_search_summary AS
            SELECT
                m.id AS song_id,
                m.release_year,
                m.decade,
                m.main_genre,
                m.artist_id,
                m.spotify_artist_id,
                m.spotify_song_id,
                CAST(COALESCE(SUM(t.indexed_windows), 0) AS INTEGER) AS indexed_windows,
                CAST(COALESCE(SUM(t.unique_indexed_harmonic_classes), 0) AS INTEGER)
                    AS indexed_features
            FROM song_metadata m
            LEFT JOIN song_harmonic_totals t ON t.song_id = m.id
            GROUP BY m.id, m.release_year, m.decade, m.main_genre,
                     m.artist_id, m.spotify_artist_id, m.spotify_song_id
            ORDER BY indexed_features DESC, song_id
            """,
        )
        run_step(
            con,
            "index metadata",
            """
            CREATE TABLE song_harmonic_similarity_index_metadata AS
            SELECT key, value
            FROM src.song_harmonic_similarity_index_metadata
            """,
        )
        run_step(
            con,
            "compatibility view",
            """
            CREATE VIEW song_harmonic_terms AS
            SELECT song_id, n, harmonic_id, count
            FROM song_harmonic_tfidf
            """,
        )
        if args.with_indexes:
            run_step(
                con,
                "runtime indexes",
                """
                CREATE INDEX idx_tfidf_song ON song_harmonic_tfidf(song_id);
                CREATE INDEX idx_tfidf_feature ON song_harmonic_tfidf(n, harmonic_id);
                CREATE INDEX idx_norm_song_n ON song_harmonic_norm_components(song_id, n);
                CREATE INDEX idx_search_song ON song_search_summary(song_id);
                """,
            )
        con.execute("CHECKPOINT")
        con.execute("DETACH src")
    finally:
        con.close()

    tmp_output.replace(output)
    size_gb = output.stat().st_size / 1e9
    print(f"[build] wrote {output} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
