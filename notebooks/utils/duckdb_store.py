from pathlib import Path

import duckdb
import pandas as pd


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def configure_connection(con: duckdb.DuckDBPyConnection) -> None:
    """
    Keep DuckDB output notebook-friendly.

    DuckDB's native progress bar can render as black boxes in Jupyter, so this
    helper disables it for long CSV import/grouping steps.
    """
    con.execute("PRAGMA threads=4")
    try:
        con.execute("PRAGMA disable_progress_bar")
    except duckdb.Error:
        con.execute("SET enable_progress_bar = false")


def create_harmonic_trends_database_from_ngrams(
    *,
    db_path,
    ngram_dir,
    overwrite: bool = True,
) -> Path:
    """
    Create a DuckDB database directly from notebook 2 n-gram outputs.

    This is the full-corpus path: it avoids loading tens of millions of exact
    n-gram rows into Pandas.
    """
    db_path = Path(db_path)
    ngram_dir = Path(ngram_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and db_path.exists():
        db_path.unlink()

    exact_path = _first_existing(
        [
            ngram_dir / "exact_ngram_vocab.csv.gz",
            ngram_dir / "exact_ngram_vocab.csv",
        ]
    )
    if exact_path is None:
        raise FileNotFoundError("Missing exact_ngram_vocab.csv.gz or exact_ngram_vocab.csv")

    song_summary_path = _first_existing(
        [
            ngram_dir / "song_ngram_summary.csv.gz",
            ngram_dir / "song_ngram_summary.csv",
        ]
    )
    run_summary_path = _first_existing([ngram_dir / "ngram_run_summary.csv"])
    parse_errors_path = _first_existing([ngram_dir / "harmonic_parse_errors.csv"])

    con = duckdb.connect(str(db_path))
    try:
        configure_connection(con)

        con.execute(
            """
            CREATE TABLE exact_ngrams AS
            SELECT *
            FROM read_csv_auto(?, header=true)
            """,
            [str(exact_path)],
        )

        exact_columns = {
            row[1] for row in con.execute("PRAGMA table_info('exact_ngrams')").fetchall()
        }
        required = {
            "exact_ngram_id",
            "n",
            "ngram_json",
            "ngram",
            "count",
            "frequency",
            "harmonic_key",
            "harmonic_id",
        }
        missing = sorted(required - exact_columns)
        if missing:
            raise ValueError(
                "exact_ngrams is missing required columns. "
                f"Rerun notebook 2 with the current schema. Missing: {missing}"
            )

        con.execute(
            """
            CREATE TABLE harmonic_ngrams AS
            SELECT
                harmonic_id,
                n,
                harmonic_key,
                SUM(count)::BIGINT AS count,
                COUNT(DISTINCT exact_ngram_id)::BIGINT AS n_exact_ngrams,
                FIRST(ngram_json ORDER BY count DESC, exact_ngram_id) AS example_ngram_json,
                FIRST(ngram ORDER BY count DESC, exact_ngram_id) AS example_ngram
            FROM exact_ngrams
            WHERE harmonic_id IS NOT NULL
            GROUP BY harmonic_id, n, harmonic_key
            """
        )

        con.execute(
            """
            CREATE OR REPLACE TABLE harmonic_ngrams AS
            SELECT
                *,
                count::DOUBLE / SUM(count) OVER (PARTITION BY n) AS frequency
            FROM harmonic_ngrams
            """
        )

        con.execute(
            """
            CREATE TABLE exact_to_harmonic AS
            SELECT exact_ngram_id, n, ngram_json, ngram, harmonic_id, harmonic_key, count
            FROM exact_ngrams
            """
        )

        con.execute(
            """
            CREATE TABLE frequency_validation AS
            SELECT
                v.n,
                v.V_count,
                h.H_count,
                v.V_frequency_sum,
                h.H_frequency_sum,
                v.V_count = h.H_count AS counts_match
            FROM (
                SELECT n, SUM(count)::BIGINT AS V_count, SUM(frequency) AS V_frequency_sum
                FROM exact_ngrams
                GROUP BY n
            ) v
            JOIN (
                SELECT n, SUM(count)::BIGINT AS H_count, SUM(frequency) AS H_frequency_sum
                FROM harmonic_ngrams
                GROUP BY n
            ) h USING (n)
            ORDER BY n
            """
        )

        if song_summary_path is not None:
            con.execute(
                """
                CREATE TABLE song_ngram_summary AS
                SELECT *
                FROM read_csv_auto(?, header=true)
                """,
                [str(song_summary_path)],
            )

        if run_summary_path is not None:
            con.execute(
                """
                CREATE TABLE ngram_run_summary AS
                SELECT *
                FROM read_csv_auto(?, header=true)
                """,
                [str(run_summary_path)],
            )

        if parse_errors_path is not None:
            con.execute(
                """
                CREATE TABLE harmonic_parse_errors AS
                SELECT *
                FROM read_csv_auto(?, header=true)
                """,
                [str(parse_errors_path)],
            )

        con.execute(
            """
            CREATE TABLE ngram_distances (
                source_id VARCHAR,
                target_id VARCHAR,
                n INTEGER,
                vocab VARCHAR,
                metric VARCHAR,
                distance DOUBLE
            )
            """
        )

        con.execute(
            """
            CREATE TABLE metadata (
                key VARCHAR,
                value VARCHAR
            )
            """
        )
        con.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [
                ("ngram_dir", str(ngram_dir)),
                ("exact_source", str(exact_path)),
                ("id_policy", "Use exact_ngram_id for V_n and harmonic_id for H_n; keep harmonic_key for matrix reconstruction."),
                ("frequency_policy", "V_n comes from notebook 2 exact counts; H_n is computed as the SQL pushforward grouped by harmonic_id."),
                ("distance_policy", "ngram_distances is reserved for future Euclidean/Levenshtein comparisons."),
            ],
        )

        con.execute("CREATE INDEX exact_ngrams_id_idx ON exact_ngrams(exact_ngram_id)")
        con.execute("CREATE INDEX exact_ngrams_n_idx ON exact_ngrams(n)")
        con.execute("CREATE INDEX exact_ngrams_harmonic_id_idx ON exact_ngrams(harmonic_id)")
        con.execute("CREATE INDEX harmonic_ngrams_id_idx ON harmonic_ngrams(harmonic_id)")
        con.execute("CREATE INDEX harmonic_ngrams_n_idx ON harmonic_ngrams(n)")
        con.execute("CREATE INDEX exact_to_harmonic_exact_idx ON exact_to_harmonic(exact_ngram_id)")
        con.execute("CREATE INDEX exact_to_harmonic_harmonic_idx ON exact_to_harmonic(harmonic_id)")
        con.execute("CREATE INDEX ngram_distances_source_idx ON ngram_distances(source_id)")
        con.execute("CREATE INDEX ngram_distances_target_idx ON ngram_distances(target_id)")
        con.execute("CREATE INDEX ngram_distances_metric_idx ON ngram_distances(vocab, metric, n)")
    finally:
        con.close()

    return db_path


def create_harmonic_trends_database_from_dataframes(
    *,
    db_path,
    exact_vocab: pd.DataFrame,
    song_summary: pd.DataFrame | None = None,
    run_summary: pd.DataFrame | None = None,
    parse_errors: pd.DataFrame | None = None,
    overwrite: bool = True,
    metadata: dict[str, str] | None = None,
    verbose: bool = True,
) -> Path:
    """
    Create the analysis DuckDB database from in-memory notebook 2 outputs.

    This is the preferred current path. Notebook 2 already has the global
    counters in memory after streaming the raw songs, so writing directly to
    DuckDB avoids the expensive CSV.gz handoff that older versions used.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if overwrite and db_path.exists():
        if verbose:
            print(f"Removing existing DuckDB database: {db_path}")
        db_path.unlink()

    required = {
        "exact_ngram_id",
        "n",
        "ngram_json",
        "ngram",
        "count",
        "frequency",
        "harmonic_key",
        "harmonic_id",
    }
    missing = sorted(required - set(exact_vocab.columns))
    if missing:
        raise ValueError(f"exact_vocab is missing required columns: {missing}")

    con = duckdb.connect(str(db_path))
    try:
        configure_connection(con)

        if verbose:
            print(f"Creating DuckDB database: {db_path}")
            print(f"Registering exact vocabulary rows: {len(exact_vocab):,}")
        con.register("exact_vocab_df", exact_vocab)
        con.execute("CREATE TABLE exact_ngrams AS SELECT * FROM exact_vocab_df")

        if verbose:
            print("Building harmonic_ngrams from exact_ngrams...")
        con.execute(
            """
            CREATE TABLE harmonic_ngrams AS
            SELECT
                harmonic_id,
                n,
                harmonic_key,
                SUM(count)::BIGINT AS count,
                COUNT(DISTINCT exact_ngram_id)::BIGINT AS n_exact_ngrams,
                FIRST(ngram_json ORDER BY count DESC, exact_ngram_id) AS example_ngram_json,
                FIRST(ngram ORDER BY count DESC, exact_ngram_id) AS example_ngram
            FROM exact_ngrams
            WHERE harmonic_id IS NOT NULL
            GROUP BY harmonic_id, n, harmonic_key
            """
        )

        if verbose:
            print("Normalizing harmonic frequencies...")
        con.execute(
            """
            CREATE OR REPLACE TABLE harmonic_ngrams AS
            SELECT
                *,
                count::DOUBLE / SUM(count) OVER (PARTITION BY n) AS frequency
            FROM harmonic_ngrams
            """
        )

        if verbose:
            print("Creating exact_to_harmonic map and frequency validation...")
        con.execute(
            """
            CREATE TABLE exact_to_harmonic AS
            SELECT exact_ngram_id, n, ngram_json, ngram, harmonic_id, harmonic_key, count
            FROM exact_ngrams
            """
        )

        con.execute(
            """
            CREATE TABLE frequency_validation AS
            SELECT
                v.n,
                v.V_count,
                h.H_count,
                v.V_frequency_sum,
                h.H_frequency_sum,
                v.V_count = h.H_count AS counts_match
            FROM (
                SELECT n, SUM(count)::BIGINT AS V_count, SUM(frequency) AS V_frequency_sum
                FROM exact_ngrams
                GROUP BY n
            ) v
            JOIN (
                SELECT n, SUM(count)::BIGINT AS H_count, SUM(frequency) AS H_frequency_sum
                FROM harmonic_ngrams
                GROUP BY n
            ) h USING (n)
            ORDER BY n
            """
        )

        if song_summary is not None:
            if verbose:
                print(f"Writing song_ngram_summary rows: {len(song_summary):,}")
            con.register("song_summary_df", song_summary)
            con.execute("CREATE TABLE song_ngram_summary AS SELECT * FROM song_summary_df")

        if run_summary is not None:
            if verbose:
                print(f"Writing ngram_run_summary rows: {len(run_summary):,}")
            con.register("run_summary_df", run_summary)
            con.execute("CREATE TABLE ngram_run_summary AS SELECT * FROM run_summary_df")

        if parse_errors is not None:
            if verbose:
                print(f"Writing harmonic_parse_errors rows: {len(parse_errors):,}")
            con.register("parse_errors_df", parse_errors)
            con.execute("CREATE TABLE harmonic_parse_errors AS SELECT * FROM parse_errors_df")

        if verbose:
            print("Creating metadata and reserved distance table...")
        con.execute(
            """
            CREATE TABLE ngram_distances (
                source_id VARCHAR,
                target_id VARCHAR,
                n INTEGER,
                vocab VARCHAR,
                metric VARCHAR,
                distance DOUBLE
            )
            """
        )

        con.execute(
            """
            CREATE TABLE metadata (
                key VARCHAR,
                value VARCHAR
            )
            """
        )
        metadata_rows = {
            "source": "notebook_2_direct_duckdb",
            "id_policy": "Use exact_ngram_id for V_n and harmonic_id for H_n; keep harmonic_key for matrix reconstruction.",
            "frequency_policy": "V_n comes from notebook 2 exact counts; H_n is computed as the SQL pushforward grouped by harmonic_id.",
            "distance_policy": "ngram_distances is reserved for future Euclidean/Levenshtein comparisons.",
        }
        if metadata:
            metadata_rows.update(metadata)
        con.executemany("INSERT INTO metadata VALUES (?, ?)", list(metadata_rows.items()))

        if verbose:
            print("Creating indexes. This can take several minutes on the full corpus...")
        con.execute("CREATE INDEX exact_ngrams_id_idx ON exact_ngrams(exact_ngram_id)")
        con.execute("CREATE INDEX exact_ngrams_n_idx ON exact_ngrams(n)")
        con.execute("CREATE INDEX exact_ngrams_harmonic_id_idx ON exact_ngrams(harmonic_id)")
        con.execute("CREATE INDEX harmonic_ngrams_id_idx ON harmonic_ngrams(harmonic_id)")
        con.execute("CREATE INDEX harmonic_ngrams_n_idx ON harmonic_ngrams(n)")
        con.execute("CREATE INDEX exact_to_harmonic_exact_idx ON exact_to_harmonic(exact_ngram_id)")
        con.execute("CREATE INDEX exact_to_harmonic_harmonic_idx ON exact_to_harmonic(harmonic_id)")
        con.execute("CREATE INDEX ngram_distances_source_idx ON ngram_distances(source_id)")
        con.execute("CREATE INDEX ngram_distances_target_idx ON ngram_distances(target_id)")
        con.execute("CREATE INDEX ngram_distances_metric_idx ON ngram_distances(vocab, metric, n)")
        if verbose:
            print("DuckDB global n-gram store complete.")
    finally:
        con.close()

    return db_path


def export_core_tables(
    *,
    db_path,
    frequency_dir,
) -> dict[str, Path]:
    """
    Export compact CSV.gz frequency tables from the DuckDB database.
    """
    frequency_dir = Path(frequency_dir)
    frequency_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "V_table": frequency_dir / "V_frequency_table.csv.gz",
        "H_table": frequency_dir / "H_frequency_table.csv.gz",
        "V_to_H_map": frequency_dir / "V_to_H_map.csv.gz",
        "validation": frequency_dir / "frequency_validation.csv",
    }

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        configure_connection(con)
        con.execute(
            "COPY exact_ngrams TO ? (HEADER, COMPRESSION gzip)",
            [str(paths["V_table"])],
        )
        con.execute(
            "COPY harmonic_ngrams TO ? (HEADER, COMPRESSION gzip)",
            [str(paths["H_table"])],
        )
        con.execute(
            "COPY exact_to_harmonic TO ? (HEADER, COMPRESSION gzip)",
            [str(paths["V_to_H_map"])],
        )
        con.execute(
            "COPY frequency_validation TO ? (HEADER)",
            [str(paths["validation"])],
        )
    finally:
        con.close()

    return paths


def validate_harmonic_trends_database(db_path) -> pd.DataFrame:
    """
    Return validation checks for the DuckDB database.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        configure_connection(con)
        counts = con.execute(
            """
            SELECT
                v.n,
                SUM(v.count) AS exact_count,
                h.harmonic_count
            FROM exact_ngrams v
            JOIN (
                SELECT n, SUM(count) AS harmonic_count
                FROM harmonic_ngrams
                GROUP BY n
            ) h USING (n)
            GROUP BY v.n, h.harmonic_count
            ORDER BY v.n
            """
        ).fetchdf()
        records = []
        for row in counts.itertuples(index=False):
            records.append(
                {
                    "check_name": "exact_count_equals_harmonic_count",
                    "n": row.n,
                    "passed": row.exact_count == row.harmonic_count,
                    "exact_count": row.exact_count,
                    "harmonic_count": row.harmonic_count,
                }
            )

        collisions = con.execute(
            """
            SELECT n, harmonic_id, COUNT(DISTINCT harmonic_key) AS n_keys
            FROM harmonic_ngrams
            GROUP BY n, harmonic_id
            HAVING COUNT(DISTINCT harmonic_key) > 1
            """
        ).fetchdf()
        records.append(
            {
                "check_name": "harmonic_id_collision_free",
                "n": None,
                "passed": collisions.empty,
                "exact_count": None,
                "harmonic_count": None,
            }
        )

        map_mismatch = con.execute(
            """
            SELECT COUNT(*) AS n_bad
            FROM exact_to_harmonic m
            LEFT JOIN exact_ngrams v USING (exact_ngram_id)
            LEFT JOIN harmonic_ngrams h ON m.harmonic_id = h.harmonic_id
            WHERE v.exact_ngram_id IS NULL OR h.harmonic_id IS NULL
            """
        ).fetchone()[0]
        records.append(
            {
                "check_name": "exact_to_harmonic_references_exist",
                "n": None,
                "passed": map_mismatch == 0,
                "exact_count": map_mismatch,
                "harmonic_count": None,
            }
        )

        return pd.DataFrame.from_records(records)
    finally:
        con.close()
