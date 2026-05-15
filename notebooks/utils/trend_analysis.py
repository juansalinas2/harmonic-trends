from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import duckdb
import pandas as pd
from tqdm import tqdm

from utils import duckdb_store as ds
from utils import ngram_features as ng


DEFAULT_STRATA = ("year", "decade", "main_genre")


def _clean_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def release_year(value) -> int | None:
    text = _clean_value(value)
    if text is None:
        return None
    match = re.match(r"^(\d{4})", text)
    return int(match.group(1)) if match else None


def decade_value(row: dict) -> int | None:
    explicit = _clean_value(row.get("decade"))
    if explicit is not None:
        try:
            return int(float(explicit))
        except ValueError:
            pass

    year = release_year(row.get("release_date"))
    return (year // 10) * 10 if year is not None else None


def genre_tags(value) -> list[str]:
    text = _clean_value(value)
    if text is None:
        return []
    quoted = re.findall(r"'([^']+)'", text)
    if quoted:
        return quoted
    return [part.strip() for part in text.split(",") if part.strip()]


def row_strata(row: dict, strata: tuple[str, ...] = DEFAULT_STRATA) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    for stratum in strata:
        if stratum == "year":
            year = release_year(row.get("release_date"))
            if year is not None:
                out.append(("year", str(year)))
        elif stratum == "decade":
            decade = decade_value(row)
            if decade is not None:
                out.append(("decade", str(decade)))
        elif stratum == "main_genre":
            value = _clean_value(row.get("main_genre"))
            if value is not None:
                out.append(("main_genre", value))
        elif stratum == "rock_genre":
            value = _clean_value(row.get("rock_genre"))
            if value is not None:
                out.append(("rock_genre", value))
        elif stratum == "genre_tag":
            out.extend(("genre_tag", tag) for tag in genre_tags(row.get("genres")))
        else:
            value = _clean_value(row.get(stratum))
            if value is not None:
                out.append((stratum, value))

    return out


def create_song_metadata_table(
    *,
    db_path,
    raw_path,
    overwrite: bool = True,
) -> Path:
    """
    Add compact song metadata to the analysis DuckDB database.

    This table is intentionally separate from the n-gram vocabulary tables. It
    stores stratification fields only; it does not duplicate chord strings.
    """
    db_path = Path(db_path)
    raw_path = Path(raw_path)

    con = duckdb.connect(str(db_path))
    try:
        ds.configure_connection(con)
        if overwrite:
            con.execute("DROP TABLE IF EXISTS song_metadata")

        con.execute(
            """
            CREATE TABLE song_metadata AS
            SELECT
                id::BIGINT AS id,
                release_date,
                TRY_CAST(SUBSTRING(CAST(release_date AS VARCHAR), 1, 4) AS INTEGER) AS release_year,
                TRY_CAST(decade AS INTEGER) AS decade,
                main_genre,
                rock_genre,
                genres,
                artist_id,
                spotify_song_id,
                spotify_artist_id
            FROM read_csv_auto(?, header=true, columns={
                'id': 'BIGINT',
                'chords': 'VARCHAR',
                'release_date': 'VARCHAR',
                'genres': 'VARCHAR',
                'decade': 'VARCHAR',
                'rock_genre': 'VARCHAR',
                'artist_id': 'VARCHAR',
                'main_genre': 'VARCHAR',
                'spotify_song_id': 'VARCHAR',
                'spotify_artist_id': 'VARCHAR'
            })
            """,
            [str(raw_path)],
        )
        con.execute("CREATE INDEX IF NOT EXISTS song_metadata_id_idx ON song_metadata(id)")
        con.execute("CREATE INDEX IF NOT EXISTS song_metadata_year_idx ON song_metadata(release_year)")
        con.execute("CREATE INDEX IF NOT EXISTS song_metadata_decade_idx ON song_metadata(decade)")
        con.execute("CREATE INDEX IF NOT EXISTS song_metadata_main_genre_idx ON song_metadata(main_genre)")
    finally:
        con.close()

    return db_path


def top_global_targets(
    *,
    db_path,
    ns: tuple[int, ...],
    top_k_per_n: int = 100,
) -> dict[str, dict[int, set[str] | dict[str, str]]]:
    """
    Return top exact IDs, top harmonic IDs, and exact-ID membership maps for
    the selected harmonic classes.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        ds.configure_connection(con)
        exact: dict[int, set[str]] = {}
        harmonic: dict[int, set[str]] = {}
        harmonic_exact_members: dict[int, dict[str, str]] = {}
        for n in ns:
            exact[n] = {
                row[0]
                for row in con.execute(
                    """
                    SELECT exact_ngram_id
                    FROM exact_ngrams
                    WHERE n = ?
                    ORDER BY count DESC, exact_ngram_id
                    LIMIT ?
                    """,
                    [n, top_k_per_n],
                ).fetchall()
            }
            harmonic[n] = {
                row[0]
                for row in con.execute(
                    """
                    SELECT harmonic_id
                    FROM harmonic_ngrams
                    WHERE n = ?
                    ORDER BY count DESC, harmonic_id
                    LIMIT ?
                    """,
                    [n, top_k_per_n],
                ).fetchall()
            }
            harmonic_exact_members[n] = {
                row[0]: row[1]
                for row in con.execute(
                    """
                    SELECT exact_ngram_id, harmonic_id
                    FROM exact_ngrams
                    WHERE n = ?
                      AND harmonic_id IN (
                          SELECT harmonic_id
                          FROM harmonic_ngrams
                          WHERE n = ?
                          ORDER BY count DESC, harmonic_id
                          LIMIT ?
                      )
                    """,
                    [n, n, top_k_per_n],
                ).fetchall()
            }
    finally:
        con.close()

    return {"V": exact, "H": harmonic, "H_members": harmonic_exact_members}


def supported_harmonic_targets(
    *,
    db_path,
    ns: tuple[int, ...],
    min_global_harmonic_count: int = 1000,
) -> dict[str, dict[int, set[str] | dict[str, str]]]:
    """
    Return support-thresholded harmonic IDs and exact-ID membership maps.

    This is broader than ``top_global_targets``. It is intended for
    corpus-linguistic document-term counts, where document frequency and IDF
    need more than the globally most common 100 targets.
    """
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        ds.configure_connection(con)
        harmonic: dict[int, set[str]] = {}
        harmonic_exact_members: dict[int, dict[str, str]] = {}
        for n in ns:
            harmonic[n] = {
                row[0]
                for row in con.execute(
                    """
                    SELECT harmonic_id
                    FROM harmonic_ngrams
                    WHERE n = ?
                      AND count >= ?
                    ORDER BY count DESC, harmonic_id
                    """,
                    [n, min_global_harmonic_count],
                ).fetchall()
            }
            harmonic_exact_members[n] = {
                row[0]: row[1]
                for row in con.execute(
                    """
                    SELECT v.exact_ngram_id, v.harmonic_id
                    FROM exact_ngrams v
                    JOIN harmonic_ngrams h USING (harmonic_id)
                    WHERE v.n = ?
                      AND h.count >= ?
                    """,
                    [n, min_global_harmonic_count],
                ).fetchall()
            }
    finally:
        con.close()

    return {"H": harmonic, "H_members": harmonic_exact_members}


def _counts_frame(counter: Counter, id_col: str) -> pd.DataFrame:
    rows = [
        {
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "n": n,
            id_col: item_id,
            "count": count,
        }
        for (stratum_type, stratum_value, n, item_id), count in counter.items()
    ]
    return pd.DataFrame(rows)


def _totals_frame(counter: Counter) -> pd.DataFrame:
    rows = [
        {
            "stratum_type": stratum_type,
            "stratum_value": stratum_value,
            "n": n,
            "total_windows": count,
        }
        for (stratum_type, stratum_value, n), count in counter.items()
    ]
    return pd.DataFrame(rows)


def build_stratified_target_trends(
    *,
    db_path,
    raw_path,
    ns: tuple[int, ...],
    top_k_per_n: int = 100,
    strata: tuple[str, ...] = DEFAULT_STRATA,
    chunksize: int = 25_000,
    max_rows: int | None = None,
    show_progress: bool = True,
    overwrite: bool = True,
) -> dict[str, int]:
    """
    Count selected global V_n/H_n targets by metadata strata.

    This deliberately stores trends for top global targets rather than every
    exact n-gram crossed with every stratum. That keeps notebook 4 exploratory
    and avoids producing a table larger than the global vocabulary.

    Harmonic target membership is read from DuckDB's exact_ngrams table instead
    of recomputing matrix keys while streaming raw songs.
    """
    db_path = Path(db_path)
    raw_path = Path(raw_path)
    if show_progress:
        tqdm.write("Selecting top global exact and harmonic targets from DuckDB...")
    targets = top_global_targets(db_path=db_path, ns=ns, top_k_per_n=top_k_per_n)

    exact_counts: Counter = Counter()
    harmonic_counts: Counter = Counter()
    totals: Counter = Counter()
    songs: Counter = Counter()

    usecols = ["id", "chords", "release_date", "decade", "main_genre", "rock_genre", "genres"]
    read_kwargs = {
        "usecols": usecols,
        "chunksize": chunksize,
        "low_memory": False,
    }
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows

    total_rows = max_rows
    if total_rows is None:
        try:
            total_rows = sum(1 for _ in open(raw_path, encoding="utf-8")) - 1
        except OSError:
            total_rows = None

    reader = pd.read_csv(raw_path, **read_kwargs)
    chunk_iter = tqdm(
        reader,
        desc="Counting stratified trend targets",
        unit="chunk",
        disable=not show_progress,
    )
    row_bar = tqdm(
        total=total_rows,
        desc="Scanning songs",
        unit="song",
        disable=not show_progress,
    )

    processed_rows = 0
    for chunk in chunk_iter:
        processed_rows += len(chunk)
        if show_progress:
            chunk_iter.set_postfix(
                {
                    "rows": processed_rows,
                    "V_hits": sum(exact_counts.values()),
                    "H_hits": sum(harmonic_counts.values()),
                }
            )

        for row in chunk.itertuples(index=False):
            row_data = row._asdict()
            row_strata_values = row_strata(row_data, strata=strata)
            if not row_strata_values:
                continue

            tokens = ng.chord_tokens(row_data["chords"])
            for stratum_type, stratum_value in row_strata_values:
                songs[(stratum_type, stratum_value)] += 1

            for n in ns:
                ngrams = ng.chord_ngrams(tokens, n)
                if not ngrams:
                    continue

                for stratum_type, stratum_value in row_strata_values:
                    totals[(stratum_type, stratum_value, n)] += len(ngrams)

                exact_targets = targets["V"][n]
                harmonic_targets = targets["H"][n]
                harmonic_exact_members = targets["H_members"][n]

                for ngram in ngrams:
                    exact_id = ng.exact_ngram_id(ngram)

                    if exact_id in exact_targets:
                        for stratum_type, stratum_value in row_strata_values:
                            exact_counts[(stratum_type, stratum_value, n, exact_id)] += 1

                    harmonic_id = harmonic_exact_members.get(exact_id)
                    if harmonic_id in harmonic_targets:
                        for stratum_type, stratum_value in row_strata_values:
                            harmonic_counts[(stratum_type, stratum_value, n, harmonic_id)] += 1

        row_bar.update(len(chunk))

    row_bar.close()

    exact_df = _counts_frame(exact_counts, "exact_ngram_id")
    harmonic_df = _counts_frame(harmonic_counts, "harmonic_id")
    totals_df = _totals_frame(totals)
    songs_df = pd.DataFrame(
        [
            {
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "song_count": count,
            }
            for (stratum_type, stratum_value), count in songs.items()
        ]
    )

    if show_progress:
        tqdm.write(
            "Writing trend tables to DuckDB "
            f"({len(exact_df):,} exact rows, {len(harmonic_df):,} harmonic rows)..."
        )

    con = duckdb.connect(str(db_path))
    try:
        ds.configure_connection(con)
        if overwrite:
            con.execute("DROP VIEW IF EXISTS trend_exact")
            con.execute("DROP VIEW IF EXISTS trend_harmonic")
            con.execute("DROP VIEW IF EXISTS trend_comparison")
            con.execute("DROP TABLE IF EXISTS trend_exact_counts")
            con.execute("DROP TABLE IF EXISTS trend_harmonic_counts")
            con.execute("DROP TABLE IF EXISTS trend_stratum_totals")
            con.execute("DROP TABLE IF EXISTS trend_stratum_songs")

        con.register("trend_exact_counts_df", exact_df)
        con.register("trend_harmonic_counts_df", harmonic_df)
        con.register("trend_stratum_totals_df", totals_df)
        con.register("trend_stratum_songs_df", songs_df)

        con.execute("CREATE TABLE trend_exact_counts AS SELECT * FROM trend_exact_counts_df")
        con.execute("CREATE TABLE trend_harmonic_counts AS SELECT * FROM trend_harmonic_counts_df")
        con.execute("CREATE TABLE trend_stratum_totals AS SELECT * FROM trend_stratum_totals_df")
        con.execute("CREATE TABLE trend_stratum_songs AS SELECT * FROM trend_stratum_songs_df")

        con.execute(
            """
            CREATE VIEW trend_exact AS
            SELECT
                c.stratum_type,
                c.stratum_value,
                c.n,
                c.exact_ngram_id,
                v.ngram,
                v.harmonic_id,
                c.count,
                t.total_windows,
                c.count::DOUBLE / NULLIF(t.total_windows, 0) AS frequency
            FROM trend_exact_counts c
            JOIN trend_stratum_totals t USING (stratum_type, stratum_value, n)
            LEFT JOIN exact_ngrams v USING (exact_ngram_id)
            """
        )
        con.execute(
            """
            CREATE VIEW trend_harmonic AS
            SELECT
                c.stratum_type,
                c.stratum_value,
                c.n,
                c.harmonic_id,
                h.example_ngram,
                c.count,
                t.total_windows,
                c.count::DOUBLE / NULLIF(t.total_windows, 0) AS frequency
            FROM trend_harmonic_counts c
            JOIN trend_stratum_totals t USING (stratum_type, stratum_value, n)
            LEFT JOIN harmonic_ngrams h USING (harmonic_id)
            """
        )
        con.execute(
            """
            CREATE VIEW trend_comparison AS
            SELECT
                e.stratum_type,
                e.stratum_value,
                e.n,
                e.exact_ngram_id,
                e.ngram,
                e.harmonic_id,
                e.count AS exact_count,
                e.frequency AS exact_frequency,
                h.count AS harmonic_count,
                h.frequency AS harmonic_frequency,
                h.frequency / NULLIF(e.frequency, 0) AS harmonic_to_exact_frequency_ratio
            FROM trend_exact e
            LEFT JOIN trend_harmonic h
              ON e.stratum_type = h.stratum_type
             AND e.stratum_value = h.stratum_value
             AND e.n = h.n
             AND e.harmonic_id = h.harmonic_id
            """
        )

        con.execute("CREATE INDEX IF NOT EXISTS trend_exact_idx ON trend_exact_counts(stratum_type, stratum_value, n)")
        con.execute("CREATE INDEX IF NOT EXISTS trend_harmonic_idx ON trend_harmonic_counts(stratum_type, stratum_value, n)")
        con.execute("CREATE INDEX IF NOT EXISTS trend_totals_idx ON trend_stratum_totals(stratum_type, stratum_value, n)")
    finally:
        con.close()

    if show_progress:
        tqdm.write("Stratified trend tables complete.")

    return {
        "exact_trend_rows": len(exact_df),
        "harmonic_trend_rows": len(harmonic_df),
        "stratum_total_rows": len(totals_df),
        "stratum_song_rows": len(songs_df),
    }


def _document_values(
    row_data: dict,
    *,
    include_main_genre: bool = True,
    include_decade_1950_plus: bool = True,
    min_decade: int = 1950,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    if include_main_genre:
        value = _clean_value(row_data.get("main_genre"))
        if value is not None:
            out.append(("main_genre", value))

    if include_decade_1950_plus:
        decade = decade_value(row_data)
        if decade is not None and decade >= min_decade:
            out.append(("decade_1950_plus", str(decade)))

    return out


def build_harmonic_document_counts(
    *,
    db_path,
    raw_path,
    ns: tuple[int, ...],
    min_global_harmonic_count: int = 1000,
    min_decade: int = 1950,
    include_main_genre: bool = True,
    include_decade_1950_plus: bool = True,
    chunksize: int = 25_000,
    max_rows: int | None = None,
    show_progress: bool = True,
    overwrite: bool = True,
) -> dict[str, int]:
    """
    Build broad support-thresholded harmonic document-term counts.

    The output tables are designed for corpus-linguistic analysis:

    - ``harmonic_document_counts``: selected H_n counts by document.
    - ``harmonic_document_totals``: denominator windows by document and n.
    - ``harmonic_document_songs``: song counts by document.
    - ``harmonic_document_terms``: counts joined to examples, frequencies, and
      global counts.

    This deliberately differs from notebook 4's target trend tables. Notebook 4
    tracks a small fixed set of globally top targets; this function counts a
    broader support-thresholded harmonic vocabulary so document frequency and
    TF-IDF are meaningful.
    """
    db_path = Path(db_path)
    raw_path = Path(raw_path)
    if show_progress:
        tqdm.write(
            "Selecting support-thresholded harmonic targets from DuckDB "
            f"(min_global_harmonic_count={min_global_harmonic_count:,})..."
        )
    targets = supported_harmonic_targets(
        db_path=db_path,
        ns=ns,
        min_global_harmonic_count=min_global_harmonic_count,
    )

    harmonic_counts: Counter = Counter()
    totals: Counter = Counter()
    songs: Counter = Counter()

    usecols = ["id", "chords", "release_date", "decade", "main_genre"]
    read_kwargs = {
        "usecols": usecols,
        "chunksize": chunksize,
        "low_memory": False,
    }
    if max_rows is not None:
        read_kwargs["nrows"] = max_rows

    total_rows = max_rows
    if total_rows is None:
        try:
            total_rows = sum(1 for _ in open(raw_path, encoding="utf-8")) - 1
        except OSError:
            total_rows = None

    reader = pd.read_csv(raw_path, **read_kwargs)
    chunk_iter = tqdm(
        reader,
        desc="Counting harmonic document terms",
        unit="chunk",
        disable=not show_progress,
    )
    row_bar = tqdm(
        total=total_rows,
        desc="Scanning songs",
        unit="song",
        disable=not show_progress,
    )

    processed_rows = 0
    for chunk in chunk_iter:
        processed_rows += len(chunk)
        if show_progress:
            chunk_iter.set_postfix(
                {
                    "rows": processed_rows,
                    "H_hits": sum(harmonic_counts.values()),
                }
            )

        for row in chunk.itertuples(index=False):
            row_data = row._asdict()
            documents = _document_values(
                row_data,
                include_main_genre=include_main_genre,
                include_decade_1950_plus=include_decade_1950_plus,
                min_decade=min_decade,
            )
            if not documents:
                continue

            tokens = ng.chord_tokens(row_data["chords"])
            for document_type, document_value in documents:
                songs[(document_type, document_value)] += 1

            for n in ns:
                ngrams = ng.chord_ngrams(tokens, n)
                if not ngrams:
                    continue

                for document_type, document_value in documents:
                    totals[(document_type, document_value, n)] += len(ngrams)

                harmonic_exact_members = targets["H_members"][n]
                for ngram in ngrams:
                    exact_id = ng.exact_ngram_id(ngram)
                    harmonic_id = harmonic_exact_members.get(exact_id)
                    if harmonic_id is None:
                        continue

                    for document_type, document_value in documents:
                        harmonic_counts[(document_type, document_value, n, harmonic_id)] += 1

        row_bar.update(len(chunk))

    row_bar.close()

    counts_df = _counts_frame(harmonic_counts, "harmonic_id").rename(
        columns={"stratum_type": "document_type", "stratum_value": "document_value"}
    )
    totals_df = _totals_frame(totals).rename(
        columns={"stratum_type": "document_type", "stratum_value": "document_value"}
    )
    songs_df = pd.DataFrame(
        [
            {
                "document_type": document_type,
                "document_value": document_value,
                "song_count": count,
            }
            for (document_type, document_value), count in songs.items()
        ]
    )
    vocabulary_df = pd.DataFrame(
        [
            {"n": n, "harmonic_id": harmonic_id}
            for n, harmonic_ids in targets["H"].items()
            for harmonic_id in harmonic_ids
        ]
    )

    if show_progress:
        tqdm.write(
            "Writing harmonic document-term tables to DuckDB "
            f"({len(counts_df):,} count rows, {len(vocabulary_df):,} vocabulary rows)..."
        )

    con = duckdb.connect(str(db_path))
    try:
        ds.configure_connection(con)
        if overwrite:
            con.execute("DROP VIEW IF EXISTS harmonic_document_terms")
            con.execute("DROP TABLE IF EXISTS harmonic_document_counts")
            con.execute("DROP TABLE IF EXISTS harmonic_document_totals")
            con.execute("DROP TABLE IF EXISTS harmonic_document_songs")
            con.execute("DROP TABLE IF EXISTS harmonic_document_vocabulary")

        con.register("harmonic_document_counts_df", counts_df)
        con.register("harmonic_document_totals_df", totals_df)
        con.register("harmonic_document_songs_df", songs_df)
        con.register("harmonic_document_vocabulary_df", vocabulary_df)

        con.execute("CREATE TABLE harmonic_document_counts AS SELECT * FROM harmonic_document_counts_df")
        con.execute("CREATE TABLE harmonic_document_totals AS SELECT * FROM harmonic_document_totals_df")
        con.execute("CREATE TABLE harmonic_document_songs AS SELECT * FROM harmonic_document_songs_df")
        con.execute("CREATE TABLE harmonic_document_vocabulary AS SELECT * FROM harmonic_document_vocabulary_df")

        con.execute(
            """
            CREATE VIEW harmonic_document_terms AS
            SELECT
                c.document_type,
                c.document_value,
                c.document_type || ':' || c.document_value AS document_id,
                c.n,
                c.harmonic_id,
                h.example_ngram,
                h.count AS global_count,
                h.frequency AS global_frequency,
                c.count,
                t.total_windows,
                c.count::DOUBLE / NULLIF(t.total_windows, 0) AS frequency,
                s.song_count
            FROM harmonic_document_counts c
            JOIN harmonic_document_totals t USING (document_type, document_value, n)
            JOIN harmonic_document_songs s USING (document_type, document_value)
            LEFT JOIN harmonic_ngrams h USING (harmonic_id)
            """
        )

        con.execute("CREATE INDEX IF NOT EXISTS harmonic_document_counts_idx ON harmonic_document_counts(document_type, document_value, n)")
        con.execute("CREATE INDEX IF NOT EXISTS harmonic_document_counts_h_idx ON harmonic_document_counts(harmonic_id)")
        con.execute("CREATE INDEX IF NOT EXISTS harmonic_document_totals_idx ON harmonic_document_totals(document_type, document_value, n)")
    finally:
        con.close()

    if show_progress:
        tqdm.write("Harmonic document-term tables complete.")

    return {
        "harmonic_document_count_rows": len(counts_df),
        "harmonic_document_total_rows": len(totals_df),
        "harmonic_document_song_rows": len(songs_df),
        "harmonic_document_vocabulary_rows": len(vocabulary_df),
        "min_global_harmonic_count": min_global_harmonic_count,
    }
