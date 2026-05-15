from collections import Counter
from functools import lru_cache
import hashlib
import json
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import chords_to_vectors as ctv
from utils import normalize_chords as nc


DEFAULT_NS = tuple(range(3, 9))
NGRAM_SEP = " "
PARSE_ERROR_COLUMNS = [
    "n",
    "error_type",
    "message",
    "count",
    "song_id",
    "ngram_json",
    "ngram",
]


def chord_tokens(chord_string: str) -> list[str]:
    """
    Return chord-like tokens after removing section markers.
    """
    if chord_string is None:
        return []

    s = nc.remove_sections(str(chord_string))
    s = s.replace(",", " ")
    s = s.replace("♯", "#").replace("♭", "b")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []

    return [tok for tok in s.split(" ") if re.match(r"^[A-G]", tok)]


def chord_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_label(ngram: tuple[str, ...]) -> str:
    return NGRAM_SEP.join(ngram)


@lru_cache(maxsize=None)
def chord_pitch_bits(token: str, include_bass: bool = True) -> tuple[int, ...]:
    """
    Return a cached 12-bit chord vector as a tuple.
    """
    root, rest, bass = ctv.parse_token(token)
    root_pc = ctv._pc(root)
    pcs = {(root_pc + i) % 12 for i in ctv.chord_intervals(rest)}
    if include_bass and bass is not None:
        pcs.add(ctv._pc(bass))

    return tuple(1 if pc in pcs else 0 for pc in range(12))


def chord_pitch_vector(token: str, *, include_bass: bool = True) -> np.ndarray:
    """
    Return a 12-vector for one chord.
    """
    return np.asarray(chord_pitch_bits(token, include_bass), dtype=np.uint8)


def ngram_matrix(ngram: tuple[str, ...], *, include_bass: bool = True) -> np.ndarray:
    """
    Return the 12 x n pitch-class matrix for a chord n-gram.
    """
    cols = [chord_pitch_vector(token, include_bass=include_bass) for token in ngram]
    return np.stack(cols, axis=1) if cols else np.zeros((12, 0), dtype=np.uint8)


def matrix_key(matrix: np.ndarray) -> str:
    """
    Stable text key for a binary 12 x n matrix.
    """
    matrix = np.asarray(matrix, dtype=np.uint8)
    bits = "".join(str(int(v)) for v in matrix.reshape(-1))
    return f"{matrix.shape[1]}:{bits}"


def matrix_from_key(key: str) -> np.ndarray:
    """
    Reconstruct a binary 12 x n matrix from a serialized harmonic_key.
    """
    try:
        n_text, bits = str(key).split(":", 1)
        n = int(n_text)
    except ValueError as exc:
        raise ValueError(f"Invalid matrix key format: {key!r}") from exc

    expected = 12 * n
    if len(bits) != expected:
        raise ValueError(
            f"Invalid matrix key length for n={n}: expected {expected} bits, got {len(bits)}"
        )
    if any(bit not in {"0", "1"} for bit in bits):
        raise ValueError(f"Matrix key contains non-binary characters: {key!r}")

    values = np.fromiter((int(bit) for bit in bits), dtype=np.uint8, count=expected)
    return values.reshape(12, n)


def harmonic_key_l2_distance(left: str, right: str) -> float:
    """
    Euclidean distance between two harmonic_key matrices.

    For binary matrices this is sqrt(Hamming distance), but keeping the L2 name
    makes the intended geometry explicit.
    """
    left_matrix = matrix_from_key(left)
    right_matrix = matrix_from_key(right)
    if left_matrix.shape != right_matrix.shape:
        raise ValueError(
            f"Cannot compare harmonic keys with shapes {left_matrix.shape} and {right_matrix.shape}"
        )

    diff = left_matrix.astype(np.int16) - right_matrix.astype(np.int16)
    return float(np.sqrt(np.sum(diff * diff)))


def canonical_transposition_key(matrix: np.ndarray) -> str:
    """
    Canonical key modulo the Z/12 action by pitch-class row shifts.
    """
    matrix = np.asarray(matrix, dtype=np.uint8)
    if matrix.shape[0] != 12:
        raise ValueError(f"Expected a 12 x n matrix, got shape {matrix.shape}")

    shifted_keys = [matrix_key(np.roll(matrix, shift, axis=0)) for shift in range(12)]
    return min(shifted_keys)


@lru_cache(maxsize=None)
def root_pc(token: str) -> int:
    """
    Return the pitch class of a chord token's root, ignoring slash bass.
    """
    root, _, _ = ctv.parse_token(token)
    return ctv._pc(root)


def first_root_normalized_key(
    ngram: tuple[str, ...],
    *,
    include_bass: bool = True,
) -> str:
    """
    Return the 12 x n matrix key after transposing the first chord root to C.

    Slash bass notes do not determine the representative, but they are included
    in the matrix when include_bass=True and are shifted with the rest of the
    pitch classes.
    """
    if not ngram:
        raise ValueError("ngram must be non-empty")

    matrix = ngram_matrix(ngram, include_bass=include_bass)
    return matrix_key(np.roll(matrix, -root_pc(ngram[0]), axis=0))


def exact_ngram_id(ngram: tuple[str, ...]) -> str:
    """
    Stable id for an exact n-gram.
    """
    payload = json.dumps(list(ngram), separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{len(ngram)}_{digest}"


def harmonic_id(harmonic_key: str) -> str | None:
    """
    Stable compact id for a harmonic_key.
    """
    if harmonic_key is None or pd.isna(harmonic_key):
        return None
    n = str(harmonic_key).split(":", 1)[0]
    digest = hashlib.sha1(str(harmonic_key).encode("utf-8")).hexdigest()[:16]
    return f"H{n}_{digest}"


def _finalize_exact_vocab(
    exact_counters: dict[int, Counter],
    exact_to_harmonic: dict[tuple[str, ...], str | None],
) -> pd.DataFrame:
    rows = []

    for n, counter in exact_counters.items():
        total = sum(counter.values())
        for ngram, count in counter.most_common():
            rows.append(
                {
                    "exact_ngram_id": exact_ngram_id(ngram),
                    "n": n,
                    "ngram_json": json.dumps(list(ngram)),
                    "ngram": ngram_label(ngram),
                    "count": count,
                    "frequency": count / total if total else 0.0,
                    "harmonic_key": exact_to_harmonic.get(ngram),
                    "harmonic_id": harmonic_id(exact_to_harmonic.get(ngram)),
                }
            )

    return pd.DataFrame(rows)


def _finalize_harmonic_vocab(
    harmonic_counters: dict[int, Counter],
    harmonic_examples: dict[tuple[int, str], tuple[str, ...]],
) -> pd.DataFrame:
    rows = []

    for n, counter in harmonic_counters.items():
        total = sum(counter.values())
        for key, count in counter.most_common():
            example = harmonic_examples[(n, key)]
            rows.append(
                {
                    "harmonic_id": harmonic_id(key),
                    "n": n,
                    "harmonic_key": key,
                    "count": count,
                    "frequency": count / total if total else 0.0,
                    "example_ngram_json": json.dumps(list(example)),
                    "example_ngram": ngram_label(example),
                }
            )

    return pd.DataFrame(rows)


def ngram_run_summary(
    exact_vocab: pd.DataFrame,
    harmonic_vocab: pd.DataFrame,
    song_summary: pd.DataFrame,
    parse_errors: pd.DataFrame,
    *,
    ns: tuple[int, ...] = DEFAULT_NS,
) -> pd.DataFrame:
    """
    Summarize total occurrences, vocabulary sizes, and harmonic parse failures by n.
    """
    rows = []
    parse_error_counts = (
        parse_errors.groupby("n")["count"].sum().to_dict()
        if not parse_errors.empty and "n" in parse_errors.columns
        else {}
    )

    for n in ns:
        total_windows = (
            int(song_summary[f"n_ngrams_{n}"].sum())
            if f"n_ngrams_{n}" in song_summary.columns
            else 0
        )
        exact_rows = exact_vocab[exact_vocab["n"] == n]
        harmonic_rows = harmonic_vocab[harmonic_vocab["n"] == n]
        parse_error_count = int(parse_error_counts.get(n, 0))

        rows.append(
            {
                "n": n,
                "total_windows": total_windows,
                "exact_occurrences": int(exact_rows["count"].sum()) if not exact_rows.empty else 0,
                "unique_exact_ngrams": int(len(exact_rows)),
                "harmonic_occurrences": int(harmonic_rows["count"].sum()) if not harmonic_rows.empty else 0,
                "unique_harmonic_classes": int(len(harmonic_rows)),
                "parse_error_occurrences": parse_error_count,
                "parse_error_rate": parse_error_count / total_windows if total_windows else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_ngram_vocabularies_from_frame(
    df: pd.DataFrame,
    *,
    chord_col: str = "chords",
    id_col: str | None = "id",
    ns: tuple[int, ...] = DEFAULT_NS,
    include_bass: bool = True,
    max_error_examples: int = 25,
) -> dict[str, pd.DataFrame]:
    """
    Build exact and harmonic n-gram vocabularies without storing song n-gram lists.

    Outputs:
    - exact_vocab: one row per exact chord n-gram.
    - harmonic_vocab: one row per first-root-normalized harmonic class.
    - song_summary: one row per song with only n_chords and n_ngrams_* counts.
    - parse_errors: compact diagnostics for n-grams that could not be harmonically encoded.
    """
    exact_counters = {n: Counter() for n in ns}
    harmonic_counters = {n: Counter() for n in ns}
    harmonic_examples = {}
    exact_to_harmonic = {}
    exact_to_error = {}
    parse_error_counts = Counter()
    parse_error_examples = {}
    song_summary_rows = []

    for row in df.itertuples(index=False):
        row_data = row._asdict()
        song_id = row_data.get(id_col) if id_col is not None else None
        tokens = chord_tokens(row_data[chord_col])

        summary = {"n_chords": len(tokens)}
        if id_col is not None:
            summary[id_col] = song_id

        for n in ns:
            ngrams = chord_ngrams(tokens, n)
            summary[f"n_ngrams_{n}"] = len(ngrams)

            for ngram in ngrams:
                exact_counters[n][ngram] += 1

                if ngram not in exact_to_harmonic:
                    try:
                        exact_to_harmonic[ngram] = first_root_normalized_key(ngram, include_bass=include_bass)
                    except Exception as exc:
                        exact_to_harmonic[ngram] = None
                        error_key = (n, type(exc).__name__, str(exc))
                        exact_to_error[ngram] = error_key
                        parse_error_examples.setdefault(
                            error_key,
                            {
                                "song_id": song_id,
                                "ngram_json": json.dumps(list(ngram)),
                                "ngram": ngram_label(ngram),
                            },
                        )

                harmonic_key = exact_to_harmonic[ngram]
                if harmonic_key is not None:
                    harmonic_counters[n][harmonic_key] += 1
                    harmonic_examples.setdefault((n, harmonic_key), ngram)
                else:
                    parse_error_counts[exact_to_error[ngram]] += 1

        song_summary_rows.append(summary)

    error_rows = []
    for (n, error_type, message), count in parse_error_counts.most_common(max_error_examples):
        example = parse_error_examples[(n, error_type, message)]
        error_rows.append(
            {
                "n": n,
                "error_type": error_type,
                "message": message,
                "count": count,
                **example,
            }
        )

    return {
        "exact_vocab": _finalize_exact_vocab(exact_counters, exact_to_harmonic),
        "harmonic_vocab": _finalize_harmonic_vocab(harmonic_counters, harmonic_examples),
        "song_summary": pd.DataFrame(song_summary_rows),
        "parse_errors": pd.DataFrame(error_rows, columns=PARSE_ERROR_COLUMNS),
    }


def build_ngram_vocabularies_from_csv(
    csv_path,
    *,
    chord_col: str = "chords",
    id_col: str | None = "id",
    ns: tuple[int, ...] = DEFAULT_NS,
    include_bass: bool = True,
    chunksize: int = 50_000,
    max_rows: int | None = None,
    max_error_examples: int = 25,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Chunked version of build_ngram_vocabularies_from_frame for the full corpus.
    """
    usecols = [chord_col]
    if id_col is not None:
        usecols.insert(0, id_col)

    exact_counters = {n: Counter() for n in ns}
    harmonic_counters = {n: Counter() for n in ns}
    harmonic_examples = {}
    exact_to_harmonic = {}
    exact_to_error = {}
    parse_error_counts = Counter()
    parse_error_examples = {}
    song_summary_chunks = []

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
            total_rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        except OSError:
            total_rows = None

    processed_rows = 0
    reader = pd.read_csv(csv_path, **read_kwargs)
    chunk_iter = tqdm(
        reader,
        desc="Processing chord rows",
        unit="chunk",
        disable=not show_progress,
    )

    row_bar = tqdm(
        total=total_rows,
        desc="Counting n-grams",
        unit="song",
        disable=not show_progress,
    )

    for chunk in chunk_iter:
        song_summary_rows = []
        processed_rows += len(chunk)
        if show_progress:
            chunk_iter.set_postfix(
                {
                    "rows": processed_rows,
                    "exact": sum(len(counter) for counter in exact_counters.values()),
                    "harmonic": sum(len(counter) for counter in harmonic_counters.values()),
                    "errors": sum(parse_error_counts.values()),
                }
            )

        for row in chunk.itertuples(index=False):
            row_data = row._asdict()
            song_id = row_data.get(id_col) if id_col is not None else None
            tokens = chord_tokens(row_data[chord_col])

            summary = {"n_chords": len(tokens)}
            if id_col is not None:
                summary[id_col] = song_id

            for n in ns:
                ngrams = chord_ngrams(tokens, n)
                summary[f"n_ngrams_{n}"] = len(ngrams)

                for ngram in ngrams:
                    exact_counters[n][ngram] += 1

                    if ngram not in exact_to_harmonic:
                        try:
                            exact_to_harmonic[ngram] = first_root_normalized_key(ngram, include_bass=include_bass)
                        except Exception as exc:
                            exact_to_harmonic[ngram] = None
                            error_key = (n, type(exc).__name__, str(exc))
                            exact_to_error[ngram] = error_key
                            parse_error_examples.setdefault(
                                error_key,
                                {
                                    "song_id": song_id,
                                    "ngram_json": json.dumps(list(ngram)),
                                    "ngram": ngram_label(ngram),
                                },
                            )

                    harmonic_key = exact_to_harmonic[ngram]
                    if harmonic_key is not None:
                        harmonic_counters[n][harmonic_key] += 1
                        harmonic_examples.setdefault((n, harmonic_key), ngram)
                    else:
                        parse_error_counts[exact_to_error[ngram]] += 1

            song_summary_rows.append(summary)

        row_bar.update(len(chunk))
        song_summary_chunks.append(pd.DataFrame(song_summary_rows))

    row_bar.close()

    song_summary = (
        pd.concat(song_summary_chunks, ignore_index=True)
        if song_summary_chunks
        else pd.DataFrame()
    )

    error_rows = []
    for (n, error_type, message), count in parse_error_counts.most_common(max_error_examples):
        example = parse_error_examples[(n, error_type, message)]
        error_rows.append(
            {
                "n": n,
                "error_type": error_type,
                "message": message,
                "count": count,
                **example,
            }
        )

    return {
        "exact_vocab": _finalize_exact_vocab(exact_counters, exact_to_harmonic),
        "harmonic_vocab": _finalize_harmonic_vocab(harmonic_counters, harmonic_examples),
        "song_summary": song_summary,
        "parse_errors": pd.DataFrame(error_rows, columns=PARSE_ERROR_COLUMNS),
    }


def exact_ngram_frequency_table(
    chord_series: pd.Series,
    *,
    ns: tuple[int, ...] = DEFAULT_NS,
) -> pd.DataFrame:
    rows = []
    counters = {n: Counter() for n in ns}

    for chord_string in chord_series:
        tokens = chord_tokens(chord_string)
        for n in ns:
            counters[n].update(ngram_label(ngram) for ngram in chord_ngrams(tokens, n))

    for n, counter in counters.items():
        total = sum(counter.values())
        for label, count in counter.most_common():
            rows.append(
                {
                    "n": n,
                    "ngram": label,
                    "count": count,
                    "frequency": count / total if total else 0.0,
                }
            )

    return pd.DataFrame(rows)


def harmonic_ngram_frequency_table(
    chord_series: pd.Series,
    *,
    ns: tuple[int, ...] = DEFAULT_NS,
    include_bass: bool = True,
) -> pd.DataFrame:
    rows = []
    counters = {n: Counter() for n in ns}
    examples = {}

    for chord_string in chord_series:
        tokens = chord_tokens(chord_string)
        for n in ns:
            for ngram in chord_ngrams(tokens, n):
                key = first_root_normalized_key(ngram, include_bass=include_bass)
                counters[n][key] += 1
                examples.setdefault((n, key), ngram_label(ngram))

    for n, counter in counters.items():
        total = sum(counter.values())
        for key, count in counter.most_common():
            rows.append(
                {
                    "n": n,
                    "harmonic_key": key,
                    "count": count,
                    "frequency": count / total if total else 0.0,
                    "example_ngram": examples[(n, key)],
                }
            )

    return pd.DataFrame(rows)
